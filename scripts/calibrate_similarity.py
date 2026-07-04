"""Live calibration: measure real Pinecone cosine scores for RU phrase pairs.

This script makes REAL Pinecone and OpenAI API calls when executed. It is a
one-off operator tool, not part of the installed package: ``Settings`` and
``PineconeManager`` are only ever constructed inside :func:`main`, never at
import time, so importing this module is always safe and network-free.

All calibration data is written to a single, throwaway namespace named
``calibration-<uuid>`` and is always deleted afterwards -- this script never
touches ``MEMORY_NAMESPACE_PREFIX`` or a real Telegram user's namespace, and
never deletes the Pinecone index itself.

Pinecone is eventually consistent: an acknowledged upsert does not guarantee
that a subsequent filtered query immediately sees it. After each reference
upsert, this script polls the same filtered query with a bounded timeout
(``--consistency-timeout`` / ``--poll-interval``) instead of assuming
instant visibility -- it never repeats the upsert or recreates an embedding
while waiting.

Usage::

    python scripts/calibrate_similarity.py
    python scripts/calibrate_similarity.py --consistency-timeout 30 --poll-interval 2
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from typing import Any

from telegram_vector_memory_bot.config import get_settings
from telegram_vector_memory_bot.models import VectorMatch
from telegram_vector_memory_bot.pinecone_manager import PineconeManager, VectorMemoryError

logger = logging.getLogger(__name__)

_NAMESPACE_PREFIX = "calibration"
_RECORD_TYPE = "calibration_reference"

DEFAULT_CONSISTENCY_TIMEOUT = 20.0
DEFAULT_POLL_INTERVAL = 1.0

# Fixed, realistic Russian-language phrase pairs spanning a range of expected
# similarity: a likely paraphrase, a related-but-distinct preference, a
# potentially conflicting preference, and an unrelated statement.
CALIBRATION_PAIRS: list[dict[str, str]] = [
    {
        "pair_id": "A_likely_paraphrase",
        "category": "likely_paraphrase",
        "reference": "Я предпочитаю короткие ответы без лишних подробностей.",
        "candidate": "Пожалуйста, отвечай мне кратко и по существу.",
    },
    {
        "pair_id": "B_related_preference",
        "category": "related_preference_phrasing",
        "reference": "По будням я обычно тренируюсь вечером.",
        "candidate": "После работы я часто занимаюсь спортом.",
    },
    {
        "pair_id": "C_potential_conflict",
        "category": "potentially_conflicting_preference",
        "reference": "Я предпочитаю короткие ответы.",
        "candidate": "Мне нужны подробные объяснения с примерами.",
    },
    {
        "pair_id": "D_unrelated",
        "category": "unrelated",
        "reference": "Я предпочитаю короткие ответы.",
        "candidate": "По субботам я готовлю пасту.",
    },
]

# Which CALIBRATION_PAIRS categories are labeled duplicate-like (positive) versus
# should-remain-separate (negative), for this educational MVP's calibration
# interpretation. This labeling is fixed by category, not derived from scores --
# a pair's score never decides its own label.
_POSITIVE_CATEGORIES = frozenset({"likely_paraphrase", "related_preference_phrasing"})
_NEGATIVE_CATEGORIES = frozenset(
    {"potentially_conflicting_preference", "unrelated"}
)

# The project's currently selected operating threshold, used only as a
# reporting default when a caller does not pass the live ``Settings`` value.
# Chosen for the default text-embedding-3-small configuration; must be
# reconfirmed (and this constant updated) if the embedding model or memory
# policy ever changes.
DEFAULT_SELECTED_THRESHOLD = 0.50


class CalibrationError(Exception):
    """Raised when a calibration pair cannot be measured as expected."""


class CalibrationVisibilityTimeoutError(CalibrationError):
    """A reference vector never became query-visible within the configured timeout."""


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser. No required arguments today, but this keeps
    ``--help`` available and gives the script room to grow."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure real Pinecone cosine similarity scores for a fixed set of "
            "Russian-language phrase pairs, to help a human choose a semantic "
            "duplicate threshold. Makes live Pinecone and OpenAI API calls; "
            "all calibration data is written to a throwaway namespace and "
            "deleted before the script exits."
        )
    )
    parser.add_argument(
        "--consistency-timeout",
        type=float,
        default=DEFAULT_CONSISTENCY_TIMEOUT,
        help=(
            "Maximum seconds to wait for an upserted reference vector to "
            "become query-visible before failing "
            f"(default: {DEFAULT_CONSISTENCY_TIMEOUT})."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help=(
            "Seconds to wait between visibility polling attempts "
            f"(default: {DEFAULT_POLL_INTERVAL})."
        ),
    )
    return parser


def _validate_consistency_arguments(consistency_timeout: float, poll_interval: float) -> None:
    """Validate the shared consistency-polling CLI options.

    Raises ``CalibrationError`` (not the built-in ``ValueError``) so callers
    can rely on the same exception type already used for other calibration
    failures.
    """
    if not (consistency_timeout > 0):
        raise CalibrationError("--consistency-timeout must be a positive number")
    if not (poll_interval > 0):
        raise CalibrationError("--poll-interval must be a positive number")
    if poll_interval > consistency_timeout:
        raise CalibrationError("--poll-interval must not be greater than --consistency-timeout")


def new_calibration_namespace() -> str:
    """Generate a unique, throwaway namespace -- never a real user namespace."""
    return f"{_NAMESPACE_PREFIX}-{uuid.uuid4()}"


def _reference_vector_id(pair_id: str) -> str:
    return f"calib-ref-{pair_id}"


def _wait_for_reference_visibility(
    manager: PineconeManager,
    *,
    namespace: str,
    pair_id: str,
    candidate_embedding: list[float],
    consistency_timeout: float,
    poll_interval: float,
) -> tuple[VectorMatch, int]:
    """Poll the same filtered query until exactly one reference match is visible.

    Pinecone upserts are eventually consistent: an acknowledged upsert may not
    be immediately visible to a query. This never repeats the upsert or
    recreates either embedding -- it only re-issues the same filtered query,
    bounded by ``consistency_timeout``. ``top_k=2`` (rather than 1) lets a
    genuinely malformed state -- more than one reference vector matching this
    pair's id -- be detected and fail immediately instead of silently picking
    one match.
    """
    deadline = time.monotonic() + consistency_timeout
    attempts = 0
    while True:
        attempts += 1
        matches = manager.query_by_vector(
            values=candidate_embedding,
            namespace=namespace,
            top_k=2,
            metadata_filter={"pair_id": {"$eq": pair_id}},
        )

        if len(matches) > 1:
            raise CalibrationError(
                f"expected at most one reference match for pair {pair_id!r}, "
                f"got {len(matches)} (malformed calibration state)"
            )

        if len(matches) == 1:
            return matches[0], attempts

        if time.monotonic() >= deadline:
            raise CalibrationVisibilityTimeoutError(
                f"reference vector for pair {pair_id!r} did not become "
                f"query-visible within {consistency_timeout}s"
            )

        logger.info(
            "Reference for pair %s not yet query-visible (attempt %d), retrying",
            pair_id,
            attempts,
        )
        time.sleep(poll_interval)


def _measure_pair(
    manager: PineconeManager,
    namespace: str,
    pair: dict[str, str],
    *,
    consistency_timeout: float,
    poll_interval: float,
) -> dict[str, Any]:
    pair_id = pair["pair_id"]
    logger.info("Measuring calibration pair %s", pair_id)

    reference_embedding = manager.create_embedding(pair["reference"])
    manager.upsert_vector(
        vector_id=_reference_vector_id(pair_id),
        values=reference_embedding,
        metadata={"pair_id": pair_id, "record_type": _RECORD_TYPE},
        namespace=namespace,
    )

    candidate_embedding = manager.create_embedding(pair["candidate"])

    match, attempts = _wait_for_reference_visibility(
        manager,
        namespace=namespace,
        pair_id=pair_id,
        candidate_embedding=candidate_embedding,
        consistency_timeout=consistency_timeout,
        poll_interval=poll_interval,
    )
    logger.info("Pair %s became query-visible after %d attempt(s)", pair_id, attempts)

    return {
        "pair_id": pair_id,
        "category": pair["category"],
        "score": match.score,
        "visibility_attempts": attempts,
    }


def run_calibration(
    manager: PineconeManager,
    embedding_model: str,
    *,
    consistency_timeout: float = DEFAULT_CONSISTENCY_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    similarity_threshold: float = DEFAULT_SELECTED_THRESHOLD,
) -> dict[str, Any]:
    """Measure all calibration pairs in one throwaway namespace, then clean up.

    The namespace is always deleted in a ``finally`` block, even if a pair
    fails partway through -- calibration never leaves data behind.
    """
    namespace = new_calibration_namespace()
    logger.info("Using temporary calibration namespace")

    pairs_result: list[dict[str, Any]] = []
    try:
        for pair in CALIBRATION_PAIRS:
            pairs_result.append(
                _measure_pair(
                    manager,
                    namespace,
                    pair,
                    consistency_timeout=consistency_timeout,
                    poll_interval=poll_interval,
                )
            )
    finally:
        manager.delete_namespace(namespace)
        logger.info("Deleted temporary calibration namespace")

    return {
        "embedding_model": embedding_model,
        "index_name": manager.index_info.name,
        "index_dimension": manager.index_info.dimension,
        "index_metric": manager.index_info.metric,
        "pairs": pairs_result,
        "calibration_interpretation": _describe_calibration_interpretation(
            pairs_result, similarity_threshold
        ),
    }


def _describe_calibration_interpretation(
    pairs_result: list[dict[str, Any]], similarity_threshold: float
) -> dict[str, Any]:
    """Structured interpretation of this run's scores -- never a substitute for
    human judgment across repeated runs and real usage.

    Pairs are labeled positive (duplicate-like: ``_POSITIVE_CATEGORIES``) or
    negative (should remain separate: ``_NEGATIVE_CATEGORIES``) by their fixed
    category, never by their observed score. When every negative score is
    below every positive score, the half-open interval
    ``(maximum_negative_score, minimum_positive_score]`` is reported as the
    candidate threshold interval implied by this run. When scores overlap, no
    interval is fabricated -- the note says plainly that this run's data
    cannot decide a threshold.
    """
    positive_scores = [p["score"] for p in pairs_result if p["category"] in _POSITIVE_CATEGORIES]
    negative_scores = [p["score"] for p in pairs_result if p["category"] in _NEGATIVE_CATEGORIES]

    if not positive_scores or not negative_scores:
        return {
            "minimum_positive_score": None,
            "maximum_negative_score": None,
            "separation_exists": None,
            "candidate_threshold_interval": None,
            "selected_project_threshold": similarity_threshold,
            "note": (
                "This run did not include both a labeled positive and a "
                "labeled negative calibration pair, so no separation could be "
                "computed."
            ),
        }

    minimum_positive_score = min(positive_scores)
    maximum_negative_score = max(negative_scores)
    separation_exists = maximum_negative_score < minimum_positive_score

    if separation_exists:
        candidate_threshold_interval: dict[str, float] | None = {
            "greater_than": maximum_negative_score,
            "less_than_or_equal_to": minimum_positive_score,
        }
        note = (
            "Every negative example scored below every positive example in "
            "this run, so any threshold in "
            f"({maximum_negative_score:.4f}, {minimum_positive_score:.4f}] "
            "would separate them. This is one run's data, not a universal "
            "constant -- confirm with repeated runs and real usage before "
            "changing MEMORY_SIMILARITY_THRESHOLD in .env, and recalibrate if "
            "the embedding model or memory policy changes."
        )
    else:
        candidate_threshold_interval = None
        note = (
            "No single cosine threshold separates this run's labeled "
            "positive and negative examples: the highest-scoring negative "
            f"example ({maximum_negative_score:.4f}) scored at or above the "
            f"lowest-scoring positive example ({minimum_positive_score:.4f}). "
            "This is not a recommendation to change MEMORY_SIMILARITY_THRESHOLD "
            "-- it means this run's pairs cannot decide it alone."
        )

    return {
        "minimum_positive_score": minimum_positive_score,
        "maximum_negative_score": maximum_negative_score,
        "separation_exists": separation_exists,
        "candidate_threshold_interval": candidate_threshold_interval,
        "selected_project_threshold": similarity_threshold,
        "note": note,
    }


def print_report(summary: dict[str, Any]) -> None:
    """Print a compact table plus a machine-readable JSON summary.

    Includes the number of visibility polling attempts per pair, but never
    embeddings or full stored metadata.
    """
    print(f"{'pair_id':<24}{'category':<32}{'score':>8}{'attempts':>10}")
    for row in summary["pairs"]:
        print(
            f"{row['pair_id']:<24}{row['category']:<32}{row['score']:>8.4f}"
            f"{row['visibility_attempts']:>10}"
        )
    print()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        _validate_consistency_arguments(args.consistency_timeout, args.poll_interval)
    except CalibrationError as exc:
        logger.error("Invalid consistency options: %s", exc)
        return 2

    settings = get_settings()
    manager = PineconeManager(settings)

    try:
        summary = run_calibration(
            manager,
            settings.OPENAI_EMBEDDING_MODEL,
            consistency_timeout=args.consistency_timeout,
            poll_interval=args.poll_interval,
            similarity_threshold=settings.MEMORY_SIMILARITY_THRESHOLD,
        )
    except (VectorMemoryError, CalibrationError) as exc:
        logger.error("Calibration failed: %s", exc)
        return 1

    print_report(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
