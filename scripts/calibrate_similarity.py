"""Live calibration: measure real Pinecone cosine scores for RU phrase pairs.

This script makes REAL Pinecone and OpenAI API calls when executed. It is a
one-off operator tool, not part of the installed package: ``Settings`` and
``PineconeManager`` are only ever constructed inside :func:`main`, never at
import time, so importing this module is always safe and network-free.

All calibration data is written to a single, throwaway namespace named
``calibration-<uuid>`` and is always deleted afterwards -- this script never
touches ``MEMORY_NAMESPACE_PREFIX`` or a real Telegram user's namespace, and
never deletes the Pinecone index itself.

Usage::

    python scripts/calibrate_similarity.py
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from typing import Any

from telegram_vector_memory_bot.config import get_settings
from telegram_vector_memory_bot.pinecone_manager import PineconeManager, VectorMemoryError

logger = logging.getLogger(__name__)

_NAMESPACE_PREFIX = "calibration"
_RECORD_TYPE = "calibration_reference"

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


class CalibrationError(Exception):
    """Raised when a calibration pair cannot be measured as expected."""


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser. No required arguments today, but this keeps
    ``--help`` available and gives the script room to grow."""
    return argparse.ArgumentParser(
        description=(
            "Measure real Pinecone cosine similarity scores for a fixed set of "
            "Russian-language phrase pairs, to help a human choose a semantic "
            "duplicate threshold. Makes live Pinecone and OpenAI API calls; "
            "all calibration data is written to a throwaway namespace and "
            "deleted before the script exits."
        )
    )


def new_calibration_namespace() -> str:
    """Generate a unique, throwaway namespace -- never a real user namespace."""
    return f"{_NAMESPACE_PREFIX}-{uuid.uuid4()}"


def _reference_vector_id(pair_id: str) -> str:
    return f"calib-ref-{pair_id}"


def _measure_pair(
    manager: PineconeManager, namespace: str, pair: dict[str, str]
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
    matches = manager.query_by_vector(
        values=candidate_embedding,
        namespace=namespace,
        top_k=1,
        metadata_filter={"pair_id": {"$eq": pair_id}},
    )

    if len(matches) != 1:
        raise CalibrationError(
            f"expected exactly one reference match for pair {pair_id!r}, "
            f"got {len(matches)}"
        )

    return {"pair_id": pair_id, "category": pair["category"], "score": matches[0].score}


def run_calibration(manager: PineconeManager, embedding_model: str) -> dict[str, Any]:
    """Measure all calibration pairs in one throwaway namespace, then clean up.

    The namespace is always deleted in a ``finally`` block, even if a pair
    fails partway through -- calibration never leaves data behind.
    """
    namespace = new_calibration_namespace()
    logger.info("Using temporary calibration namespace")

    pairs_result: list[dict[str, Any]] = []
    try:
        for pair in CALIBRATION_PAIRS:
            pairs_result.append(_measure_pair(manager, namespace, pair))
    finally:
        manager.delete_namespace(namespace)
        logger.info("Deleted temporary calibration namespace")

    return {
        "embedding_model": embedding_model,
        "index_name": manager.index_info.name,
        "index_dimension": manager.index_info.dimension,
        "index_metric": manager.index_info.metric,
        "pairs": pairs_result,
        "suggested_threshold_range": _describe_suggested_range(pairs_result),
    }


def _describe_suggested_range(pairs_result: list[dict[str, Any]]) -> dict[str, Any]:
    """Purely descriptive summary -- never a computed universal constant.

    A single observed score never proves semantic equivalence on its own;
    choosing MEMORY_SIMILARITY_THRESHOLD requires human judgment across
    repeated runs and real usage, not this run's numbers alone.
    """
    return {
        "note": (
            "Descriptive only, derived from this single run's observed scores. "
            "A threshold should sit above the observed 'related'/'potentially "
            "conflicting'/'unrelated' scores and at or below the observed "
            "'likely paraphrase' score -- but confirm with more than one run "
            "before changing MEMORY_SIMILARITY_THRESHOLD in .env."
        ),
        "observed_scores_by_pair": {p["pair_id"]: p["score"] for p in pairs_result},
    }


def print_report(summary: dict[str, Any]) -> None:
    """Print a compact table plus a machine-readable JSON summary."""
    print(f"{'pair_id':<24}{'category':<32}{'score':>8}")
    for row in summary["pairs"]:
        print(f"{row['pair_id']:<24}{row['category']:<32}{row['score']:>8.4f}")
    print()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    settings = get_settings()
    manager = PineconeManager(settings)

    try:
        summary = run_calibration(manager, settings.OPENAI_EMBEDDING_MODEL)
    except (VectorMemoryError, CalibrationError) as exc:
        logger.error("Calibration failed: %s", exc)
        return 1

    print_report(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
