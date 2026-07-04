"""Live smoke test: exercises MemoryService end-to-end via a synthetic user.

This script makes REAL Pinecone and OpenAI API calls when executed. It is a
one-off operator tool, not part of the installed package: ``Settings``,
``PineconeManager``, and ``MemoryService`` are only ever constructed inside
:func:`main`, never at import time, so importing this module is always safe
and network-free.

The default ``--user-id`` (900000001) is a synthetic placeholder, not a real
Telegram user ID. All data for the given user ID is removed both before and
after the run, so the namespace is left exactly as it was found.

Usage::

    python scripts/smoke_test_memory.py
    python scripts/smoke_test_memory.py --require-semantic-skip
    python scripts/smoke_test_memory.py --user-id 900000042
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from telegram_vector_memory_bot.config import get_settings
from telegram_vector_memory_bot.memory_service import MemoryService, MemoryServiceError
from telegram_vector_memory_bot.models import MemoryAction, MemoryReason, MemoryWriteResult
from telegram_vector_memory_bot.pinecone_manager import PineconeManager, VectorMemoryError

logger = logging.getLogger(__name__)

# A reserved, clearly-synthetic Telegram user ID block. Real Telegram user IDs
# in current use are well below this range, so rejecting IDs outside of it
# is a practical (not foolproof) guard against accidentally pointing this
# script at a real user's namespace.
DEFAULT_SYNTHETIC_USER_ID = 900000001
_MIN_SYNTHETIC_USER_ID = 900000000
_MAX_SYNTHETIC_USER_ID = 999999999

SHORT_ANSWERS_TEXT = "Я предпочитаю короткие ответы без лишних подробностей."
PARAPHRASE_TEXT = "Пожалуйста, отвечай мне кратко и по существу."
DIFFERENT_MEMORY_TEXT = "По будням я обычно тренируюсь вечером."
RECALL_QUERY_TEXT = "Когда мне удобнее заниматься спортом?"

_LABEL_LIMIT = 40


class SmokeTestError(Exception):
    """Raised when a live smoke-test expectation is not met."""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a live end-to-end smoke test through MemoryService using a "
            "synthetic Telegram user ID. Makes live Pinecone and OpenAI API "
            "calls; all data for the given user ID is removed before and "
            "after the run."
        )
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=DEFAULT_SYNTHETIC_USER_ID,
        help=(
            "Positive, synthetic Telegram user ID to use for this smoke test "
            f"(default: {DEFAULT_SYNTHETIC_USER_ID}, a synthetic placeholder -- "
            "never a real Telegram user ID)."
        ),
    )
    parser.add_argument(
        "--require-semantic-skip",
        action="store_true",
        help=(
            "Fail the smoke test if the paraphrase is not classified as "
            "skipped/semantic_duplicate under the current threshold."
        ),
    )
    return parser


def validate_user_id(user_id: int) -> None:
    """Reject non-positive IDs and IDs outside the synthetic placeholder range."""
    if user_id <= 0:
        raise SmokeTestError("--user-id must be positive")
    if not (_MIN_SYNTHETIC_USER_ID <= user_id <= _MAX_SYNTHETIC_USER_ID):
        raise SmokeTestError(
            "--user-id does not look like a synthetic placeholder ID "
            f"(expected a value between {_MIN_SYNTHETIC_USER_ID} and "
            f"{_MAX_SYNTHETIC_USER_ID}); refusing to risk touching a real "
            "Telegram user's namespace"
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeTestError(message)


def _result_summary(result: MemoryWriteResult) -> dict[str, Any]:
    return {
        "action": result.action.value,
        "reason": result.reason.value,
        "existing_id": result.existing_id,
        "similarity_score": result.similarity_score,
    }


def _short_label(text: str, limit: int = _LABEL_LIMIT) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 1] + "…"


def run_smoke_test(
    service: MemoryService, *, user_id: int, require_semantic_skip: bool
) -> dict[str, Any]:
    """Run the live scenario for *user_id*, always cleaning up afterwards.

    Cleanup runs both before the scenario (removing stale data from a
    previous interrupted run) and in a ``finally`` block afterwards, so a
    failure partway through never leaves synthetic data behind. A cleanup
    failure is logged but never replaces an original scenario failure.
    """
    logger.info("Using synthetic user namespace")
    service.forget_user(user_id=user_id)

    summary: dict[str, Any] = {}
    try:
        first_write = service.remember(user_id=user_id, text=SHORT_ANSWERS_TEXT)
        _require(
            first_write.action == MemoryAction.INSERTED
            and first_write.reason == MemoryReason.NEW_MEMORY,
            f"expected first write inserted/new_memory, got "
            f"{first_write.action}/{first_write.reason}",
        )
        summary["first_write"] = _result_summary(first_write)

        exact_duplicate = service.remember(user_id=user_id, text=SHORT_ANSWERS_TEXT)
        _require(
            exact_duplicate.action == MemoryAction.SKIPPED
            and exact_duplicate.reason == MemoryReason.EXACT_DUPLICATE,
            f"expected exact duplicate skipped/exact_duplicate, got "
            f"{exact_duplicate.action}/{exact_duplicate.reason}",
        )
        summary["exact_duplicate"] = _result_summary(exact_duplicate)

        paraphrase = service.remember(user_id=user_id, text=PARAPHRASE_TEXT)
        if require_semantic_skip:
            _require(
                paraphrase.action == MemoryAction.SKIPPED
                and paraphrase.reason == MemoryReason.SEMANTIC_DUPLICATE,
                "expected paraphrase skipped/semantic_duplicate with "
                f"--require-semantic-skip, got {paraphrase.action}/{paraphrase.reason}",
            )
        summary["semantic_paraphrase"] = _result_summary(paraphrase)

        different_memory = service.remember(user_id=user_id, text=DIFFERENT_MEMORY_TEXT)
        _require(
            different_memory.action == MemoryAction.INSERTED
            and different_memory.reason == MemoryReason.NEW_MEMORY,
            f"expected different memory inserted/new_memory, got "
            f"{different_memory.action}/{different_memory.reason}",
        )
        summary["different_memory"] = _result_summary(different_memory)

        recalled = service.recall(user_id=user_id, query=RECALL_QUERY_TEXT)
        _require(len(recalled) >= 1, "expected recall to return at least one result")
        summary["recall_count"] = len(recalled)
        summary["recall_results"] = [
            {"memory_id": m.memory_id, "score": m.score, "label": _short_label(m.text)}
            for m in recalled
        ]

        service.forget_user(user_id=user_id)
        post_forget_recall = service.recall(user_id=user_id, query=RECALL_QUERY_TEXT)
        _require(len(post_forget_recall) == 0, "expected recall after forget_user to be empty")
        summary["cleanup_verified"] = True
    finally:
        try:
            service.forget_user(user_id=user_id)
        except Exception:
            logger.exception("Safety-net cleanup (forget_user) failed")
            summary["cleanup_verified"] = False

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        validate_user_id(args.user_id)
    except SmokeTestError as exc:
        logger.error("Invalid --user-id: %s", exc)
        return 2

    settings = get_settings()
    manager = PineconeManager(settings)
    service = MemoryService(manager=manager, settings=settings)

    try:
        summary = run_smoke_test(
            service,
            user_id=args.user_id,
            require_semantic_skip=args.require_semantic_skip,
        )
    except (SmokeTestError, VectorMemoryError, MemoryServiceError) as exc:
        logger.error("Smoke test failed: %s", exc)
        return 1

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
