"""Unit tests for telegram_vector_memory_bot.models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from telegram_vector_memory_bot.models import (
    MemoryAction,
    MemoryReason,
    MemoryRecord,
    MemoryWriteResult,
)


def test_valid_inserted_result() -> None:
    result = MemoryWriteResult(
        action=MemoryAction.INSERTED,
        reason=MemoryReason.NEW_MEMORY,
        memory_id="mem-1",
        existing_id=None,
        similarity_score=None,
    )

    assert result.action is MemoryAction.INSERTED
    assert result.memory_id == "mem-1"


def test_inserted_without_memory_id_rejected() -> None:
    with pytest.raises(ValidationError):
        MemoryWriteResult(
            action=MemoryAction.INSERTED,
            reason=MemoryReason.NEW_MEMORY,
            memory_id=None,
        )


def test_valid_exact_duplicate_result() -> None:
    result = MemoryWriteResult(
        action=MemoryAction.SKIPPED,
        reason=MemoryReason.EXACT_DUPLICATE,
        existing_id="mem-1",
        similarity_score=1.0,
    )

    assert result.reason is MemoryReason.EXACT_DUPLICATE
    assert result.existing_id == "mem-1"


def test_valid_semantic_duplicate_result() -> None:
    result = MemoryWriteResult(
        action=MemoryAction.SKIPPED,
        reason=MemoryReason.SEMANTIC_DUPLICATE,
        existing_id="mem-1",
        similarity_score=0.95,
    )

    assert result.reason is MemoryReason.SEMANTIC_DUPLICATE
    assert result.similarity_score == 0.95


def test_duplicate_without_existing_id_rejected() -> None:
    with pytest.raises(ValidationError):
        MemoryWriteResult(
            action=MemoryAction.SKIPPED,
            reason=MemoryReason.SEMANTIC_DUPLICATE,
            existing_id=None,
            similarity_score=0.95,
        )


def test_duplicate_with_inserted_action_rejected() -> None:
    with pytest.raises(ValidationError):
        MemoryWriteResult(
            action=MemoryAction.INSERTED,
            reason=MemoryReason.SEMANTIC_DUPLICATE,
            memory_id="mem-1",
            existing_id="mem-2",
            similarity_score=0.95,
        )


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_similarity_score_out_of_range_rejected(score: float) -> None:
    with pytest.raises(ValidationError):
        MemoryWriteResult(
            action=MemoryAction.SKIPPED,
            reason=MemoryReason.SEMANTIC_DUPLICATE,
            existing_id="mem-1",
            similarity_score=score,
        )


def test_failed_result_allows_none_memory_id() -> None:
    result = MemoryWriteResult(
        action=MemoryAction.FAILED,
        reason=MemoryReason.STORAGE_ERROR,
        memory_id=None,
        existing_id=None,
        similarity_score=None,
    )

    assert result.action is MemoryAction.FAILED
    assert result.memory_id is None


def test_valid_memory_record() -> None:
    record = MemoryRecord(
        memory_id="mem-1",
        user_id=123,
        text="Remember to buy milk",
        content_hash="abc123",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        username="jdoe",
        first_name="Jane",
        last_name="Doe",
    )

    assert record.user_id == 123
    assert record.source == "telegram"


def test_memory_record_empty_text_rejected() -> None:
    with pytest.raises(ValidationError):
        MemoryRecord(
            memory_id="mem-1",
            user_id=123,
            text="   ",
            content_hash="abc123",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_memory_record_non_positive_user_id_rejected() -> None:
    with pytest.raises(ValidationError):
        MemoryRecord(
            memory_id="mem-1",
            user_id=0,
            text="Remember to buy milk",
            content_hash="abc123",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_memory_record_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        MemoryRecord(
            memory_id="mem-1",
            user_id=123,
            text="Remember to buy milk",
            content_hash="abc123",
            created_at=datetime(2026, 1, 1),
        )


def test_memory_record_optional_telegram_fields_default_to_none() -> None:
    record = MemoryRecord(
        memory_id="mem-1",
        user_id=123,
        text="Remember to buy milk",
        content_hash="abc123",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert record.username is None
    assert record.first_name is None
    assert record.last_name is None
