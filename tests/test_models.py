"""Unit tests for telegram_vector_memory_bot.models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from telegram_vector_memory_bot.models import (
    IndexInfo,
    MemoryAction,
    MemoryReason,
    MemoryRecord,
    MemoryWriteResult,
    RecalledMemory,
    VectorMatch,
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


@pytest.mark.parametrize("score", [-1.0, 1.0])
def test_similarity_score_boundary_values_accepted(score: float) -> None:
    result = MemoryWriteResult(
        action=MemoryAction.SKIPPED,
        reason=MemoryReason.SEMANTIC_DUPLICATE,
        existing_id="mem-1",
        similarity_score=score,
    )

    assert result.similarity_score == score


@pytest.mark.parametrize("score", [-1.01, 1.01])
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


def test_valid_index_info() -> None:
    info = IndexInfo(
        name="my-index",
        host="my-index-abc123.svc.pinecone.io",
        dimension=1536,
        metric="cosine",
        ready=True,
        state="Ready",
    )

    assert info.dimension == 1536
    assert info.state == "Ready"


@pytest.mark.parametrize("name", ["", "   "])
def test_index_info_empty_name_rejected(name: str) -> None:
    with pytest.raises(ValidationError):
        IndexInfo(
            name=name,
            host="my-index-abc123.svc.pinecone.io",
            dimension=1536,
            metric="cosine",
            ready=True,
        )


@pytest.mark.parametrize("host", ["", "   "])
def test_index_info_empty_host_rejected(host: str) -> None:
    with pytest.raises(ValidationError):
        IndexInfo(
            name="my-index",
            host=host,
            dimension=1536,
            metric="cosine",
            ready=True,
        )


@pytest.mark.parametrize("dimension", [0, -1])
def test_index_info_non_positive_dimension_rejected(dimension: int) -> None:
    with pytest.raises(ValidationError):
        IndexInfo(
            name="my-index",
            host="my-index-abc123.svc.pinecone.io",
            dimension=dimension,
            metric="cosine",
            ready=True,
        )


def test_index_info_empty_metric_rejected() -> None:
    with pytest.raises(ValidationError):
        IndexInfo(
            name="my-index",
            host="my-index-abc123.svc.pinecone.io",
            dimension=1536,
            metric="   ",
            ready=True,
        )


def test_index_info_state_defaults_to_none() -> None:
    info = IndexInfo(
        name="my-index",
        host="my-index-abc123.svc.pinecone.io",
        dimension=1536,
        metric="cosine",
        ready=True,
    )

    assert info.state is None


def test_valid_vector_match() -> None:
    match = VectorMatch(vector_id="mem-1", score=0.42, metadata={"user_id": 1})

    assert match.vector_id == "mem-1"
    assert match.metadata == {"user_id": 1}


def test_vector_match_metadata_defaults_to_empty_dict() -> None:
    match = VectorMatch(vector_id="mem-1", score=0.42)

    assert match.metadata == {}


def test_vector_match_empty_id_rejected() -> None:
    with pytest.raises(ValidationError):
        VectorMatch(vector_id="   ", score=0.42)


@pytest.mark.parametrize("score", [-1.01, 1.01])
def test_vector_match_score_out_of_range_rejected(score: float) -> None:
    with pytest.raises(ValidationError):
        VectorMatch(vector_id="mem-1", score=score)


@pytest.mark.parametrize("score", [-1.0, 1.0])
def test_vector_match_score_boundary_values_accepted(score: float) -> None:
    match = VectorMatch(vector_id="mem-1", score=score)

    assert match.score == score


def test_valid_recalled_memory() -> None:
    memory = RecalledMemory(
        memory_id="mem-1",
        text="Я предпочитаю короткие ответы.",
        score=0.95,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        source="telegram",
        content_hash="abc123",
        username="jdoe",
        first_name="Jane",
        last_name="Doe",
    )

    assert memory.memory_id == "mem-1"
    assert memory.score == 0.95


def test_recalled_memory_optional_telegram_fields_default_to_none() -> None:
    memory = RecalledMemory(
        memory_id="mem-1",
        text="Пиши мне кратко и по существу.",
        score=0.5,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        source="telegram",
        content_hash="abc123",
    )

    assert memory.username is None
    assert memory.first_name is None
    assert memory.last_name is None


@pytest.mark.parametrize("field", ["memory_id", "text", "source", "content_hash"])
def test_recalled_memory_empty_required_field_rejected(field: str) -> None:
    data = {
        "memory_id": "mem-1",
        "text": "Пиши мне кратко и по существу.",
        "score": 0.5,
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "source": "telegram",
        "content_hash": "abc123",
    }
    data[field] = "   "

    with pytest.raises(ValidationError):
        RecalledMemory(**data)


@pytest.mark.parametrize("score", [-1.01, 1.01])
def test_recalled_memory_score_out_of_range_rejected(score: float) -> None:
    with pytest.raises(ValidationError):
        RecalledMemory(
            memory_id="mem-1",
            text="Пиши мне кратко и по существу.",
            score=score,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            source="telegram",
            content_hash="abc123",
        )


def test_recalled_memory_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        RecalledMemory(
            memory_id="mem-1",
            text="Пиши мне кратко и по существу.",
            score=0.5,
            created_at=datetime(2026, 1, 1),
            source="telegram",
            content_hash="abc123",
        )


def test_recalled_memory_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        RecalledMemory(
            memory_id="mem-1",
            text="Пиши мне кратко и по существу.",
            score=0.5,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            source="telegram",
            content_hash="abc123",
            bot_response="this field does not exist",
        )
