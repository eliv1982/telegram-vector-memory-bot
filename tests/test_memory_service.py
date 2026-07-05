"""Unit tests for telegram_vector_memory_bot.memory_service.

PineconeManager is replaced with a small fake test double throughout. No
test performs a real network request or reads the user's real .env file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from telegram_vector_memory_bot.config import Settings
from telegram_vector_memory_bot.memory_policy import MemoryPolicy
from telegram_vector_memory_bot.memory_service import (
    MemoryService,
    MemoryServiceError,
    StoredMemoryFormatError,
)
from telegram_vector_memory_bot.models import MemoryAction, MemoryReason, VectorMatch
from telegram_vector_memory_bot.pinecone_manager import VectorQueryError

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
DEFAULT_EMBEDDING = [0.1, 0.2, 0.3, 0.4]

SHORT_ANSWERS_RU = "Я предпочитаю короткие ответы."
BRIEF_RU = "Пиши мне кратко и по существу."
NO_MORE_SHORT_RU = "Я больше не хочу коротких ответов."


def _fixed_clock() -> datetime:
    return FIXED_NOW


class FakeManager:
    """Minimal fake standing in for PineconeManager. Records call arguments."""

    def __init__(self) -> None:
        self.fetch_calls: list[dict[str, Any]] = []
        self.create_embedding_calls: list[str] = []
        self.query_by_vector_calls: list[dict[str, Any]] = []
        self.query_by_text_calls: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []
        self.delete_calls: list[str] = []
        self.namespace_vector_count_calls: list[str] = []

        self.fetch_response: dict[str, dict[str, Any]] = {}
        self.embedding_response: list[float] = list(DEFAULT_EMBEDDING)
        self.query_by_vector_response: list[VectorMatch] = []
        self.query_by_text_response: list[VectorMatch] = []
        self.namespace_vector_count_response: int = 0

        self.raise_on_fetch: Exception | None = None
        self.raise_on_create_embedding: Exception | None = None
        self.raise_on_query_by_vector: Exception | None = None
        self.raise_on_query_by_text: Exception | None = None
        self.raise_on_upsert: Exception | None = None
        self.raise_on_delete: Exception | None = None
        self.raise_on_namespace_vector_count: Exception | None = None

    def fetch_vectors(
        self, *, vector_ids: Sequence[str], namespace: str
    ) -> dict[str, dict[str, Any]]:
        self.fetch_calls.append({"vector_ids": list(vector_ids), "namespace": namespace})
        if self.raise_on_fetch is not None:
            raise self.raise_on_fetch
        return self.fetch_response

    def create_embedding(self, text: str) -> list[float]:
        self.create_embedding_calls.append(text)
        if self.raise_on_create_embedding is not None:
            raise self.raise_on_create_embedding
        return self.embedding_response

    def query_by_vector(
        self,
        *,
        values: Sequence[float],
        namespace: str,
        top_k: int,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[VectorMatch]:
        self.query_by_vector_calls.append(
            {
                "values": list(values),
                "namespace": namespace,
                "top_k": top_k,
                "metadata_filter": metadata_filter,
            }
        )
        if self.raise_on_query_by_vector is not None:
            raise self.raise_on_query_by_vector
        return self.query_by_vector_response

    def query_by_text(
        self,
        *,
        text: str,
        namespace: str,
        top_k: int,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[VectorMatch]:
        self.query_by_text_calls.append(
            {
                "text": text,
                "namespace": namespace,
                "top_k": top_k,
                "metadata_filter": metadata_filter,
            }
        )
        if self.raise_on_query_by_text is not None:
            raise self.raise_on_query_by_text
        return self.query_by_text_response

    def upsert_vector(
        self,
        *,
        vector_id: str,
        values: Sequence[float],
        metadata: Mapping[str, Any],
        namespace: str,
    ) -> None:
        self.upsert_calls.append(
            {
                "vector_id": vector_id,
                "values": list(values),
                "metadata": dict(metadata),
                "namespace": namespace,
            }
        )
        if self.raise_on_upsert is not None:
            raise self.raise_on_upsert

    def delete_namespace(self, namespace: str) -> None:
        self.delete_calls.append(namespace)
        if self.raise_on_delete is not None:
            raise self.raise_on_delete

    def get_namespace_vector_count(self, *, namespace: str) -> int:
        self.namespace_vector_count_calls.append(namespace)
        if self.raise_on_namespace_vector_count is not None:
            raise self.raise_on_namespace_vector_count
        return self.namespace_vector_count_response


def _build_settings(**overrides: Any) -> Settings:
    data: dict[str, Any] = {
        "PINECONE_API_KEY": "test-pinecone-key",
        "PINECONE_INDEX_NAME": "test-index",
        "OPENAI_API_KEY": "test-openai-key",
        "OPENAI_CHAT_MODEL": "gpt-4o-mini",
        "TELEGRAM_BOT_TOKEN": "test-telegram-token",
    }
    data.update(overrides)
    return Settings(_env_file=None, **data)


def _recalled_metadata(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "text": SHORT_ANSWERS_RU,
        "content_hash": "abc123",
        "created_at": "2026-01-01T12:00:00+00:00",
        "source": "telegram",
    }
    data.update(overrides)
    return data


@pytest.fixture
def manager() -> FakeManager:
    return FakeManager()


@pytest.fixture
def settings() -> Settings:
    return _build_settings()


@pytest.fixture
def service(manager: FakeManager, settings: Settings) -> MemoryService:
    return MemoryService(manager=manager, settings=settings, clock=_fixed_clock)


# ---------------------------------------------------------------------------
# Exact duplicate
# ---------------------------------------------------------------------------


def test_exact_duplicate_returns_skipped(service: MemoryService, manager: FakeManager) -> None:
    policy = MemoryPolicy(0.90)
    memory_id = policy.memory_id_for_text(SHORT_ANSWERS_RU)
    manager.fetch_response = {memory_id: {"values": DEFAULT_EMBEDDING, "metadata": {}}}

    result = service.remember(user_id=1, text=SHORT_ANSWERS_RU)

    assert result.action == MemoryAction.SKIPPED
    assert result.reason == MemoryReason.EXACT_DUPLICATE
    assert result.existing_id == memory_id
    assert result.memory_id is None
    assert result.similarity_score is None


def test_exact_duplicate_does_not_call_create_embedding(
    service: MemoryService, manager: FakeManager
) -> None:
    policy = MemoryPolicy(0.90)
    memory_id = policy.memory_id_for_text(SHORT_ANSWERS_RU)
    manager.fetch_response = {memory_id: {"values": DEFAULT_EMBEDDING, "metadata": {}}}

    service.remember(user_id=1, text=SHORT_ANSWERS_RU)

    assert manager.create_embedding_calls == []


def test_exact_duplicate_does_not_query(service: MemoryService, manager: FakeManager) -> None:
    policy = MemoryPolicy(0.90)
    memory_id = policy.memory_id_for_text(SHORT_ANSWERS_RU)
    manager.fetch_response = {memory_id: {"values": DEFAULT_EMBEDDING, "metadata": {}}}

    service.remember(user_id=1, text=SHORT_ANSWERS_RU)

    assert manager.query_by_vector_calls == []


def test_exact_duplicate_does_not_upsert(service: MemoryService, manager: FakeManager) -> None:
    policy = MemoryPolicy(0.90)
    memory_id = policy.memory_id_for_text(SHORT_ANSWERS_RU)
    manager.fetch_response = {memory_id: {"values": DEFAULT_EMBEDDING, "metadata": {}}}

    service.remember(user_id=1, text=SHORT_ANSWERS_RU)

    assert manager.upsert_calls == []


# ---------------------------------------------------------------------------
# New memory
# ---------------------------------------------------------------------------


def test_new_memory_is_inserted_with_expected_result(
    service: MemoryService, manager: FakeManager
) -> None:
    policy = MemoryPolicy(0.90)
    expected_id = policy.memory_id_for_text(SHORT_ANSWERS_RU)

    result = service.remember(user_id=7, text=SHORT_ANSWERS_RU)

    assert result.action == MemoryAction.INSERTED
    assert result.reason == MemoryReason.NEW_MEMORY
    assert result.memory_id == expected_id
    assert result.existing_id is None
    assert result.similarity_score is None


def test_new_memory_creates_exactly_one_embedding(
    service: MemoryService, manager: FakeManager
) -> None:
    service.remember(user_id=7, text=SHORT_ANSWERS_RU)

    assert manager.create_embedding_calls == [SHORT_ANSWERS_RU]


def test_new_memory_query_uses_exact_namespace_and_top_k_one(
    service: MemoryService, manager: FakeManager
) -> None:
    service.remember(user_id=7, text=SHORT_ANSWERS_RU)

    assert len(manager.query_by_vector_calls) == 1
    call = manager.query_by_vector_calls[0]
    assert call["namespace"] == "telegram-user-7"
    assert call["top_k"] == 1
    assert call["metadata_filter"] == {"record_type": {"$eq": "user_memory"}}


def test_new_memory_upsert_reuses_same_embedding(
    service: MemoryService, manager: FakeManager
) -> None:
    service.remember(user_id=7, text=SHORT_ANSWERS_RU)

    assert manager.upsert_calls[0]["values"] == DEFAULT_EMBEDDING
    assert manager.query_by_vector_calls[0]["values"] == DEFAULT_EMBEDDING


def test_new_memory_uses_deterministic_id(service: MemoryService, manager: FakeManager) -> None:
    policy = MemoryPolicy(0.90)
    expected_id = policy.memory_id_for_text(SHORT_ANSWERS_RU)

    service.remember(user_id=7, text=SHORT_ANSWERS_RU)

    assert manager.upsert_calls[0]["vector_id"] == expected_id


def test_new_memory_safe_metadata_is_correct(service: MemoryService, manager: FakeManager) -> None:
    policy = MemoryPolicy(0.90)

    service.remember(
        user_id=7,
        text=SHORT_ANSWERS_RU,
        username="jdoe",
        first_name="Jane",
        last_name=None,
    )

    metadata = manager.upsert_calls[0]["metadata"]
    assert metadata["user_id"] == 7
    assert metadata["text"] == SHORT_ANSWERS_RU
    assert metadata["content_hash"] == policy.content_hash(SHORT_ANSWERS_RU)
    assert metadata["source"] == "telegram"
    assert metadata["record_type"] == "user_memory"
    assert metadata["username"] == "jdoe"
    assert metadata["first_name"] == "Jane"


def test_new_memory_last_name_included_when_present(
    service: MemoryService, manager: FakeManager
) -> None:
    service.remember(user_id=7, text=SHORT_ANSWERS_RU, last_name="Doe")

    assert manager.upsert_calls[0]["metadata"]["last_name"] == "Doe"


def test_new_memory_bot_response_is_absent(service: MemoryService, manager: FakeManager) -> None:
    service.remember(user_id=7, text=SHORT_ANSWERS_RU)

    assert "bot_response" not in manager.upsert_calls[0]["metadata"]


def test_new_memory_none_telegram_fields_are_omitted(
    service: MemoryService, manager: FakeManager
) -> None:
    service.remember(user_id=7, text=SHORT_ANSWERS_RU)

    metadata = manager.upsert_calls[0]["metadata"]
    assert "username" not in metadata
    assert "first_name" not in metadata
    assert "last_name" not in metadata


def test_new_memory_does_not_mutate_caller_supplied_strings(
    service: MemoryService, manager: FakeManager
) -> None:
    username = "jdoe"

    service.remember(user_id=7, text=SHORT_ANSWERS_RU, username=username)

    assert username == "jdoe"
    assert manager.upsert_calls[0]["metadata"]["username"] == "jdoe"


def test_new_memory_created_at_is_utc_iso8601(
    service: MemoryService, manager: FakeManager
) -> None:
    service.remember(user_id=7, text=SHORT_ANSWERS_RU)

    assert manager.upsert_calls[0]["metadata"]["created_at"] == "2026-01-01T12:00:00+00:00"


# ---------------------------------------------------------------------------
# Semantic duplicate
# ---------------------------------------------------------------------------


def test_high_score_paraphrase_is_skipped(service: MemoryService, manager: FakeManager) -> None:
    manager.query_by_vector_response = [
        VectorMatch(vector_id="mem-existing", score=0.95, metadata={"text": BRIEF_RU})
    ]

    result = service.remember(user_id=1, text=SHORT_ANSWERS_RU)

    assert result.action == MemoryAction.SKIPPED
    assert result.reason == MemoryReason.SEMANTIC_DUPLICATE
    assert manager.query_by_vector_calls[0]["metadata_filter"] == {
        "record_type": {"$eq": "user_memory"}
    }


def test_semantic_duplicate_does_not_upsert(service: MemoryService, manager: FakeManager) -> None:
    manager.query_by_vector_response = [
        VectorMatch(vector_id="mem-existing", score=0.95, metadata={"text": BRIEF_RU})
    ]

    service.remember(user_id=1, text=SHORT_ANSWERS_RU)

    assert manager.upsert_calls == []


def test_semantic_duplicate_returns_existing_id_and_score(
    service: MemoryService, manager: FakeManager
) -> None:
    manager.query_by_vector_response = [
        VectorMatch(vector_id="mem-existing", score=0.95, metadata={"text": BRIEF_RU})
    ]

    result = service.remember(user_id=1, text=SHORT_ANSWERS_RU)

    assert result.existing_id == "mem-existing"
    assert result.similarity_score == 0.95
    assert result.memory_id is None


def test_semantic_duplicate_reads_candidate_text_from_metadata(
    service: MemoryService, manager: FakeManager
) -> None:
    manager.query_by_vector_response = [
        VectorMatch(vector_id="mem-existing", score=0.95, metadata={"text": NO_MORE_SHORT_RU})
    ]

    # NO_MORE_SHORT_RU is negated, SHORT_ANSWERS_RU is not -> not a duplicate,
    # proving the candidate's *own* metadata text (not the new text) was used.
    result = service.remember(user_id=1, text=SHORT_ANSWERS_RU)

    assert result.action == MemoryAction.INSERTED


# ---------------------------------------------------------------------------
# Negation guard
# ---------------------------------------------------------------------------


def test_negation_mismatch_inserts_new_memory_despite_high_score(
    service: MemoryService, manager: FakeManager
) -> None:
    manager.query_by_vector_response = [
        VectorMatch(vector_id="mem-existing", score=0.99, metadata={"text": SHORT_ANSWERS_RU})
    ]

    result = service.remember(user_id=1, text=NO_MORE_SHORT_RU)

    assert result.action == MemoryAction.INSERTED
    assert result.reason == MemoryReason.NEW_MEMORY
    assert manager.query_by_vector_calls[0]["metadata_filter"] == {
        "record_type": {"$eq": "user_memory"}
    }


def test_negation_mismatch_never_updates_existing_candidate(
    service: MemoryService, manager: FakeManager
) -> None:
    manager.query_by_vector_response = [
        VectorMatch(vector_id="mem-existing", score=0.99, metadata={"text": SHORT_ANSWERS_RU})
    ]

    service.remember(user_id=1, text=NO_MORE_SHORT_RU)

    assert len(manager.upsert_calls) == 1
    assert manager.upsert_calls[0]["vector_id"] != "mem-existing"


# ---------------------------------------------------------------------------
# Conservative malformed candidate
# ---------------------------------------------------------------------------


def test_missing_candidate_text_inserts_conservatively(
    service: MemoryService, manager: FakeManager
) -> None:
    manager.query_by_vector_response = [
        VectorMatch(vector_id="mem-existing", score=0.99, metadata={})
    ]

    result = service.remember(user_id=1, text=SHORT_ANSWERS_RU)

    assert result.action == MemoryAction.INSERTED


def test_non_string_candidate_text_inserts_conservatively(
    service: MemoryService, manager: FakeManager
) -> None:
    manager.query_by_vector_response = [
        VectorMatch(vector_id="mem-existing", score=0.99, metadata={"text": 12345})
    ]

    result = service.remember(user_id=1, text=SHORT_ANSWERS_RU)

    assert result.action == MemoryAction.INSERTED


def test_empty_candidate_text_inserts_conservatively(
    service: MemoryService, manager: FakeManager
) -> None:
    manager.query_by_vector_response = [
        VectorMatch(vector_id="mem-existing", score=0.99, metadata={"text": "   "})
    ]

    result = service.remember(user_id=1, text=SHORT_ANSWERS_RU)

    assert result.action == MemoryAction.INSERTED


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def test_different_users_get_different_namespaces(service: MemoryService) -> None:
    assert service.namespace_for_user(1) != service.namespace_for_user(2)


def test_same_text_two_users_checked_in_separate_namespaces(
    service: MemoryService, manager: FakeManager
) -> None:
    service.remember(user_id=1, text=SHORT_ANSWERS_RU)
    service.remember(user_id=2, text=SHORT_ANSWERS_RU)

    namespaces_used = {call["namespace"] for call in manager.fetch_calls}
    assert namespaces_used == {"telegram-user-1", "telegram-user-2"}


def test_recall_queries_only_requested_user_namespace(
    service: MemoryService, manager: FakeManager
) -> None:
    service.recall(user_id=5, query="короткие ответы")

    assert manager.query_by_text_calls[0]["namespace"] == "telegram-user-5"


def test_forget_deletes_only_requested_user_namespace(
    service: MemoryService, manager: FakeManager
) -> None:
    service.forget_user(user_id=9)

    assert manager.delete_calls == ["telegram-user-9"]


# ---------------------------------------------------------------------------
# get_memory_count
# ---------------------------------------------------------------------------


def test_get_memory_count_returns_manager_result(
    service: MemoryService, manager: FakeManager
) -> None:
    manager.namespace_vector_count_response = 4

    count = service.get_memory_count(user_id=1)

    assert count == 4


def test_get_memory_count_uses_exact_namespace_for_user(
    service: MemoryService, manager: FakeManager
) -> None:
    service.get_memory_count(user_id=9)

    assert manager.namespace_vector_count_calls == ["telegram-user-9"]


def test_get_memory_count_two_users_use_distinct_namespaces(
    service: MemoryService, manager: FakeManager
) -> None:
    service.get_memory_count(user_id=1)
    service.get_memory_count(user_id=2)

    assert manager.namespace_vector_count_calls == ["telegram-user-1", "telegram-user-2"]


def test_get_memory_count_makes_no_embedding_or_query_call(
    service: MemoryService, manager: FakeManager
) -> None:
    service.get_memory_count(user_id=1)

    assert manager.create_embedding_calls == []
    assert manager.query_by_text_calls == []
    assert manager.query_by_vector_calls == []


@pytest.mark.parametrize("user_id", [0, -1])
def test_get_memory_count_invalid_user_id_rejected_before_manager_calls(
    service: MemoryService, manager: FakeManager, user_id: int
) -> None:
    with pytest.raises(ValueError):
        service.get_memory_count(user_id=user_id)

    assert manager.namespace_vector_count_calls == []


def test_get_memory_count_propagates_manager_error(
    service: MemoryService, manager: FakeManager
) -> None:
    original = VectorQueryError("stats failed")
    manager.raise_on_namespace_vector_count = original

    with pytest.raises(VectorQueryError) as exc_info:
        service.get_memory_count(user_id=1)

    assert exc_info.value is original


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------


def test_recall_parses_valid_matches(service: MemoryService, manager: FakeManager) -> None:
    manager.query_by_text_response = [
        VectorMatch(vector_id="mem-1", score=0.9, metadata=_recalled_metadata()),
    ]

    results = service.recall(user_id=1, query="короткие ответы")

    assert len(results) == 1
    assert results[0].memory_id == "mem-1"
    assert results[0].text == SHORT_ANSWERS_RU
    assert results[0].score == 0.9


def test_recall_preserves_order(service: MemoryService, manager: FakeManager) -> None:
    manager.query_by_text_response = [
        VectorMatch(
            vector_id="mem-low", score=0.5, metadata=_recalled_metadata(text="низкий скор")
        ),
        VectorMatch(
            vector_id="mem-high", score=0.9, metadata=_recalled_metadata(text="высокий скор")
        ),
    ]

    results = service.recall(user_id=1, query="q")

    assert [r.memory_id for r in results] == ["mem-low", "mem-high"]


def test_recall_default_top_k_from_settings(
    service: MemoryService, manager: FakeManager, settings: Settings
) -> None:
    service.recall(user_id=1, query="q")

    assert manager.query_by_text_calls[0]["top_k"] == settings.MEMORY_TOP_K


def test_recall_explicit_top_k_passed_through(
    service: MemoryService, manager: FakeManager
) -> None:
    service.recall(user_id=1, query="q", top_k=3)

    assert manager.query_by_text_calls[0]["top_k"] == 3


def test_recall_uses_record_type_metadata_filter(
    service: MemoryService, manager: FakeManager
) -> None:
    service.recall(user_id=1, query="q")

    assert manager.query_by_text_calls[0]["metadata_filter"] == {
        "record_type": {"$eq": "user_memory"}
    }


def test_recall_missing_required_metadata_raises(
    service: MemoryService, manager: FakeManager
) -> None:
    manager.query_by_text_response = [
        VectorMatch(
            vector_id="mem-1",
            score=0.9,
            metadata={
                "content_hash": "abc",
                "created_at": "2026-01-01T12:00:00+00:00",
                "source": "telegram",
            },
        )
    ]

    with pytest.raises(StoredMemoryFormatError):
        service.recall(user_id=1, query="q")


def test_recall_missing_created_at_key_raises(
    service: MemoryService, manager: FakeManager
) -> None:
    manager.query_by_text_response = [
        VectorMatch(
            vector_id="mem-1",
            score=0.9,
            metadata={"text": SHORT_ANSWERS_RU, "content_hash": "abc", "source": "telegram"},
        )
    ]

    with pytest.raises(StoredMemoryFormatError):
        service.recall(user_id=1, query="q")


def test_recall_malformed_timestamp_raises(service: MemoryService, manager: FakeManager) -> None:
    manager.query_by_text_response = [
        VectorMatch(
            vector_id="mem-1", score=0.9, metadata=_recalled_metadata(created_at="not-a-date")
        )
    ]

    with pytest.raises(StoredMemoryFormatError):
        service.recall(user_id=1, query="q")


def test_recall_naive_timestamp_raises(service: MemoryService, manager: FakeManager) -> None:
    manager.query_by_text_response = [
        VectorMatch(
            vector_id="mem-1",
            score=0.9,
            metadata=_recalled_metadata(created_at="2026-01-01T12:00:00"),
        )
    ]

    with pytest.raises(StoredMemoryFormatError):
        service.recall(user_id=1, query="q")


def test_recall_optional_telegram_metadata_may_be_absent(
    service: MemoryService, manager: FakeManager
) -> None:
    manager.query_by_text_response = [
        VectorMatch(vector_id="mem-1", score=0.9, metadata=_recalled_metadata())
    ]

    results = service.recall(user_id=1, query="q")

    assert results[0].username is None
    assert results[0].first_name is None
    assert results[0].last_name is None


# ---------------------------------------------------------------------------
# Validation and failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("user_id", [0, -1])
def test_remember_invalid_user_id_rejected_before_manager_calls(
    service: MemoryService, manager: FakeManager, user_id: int
) -> None:
    with pytest.raises(ValueError):
        service.remember(user_id=user_id, text=SHORT_ANSWERS_RU)

    assert manager.fetch_calls == []
    assert manager.create_embedding_calls == []


@pytest.mark.parametrize("text", ["", "   "])
def test_remember_empty_text_rejected_before_manager_calls(
    service: MemoryService, manager: FakeManager, text: str
) -> None:
    with pytest.raises(ValueError):
        service.remember(user_id=1, text=text)

    assert manager.fetch_calls == []


@pytest.mark.parametrize("user_id", [0, -1])
def test_recall_invalid_user_id_rejected_before_manager_calls(
    service: MemoryService, manager: FakeManager, user_id: int
) -> None:
    with pytest.raises(ValueError):
        service.recall(user_id=user_id, query="q")

    assert manager.query_by_text_calls == []


@pytest.mark.parametrize("query", ["", "   "])
def test_recall_empty_query_rejected_before_manager_calls(
    service: MemoryService, manager: FakeManager, query: str
) -> None:
    with pytest.raises(ValueError):
        service.recall(user_id=1, query=query)

    assert manager.query_by_text_calls == []


@pytest.mark.parametrize("user_id", [0, -1])
def test_forget_user_invalid_user_id_rejected_before_manager_calls(
    service: MemoryService, manager: FakeManager, user_id: int
) -> None:
    with pytest.raises(ValueError):
        service.forget_user(user_id=user_id)

    assert manager.delete_calls == []


def test_default_clock_produces_timezone_aware_utc_now(
    manager: FakeManager, settings: Settings
) -> None:
    service = MemoryService(manager=manager, settings=settings)

    before = datetime.now(UTC)
    service.remember(user_id=1, text=SHORT_ANSWERS_RU)
    after = datetime.now(UTC)

    created_at = datetime.fromisoformat(manager.upsert_calls[0]["metadata"]["created_at"])
    assert before <= created_at <= after


def test_naive_injected_clock_rejected(manager: FakeManager, settings: Settings) -> None:
    def naive_clock() -> datetime:
        return datetime(2026, 1, 1, 12, 0, 0)

    service = MemoryService(manager=manager, settings=settings, clock=naive_clock)

    with pytest.raises(MemoryServiceError):
        service.remember(user_id=1, text=SHORT_ANSWERS_RU)


def test_manager_infrastructure_exception_propagates_unchanged(
    service: MemoryService, manager: FakeManager
) -> None:
    manager.raise_on_fetch = VectorQueryError("boom")

    with pytest.raises(VectorQueryError):
        service.remember(user_id=1, text=SHORT_ANSWERS_RU)


def test_memory_service_creates_no_clients(service: MemoryService) -> None:
    assert not hasattr(service, "_pinecone")
    assert not hasattr(service, "_openai")
