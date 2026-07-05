"""Unit tests for telegram_vector_memory_bot.pinecone_manager.

All Pinecone and OpenAI clients are fakes or mocks. No test performs a real
network request or reads the user's real .env file.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest
from pinecone import ForbiddenError, NotFoundError, RateLimitError, ServiceError, UnauthorizedError

from telegram_vector_memory_bot import pinecone_manager
from telegram_vector_memory_bot.config import Settings
from telegram_vector_memory_bot.models import VectorMatch
from telegram_vector_memory_bot.pinecone_manager import (
    EmbeddingGenerationError,
    IndexConfigurationError,
    PineconeManager,
    VectorQueryError,
    VectorStorageError,
)

DIMENSION = 4
VALID_VALUES = [0.1, 0.2, 0.3, 0.4]


# ---------------------------------------------------------------------------
# Fakes and fixtures
# ---------------------------------------------------------------------------


def _dict_index_description(
    *,
    name: str = "test-index",
    host: str = "test-index-abc123.svc.pinecone.io",
    dimension: Any = DIMENSION,
    metric: str = "cosine",
    ready: Any = True,
    state: str | None = "Ready",
) -> dict[str, Any]:
    return {
        "name": name,
        "host": host,
        "dimension": dimension,
        "metric": metric,
        "status": {"ready": ready, "state": state},
    }


def _attr_index_description(**overrides: Any) -> SimpleNamespace:
    data = _dict_index_description(**overrides)
    status = data["status"]
    return SimpleNamespace(
        name=data["name"],
        host=data["host"],
        dimension=data["dimension"],
        metric=data["metric"],
        status=SimpleNamespace(ready=status["ready"], state=status["state"]),
    )


class ToDictIndexDescription:
    """A response object that only exposes fields through to_dict()."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def to_dict(self) -> dict[str, Any]:
        return self._data


class FakeIndexHandle:
    def __init__(self) -> None:
        self.upsert_calls: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, Any]] = []
        self.fetch_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.describe_index_stats_calls = 0

        self.upsert_response: Any = {"upserted_count": 1}
        self.query_response: Any = {"matches": []}
        self.fetch_response: Any = {"vectors": {}}
        self.stats_response: Any = {"total_vector_count": 0}

        self.raise_on_upsert: Exception | None = None
        self.raise_on_query: Exception | None = None
        self.raise_on_fetch: Exception | None = None
        self.raise_on_delete: Exception | None = None
        self.raise_on_stats: Exception | None = None

    def upsert(self, **kwargs: Any) -> Any:
        self.upsert_calls.append(kwargs)
        if self.raise_on_upsert is not None:
            raise self.raise_on_upsert
        return self.upsert_response

    def query(self, **kwargs: Any) -> Any:
        self.query_calls.append(kwargs)
        if self.raise_on_query is not None:
            raise self.raise_on_query
        return self.query_response

    def fetch(self, **kwargs: Any) -> Any:
        self.fetch_calls.append(kwargs)
        if self.raise_on_fetch is not None:
            raise self.raise_on_fetch
        return self.fetch_response

    def delete(self, **kwargs: Any) -> Any:
        self.delete_calls.append(kwargs)
        if self.raise_on_delete is not None:
            raise self.raise_on_delete
        return None

    def describe_index_stats(self) -> Any:
        self.describe_index_stats_calls += 1
        if self.raise_on_stats is not None:
            raise self.raise_on_stats
        return self.stats_response


class FakePineconeClient:
    """Fake control-plane client. Deliberately exposes no index-deletion method."""

    def __init__(self, index_description: Any, index_handle: FakeIndexHandle | None = None) -> None:
        self.describe_index_calls: list[str] = []
        self.index_calls: list[dict[str, Any]] = []
        self._index_description = index_description
        self._index_handle = index_handle or FakeIndexHandle()
        self.raise_on_describe_index: Exception | None = None

    def describe_index(self, name: str) -> Any:
        self.describe_index_calls.append(name)
        if self.raise_on_describe_index is not None:
            raise self.raise_on_describe_index
        return self._index_description

    def Index(self, *, host: str = "", name: str = "") -> FakeIndexHandle:
        self.index_calls.append({"host": host, "name": name})
        return self._index_handle

    def __getattr__(self, name: str) -> Any:
        if name in {"delete_index", "delete"}:
            raise AssertionError(f"PineconeManager must never call Pinecone.{name}")
        raise AttributeError(name)


class FakeOpenAIEmbeddings:
    def __init__(self, response: Any = None, exception: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response
        self.exception = exception

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.exception is not None:
            raise self.exception
        return self.response


class FakeOpenAIClient:
    def __init__(
        self,
        embeddings_response: Any = None,
        embeddings_exception: Exception | None = None,
    ) -> None:
        self.embeddings = FakeOpenAIEmbeddings(embeddings_response, embeddings_exception)


def _embedding_response(values: list[float] | None = None) -> dict[str, Any]:
    return {"data": [{"embedding": list(values if values is not None else VALID_VALUES)}]}


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


@pytest.fixture
def fake_pinecone() -> FakePineconeClient:
    return FakePineconeClient(_dict_index_description())


@pytest.fixture
def fake_openai() -> FakeOpenAIClient:
    return FakeOpenAIClient(_embedding_response())


@pytest.fixture
def manager(fake_pinecone: FakePineconeClient, fake_openai: FakeOpenAIClient) -> PineconeManager:
    return PineconeManager(
        _build_settings(), pinecone_client=fake_pinecone, openai_client=fake_openai
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_successful_initialization_with_injected_clients(
    fake_pinecone: FakePineconeClient, fake_openai: FakeOpenAIClient
) -> None:
    mgr = PineconeManager(
        _build_settings(), pinecone_client=fake_pinecone, openai_client=fake_openai
    )

    info = mgr.index_info
    assert info.name == "test-index"
    assert info.host == "test-index-abc123.svc.pinecone.io"
    assert info.dimension == DIMENSION
    assert info.metric == "cosine"
    assert info.ready is True
    assert info.state == "Ready"


def test_index_is_described_exactly_once(
    manager: PineconeManager,
    fake_pinecone: FakePineconeClient,
) -> None:
    manager.describe_index_stats()
    manager.describe_index_stats()

    assert fake_pinecone.describe_index_calls == ["test-index"]


def test_index_handle_created_using_host_not_name(
    fake_pinecone: FakePineconeClient, fake_openai: FakeOpenAIClient
) -> None:
    PineconeManager(_build_settings(), pinecone_client=fake_pinecone, openai_client=fake_openai)

    assert fake_pinecone.index_calls == [
        {"host": "test-index-abc123.svc.pinecone.io", "name": ""}
    ]


def test_cached_index_handle_is_reused(
    manager: PineconeManager,
    fake_pinecone: FakePineconeClient,
) -> None:
    handle = fake_pinecone._index_handle
    manager.describe_index_stats()
    manager.upsert_vector(
        vector_id="mem-1", values=VALID_VALUES, metadata={}, namespace="user-1"
    )

    assert len(fake_pinecone.index_calls) == 1
    assert handle.describe_index_stats_calls == 1
    assert len(handle.upsert_calls) == 1


def test_not_ready_index_rejected(fake_openai: FakeOpenAIClient) -> None:
    fake_pinecone = FakePineconeClient(_dict_index_description(ready=False))

    with pytest.raises(IndexConfigurationError):
        PineconeManager(_build_settings(), pinecone_client=fake_pinecone, openai_client=fake_openai)


def test_non_cosine_index_rejected(fake_openai: FakeOpenAIClient) -> None:
    fake_pinecone = FakePineconeClient(_dict_index_description(metric="euclidean"))

    with pytest.raises(IndexConfigurationError):
        PineconeManager(_build_settings(), pinecone_client=fake_pinecone, openai_client=fake_openai)


def test_cosine_metric_case_insensitive_accepted(fake_openai: FakeOpenAIClient) -> None:
    fake_pinecone = FakePineconeClient(_dict_index_description(metric="COSINE"))

    mgr = PineconeManager(
        _build_settings(), pinecone_client=fake_pinecone, openai_client=fake_openai
    )

    assert mgr.index_info.metric == "COSINE"


@pytest.mark.parametrize("name", [None, "", "   "])
def test_missing_name_rejected(name: Any, fake_openai: FakeOpenAIClient) -> None:
    description = _dict_index_description()
    description["name"] = name
    fake_pinecone = FakePineconeClient(description)

    with pytest.raises(IndexConfigurationError):
        PineconeManager(_build_settings(), pinecone_client=fake_pinecone, openai_client=fake_openai)


@pytest.mark.parametrize("host", [None, "", "   "])
def test_missing_host_rejected(host: Any, fake_openai: FakeOpenAIClient) -> None:
    description = _dict_index_description()
    description["host"] = host
    fake_pinecone = FakePineconeClient(description)

    with pytest.raises(IndexConfigurationError):
        PineconeManager(_build_settings(), pinecone_client=fake_pinecone, openai_client=fake_openai)


@pytest.mark.parametrize("dimension", [None, 0, -1, "1536", True])
def test_invalid_or_missing_dimension_rejected(
    dimension: Any,
    fake_openai: FakeOpenAIClient,
) -> None:
    fake_pinecone = FakePineconeClient(_dict_index_description(dimension=dimension))

    with pytest.raises(IndexConfigurationError):
        PineconeManager(_build_settings(), pinecone_client=fake_pinecone, openai_client=fake_openai)


def test_dict_style_index_description(fake_openai: FakeOpenAIClient) -> None:
    fake_pinecone = FakePineconeClient(_dict_index_description())

    mgr = PineconeManager(
        _build_settings(), pinecone_client=fake_pinecone, openai_client=fake_openai
    )

    assert mgr.index_info.name == "test-index"


def test_attribute_style_index_description(fake_openai: FakeOpenAIClient) -> None:
    fake_pinecone = FakePineconeClient(_attr_index_description())

    mgr = PineconeManager(
        _build_settings(), pinecone_client=fake_pinecone, openai_client=fake_openai
    )

    assert mgr.index_info.name == "test-index"
    assert mgr.index_info.ready is True


def test_index_description_exposing_to_dict(fake_openai: FakeOpenAIClient) -> None:
    fake_pinecone = FakePineconeClient(ToDictIndexDescription(_dict_index_description()))

    mgr = PineconeManager(
        _build_settings(), pinecone_client=fake_pinecone, openai_client=fake_openai
    )

    assert mgr.index_info.name == "test-index"
    assert mgr.index_info.dimension == DIMENSION


def test_index_description_with_raising_to_dict_falls_back_to_attributes(
    fake_openai: FakeOpenAIClient,
) -> None:
    data = _dict_index_description()

    class RaisingToDictIndexDescription:
        """Exposes real attributes, but to_dict() itself is broken."""

        def __init__(self, values: dict[str, Any]) -> None:
            self.name = values["name"]
            self.host = values["host"]
            self.dimension = values["dimension"]
            self.metric = values["metric"]
            self.status = SimpleNamespace(**values["status"])

        def to_dict(self) -> dict[str, Any]:
            raise RuntimeError("to_dict exploded")

    fake_pinecone = FakePineconeClient(RaisingToDictIndexDescription(data))

    mgr = PineconeManager(
        _build_settings(), pinecone_client=fake_pinecone, openai_client=fake_openai
    )

    assert mgr.index_info.name == "test-index"
    assert mgr.index_info.ready is True


def test_describe_index_failure_wrapped(fake_openai: FakeOpenAIClient) -> None:
    fake_pinecone = FakePineconeClient(_dict_index_description())
    fake_pinecone.raise_on_describe_index = RuntimeError("boom")

    with pytest.raises(IndexConfigurationError) as exc_info:
        PineconeManager(_build_settings(), pinecone_client=fake_pinecone, openai_client=fake_openai)

    assert exc_info.value.__cause__ is fake_pinecone.raise_on_describe_index


# ---------------------------------------------------------------------------
# OpenAI construction
# ---------------------------------------------------------------------------


def test_openai_base_url_passed_when_configured(
    monkeypatch: pytest.MonkeyPatch, fake_pinecone: FakePineconeClient
) -> None:
    captured: dict[str, Any] = {}

    class FakeOpenAIConstructor:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.embeddings = FakeOpenAIEmbeddings(_embedding_response())

    monkeypatch.setattr(pinecone_manager, "OpenAI", FakeOpenAIConstructor)
    settings = _build_settings(OPENAI_BASE_URL="https://proxy.example.com/v1")

    PineconeManager(settings, pinecone_client=fake_pinecone)

    assert captured["base_url"] == "https://proxy.example.com/v1"
    assert captured["api_key"] == "test-openai-key"


def test_openai_base_url_omitted_when_not_configured(
    monkeypatch: pytest.MonkeyPatch, fake_pinecone: FakePineconeClient
) -> None:
    captured: dict[str, Any] = {}

    class FakeOpenAIConstructor:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.embeddings = FakeOpenAIEmbeddings(_embedding_response())

    monkeypatch.setattr(pinecone_manager, "OpenAI", FakeOpenAIConstructor)
    settings = _build_settings()

    PineconeManager(settings, pinecone_client=fake_pinecone)

    assert "base_url" not in captured
    assert captured["api_key"] == "test-openai-key"


def test_openai_base_url_omitted_when_normalized_to_none(
    monkeypatch: pytest.MonkeyPatch, fake_pinecone: FakePineconeClient
) -> None:
    # Settings normalizes an empty OPENAI_BASE_URL (as in .env.example) to
    # None; PineconeManager must still omit base_url in that case.
    captured: dict[str, Any] = {}

    class FakeOpenAIConstructor:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.embeddings = FakeOpenAIEmbeddings(_embedding_response())

    monkeypatch.setattr(pinecone_manager, "OpenAI", FakeOpenAIConstructor)
    settings = _build_settings(OPENAI_BASE_URL="")

    assert settings.OPENAI_BASE_URL is None

    PineconeManager(settings, pinecone_client=fake_pinecone)

    assert "base_url" not in captured
    assert captured["api_key"] == "test-openai-key"


def test_secrets_unwrapped_only_for_client_construction(
    monkeypatch: pytest.MonkeyPatch, fake_pinecone: FakePineconeClient
) -> None:
    captured: dict[str, Any] = {}

    class FakeOpenAIConstructor:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.embeddings = FakeOpenAIEmbeddings(_embedding_response())

    monkeypatch.setattr(pinecone_manager, "OpenAI", FakeOpenAIConstructor)
    settings = _build_settings()

    PineconeManager(settings, pinecone_client=fake_pinecone)

    assert captured["api_key"] == "test-openai-key"
    assert isinstance(captured["api_key"], str)
    assert isinstance(settings.OPENAI_API_KEY.get_secret_value(), str)


def test_wrapped_exception_message_does_not_expose_secret_value(manager: PineconeManager) -> None:
    secret = "sk-super-secret-leaked-value"

    class LeakyError(Exception):
        pass

    manager._openai.embeddings.exception = LeakyError(f"Authorization: Bearer {secret}")

    with pytest.raises(EmbeddingGenerationError) as exc_info:
        manager.create_embedding("hello world")

    assert secret not in str(exc_info.value)
    assert secret not in repr(exc_info.value)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def test_valid_embedding(manager: PineconeManager) -> None:
    result = manager.create_embedding("remember to buy milk")

    assert result == VALID_VALUES
    assert all(isinstance(v, float) for v in result)


def test_whitespace_only_text_rejected_without_calling_openai(
    manager: PineconeManager, fake_openai: FakeOpenAIClient
) -> None:
    with pytest.raises(EmbeddingGenerationError):
        manager.create_embedding("   ")

    assert fake_openai.embeddings.calls == []


def test_missing_embedding_data(manager: PineconeManager) -> None:
    manager._openai.embeddings.response = {"data": []}

    with pytest.raises(EmbeddingGenerationError):
        manager.create_embedding("hello")


def test_empty_embedding_vector(manager: PineconeManager) -> None:
    manager._openai.embeddings.response = {"data": [{"embedding": []}]}

    with pytest.raises(EmbeddingGenerationError):
        manager.create_embedding("hello")


def test_embedding_bool_value_rejected(manager: PineconeManager) -> None:
    manager._openai.embeddings.response = {"data": [{"embedding": [0.1, True, 0.3, 0.4]}]}

    with pytest.raises(EmbeddingGenerationError):
        manager.create_embedding("hello")


def test_embedding_malformed_value_rejected(manager: PineconeManager) -> None:
    manager._openai.embeddings.response = {"data": [{"embedding": [0.1, "oops", 0.3, 0.4]}]}

    with pytest.raises(EmbeddingGenerationError):
        manager.create_embedding("hello")


def test_embedding_non_sequence_value_rejected(manager: PineconeManager) -> None:
    manager._openai.embeddings.response = {"data": [{"embedding": "not-a-list"}]}

    with pytest.raises(EmbeddingGenerationError):
        manager.create_embedding("hello")


def test_embedding_dimension_mismatch(manager: PineconeManager) -> None:
    manager._openai.embeddings.response = {"data": [{"embedding": [0.1, 0.2]}]}

    with pytest.raises(IndexConfigurationError):
        manager.create_embedding("hello")


def test_openai_exception_wrapped_and_chained(manager: PineconeManager) -> None:
    original = RuntimeError("openai down")
    manager._openai.embeddings.exception = original

    with pytest.raises(EmbeddingGenerationError) as exc_info:
        manager.create_embedding("hello")

    assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------


def test_valid_upsert(manager: PineconeManager, fake_pinecone: FakePineconeClient) -> None:
    manager.upsert_vector(
        vector_id="mem-1", values=VALID_VALUES, metadata={"text": "hi"}, namespace="user-1"
    )

    handle = fake_pinecone._index_handle
    assert len(handle.upsert_calls) == 1
    call = handle.upsert_calls[0]
    assert call["namespace"] == "user-1"
    assert call["vectors"] == [
        {"id": "mem-1", "values": VALID_VALUES, "metadata": {"text": "hi"}}
    ]


def test_upsert_metadata_not_mutated(
    manager: PineconeManager,
    fake_pinecone: FakePineconeClient,
) -> None:
    original_metadata = {"note": "hello"}

    manager.upsert_vector(
        vector_id="mem-1", values=VALID_VALUES, metadata=original_metadata, namespace="user-1"
    )

    sent_metadata = fake_pinecone._index_handle.upsert_calls[0]["vectors"][0]["metadata"]
    assert sent_metadata == original_metadata
    assert sent_metadata is not original_metadata


def test_upsert_empty_id_rejected(manager: PineconeManager) -> None:
    with pytest.raises(VectorStorageError):
        manager.upsert_vector(vector_id="   ", values=VALID_VALUES, metadata={}, namespace="user-1")


def test_upsert_empty_namespace_rejected(manager: PineconeManager) -> None:
    with pytest.raises(VectorStorageError):
        manager.upsert_vector(vector_id="mem-1", values=VALID_VALUES, metadata={}, namespace="   ")


def test_upsert_bool_vector_value_rejected(manager: PineconeManager) -> None:
    with pytest.raises(VectorStorageError):
        manager.upsert_vector(
            vector_id="mem-1", values=[0.1, True, 0.3, 0.4], metadata={}, namespace="user-1"
        )


def test_upsert_vector_dimension_mismatch_rejected(manager: PineconeManager) -> None:
    with pytest.raises(VectorStorageError):
        manager.upsert_vector(vector_id="mem-1", values=[0.1, 0.2], metadata={}, namespace="user-1")


def test_upsert_response_count_mismatch_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.upsert_response = {"upserted_count": 0}

    with pytest.raises(VectorStorageError):
        manager.upsert_vector(
            vector_id="mem-1", values=VALID_VALUES, metadata={}, namespace="user-1"
        )


def test_upsert_pinecone_exception_wrapped_and_chained(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    original = RuntimeError("pinecone down")
    fake_pinecone._index_handle.raise_on_upsert = original

    with pytest.raises(VectorStorageError) as exc_info:
        manager.upsert_vector(
            vector_id="mem-1", values=VALID_VALUES, metadata={}, namespace="user-1"
        )

    assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def test_valid_dict_style_matches(
    manager: PineconeManager,
    fake_pinecone: FakePineconeClient,
) -> None:
    fake_pinecone._index_handle.query_response = {
        "matches": [
            {"id": "mem-1", "score": 0.9, "metadata": {"text": "a"}},
            {"id": "mem-2", "score": 0.5, "metadata": {"text": "b"}},
        ]
    }

    matches = manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)

    assert matches == [
        VectorMatch(vector_id="mem-1", score=0.9, metadata={"text": "a"}),
        VectorMatch(vector_id="mem-2", score=0.5, metadata={"text": "b"}),
    ]


def test_valid_attribute_style_matches(
    manager: PineconeManager,
    fake_pinecone: FakePineconeClient,
) -> None:
    fake_pinecone._index_handle.query_response = SimpleNamespace(
        matches=[
            SimpleNamespace(id="mem-1", score=0.9, metadata={"text": "a"}),
            SimpleNamespace(id="mem-2", score=0.5, metadata={"text": "b"}),
        ]
    )

    matches = manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)

    assert [m.vector_id for m in matches] == ["mem-1", "mem-2"]


def test_query_match_order_is_preserved(
    manager: PineconeManager,
    fake_pinecone: FakePineconeClient,
) -> None:
    fake_pinecone._index_handle.query_response = {
        "matches": [
            {"id": "mem-low", "score": 0.1, "metadata": {}},
            {"id": "mem-high", "score": 0.9, "metadata": {}},
        ]
    }

    matches = manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)

    assert [m.vector_id for m in matches] == ["mem-low", "mem-high"]


def test_query_missing_metadata_becomes_empty_dict(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.query_response = {"matches": [{"id": "mem-1", "score": 0.9}]}

    matches = manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)

    assert matches[0].metadata == {}


def test_query_invalid_score_below_negative_one_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.query_response = {
        "matches": [{"id": "mem-1", "score": -1.01, "metadata": {}}]
    }

    with pytest.raises(VectorQueryError):
        manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)


def test_query_invalid_score_above_one_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.query_response = {
        "matches": [{"id": "mem-1", "score": 1.01, "metadata": {}}]
    }

    with pytest.raises(VectorQueryError):
        manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)


def test_query_bool_score_rejected(
    manager: PineconeManager,
    fake_pinecone: FakePineconeClient,
) -> None:
    fake_pinecone._index_handle.query_response = {
        "matches": [{"id": "mem-1", "score": True, "metadata": {}}]
    }

    with pytest.raises(VectorQueryError):
        manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)


# ---------------------------------------------------------------------------
# Cosine score boundary normalization
# ---------------------------------------------------------------------------


def test_query_score_exactly_one_unchanged(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.query_response = {
        "matches": [{"id": "mem-1", "score": 1.0, "metadata": {}}]
    }

    matches = manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)

    assert matches[0].score == 1.0


def test_query_score_exactly_negative_one_unchanged(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.query_response = {
        "matches": [{"id": "mem-1", "score": -1.0, "metadata": {}}]
    }

    matches = manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)

    assert matches[0].score == -1.0


def test_query_score_live_observed_positive_overshoot_normalized(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    # Real value observed against the live Pinecone index during Stage 5C
    # acceptance: delta_from_one = 0.0003201999999999927.
    fake_pinecone._index_handle.query_response = {
        "matches": [{"id": "mem-1", "score": 1.0003202, "metadata": {}}]
    }

    matches = manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)

    assert matches[0].score == 1.0


def test_query_score_symmetric_negative_overshoot_normalized(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.query_response = {
        "matches": [{"id": "mem-1", "score": -1.0003202, "metadata": {}}]
    }

    matches = manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)

    assert matches[0].score == -1.0


def test_query_score_exact_positive_tolerance_boundary_normalized(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.query_response = {
        "matches": [{"id": "mem-1", "score": 1.001, "metadata": {}}]
    }

    matches = manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)

    assert matches[0].score == 1.0


def test_query_score_exact_negative_tolerance_boundary_normalized(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.query_response = {
        "matches": [{"id": "mem-1", "score": -1.001, "metadata": {}}]
    }

    matches = manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)

    assert matches[0].score == -1.0


def test_query_score_positive_overshoot_beyond_epsilon_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.query_response = {
        "matches": [{"id": "mem-1", "score": 1.0011, "metadata": {}}]
    }

    with pytest.raises(VectorQueryError):
        manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)


def test_query_score_negative_overshoot_beyond_epsilon_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.query_response = {
        "matches": [{"id": "mem-1", "score": -1.0011, "metadata": {}}]
    }

    with pytest.raises(VectorQueryError):
        manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)


def test_query_score_nan_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.query_response = {
        "matches": [{"id": "mem-1", "score": float("nan"), "metadata": {}}]
    }

    with pytest.raises(VectorQueryError):
        manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)


def test_query_score_positive_infinity_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.query_response = {
        "matches": [{"id": "mem-1", "score": float("inf"), "metadata": {}}]
    }

    with pytest.raises(VectorQueryError):
        manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)


def test_query_score_negative_infinity_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.query_response = {
        "matches": [{"id": "mem-1", "score": float("-inf"), "metadata": {}}]
    }

    with pytest.raises(VectorQueryError):
        manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)


def test_query_integer_score_accepted_and_normalized_to_float(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.query_response = {
        "matches": [{"id": "mem-1", "score": 1, "metadata": {}}]
    }

    matches = manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)

    assert matches[0].score == 1.0
    assert isinstance(matches[0].score, float)


def test_query_dict_style_response_uses_normalized_score(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.query_response = {
        "matches": [{"id": "mem-1", "score": 1.0000001, "metadata": {}}]
    }

    matches = manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)

    assert matches[0].score == 1.0


def test_query_attribute_style_response_uses_normalized_score(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.query_response = SimpleNamespace(
        matches=[SimpleNamespace(id="mem-1", score=-1.0000001, metadata={})]
    )

    matches = manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)

    assert matches[0].score == -1.0


def test_query_match_order_preserved_with_normalized_scores(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.query_response = {
        "matches": [
            {"id": "mem-overshoot", "score": 1.0000001, "metadata": {}},
            {"id": "mem-plain", "score": 0.5, "metadata": {}},
        ]
    }

    matches = manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)

    assert [m.vector_id for m in matches] == ["mem-overshoot", "mem-plain"]
    assert matches[0].score == 1.0
    assert matches[1].score == 0.5


def test_query_non_numeric_score_wrapped_as_vector_query_error(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.query_response = {
        "matches": [{"id": "mem-1", "score": "not-a-number", "metadata": {}}]
    }

    with pytest.raises(VectorQueryError):
        manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)


def test_query_score_error_message_does_not_expose_injected_secret(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    secret = "sk-FAKE-INJECTED-SECRET-VALUE"
    fake_pinecone._index_handle.query_response = {
        "matches": [{"id": "mem-1", "score": 1.01, "metadata": {"text": secret}}]
    }

    with pytest.raises(VectorQueryError) as exc_info:
        manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)

    assert secret not in str(exc_info.value)
    assert secret not in repr(exc_info.value)


def test_query_missing_match_id_rejected(
    manager: PineconeManager,
    fake_pinecone: FakePineconeClient,
) -> None:
    fake_pinecone._index_handle.query_response = {"matches": [{"score": 0.9, "metadata": {}}]}

    with pytest.raises(VectorQueryError):
        manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)


def test_query_match_metadata_not_a_mapping_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.query_response = {
        "matches": [{"id": "mem-1", "score": 0.9, "metadata": "not-a-mapping"}]
    }

    with pytest.raises(VectorQueryError):
        manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)


def test_query_response_missing_matches_list_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.query_response = {"no_matches_here": []}

    with pytest.raises(VectorQueryError):
        manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)


@pytest.mark.parametrize("top_k", [0, 21, -1, True])
def test_query_invalid_top_k_rejected(manager: PineconeManager, top_k: Any) -> None:
    with pytest.raises(VectorQueryError):
        manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=top_k)


def test_query_metadata_filter_omitted_when_none(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5, metadata_filter=None)

    call = fake_pinecone._index_handle.query_calls[0]
    assert "filter" not in call


def test_query_metadata_filter_passed_when_provided(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    manager.query_by_vector(
        values=VALID_VALUES, namespace="user-1", top_k=5, metadata_filter={"source": "telegram"}
    )

    call = fake_pinecone._index_handle.query_calls[0]
    assert call["filter"] == {"source": "telegram"}


def test_query_pinecone_exception_wrapped(
    manager: PineconeManager,
    fake_pinecone: FakePineconeClient,
) -> None:
    original = RuntimeError("pinecone query failed")
    fake_pinecone._index_handle.raise_on_query = original

    with pytest.raises(VectorQueryError) as exc_info:
        manager.query_by_vector(values=VALID_VALUES, namespace="user-1", top_k=5)

    assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# query_by_text
# ---------------------------------------------------------------------------


def test_query_by_text_creates_exactly_one_embedding(
    manager: PineconeManager, fake_openai: FakeOpenAIClient, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.query_response = {"matches": []}

    manager.query_by_text(text="hello", namespace="user-1", top_k=5)

    assert len(fake_openai.embeddings.calls) == 1


def test_query_by_text_delegates_to_vector_query(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.query_response = {"matches": []}

    manager.query_by_text(
        text="hello", namespace="user-1", top_k=3, metadata_filter={"source": "telegram"}
    )

    call = fake_pinecone._index_handle.query_calls[0]
    assert call["namespace"] == "user-1"
    assert call["top_k"] == 3
    assert call["filter"] == {"source": "telegram"}
    assert call["vector"] == VALID_VALUES


# ---------------------------------------------------------------------------
# Fetch and deletion
# ---------------------------------------------------------------------------


def test_valid_fetch(manager: PineconeManager, fake_pinecone: FakePineconeClient) -> None:
    fake_pinecone._index_handle.fetch_response = {
        "vectors": {
            "mem-1": {"values": VALID_VALUES, "metadata": {"text": "a"}},
        }
    }

    result = manager.fetch_vectors(vector_ids=["mem-1"], namespace="user-1")

    assert result == {"mem-1": {"values": VALID_VALUES, "metadata": {"text": "a"}}}
    assert fake_pinecone._index_handle.fetch_calls[0]["namespace"] == "user-1"


def test_fetch_duplicate_ids_rejected(manager: PineconeManager) -> None:
    with pytest.raises(VectorQueryError):
        manager.fetch_vectors(vector_ids=["mem-1", "mem-1"], namespace="user-1")


def test_fetch_empty_collection_rejected(manager: PineconeManager) -> None:
    with pytest.raises(VectorQueryError):
        manager.fetch_vectors(vector_ids=[], namespace="user-1")


def test_fetch_empty_id_in_collection_rejected(manager: PineconeManager) -> None:
    with pytest.raises(VectorQueryError):
        manager.fetch_vectors(vector_ids=["mem-1", "   "], namespace="user-1")


def test_fetch_response_missing_vectors_mapping_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.fetch_response = {"no_vectors_here": {}}

    with pytest.raises(VectorQueryError):
        manager.fetch_vectors(vector_ids=["mem-1"], namespace="user-1")


def test_fetch_missing_metadata_becomes_empty_dict(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.fetch_response = {
        "vectors": {"mem-1": {"values": VALID_VALUES}}
    }

    result = manager.fetch_vectors(vector_ids=["mem-1"], namespace="user-1")

    assert result == {"mem-1": {"values": VALID_VALUES, "metadata": {}}}


def test_fetch_vector_metadata_not_a_mapping_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.fetch_response = {
        "vectors": {"mem-1": {"values": VALID_VALUES, "metadata": "not-a-mapping"}}
    }

    with pytest.raises(VectorQueryError):
        manager.fetch_vectors(vector_ids=["mem-1"], namespace="user-1")


def test_fetch_pinecone_exception_wrapped(
    manager: PineconeManager,
    fake_pinecone: FakePineconeClient,
) -> None:
    original = RuntimeError("fetch failed")
    fake_pinecone._index_handle.raise_on_fetch = original

    with pytest.raises(VectorQueryError) as exc_info:
        manager.fetch_vectors(vector_ids=["mem-1"], namespace="user-1")

    assert exc_info.value.__cause__ is original


def test_delete_uses_delete_all_and_exact_namespace(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    manager.delete_namespace("user-1")

    calls = fake_pinecone._index_handle.delete_calls
    assert calls == [{"delete_all": True, "namespace": "user-1"}]


def test_delete_empty_namespace_rejected(manager: PineconeManager) -> None:
    with pytest.raises(VectorStorageError):
        manager.delete_namespace("   ")


def test_delete_pinecone_exception_wrapped(
    manager: PineconeManager,
    fake_pinecone: FakePineconeClient,
) -> None:
    original = RuntimeError("delete failed")
    fake_pinecone._index_handle.raise_on_delete = original

    with pytest.raises(VectorStorageError) as exc_info:
        manager.delete_namespace("user-1")

    assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Idempotent namespace deletion
# ---------------------------------------------------------------------------


def test_delete_official_not_found_error_treated_as_success(
    manager: PineconeManager,
    fake_pinecone: FakePineconeClient,
) -> None:
    fake_pinecone._index_handle.raise_on_delete = NotFoundError("namespace not found")

    manager.delete_namespace("user-1")

    assert fake_pinecone._index_handle.delete_calls == [
        {"delete_all": True, "namespace": "user-1"}
    ]


def test_delete_status_code_404_treated_as_success(
    manager: PineconeManager,
    fake_pinecone: FakePineconeClient,
) -> None:
    class FakeApiException(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__("not found")
            self.status_code = status_code

    fake_pinecone._index_handle.raise_on_delete = FakeApiException(404)

    manager.delete_namespace("user-1")

    assert fake_pinecone._index_handle.delete_calls == [
        {"delete_all": True, "namespace": "user-1"}
    ]


def test_delete_legacy_status_404_treated_as_success(
    manager: PineconeManager,
    fake_pinecone: FakePineconeClient,
) -> None:
    class LegacyApiException(Exception):
        def __init__(self, status: int) -> None:
            super().__init__("not found")
            self.status = status

    fake_pinecone._index_handle.raise_on_delete = LegacyApiException(404)

    manager.delete_namespace("user-1")

    assert fake_pinecone._index_handle.delete_calls == [
        {"delete_all": True, "namespace": "user-1"}
    ]


def test_delete_401_error_not_swallowed(
    manager: PineconeManager,
    fake_pinecone: FakePineconeClient,
) -> None:
    original = UnauthorizedError("invalid api key")
    fake_pinecone._index_handle.raise_on_delete = original

    with pytest.raises(VectorStorageError) as exc_info:
        manager.delete_namespace("user-1")

    assert exc_info.value.__cause__ is original


def test_delete_403_error_not_swallowed(
    manager: PineconeManager,
    fake_pinecone: FakePineconeClient,
) -> None:
    original = ForbiddenError("forbidden")
    fake_pinecone._index_handle.raise_on_delete = original

    with pytest.raises(VectorStorageError) as exc_info:
        manager.delete_namespace("user-1")

    assert exc_info.value.__cause__ is original


def test_delete_429_error_not_swallowed(
    manager: PineconeManager,
    fake_pinecone: FakePineconeClient,
) -> None:
    original = RateLimitError("rate limited")
    fake_pinecone._index_handle.raise_on_delete = original

    with pytest.raises(VectorStorageError) as exc_info:
        manager.delete_namespace("user-1")

    assert exc_info.value.__cause__ is original


def test_delete_500_error_not_swallowed(
    manager: PineconeManager,
    fake_pinecone: FakePineconeClient,
) -> None:
    original = ServiceError("internal error")
    fake_pinecone._index_handle.raise_on_delete = original

    with pytest.raises(VectorStorageError) as exc_info:
        manager.delete_namespace("user-1")

    assert exc_info.value.__cause__ is original


def test_delete_non_404_exception_message_does_not_expose_injected_secret(
    manager: PineconeManager,
    fake_pinecone: FakePineconeClient,
) -> None:
    secret = "sk-FAKE-INJECTED-SECRET-VALUE"
    original = ServiceError(f"Authorization: Bearer {secret}")
    fake_pinecone._index_handle.raise_on_delete = original

    with pytest.raises(VectorStorageError) as exc_info:
        manager.delete_namespace("user-1")

    assert secret not in str(exc_info.value)
    assert secret not in repr(exc_info.value)


def test_no_method_can_delete_the_entire_index(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    # FakePineconeClient raises AssertionError if any index-deletion-style
    # control-plane method is ever accessed; reaching this point without
    # error proves delete_namespace never touches the index itself.
    manager.delete_namespace("user-1")

    assert fake_pinecone._index_handle.delete_calls == [{"delete_all": True, "namespace": "user-1"}]


# ---------------------------------------------------------------------------
# Stats and import behavior
# ---------------------------------------------------------------------------


def test_stats_response_becomes_plain_dict(
    manager: PineconeManager,
    fake_pinecone: FakePineconeClient,
) -> None:
    raw_response = {"total_vector_count": 42, "namespaces": {"user-1": {"vector_count": 42}}}
    fake_pinecone._index_handle.stats_response = raw_response

    result = manager.describe_index_stats()

    assert result == raw_response
    assert result is not raw_response


def test_stats_response_object_exposing_to_dict(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.stats_response = ToDictIndexDescription(
        {"total_vector_count": 7}
    )

    result = manager.describe_index_stats()

    assert result == {"total_vector_count": 7}


def test_stats_response_unparseable_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.stats_response = 12345

    with pytest.raises(VectorQueryError):
        manager.describe_index_stats()


def test_stats_pinecone_exception_wrapped(
    manager: PineconeManager,
    fake_pinecone: FakePineconeClient,
) -> None:
    original = RuntimeError("stats failed")
    fake_pinecone._index_handle.raise_on_stats = original

    with pytest.raises(VectorQueryError) as exc_info:
        manager.describe_index_stats()

    assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# get_namespace_vector_count
# ---------------------------------------------------------------------------


def test_namespace_vector_count_returned(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.stats_response = {
        "namespaces": {"user-1": {"vector_count": 7}}
    }

    count = manager.get_namespace_vector_count(namespace="user-1")

    assert count == 7


def test_namespace_vector_count_absent_namespace_returns_zero(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.stats_response = {
        "namespaces": {"some-other-user": {"vector_count": 3}}
    }

    count = manager.get_namespace_vector_count(namespace="user-1")

    assert count == 0


def test_namespace_vector_count_present_key_mapped_to_none_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    # The key genuinely exists in the response but its value is null -- a
    # malformed/broken response, not an empty namespace -- must not be
    # silently treated as a count of 0.
    fake_pinecone._index_handle.stats_response = {"namespaces": {"user-1": None}}

    with pytest.raises(VectorQueryError):
        manager.get_namespace_vector_count(namespace="user-1")


def test_namespace_vector_count_present_malformed_summary_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    # The key is present but its value has no usable 'vector_count' at all.
    fake_pinecone._index_handle.stats_response = {"namespaces": {"user-1": 12345}}

    with pytest.raises(VectorQueryError):
        manager.get_namespace_vector_count(namespace="user-1")


def test_namespace_vector_count_empty_namespaces_mapping_returns_zero(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.stats_response = {"namespaces": {}}

    count = manager.get_namespace_vector_count(namespace="user-1")

    assert count == 0


def test_namespace_vector_count_mapping_style_response_supported(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.stats_response = {
        "total_vector_count": 7,
        "namespaces": {"user-1": {"vector_count": 7}},
    }

    count = manager.get_namespace_vector_count(namespace="user-1")

    assert count == 7


def test_namespace_vector_count_attribute_style_response_supported(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.stats_response = SimpleNamespace(
        namespaces={"user-1": SimpleNamespace(vector_count=9)}
    )

    count = manager.get_namespace_vector_count(namespace="user-1")

    assert count == 9


def test_namespace_vector_count_bool_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.stats_response = {
        "namespaces": {"user-1": {"vector_count": True}}
    }

    with pytest.raises(VectorQueryError):
        manager.get_namespace_vector_count(namespace="user-1")


def test_namespace_vector_count_negative_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.stats_response = {
        "namespaces": {"user-1": {"vector_count": -1}}
    }

    with pytest.raises(VectorQueryError):
        manager.get_namespace_vector_count(namespace="user-1")


def test_namespace_vector_count_non_integral_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.stats_response = {
        "namespaces": {"user-1": {"vector_count": 7.5}}
    }

    with pytest.raises(VectorQueryError):
        manager.get_namespace_vector_count(namespace="user-1")


def test_namespace_vector_count_missing_vector_count_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.stats_response = {"namespaces": {"user-1": {}}}

    with pytest.raises(VectorQueryError):
        manager.get_namespace_vector_count(namespace="user-1")


def test_namespace_vector_count_missing_namespaces_key_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.stats_response = {"total_vector_count": 0}

    with pytest.raises(VectorQueryError):
        manager.get_namespace_vector_count(namespace="user-1")


def test_namespace_vector_count_non_mapping_namespaces_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.stats_response = {"namespaces": "not-a-mapping"}

    with pytest.raises(VectorQueryError):
        manager.get_namespace_vector_count(namespace="user-1")


def test_namespace_vector_count_unparseable_response_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    fake_pinecone._index_handle.stats_response = 12345

    with pytest.raises(VectorQueryError):
        manager.get_namespace_vector_count(namespace="user-1")


def test_namespace_vector_count_pinecone_exception_wrapped(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    original = RuntimeError("stats failed")
    fake_pinecone._index_handle.raise_on_stats = original

    with pytest.raises(VectorQueryError) as exc_info:
        manager.get_namespace_vector_count(namespace="user-1")

    assert exc_info.value.__cause__ is original


def test_namespace_vector_count_blank_namespace_rejected(
    manager: PineconeManager, fake_pinecone: FakePineconeClient
) -> None:
    with pytest.raises(VectorQueryError):
        manager.get_namespace_vector_count(namespace="   ")


def test_namespace_vector_count_no_query_or_embedding_call_made(
    manager: PineconeManager, fake_pinecone: FakePineconeClient, fake_openai: FakeOpenAIClient
) -> None:
    fake_pinecone._index_handle.stats_response = {
        "namespaces": {"user-1": {"vector_count": 2}}
    }

    manager.get_namespace_vector_count(namespace="user-1")

    assert fake_pinecone._index_handle.query_calls == []
    assert fake_openai.embeddings.calls == []


def test_importing_package_does_not_create_external_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai
    import pinecone

    def _explode(self: Any, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("client constructor must not be called at import time")

    monkeypatch.setattr(pinecone.Pinecone, "__init__", _explode)
    monkeypatch.setattr(openai.OpenAI, "__init__", _explode)

    reloaded = importlib.reload(pinecone_manager)

    assert reloaded.PineconeManager is not None
