"""Offline tests for hay_v2_bot.services.document_rag and v2 Pinecone storage."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from hay_v2_bot.config import DocumentProcessingSettings, DocumentRagSettings
from hay_v2_bot.models import (
    INSUFFICIENT_DOCUMENT_ANSWER,
    PDF_CONTENT_TYPE,
    DocumentConversionRequest,
    DocumentConversionResult,
)
from hay_v2_bot.services import (
    DocumentIngestionError,
    DocumentQuestionError,
    DocumentRagService,
    DocumentRagServiceError,
    DocumentSummaryError,
)
from hay_v2_bot.storage import (
    DocumentCleanupError,
    DocumentIndexUnavailableError,
    PineconeDocumentStoreFactory,
    document_namespace_for_user,
)
from haystack import Document
from haystack.dataclasses import ChatMessage

UPLOAD_TIME = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)
VALID_HASH = "a" * 64


class FakeAdapter:
    def __init__(
        self,
        result: DocumentConversionResult | None = None,
        *,
        exception: BaseException | None = None,
    ) -> None:
        self._result = result if result is not None else _conversion_result()
        self._exception = exception
        self.calls: list[DocumentConversionRequest] = []

    def convert(self, request: DocumentConversionRequest) -> DocumentConversionResult:
        self.calls.append(request)
        if self._exception is not None:
            raise self._exception
        return self._result


class FakePipelineRunner:
    def __init__(self, result: Any = None, *, exception: BaseException | None = None) -> None:
        self._result = result
        self._exception = exception
        self.calls: list[Any] = []

    def run(self, payload: Any) -> Any:
        self.calls.append(payload)
        if self._exception is not None:
            raise self._exception
        return self._result


class FakeComponent:
    def __init__(self, result: Any = None, *, exception: BaseException | None = None) -> None:
        self._result = result
        self._exception = exception
        self.calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._exception is not None:
            raise self._exception
        if callable(self._result):
            return self._result(**kwargs)
        return self._result


class FakeRagPipeline:
    def __init__(self, components: dict[str, FakeComponent]) -> None:
        self._components = components

    def get_component(self, name: str) -> FakeComponent:
        return self._components[name]


class FakeStoreFactory:
    def __init__(self) -> None:
        self.created_user_ids: list[int] = []
        self.delete_calls: list[tuple[int, tuple[str, ...]]] = []
        self.delete_namespace_calls: list[int] = []
        self.delete_namespace_exception: BaseException | None = None
        self.delete_exception: BaseException | None = None
        self.created_stores: dict[int, dict[str, object]] = {}

    def create_document_store(self, user_id: int) -> dict[str, object]:
        self.created_user_ids.append(user_id)
        store = {
            "namespace": document_namespace_for_user(user_id),
            "user_id": user_id,
        }
        self.created_stores[user_id] = store
        return store

    def delete_documents(self, user_id: int, document_ids: Sequence[str]) -> None:
        self.delete_calls.append((user_id, tuple(document_ids)))
        if self.delete_exception is not None:
            raise self.delete_exception

    def delete_user_namespace(self, user_id: int) -> None:
        self.delete_namespace_calls.append(user_id)
        if self.delete_namespace_exception is not None:
            raise self.delete_namespace_exception


class FakeNotFoundError(Exception):
    status_code = 404


class FakePineconeIndex:
    def __init__(self, *, existing_ids: Sequence[str] = ()) -> None:
        self.delete_calls: list[dict[str, object]] = []
        self.delete_all_calls: list[str] = []
        self.delete_exception: BaseException | None = None
        self.fetch_calls: list[dict[str, object]] = []
        self.existing_ids = set(existing_ids)

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        namespace: str = "",
        delete_all: bool = False,
        **_: Any,
    ) -> None:
        if self.delete_exception is not None:
            raise self.delete_exception
        if delete_all:
            self.delete_all_calls.append(namespace)
            return
        self.delete_calls.append({"ids": ids, "namespace": namespace})
        if ids is not None:
            for document_id in ids:
                self.existing_ids.discard(document_id)

    def fetch(self, *, ids: list[str], namespace: str = "", **_: Any) -> dict[str, object]:
        self.fetch_calls.append({"ids": ids, "namespace": namespace})
        return {
            "vectors": {
                document_id: {}
                for document_id in ids
                if document_id in self.existing_ids
            }
        }


class FakePineconeClient:
    def __init__(
        self,
        *,
        description: dict[str, object] | None = None,
        index_handle: FakePineconeIndex | None = None,
        describe_exception: BaseException | None = None,
    ) -> None:
        self.description = description or {
            "name": "document-index",
            "host": "https://pinecone.invalid/index",
            "dimension": 1536,
            "metric": "cosine",
            "status": {"ready": True, "state": "Ready"},
        }
        self.index_handle = index_handle or FakePineconeIndex()
        self.describe_exception = describe_exception
        self.describe_calls: list[str] = []
        self.index_calls: list[str] = []
        self.create_index_calls: list[dict[str, object]] = []

    def describe_index(self, name: str) -> dict[str, object]:
        self.describe_calls.append(name)
        if self.describe_exception is not None:
            raise self.describe_exception
        return self.description

    def Index(self, *, host: str = "", **_: Any) -> FakePineconeIndex:
        self.index_calls.append(host)
        return self.index_handle

    def create_index(self, **kwargs: object) -> None:
        self.create_index_calls.append(kwargs)


def _processing_settings() -> DocumentProcessingSettings:
    return DocumentProcessingSettings(_env_file=None)


def _rag_settings(**overrides: object) -> DocumentRagSettings:
    return DocumentRagSettings(
        _env_file=None,
        PINECONE_API_KEY="pinecone-key",
        PINECONE_INDEX_NAME="document-index",
        OPENAI_API_KEY="openai-key",
        OPENAI_BASE_URL="https://example.invalid/v1",
        OPENAI_EMBEDDING_MODEL="embedding-model",
        OPENAI_CHAT_MODEL="chat-model",
        **overrides,
    )


def _request(user_id: int = 123) -> DocumentConversionRequest:
    return DocumentConversionRequest(
        local_path=Path("sample.pdf"),
        user_id=user_id,
        file_name="sample.pdf",
        content_type=PDF_CONTENT_TYPE,
        uploaded_at=UPLOAD_TIME,
    )


def _documents() -> tuple[Document, ...]:
    return (
        Document(
            id="doc-1",
            content="First chunk with the budget fact.",
            meta={
                "record_type": "document_chunk",
                "user_id": 123,
                "file_name": "sample.pdf",
                "file_hash": VALID_HASH,
                "chunk_index": 0,
                "content_type": PDF_CONTENT_TYPE,
                "uploaded_at": UPLOAD_TIME.isoformat(),
            },
        ),
        Document(
            id="doc-2",
            content="Second chunk with the schedule fact.",
            meta={
                "record_type": "document_chunk",
                "user_id": 123,
                "file_name": "sample.pdf",
                "file_hash": VALID_HASH,
                "chunk_index": 1,
                "content_type": PDF_CONTENT_TYPE,
                "uploaded_at": UPLOAD_TIME.isoformat(),
                "page_number": 2,
            },
        ),
    )


def _conversion_result(documents: Sequence[Document] | None = None) -> DocumentConversionResult:
    return DocumentConversionResult(
        file_hash=VALID_HASH,
        file_name="sample.pdf",
        content_type=PDF_CONTENT_TYPE,
        documents=list(documents) if documents is not None else list(_documents()),
    )


def _contract_json(
    *,
    answerable: bool = True,
    answer: str | None = "Бюджет составляет 4,2 млн евро.",
    source_ids: Sequence[str] = ("DOC_1",),
) -> str:
    return json.dumps(
        {"answerable": answerable, "answer": answer, "source_ids": list(source_ids)},
        ensure_ascii=False,
    )


def _reply(text: str) -> dict[str, Any]:
    return {"replies": [ChatMessage.from_assistant(text=text)]}


def _retrieved_documents(count: int = 3, *, scores: Sequence[float | None] = ()) -> list[Document]:
    return [
        Document(
            id=f"doc-{position}",
            content=f"Chunk {position}",
            score=scores[position - 1] if position <= len(scores) else None,
            meta={"file_name": "sample.pdf", "chunk_index": position - 1, "page_number": position},
        )
        for position in range(1, count + 1)
    ]


def _answer_for_reply(
    reply_text: str,
    documents: Sequence[Document] | None = None,
) -> tuple[Any, FakeComponent, FakeComponent]:
    retrieved = _retrieved_documents() if documents is None else list(documents)
    prompt_builder = FakeComponent(result={"prompt": [ChatMessage.from_user(text="prompt")]})
    generator = FakeComponent(result=_reply(reply_text))
    service = DocumentRagService(
        _processing_settings(),
        _rag_settings(),
        adapter=FakeAdapter(),
        document_store_factory=FakeStoreFactory(),
        rag_pipeline_factory=lambda settings, store: FakeRagPipeline(
            {
                "text_embedder": FakeComponent(result={"embedding": [0.1]}),
                "retriever": FakeComponent(result={"documents": retrieved}),
                "prompt_builder": prompt_builder,
                "generator": generator,
            }
        ),
    )
    return service.answer_question(123, "Какой бюджет?"), prompt_builder, generator


def _assert_fallback(answer: Any) -> None:
    assert answer.fallback_used is True
    assert answer.sources == ()
    assert answer.used_document_count == 0


def test_pinecone_factory_uses_document_namespace_and_never_creates_index() -> None:
    fake_index = FakePineconeIndex()
    fake_client = FakePineconeClient(index_handle=fake_index)
    factory = PineconeDocumentStoreFactory(_rag_settings(), pinecone_client=fake_client)

    index_info = factory.describe_index()
    store = factory.create_document_store(900000002)

    assert index_info.name == "document-index"
    assert index_info.dimension == 1536
    assert store.namespace == "telegram-documents-user-900000002"
    assert store.dimension == 1536
    assert store.metric == "cosine"
    assert fake_client.describe_calls == ["document-index"]
    assert fake_client.index_calls == ["https://pinecone.invalid/index"]
    assert fake_client.create_index_calls == []


def test_pinecone_factory_delete_and_fetch_use_only_specified_ids() -> None:
    fake_index = FakePineconeIndex(existing_ids=("doc-3",))
    fake_client = FakePineconeClient(index_handle=fake_index)
    factory = PineconeDocumentStoreFactory(_rag_settings(), pinecone_client=fake_client)

    factory.delete_documents(123, ["doc-1", "doc-2", "doc-1"])
    remaining_ids = factory.fetch_existing_document_ids(123, ["doc-3", "doc-4"])

    assert fake_index.delete_calls == [
        {"ids": ["doc-1", "doc-2"], "namespace": "telegram-documents-user-123"}
    ]
    assert fake_index.fetch_calls == [
        {"ids": ["doc-3", "doc-4"], "namespace": "telegram-documents-user-123"}
    ]
    assert remaining_ids == ("doc-3",)


def test_pinecone_factory_delete_user_namespace_deletes_only_that_documents_namespace() -> None:
    fake_index = FakePineconeIndex(existing_ids=("doc-1",))
    fake_client = FakePineconeClient(index_handle=fake_index)
    factory = PineconeDocumentStoreFactory(_rag_settings(), pinecone_client=fake_client)

    factory.delete_user_namespace(123)

    assert fake_index.delete_all_calls == ["telegram-documents-user-123"]
    # No per-id deletion, no other namespace (notably not the v1 memory namespace).
    assert fake_index.delete_calls == []
    assert fake_client.create_index_calls == []


def test_pinecone_factory_delete_user_namespace_uses_a_different_namespace_per_user() -> None:
    fake_index = FakePineconeIndex()
    factory = PineconeDocumentStoreFactory(
        _rag_settings(), pinecone_client=FakePineconeClient(index_handle=fake_index)
    )

    factory.delete_user_namespace(123)
    factory.delete_user_namespace(456)

    assert fake_index.delete_all_calls == [
        "telegram-documents-user-123",
        "telegram-documents-user-456",
    ]


def test_pinecone_factory_delete_user_namespace_tolerates_missing_namespace() -> None:
    fake_index = FakePineconeIndex()
    fake_index.delete_exception = FakeNotFoundError("namespace not found")
    factory = PineconeDocumentStoreFactory(
        _rag_settings(), pinecone_client=FakePineconeClient(index_handle=fake_index)
    )

    factory.delete_user_namespace(123)  # must not raise: already absent == already deleted

    assert fake_index.delete_all_calls == []


def test_pinecone_factory_delete_user_namespace_failure_raises_safe_cleanup_error() -> None:
    fake_index = FakePineconeIndex()
    fake_index.delete_exception = RuntimeError("boom api-key=SECRET-123 host=internal.invalid")
    factory = PineconeDocumentStoreFactory(
        _rag_settings(), pinecone_client=FakePineconeClient(index_handle=fake_index)
    )

    with pytest.raises(DocumentCleanupError) as exc_info:
        factory.delete_user_namespace(123)

    assert "SECRET-123" not in str(exc_info.value)
    assert "internal.invalid" not in str(exc_info.value)
    assert "telegram-documents-user-123" not in str(exc_info.value)


def test_pinecone_factory_delete_user_namespace_unavailable_index_raises_cleanup_error() -> None:
    fake_client = FakePineconeClient(describe_exception=RuntimeError("index down"))
    factory = PineconeDocumentStoreFactory(_rag_settings(), pinecone_client=fake_client)

    with pytest.raises(DocumentCleanupError):
        factory.delete_user_namespace(123)


@pytest.mark.parametrize("user_id", [0, -5, True, "123", None])
def test_pinecone_factory_delete_user_namespace_rejects_invalid_user_id_before_any_call(
    user_id: object,
) -> None:
    fake_index = FakePineconeIndex()
    fake_client = FakePineconeClient(index_handle=fake_index)
    factory = PineconeDocumentStoreFactory(_rag_settings(), pinecone_client=fake_client)

    with pytest.raises((TypeError, ValueError)):
        factory.delete_user_namespace(user_id)

    assert fake_index.delete_all_calls == []
    assert fake_client.describe_calls == []


def test_pinecone_factory_missing_index_raises_controlled_error() -> None:
    fake_client = FakePineconeClient(describe_exception=FakeNotFoundError())
    factory = PineconeDocumentStoreFactory(_rag_settings(), pinecone_client=fake_client)

    with pytest.raises(DocumentIndexUnavailableError):
        factory.describe_index()

    assert fake_client.create_index_calls == []


def test_ingest_and_summarize_successful_conversion_write_and_summary() -> None:
    adapter = FakeAdapter(result=_conversion_result())
    store_factory = FakeStoreFactory()
    ingestion_pipeline = FakePipelineRunner(result={"writer": {"documents_written": 2}})
    summary_pipeline = FakePipelineRunner(
        result={
            "generator": {
                "replies": [
                    ChatMessage.from_assistant(
                        text=(
                            '"Документ описывает запуск пилотного проекта Orion, '
                            "его бюджет, порядок эскалации инцидентов и правила "
                            'пересмотра документа."'
                        )
                    )
                ]
            }
        }
    )
    service = DocumentRagService(
        _processing_settings(),
        _rag_settings(),
        adapter=adapter,
        document_store_factory=store_factory,
        ingestion_pipeline_factory=lambda settings, store: ingestion_pipeline,
        summary_pipeline_factory=lambda settings: summary_pipeline,
    )

    outcome = service.ingest_and_summarize(_request())

    assert outcome.file_hash == VALID_HASH
    assert outcome.chunk_count == 2
    assert outcome.documents_written == 2
    assert outcome.document_ids == ("doc-1", "doc-2")
    assert outcome.summary.startswith("Документ описывает запуск пилотного проекта Orion")
    assert "бюджет" in outcome.summary
    assert "эскалации" in outcome.summary
    assert "пересмотра документа" in outcome.summary
    assert store_factory.created_user_ids == [123]
    assert ingestion_pipeline.calls == [{"embedder": {"documents": list(_documents())}}]
    assert summary_pipeline.calls[0]["prompt_builder"]["file_name"] == "sample.pdf"
    assert "[Chunk 0]" in summary_pipeline.calls[0]["prompt_builder"]["document_context"]


def test_ingest_and_summarize_writer_count_mismatch_raises_controlled_error() -> None:
    service = DocumentRagService(
        _processing_settings(),
        _rag_settings(),
        adapter=FakeAdapter(result=_conversion_result()),
        document_store_factory=FakeStoreFactory(),
        ingestion_pipeline_factory=lambda settings, store: FakePipelineRunner(
            result={"writer": {"documents_written": 1}}
        ),
        summary_pipeline_factory=lambda settings: FakePipelineRunner(result={}),
    )

    with pytest.raises(DocumentIngestionError) as exc_info:
        service.ingest_and_summarize(_request())

    assert exc_info.value.document_ids == ("doc-1", "doc-2")


def test_ingest_and_summarize_embedding_or_write_failure_preserves_document_ids() -> None:
    runtime_error = RuntimeError("boom")
    service = DocumentRagService(
        _processing_settings(),
        _rag_settings(),
        adapter=FakeAdapter(result=_conversion_result()),
        document_store_factory=FakeStoreFactory(),
        ingestion_pipeline_factory=lambda settings, store: FakePipelineRunner(
            exception=runtime_error
        ),
        summary_pipeline_factory=lambda settings: FakePipelineRunner(result={}),
    )

    with pytest.raises(DocumentIngestionError) as exc_info:
        service.ingest_and_summarize(_request())

    assert exc_info.value.document_ids == ("doc-1", "doc-2")
    assert exc_info.value.__cause__ is runtime_error


def test_ingest_and_summarize_summary_failure_after_successful_write_preserves_document_ids(
) -> None:
    runtime_error = RuntimeError("summary failed")
    service = DocumentRagService(
        _processing_settings(),
        _rag_settings(),
        adapter=FakeAdapter(result=_conversion_result()),
        document_store_factory=FakeStoreFactory(),
        ingestion_pipeline_factory=lambda settings, store: FakePipelineRunner(
            result={"writer": {"documents_written": 2}}
        ),
        summary_pipeline_factory=lambda settings: FakePipelineRunner(exception=runtime_error),
    )

    with pytest.raises(DocumentSummaryError) as exc_info:
        service.ingest_and_summarize(_request())

    assert exc_info.value.document_ids == ("doc-1", "doc-2")
    assert exc_info.value.__cause__ is runtime_error


def test_answer_question_returns_grounded_sources_preserves_order_and_deduplicates_ids() -> None:
    retrieved_documents = (
        Document(
            id="doc-1",
            content="Budget chunk",
            score=0.9,
            meta={
                "file_name": "sample.pdf",
                "chunk_index": 0,
                "page_number": 3,
                "local_path": "C:/secret/sample.pdf",
                "arbitrary": "ignore-me",
            },
        ),
        Document(
            id="doc-1",
            content="Duplicate chunk",
            score=0.8,
            meta={"file_name": "sample.pdf", "chunk_index": 0},
        ),
        Document(
            id="doc-2",
            content="Second chunk",
            score=float("nan"),
            meta={"file_name": "sample.pdf", "chunk_index": 1},
        ),
    )
    embedder = FakeComponent(result={"embedding": [0.1, 0.2, 0.3]})
    retriever = FakeComponent(result={"documents": list(retrieved_documents)})
    prompt_builder = FakeComponent(result={"prompt": [ChatMessage.from_user(text="prompt")]})
    generator = FakeComponent(
        result=_reply(
            _contract_json(
                answer="Утвержденный бюджет пилотного проекта Orion составляет 4,2 миллиона евро.",
                # DOC_1 and DOC_2 share a storage id: cited twice, sourced once.
                source_ids=["DOC_1", "DOC_2", "DOC_3"],
            )
        )
    )
    store_factory = FakeStoreFactory()
    service = DocumentRagService(
        _processing_settings(),
        _rag_settings(),
        adapter=FakeAdapter(),
        document_store_factory=store_factory,
        rag_pipeline_factory=lambda settings, store: FakeRagPipeline(
            {
                "text_embedder": embedder,
                "retriever": retriever,
                "prompt_builder": prompt_builder,
                "generator": generator,
            }
        ),
    )

    answer = service.answer_question(123, "  What is the approved budget?  ")

    assert answer.answer.startswith("Утвержденный бюджет пилотного проекта Orion")
    assert "4,2" in answer.answer
    assert "миллиона евро" in answer.answer
    assert answer.fallback_used is False
    assert answer.used_document_count == 2
    assert [source.document_id for source in answer.sources] == ["doc-1", "doc-2"]
    assert answer.sources[0].file_name == "sample.pdf"
    assert answer.sources[0].chunk_index == 0
    assert answer.sources[0].page_number == 3
    assert answer.sources[0].score == 0.9
    assert answer.sources[1].score is None
    assert "local_path" not in answer.sources[0].model_dump(exclude_none=True)
    assert "arbitrary" not in answer.sources[0].model_dump(exclude_none=True)
    assert prompt_builder.calls == [
        {"question": "What is the approved budget?", "documents": list(retrieved_documents)}
    ]
    assert generator.calls == [{"messages": [ChatMessage.from_user(text="prompt")]}]


def test_answer_question_returns_exact_fallback_without_calling_generator_when_no_documents(
) -> None:
    embedder = FakeComponent(result={"embedding": [0.1, 0.2, 0.3]})
    retriever = FakeComponent(result={"documents": []})
    prompt_builder = FakeComponent(result={"prompt": [ChatMessage.from_user(text="prompt")]})
    generator = FakeComponent(result={"replies": [ChatMessage.from_assistant(text="unused")]})
    service = DocumentRagService(
        _processing_settings(),
        _rag_settings(),
        adapter=FakeAdapter(),
        document_store_factory=FakeStoreFactory(),
        rag_pipeline_factory=lambda settings, store: FakeRagPipeline(
            {
                "text_embedder": embedder,
                "retriever": retriever,
                "prompt_builder": prompt_builder,
                "generator": generator,
            }
        ),
    )

    answer = service.answer_question(123, "What is the approved budget?")

    assert answer.answer == INSUFFICIENT_DOCUMENT_ANSWER
    assert answer.sources == ()
    assert answer.used_document_count == 0
    assert answer.fallback_used is True
    assert prompt_builder.calls == []
    assert generator.calls == []


def test_answer_question_treats_the_old_sentinel_sentence_as_malformed_output() -> None:
    # The old contract matched this sentence exactly and *kept* the retrieved sources. Under
    # the machine contract it is just prose: a fallback with no sources at all.
    answer, _, _ = _answer_for_reply(INSUFFICIENT_DOCUMENT_ANSWER, _retrieved_documents(1))

    assert answer.answer == INSUFFICIENT_DOCUMENT_ANSWER  # inert placeholder, never routed on
    _assert_fallback(answer)


def test_english_document_chunks_still_allow_russian_summary_and_rag_answer() -> None:
    english_documents = (
        Document(
            id="doc-1",
            content="The Orion pilot starts on 15 September 2026.",
            meta={
                "record_type": "document_chunk",
                "user_id": 123,
                "file_name": "sample.pdf",
                "file_hash": VALID_HASH,
                "chunk_index": 0,
                "content_type": PDF_CONTENT_TYPE,
                "uploaded_at": UPLOAD_TIME.isoformat(),
            },
        ),
        Document(
            id="doc-2",
            content="The approved budget is 4.2 million euros.",
            meta={
                "record_type": "document_chunk",
                "user_id": 123,
                "file_name": "sample.pdf",
                "file_hash": VALID_HASH,
                "chunk_index": 1,
                "content_type": PDF_CONTENT_TYPE,
                "uploaded_at": UPLOAD_TIME.isoformat(),
                "page_number": 1,
            },
        ),
    )
    summary_pipeline = FakePipelineRunner(
        result={
            "generator": {
                "replies": [
                    ChatMessage.from_assistant(
                        text=(
                            "Документ описывает запуск пилотного проекта Orion, "
                            "его бюджет, порядок эскалации инцидентов и правила "
                            "пересмотра документа."
                        )
                    )
                ]
            }
        }
    )
    rag_pipeline = FakeRagPipeline(
        {
            "text_embedder": FakeComponent(result={"embedding": [0.1, 0.2, 0.3]}),
            "retriever": FakeComponent(result={"documents": list(english_documents)}),
            "prompt_builder": FakeComponent(
                result={"prompt": [ChatMessage.from_user(text="prompt")]}
            ),
            "generator": FakeComponent(
                result=_reply(
                    _contract_json(
                        answer=(
                            "Утвержденный бюджет пилотного проекта Orion "
                            "составляет 4,2 миллиона евро."
                        ),
                        source_ids=["DOC_2"],
                    )
                )
            ),
        }
    )
    service = DocumentRagService(
        _processing_settings(),
        _rag_settings(),
        adapter=FakeAdapter(result=_conversion_result(english_documents)),
        document_store_factory=FakeStoreFactory(),
        ingestion_pipeline_factory=lambda settings, store: FakePipelineRunner(
            result={"writer": {"documents_written": 2}}
        ),
        summary_pipeline_factory=lambda settings: summary_pipeline,
        rag_pipeline_factory=lambda settings, store: rag_pipeline,
    )

    outcome = service.ingest_and_summarize(_request())
    answer = service.answer_question(123, "Какой бюджет утвержден для пилотного проекта Orion?")

    assert "Orion" in outcome.summary
    assert "бюджет" in outcome.summary
    assert "4,2" in answer.answer
    assert "евро" in answer.answer
    assert "Orion" in answer.answer
    assert answer.fallback_used is False


def test_answer_question_rejects_blank_question_overlong_question_and_bool_user_id() -> None:
    service = DocumentRagService(
        _processing_settings(),
        _rag_settings(max_question_chars=5),
        adapter=FakeAdapter(),
        document_store_factory=FakeStoreFactory(),
        rag_pipeline_factory=lambda settings, store: FakeRagPipeline({}),
    )

    with pytest.raises(DocumentQuestionError):
        service.answer_question(123, "   ")
    with pytest.raises(DocumentQuestionError):
        service.answer_question(123, "123456")
    with pytest.raises(DocumentQuestionError):
        service.answer_question(True, "Valid?")


def test_service_uses_another_store_for_another_user_and_delete_uses_only_specified_ids() -> None:
    store_factory = FakeStoreFactory()
    service = DocumentRagService(
        _processing_settings(),
        _rag_settings(),
        adapter=FakeAdapter(),
        document_store_factory=store_factory,
        rag_pipeline_factory=lambda settings, store: FakeRagPipeline(
            {
                "text_embedder": FakeComponent(result={"embedding": [0.1]}),
                "retriever": FakeComponent(result={"documents": []}),
                "prompt_builder": FakeComponent(result={}),
                "generator": FakeComponent(result={}),
            }
        ),
    )

    service.answer_question(123, "Question?")
    service.answer_question(456, "Question?")
    service.delete_documents(456, ["doc-7", "doc-8"])

    assert store_factory.created_user_ids == [123, 456]
    assert store_factory.created_stores[123]["namespace"] == "telegram-documents-user-123"
    assert store_factory.created_stores[456]["namespace"] == "telegram-documents-user-456"
    assert store_factory.delete_calls == [(456, ("doc-7", "doc-8"))]


def test_service_delete_user_documents_deletes_the_users_whole_document_namespace() -> None:
    store_factory = FakeStoreFactory()
    service = DocumentRagService(
        _processing_settings(),
        _rag_settings(),
        adapter=FakeAdapter(),
        document_store_factory=store_factory,
    )

    service.delete_user_documents(456)

    assert store_factory.delete_namespace_calls == [456]
    assert store_factory.delete_calls == []
    assert store_factory.created_user_ids == []


def test_service_delete_user_documents_wraps_store_failure_in_safe_service_error() -> None:
    store_factory = FakeStoreFactory()
    store_factory.delete_namespace_exception = DocumentCleanupError("raw provider detail 9999")
    service = DocumentRagService(
        _processing_settings(),
        _rag_settings(),
        adapter=FakeAdapter(),
        document_store_factory=store_factory,
    )

    with pytest.raises(DocumentRagServiceError) as exc_info:
        service.delete_user_documents(456)

    assert str(exc_info.value) == "Document cleanup failed"
    assert "9999" not in str(exc_info.value)


@pytest.mark.parametrize("user_id", [0, -1, True, "7"])
def test_service_delete_user_documents_rejects_invalid_user_id_before_store_call(
    user_id: object,
) -> None:
    store_factory = FakeStoreFactory()
    service = DocumentRagService(
        _processing_settings(),
        _rag_settings(),
        adapter=FakeAdapter(),
        document_store_factory=store_factory,
    )

    with pytest.raises(DocumentRagServiceError):
        service.delete_user_documents(user_id)

    assert store_factory.delete_namespace_calls == []


# ---------------------------------------------------------------------------
# Document-vs-Agent routing: explicit, evidence-gated, fail-closed
# ---------------------------------------------------------------------------


def test_valid_contract_reply_returns_a_document_answer_with_only_the_cited_sources() -> None:
    documents = _retrieved_documents(3, scores=(0.9, 0.8, 0.7))

    answer, prompt_builder, generator = _answer_for_reply(
        _contract_json(source_ids=["DOC_2"]), documents
    )

    assert answer.fallback_used is False
    assert answer.answer == "Бюджет составляет 4,2 млн евро."
    # Only the cited chunk is a source -- not the other two that were retrieved.
    assert [source.document_id for source in answer.sources] == ["doc-2"]
    assert answer.used_document_count == 1
    assert answer.sources[0].chunk_index == 1
    assert answer.sources[0].page_number == 2
    # All three chunks were supplied to the model; the generator ran exactly once.
    assert prompt_builder.calls[0]["documents"] == documents
    assert len(generator.calls) == 1


def test_cited_sources_keep_citation_order_and_are_deduplicated() -> None:
    answer, _, _ = _answer_for_reply(_contract_json(source_ids=["DOC_3", "DOC_1", "DOC_3"]))

    assert [source.document_id for source in answer.sources] == ["doc-3", "doc-1"]


def test_sources_are_built_only_from_documents_actually_supplied_to_the_model() -> None:
    answer, prompt_builder, _ = _answer_for_reply(
        _contract_json(source_ids=["DOC_1", "DOC_3"]), _retrieved_documents(3)
    )

    supplied_ids = {document.id for document in prompt_builder.calls[0]["documents"]}
    assert {source.document_id for source in answer.sources} <= supplied_ids


def test_explicit_not_answerable_falls_back_even_though_documents_were_retrieved() -> None:
    answer, _, generator = _answer_for_reply(
        _contract_json(answerable=False, answer=None, source_ids=[]),
        _retrieved_documents(3, scores=(0.99, 0.98, 0.97)),
    )

    _assert_fallback(answer)
    assert len(generator.calls) == 1


_MALFORMED_REPLIES = [
    pytest.param("not json at all", id="prose"),
    pytest.param("[]", id="array"),
    pytest.param('{"answerable": true, "answer": "x"}', id="missing-source_ids"),
    pytest.param('{"answer": "x", "source_ids": ["DOC_1"]}', id="missing-answerable"),
    pytest.param(_contract_json().replace("true", '"true"'), id="answerable-string"),
    pytest.param('{"answerable": true, "answer": 5, "source_ids": ["DOC_1"]}', id="answer-number"),
    pytest.param(_contract_json(answer="   "), id="blank-answer"),
    pytest.param(_contract_json(answer=None), id="null-answer"),
    pytest.param(_contract_json(source_ids=[]), id="no-sources"),
    pytest.param(_contract_json(source_ids=["DOC_9"]), id="unknown-source"),
    pytest.param(_contract_json(source_ids=["doc-1"]), id="storage-id-instead-of-label"),
    pytest.param(_contract_json(source_ids=["DOC_1", "DOC_9"]), id="valid-and-fabricated-mixed"),
]


@pytest.mark.parametrize("reply_text", _MALFORMED_REPLIES)
def test_malformed_or_unsafe_generator_output_falls_back_without_any_document_answer(
    reply_text: str,
) -> None:
    answer, _, _ = _answer_for_reply(reply_text)

    _assert_fallback(answer)


_SENTINEL_VARIANTS = [
    pytest.param(INSUFFICIENT_DOCUMENT_ANSWER, id="exact"),
    pytest.param(f'"{INSUFFICIENT_DOCUMENT_ANSWER}"', id="quoted"),
    pytest.param(INSUFFICIENT_DOCUMENT_ANSWER + "!", id="plus-punctuation"),
    pytest.param(
        INSUFFICIENT_DOCUMENT_ANSWER + " Попробуйте уточнить вопрос.", id="plus-sentence"
    ),
    pytest.param("В документах нет данных, чтобы ответить на этот вопрос.", id="paraphrase"),
    pytest.param("NO_ANSWER", id="internal-token"),
]


@pytest.mark.parametrize("reply_text", _SENTINEL_VARIANTS)
def test_old_sentinel_variants_never_become_a_grounded_document_answer(reply_text: str) -> None:
    # Before the machine contract, every variant except the exact sentence was returned to the
    # user as a *grounded* answer together with a source block.
    answer, _, _ = _answer_for_reply(reply_text)

    _assert_fallback(answer)


def test_sentinel_text_inside_an_answerable_contract_reply_is_not_treated_as_fallback() -> None:
    # Routing reads the machine fields, not the prose: the sentence has no special authority.
    answer, _, _ = _answer_for_reply(
        _contract_json(answer=INSUFFICIENT_DOCUMENT_ANSWER, source_ids=["DOC_1"])
    )

    assert answer.fallback_used is False
    assert [source.document_id for source in answer.sources] == ["doc-1"]


def test_no_relevance_threshold_a_low_scored_cited_chunk_is_still_a_document_answer() -> None:
    answer, _, _ = _answer_for_reply(
        _contract_json(source_ids=["DOC_1"]),
        _retrieved_documents(1, scores=(0.01,)),
    )

    assert answer.fallback_used is False
    assert answer.sources[0].score == 0.01  # existing runtime data preserved, not filtered


def test_zero_retrieved_documents_falls_back_without_calling_the_model() -> None:
    answer, prompt_builder, generator = _answer_for_reply(_contract_json(), documents=[])

    _assert_fallback(answer)
    assert prompt_builder.calls == []
    assert generator.calls == []


def test_expected_fallback_is_logged_at_info_with_only_a_reason_code(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger="hay_v2_bot.services.document_rag"):
        _answer_for_reply(_contract_json(answerable=False, answer=None, source_ids=[]))

    records = [r for r in caplog.records if "document_answer_fallback" in r.getMessage()]
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (logging.INFO, "event=document_answer_fallback reason=not_answerable")
    ]


@pytest.mark.parametrize(
    ("reply_text", "reason"),
    [
        ("TOP-SECRET-USER-DATA prose", "invalid_format"),
        (_contract_json(answer="TOP-SECRET-USER-DATA", source_ids=["DOC_9"]), "unknown_source"),
        (_contract_json(answer="TOP-SECRET-USER-DATA", source_ids=[]), "no_sources"),
    ],
)
def test_malformed_fallback_is_a_warning_and_never_logs_model_output(
    caplog: pytest.LogCaptureFixture, reply_text: str, reason: str
) -> None:
    with caplog.at_level(logging.DEBUG, logger="hay_v2_bot.services.document_rag"):
        _answer_for_reply(reply_text)

    records = [r for r in caplog.records if "document_answer_fallback" in r.getMessage()]
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (logging.WARNING, f"event=document_answer_fallback reason={reason}")
    ]
    assert "TOP-SECRET-USER-DATA" not in caplog.text


@pytest.mark.parametrize("replies", [[], [ChatMessage.from_assistant(text="   ")]])
def test_reply_with_no_usable_text_is_still_a_controlled_question_error(
    replies: list[ChatMessage],
) -> None:
    # Structural failures of the chat reply (nothing to parse, as opposed to a reply that
    # violates the contract) keep their old behaviour: a controlled DocumentQuestionError,
    # which the Telegram handler turns into an Agent fallback.
    service = DocumentRagService(
        _processing_settings(),
        _rag_settings(),
        adapter=FakeAdapter(),
        document_store_factory=FakeStoreFactory(),
        rag_pipeline_factory=lambda settings, store: FakeRagPipeline(
            {
                "text_embedder": FakeComponent(result={"embedding": [0.1]}),
                "retriever": FakeComponent(result={"documents": _retrieved_documents(1)}),
                "prompt_builder": FakeComponent(
                    result={"prompt": [ChatMessage.from_user(text="prompt")]}
                ),
                "generator": FakeComponent(result={"replies": replies}),
            }
        ),
    )

    with pytest.raises(DocumentQuestionError):
        service.answer_question(123, "Какой бюджет?")


# ---------------------------------------------------------------------------
# Upload lifecycle: summary-only failure and partial writes
# ---------------------------------------------------------------------------


class WritingPipeline:
    """Fake ingestion pipeline that really "stores" the documents it is given."""

    def __init__(self, stored: dict[str, Document]) -> None:
        self._stored = stored

    def run(self, payload: Any) -> Any:
        for document in payload["embedder"]["documents"]:
            self._stored[document.id] = document
        return {"writer": {"documents_written": len(payload["embedder"]["documents"])}}


def test_summary_only_failure_keeps_the_indexed_chunks_and_they_stay_answerable() -> None:
    stored: dict[str, Document] = {}
    store_factory = FakeStoreFactory()
    service = DocumentRagService(
        _processing_settings(),
        _rag_settings(),
        adapter=FakeAdapter(result=_conversion_result()),
        document_store_factory=store_factory,
        ingestion_pipeline_factory=lambda settings, store: WritingPipeline(stored),
        summary_pipeline_factory=lambda settings: FakePipelineRunner(
            exception=RuntimeError("summary provider down")
        ),
        rag_pipeline_factory=lambda settings, store: FakeRagPipeline(
            {
                "text_embedder": FakeComponent(result={"embedding": [0.1]}),
                "retriever": FakeComponent(
                    result=lambda **_: {"documents": list(stored.values())}
                ),
                "prompt_builder": FakeComponent(
                    result={"prompt": [ChatMessage.from_user(text="prompt")]}
                ),
                "generator": FakeComponent(
                    result=_reply(_contract_json(source_ids=["DOC_2"]))
                ),
            }
        ),
    )

    with pytest.raises(DocumentSummaryError) as exc_info:
        service.ingest_and_summarize(_request())

    # Indexing succeeded, so the failure is reported *with* the ids and nothing is deleted.
    assert exc_info.value.document_ids == ("doc-1", "doc-2")
    assert set(stored) == {"doc-1", "doc-2"}
    assert store_factory.delete_calls == []
    assert store_factory.delete_namespace_calls == []

    # The stored document is still usable for grounded answers.
    answer = service.answer_question(123, "Какой бюджет?")
    assert answer.fallback_used is False
    assert [source.document_id for source in answer.sources] == ["doc-2"]


def _service_reporting_write_count(
    written: int,
    store_factory: FakeStoreFactory,
    *,
    pipeline_exception: BaseException | None = None,
    summary_pipeline: FakePipelineRunner | None = None,
) -> DocumentRagService:
    if pipeline_exception is not None:
        ingestion_pipeline = FakePipelineRunner(exception=pipeline_exception)
    else:
        ingestion_pipeline = FakePipelineRunner(result={"writer": {"documents_written": written}})
    summary = summary_pipeline if summary_pipeline is not None else FakePipelineRunner(result={})
    return DocumentRagService(
        _processing_settings(),
        _rag_settings(),
        adapter=FakeAdapter(result=_conversion_result()),  # two chunks: doc-1, doc-2
        document_store_factory=store_factory,
        ingestion_pipeline_factory=lambda settings, store: ingestion_pipeline,
        summary_pipeline_factory=lambda settings: summary,
    )


def test_partial_write_is_reported_as_failed_without_any_destructive_cleanup() -> None:
    store_factory = FakeStoreFactory()
    service = _service_reporting_write_count(1, store_factory)  # 1 of 2 chunks committed

    with pytest.raises(DocumentIngestionError) as exc_info:
        service.ingest_and_summarize(_request(user_id=123))

    # Reported as a failed/incomplete ingestion with the existing fixed, user-safe message
    # (the handler maps DocumentIngestionError to PROCESSING_FAILURE_MESSAGE) ...
    assert str(exc_info.value) == "Document store reported an unexpected write count"
    assert exc_info.value.document_ids == ("doc-1", "doc-2")
    # ... and nothing is deleted: the deterministic ids may belong to an earlier upload.
    assert store_factory.delete_calls == []
    assert store_factory.delete_namespace_calls == []


@pytest.mark.parametrize("written", [0, 1, 3])
def test_no_write_count_mismatch_ever_deletes_anything(written: int) -> None:
    # 0: nothing committed. 1 of 2: a partial write. 3 of 2: more than expected.
    store_factory = FakeStoreFactory()
    service = _service_reporting_write_count(written, store_factory)

    with pytest.raises(DocumentIngestionError):
        service.ingest_and_summarize(_request())

    assert store_factory.delete_calls == []
    assert store_factory.delete_namespace_calls == []


class PartiallyCommittingPipeline:
    """Fake ingestion pipeline that upserts by id but only commits the first N documents.

    Mirrors the Pinecone SDK, which reports failed upsert batches through a short
    ``documents_written`` count instead of raising.
    """

    def __init__(self, stored: dict[str, Document], commit_limit: int | None) -> None:
        self._stored = stored
        self.commit_limit = commit_limit  # None: commit everything

    def run(self, payload: Any) -> Any:
        documents = payload["embedder"]["documents"]
        committed = documents if self.commit_limit is None else documents[: self.commit_limit]
        for document in committed:
            self._stored[document.id] = document  # upsert/overwrite by deterministic id
        return {"writer": {"documents_written": len(committed)}}


def _service_over_partially_committing_store(
    stored: dict[str, Document],
    pipeline: PartiallyCommittingPipeline,
    store_factory: FakeStoreFactory,
) -> DocumentRagService:
    return DocumentRagService(
        _processing_settings(),
        _rag_settings(),
        adapter=FakeAdapter(result=_conversion_result()),  # two chunks: doc-1, doc-2
        document_store_factory=store_factory,
        ingestion_pipeline_factory=lambda settings, store: pipeline,
        summary_pipeline_factory=lambda settings: FakePipelineRunner(
            result={
                "generator": {
                    "replies": [ChatMessage.from_assistant(text="Документ описывает пилот Orion.")]
                }
            }
        ),
    )


def test_failed_reupload_of_an_identical_file_keeps_the_earlier_successful_chunks() -> None:
    # The hazard that rules out cleanup: chunk ids are derived from the file content, so a
    # re-upload of the same file targets the very ids an earlier, complete upload wrote.
    stored: dict[str, Document] = {}
    store_factory = FakeStoreFactory()
    pipeline = PartiallyCommittingPipeline(stored, commit_limit=None)
    service = _service_over_partially_committing_store(stored, pipeline, store_factory)
    service.ingest_and_summarize(_request())  # first upload: complete
    assert set(stored) == {"doc-1", "doc-2"}

    pipeline.commit_limit = 1  # the identical re-upload fails part-way
    with pytest.raises(DocumentIngestionError):
        service.ingest_and_summarize(_request())

    assert set(stored) == {"doc-1", "doc-2"}  # the earlier upload's chunks survive
    assert store_factory.delete_calls == []
    assert store_factory.delete_namespace_calls == []


def test_retry_of_the_same_document_completes_after_a_partial_write() -> None:
    stored: dict[str, Document] = {}
    store_factory = FakeStoreFactory()
    pipeline = PartiallyCommittingPipeline(stored, commit_limit=1)
    service = _service_over_partially_committing_store(stored, pipeline, store_factory)

    with pytest.raises(DocumentIngestionError):
        service.ingest_and_summarize(_request())  # only doc-1 committed; reported as failed
    assert set(stored) == {"doc-1"}

    pipeline.commit_limit = None  # the user sends the same file again
    outcome = service.ingest_and_summarize(_request())

    assert set(stored) == {"doc-1", "doc-2"}
    assert outcome.documents_written == outcome.chunk_count == 2
    assert store_factory.delete_calls == []
    assert store_factory.delete_namespace_calls == []


def test_embedding_or_pipeline_exception_does_not_trigger_cleanup() -> None:
    # The SDK reports failed upsert batches through the write count rather than raising, so a
    # raised pipeline error (typically the embedder) comes with no evidence of committed
    # chunks; deleting the deterministic ids could remove an identical, earlier upload.
    store_factory = FakeStoreFactory()
    service = _service_reporting_write_count(
        0, store_factory, pipeline_exception=RuntimeError("embedder rate limited")
    )

    with pytest.raises(DocumentIngestionError):
        service.ingest_and_summarize(_request())

    assert store_factory.delete_calls == []


def test_successful_ingestion_never_deletes_anything() -> None:
    store_factory = FakeStoreFactory()
    service = _service_reporting_write_count(
        2,
        store_factory,
        summary_pipeline=FakePipelineRunner(
            result={
                "generator": {
                    "replies": [ChatMessage.from_assistant(text="Документ описывает пилот Orion.")]
                }
            }
        ),
    )

    outcome = service.ingest_and_summarize(_request())

    assert outcome.summary == "Документ описывает пилот Orion."
    assert outcome.documents_written == 2
    assert store_factory.delete_calls == []
    assert store_factory.delete_namespace_calls == []
