"""Offline tests for hay_v2_bot.services.document_rag and v2 Pinecone storage."""

from __future__ import annotations

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
    DocumentSummaryError,
)
from hay_v2_bot.storage import (
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


class FakeNotFoundError(Exception):
    status_code = 404


class FakePineconeIndex:
    def __init__(self, *, existing_ids: Sequence[str] = ()) -> None:
        self.delete_calls: list[dict[str, object]] = []
        self.fetch_calls: list[dict[str, object]] = []
        self.existing_ids = set(existing_ids)

    def delete(self, *, ids: list[str] | None = None, namespace: str = "", **_: Any) -> None:
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
        result={
            "replies": [
                ChatMessage.from_assistant(
                    text="Утвержденный бюджет пилотного проекта Orion составляет 4,2 миллиона евро."
                )
            ]
        }
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


def test_answer_question_marks_exact_fallback_reply() -> None:
    retrieved_documents = [
        Document(id="doc-1", content="Chunk", meta={"file_name": "sample.pdf", "chunk_index": 0})
    ]
    service = DocumentRagService(
        _processing_settings(),
        _rag_settings(),
        adapter=FakeAdapter(),
        document_store_factory=FakeStoreFactory(),
        rag_pipeline_factory=lambda settings, store: FakeRagPipeline(
            {
                "text_embedder": FakeComponent(result={"embedding": [0.1]}),
                "retriever": FakeComponent(result={"documents": retrieved_documents}),
                "prompt_builder": FakeComponent(
                    result={"prompt": [ChatMessage.from_user(text="prompt")]}
                ),
                "generator": FakeComponent(
                    result={
                        "replies": [
                            ChatMessage.from_assistant(text=INSUFFICIENT_DOCUMENT_ANSWER)
                        ]
                    }
                ),
            }
        ),
    )

    answer = service.answer_question(123, "What is the approved budget?")

    assert answer.answer == INSUFFICIENT_DOCUMENT_ANSWER
    assert answer.fallback_used is True
    assert answer.used_document_count == 1


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
                result={
                    "replies": [
                        ChatMessage.from_assistant(
                            text=(
                                "Утвержденный бюджет пилотного проекта Orion "
                                "составляет 4,2 миллиона евро."
                            )
                        )
                    ]
                }
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
