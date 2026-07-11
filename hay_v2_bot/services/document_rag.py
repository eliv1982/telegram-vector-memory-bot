"""Document RAG service built on the v2 Docling adapter and Haystack pipelines."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from haystack import Pipeline

from hay_v2_bot.adapters import DoclingDocumentAdapter, DocumentAdapterError
from hay_v2_bot.components import (
    build_sources,
    build_summary_context,
    extract_chat_reply_text,
    normalize_single_sentence_summary,
)
from hay_v2_bot.config import DocumentProcessingSettings, DocumentRagSettings
from hay_v2_bot.models import (
    INSUFFICIENT_DOCUMENT_ANSWER,
    DocumentAnswer,
    DocumentConversionRequest,
    DocumentIngestionOutcome,
)
from hay_v2_bot.pipelines import (
    build_ingestion_pipeline,
    build_rag_pipeline,
    build_summary_pipeline,
)
from hay_v2_bot.storage import DocumentStoreError, PineconeDocumentStoreFactory


class DocumentRagServiceError(Exception):
    """Base class for safe public document RAG service failures."""

    def __init__(self, message: str, *, document_ids: Sequence[str] | None = None) -> None:
        super().__init__(message)
        self.document_ids = tuple(document_ids or ())


class DocumentIngestionError(DocumentRagServiceError):
    """Raised when document conversion, embedding, or writing fails."""


class DocumentSummaryError(DocumentRagServiceError):
    """Raised when summary generation fails after a successful write path."""


class DocumentQuestionError(DocumentRagServiceError):
    """Raised when a grounded document answer cannot be produced safely."""


class DocumentRagService:
    """Convert, store, summarize, retrieve, and answer document questions."""

    def __init__(
        self,
        processing_settings: DocumentProcessingSettings,
        rag_settings: DocumentRagSettings,
        *,
        adapter: DoclingDocumentAdapter | None = None,
        document_store_factory: PineconeDocumentStoreFactory | None = None,
        ingestion_pipeline_factory: Callable[[DocumentRagSettings, Any], Pipeline] | None = None,
        summary_pipeline_factory: Callable[[DocumentRagSettings], Pipeline] | None = None,
        rag_pipeline_factory: Callable[[DocumentRagSettings, Any], Pipeline] | None = None,
    ) -> None:
        self._processing_settings = processing_settings
        self._rag_settings = rag_settings
        self._adapter = (
            adapter if adapter is not None else DoclingDocumentAdapter(settings=processing_settings)
        )
        self._document_store_factory = (
            document_store_factory
            if document_store_factory is not None
            else PineconeDocumentStoreFactory(rag_settings)
        )
        self._ingestion_pipeline_factory = (
            ingestion_pipeline_factory
            if ingestion_pipeline_factory is not None
            else build_ingestion_pipeline
        )
        self._summary_pipeline_factory = (
            summary_pipeline_factory
            if summary_pipeline_factory is not None
            else build_summary_pipeline
        )
        self._rag_pipeline_factory = (
            rag_pipeline_factory if rag_pipeline_factory is not None else build_rag_pipeline
        )

    def ingest_and_summarize(
        self,
        request: DocumentConversionRequest,
    ) -> DocumentIngestionOutcome:
        """Convert one document, write its chunks, and summarize it in one sentence."""
        try:
            conversion_result = self._adapter.convert(request)
        except DocumentAdapterError as exc:
            raise DocumentIngestionError("Document conversion failed") from exc

        document_ids = tuple(document.id for document in conversion_result.documents)
        try:
            document_store = self._document_store_factory.create_document_store(request.user_id)
        except DocumentStoreError as exc:
            raise DocumentIngestionError("Document store is unavailable") from exc

        try:
            ingestion_pipeline = self._ingestion_pipeline_factory(
                self._rag_settings,
                document_store,
            )
            ingestion_result = ingestion_pipeline.run(
                {"embedder": {"documents": list(conversion_result.documents)}}
            )
        except Exception as exc:
            raise DocumentIngestionError(
                "Embedding or document write failed",
                document_ids=document_ids,
            ) from exc

        try:
            documents_written = _extract_documents_written(ingestion_result)
        except DocumentIngestionError as exc:
            raise DocumentIngestionError(str(exc), document_ids=document_ids) from exc
        if documents_written != conversion_result.chunk_count:
            raise DocumentIngestionError(
                "Document store reported an unexpected write count",
                document_ids=document_ids,
            )

        try:
            summary_context = build_summary_context(
                conversion_result.documents,
                self._rag_settings.max_summary_chars,
            )
            summary_pipeline = self._summary_pipeline_factory(self._rag_settings)
            summary_result = summary_pipeline.run(
                {
                    "prompt_builder": {
                        "file_name": conversion_result.file_name,
                        "document_context": summary_context,
                    }
                }
            )
            summary_text = extract_chat_reply_text(_extract_replies(summary_result))
            summary = normalize_single_sentence_summary(summary_text)
        except Exception as exc:
            raise DocumentSummaryError(
                "Document summary generation failed",
                document_ids=document_ids,
            ) from exc

        return DocumentIngestionOutcome(
            file_hash=conversion_result.file_hash,
            file_name=conversion_result.file_name,
            content_type=conversion_result.content_type,
            chunk_count=conversion_result.chunk_count,
            documents_written=documents_written,
            document_ids=document_ids,
            summary=summary,
        )

    def answer_question(self, user_id: int, question: str) -> DocumentAnswer:
        """Answer one user question using only the user's stored document chunks."""
        user_id = _validate_user_id(user_id)
        normalized_question = _normalize_question(question, self._rag_settings.max_question_chars)
        try:
            document_store = self._document_store_factory.create_document_store(user_id)
        except DocumentStoreError as exc:
            raise DocumentQuestionError("Document store is unavailable") from exc

        try:
            rag_pipeline = self._rag_pipeline_factory(self._rag_settings, document_store)
            text_embedder = rag_pipeline.get_component("text_embedder")
            retriever = rag_pipeline.get_component("retriever")
            prompt_builder = rag_pipeline.get_component("prompt_builder")
            generator = rag_pipeline.get_component("generator")

            embedding_result = text_embedder.run(text=normalized_question)
            query_embedding = embedding_result.get("embedding")
            retrieval_result = retriever.run(query_embedding=query_embedding)
            retrieved_documents = tuple(retrieval_result.get("documents", ()))
            if not retrieved_documents:
                return DocumentAnswer(
                    answer=INSUFFICIENT_DOCUMENT_ANSWER,
                    sources=(),
                    used_document_count=0,
                    fallback_used=True,
                )

            prompt_result = prompt_builder.run(
                question=normalized_question,
                documents=list(retrieved_documents),
            )
            generator_result = generator.run(messages=prompt_result["prompt"])
            answer_text = extract_chat_reply_text(generator_result.get("replies", ()))
            sources = build_sources(retrieved_documents)
        except DocumentRagServiceError:
            raise
        except Exception as exc:
            raise DocumentQuestionError("Question answering failed") from exc

        return DocumentAnswer(
            answer=answer_text,
            sources=sources,
            used_document_count=len(sources),
            fallback_used=answer_text == INSUFFICIENT_DOCUMENT_ANSWER,
        )

    def delete_documents(self, user_id: int, document_ids: Sequence[str]) -> None:
        """Delete only the specified document IDs from the user's document namespace."""
        user_id = _validate_user_id(user_id)
        if (
            isinstance(document_ids, str | bytes)
            or not isinstance(document_ids, Sequence)
            or not document_ids
        ):
            raise DocumentRagServiceError("document_ids must be a non-empty sequence")
        try:
            self._document_store_factory.delete_documents(user_id, document_ids)
        except DocumentStoreError as exc:
            raise DocumentRagServiceError("Document cleanup failed") from exc


def _validate_user_id(user_id: int) -> int:
    if isinstance(user_id, bool) or not isinstance(user_id, int):
        raise DocumentQuestionError("user_id must be a positive integer")
    if user_id <= 0:
        raise DocumentQuestionError("user_id must be a positive integer")
    return user_id


def _normalize_question(question: str, max_question_chars: int) -> str:
    if not isinstance(question, str):
        raise DocumentQuestionError("Question must be a string")
    normalized = question.strip()
    if not normalized:
        raise DocumentQuestionError("Question must not be empty")
    if len(normalized) > max_question_chars:
        raise DocumentQuestionError("Question exceeds the configured length limit")
    return normalized


def _extract_documents_written(pipeline_result: Any) -> int:
    writer_result = pipeline_result.get("writer") if isinstance(pipeline_result, dict) else None
    documents_written = (
        writer_result.get("documents_written") if isinstance(writer_result, dict) else None
    )
    if isinstance(documents_written, bool) or not isinstance(documents_written, int):
        raise DocumentIngestionError("Document store reported an invalid write count")
    if documents_written < 0:
        raise DocumentIngestionError("Document store reported an invalid write count")
    return documents_written


def _extract_replies(pipeline_result: Any) -> Any:
    generator_result = (
        pipeline_result.get("generator") if isinstance(pipeline_result, dict) else None
    )
    replies = generator_result.get("replies") if isinstance(generator_result, dict) else None
    if replies is None:
        raise DocumentSummaryError("Document summary generation failed")
    return replies
