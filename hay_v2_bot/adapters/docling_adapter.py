"""Offline-safe adapter around Docling document conversion."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

from haystack import Document
from haystack_integrations.components.converters.docling import (
    DoclingConverter,
    ExportType,
)
from pydantic import ValidationError

from hay_v2_bot.config import DocumentProcessingSettings
from hay_v2_bot.models import (
    DOCX_CONTENT_TYPE,
    PDF_CONTENT_TYPE,
    DocumentChunkMetadata,
    DocumentConversionRequest,
    DocumentConversionResult,
)
from hay_v2_bot.storage import build_document_chunk_id

_HASH_BLOCK_SIZE = 1024 * 1024
_SUFFIX_BY_CONTENT_TYPE = {
    PDF_CONTENT_TYPE: ".pdf",
    DOCX_CONTENT_TYPE: ".docx",
}


class DocumentAdapterError(Exception):
    """Base class for document adapter failures."""


class InvalidDocumentInputError(DocumentAdapterError):
    """Raised when a document request or file is invalid."""


class DocumentTooLargeError(InvalidDocumentInputError):
    """Raised when a document exceeds the configured size limit."""


class UnsupportedDocumentTypeError(InvalidDocumentInputError):
    """Raised when a document type or suffix is unsupported."""


class DocumentConversionError(DocumentAdapterError):
    """Raised when the underlying converter fails or returns invalid output."""


class EmptyDocumentError(InvalidDocumentInputError):
    """Raised when a document yields no non-blank textual chunks."""


class TooManyDocumentChunksError(InvalidDocumentInputError):
    """Raised when a converted document exceeds the chunk limit."""


class DoclingConverterProtocol(Protocol):
    """Narrow typed boundary for injected Docling-compatible converters."""

    def run(
        self,
        *,
        paths: list[str | Path] | None = None,
        sources: Any = None,
        meta: Any = None,
    ) -> Mapping[str, object]:
        """Convert one or more documents into Haystack documents."""


class DoclingDocumentAdapter:
    """Validate a local document file and normalize Docling chunk output."""

    def __init__(
        self,
        settings: DocumentProcessingSettings,
        converter: DoclingConverterProtocol | None = None,
    ) -> None:
        self._settings = settings
        self._converter = (
            converter
            if converter is not None
            else DoclingConverter(export_type=ExportType.DOC_CHUNKS)
        )

    def convert(self, request: DocumentConversionRequest) -> DocumentConversionResult:
        """Convert one validated local document into normalized chunk records."""
        self._validate_file(request)
        file_hash = self._compute_file_hash(request.local_path)
        converter_documents = self._run_converter(request.local_path)
        documents = self._normalize_documents(
            converter_documents=converter_documents,
            request=request,
            file_hash=file_hash,
        )
        return DocumentConversionResult(
            file_hash=file_hash,
            file_name=request.file_name,
            content_type=request.content_type,
            documents=tuple(documents),
        )

    def _validate_file(self, request: DocumentConversionRequest) -> None:
        path = request.local_path
        if not path.exists():
            raise InvalidDocumentInputError("Document file does not exist")
        if not path.is_file():
            raise InvalidDocumentInputError("Document path must point to a regular file")
        if path.is_symlink():
            raise InvalidDocumentInputError("Symbolic links are not allowed")

        file_size = path.stat().st_size
        if file_size == 0:
            raise InvalidDocumentInputError("Document file must not be empty")
        if file_size > self._settings.max_file_bytes:
            raise DocumentTooLargeError("Document file exceeds the configured size limit")

        expected_suffix = self._expected_suffix(request.content_type)
        if path.suffix.lower() != expected_suffix:
            raise UnsupportedDocumentTypeError(
                "Document file suffix does not match the declared content type"
            )
        if Path(request.file_name).suffix.lower() != expected_suffix:
            raise UnsupportedDocumentTypeError(
                "Original filename suffix does not match the declared content type"
            )

    def _expected_suffix(self, content_type: str) -> str:
        try:
            return _SUFFIX_BY_CONTENT_TYPE[content_type]
        except KeyError as exc:
            raise UnsupportedDocumentTypeError("Unsupported document content type") from exc

    def _compute_file_hash(self, path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_HASH_BLOCK_SIZE), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _run_converter(self, path: Path) -> list[Document]:
        try:
            result = self._converter.run(paths=[path])
        except Exception as exc:
            raise DocumentConversionError("Document conversion failed") from exc

        if not isinstance(result, Mapping) or "documents" not in result:
            raise DocumentConversionError("Converter returned an invalid result")

        documents = result["documents"]
        if isinstance(documents, str | bytes) or not isinstance(documents, Iterable):
            raise DocumentConversionError("Converter returned an invalid result")

        converted_documents = list(documents)
        for document in converted_documents:
            if not isinstance(document, Document):
                raise DocumentConversionError("Converter returned an invalid document chunk")
        return converted_documents

    def _normalize_documents(
        self,
        *,
        converter_documents: Iterable[Document],
        request: DocumentConversionRequest,
        file_hash: str,
    ) -> list[Document]:
        normalized_documents: list[Document] = []

        for source_document in converter_documents:
            if not isinstance(source_document.content, str):
                raise DocumentConversionError("Converter returned a non-text document chunk")

            content = source_document.content.strip()
            if not content:
                continue

            chunk_index = len(normalized_documents)
            if chunk_index >= self._settings.max_chunks_per_document:
                raise TooManyDocumentChunksError(
                    "Document produced too many non-blank chunks"
                )

            metadata = self._build_chunk_metadata(
                source_document=source_document,
                request=request,
                file_hash=file_hash,
                chunk_index=chunk_index,
            )
            normalized_documents.append(
                Document(
                    id=build_document_chunk_id(file_hash, chunk_index),
                    content=content,
                    meta=metadata.to_pinecone_metadata(),
                )
            )

        if not normalized_documents:
            raise EmptyDocumentError("Document produced no non-blank text chunks")
        return normalized_documents

    def _build_chunk_metadata(
        self,
        *,
        source_document: Document,
        request: DocumentConversionRequest,
        file_hash: str,
        chunk_index: int,
    ) -> DocumentChunkMetadata:
        source_meta = source_document.meta
        if source_meta is None:
            source_meta = {}
        if not isinstance(source_meta, Mapping):
            raise DocumentConversionError("Converter returned invalid document metadata")

        metadata_payload: dict[str, object] = {
            "user_id": request.user_id,
            "file_name": request.file_name,
            "file_hash": file_hash,
            "chunk_index": chunk_index,
            "content_type": request.content_type,
            "uploaded_at": request.uploaded_at,
        }
        if "page_number" in source_meta:
            metadata_payload["page_number"] = source_meta["page_number"]
        if "headings" in source_meta:
            metadata_payload["headings"] = source_meta["headings"]

        try:
            return DocumentChunkMetadata(**metadata_payload)
        except ValidationError as exc:
            raise InvalidDocumentInputError("Document chunk metadata is invalid") from exc
