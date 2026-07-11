"""Document metadata models for hay_v2_bot."""

from .conversion import DocumentConversionRequest, DocumentConversionResult
from .documents import (
    DOCX_CONTENT_TYPE,
    PDF_CONTENT_TYPE,
    SUPPORTED_DOCUMENT_CONTENT_TYPES,
    DocumentChunkMetadata,
)
from .rag import (
    INSUFFICIENT_DOCUMENT_ANSWER,
    DocumentAnswer,
    DocumentIngestionOutcome,
    DocumentSource,
)

__all__ = [
    "DOCX_CONTENT_TYPE",
    "PDF_CONTENT_TYPE",
    "SUPPORTED_DOCUMENT_CONTENT_TYPES",
    "DocumentConversionRequest",
    "DocumentConversionResult",
    "DocumentChunkMetadata",
    "DocumentAnswer",
    "DocumentIngestionOutcome",
    "DocumentSource",
    "INSUFFICIENT_DOCUMENT_ANSWER",
]
