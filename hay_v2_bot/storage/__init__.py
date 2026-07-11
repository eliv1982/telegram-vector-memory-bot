"""Storage helpers for hay_v2_bot."""

from .identifiers import build_document_chunk_id, validate_file_hash
from .namespaces import DOCUMENT_NAMESPACE_PREFIX, document_namespace_for_user
from .pinecone import (
    DocumentCleanupError,
    DocumentIndexContractError,
    DocumentIndexInfo,
    DocumentIndexUnavailableError,
    DocumentStoreError,
    PineconeDocumentStoreFactory,
)

__all__ = [
    "DOCUMENT_NAMESPACE_PREFIX",
    "build_document_chunk_id",
    "document_namespace_for_user",
    "validate_file_hash",
    "DocumentCleanupError",
    "DocumentIndexContractError",
    "DocumentIndexInfo",
    "DocumentIndexUnavailableError",
    "DocumentStoreError",
    "PineconeDocumentStoreFactory",
]
