"""Deterministic storage identifiers for v2 document records."""

from __future__ import annotations

import re
from typing import Final

_FILE_HASH_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")


def validate_file_hash(file_hash: object) -> str:
    """Validate a lowercase SHA-256 file hash and return it unchanged."""
    if not isinstance(file_hash, str):
        raise TypeError("file_hash must be a string")
    if not _FILE_HASH_PATTERN.fullmatch(file_hash):
        raise ValueError("file_hash must be exactly 64 lowercase hexadecimal characters")
    return file_hash


def build_document_chunk_id(file_hash: str, chunk_index: int) -> str:
    """Build a deterministic Pinecone record ID for a document chunk."""
    validated_hash = validate_file_hash(file_hash)
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int):
        raise TypeError("chunk_index must be a non-boolean integer")
    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative")
    return f"doc-{validated_hash}-chunk-{chunk_index:06d}"
