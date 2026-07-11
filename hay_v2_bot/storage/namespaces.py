"""Namespace helpers for v2 document storage."""

from __future__ import annotations

from typing import Final

DOCUMENT_NAMESPACE_PREFIX: Final = "telegram-documents-user"


def document_namespace_for_user(user_id: int) -> str:
    """Return the per-user Pinecone namespace for uploaded documents."""
    if isinstance(user_id, bool) or not isinstance(user_id, int):
        raise TypeError("user_id must be a non-boolean integer")
    if user_id <= 0:
        raise ValueError("user_id must be positive")
    return f"{DOCUMENT_NAMESPACE_PREFIX}-{user_id}"
