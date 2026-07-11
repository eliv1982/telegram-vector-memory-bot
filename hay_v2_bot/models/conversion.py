"""Immutable request and result models for document conversion."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from haystack import Document
from pydantic import BaseModel, ConfigDict, StrictStr, field_validator

from hay_v2_bot.storage import validate_file_hash

from .documents import (
    StrictPositiveInt,
    SupportedDocumentContentType,
    normalize_utc_datetime,
    validate_base_file_name,
)


class DocumentConversionRequest(BaseModel):
    """Immutable request for converting one local document file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    local_path: Path
    user_id: StrictPositiveInt
    file_name: StrictStr
    content_type: SupportedDocumentContentType
    uploaded_at: datetime

    @field_validator("file_name")
    @classmethod
    def _validate_file_name(cls, value: str) -> str:
        return validate_base_file_name(value)

    @field_validator("uploaded_at")
    @classmethod
    def _normalize_uploaded_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, field_name="uploaded_at")


class DocumentConversionResult(BaseModel):
    """Immutable normalized result for a converted document."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    file_hash: StrictStr
    file_name: StrictStr
    content_type: SupportedDocumentContentType
    documents: tuple[Document, ...]

    @field_validator("file_hash")
    @classmethod
    def _validate_file_hash(cls, value: str) -> str:
        return validate_file_hash(value)

    @field_validator("file_name")
    @classmethod
    def _validate_file_name(cls, value: str) -> str:
        return validate_base_file_name(value)

    @field_validator("documents", mode="before")
    @classmethod
    def _validate_documents(cls, value: Any) -> tuple[Document, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            raise ValueError("documents must be a non-empty sequence of Haystack Document objects")
        documents = tuple(value)
        if not documents:
            raise ValueError("documents must not be empty")
        for document in documents:
            if not isinstance(document, Document):
                raise ValueError("documents must contain only Haystack Document objects")
        return documents

    @property
    def chunk_count(self) -> int:
        """Return the number of normalized chunk documents."""
        return len(self.documents)
