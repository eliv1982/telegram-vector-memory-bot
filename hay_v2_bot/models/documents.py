"""Strict document metadata contracts for Pinecone-backed document chunks."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from hay_v2_bot.storage.identifiers import validate_file_hash

PDF_CONTENT_TYPE: Final = "application/pdf"
DOCX_CONTENT_TYPE: Final = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
SUPPORTED_DOCUMENT_CONTENT_TYPES: Final = frozenset({PDF_CONTENT_TYPE, DOCX_CONTENT_TYPE})
_DOCUMENT_CHUNK_RECORD_TYPE: Final = "document_chunk"
_PATH_ONLY_NAMES: Final = frozenset({".", ".."})

StrictPositiveInt = Annotated[int, Field(strict=True, gt=0)]
StrictNonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
SupportedDocumentContentType = Literal[
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
]


def validate_base_file_name(value: str) -> str:
    """Validate that a value is a non-blank base filename."""
    if not value.strip():
        raise ValueError("file_name must not be empty or whitespace-only")
    if value in _PATH_ONLY_NAMES:
        raise ValueError("file_name must be a base filename, not a path")
    if PurePosixPath(value).name != value or PureWindowsPath(value).name != value:
        raise ValueError("file_name must be a base filename, not a path")
    return value


def normalize_utc_datetime(value: datetime, *, field_name: str) -> datetime:
    """Require a timezone-aware datetime and normalize it to UTC."""
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


class DocumentChunkMetadata(BaseModel):
    """Immutable metadata contract for a stored document chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["document_chunk"] = _DOCUMENT_CHUNK_RECORD_TYPE
    user_id: StrictPositiveInt
    file_name: StrictStr
    file_hash: StrictStr
    chunk_index: StrictNonNegativeInt
    content_type: SupportedDocumentContentType
    uploaded_at: datetime
    page_number: StrictPositiveInt | None = None
    headings: tuple[StrictStr, ...] | None = None

    @field_validator("file_name")
    @classmethod
    def _validate_file_name(cls, value: str) -> str:
        return validate_base_file_name(value)

    @field_validator("file_hash")
    @classmethod
    def _validate_file_hash(cls, value: str) -> str:
        return validate_file_hash(value)

    @field_validator("uploaded_at")
    @classmethod
    def _normalize_uploaded_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, field_name="uploaded_at")

    @field_validator("headings", mode="before")
    @classmethod
    def _normalize_headings(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            raise TypeError("headings must be a sequence of strings")
        return tuple(value)

    @field_validator("headings")
    @classmethod
    def _validate_headings(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is None:
            return None
        for heading in value:
            if not heading.strip():
                raise ValueError("headings must not contain blank strings")
        return value

    def to_pinecone_metadata(self) -> dict[str, str | int | list[str]]:
        """Return JSON-compatible metadata safe to send to Pinecone."""
        metadata: dict[str, str | int | list[str]] = {
            "record_type": self.record_type,
            "user_id": self.user_id,
            "file_name": self.file_name,
            "file_hash": self.file_hash,
            "chunk_index": self.chunk_index,
            "content_type": self.content_type,
            "uploaded_at": self.uploaded_at.astimezone(UTC).isoformat(),
        }
        if self.page_number is not None:
            metadata["page_number"] = self.page_number
        if self.headings is not None:
            metadata["headings"] = list(self.headings)
        return metadata
