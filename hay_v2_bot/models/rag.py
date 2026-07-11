"""Immutable result models for the Stage 5 document RAG core."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, field_validator, model_validator

from hay_v2_bot.storage import validate_file_hash

from .documents import (
    StrictNonNegativeInt,
    StrictPositiveInt,
    SupportedDocumentContentType,
    validate_base_file_name,
)

INSUFFICIENT_DOCUMENT_ANSWER = "В загруженных документах недостаточно информации для ответа."


def _require_non_blank(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty or whitespace-only")
    return value


def _sequence_to_tuple(value: Any, *, field_name: str) -> Any:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence")
    return tuple(value)


class DocumentSource(BaseModel):
    """Immutable public source information for one retrieved document chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: StrictStr
    file_name: StrictStr
    chunk_index: StrictNonNegativeInt
    page_number: StrictPositiveInt | None = None
    score: float | None = None

    @field_validator("document_id")
    @classmethod
    def _validate_document_id(cls, value: str) -> str:
        return _require_non_blank(value, "document_id")

    @field_validator("file_name")
    @classmethod
    def _validate_file_name(cls, value: str) -> str:
        return validate_base_file_name(value)

    @field_validator("score")
    @classmethod
    def _validate_score(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise TypeError("score must be numeric when present")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("score must be finite when present")
        return normalized


class DocumentIngestionOutcome(BaseModel):
    """Immutable outcome for a successful document ingestion and summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file_hash: StrictStr
    file_name: StrictStr
    content_type: SupportedDocumentContentType
    chunk_count: StrictPositiveInt
    documents_written: StrictNonNegativeInt
    document_ids: tuple[StrictStr, ...]
    summary: StrictStr

    @field_validator("file_hash")
    @classmethod
    def _validate_file_hash(cls, value: str) -> str:
        return validate_file_hash(value)

    @field_validator("file_name")
    @classmethod
    def _validate_file_name(cls, value: str) -> str:
        return validate_base_file_name(value)

    @field_validator("document_ids", mode="before")
    @classmethod
    def _normalize_document_ids(cls, value: Any) -> Any:
        return _sequence_to_tuple(value, field_name="document_ids")

    @field_validator("document_ids")
    @classmethod
    def _validate_document_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("document_ids must not be empty")
        for document_id in value:
            _require_non_blank(document_id, "document_ids")
        return value

    @field_validator("summary")
    @classmethod
    def _validate_summary(cls, value: str) -> str:
        return _require_non_blank(value, "summary")

    @model_validator(mode="after")
    def _validate_counts(self) -> DocumentIngestionOutcome:
        if self.chunk_count != len(self.document_ids):
            raise ValueError("chunk_count must equal the number of document_ids")
        return self


class DocumentAnswer(BaseModel):
    """Immutable public answer payload for one grounded document question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: StrictStr
    sources: tuple[DocumentSource, ...] = ()
    used_document_count: StrictNonNegativeInt
    fallback_used: StrictBool

    @field_validator("answer")
    @classmethod
    def _validate_answer(cls, value: str) -> str:
        return _require_non_blank(value, "answer")

    @field_validator("sources", mode="before")
    @classmethod
    def _normalize_sources(cls, value: Any) -> Any:
        return _sequence_to_tuple(value, field_name="sources")

    @field_validator("sources")
    @classmethod
    def _validate_sources(cls, value: tuple[DocumentSource, ...]) -> tuple[DocumentSource, ...]:
        return value

    @model_validator(mode="after")
    def _validate_source_count(self) -> DocumentAnswer:
        if self.used_document_count != len(self.sources):
            raise ValueError("used_document_count must equal len(sources)")
        return self
