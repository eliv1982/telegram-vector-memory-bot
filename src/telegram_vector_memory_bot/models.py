"""Domain models for memory write operations and stored memory records."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class MemoryAction(StrEnum):
    """Outcome of an attempt to write a memory."""

    INSERTED = "inserted"
    SKIPPED = "skipped"
    FAILED = "failed"


class MemoryReason(StrEnum):
    """Reason behind a ``MemoryAction``."""

    NEW_MEMORY = "new_memory"
    EXACT_DUPLICATE = "exact_duplicate"
    SEMANTIC_DUPLICATE = "semantic_duplicate"
    VALIDATION_ERROR = "validation_error"
    STORAGE_ERROR = "storage_error"


_DUPLICATE_REASONS = {MemoryReason.EXACT_DUPLICATE, MemoryReason.SEMANTIC_DUPLICATE}


class MemoryWriteResult(BaseModel):
    """Result of attempting to write a memory to the vector store."""

    action: MemoryAction
    reason: MemoryReason
    memory_id: str | None = None
    existing_id: str | None = None
    similarity_score: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_consistency(self) -> MemoryWriteResult:
        if self.action is MemoryAction.INSERTED and self.memory_id is None:
            raise ValueError("action 'inserted' requires memory_id to be set")

        if self.reason in _DUPLICATE_REASONS:
            if self.action is not MemoryAction.SKIPPED:
                raise ValueError(
                    f"reason '{self.reason.value}' requires action 'skipped'"
                )
            if self.existing_id is None:
                raise ValueError(
                    f"reason '{self.reason.value}' requires existing_id to be set"
                )

        return self


class MemoryRecord(BaseModel):
    """A single stored memory attributed to a Telegram user."""

    memory_id: str
    user_id: int = Field(gt=0)
    text: str
    content_hash: str
    created_at: datetime
    source: str = "telegram"
    username: str | None = None
    first_name: str | None = None
    last_name: str | None = None

    @model_validator(mode="after")
    def _validate_record(self) -> MemoryRecord:
        if not self.text.strip():
            raise ValueError("text must not be empty or whitespace-only")
        if not self.content_hash.strip():
            raise ValueError("content_hash must not be empty or whitespace-only")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return self
