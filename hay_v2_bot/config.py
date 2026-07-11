"""Typed document-processing configuration for hay_v2_bot."""

from __future__ import annotations

from typing import Any

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_MAX_FILE_BYTES = 20 * 1024 * 1024
_DEFAULT_MAX_CHUNKS_PER_DOCUMENT = 2000


class DocumentProcessingSettings(BaseSettings):
    """Immutable settings for offline-safe document processing."""

    model_config = SettingsConfigDict(
        env_prefix="DOCUSCOPE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    max_file_bytes: int = Field(default=_DEFAULT_MAX_FILE_BYTES, gt=0)
    max_chunks_per_document: int = Field(default=_DEFAULT_MAX_CHUNKS_PER_DOCUMENT, gt=0)

    @field_validator("max_file_bytes", "max_chunks_per_document", mode="before")
    @classmethod
    def _reject_bool_values(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be a positive integer")
        return value
