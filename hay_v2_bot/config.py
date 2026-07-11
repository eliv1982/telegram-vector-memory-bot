"""Typed document-processing configuration for hay_v2_bot."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_MAX_FILE_BYTES = 20 * 1024 * 1024
_DEFAULT_MAX_CHUNKS_PER_DOCUMENT = 2000
_DEFAULT_EMBEDDING_DIMENSIONS = 1536
_DEFAULT_RETRIEVAL_TOP_K = 4
_DEFAULT_MAX_SUMMARY_CHARS = 12000
_DEFAULT_MAX_QUESTION_CHARS = 4000


def _require_non_blank(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty or whitespace-only")
    return value


def _looks_like_base_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


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


class DocumentRagSettings(BaseSettings):
    """Immutable settings for the non-Telegram document RAG core."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    PINECONE_API_KEY: SecretStr
    PINECONE_INDEX_NAME: str

    OPENAI_API_KEY: SecretStr
    OPENAI_BASE_URL: str | None = None
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_CHAT_MODEL: str

    embedding_dimensions: int = Field(
        default=_DEFAULT_EMBEDDING_DIMENSIONS,
        gt=0,
        validation_alias="DOCUSCOPE_EMBEDDING_DIMENSIONS",
    )
    retrieval_top_k: int = Field(
        default=_DEFAULT_RETRIEVAL_TOP_K,
        gt=0,
        le=20,
        validation_alias="DOCUSCOPE_RETRIEVAL_TOP_K",
    )
    max_summary_chars: int = Field(
        default=_DEFAULT_MAX_SUMMARY_CHARS,
        gt=0,
        validation_alias="DOCUSCOPE_MAX_SUMMARY_CHARS",
    )
    max_question_chars: int = Field(
        default=_DEFAULT_MAX_QUESTION_CHARS,
        gt=0,
        validation_alias="DOCUSCOPE_MAX_QUESTION_CHARS",
    )

    @field_validator("PINECONE_INDEX_NAME", "OPENAI_EMBEDDING_MODEL", "OPENAI_CHAT_MODEL")
    @classmethod
    def _validate_non_blank(cls, value: str, info: ValidationInfo) -> str:
        return _require_non_blank(value, info.field_name or "value")

    @field_validator("PINECONE_API_KEY", "OPENAI_API_KEY")
    @classmethod
    def _validate_secret_non_blank(cls, value: SecretStr, info: ValidationInfo) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError(f"{info.field_name or 'value'} must not be empty or whitespace-only")
        return value

    @field_validator("OPENAI_BASE_URL", mode="before")
    @classmethod
    def _normalize_optional_base_url(cls, value: Any) -> Any:
        if value is None or not isinstance(value, str):
            return value
        if not value.strip():
            return None
        if not _looks_like_base_url(value):
            raise ValueError("OPENAI_BASE_URL must be a valid http(s) URL")
        return value

    @field_validator(
        "embedding_dimensions",
        "retrieval_top_k",
        "max_summary_chars",
        "max_question_chars",
        mode="before",
    )
    @classmethod
    def _reject_bool_rag_integers(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be a positive integer")
        return value
