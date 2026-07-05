"""Typed application configuration.

Settings are read from environment variables and an optional ``.env`` file.
No external clients (Pinecone, OpenAI, Telegram) are created here or at
import time -- this module only defines and validates configuration.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_NAMESPACE_PREFIX_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _require_non_blank(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty or whitespace-only")
    return value


def _looks_like_base_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


class Settings(BaseSettings):
    """Application configuration loaded from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PINECONE_API_KEY: SecretStr
    PINECONE_INDEX_NAME: str

    OPENAI_API_KEY: SecretStr
    OPENAI_BASE_URL: str | None = None
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_CHAT_MODEL: str

    TELEGRAM_BOT_TOKEN: SecretStr

    MEMORY_SIMILARITY_THRESHOLD: float = Field(default=0.50, ge=0.0, le=1.0)
    MEMORY_TOP_K: int = Field(default=5, ge=1, le=20)
    MEMORY_NAMESPACE_PREFIX: str = "telegram-user"

    LOG_LEVEL: str = "INFO"

    @field_validator("PINECONE_INDEX_NAME", "OPENAI_EMBEDDING_MODEL", "OPENAI_CHAT_MODEL")
    @classmethod
    def _validate_non_blank(cls, value: str, info: ValidationInfo) -> str:
        return _require_non_blank(value, info.field_name or "value")

    @field_validator("PINECONE_API_KEY", "OPENAI_API_KEY", "TELEGRAM_BOT_TOKEN")
    @classmethod
    def _validate_secret_non_blank(cls, value: SecretStr, info: ValidationInfo) -> SecretStr:
        """Reject whitespace-only secrets without ever exposing the value.

        Uses ``get_secret_value()`` only to check blankness; the raw value is
        never interpolated into the raised error message.
        """
        if not value.get_secret_value().strip():
            raise ValueError(f"{info.field_name or 'value'} must not be empty or whitespace-only")
        return value

    @field_validator("OPENAI_BASE_URL", mode="before")
    @classmethod
    def _normalize_optional_base_url(cls, value: Any) -> Any:
        """Treat missing, empty, and whitespace-only values as unset (None).

        This field alone is optional and blank-tolerant, matching
        ``.env.example``'s intentionally empty ``OPENAI_BASE_URL=`` line for
        the documented direct-OpenAI configuration. A non-blank value must
        still be a well-formed http(s) URL.
        """
        if value is None or not isinstance(value, str):
            return value
        if not value.strip():
            return None
        if not _looks_like_base_url(value):
            raise ValueError("OPENAI_BASE_URL must be a valid http(s) URL")
        return value

    @field_validator("MEMORY_NAMESPACE_PREFIX")
    @classmethod
    def _validate_namespace_prefix(cls, value: str) -> str:
        value = _require_non_blank(value, "MEMORY_NAMESPACE_PREFIX")
        if not _NAMESPACE_PREFIX_PATTERN.fullmatch(value):
            raise ValueError(
                "MEMORY_NAMESPACE_PREFIX must contain only letters, digits, "
                "hyphens, and underscores"
            )
        return value

    @field_validator("LOG_LEVEL")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        return _require_non_blank(value, "LOG_LEVEL").upper()


@lru_cache
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance, loading it on first access."""
    return Settings()
