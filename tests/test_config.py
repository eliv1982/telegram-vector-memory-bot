"""Unit tests for telegram_vector_memory_bot.config.

All tests build Settings with ``_env_file=None`` and rely solely on
monkeypatched environment variables, so no real user .env file is ever read.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from telegram_vector_memory_bot.config import Settings, get_settings

REQUIRED_ENV: dict[str, str] = {
    "PINECONE_API_KEY": "test-pinecone-key",
    "PINECONE_INDEX_NAME": "test-index",
    "OPENAI_API_KEY": "test-openai-key",
    "OPENAI_CHAT_MODEL": "gpt-4o-mini",
    "TELEGRAM_BOT_TOKEN": "test-telegram-token",
}


@pytest.fixture(autouse=True)
def _clear_settings_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    # get_settings() uses Settings()'s default env_file=".env", relative to the
    # current working directory. Chdir into an empty tmp_path so a real,
    # developer-local .env (e.g. one set up for the live Stage 4A scripts)
    # never leaks into these tests, even for the two tests below that call
    # get_settings() directly instead of Settings(_env_file=None).
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _set_required_env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)


def test_settings_created_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_env(monkeypatch)

    settings = Settings(_env_file=None)

    assert settings.PINECONE_INDEX_NAME == "test-index"
    assert settings.OPENAI_CHAT_MODEL == "gpt-4o-mini"
    assert settings.OPENAI_EMBEDDING_MODEL == "text-embedding-3-small"
    assert settings.MEMORY_SIMILARITY_THRESHOLD == 0.90
    assert settings.MEMORY_TOP_K == 5
    assert settings.MEMORY_NAMESPACE_PREFIX == "telegram-user"
    assert settings.LOG_LEVEL == "INFO"


def test_missing_required_variable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in REQUIRED_ENV.items():
        if key == "OPENAI_CHAT_MODEL":
            continue
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_similarity_threshold_below_zero_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_env(monkeypatch, MEMORY_SIMILARITY_THRESHOLD="-0.01")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_similarity_threshold_above_one_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_env(monkeypatch, MEMORY_SIMILARITY_THRESHOLD="1.01")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("top_k", ["0", "21", "-1"])
def test_invalid_top_k_rejected(monkeypatch: pytest.MonkeyPatch, top_k: str) -> None:
    _set_required_env(monkeypatch, MEMORY_TOP_K=top_k)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("index_name", ["", "   "])
def test_empty_index_name_rejected(monkeypatch: pytest.MonkeyPatch, index_name: str) -> None:
    _set_required_env(monkeypatch, PINECONE_INDEX_NAME=index_name)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_openai_base_url_absent_becomes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_env(monkeypatch)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.OPENAI_BASE_URL is None


def test_openai_base_url_empty_string_becomes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # Matches .env.example's intentionally empty `OPENAI_BASE_URL=` line.
    _set_required_env(monkeypatch, OPENAI_BASE_URL="")

    settings = Settings(_env_file=None)

    assert settings.OPENAI_BASE_URL is None


def test_openai_base_url_whitespace_only_becomes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_env(monkeypatch, OPENAI_BASE_URL="   ")

    settings = Settings(_env_file=None)

    assert settings.OPENAI_BASE_URL is None


def test_valid_openai_base_url_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_env(monkeypatch, OPENAI_BASE_URL="https://api.openai.com/v1")

    settings = Settings(_env_file=None)

    assert settings.OPENAI_BASE_URL == "https://api.openai.com/v1"


@pytest.mark.parametrize(
    "base_url",
    ["not-a-url", "ftp://api.openai.com", "http://", "https://", "just some text"],
)
def test_malformed_openai_base_url_rejected(
    monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    _set_required_env(monkeypatch, OPENAI_BASE_URL=base_url)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("prefix", ["bad prefix", "bad/prefix", "bad.prefix", ""])
def test_invalid_namespace_prefix_rejected(monkeypatch: pytest.MonkeyPatch, prefix: str) -> None:
    _set_required_env(monkeypatch, MEMORY_NAMESPACE_PREFIX=prefix)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_log_level_is_normalized_to_uppercase(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_env(monkeypatch, LOG_LEVEL="debug")

    settings = Settings(_env_file=None)

    assert settings.LOG_LEVEL == "DEBUG"


def test_secret_str_fields_do_not_leak_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_env(monkeypatch)

    settings = Settings(_env_file=None)
    rendered = repr(settings)

    assert "test-pinecone-key" not in rendered
    assert "test-openai-key" not in rendered
    assert "test-telegram-token" not in rendered
    assert settings.PINECONE_API_KEY.get_secret_value() == "test-pinecone-key"


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_env(monkeypatch)

    first = get_settings()
    second = get_settings()

    assert first is second


def test_get_settings_cache_can_be_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_env(monkeypatch)
    first = get_settings()

    get_settings.cache_clear()
    monkeypatch.setenv("PINECONE_INDEX_NAME", "different-index")
    second = get_settings()

    assert first is not second
    assert second.PINECONE_INDEX_NAME == "different-index"
