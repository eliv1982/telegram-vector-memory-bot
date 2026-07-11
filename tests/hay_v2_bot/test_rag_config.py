"""Unit tests for hay_v2_bot.config.DocumentRagSettings."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from hay_v2_bot.config import DocumentRagSettings
from pydantic import ValidationError


@pytest.fixture(autouse=True)
def _isolate_from_real_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    yield


def _set_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", "pinecone-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "document-index")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "chat-model")


def test_defaults_with_exact_v1_environment_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_env(monkeypatch)

    settings = DocumentRagSettings(_env_file=None)

    assert settings.PINECONE_INDEX_NAME == "document-index"
    assert settings.OPENAI_EMBEDDING_MODEL == "text-embedding-3-small"
    assert settings.OPENAI_CHAT_MODEL == "chat-model"
    assert settings.OPENAI_BASE_URL is None
    assert settings.embedding_dimensions == 1536
    assert settings.retrieval_top_k == 4
    assert settings.max_summary_chars == 12000
    assert settings.max_question_chars == 4000


def test_environment_overrides_are_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_env(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "embedding-model")
    monkeypatch.setenv("DOCUSCOPE_EMBEDDING_DIMENSIONS", "3072")
    monkeypatch.setenv("DOCUSCOPE_RETRIEVAL_TOP_K", "7")
    monkeypatch.setenv("DOCUSCOPE_MAX_SUMMARY_CHARS", "9000")
    monkeypatch.setenv("DOCUSCOPE_MAX_QUESTION_CHARS", "1234")

    settings = DocumentRagSettings(_env_file=None)

    assert settings.OPENAI_BASE_URL == "https://example.invalid/v1"
    assert settings.OPENAI_EMBEDDING_MODEL == "embedding-model"
    assert settings.embedding_dimensions == 3072
    assert settings.retrieval_top_k == 7
    assert settings.max_summary_chars == 9000
    assert settings.max_question_chars == 1234


def test_unrelated_environment_values_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_env(monkeypatch)
    monkeypatch.setenv("UNRELATED_SETTING", "value")
    monkeypatch.setenv("DOCUSCOPE_OPENAI_API_KEY", "wrong-alias")

    settings = DocumentRagSettings(_env_file=None)

    assert settings.OPENAI_API_KEY.get_secret_value() == "openai-key"
    assert settings.embedding_dimensions == 1536


def test_blank_base_url_becomes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_env(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "   ")

    settings = DocumentRagSettings(_env_file=None)

    assert settings.OPENAI_BASE_URL is None


def test_invalid_base_url_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_env(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "ftp://example.invalid")

    with pytest.raises(ValidationError):
        DocumentRagSettings(_env_file=None)


@pytest.mark.parametrize(
    "missing_key",
    [
        "PINECONE_API_KEY",
        "OPENAI_API_KEY",
    ],
)
def test_missing_required_secrets_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    missing_key: str,
) -> None:
    _set_required_env(monkeypatch)
    monkeypatch.delenv(missing_key)

    with pytest.raises(ValidationError):
        DocumentRagSettings(_env_file=None)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("embedding_dimensions", 0),
        ("embedding_dimensions", -1),
        ("retrieval_top_k", 0),
        ("retrieval_top_k", 21),
        ("max_summary_chars", 0),
        ("max_question_chars", 0),
    ],
)
def test_integer_constraints_are_enforced(field_name: str, value: int) -> None:
    with pytest.raises(ValidationError):
        DocumentRagSettings(
            _env_file=None,
            PINECONE_API_KEY="pinecone-key",
            PINECONE_INDEX_NAME="document-index",
            OPENAI_API_KEY="openai-key",
            OPENAI_CHAT_MODEL="chat-model",
            **{field_name: value},
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("embedding_dimensions", True),
        ("retrieval_top_k", False),
        ("max_summary_chars", True),
        ("max_question_chars", False),
    ],
)
def test_direct_bool_values_are_rejected(field_name: str, value: bool) -> None:
    with pytest.raises(ValidationError):
        DocumentRagSettings(
            _env_file=None,
            PINECONE_API_KEY="pinecone-key",
            PINECONE_INDEX_NAME="document-index",
            OPENAI_API_KEY="openai-key",
            OPENAI_CHAT_MODEL="chat-model",
            **{field_name: value},
        )


def test_real_repository_dotenv_is_not_read_when_env_file_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "PINECONE_API_KEY=wrong-pinecone-key",
                "PINECONE_INDEX_NAME=wrong-index",
                "OPENAI_API_KEY=wrong-openai-key",
                "OPENAI_CHAT_MODEL=wrong-chat-model",
                "DOCUSCOPE_RETRIEVAL_TOP_K=19",
            ]
        ),
        encoding="utf-8",
    )
    _set_required_env(monkeypatch)

    settings = DocumentRagSettings(_env_file=None)

    assert settings.PINECONE_API_KEY.get_secret_value() == "pinecone-key"
    assert settings.PINECONE_INDEX_NAME == "document-index"
    assert settings.OPENAI_API_KEY.get_secret_value() == "openai-key"
    assert settings.OPENAI_CHAT_MODEL == "chat-model"
    assert settings.retrieval_top_k == 4
