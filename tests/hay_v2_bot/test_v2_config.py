"""Unit tests for hay_v2_bot.config document-processing settings."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from hay_v2_bot.config import DocumentProcessingSettings
from pydantic import ValidationError


@pytest.fixture(autouse=True)
def _isolate_from_real_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    yield


def test_defaults() -> None:
    settings = DocumentProcessingSettings(_env_file=None)

    assert settings.max_file_bytes == 20 * 1024 * 1024
    assert settings.max_chunks_per_document == 2000


def test_max_file_bytes_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCUSCOPE_MAX_FILE_BYTES", "1048576")

    settings = DocumentProcessingSettings(_env_file=None)

    assert settings.max_file_bytes == 1048576


def test_max_chunks_per_document_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCUSCOPE_MAX_CHUNKS_PER_DOCUMENT", "321")

    settings = DocumentProcessingSettings(_env_file=None)

    assert settings.max_chunks_per_document == 321


def test_unrelated_environment_values_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNRELATED_SETTING", "value")

    settings = DocumentProcessingSettings(_env_file=None)

    assert settings.max_file_bytes == 20 * 1024 * 1024
    assert settings.max_chunks_per_document == 2000


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_file_bytes", 0),
        ("max_file_bytes", -1),
        ("max_chunks_per_document", 0),
        ("max_chunks_per_document", -1),
    ],
)
def test_zero_and_negative_values_rejected(field_name: str, value: int) -> None:
    with pytest.raises(ValidationError):
        DocumentProcessingSettings(_env_file=None, **{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_file_bytes", True),
        ("max_chunks_per_document", False),
    ],
)
def test_direct_bool_values_rejected(field_name: str, value: bool) -> None:
    with pytest.raises(ValidationError):
        DocumentProcessingSettings(_env_file=None, **{field_name: value})


@pytest.mark.parametrize(
    ("environment_key", "value"),
    [
        ("DOCUSCOPE_MAX_FILE_BYTES", "not-an-int"),
        ("DOCUSCOPE_MAX_CHUNKS_PER_DOCUMENT", "1.5"),
    ],
)
def test_invalid_strings_rejected(
    monkeypatch: pytest.MonkeyPatch,
    environment_key: str,
    value: str,
) -> None:
    monkeypatch.setenv(environment_key, value)

    with pytest.raises(ValidationError):
        DocumentProcessingSettings(_env_file=None)
