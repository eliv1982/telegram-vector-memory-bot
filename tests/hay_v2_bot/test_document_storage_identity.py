"""Unit tests for hay_v2_bot storage namespaces and identifiers."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest
from hay_v2_bot.storage import (
    build_document_chunk_id,
    document_namespace_for_user,
    validate_file_hash,
)


def _load_v1_memory_policy_class() -> type:
    repo_root = Path(__file__).resolve().parents[2]
    namespace = runpy.run_path(
        str(repo_root / "src" / "telegram_vector_memory_bot" / "memory_policy.py")
    )
    return namespace["MemoryPolicy"]


MemoryPolicy = _load_v1_memory_policy_class()

FILE_HASH = "0123456789abcdef" * 4


def test_valid_namespace() -> None:
    assert document_namespace_for_user(123) == "telegram-documents-user-123"


@pytest.mark.parametrize("user_id", [True, False, 0, -1, 1.5, "123", None])
def test_invalid_namespace_user_ids_rejected(user_id: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        document_namespace_for_user(user_id)  # type: ignore[arg-type]


def test_document_namespace_is_distinct_from_real_v1_memory_namespace() -> None:
    memory_namespace = MemoryPolicy(0.50).namespace_for_user(prefix="telegram-user", user_id=123)
    document_namespace = document_namespace_for_user(123)

    assert memory_namespace == "telegram-user-123"
    assert document_namespace == "telegram-documents-user-123"
    assert document_namespace != memory_namespace


def test_valid_file_hash_passes_through() -> None:
    assert validate_file_hash(FILE_HASH) == FILE_HASH


@pytest.mark.parametrize(
    "file_hash",
    [
        True,
        123,
        "",
        " ",
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "sha256:" + ("a" * 64),
        ("a" * 63) + "g",
        ("a" * 32) + "\n" + ("a" * 31),
    ],
)
def test_invalid_file_hashes_rejected(file_hash: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_file_hash(file_hash)


def test_deterministic_chunk_id() -> None:
    expected = f"doc-{FILE_HASH}-chunk-000007"
    assert build_document_chunk_id(FILE_HASH, 7) == expected
    assert build_document_chunk_id(FILE_HASH, 7) == expected


def test_different_chunk_indexes_produce_different_ids() -> None:
    assert build_document_chunk_id(FILE_HASH, 0) != build_document_chunk_id(FILE_HASH, 1)


@pytest.mark.parametrize("chunk_index", [True, False, -1, 1.5, "1", None])
def test_invalid_chunk_index_rejected(chunk_index: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_document_chunk_id(FILE_HASH, chunk_index)  # type: ignore[arg-type]


def test_invalid_hash_rejected_by_chunk_id_builder() -> None:
    with pytest.raises(ValueError):
        build_document_chunk_id("A" * 64, 0)
