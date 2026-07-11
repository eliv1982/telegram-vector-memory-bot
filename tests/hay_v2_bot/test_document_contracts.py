"""Unit tests for hay_v2_bot document metadata contracts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from hay_v2_bot.models import (
    DOCX_CONTENT_TYPE,
    PDF_CONTENT_TYPE,
    SUPPORTED_DOCUMENT_CONTENT_TYPES,
    DocumentChunkMetadata,
)
from pydantic import ValidationError

FILE_HASH = "a" * 64


def test_supported_document_types_are_exactly_pdf_and_docx() -> None:
    assert SUPPORTED_DOCUMENT_CONTENT_TYPES == {PDF_CONTENT_TYPE, DOCX_CONTENT_TYPE}


def test_valid_document_chunk_metadata_with_pdf() -> None:
    metadata = DocumentChunkMetadata(
        user_id=123,
        file_name="report.pdf",
        file_hash=FILE_HASH,
        chunk_index=0,
        content_type=PDF_CONTENT_TYPE,
        uploaded_at=datetime(2026, 7, 11, 12, 0, tzinfo=UTC),
        page_number=2,
        headings=["Intro", "Findings"],
    )

    assert metadata.record_type == "document_chunk"
    assert metadata.user_id == 123
    assert metadata.file_name == "report.pdf"
    assert metadata.file_hash == FILE_HASH
    assert metadata.chunk_index == 0
    assert metadata.content_type == PDF_CONTENT_TYPE
    assert metadata.page_number == 2
    assert metadata.headings == ("Intro", "Findings")
    assert metadata.uploaded_at == datetime(2026, 7, 11, 12, 0, tzinfo=UTC)


def test_valid_document_chunk_metadata_with_docx() -> None:
    metadata = DocumentChunkMetadata(
        user_id=123,
        file_name="report.docx",
        file_hash=FILE_HASH,
        chunk_index=1,
        content_type=DOCX_CONTENT_TYPE,
        uploaded_at=datetime(2026, 7, 11, 12, 0, tzinfo=UTC),
    )

    assert metadata.content_type == DOCX_CONTENT_TYPE


def test_unsupported_content_type_rejected() -> None:
    with pytest.raises(ValidationError):
        DocumentChunkMetadata(
            user_id=123,
            file_name="report.txt",
            file_hash=FILE_HASH,
            chunk_index=0,
            content_type="text/plain",
            uploaded_at=datetime(2026, 7, 11, 12, 0, tzinfo=UTC),
        )


def test_windows_path_rejected() -> None:
    with pytest.raises(ValidationError):
        DocumentChunkMetadata(
            user_id=123,
            file_name="folder\\report.pdf",
            file_hash=FILE_HASH,
            chunk_index=0,
            content_type=PDF_CONTENT_TYPE,
            uploaded_at=datetime(2026, 7, 11, 12, 0, tzinfo=UTC),
        )


def test_posix_path_rejected() -> None:
    with pytest.raises(ValidationError):
        DocumentChunkMetadata(
            user_id=123,
            file_name="folder/report.pdf",
            file_hash=FILE_HASH,
            chunk_index=0,
            content_type=PDF_CONTENT_TYPE,
            uploaded_at=datetime(2026, 7, 11, 12, 0, tzinfo=UTC),
        )


@pytest.mark.parametrize("file_name", ["", "   "])
def test_blank_file_name_rejected(file_name: str) -> None:
    with pytest.raises(ValidationError):
        DocumentChunkMetadata(
            user_id=123,
            file_name=file_name,
            file_hash=FILE_HASH,
            chunk_index=0,
            content_type=PDF_CONTENT_TYPE,
            uploaded_at=datetime(2026, 7, 11, 12, 0, tzinfo=UTC),
        )


@pytest.mark.parametrize("user_id", [True, 0, -1])
def test_invalid_user_id_rejected(user_id: int) -> None:
    with pytest.raises(ValidationError):
        DocumentChunkMetadata(
            user_id=user_id,
            file_name="report.pdf",
            file_hash=FILE_HASH,
            chunk_index=0,
            content_type=PDF_CONTENT_TYPE,
            uploaded_at=datetime(2026, 7, 11, 12, 0, tzinfo=UTC),
        )


@pytest.mark.parametrize("chunk_index", [True, -1])
def test_invalid_chunk_index_rejected(chunk_index: int) -> None:
    with pytest.raises(ValidationError):
        DocumentChunkMetadata(
            user_id=123,
            file_name="report.pdf",
            file_hash=FILE_HASH,
            chunk_index=chunk_index,
            content_type=PDF_CONTENT_TYPE,
            uploaded_at=datetime(2026, 7, 11, 12, 0, tzinfo=UTC),
        )


@pytest.mark.parametrize("page_number", [True, 0, -1])
def test_invalid_page_number_rejected(page_number: int) -> None:
    with pytest.raises(ValidationError):
        DocumentChunkMetadata(
            user_id=123,
            file_name="report.pdf",
            file_hash=FILE_HASH,
            chunk_index=0,
            content_type=PDF_CONTENT_TYPE,
            uploaded_at=datetime(2026, 7, 11, 12, 0, tzinfo=UTC),
            page_number=page_number,
        )


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        DocumentChunkMetadata(
            user_id=123,
            file_name="report.pdf",
            file_hash=FILE_HASH,
            chunk_index=0,
            content_type=PDF_CONTENT_TYPE,
            uploaded_at=datetime(2026, 7, 11, 12, 0),
        )


def test_uploaded_at_normalized_to_utc() -> None:
    source_time = datetime(2026, 7, 11, 15, 0, tzinfo=timezone(timedelta(hours=3)))
    metadata = DocumentChunkMetadata(
        user_id=123,
        file_name="report.pdf",
        file_hash=FILE_HASH,
        chunk_index=0,
        content_type=PDF_CONTENT_TYPE,
        uploaded_at=source_time,
    )

    assert metadata.uploaded_at == datetime(2026, 7, 11, 12, 0, tzinfo=UTC)


def test_model_is_immutable() -> None:
    metadata = DocumentChunkMetadata(
        user_id=123,
        file_name="report.pdf",
        file_hash=FILE_HASH,
        chunk_index=0,
        content_type=PDF_CONTENT_TYPE,
        uploaded_at=datetime(2026, 7, 11, 12, 0, tzinfo=UTC),
    )

    with pytest.raises(ValidationError):
        metadata.file_name = "changed.pdf"  # type: ignore[misc]


def test_headings_are_stored_immutably() -> None:
    metadata = DocumentChunkMetadata(
        user_id=123,
        file_name="report.pdf",
        file_hash=FILE_HASH,
        chunk_index=0,
        content_type=PDF_CONTENT_TYPE,
        uploaded_at=datetime(2026, 7, 11, 12, 0, tzinfo=UTC),
        headings=["Intro", "Summary"],
    )

    assert isinstance(metadata.headings, tuple)
    with pytest.raises(TypeError):
        metadata.headings[0] = "Changed"  # type: ignore[index]


def test_blank_heading_rejected() -> None:
    with pytest.raises(ValidationError):
        DocumentChunkMetadata(
            user_id=123,
            file_name="report.pdf",
            file_hash=FILE_HASH,
            chunk_index=0,
            content_type=PDF_CONTENT_TYPE,
            uploaded_at=datetime(2026, 7, 11, 12, 0, tzinfo=UTC),
            headings=["Intro", "   "],
        )


def test_pinecone_metadata_is_json_safe_and_serialized() -> None:
    metadata = DocumentChunkMetadata(
        user_id=123,
        file_name="report.pdf",
        file_hash=FILE_HASH,
        chunk_index=7,
        content_type=PDF_CONTENT_TYPE,
        uploaded_at=datetime(2026, 7, 11, 12, 0, tzinfo=UTC),
        page_number=3,
        headings=["Intro", "Summary"],
    )

    assert metadata.to_pinecone_metadata() == {
        "record_type": "document_chunk",
        "user_id": 123,
        "file_name": "report.pdf",
        "file_hash": FILE_HASH,
        "chunk_index": 7,
        "content_type": PDF_CONTENT_TYPE,
        "uploaded_at": "2026-07-11T12:00:00+00:00",
        "page_number": 3,
        "headings": ["Intro", "Summary"],
    }


def test_absent_optional_metadata_is_omitted_from_pinecone_payload() -> None:
    metadata = DocumentChunkMetadata(
        user_id=123,
        file_name="report.pdf",
        file_hash=FILE_HASH,
        chunk_index=0,
        content_type=PDF_CONTENT_TYPE,
        uploaded_at=datetime(2026, 7, 11, 12, 0, tzinfo=UTC),
    )

    pinecone_metadata = metadata.to_pinecone_metadata()
    assert "page_number" not in pinecone_metadata
    assert "headings" not in pinecone_metadata
