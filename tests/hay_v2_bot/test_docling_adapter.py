"""Offline unit tests for hay_v2_bot.adapters.docling_adapter."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from hay_v2_bot.adapters import (
    DoclingDocumentAdapter,
    DocumentConversionError,
    DocumentTooLargeError,
    EmptyDocumentError,
    InvalidDocumentInputError,
    TooManyDocumentChunksError,
    UnsupportedDocumentTypeError,
)
from hay_v2_bot.config import DocumentProcessingSettings
from hay_v2_bot.models import (
    DOCX_CONTENT_TYPE,
    PDF_CONTENT_TYPE,
    DocumentConversionRequest,
    DocumentConversionResult,
)
from hay_v2_bot.storage import build_document_chunk_id
from haystack import Document
from haystack_integrations.components.converters.docling import ExportType
from pydantic import ValidationError

UPLOAD_TIME = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)


class FakeConverter:
    def __init__(
        self,
        result: dict[str, Any] | None = None,
        *,
        exception: BaseException | None = None,
    ) -> None:
        self._result = result if result is not None else {"documents": [Document(content="chunk")]}
        self._exception = exception
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        *,
        paths: list[str | Path] | None = None,
        sources: Any = None,
        meta: Any = None,
    ) -> dict[str, Any]:
        self.calls.append({"paths": paths, "sources": sources, "meta": meta})
        if self._exception is not None:
            raise self._exception
        return self._result


def _settings(**overrides: int) -> DocumentProcessingSettings:
    return DocumentProcessingSettings(_env_file=None, **overrides)


def _write_file(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path


def _request(
    local_path: Path,
    *,
    user_id: int = 123,
    file_name: str | None = None,
    content_type: str = PDF_CONTENT_TYPE,
    uploaded_at: datetime = UPLOAD_TIME,
) -> DocumentConversionRequest:
    if file_name is None:
        file_name = local_path.name
    return DocumentConversionRequest(
        local_path=local_path,
        user_id=user_id,
        file_name=file_name,
        content_type=content_type,
        uploaded_at=uploaded_at,
    )


def test_default_adapter_constructs_docling_converter_for_doc_chunks_without_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {"export_type": None, "run_called": False, "warm_up_called": False}

    class FakeDoclingConverter:
        def __init__(self, *, export_type: ExportType) -> None:
            calls["export_type"] = export_type

        def run(self, **_: Any) -> dict[str, Any]:
            calls["run_called"] = True
            return {"documents": [Document(content="chunk")]}

        def warm_up(self) -> None:
            calls["warm_up_called"] = True

    monkeypatch.setattr(
        "hay_v2_bot.adapters.docling_adapter.DoclingConverter",
        FakeDoclingConverter,
    )

    adapter = DoclingDocumentAdapter(settings=_settings())

    assert isinstance(adapter, DoclingDocumentAdapter)
    assert calls["export_type"] is ExportType.DOC_CHUNKS
    assert calls["run_called"] is False
    assert calls["warm_up_called"] is False


def test_injected_fake_converter_is_accepted() -> None:
    converter = FakeConverter()
    adapter = DoclingDocumentAdapter(settings=_settings(), converter=converter)

    assert adapter._converter is converter


@pytest.mark.parametrize(
    ("file_name", "content_type", "suffix", "content"),
    [
        ("report.pdf", PDF_CONTENT_TYPE, ".pdf", b"%PDF-1.7\ncontent"),
        (
            "report.docx",
            DOCX_CONTENT_TYPE,
            ".docx",
            b"PK\x03\x04docx-bytes",
        ),
    ],
)
def test_valid_pdf_and_docx_requests_convert(
    tmp_path: Path,
    file_name: str,
    content_type: str,
    suffix: str,
    content: bytes,
) -> None:
    local_path = _write_file(tmp_path / f"upload{suffix}", content)
    converter = FakeConverter(result={"documents": [Document(content=" normalized ")]})
    adapter = DoclingDocumentAdapter(settings=_settings(), converter=converter)

    result = adapter.convert(_request(local_path, file_name=file_name, content_type=content_type))

    assert result.file_name == file_name
    assert result.content_type == content_type
    assert result.chunk_count == 1


def test_missing_path_is_rejected_before_converter_runs(tmp_path: Path) -> None:
    converter = FakeConverter()
    adapter = DoclingDocumentAdapter(settings=_settings(), converter=converter)

    with pytest.raises(InvalidDocumentInputError):
        adapter.convert(_request(tmp_path / "missing.pdf"))

    assert converter.calls == []


def test_directory_instead_of_file_is_rejected(tmp_path: Path) -> None:
    local_path = tmp_path / "folder.pdf"
    local_path.mkdir()
    converter = FakeConverter()
    adapter = DoclingDocumentAdapter(settings=_settings(), converter=converter)

    with pytest.raises(InvalidDocumentInputError):
        adapter.convert(_request(local_path))

    assert converter.calls == []


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    local_path = _write_file(tmp_path / "empty.pdf", b"")
    converter = FakeConverter()
    adapter = DoclingDocumentAdapter(settings=_settings(), converter=converter)

    with pytest.raises(InvalidDocumentInputError):
        adapter.convert(_request(local_path))

    assert converter.calls == []


def test_oversized_file_is_rejected(tmp_path: Path) -> None:
    local_path = _write_file(tmp_path / "large.pdf", b"1234")
    converter = FakeConverter()
    adapter = DoclingDocumentAdapter(settings=_settings(max_file_bytes=3), converter=converter)

    with pytest.raises(DocumentTooLargeError):
        adapter.convert(_request(local_path))

    assert converter.calls == []


def test_pdf_mime_with_docx_suffix_is_rejected(tmp_path: Path) -> None:
    local_path = _write_file(tmp_path / "upload.docx", b"PK\x03\x04docx-bytes")
    adapter = DoclingDocumentAdapter(settings=_settings(), converter=FakeConverter())

    with pytest.raises(UnsupportedDocumentTypeError):
        adapter.convert(_request(local_path, content_type=PDF_CONTENT_TYPE))


def test_docx_mime_with_pdf_suffix_is_rejected(tmp_path: Path) -> None:
    local_path = _write_file(tmp_path / "upload.pdf", b"%PDF-1.7\ncontent")
    adapter = DoclingDocumentAdapter(settings=_settings(), converter=FakeConverter())

    with pytest.raises(UnsupportedDocumentTypeError):
        adapter.convert(
            _request(local_path, content_type=DOCX_CONTENT_TYPE, file_name="report.docx")
        )


def test_original_file_name_suffix_mismatch_is_rejected(tmp_path: Path) -> None:
    local_path = _write_file(tmp_path / "upload.pdf", b"%PDF-1.7\ncontent")
    adapter = DoclingDocumentAdapter(settings=_settings(), converter=FakeConverter())

    with pytest.raises(UnsupportedDocumentTypeError):
        adapter.convert(_request(local_path, file_name="report.docx"))


def test_case_insensitive_suffix_matching_is_accepted(tmp_path: Path) -> None:
    local_path = _write_file(tmp_path / "upload.PDF", b"%PDF-1.7\ncontent")
    converter = FakeConverter(result={"documents": [Document(content="chunk")]})
    adapter = DoclingDocumentAdapter(settings=_settings(), converter=converter)

    result = adapter.convert(_request(local_path, file_name="REPORT.PDF"))

    assert result.file_name == "REPORT.PDF"
    assert converter.calls == [{"paths": [local_path], "sources": None, "meta": None}]


def test_symbolic_links_are_rejected_when_supported(tmp_path: Path) -> None:
    target = _write_file(tmp_path / "target.pdf", b"%PDF-1.7\ncontent")
    link_path = tmp_path / "linked.pdf"
    try:
        link_path.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation is not permitted on this platform: {exc}")

    adapter = DoclingDocumentAdapter(settings=_settings(), converter=FakeConverter())

    with pytest.raises(InvalidDocumentInputError):
        adapter.convert(_request(link_path))


def test_hash_is_deterministic_lowercase_and_64_characters(tmp_path: Path) -> None:
    content = b"%PDF-1.7\nsame-content"
    local_path = _write_file(tmp_path / "hash.pdf", content)
    adapter = DoclingDocumentAdapter(settings=_settings(), converter=FakeConverter())

    first = adapter.convert(_request(local_path))
    second = adapter.convert(_request(local_path))
    expected_hash = hashlib.sha256(content).hexdigest()

    assert first.file_hash == expected_hash
    assert second.file_hash == expected_hash
    assert len(first.file_hash) == 64
    assert first.file_hash == first.file_hash.lower()


def test_different_file_bytes_produce_different_hashes(tmp_path: Path) -> None:
    first_path = _write_file(tmp_path / "first.pdf", b"%PDF-1.7\none")
    second_path = _write_file(tmp_path / "second.pdf", b"%PDF-1.7\ntwo")
    adapter = DoclingDocumentAdapter(settings=_settings(), converter=FakeConverter())

    first = adapter.convert(_request(first_path, file_name="first.pdf"))
    second = adapter.convert(_request(second_path, file_name="second.pdf"))

    assert first.file_hash != second.file_hash


def test_hash_is_computed_before_chunk_ids_are_produced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    local_path = _write_file(tmp_path / "order.pdf", b"%PDF-1.7\ncontent")
    converter = FakeConverter(result={"documents": [Document(content="chunk")]})
    adapter = DoclingDocumentAdapter(settings=_settings(), converter=converter)

    original_compute_file_hash = DoclingDocumentAdapter._compute_file_hash
    original_build_document_chunk_id = build_document_chunk_id

    def tracking_compute_file_hash(self: DoclingDocumentAdapter, path: Path) -> str:
        events.append("hash")
        return original_compute_file_hash(self, path)

    def tracking_build_document_chunk_id(file_hash: str, chunk_index: int) -> str:
        events.append("id")
        return original_build_document_chunk_id(file_hash, chunk_index)

    original_run = converter.run

    def tracking_run(
        *,
        paths: list[str | Path] | None = None,
        sources: Any = None,
        meta: Any = None,
    ) -> dict[str, Any]:
        events.append("converter")
        return original_run(paths=paths, sources=sources, meta=meta)

    monkeypatch.setattr(DoclingDocumentAdapter, "_compute_file_hash", tracking_compute_file_hash)
    monkeypatch.setattr(
        "hay_v2_bot.adapters.docling_adapter.build_document_chunk_id",
        tracking_build_document_chunk_id,
    )
    converter.run = tracking_run

    result = adapter.convert(_request(local_path))

    assert result.documents[0].id == build_document_chunk_id(result.file_hash, 0)
    assert events == ["hash", "converter", "id"]


def test_converter_receives_exactly_one_local_path_via_paths_argument(tmp_path: Path) -> None:
    local_path = _write_file(tmp_path / "input.pdf", b"%PDF-1.7\ncontent")
    converter = FakeConverter()
    adapter = DoclingDocumentAdapter(settings=_settings(), converter=converter)

    adapter.convert(_request(local_path))

    assert converter.calls == [{"paths": [local_path], "sources": None, "meta": None}]


def test_converter_runtime_failure_becomes_document_conversion_error(tmp_path: Path) -> None:
    local_path = _write_file(tmp_path / "boom.pdf", b"%PDF-1.7\ncontent")
    runtime_error = RuntimeError("boom")
    adapter = DoclingDocumentAdapter(
        settings=_settings(),
        converter=FakeConverter(exception=runtime_error),
    )

    with pytest.raises(DocumentConversionError) as exc_info:
        adapter.convert(_request(local_path))

    assert exc_info.value.__cause__ is runtime_error


@pytest.mark.parametrize("exception", [KeyboardInterrupt(), SystemExit(2)])
def test_keyboard_interrupt_and_system_exit_are_not_swallowed(
    tmp_path: Path,
    exception: BaseException,
) -> None:
    local_path = _write_file(tmp_path / "interrupt.pdf", b"%PDF-1.7\ncontent")
    adapter = DoclingDocumentAdapter(
        settings=_settings(),
        converter=FakeConverter(exception=exception),
    )

    with pytest.raises(type(exception)):
        adapter.convert(_request(local_path))


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"documents": None},
        {"documents": "bad"},
    ],
)
def test_malformed_converter_result_is_rejected(tmp_path: Path, result: dict[str, Any]) -> None:
    local_path = _write_file(tmp_path / "malformed.pdf", b"%PDF-1.7\ncontent")
    adapter = DoclingDocumentAdapter(settings=_settings(), converter=FakeConverter(result=result))

    with pytest.raises(DocumentConversionError):
        adapter.convert(_request(local_path))


def test_non_document_values_are_rejected(tmp_path: Path) -> None:
    local_path = _write_file(tmp_path / "bad.pdf", b"%PDF-1.7\ncontent")
    adapter = DoclingDocumentAdapter(
        settings=_settings(),
        converter=FakeConverter(result={"documents": ["not-a-document"]}),
    )

    with pytest.raises(DocumentConversionError):
        adapter.convert(_request(local_path))


def test_output_order_is_preserved_blank_chunks_removed_and_metadata_allowlisted(
    tmp_path: Path,
) -> None:
    local_path = _write_file(tmp_path / "normalize.pdf", b"%PDF-1.7\ncontent")
    converter = FakeConverter(
        result={
            "documents": [
                Document(
                    id="ignored-id-1",
                    content="  First chunk  ",
                    meta={
                        "page_number": 1,
                        "headings": ["Intro", "Summary"],
                        "unknown": "discard-me",
                        "local_path": "C:/secret.pdf",
                    },
                ),
                Document(id="ignored-id-2", content="   ", meta={"page_number": 2}),
                Document(id="ignored-id-3", content="\nSecond chunk\n", meta={}),
            ]
        }
    )
    adapter = DoclingDocumentAdapter(settings=_settings(), converter=converter)

    result = adapter.convert(_request(local_path, user_id=987, file_name="normalize.pdf"))

    expected_hash = hashlib.sha256(local_path.read_bytes()).hexdigest()

    assert [document.content for document in result.documents] == ["First chunk", "Second chunk"]
    assert [document.id for document in result.documents] == [
        build_document_chunk_id(expected_hash, 0),
        build_document_chunk_id(expected_hash, 1),
    ]
    assert result.documents[0].meta == {
        "record_type": "document_chunk",
        "user_id": 987,
        "file_name": "normalize.pdf",
        "file_hash": expected_hash,
        "chunk_index": 0,
        "content_type": PDF_CONTENT_TYPE,
        "uploaded_at": "2026-07-11T12:00:00+00:00",
        "page_number": 1,
        "headings": ["Intro", "Summary"],
    }
    assert result.documents[1].meta == {
        "record_type": "document_chunk",
        "user_id": 987,
        "file_name": "normalize.pdf",
        "file_hash": expected_hash,
        "chunk_index": 1,
        "content_type": PDF_CONTENT_TYPE,
        "uploaded_at": "2026-07-11T12:00:00+00:00",
    }
    assert "unknown" not in result.documents[0].meta
    assert "local_path" not in result.documents[0].meta


@pytest.mark.parametrize(
    "meta",
    [
        {"page_number": 0},
        {"headings": ["Intro", "   "]},
    ],
)
def test_invalid_direct_optional_metadata_raises_controlled_project_error(
    tmp_path: Path,
    meta: dict[str, Any],
) -> None:
    local_path = _write_file(tmp_path / "invalid-meta.pdf", b"%PDF-1.7\ncontent")
    adapter = DoclingDocumentAdapter(
        settings=_settings(),
        converter=FakeConverter(result={"documents": [Document(content="chunk", meta=meta)]}),
    )

    with pytest.raises(InvalidDocumentInputError):
        adapter.convert(_request(local_path))


def test_all_blank_output_raises_empty_document_error(tmp_path: Path) -> None:
    local_path = _write_file(tmp_path / "blank.pdf", b"%PDF-1.7\ncontent")
    adapter = DoclingDocumentAdapter(
        settings=_settings(),
        converter=FakeConverter(
            result={"documents": [Document(content="  "), Document(content="\n")]}
        ),
    )

    with pytest.raises(EmptyDocumentError):
        adapter.convert(_request(local_path))


def test_too_many_retained_chunks_raises_error(tmp_path: Path) -> None:
    local_path = _write_file(tmp_path / "many.pdf", b"%PDF-1.7\ncontent")
    adapter = DoclingDocumentAdapter(
        settings=_settings(max_chunks_per_document=1),
        converter=FakeConverter(
            result={
                "documents": [
                    Document(content="first"),
                    Document(content="   "),
                    Document(content="second"),
                ]
            }
        ),
    )

    with pytest.raises(TooManyDocumentChunksError):
        adapter.convert(_request(local_path))


def test_non_text_document_chunk_is_rejected(tmp_path: Path) -> None:
    local_path = _write_file(tmp_path / "non-text.pdf", b"%PDF-1.7\ncontent")
    adapter = DoclingDocumentAdapter(
        settings=_settings(),
        converter=FakeConverter(result={"documents": [Document(content=None)]}),
    )

    with pytest.raises(DocumentConversionError):
        adapter.convert(_request(local_path))


def test_request_model_is_immutable_and_normalizes_uploaded_at_to_utc(tmp_path: Path) -> None:
    request = _request(
        tmp_path / "missing.pdf",
        uploaded_at=datetime(2026, 7, 11, 15, 0, tzinfo=timezone(timedelta(hours=3))),
    )

    assert request.uploaded_at == UPLOAD_TIME
    with pytest.raises(ValidationError):
        request.file_name = "changed.pdf"  # type: ignore[misc]


def test_request_rejects_bool_user_id_and_allows_missing_local_path_in_model(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "still-missing.pdf"

    request = _request(missing_path)

    assert request.local_path == missing_path
    with pytest.raises(ValidationError):
        _request(missing_path, user_id=True)


def test_result_model_is_immutable_and_uses_tuple_documents() -> None:
    result = DocumentConversionResult(
        file_hash="a" * 64,
        file_name="report.pdf",
        content_type=PDF_CONTENT_TYPE,
        documents=[Document(content="chunk")],
    )

    assert isinstance(result.documents, tuple)
    assert result.chunk_count == 1
    with pytest.raises(ValidationError):
        result.file_name = "changed.pdf"  # type: ignore[misc]


def test_result_requires_non_empty_document_tuple() -> None:
    with pytest.raises(ValidationError):
        DocumentConversionResult(
            file_hash="a" * 64,
            file_name="report.pdf",
            content_type=PDF_CONTENT_TYPE,
            documents=[],
        )
