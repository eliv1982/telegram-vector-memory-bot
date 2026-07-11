"""Offline tests for scripts/smoke_test_docling.py."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import pytest
from hay_v2_bot.adapters import InvalidDocumentInputError
from hay_v2_bot.models import (
    DOCX_CONTENT_TYPE,
    PDF_CONTENT_TYPE,
    DocumentConversionResult,
)
from haystack import Document

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
VALID_HASH = "a" * 64


def _load_script() -> dict[str, Any]:
    namespace = runpy.run_path(str(SCRIPTS_DIR / "smoke_test_docling.py"))
    return namespace["main"].__globals__


def _make_result(
    *,
    content_type: str = PDF_CONTENT_TYPE,
    documents: list[Document] | None = None,
) -> DocumentConversionResult:
    if documents is None:
        documents = [
            Document(
                id="doc-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-chunk-000000",
                content="First line\nSecond line with a useful fact.",
                meta={
                    "record_type": "document_chunk",
                    "user_id": 123,
                    "file_name": "sample.pdf",
                    "file_hash": VALID_HASH,
                    "chunk_index": 0,
                    "content_type": content_type,
                    "uploaded_at": "2026-07-11T12:00:00+00:00",
                },
            )
        ]
    return DocumentConversionResult(
        file_hash=VALID_HASH,
        file_name="sample.pdf" if content_type == PDF_CONTENT_TYPE else "sample.docx",
        content_type=content_type,
        documents=documents,
    )


class FakeRealConverter:
    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        *,
        paths: list[str | Path] | None = None,
        sources: Any = None,
        meta: Any = None,
    ) -> dict[str, Any]:
        self.calls.append({"paths": paths, "sources": sources, "meta": meta})
        return self._result


class FakeAdapter:
    def __init__(self, settings: Any, converter: Any) -> None:
        self.settings = settings
        self.converter = converter

    def convert(self, request: Any) -> DocumentConversionResult:
        self.converter.run(paths=[request.local_path])
        return _make_result(
            documents=[
                Document(
                    id="normalized-0",
                    content="Line one\nLine two\nLine three",
                    meta={
                        "record_type": "document_chunk",
                        "user_id": request.user_id,
                        "file_name": request.file_name,
                        "file_hash": VALID_HASH,
                        "chunk_index": 0,
                        "content_type": request.content_type,
                        "uploaded_at": "2026-07-11T12:00:00+00:00",
                    },
                ),
                Document(
                    id="normalized-1",
                    content="Second normalized chunk",
                    meta={
                        "record_type": "document_chunk",
                        "user_id": request.user_id,
                        "file_name": request.file_name,
                        "file_hash": VALID_HASH,
                        "chunk_index": 1,
                        "content_type": request.content_type,
                        "uploaded_at": "2026-07-11T12:00:00+00:00",
                        "page_number": 2,
                        "headings": ["Document control"],
                    },
                ),
            ]
        )


class FailingAdapter:
    def __init__(self, settings: Any, converter: Any) -> None:
        self.settings = settings
        self.converter = converter

    def convert(self, request: Any) -> DocumentConversionResult:
        raise InvalidDocumentInputError("Document file does not exist")


class InterruptingAdapter:
    def __init__(self, settings: Any, converter: Any) -> None:
        self.settings = settings
        self.converter = converter

    def convert(self, request: Any) -> DocumentConversionResult:
        raise KeyboardInterrupt()


@pytest.fixture
def script_ns() -> dict[str, Any]:
    return _load_script()


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"%PDF-1.7\nsample")
    return path


@pytest.fixture
def sample_docx(tmp_path: Path) -> Path:
    path = tmp_path / "sample.docx"
    path.write_bytes(b"PK\x03\x04docx")
    return path


def test_cli_parser_accepts_valid_pdf(script_ns: dict[str, Any], sample_pdf: Path) -> None:
    parser = script_ns["build_arg_parser"]()

    parsed = parser.parse_args(
        [
            "--file",
            str(sample_pdf),
            "--content-type",
            PDF_CONTENT_TYPE,
            "--user-id",
            "123",
        ]
    )

    assert parsed.file == sample_pdf
    assert parsed.content_type == PDF_CONTENT_TYPE
    assert parsed.user_id == 123


def test_cli_parser_accepts_valid_docx(script_ns: dict[str, Any], sample_docx: Path) -> None:
    parser = script_ns["build_arg_parser"]()

    parsed = parser.parse_args(
        [
            "--file",
            str(sample_docx),
            "--content-type",
            DOCX_CONTENT_TYPE,
            "--user-id",
            "456",
        ]
    )

    assert parsed.file == sample_docx
    assert parsed.content_type == DOCX_CONTENT_TYPE
    assert parsed.user_id == 456


def test_unsupported_mime_is_rejected(script_ns: dict[str, Any], sample_pdf: Path) -> None:
    parser = script_ns["build_arg_parser"]()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--file",
                str(sample_pdf),
                "--content-type",
                "text/plain",
                "--user-id",
                "123",
            ]
        )


def test_invalid_user_id_is_rejected(script_ns: dict[str, Any], sample_pdf: Path) -> None:
    parser = script_ns["build_arg_parser"]()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--file",
                str(sample_pdf),
                "--content-type",
                PDF_CONTENT_TYPE,
                "--user-id",
                "0",
            ]
        )


def test_invalid_preview_length_is_rejected(script_ns: dict[str, Any], sample_pdf: Path) -> None:
    parser = script_ns["build_arg_parser"]()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--file",
                str(sample_pdf),
                "--content-type",
                PDF_CONTENT_TYPE,
                "--user-id",
                "123",
                "--preview-chars",
                "-1",
            ]
        )


def test_settings_are_created_without_reading_dotenv(
    script_ns: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("DOCUSCOPE_MAX_FILE_BYTES=1\n", encoding="utf-8")

    settings = script_ns["create_settings"]()

    assert settings.max_file_bytes == 20 * 1024 * 1024
    assert settings.max_chunks_per_document == 2000


def test_wrapper_delegates_once_and_retains_captured_raw_documents(
    script_ns: dict[str, Any],
) -> None:
    raw_documents = [
        Document(content="Raw one", meta={"page_number": 1}),
        Document(content="Raw two", meta={"headings": ["Intro"]}),
    ]
    real_converter = FakeRealConverter({"documents": raw_documents})
    wrapper = script_ns["CapturingDoclingConverterWrapper"](converter=real_converter)

    result = wrapper.run(paths=[Path("sample.pdf")])

    assert result["documents"] == raw_documents
    assert wrapper.run_call_count == 1
    assert wrapper.raw_documents == tuple(raw_documents)
    assert real_converter.calls == [{"paths": [Path("sample.pdf")], "sources": None, "meta": None}]


def test_report_uses_adapter_result_for_normalized_output(
    script_ns: dict[str, Any],
    sample_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_documents = [
        Document(
            content="Raw chunk",
            meta={
                "page_number": 1,
                "headings": ["Intro"],
                "section": {"items": [{"label": "A"}]},
            },
        )
    ]
    wrapper = script_ns["CapturingDoclingConverterWrapper"](
        converter=FakeRealConverter({"documents": raw_documents})
    )
    monkeypatch.setitem(script_ns, "capture_cache_activity_snapshot", lambda: {})
    monkeypatch.setitem(script_ns, "detect_cache_activity", lambda before, after: None)

    report = script_ns["run_smoke_conversion"](
        file_path=sample_pdf,
        content_type=PDF_CONTENT_TYPE,
        user_id=123,
        preview_chars=12,
        settings_factory=script_ns["create_settings"],
        wrapper_factory=lambda: wrapper,
        adapter_class=FakeAdapter,
    )

    assert report["normalized_chunk_count"] == 2
    assert report["normalized_document_ids"] == ["normalized-0", "normalized-1"]
    assert report["first_normalized_chunk_preview"] == "Line one Lin"
    assert report["last_normalized_chunk_preview"] == "Second norma"
    assert report["normalized_chunks_with_page_number_count"] == 1
    assert report["normalized_chunks_with_headings_count"] == 1
    assert report["raw_chunk_count"] == 1
    assert set(report) >= {
        "status",
        "file_name",
        "content_type",
        "file_size_bytes",
        "file_hash",
        "raw_chunk_count",
        "normalized_chunk_count",
        "normalized_document_ids",
        "first_normalized_chunk_preview",
        "last_normalized_chunk_preview",
        "normalized_metadata_sample",
        "raw_metadata_top_level_keys",
        "raw_metadata_key_paths",
        "raw_chunks_with_direct_page_number_count",
        "raw_chunks_with_direct_headings_count",
        "normalized_chunks_with_page_number_count",
        "normalized_chunks_with_headings_count",
        "duration_seconds",
        "model_or_tokenizer_cache_activity_observed",
        "warnings",
    }


def test_report_omits_absolute_paths_and_sensitive_keys_and_unknown_object_values(
    script_ns: dict[str, Any],
    sample_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local_secret_path = tmp_path / "private" / "secret.txt"
    raw_documents = [
        Document(
            content="Raw chunk",
            meta={
                "page_number": 1,
                "headings": ["Intro"],
                "source_path": str(local_secret_path),
                "auth_token": "secret-token",
                "metadata": {
                    "public_value": "safe",
                    "nested_url": "https://example.invalid",
                    "object_value": object(),
                },
            },
        )
    ]
    wrapper = script_ns["CapturingDoclingConverterWrapper"](
        converter=FakeRealConverter({"documents": raw_documents})
    )
    monkeypatch.setitem(script_ns, "capture_cache_activity_snapshot", lambda: {})
    monkeypatch.setitem(script_ns, "detect_cache_activity", lambda before, after: None)

    report = script_ns["run_smoke_conversion"](
        file_path=sample_pdf,
        content_type=PDF_CONTENT_TYPE,
        user_id=123,
        settings_factory=script_ns["create_settings"],
        wrapper_factory=lambda: wrapper,
        adapter_class=FakeAdapter,
    )
    rendered = json.dumps(report, ensure_ascii=False)

    assert str(sample_pdf.parent) not in rendered
    assert str(local_secret_path) not in rendered
    assert "source_path" not in report["raw_metadata_top_level_keys"]
    assert "auth_token" not in report["raw_metadata_top_level_keys"]
    assert not any("url" in path.lower() for path in report["raw_metadata_key_paths"])
    assert "object at 0x" not in rendered
    assert report["warnings"]


def test_json_output_is_valid_and_controlled_failure_has_no_traceback(
    script_ns: dict[str, Any],
    sample_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    raw_documents = [Document(content="Raw", meta={})]
    wrapper = script_ns["CapturingDoclingConverterWrapper"](
        converter=FakeRealConverter({"documents": raw_documents})
    )
    monkeypatch.setitem(script_ns, "capture_cache_activity_snapshot", lambda: {})
    monkeypatch.setitem(script_ns, "detect_cache_activity", lambda before, after: False)
    monkeypatch.setitem(script_ns, "CapturingDoclingConverterWrapper", lambda: wrapper)
    monkeypatch.setitem(script_ns, "DoclingDocumentAdapter", FakeAdapter)

    output_path = tmp_path / "report.json"
    exit_code = script_ns["main"](
        [
            "--file",
            str(sample_pdf),
            "--content-type",
            PDF_CONTENT_TYPE,
            "--user-id",
            "123",
            "--json-output",
            str(output_path),
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Smoke test passed for sample.pdf" in stdout
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "success"

    monkeypatch.setitem(
        script_ns,
        "CapturingDoclingConverterWrapper",
        script_ns["CapturingDoclingConverterWrapper"],
    )
    monkeypatch.setitem(script_ns, "DoclingDocumentAdapter", FailingAdapter)

    failure_code = script_ns["main"](
        [
            "--file",
            str(sample_pdf),
            "--content-type",
            PDF_CONTENT_TYPE,
            "--user-id",
            "123",
        ]
    )

    captured = capsys.readouterr()
    assert failure_code == script_ns["EXIT_CONTROLLED_FAILURE"]
    assert "Smoke test failed: Document file does not exist" in captured.err
    assert "Traceback" not in captured.err


def test_keyboard_interrupt_uses_documented_exit_behavior(
    script_ns: dict[str, Any],
    sample_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setitem(script_ns, "DoclingDocumentAdapter", InterruptingAdapter)

    exit_code = script_ns["main"](
        [
            "--file",
            str(sample_pdf),
            "--content-type",
            PDF_CONTENT_TYPE,
            "--user-id",
            "123",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == script_ns["EXIT_INTERRUPTED"]
    assert "Interrupted." in captured.err
    assert "DocumentConversionError" not in captured.err


def test_main_does_not_leak_service_secrets_from_environment(
    script_ns: dict[str, Any],
    sample_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("PINECONE_API_KEY", "pinecone-secret")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-secret")
    raw_documents = [Document(content="Raw", meta={"safe": "value"})]
    wrapper = script_ns["CapturingDoclingConverterWrapper"](
        converter=FakeRealConverter({"documents": raw_documents})
    )
    monkeypatch.setitem(script_ns, "capture_cache_activity_snapshot", lambda: {})
    monkeypatch.setitem(script_ns, "detect_cache_activity", lambda before, after: False)
    monkeypatch.setitem(script_ns, "CapturingDoclingConverterWrapper", lambda: wrapper)
    monkeypatch.setitem(script_ns, "DoclingDocumentAdapter", FakeAdapter)

    exit_code = script_ns["main"](
        [
            "--file",
            str(sample_pdf),
            "--content-type",
            PDF_CONTENT_TYPE,
            "--user-id",
            "123",
        ]
    )

    captured = capsys.readouterr()
    rendered = captured.out + captured.err
    assert exit_code == 0
    assert "openai-secret" not in rendered
    assert "pinecone-secret" not in rendered
    assert "telegram-secret" not in rendered
