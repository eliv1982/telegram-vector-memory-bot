"""Offline tests for scripts/smoke_test_document_rag.py."""

from __future__ import annotations

import json
import runpy
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from hay_v2_bot.models import PDF_CONTENT_TYPE
from hay_v2_bot.models.rag import DocumentAnswer, DocumentIngestionOutcome, DocumentSource
from hay_v2_bot.services import DocumentQuestionError

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
VALID_HASH = "a" * 64


class FakeStoreFactory:
    def __init__(
        self,
        *,
        remaining_sequences: Sequence[tuple[str, ...]] | None = None,
    ) -> None:
        if remaining_sequences is None:
            remaining_sequences = [()]
        self._remaining_sequences = tuple(tuple(ids) for ids in remaining_sequences)
        self._fetch_index = 0
        self.delete_calls: list[tuple[int, tuple[str, ...]]] = []
        self.fetch_calls: list[tuple[int, tuple[str, ...]]] = []

    def delete_documents(self, user_id: int, document_ids: tuple[str, ...]) -> None:
        self.delete_calls.append((user_id, tuple(document_ids)))

    def fetch_existing_document_ids(
        self,
        user_id: int,
        document_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        self.fetch_calls.append((user_id, tuple(document_ids)))
        index = min(self._fetch_index, len(self._remaining_sequences) - 1)
        self._fetch_index += 1
        return self._remaining_sequences[index]


class FakeClock:
    def __init__(self) -> None:
        self.current = 0.0
        self.sleep_calls: list[float] = []

    def monotonic(self) -> float:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.current += seconds

    def interrupting_sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        raise KeyboardInterrupt()


class FakeService:
    def __init__(
        self,
        *,
        ingestion_outcome: DocumentIngestionOutcome | None = None,
        answer: DocumentAnswer | None = None,
        ingest_exception: BaseException | None = None,
        answer_exception: BaseException | None = None,
    ) -> None:
        self._ingestion_outcome = ingestion_outcome
        self._answer = answer
        self._ingest_exception = ingest_exception
        self._answer_exception = answer_exception
        self.ingest_calls: list[Any] = []
        self.answer_calls: list[tuple[int, str]] = []

    def ingest_and_summarize(self, request: Any) -> DocumentIngestionOutcome:
        self.ingest_calls.append(request)
        if self._ingest_exception is not None:
            raise self._ingest_exception
        assert self._ingestion_outcome is not None
        return self._ingestion_outcome

    def answer_question(self, user_id: int, question: str) -> DocumentAnswer:
        self.answer_calls.append((user_id, question))
        if self._answer_exception is not None:
            raise self._answer_exception
        assert self._answer is not None
        return self._answer


def _load_script() -> dict[str, Any]:
    namespace = runpy.run_path(str(SCRIPTS_DIR / "smoke_test_document_rag.py"))
    return namespace["main"].__globals__


def _make_ingestion_outcome() -> DocumentIngestionOutcome:
    return DocumentIngestionOutcome(
        file_hash=VALID_HASH,
        file_name="docuscope_smoke.pdf",
        content_type=PDF_CONTENT_TYPE,
        chunk_count=2,
        documents_written=2,
        document_ids=("doc-1", "doc-2"),
        summary="The Orion pilot budget was approved at 4.2 million euros.",
    )


def _make_answer() -> DocumentAnswer:
    return DocumentAnswer(
        answer="The approved budget for the Orion pilot is 4.2 million euros.",
        sources=(
            DocumentSource(
                document_id="doc-1",
                file_name="docuscope_smoke.pdf",
                chunk_index=0,
                page_number=1,
                score=0.91,
            ),
        ),
        used_document_count=1,
        fallback_used=False,
    )


@pytest.fixture
def script_ns() -> dict[str, Any]:
    return _load_script()


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "docuscope_smoke.pdf"
    path.write_bytes(b"%PDF-1.7\nsample")
    return path


def test_cli_parser_accepts_controlled_pdf_and_required_question(
    script_ns: dict[str, Any],
    sample_pdf: Path,
) -> None:
    parser = script_ns["build_arg_parser"]()

    parsed = parser.parse_args(
        [
            "--file",
            str(sample_pdf),
            "--content-type",
            PDF_CONTENT_TYPE,
            "--user-id",
            "900000002",
            "--question",
            "What is the approved budget for the Orion pilot?",
        ]
    )

    assert parsed.file == sample_pdf
    assert parsed.content_type == PDF_CONTENT_TYPE
    assert parsed.user_id == 900000002
    assert parsed.question == "What is the approved budget for the Orion pilot?"


def test_run_document_rag_smoke_success_report_is_json_safe_and_cleanup_verified(
    script_ns: dict[str, Any],
    sample_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("PINECONE_API_KEY", "pinecone-secret")
    store_factory = FakeStoreFactory(remaining_sequences=[()])
    clock = FakeClock()
    service = FakeService(ingestion_outcome=_make_ingestion_outcome(), answer=_make_answer())

    report = script_ns["run_document_rag_smoke"](
        file_path=sample_pdf,
        content_type=PDF_CONTENT_TYPE,
        user_id=900000002,
        question="What is the approved budget for the Orion pilot?",
        processing_settings_factory=lambda: object(),
        rag_settings_factory=lambda: object(),
        document_store_factory_builder=lambda _: store_factory,
        service_factory=lambda *_: service,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    rendered = json.dumps(report, ensure_ascii=False)

    assert report["status"] == "success"
    assert report["file_name"] == "docuscope_smoke.pdf"
    assert report["content_type"] == PDF_CONTENT_TYPE
    assert report["chunk_count"] == 2
    assert report["documents_written"] == 2
    assert report["summary"] == "The Orion pilot budget was approved at 4.2 million euros."
    assert report["answer"] == "The approved budget for the Orion pilot is 4.2 million euros."
    assert report["fallback_used"] is False
    assert report["sources"] == [
        {
            "document_id": "doc-1",
            "file_name": "docuscope_smoke.pdf",
            "chunk_index": 0,
            "page_number": 1,
            "score": 0.91,
        }
    ]
    assert report["cleanup_attempted"] is True
    assert report["cleanup_succeeded"] is True
    assert report["cleanup_verification_poll_count"] == 1
    assert store_factory.delete_calls == [(900000002, ("doc-1", "doc-2"))]
    assert store_factory.fetch_calls == [(900000002, ("doc-1", "doc-2"))]
    assert clock.sleep_calls == []
    assert str(sample_pdf.parent) not in rendered
    assert "openai-secret" not in rendered
    assert "pinecone-secret" not in rendered


def test_cleanup_polling_handles_one_stale_result_then_success(
    script_ns: dict[str, Any],
    sample_pdf: Path,
) -> None:
    store_factory = FakeStoreFactory(remaining_sequences=[("doc-1",), ()])
    clock = FakeClock()
    service = FakeService(ingestion_outcome=_make_ingestion_outcome(), answer=_make_answer())

    report = script_ns["run_document_rag_smoke"](
        file_path=sample_pdf,
        content_type=PDF_CONTENT_TYPE,
        user_id=900000002,
        question="What is the approved budget for the Orion pilot?",
        processing_settings_factory=lambda: object(),
        rag_settings_factory=lambda: object(),
        document_store_factory_builder=lambda _: store_factory,
        service_factory=lambda *_: service,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert report["cleanup_attempted"] is True
    assert report["cleanup_succeeded"] is True
    assert report["cleanup_verification_poll_count"] == 2
    assert store_factory.delete_calls == [(900000002, ("doc-1", "doc-2"))]
    assert store_factory.fetch_calls == [
        (900000002, ("doc-1", "doc-2")),
        (900000002, ("doc-1", "doc-2")),
    ]
    assert clock.sleep_calls == [0.5]


def test_cleanup_polling_handles_multiple_stale_polls_then_success(
    script_ns: dict[str, Any],
    sample_pdf: Path,
) -> None:
    store_factory = FakeStoreFactory(
        remaining_sequences=[
            ("doc-1", "doc-2"),
            ("doc-1",),
            ("doc-2",),
            (),
        ]
    )
    clock = FakeClock()
    service = FakeService(ingestion_outcome=_make_ingestion_outcome(), answer=_make_answer())

    report = script_ns["run_document_rag_smoke"](
        file_path=sample_pdf,
        content_type=PDF_CONTENT_TYPE,
        user_id=900000002,
        question="What is the approved budget for the Orion pilot?",
        processing_settings_factory=lambda: object(),
        rag_settings_factory=lambda: object(),
        document_store_factory_builder=lambda _: store_factory,
        service_factory=lambda *_: service,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert report["cleanup_attempted"] is True
    assert report["cleanup_succeeded"] is True
    assert report["cleanup_verification_poll_count"] == 4
    assert store_factory.delete_calls == [(900000002, ("doc-1", "doc-2"))]
    assert store_factory.fetch_calls == [
        (900000002, ("doc-1", "doc-2")),
        (900000002, ("doc-1", "doc-2")),
        (900000002, ("doc-1", "doc-2")),
        (900000002, ("doc-1", "doc-2")),
    ]
    assert clock.sleep_calls == [0.5, 0.5, 0.5]


def test_main_writes_final_json_with_automatic_cleanup_success_after_stale_visibility(
    script_ns: dict[str, Any],
    sample_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    store_factory = FakeStoreFactory(remaining_sequences=[("doc-1",), ()])
    clock = FakeClock()
    service = FakeService(ingestion_outcome=_make_ingestion_outcome(), answer=_make_answer())
    output_path = tmp_path / "report.json"

    monkeypatch.setattr(script_ns["time"], "monotonic", clock.monotonic)
    monkeypatch.setattr(script_ns["time"], "sleep", clock.sleep)
    script_ns["create_processing_settings"] = lambda: object()
    script_ns["create_rag_settings"] = lambda: object()
    script_ns["create_document_store_factory"] = lambda _: store_factory
    script_ns["create_document_rag_service"] = lambda *_: service

    exit_code = script_ns["main"](
        [
            "--file",
            str(sample_pdf),
            "--content-type",
            PDF_CONTENT_TYPE,
            "--user-id",
            "900000002",
            "--question",
            "What is the approved budget for the Orion pilot?",
            "--json-output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "Smoke test passed for docuscope_smoke.pdf" in captured.out
    assert report["status"] == "success"
    assert report["cleanup_attempted"] is True
    assert report["cleanup_succeeded"] is True
    assert report["cleanup_verification_poll_count"] == 2
    assert store_factory.delete_calls == [(900000002, ("doc-1", "doc-2"))]
    assert store_factory.fetch_calls == [
        (900000002, ("doc-1", "doc-2")),
        (900000002, ("doc-1", "doc-2")),
    ]
    assert clock.sleep_calls == [0.5]


def test_cleanup_timeout_returns_non_zero_and_json_contains_final_cleanup_state(
    script_ns: dict[str, Any],
    sample_pdf: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_factory = FakeStoreFactory(remaining_sequences=[("doc-1",)])
    clock = FakeClock()
    service = FakeService(ingestion_outcome=_make_ingestion_outcome(), answer=_make_answer())
    output_path = tmp_path / "report.json"

    monkeypatch.setattr(script_ns["time"], "monotonic", clock.monotonic)
    monkeypatch.setattr(script_ns["time"], "sleep", clock.sleep)
    script_ns["CLEANUP_POLL_INTERVAL_SECONDS"] = 0.5
    script_ns["CLEANUP_TIMEOUT_SECONDS"] = 1.0
    script_ns["create_processing_settings"] = lambda: object()
    script_ns["create_rag_settings"] = lambda: object()
    script_ns["create_document_store_factory"] = lambda _: store_factory
    script_ns["create_document_rag_service"] = lambda *_: service

    exit_code = script_ns["main"](
        [
            "--file",
            str(sample_pdf),
            "--content-type",
            PDF_CONTENT_TYPE,
            "--user-id",
            "900000002",
            "--question",
            "What is the approved budget for the Orion pilot?",
            "--json-output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == script_ns["EXIT_CONTROLLED_FAILURE"]
    assert (
        "Smoke test failed: Cleanup verification timed out before all inserted "
        "document IDs disappeared."
        in captured.err
    )
    assert "Traceback" not in captured.err
    assert report["status"] == "failure"
    assert report["cleanup_attempted"] is True
    assert report["cleanup_succeeded"] is False
    assert report["cleanup_verification_poll_count"] == 3
    assert report["document_ids"] == ["doc-1", "doc-2"]
    assert report["warnings"] == [
        "Some inserted document IDs remained visible after the cleanup verification timeout."
    ]
    assert store_factory.delete_calls == [(900000002, ("doc-1", "doc-2"))]
    assert store_factory.fetch_calls == [
        (900000002, ("doc-1", "doc-2")),
        (900000002, ("doc-1", "doc-2")),
        (900000002, ("doc-1", "doc-2")),
    ]
    assert clock.sleep_calls == [0.5, 0.5]


def test_run_document_rag_smoke_cleanup_happens_in_finally_after_question_failure(
    script_ns: dict[str, Any],
    sample_pdf: Path,
) -> None:
    store_factory = FakeStoreFactory(remaining_sequences=[("doc-1",), ()])
    clock = FakeClock()
    service = FakeService(
        ingestion_outcome=_make_ingestion_outcome(),
        answer_exception=DocumentQuestionError("Question answering failed"),
    )

    with pytest.raises(script_ns["SmokeScriptControlledFailure"]) as exc_info:
        script_ns["run_document_rag_smoke"](
            file_path=sample_pdf,
            content_type=PDF_CONTENT_TYPE,
            user_id=900000002,
            question="What is the approved budget for the Orion pilot?",
            processing_settings_factory=lambda: object(),
            rag_settings_factory=lambda: object(),
            document_store_factory_builder=lambda _: store_factory,
            service_factory=lambda *_: service,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    report = exc_info.value.report
    assert report["status"] == "failure"
    assert report["cleanup_attempted"] is True
    assert report["cleanup_succeeded"] is True
    assert report["cleanup_verification_poll_count"] == 2
    assert store_factory.delete_calls == [(900000002, ("doc-1", "doc-2"))]
    assert store_factory.fetch_calls == [
        (900000002, ("doc-1", "doc-2")),
        (900000002, ("doc-1", "doc-2")),
    ]


def test_keyboard_interrupt_during_cleanup_polling_returns_130(
    script_ns: dict[str, Any],
    sample_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store_factory = FakeStoreFactory(remaining_sequences=[("doc-1",)])
    clock = FakeClock()
    service = FakeService(ingestion_outcome=_make_ingestion_outcome(), answer=_make_answer())

    monkeypatch.setattr(script_ns["time"], "monotonic", clock.monotonic)
    monkeypatch.setattr(script_ns["time"], "sleep", clock.interrupting_sleep)
    script_ns["create_processing_settings"] = lambda: object()
    script_ns["create_rag_settings"] = lambda: object()
    script_ns["create_document_store_factory"] = lambda _: store_factory
    script_ns["create_document_rag_service"] = lambda *_: service

    exit_code = script_ns["main"](
        [
            "--file",
            str(sample_pdf),
            "--content-type",
            PDF_CONTENT_TYPE,
            "--user-id",
            "900000002",
            "--question",
            "What is the approved budget for the Orion pilot?",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == script_ns["EXIT_INTERRUPTED"]
    assert "Interrupted." in captured.err
    assert store_factory.delete_calls == [(900000002, ("doc-1", "doc-2"))]
    assert store_factory.fetch_calls == [(900000002, ("doc-1", "doc-2"))]
    assert clock.sleep_calls == [0.5]
