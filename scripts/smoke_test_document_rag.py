"""Run one live Stage 5 document RAG cycle and emit a safe JSON report."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hay_v2_bot.config import DocumentProcessingSettings, DocumentRagSettings  # noqa: E402
from hay_v2_bot.models import (  # noqa: E402
    SUPPORTED_DOCUMENT_CONTENT_TYPES,
    DocumentConversionRequest,
)
from hay_v2_bot.services import (  # noqa: E402
    DocumentIngestionError,
    DocumentQuestionError,
    DocumentRagService,
    DocumentSummaryError,
)
from hay_v2_bot.storage import DocumentStoreError, PineconeDocumentStoreFactory  # noqa: E402

EXIT_CONTROLLED_FAILURE = 3
EXIT_UNEXPECTED_FAILURE = 4
EXIT_INTERRUPTED = 130
CLEANUP_POLL_INTERVAL_SECONDS = 0.5
CLEANUP_TIMEOUT_SECONDS = 15.0


class SmokeScriptError(Exception):
    """Raised for controlled smoke-test failures with safe public messages."""


@dataclass(frozen=True)
class SmokeScriptControlledFailure(SmokeScriptError):
    """Raised when a controlled failure still has a safe report payload."""

    message: str
    report: dict[str, Any]

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class CleanupVerificationResult:
    """Result of bounded post-delete exact-ID visibility polling."""

    succeeded: bool
    remaining_ids: tuple[str, ...]
    poll_count: int


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the Stage 5 live smoke script."""
    parser = argparse.ArgumentParser(
        description=(
            "Run one live document RAG flow through the production Stage 5 "
            "adapter, Pinecone document store, and OpenAI-compatible models."
        )
    )
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Path to one local PDF or DOCX file.",
    )
    parser.add_argument(
        "--content-type",
        required=True,
        choices=sorted(SUPPORTED_DOCUMENT_CONTENT_TYPES),
        help="Exact MIME type for the input file.",
    )
    parser.add_argument(
        "--user-id",
        required=True,
        type=_parse_positive_int,
        help="Positive synthetic user ID used for the document namespace.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="One grounded question to answer from the uploaded document chunks.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional output path for the JSON report.",
    )
    parser.add_argument(
        "--keep-records",
        action="store_true",
        help="Skip cleanup of inserted document IDs after the smoke test.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print a traceback for unexpected internal failures.",
    )
    return parser


def create_processing_settings() -> DocumentProcessingSettings:
    """Construct production document-processing settings from .env and environment."""
    return DocumentProcessingSettings()


def create_rag_settings() -> DocumentRagSettings:
    """Construct production document-RAG settings from .env and environment."""
    return DocumentRagSettings()


def create_document_store_factory(
    rag_settings: DocumentRagSettings,
) -> PineconeDocumentStoreFactory:
    """Construct the production Pinecone document-store factory."""
    return PineconeDocumentStoreFactory(rag_settings)


def create_document_rag_service(
    processing_settings: DocumentProcessingSettings,
    rag_settings: DocumentRagSettings,
    document_store_factory: PineconeDocumentStoreFactory,
) -> DocumentRagService:
    """Construct the production document RAG service."""
    return DocumentRagService(
        processing_settings,
        rag_settings,
        document_store_factory=document_store_factory,
    )


def run_document_rag_smoke(
    *,
    file_path: Path,
    content_type: str,
    user_id: int,
    question: str,
    keep_records: bool = False,
    processing_settings_factory: Any | None = None,
    rag_settings_factory: Any | None = None,
    document_store_factory_builder: Any | None = None,
    service_factory: Any | None = None,
    cleanup_poll_interval_seconds: float | None = None,
    cleanup_timeout_seconds: float | None = None,
    monotonic: Any | None = None,
    sleep: Any | None = None,
) -> dict[str, Any]:
    """Run one full document RAG cycle and return the safe report payload."""
    processing_settings_factory = processing_settings_factory or create_processing_settings
    rag_settings_factory = rag_settings_factory or create_rag_settings
    document_store_factory_builder = (
        document_store_factory_builder or create_document_store_factory
    )
    service_factory = service_factory or create_document_rag_service
    if cleanup_poll_interval_seconds is None:
        cleanup_poll_interval_seconds = CLEANUP_POLL_INTERVAL_SECONDS
    if cleanup_timeout_seconds is None:
        cleanup_timeout_seconds = CLEANUP_TIMEOUT_SECONDS
    if monotonic is None:
        monotonic = time.monotonic
    if sleep is None:
        sleep = time.sleep

    request: DocumentConversionRequest | None = None
    normalized_question = question.strip()
    warnings: list[str] = []
    cleanup_attempted = False
    cleanup_succeeded = False
    cleanup_verification_poll_count = 0
    ingestion_duration_seconds: float | None = None
    question_duration_seconds: float | None = None
    ingestion_outcome: Any | None = None
    answer_result: Any | None = None
    inserted_document_ids: tuple[str, ...] = ()
    failure_message: str | None = None
    cleanup_failure_message: str | None = None

    try:
        request = _build_request(file_path=file_path, content_type=content_type, user_id=user_id)
        processing_settings = processing_settings_factory()
        rag_settings = rag_settings_factory()
        document_store_factory = document_store_factory_builder(rag_settings)
        service = service_factory(processing_settings, rag_settings, document_store_factory)

        ingestion_started = time.perf_counter()
        ingestion_outcome = service.ingest_and_summarize(request)
        ingestion_duration_seconds = round(time.perf_counter() - ingestion_started, 6)
        inserted_document_ids = tuple(ingestion_outcome.document_ids)

        question_started = time.perf_counter()
        answer_result = service.answer_question(user_id, normalized_question)
        question_duration_seconds = round(time.perf_counter() - question_started, 6)
    except KeyboardInterrupt:
        raise
    except (DocumentIngestionError, DocumentSummaryError, DocumentQuestionError) as exc:
        if exc.document_ids:
            inserted_document_ids = tuple(exc.document_ids)
        failure_message = str(exc)
    except (DocumentStoreError, ValidationError, SmokeScriptError) as exc:
        failure_message = _safe_controlled_failure_message(exc)
    finally:
        if "document_store_factory" in locals() and inserted_document_ids and not keep_records:
            cleanup_attempted = True
            try:
                document_store_factory.delete_documents(user_id, inserted_document_ids)
            except DocumentStoreError:
                cleanup_succeeded = False
                cleanup_failure_message = "Cleanup deletion failed."
                warnings.append("Cleanup deletion failed.")
            else:
                try:
                    cleanup_result = _poll_for_absent_document_ids(
                        document_store_factory=document_store_factory,
                        user_id=user_id,
                        document_ids=inserted_document_ids,
                        poll_interval_seconds=cleanup_poll_interval_seconds,
                        timeout_seconds=cleanup_timeout_seconds,
                        monotonic=monotonic,
                        sleep=sleep,
                    )
                except DocumentStoreError:
                    cleanup_succeeded = False
                    cleanup_failure_message = "Cleanup verification failed."
                    warnings.append("Cleanup verification failed.")
                else:
                    cleanup_succeeded = cleanup_result.succeeded
                    cleanup_verification_poll_count = cleanup_result.poll_count
                    if not cleanup_succeeded:
                        cleanup_failure_message = (
                            "Cleanup verification timed out before all inserted "
                            "document IDs disappeared."
                        )
                        warnings.append(
                            "Some inserted document IDs remained visible after "
                            "the cleanup verification timeout."
                        )
        elif keep_records and inserted_document_ids:
            warnings.append("Inserted document IDs were kept by request.")

    report = _build_report(
        status="success",
        file_name=request.file_name if request is not None else file_path.name,
        content_type=request.content_type if request is not None else content_type,
        question=normalized_question,
        ingestion_outcome=ingestion_outcome,
        answer_result=answer_result,
        ingestion_duration_seconds=ingestion_duration_seconds,
        question_duration_seconds=question_duration_seconds,
        cleanup_attempted=cleanup_attempted,
        cleanup_succeeded=cleanup_succeeded,
        cleanup_verification_poll_count=cleanup_verification_poll_count,
        inserted_document_ids=inserted_document_ids,
        warnings=warnings,
    )
    if failure_message is None and cleanup_failure_message is not None:
        failure_message = cleanup_failure_message
    if failure_message is not None:
        report["status"] = "failure"
        raise SmokeScriptControlledFailure(failure_message, report)
    return report


def _build_report(
    *,
    status: str,
    file_name: str,
    content_type: str,
    question: str,
    ingestion_outcome: Any | None,
    answer_result: Any | None,
    ingestion_duration_seconds: float | None,
    question_duration_seconds: float | None,
    cleanup_attempted: bool,
    cleanup_succeeded: bool,
    cleanup_verification_poll_count: int,
    inserted_document_ids: Sequence[str],
    warnings: Sequence[str],
) -> dict[str, Any]:
    summary = getattr(ingestion_outcome, "summary", None)
    sources = getattr(answer_result, "sources", ())
    return {
        "status": status,
        "file_name": file_name,
        "content_type": content_type,
        "file_hash": getattr(ingestion_outcome, "file_hash", None),
        "chunk_count": getattr(ingestion_outcome, "chunk_count", None),
        "documents_written": getattr(ingestion_outcome, "documents_written", None),
        "document_ids": list(inserted_document_ids),
        "summary": summary,
        "summary_character_count": len(summary) if isinstance(summary, str) else 0,
        "question": question,
        "answer": getattr(answer_result, "answer", None),
        "fallback_used": getattr(answer_result, "fallback_used", False),
        "sources": [source.model_dump(mode="json", exclude_none=True) for source in sources],
        "ingestion_duration_seconds": ingestion_duration_seconds,
        "question_duration_seconds": question_duration_seconds,
        "cleanup_attempted": cleanup_attempted,
        "cleanup_succeeded": cleanup_succeeded,
        "cleanup_verification_poll_count": cleanup_verification_poll_count,
        "warnings": list(warnings),
    }


def _poll_for_absent_document_ids(
    *,
    document_store_factory: Any,
    user_id: int,
    document_ids: Sequence[str],
    poll_interval_seconds: float,
    timeout_seconds: float,
    monotonic: Any,
    sleep: Any,
) -> CleanupVerificationResult:
    """Poll exact-ID visibility until all IDs disappear or the timeout expires."""
    if poll_interval_seconds <= 0:
        raise ValueError("cleanup poll interval must be positive")
    if timeout_seconds <= 0:
        raise ValueError("cleanup timeout must be positive")

    deadline = monotonic() + timeout_seconds
    poll_count = 0
    while True:
        remaining_ids = tuple(
            document_store_factory.fetch_existing_document_ids(user_id, tuple(document_ids))
        )
        poll_count += 1
        if not remaining_ids:
            return CleanupVerificationResult(
                succeeded=True,
                remaining_ids=(),
                poll_count=poll_count,
            )
        if monotonic() >= deadline:
            return CleanupVerificationResult(
                succeeded=False,
                remaining_ids=remaining_ids,
                poll_count=poll_count,
            )
        sleep(poll_interval_seconds)


def _safe_controlled_failure_message(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "Settings or input validation failed"
    return str(exc)


def _build_request(
    *,
    file_path: Path,
    content_type: str,
    user_id: int,
) -> DocumentConversionRequest:
    if not file_path.exists():
        raise SmokeScriptError("Input file does not exist")
    if not file_path.is_file():
        raise SmokeScriptError("Input path must point to a regular file")
    try:
        stat_result = file_path.stat()
    except OSError as exc:
        raise SmokeScriptError("Input file could not be read") from exc
    return DocumentConversionRequest(
        local_path=file_path,
        user_id=user_id,
        file_name=file_path.name,
        content_type=content_type,
        uploaded_at=datetime.fromtimestamp(stat_result.st_mtime, tz=UTC),
    )


def _parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def write_json_report(output_path: Path, report: Mapping[str, Any]) -> None:
    """Write the JSON report to *output_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        report = run_document_rag_smoke(
            file_path=args.file,
            content_type=args.content_type,
            user_id=args.user_id,
            question=args.question,
            keep_records=args.keep_records,
        )
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except SmokeScriptControlledFailure as exc:
        if args.json_output is not None:
            write_json_report(args.json_output, exc.report)
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        return EXIT_CONTROLLED_FAILURE
    except Exception:
        if args.debug:
            traceback.print_exc()
        else:
            print("Smoke test failed due to an unexpected internal error.", file=sys.stderr)
        return EXIT_UNEXPECTED_FAILURE

    print(
        "Smoke test passed for "
        f"{report['file_name']}: chunks={report['chunk_count']}, "
        f"written={report['documents_written']}, "
        f"sources={len(report['sources'])}"
    )
    if args.json_output is not None:
        write_json_report(args.json_output, report)
        print(f"JSON report written to {args.json_output.name}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
