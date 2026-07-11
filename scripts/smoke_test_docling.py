"""Run one live Docling conversion through the production adapter safely.

This script performs exactly one real Docling conversion for one local PDF or
DOCX file, using the existing ``DoclingDocumentAdapter`` for the normalized
output. It captures only the raw Haystack ``Document`` objects returned by the
real converter so the script can inspect the metadata shape without exposing
full raw contents, local absolute paths, or arbitrary object reprs.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import traceback
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from haystack import Document
from haystack_integrations.components.converters.docling import (
    DoclingConverter,
    ExportType,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hay_v2_bot.adapters import DoclingDocumentAdapter, DocumentAdapterError  # noqa: E402
from hay_v2_bot.config import DocumentProcessingSettings  # noqa: E402
from hay_v2_bot.models import (
    SUPPORTED_DOCUMENT_CONTENT_TYPES,
    DocumentConversionRequest,
    DocumentConversionResult,
)  # noqa: E402

DEFAULT_PREVIEW_CHARS = 160
EXIT_CONTROLLED_FAILURE = 3
EXIT_UNEXPECTED_FAILURE = 4
EXIT_INTERRUPTED = 130
MAX_METADATA_DEPTH = 4
SENSITIVE_KEY_PARTS = ("path", "uri", "url", "token", "secret", "key", "credential")
CACHE_DIRECTORIES = (
    Path.home() / ".cache" / "huggingface",
    Path.home() / ".cache" / "docling",
    Path.home() / ".cache" / "transformers",
    Path.home() / "AppData" / "Local" / "huggingface",
    Path.home() / "AppData" / "Local" / "docling",
    Path.home() / "AppData" / "Local" / "transformers",
)


class SmokeScriptError(Exception):
    """Raised for controlled smoke-test failures with safe public messages."""


@dataclass(frozen=True)
class CacheDirectorySnapshot:
    """Minimal cache-directory state captured before and after conversion."""

    exists: bool
    mtime_ns: int | None
    child_names: tuple[str, ...] | None


@dataclass(frozen=True)
class RawMetadataInspection:
    """Safe summary of raw Docling metadata shape."""

    top_level_keys: tuple[str, ...]
    key_paths: tuple[str, ...]
    direct_page_number_count: int
    direct_headings_count: int
    redacted_key_count: int
    skipped_unknown_value_count: int


@dataclass
class _RawMetadataAccumulator:
    top_level_keys: set[str]
    key_paths: set[str]
    direct_page_number_count: int = 0
    direct_headings_count: int = 0
    redacted_key_count: int = 0
    skipped_unknown_value_count: int = 0


class CapturingDoclingConverterWrapper:
    """Delegate to the real Docling converter once and retain raw documents."""

    def __init__(self, converter: Any | None = None) -> None:
        self._converter = (
            converter
            if converter is not None
            else DoclingConverter(export_type=ExportType.DOC_CHUNKS)
        )
        self.raw_documents: tuple[Document, ...] = ()
        self.run_call_count = 0

    def run(
        self,
        *,
        paths: list[str | Path] | None = None,
        sources: Any = None,
        meta: Any = None,
    ) -> Mapping[str, object]:
        self.run_call_count += 1
        result = self._converter.run(paths=paths, sources=sources, meta=meta)
        self.raw_documents = _capture_raw_documents(result)
        return result


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for this smoke-test script."""
    parser = argparse.ArgumentParser(
        description=(
            "Run one live Docling conversion through the production adapter and "
            "emit a safe JSON-compatible report."
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
        help="Positive synthetic user ID used for document normalization metadata.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional output path for the JSON report.",
    )
    parser.add_argument(
        "--preview-chars",
        type=_parse_positive_int,
        default=DEFAULT_PREVIEW_CHARS,
        help=f"Maximum characters to keep in chunk previews (default: {DEFAULT_PREVIEW_CHARS}).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print a traceback for unexpected internal failures.",
    )
    return parser


def create_settings() -> DocumentProcessingSettings:
    """Construct default document settings without loading any .env file."""
    return DocumentProcessingSettings(_env_file=None)


def run_smoke_conversion(
    *,
    file_path: Path,
    content_type: str,
    user_id: int,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
    settings_factory: Any | None = None,
    wrapper_factory: Any | None = None,
    adapter_class: Any | None = None,
) -> dict[str, Any]:
    """Run one conversion and return the safe report payload."""
    if settings_factory is None:
        settings_factory = create_settings
    if wrapper_factory is None:
        wrapper_factory = CapturingDoclingConverterWrapper
    if adapter_class is None:
        adapter_class = DoclingDocumentAdapter
    settings = settings_factory()
    request, file_size_bytes = _build_request(
        file_path=file_path,
        content_type=content_type,
        user_id=user_id,
    )
    cache_before = capture_cache_activity_snapshot()
    captured_converter_output = StringIO()
    with (
        contextlib.redirect_stdout(captured_converter_output),
        contextlib.redirect_stderr(captured_converter_output),
        warnings.catch_warnings(),
    ):
        warnings.filterwarnings(
            "ignore",
            message=r"The 'paths' parameter is deprecated\. Use 'sources' instead\.",
            category=DeprecationWarning,
        )
        wrapper = wrapper_factory()
        adapter = adapter_class(settings=settings, converter=wrapper)
        started_at = time.perf_counter()
        result = adapter.convert(request)
        duration_seconds = round(time.perf_counter() - started_at, 6)
    cache_after = capture_cache_activity_snapshot()

    if getattr(wrapper, "run_call_count", None) != 1:
        raise SmokeScriptError("Converter delegation count was not exactly one")

    raw_documents = tuple(getattr(wrapper, "raw_documents", ()))
    report = build_safe_report(
        request=request,
        file_size_bytes=file_size_bytes,
        raw_documents=raw_documents,
        normalized_result=result,
        preview_chars=preview_chars,
        duration_seconds=duration_seconds,
        cache_activity_observed=combine_cache_activity_observations(
            snapshot_observation=detect_cache_activity(cache_before, cache_after),
            converter_output=captured_converter_output.getvalue(),
        ),
    )
    return report


def build_safe_report(
    *,
    request: DocumentConversionRequest,
    file_size_bytes: int,
    raw_documents: Sequence[Document],
    normalized_result: DocumentConversionResult,
    preview_chars: int,
    duration_seconds: float,
    cache_activity_observed: bool | None,
) -> dict[str, Any]:
    """Build the JSON-safe smoke report."""
    inspection = inspect_raw_metadata(raw_documents)
    warnings = build_report_warnings(
        inspection=inspection,
        cache_activity_observed=cache_activity_observed,
    )
    first_document = normalized_result.documents[0]
    last_document = normalized_result.documents[-1]
    return {
        "status": "success",
        "file_name": request.file_name,
        "content_type": request.content_type,
        "file_size_bytes": file_size_bytes,
        "file_hash": normalized_result.file_hash,
        "raw_chunk_count": len(raw_documents),
        "normalized_chunk_count": normalized_result.chunk_count,
        "normalized_document_ids": [document.id for document in normalized_result.documents],
        "first_normalized_chunk_preview": make_preview(first_document.content or "", preview_chars),
        "last_normalized_chunk_preview": make_preview(last_document.content or "", preview_chars),
        "normalized_metadata_sample": safe_normalized_metadata_sample(first_document),
        "raw_metadata_top_level_keys": list(inspection.top_level_keys),
        "raw_metadata_key_paths": list(inspection.key_paths),
        "raw_chunks_with_direct_page_number_count": inspection.direct_page_number_count,
        "raw_chunks_with_direct_headings_count": inspection.direct_headings_count,
        "normalized_chunks_with_page_number_count": count_normalized_metadata_key(
            normalized_result.documents,
            "page_number",
        ),
        "normalized_chunks_with_headings_count": count_normalized_metadata_key(
            normalized_result.documents,
            "headings",
        ),
        "duration_seconds": duration_seconds,
        "model_or_tokenizer_cache_activity_observed": cache_activity_observed,
        "warnings": warnings,
    }


def build_report_warnings(
    *,
    inspection: RawMetadataInspection,
    cache_activity_observed: bool | None,
) -> list[str]:
    """Build deterministic warning strings for the report."""
    warnings: list[str] = []
    if inspection.redacted_key_count > 0:
        warnings.append(
            "Omitted raw metadata keys whose names matched the sensitive-key safety policy."
        )
    if inspection.skipped_unknown_value_count > 0:
        warnings.append("Skipped raw metadata values with unsupported object types.")
    if cache_activity_observed is None:
        warnings.append("Model/tokenizer cache activity could not be determined safely.")
    return warnings


def capture_cache_activity_snapshot() -> dict[str, CacheDirectorySnapshot]:
    """Capture a small, safe snapshot of likely model/tokenizer cache directories."""
    snapshots: dict[str, CacheDirectorySnapshot] = {}
    for directory in CACHE_DIRECTORIES:
        try:
            if directory.exists():
                snapshots[str(directory)] = CacheDirectorySnapshot(
                    exists=True,
                    mtime_ns=directory.stat().st_mtime_ns,
                    child_names=tuple(sorted(child.name for child in directory.iterdir())),
                )
            else:
                snapshots[str(directory)] = CacheDirectorySnapshot(
                    exists=False,
                    mtime_ns=None,
                    child_names=None,
                )
        except OSError:
            continue
    return snapshots


def detect_cache_activity(
    before: Mapping[str, CacheDirectorySnapshot],
    after: Mapping[str, CacheDirectorySnapshot],
) -> bool | None:
    """Return whether likely cache initialization activity was observed."""
    common_keys = sorted(set(before) & set(after))
    if not common_keys:
        return None
    for key in common_keys:
        previous = before[key]
        current = after[key]
        if previous.exists != current.exists:
            return True
        if previous.mtime_ns != current.mtime_ns:
            return True
        if previous.child_names != current.child_names:
            return True
    return False


def combine_cache_activity_observations(
    *,
    snapshot_observation: bool | None,
    converter_output: str,
) -> bool | None:
    """Combine filesystem and captured-converter signals into one safe observation."""
    output_observation = observe_cache_activity_from_output(converter_output)
    if snapshot_observation is True or output_observation is True:
        return True
    if snapshot_observation is False or output_observation is False:
        return False
    return None


def observe_cache_activity_from_output(output: str) -> bool | None:
    """Infer visible cache/model activity from captured converter output when possible."""
    lowered = output.lower()
    if not lowered.strip():
        return None
    activity_markers = (
        "initiating download",
        "download size",
        "successfully saved to",
        "loading weights",
        "using engine_name",
    )
    if any(marker in lowered for marker in activity_markers):
        return True
    return None


def inspect_raw_metadata(documents: Sequence[Document]) -> RawMetadataInspection:
    """Inspect raw Docling metadata shape without serializing raw values."""
    state = _RawMetadataAccumulator(top_level_keys=set(), key_paths=set())
    for document in documents:
        meta = document.meta
        if not isinstance(meta, Mapping):
            continue
        if "page_number" in meta:
            state.direct_page_number_count += 1
        if "headings" in meta:
            state.direct_headings_count += 1
        walk_raw_mapping(meta, path=(), depth=0, state=state, is_top_level=True)
    return RawMetadataInspection(
        top_level_keys=tuple(sorted(state.top_level_keys)),
        key_paths=tuple(sorted(state.key_paths)),
        direct_page_number_count=state.direct_page_number_count,
        direct_headings_count=state.direct_headings_count,
        redacted_key_count=state.redacted_key_count,
        skipped_unknown_value_count=state.skipped_unknown_value_count,
    )


def walk_raw_mapping(
    value: Mapping[Any, Any],
    *,
    path: tuple[str, ...],
    depth: int,
    state: _RawMetadataAccumulator,
    is_top_level: bool,
) -> None:
    """Walk a raw metadata mapping safely."""
    for key, child_value in value.items():
        if not isinstance(key, str):
            state.skipped_unknown_value_count += 1
            continue
        if is_sensitive_key_name(key):
            state.redacted_key_count += 1
            continue
        current_path = path + (key,)
        state.key_paths.add(join_metadata_path(current_path))
        if is_top_level:
            state.top_level_keys.add(key)
        if depth + 1 >= MAX_METADATA_DEPTH:
            continue
        if isinstance(child_value, Mapping):
            walk_raw_mapping(
                child_value,
                path=current_path,
                depth=depth + 1,
                state=state,
                is_top_level=False,
            )
            continue
        if is_simple_sequence(child_value):
            walk_raw_sequence(
                child_value,
                path=current_path + ("[]",),
                depth=depth + 1,
                state=state,
            )
            continue
        if is_simple_scalar(child_value):
            continue
        state.skipped_unknown_value_count += 1


def walk_raw_sequence(
    value: Sequence[Any],
    *,
    path: tuple[str, ...],
    depth: int,
    state: _RawMetadataAccumulator,
) -> None:
    """Walk a raw metadata sequence safely."""
    state.key_paths.add(join_metadata_path(path))
    if depth >= MAX_METADATA_DEPTH:
        return
    for item in value:
        if isinstance(item, Mapping):
            walk_raw_mapping(
                item,
                path=path,
                depth=depth,
                state=state,
                is_top_level=False,
            )
            continue
        if is_simple_sequence(item):
            walk_raw_sequence(
                item,
                path=path + ("[]",),
                depth=depth + 1,
                state=state,
            )
            continue
        if is_simple_scalar(item):
            continue
        state.skipped_unknown_value_count += 1


def join_metadata_path(parts: Sequence[str]) -> str:
    """Join metadata path parts while keeping sequence markers compact."""
    joined: list[str] = []
    for part in parts:
        if part == "[]":
            if joined:
                joined[-1] = joined[-1] + "[]"
            else:
                joined.append("[]")
        else:
            joined.append(part)
    return ".".join(joined)


def is_sensitive_key_name(key_name: str) -> bool:
    """Return whether a metadata key name should be omitted for safety."""
    lowered = key_name.lower()
    return any(fragment in lowered for fragment in SENSITIVE_KEY_PARTS)


def is_simple_scalar(value: Any) -> bool:
    """Return whether *value* is a simple JSON-safe scalar."""
    return value is None or isinstance(value, str | int | float | bool)


def is_simple_sequence(value: Any) -> bool:
    """Return whether *value* is a simple inspectable sequence."""
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def count_normalized_metadata_key(documents: Sequence[Document], key: str) -> int:
    """Count normalized chunks whose safe metadata contains *key*."""
    count = 0
    for document in documents:
        if isinstance(document.meta, Mapping) and key in document.meta:
            count += 1
    return count


def safe_normalized_metadata_sample(document: Document) -> dict[str, Any]:
    """Return a deterministic JSON-safe sample from the first normalized chunk."""
    if not isinstance(document.meta, Mapping):
        return {}
    return {key: document.meta[key] for key in sorted(document.meta)}


def make_preview(text: str, preview_chars: int) -> str:
    """Normalize line breaks and cap the preview length safely."""
    normalized = " ".join(part.strip() for part in text.splitlines())
    normalized = " ".join(normalized.split())
    return normalized[:preview_chars]


def _build_request(
    *,
    file_path: Path,
    content_type: str,
    user_id: int,
) -> tuple[DocumentConversionRequest, int]:
    """Build the production conversion request and return it with the file size."""
    if not file_path.exists():
        raise SmokeScriptError("Input file does not exist")
    if not file_path.is_file():
        raise SmokeScriptError("Input path must point to a regular file")
    try:
        stat_result = file_path.stat()
    except OSError as exc:
        raise SmokeScriptError("Input file could not be read") from exc
    request = DocumentConversionRequest(
        local_path=file_path,
        user_id=user_id,
        file_name=file_path.name,
        content_type=content_type,
        uploaded_at=datetime.fromtimestamp(stat_result.st_mtime, tz=UTC),
    )
    return request, stat_result.st_size


def _capture_raw_documents(result: object) -> tuple[Document, ...]:
    """Retain only raw Haystack documents from the converter result."""
    if not isinstance(result, Mapping):
        return ()
    raw_documents = result.get("documents")
    if not is_simple_sequence(raw_documents):
        return ()
    return tuple(document for document in raw_documents if isinstance(document, Document))


def _parse_positive_int(value: str) -> int:
    """Argparse type enforcing a positive integer."""
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
        report = run_smoke_conversion(
            file_path=args.file,
            content_type=args.content_type,
            user_id=args.user_id,
            preview_chars=args.preview_chars,
        )
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except (SmokeScriptError, DocumentAdapterError) as exc:
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
        f"{report['file_name']}: raw_chunks={report['raw_chunk_count']}, "
        f"normalized_chunks={report['normalized_chunk_count']}, "
        f"duration={report['duration_seconds']:.3f}s"
    )
    if args.json_output is not None:
        write_json_report(args.json_output, report)
        print(f"JSON report written to {args.json_output.name}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
