"""Offline tests for hay_v2_bot.bot.messages."""

from __future__ import annotations

from typing import Any

from hay_v2_bot.bot import messages
from hay_v2_bot.models import DocumentSource


def _source(
    document_id: str,
    file_name: str,
    chunk_index: int,
    *,
    page_number: int | None = None,
    score: float | None = None,
) -> DocumentSource:
    payload: dict[str, Any] = {
        "document_id": document_id,
        "file_name": file_name,
        "chunk_index": chunk_index,
    }
    if page_number is not None:
        payload["page_number"] = page_number
    if score is not None:
        payload["score"] = score
    return DocumentSource(**payload)


def test_upload_started_message_is_exact() -> None:
    assert (
        messages.UPLOAD_STARTED_MESSAGE
        == "Файл получен. Запускаю анализ и сохранение. Это может занять немного времени…"
    )


def test_upload_completed_message_is_exact() -> None:
    assert (
        messages.UPLOAD_COMPLETED_MESSAGE
        == "Готово. Я изучил этот файл, теперь можем его обсудить."
    )


def test_format_sources_block_uses_pdf_page_number() -> None:
    source = _source("doc-1", "docuscope_smoke.pdf", 0, page_number=1, score=0.97)

    assert messages.format_sources_block((source,)) == "Источники:\n• docuscope_smoke.pdf, стр. 1"


def test_format_sources_block_uses_docx_chunk_number() -> None:
    source = _source("doc-1", "contract.docx", 2)

    assert messages.format_sources_block((source,)) == "Источники:\n• contract.docx, фрагмент 3"


def test_format_sources_block_keeps_strongest_pdf_page_and_omits_weak_page() -> None:
    sources = (
        _source("doc-1", "aurora.pdf", 0, page_number=4, score=0.92),
        _source("doc-2", "aurora.pdf", 1, page_number=8, score=0.70),
    )

    assert messages.format_sources_block(sources) == "Источники:\n• aurora.pdf, стр. 4"


def test_format_sources_block_sorts_descending_finite_scores() -> None:
    sources = (
        _source("doc-1", "contract.docx", 3, score=0.81),
        _source("doc-2", "aurora.pdf", 0, page_number=2, score=0.95),
        _source("doc-3", "appendix.pdf", 0, page_number=5, score=0.84),
    )

    assert messages.format_sources_block(sources) == (
        "Источники:\n• aurora.pdf, стр. 2\n• appendix.pdf, стр. 5"
    )


def test_format_sources_block_allows_two_similarly_strong_sources() -> None:
    sources = (
        _source("doc-1", "aurora.pdf", 0, page_number=2, score=0.91),
        _source("doc-2", "contract.docx", 1, score=0.84),
    )

    assert messages.format_sources_block(sources) == (
        "Источники:\n• aurora.pdf, стр. 2\n• contract.docx, фрагмент 2"
    )


def test_format_sources_block_limits_output_to_two_sources() -> None:
    sources = (
        _source("doc-1", "aurora.pdf", 0, page_number=2, score=0.95),
        _source("doc-2", "appendix.pdf", 0, page_number=5, score=0.91),
        _source("doc-3", "contract.docx", 1, score=0.89),
    )

    rendered = messages.format_sources_block(sources)

    assert rendered == "Источники:\n• aurora.pdf, стр. 2\n• appendix.pdf, стр. 5"
    assert rendered.count("\n• ") == 2


def test_format_sources_block_retains_best_source_below_absolute_threshold() -> None:
    sources = (
        _source("doc-1", "aurora.pdf", 0, page_number=2, score=0.24),
        _source("doc-2", "appendix.pdf", 0, page_number=5, score=0.23),
    )

    assert messages.format_sources_block(sources) == "Источники:\n• aurora.pdf, стр. 2"


def test_format_sources_block_missing_score_sources_preserve_order() -> None:
    sources = (
        _source("doc-1", "contract.docx", 2),
        _source("doc-2", "aurora.pdf", 0, page_number=3),
        _source("doc-3", "appendix.pdf", 1, page_number=7),
    )

    assert messages.format_sources_block(sources) == (
        "Источники:\n• contract.docx, фрагмент 3\n• aurora.pdf, стр. 3"
    )


def test_format_sources_block_deduplicates_pages_and_chunks() -> None:
    sources = (
        _source("doc-1", "aurora.pdf", 0, page_number=2, score=0.40),
        _source("doc-2", "contract.docx", 3, score=0.90),
        _source("doc-3", "aurora.pdf", 7, page_number=2, score=0.95),
        _source("doc-4", "contract.docx", 3, score=0.10),
    )

    assert messages.format_sources_block(sources) == (
        "Источники:\n• aurora.pdf, стр. 2\n• contract.docx, фрагмент 4"
    )


def test_format_sources_block_never_leaks_ids_scores_or_metadata() -> None:
    source = _source("namespace/user/doc-42", "contract.docx", 0, score=0.42)

    rendered = messages.format_sources_block((source,))

    assert "namespace/user/doc-42" not in rendered
    assert "0.42" not in rendered
    assert "metadata" not in rendered
    assert "contract.docx" in rendered


def test_format_sources_block_returns_empty_string_for_empty_sources() -> None:
    assert messages.format_sources_block(()) == ""
