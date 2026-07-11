"""Concise Russian Telegram messages and source formatting helpers."""

from __future__ import annotations

from collections.abc import Sequence

from hay_v2_bot.models import DocumentSource

START_MESSAGE = (
    "Привет! Я запоминаю то, что вы мне пишете, и использую эти воспоминания "
    "как контекст в разговоре.\n"
    "Команды: /help, /memory, /forget_me."
)

HELP_MESSAGE = (
    "Доступные команды:\n"
    "/start — приветствие\n"
    "/help — эта справка\n"
    "/memory — сколько у вас сохранено воспоминаний\n"
    "/forget_me — удалить всю вашу память\n\n"
    "Обычные текстовые сообщения: я отвечаю с учётом того, что вы рассказывали раньше.\n"
    "Также можно отправить PDF или DOCX: я сохраню файл, пришлю краткое "
    "резюме и отвечу на вопросы по документу."
)

MEMORY_COUNT_FAILURE_MESSAGE = "Не удалось получить количество воспоминаний. Попробуйте позже."
FORGET_ME_SUCCESS_MESSAGE = "Готово: вся ваша память удалена."
FORGET_ME_FAILURE_MESSAGE = "Не удалось удалить память. Попробуйте позже."
UNKNOWN_COMMAND_MESSAGE = "Неизвестная команда. Наберите /help, чтобы увидеть список команд."
CHAT_FAILURE_MESSAGE = "Извините, не получилось сформировать ответ. Попробуйте ещё раз."
NON_TEXT_MESSAGE = "Пока я понимаю только текстовые сообщения."

UPLOAD_STARTED_MESSAGE = (
    "Файл получен. Запускаю анализ и сохранение. Это может занять немного времени…"
)
UPLOAD_COMPLETED_MESSAGE = "Готово. Я изучил этот файл, теперь можем его обсудить."
UNSUPPORTED_DOCUMENT_MESSAGE = "Поддерживаются только документы PDF и DOCX."
PROCESSING_FAILURE_MESSAGE = "Не удалось обработать файл. Попробуйте ещё раз позже."

_SOURCE_BLOCK_TITLE = "Источники:"
_MAX_DISPLAYED_SOURCES = 2
_MIN_SOURCE_SCORE = 0.25
_SOURCE_SCORE_WINDOW = 0.15


def format_file_too_large_message(max_file_bytes: int) -> str:
    """Return a short Telegram-safe size-limit message."""
    if max_file_bytes < 1024 * 1024:
        limit_text = f"{max_file_bytes} Б"
    else:
        limit_text = f"{max_file_bytes / (1024 * 1024):.1f} МБ"
    return f"Файл слишком большой. Максимальный размер: {limit_text}."


def format_sources_block(sources: Sequence[DocumentSource]) -> str:
    """Render a short public source block without IDs, scores, or metadata leaks."""
    if isinstance(sources, str | bytes) or not isinstance(sources, Sequence):
        raise TypeError("sources must be a sequence of DocumentSource objects")
    if not sources:
        return ""

    validated_sources: list[DocumentSource] = []
    for source in sources:
        if not isinstance(source, DocumentSource):
            raise TypeError("sources must contain only DocumentSource objects")
        validated_sources.append(source)

    selected_sources = _select_sources_for_display(validated_sources)
    if not selected_sources:
        return ""

    lines = [_format_visible_reference(source) for source in selected_sources]
    return "\n".join((_SOURCE_BLOCK_TITLE, *lines))


def _select_sources_for_display(sources: Sequence[DocumentSource]) -> tuple[DocumentSource, ...]:
    scored_sources = [source for source in sources if source.score is not None]
    if scored_sources:
        sorted_sources = sorted(scored_sources, key=lambda source: source.score, reverse=True)
        best_score = sorted_sources[0].score
        assert best_score is not None
        minimum_score = max(_MIN_SOURCE_SCORE, best_score - _SOURCE_SCORE_WINDOW)
        retained_sources = [
            source
            for source in sorted_sources
            if source.score is not None and source.score >= minimum_score
        ]
        if not retained_sources:
            retained_sources = [sorted_sources[0]]
        return _deduplicate_visible_sources(retained_sources, limit=_MAX_DISPLAYED_SOURCES)
    return _deduplicate_visible_sources(sources, limit=_MAX_DISPLAYED_SOURCES)


def _deduplicate_visible_sources(
    sources: Sequence[DocumentSource],
    *,
    limit: int,
) -> tuple[DocumentSource, ...]:
    unique_sources: list[DocumentSource] = []
    seen: set[tuple[str, str, int]] = set()
    for source in sources:
        identity = _visible_reference_identity(source)
        if identity in seen:
            continue
        seen.add(identity)
        unique_sources.append(source)
        if len(unique_sources) >= limit:
            break
    return tuple(unique_sources)


def _visible_reference_identity(source: DocumentSource) -> tuple[str, str, int]:
    if source.page_number is not None:
        return (source.file_name, "page", source.page_number)
    return (source.file_name, "chunk", source.chunk_index)


def _format_visible_reference(source: DocumentSource) -> str:
    if source.page_number is not None:
        return f"• {source.file_name}, стр. {source.page_number}"
    return f"• {source.file_name}, фрагмент {source.chunk_index + 1}"


__all__ = [
    "CHAT_FAILURE_MESSAGE",
    "FORGET_ME_FAILURE_MESSAGE",
    "FORGET_ME_SUCCESS_MESSAGE",
    "HELP_MESSAGE",
    "MEMORY_COUNT_FAILURE_MESSAGE",
    "NON_TEXT_MESSAGE",
    "PROCESSING_FAILURE_MESSAGE",
    "START_MESSAGE",
    "UNKNOWN_COMMAND_MESSAGE",
    "UNSUPPORTED_DOCUMENT_MESSAGE",
    "UPLOAD_COMPLETED_MESSAGE",
    "UPLOAD_STARTED_MESSAGE",
    "format_file_too_large_message",
    "format_sources_block",
]
