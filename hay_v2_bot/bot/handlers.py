"""Aiogram handlers for the v2 Telegram bot."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

from aiogram import Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from pydantic import ValidationError

from hay_v2_bot.bot.messages import (
    CHAT_FAILURE_MESSAGE,
    FORGET_ME_FAILURE_MESSAGE,
    FORGET_ME_SUCCESS_MESSAGE,
    HELP_MESSAGE,
    MEMORY_COUNT_FAILURE_MESSAGE,
    NON_TEXT_MESSAGE,
    PROCESSING_FAILURE_MESSAGE,
    START_MESSAGE,
    UNKNOWN_COMMAND_MESSAGE,
    UNSUPPORTED_DOCUMENT_MESSAGE,
    UPLOAD_COMPLETED_MESSAGE,
    UPLOAD_STARTED_MESSAGE,
    format_file_too_large_message,
    format_sources_block,
)
from hay_v2_bot.config import DocumentProcessingSettings
from hay_v2_bot.models import (
    DOCX_CONTENT_TYPE,
    PDF_CONTENT_TYPE,
    DocumentAnswer,
    DocumentConversionRequest,
)
from hay_v2_bot.models.documents import validate_base_file_name
from hay_v2_bot.services import (
    DocumentIngestionError,
    DocumentQuestionError,
    DocumentRagService,
    DocumentRagServiceError,
    DocumentSummaryError,
)
from telegram_vector_memory_bot.bot import split_telegram_text
from telegram_vector_memory_bot.haystack_agent import (
    HaystackAgentService,
    HaystackAgentServiceError,
)
from telegram_vector_memory_bot.memory_service import MemoryService, MemoryServiceError
from telegram_vector_memory_bot.pinecone_manager import VectorMemoryError

logger = logging.getLogger(__name__)

_COMMAND_NAME_PATTERN: Final = re.compile(r"^[A-Za-z0-9_]{1,32}$")
_PDF_SUFFIX: Final = ".pdf"
_DOCX_SUFFIX: Final = ".docx"
_SUPPORTED_CONTENT_TYPES: Final = {
    _PDF_SUFFIX: PDF_CONTENT_TYPE,
    _DOCX_SUFFIX: DOCX_CONTENT_TYPE,
}


def _pluralize_records_ru(count: int) -> str:
    remainder_100 = count % 100
    remainder_10 = count % 10
    if 11 <= remainder_100 <= 14:
        return "записей"
    if remainder_10 == 1:
        return "запись"
    if 2 <= remainder_10 <= 4:
        return "записи"
    return "записей"


def _is_command_text(text: str | None) -> bool:
    return isinstance(text, str) and text.startswith("/")


def is_ordinary_text_message(message: Message) -> bool:
    """True for plain text that is not a slash-command."""
    return isinstance(message.text, str) and not _is_command_text(message.text)


def is_non_text_message(message: Message) -> bool:
    """True only for non-text, non-document messages."""
    return message.text is None and getattr(message, "document", None) is None


async def cmd_start(message: Message) -> None:
    await _send_text_chunks(message, START_MESSAGE)


async def cmd_help(message: Message) -> None:
    await _send_text_chunks(message, HELP_MESSAGE)


async def cmd_memory(message: Message, memory_service: MemoryService) -> None:
    from_user = message.from_user
    if from_user is None:
        return

    try:
        count = await asyncio.to_thread(memory_service.get_memory_count, user_id=from_user.id)
    except VectorMemoryError as exc:
        _log_safe_warning("memory_count_failed", exc)
        await _send_text_chunks(message, MEMORY_COUNT_FAILURE_MESSAGE)
        return

    await _send_text_chunks(message, f"В памяти сохранено: {count} {_pluralize_records_ru(count)}.")


async def cmd_forget_me(message: Message, memory_service: MemoryService) -> None:
    from_user = message.from_user
    if from_user is None:
        return

    try:
        await asyncio.to_thread(memory_service.forget_user, user_id=from_user.id)
    except VectorMemoryError as exc:
        _log_safe_warning("forget_me_failed", exc)
        await _send_text_chunks(message, FORGET_ME_FAILURE_MESSAGE)
        return

    await _send_text_chunks(message, FORGET_ME_SUCCESS_MESSAGE)


async def handle_unknown_command(message: Message) -> None:
    await _send_text_chunks(message, UNKNOWN_COMMAND_MESSAGE)


async def ignore_foreign_command(message: Message) -> None:
    return None


async def ignore_malformed_command(message: Message) -> None:
    return None


async def handle_document_upload(
    message: Message,
    document_rag_service: DocumentRagService,
    processing_settings: DocumentProcessingSettings,
) -> None:
    document = getattr(message, "document", None)
    if document is None:
        return

    from_user = message.from_user
    if from_user is None or from_user.id <= 0:
        await _send_text_chunks(message, PROCESSING_FAILURE_MESSAGE)
        return

    try:
        file_name, content_type = _resolve_supported_document(
            document=document,
            max_file_bytes=processing_settings.max_file_bytes,
        )
    except _UploadRejected as exc:
        await _send_text_chunks(message, exc.reply_text)
        return

    await _send_text_chunks(message, UPLOAD_STARTED_MESSAGE)

    try:
        with TemporaryDirectory() as temp_dir:
            download_path = Path(temp_dir) / file_name
            await message.bot.download(document, destination=download_path)
            request = DocumentConversionRequest(
                local_path=download_path,
                user_id=from_user.id,
                file_name=file_name,
                content_type=content_type,
                uploaded_at=datetime.now(UTC),
            )
            outcome = await asyncio.to_thread(document_rag_service.ingest_and_summarize, request)
    except _KNOWN_UPLOAD_FAILURES as exc:
        _log_safe_warning("document_upload_failed", exc)
        await _send_text_chunks(message, PROCESSING_FAILURE_MESSAGE)
        return

    await _send_text_chunks(message, UPLOAD_COMPLETED_MESSAGE)
    await _send_text_chunks(message, outcome.summary)


async def handle_non_text_message(message: Message) -> None:
    await _send_text_chunks(message, NON_TEXT_MESSAGE)


async def handle_text_message(
    message: Message,
    memory_service: MemoryService,
    reply_service: HaystackAgentService,
    document_rag_service: DocumentRagService,
) -> None:
    user_text = message.text
    if not isinstance(user_text, str) or not user_text.strip():
        return

    from_user = message.from_user
    if from_user is None:
        return

    user_id = from_user.id
    username = from_user.username
    first_name = from_user.first_name
    last_name = from_user.last_name

    try:
        memories = await asyncio.to_thread(
            memory_service.recall,
            user_id=user_id,
            query=user_text,
        )
    except (VectorMemoryError, MemoryServiceError) as exc:
        _log_safe_warning("recall_failed", exc)
        memories = []

    document_answer: DocumentAnswer | None = None
    try:
        document_answer = await asyncio.to_thread(
            document_rag_service.answer_question,
            user_id,
            user_text,
        )
    except DocumentQuestionError as exc:
        _log_safe_warning("document_question_failed", exc)

    if _is_grounded_document_answer(document_answer):
        assert document_answer is not None
        await _send_text_chunks(message, document_answer.answer)
        sources_block = format_sources_block(document_answer.sources)
        if sources_block:
            await _send_text_chunks(message, sources_block)
        await _remember_user_message(
            memory_service=memory_service,
            user_id=user_id,
            user_text=user_text,
            username=username,
            first_name=first_name,
            last_name=last_name,
        )
        return

    try:
        reply = await reply_service.generate_reply(user_text=user_text, memories=memories)
    except HaystackAgentServiceError as exc:
        _log_safe_warning("chat_generation_failed", exc)
        await _send_text_chunks(message, CHAT_FAILURE_MESSAGE)
        return

    await _send_text_chunks(message, reply)
    await _remember_user_message(
        memory_service=memory_service,
        user_id=user_id,
        user_text=user_text,
        username=username,
        first_name=first_name,
        last_name=last_name,
    )


def create_dispatcher(
    *,
    memory_service: MemoryService,
    reply_service: HaystackAgentService,
    document_rag_service: DocumentRagService,
    processing_settings: DocumentProcessingSettings,
) -> Dispatcher:
    """Build a Dispatcher with v1 and v2 services exposed as workflow data."""
    dispatcher = Dispatcher(
        memory_service=memory_service,
        reply_service=reply_service,
        document_rag_service=document_rag_service,
        processing_settings=processing_settings,
    )
    register_handlers(dispatcher)
    return dispatcher


def register_handlers(dispatcher: Dispatcher) -> Dispatcher:
    """Register the v2 router once on the provided Dispatcher."""
    if getattr(dispatcher, "_hay_v2_handlers_registered", False):
        return dispatcher
    dispatcher.include_router(_build_router())
    dispatcher._hay_v2_handlers_registered = True
    return dispatcher


def _build_router() -> Router:
    router = Router(name="hay_v2_bot_router")
    router.message.register(cmd_start, CommandStart())
    router.message.register(cmd_help, Command("help"))
    router.message.register(cmd_memory, Command("memory"))
    router.message.register(cmd_forget_me, Command("forget_me"))
    router.message.register(handle_document_upload, F.document)
    router.message.register(handle_unknown_command, Command(_COMMAND_NAME_PATTERN))
    router.message.register(
        ignore_foreign_command,
        Command(_COMMAND_NAME_PATTERN, ignore_mention=True),
    )
    router.message.register(ignore_malformed_command, F.text.startswith("/"))
    router.message.register(handle_text_message, is_ordinary_text_message)
    router.message.register(handle_non_text_message, is_non_text_message)
    return router


async def _send_text_chunks(message: Message, text: str) -> None:
    chunks = split_telegram_text(text)
    try:
        for chunk in chunks:
            await message.answer(chunk)
    except Exception as exc:
        _log_safe_warning("send_failed", exc)
        raise


async def _remember_user_message(
    *,
    memory_service: MemoryService,
    user_id: int,
    user_text: str,
    username: str | None,
    first_name: str | None,
    last_name: str | None,
) -> None:
    try:
        await asyncio.to_thread(
            memory_service.remember,
            user_id=user_id,
            text=user_text,
            username=username,
            first_name=first_name,
            last_name=last_name,
        )
    except (VectorMemoryError, MemoryServiceError) as exc:
        _log_safe_warning("remember_failed", exc)


def _is_grounded_document_answer(answer: DocumentAnswer | None) -> bool:
    return bool(answer is not None and answer.sources and answer.fallback_used is False)


def _extract_safe_base_file_name(raw_name: object) -> str | None:
    if not isinstance(raw_name, str):
        return None
    stripped = raw_name.strip()
    if not stripped:
        return None
    base_name = stripped.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    if not base_name:
        return None
    try:
        return validate_base_file_name(base_name)
    except ValueError:
        return None


def _resolve_supported_document(*, document: object, max_file_bytes: int) -> tuple[str, str]:
    file_name = _extract_safe_base_file_name(getattr(document, "file_name", None))
    mime_type = getattr(document, "mime_type", None)
    file_size = getattr(document, "file_size", None)

    if (
        isinstance(file_size, int)
        and not isinstance(file_size, bool)
        and file_size > max_file_bytes
    ):
        raise _UploadRejected(format_file_too_large_message(max_file_bytes))

    if file_name is None:
        raise _UploadRejected(UNSUPPORTED_DOCUMENT_MESSAGE)

    suffix = Path(file_name).suffix.lower()
    expected_content_type = _SUPPORTED_CONTENT_TYPES.get(suffix)
    if expected_content_type is None:
        raise _UploadRejected(UNSUPPORTED_DOCUMENT_MESSAGE)

    normalized_mime_type = (
        mime_type.strip() if isinstance(mime_type, str) and mime_type.strip() else None
    )
    if normalized_mime_type is None:
        return file_name, expected_content_type
    if normalized_mime_type != expected_content_type:
        raise _UploadRejected(UNSUPPORTED_DOCUMENT_MESSAGE)
    return file_name, expected_content_type


def _log_safe_warning(event: str, exc: BaseException) -> None:
    logger.warning("event=%s error_type=%s", event, type(exc).__name__)


class _UploadRejected(Exception):
    def __init__(self, reply_text: str) -> None:
        super().__init__(reply_text)
        self.reply_text = reply_text


_KNOWN_UPLOAD_FAILURES = (
    DocumentIngestionError,
    DocumentSummaryError,
    DocumentRagServiceError,
    OSError,
    ValidationError,
)


__all__ = [
    "cmd_forget_me",
    "cmd_help",
    "cmd_memory",
    "cmd_start",
    "create_dispatcher",
    "handle_document_upload",
    "handle_non_text_message",
    "handle_text_message",
    "handle_unknown_command",
    "ignore_foreign_command",
    "ignore_malformed_command",
    "is_non_text_message",
    "is_ordinary_text_message",
    "register_handlers",
]
