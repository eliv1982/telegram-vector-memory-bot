"""aiogram 3 Telegram application layer.

Wires the four Stage 5 commands and the ordinary text-message flow on top of
the existing, synchronous ``MemoryService`` and ``ChatService``. This module
constructs no external client and no ``Settings`` object at import time --
everything is built inside :func:`run_bot`, which only runs when the module
is executed as a script.

Message flow for ordinary text: recall -> chat generation -> send every
reply chunk -> remember, in that order. ``remember`` only ever runs after
every reply chunk has been sent successfully; a failed send re-raises so
aiogram's own error boundary handles it, and never triggers ``remember``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Final

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from .chat_service import ChatService, ChatServiceError
from .config import get_settings
from .memory_service import MemoryService, MemoryServiceError
from .pinecone_manager import PineconeManager, VectorMemoryError

logger = logging.getLogger(__name__)

_TELEGRAM_MESSAGE_LIMIT: Final = 4096

# Fixed, safe user-facing responses. Private by design -- tests assert the
# literal text independently rather than importing these constants, so a
# change here can't silently drag a stale test expectation along with it.
_START_MESSAGE: Final = (
    "Привет! Я запоминаю то, что вы мне пишете, и использую эти воспоминания "
    "как контекст в разговоре.\n"
    "Команды: /help, /memory, /forget_me."
)
_HELP_MESSAGE: Final = (
    "Доступные команды:\n"
    "/start — приветствие\n"
    "/help — эта справка\n"
    "/memory — сколько у вас сохранено воспоминаний\n"
    "/forget_me — удалить всю вашу память\n\n"
    "Обычные текстовые сообщения: я отвечаю с учётом того, что вы рассказывали раньше."
)
_MEMORY_COUNT_FAILURE_REPLY: Final = (
    "Не удалось получить количество воспоминаний. Попробуйте позже."
)
_FORGET_ME_SUCCESS_REPLY: Final = "Готово: вся ваша память удалена."
_FORGET_ME_FAILURE_REPLY: Final = "Не удалось удалить память. Попробуйте позже."
_UNKNOWN_COMMAND_REPLY: Final = "Неизвестная команда. Наберите /help, чтобы увидеть список команд."
_CHAT_FAILURE_REPLY: Final = "Извините, не получилось сформировать ответ. Попробуйте ещё раз."
_NON_TEXT_REPLY: Final = "Пока я понимаю только текстовые сообщения."

# Matches any syntactically valid Telegram bot command name (letters, digits,
# underscore, 1-32 characters) -- used as a catch-all so the unknown-command
# route relies on aiogram's own Command filter (and its bot-mention
# validation) instead of a bespoke slash-prefix/mention parser.
_COMMAND_NAME_PATTERN: Final = re.compile(r"^[A-Za-z0-9_]{1,32}$")


def _pluralize_records_ru(count: int) -> str:
    """Russian plural form of "records" for a non-negative *count*.

    A small, fixed grammatical rule for one word -- not a general
    localization layer.
    """
    remainder_100 = count % 100
    remainder_10 = count % 10
    if 11 <= remainder_100 <= 14:
        return "записей"
    if remainder_10 == 1:
        return "запись"
    if 2 <= remainder_10 <= 4:
        return "записи"
    return "записей"


def split_telegram_text(text: str, *, limit: int = _TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Split *text* into Telegram-safe chunks measured in UTF-16 code units.

    A deterministic hard split -- it does not try to break on word or
    paragraph boundaries. Concatenating the returned chunks always
    reproduces *text* exactly, preserves order, and never produces an empty
    chunk. Every emitted chunk stays within *limit* UTF-16 code units: if a
    single Unicode code point (e.g. certain emoji, which take 2 UTF-16 code
    units) would itself exceed *limit*, that is a request that can never be
    satisfied without splitting the code point into an invalid surrogate
    half, so this raises ``ValueError`` instead of emitting an over-limit or
    empty chunk.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty or whitespace-only")
    if limit <= 0:
        raise ValueError("limit must be positive")

    chunks: list[str] = []
    current_chars: list[str] = []
    current_units = 0

    for char in text:
        char_units = 2 if ord(char) > 0xFFFF else 1
        if char_units > limit:
            raise ValueError(
                "a single character requires more UTF-16 code units than limit allows"
            )
        if current_chars and current_units + char_units > limit:
            chunks.append("".join(current_chars))
            current_chars = []
            current_units = 0
        current_chars.append(char)
        current_units += char_units

    if current_chars:
        chunks.append("".join(current_chars))

    return chunks


def _is_command_text(text: str | None) -> bool:
    return isinstance(text, str) and text.startswith("/")


def is_ordinary_text_message(message: Message) -> bool:
    """True for plain text messages that are not slash commands.

    Structurally excludes commands from the ordinary-text handler by
    content (a leading "/"), independent of handler registration order.
    """
    return isinstance(message.text, str) and not _is_command_text(message.text)


def is_non_text_message(message: Message) -> bool:
    """True only when the message genuinely has no text (photo, sticker, etc.).

    Kept structurally narrow on purpose: this is the only filter for
    :func:`handle_non_text_message`, so it must never also match malformed
    slash-prefixed text (that is absorbed earlier by
    :func:`ignore_malformed_command`) or any other text message.
    """
    return message.text is None


async def cmd_start(message: Message) -> None:
    """Send a short welcome. Touches no services."""
    await message.answer(_START_MESSAGE)


async def cmd_help(message: Message) -> None:
    """List the four commands and briefly explain ordinary messages. Touches no services."""
    await message.answer(_HELP_MESSAGE)


async def cmd_memory(message: Message, memory_service: MemoryService) -> None:
    """Reply with the total number of stored memories for this user."""
    from_user = message.from_user
    if from_user is None:
        return

    try:
        count = await asyncio.to_thread(memory_service.get_memory_count, user_id=from_user.id)
    except VectorMemoryError as exc:
        logger.warning("event=memory_count_failed error_type=%s", type(exc).__name__)
        await message.answer(_MEMORY_COUNT_FAILURE_REPLY)
        return

    await message.answer(f"В памяти сохранено: {count} {_pluralize_records_ru(count)}.")


async def cmd_forget_me(message: Message, memory_service: MemoryService) -> None:
    """Delete all stored memories for this user."""
    from_user = message.from_user
    if from_user is None:
        return

    try:
        await asyncio.to_thread(memory_service.forget_user, user_id=from_user.id)
    except VectorMemoryError as exc:
        logger.warning("event=forget_me_failed error_type=%s", type(exc).__name__)
        await message.answer(_FORGET_ME_FAILURE_REPLY)
        return

    await message.answer(_FORGET_ME_SUCCESS_REPLY)


async def handle_unknown_command(message: Message) -> None:
    """Reply to any slash command not matched by a known handler. Touches no services."""
    await message.answer(_UNKNOWN_COMMAND_REPLY)


async def ignore_foreign_command(message: Message) -> None:
    """Silently absorb a command-shaped message addressed to another bot.

    Reached only when a command's ``@mention`` suffix did not match this
    bot's own username -- aiogram's ``Command`` filter already rejected it
    for every earlier handler. Sends no reply and touches no service, so
    the update does not fall through to the ordinary-text or non-text
    catch-all handlers either.
    """
    return None


async def ignore_malformed_command(message: Message) -> None:
    """Silently absorb slash-prefixed text that is not a syntactically valid command.

    Reached only for text starting with "/" that matched neither a known
    command, the unknown-command catch-all, nor the foreign-bot-mention
    catch-all (e.g. "/", "//help", "/foo-bar"). Sends no reply and touches
    no service, so it can never fall into ordinary-text processing or the
    non-text handler either.
    """
    return None


async def handle_non_text_message(message: Message) -> None:
    """Reply to photos, stickers, voice, documents, and any other non-text update.

    Touches no services -- never calls recall, chat generation, or remember.
    """
    await message.answer(_NON_TEXT_REPLY)


async def handle_text_message(
    message: Message,
    memory_service: MemoryService,
    chat_service: ChatService,
) -> None:
    """Ordinary text-message flow: recall -> generate -> send -> remember.

    ``remember`` only runs after every reply chunk has been sent
    successfully. A send failure re-raises (after safe logging) so
    aiogram's own error boundary handles it, and skips ``remember``
    entirely by simply never reaching that line.
    """
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
        logger.warning("event=recall_failed error_type=%s", type(exc).__name__)
        memories = []

    try:
        reply = await chat_service.generate_reply(user_text=user_text, memories=memories)
    except ChatServiceError as exc:
        logger.warning("event=chat_generation_failed error_type=%s", type(exc).__name__)
        await message.answer(_CHAT_FAILURE_REPLY)
        return

    chunks = split_telegram_text(reply)
    try:
        for chunk in chunks:
            await message.answer(chunk)
    except Exception as exc:
        # Broad catch is deliberate and narrowly scoped to Telegram sending:
        # a failed send must never be followed by remember(), and aiogram's
        # own error boundary -- not a second message from us -- handles it.
        logger.warning("event=send_failed error_type=%s", type(exc).__name__)
        raise

    try:
        result = await asyncio.to_thread(
            memory_service.remember,
            user_id=user_id,
            text=user_text,
            username=username,
            first_name=first_name,
            last_name=last_name,
        )
    except (VectorMemoryError, MemoryServiceError) as exc:
        logger.warning("event=remember_failed error_type=%s", type(exc).__name__)
        return

    logger.info(
        "event=memory_write action=%s reason=%s similarity_score=%s",
        result.action,
        result.reason,
        result.similarity_score,
    )


def _build_router() -> Router:
    router = Router(name="stage5_router")

    router.message.register(cmd_start, CommandStart())
    router.message.register(cmd_help, Command("help"))
    router.message.register(cmd_memory, Command("memory"))
    router.message.register(cmd_forget_me, Command("forget_me"))
    router.message.register(handle_unknown_command, Command(_COMMAND_NAME_PATTERN))
    router.message.register(
        ignore_foreign_command, Command(_COMMAND_NAME_PATTERN, ignore_mention=True)
    )
    router.message.register(ignore_malformed_command, F.text.startswith("/"))
    router.message.register(handle_text_message, is_ordinary_text_message)
    router.message.register(handle_non_text_message, is_non_text_message)

    return router


def create_dispatcher(
    *,
    memory_service: MemoryService,
    chat_service: ChatService,
) -> Dispatcher:
    """Build the Stage 5 Dispatcher with both services as contextual data.

    Handlers receive *memory_service* / *chat_service* by parameter name via
    aiogram's workflow data -- no module-level mutable singleton is used.
    """
    dispatcher = Dispatcher(memory_service=memory_service, chat_service=chat_service)
    dispatcher.include_router(_build_router())
    return dispatcher


async def run_bot() -> None:
    """Construct every service and start long polling. Never runs at import time."""
    settings = get_settings()
    manager = PineconeManager(settings)
    memory_service = MemoryService(manager=manager, settings=settings)
    chat_service = ChatService(settings=settings)

    bot = Bot(token=settings.TELEGRAM_BOT_TOKEN.get_secret_value())
    dispatcher = create_dispatcher(memory_service=memory_service, chat_service=chat_service)

    await dispatcher.start_polling(bot)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
