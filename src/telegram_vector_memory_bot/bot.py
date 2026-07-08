"""Слой Telegram-приложения на aiogram 3.

Связывает четыре команды Stage 5 и обычный поток текстовых сообщений поверх
уже существующего синхронного ``MemoryService`` и использующего инструменты
``HaystackAgentService``. Этот модуль не создаёт ни внешнего клиента, ни
объекта ``Settings`` во время импорта -- всё строится внутри :func:`run_bot`,
который выполняется только при запуске модуля как скрипта.

``ChatService`` (простой адаптер Chat Completions без инструментов) остаётся
в кодовой базе как legacy-adapter -- полностью реализован и протестирован в
``chat_service.py`` -- но больше не подключён к живому потоку сообщений.

Поток обработки обычного текста: recall -> генерация ответа агентом (который
может вызывать инструменты) -> отправка каждого фрагмента ответа -> remember,
именно в таком порядке. ``remember`` выполняется только после того, как все
фрагменты ответа успешно отправлены; неудачная отправка выбрасывает
исключение повторно, чтобы его обработал собственный error boundary aiogram,
и ``remember`` при этом никогда не вызывается.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Final

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from .config import get_settings
from .haystack_agent import HaystackAgentService, HaystackAgentServiceError
from .memory_service import MemoryService, MemoryServiceError
from .pinecone_manager import PineconeManager, VectorMemoryError

logger = logging.getLogger(__name__)

_TELEGRAM_MESSAGE_LIMIT: Final = 4096

# Фиксированные, безопасные ответы пользователю. Намеренно приватные --
# тесты проверяют буквальный текст независимо, а не импортируют эти
# константы, так что изменение здесь не может незаметно утащить за собой
# устаревшее ожидание в тесте.
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

# Совпадает с любым синтаксически корректным именем команды Telegram-бота
# (буквы, цифры, подчёркивание, 1-32 символа) -- используется как catch-all,
# чтобы маршрут "неизвестная команда" полагался на собственный фильтр
# Command aiogram (и его валидацию bot-mention) вместо самодельного парсера
# slash-префикса/mention.
_COMMAND_NAME_PATTERN: Final = re.compile(r"^[A-Za-z0-9_]{1,32}$")


def _pluralize_records_ru(count: int) -> str:
    """Русская форма множественного числа слова "запись" для неотрицательного *count*.

    Небольшое, фиксированное грамматическое правило для одного слова -- а не
    полноценный слой локализации.
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
    """Разбить *text* на безопасные для Telegram фрагменты, измеряемые в UTF-16 code units.

    Детерминированное жёсткое разбиение -- не пытается разбивать по границам
    слов или абзацев. Конкатенация возвращённых фрагментов всегда точно
    воспроизводит *text*, сохраняет порядок и никогда не создаёт пустой
    фрагмент. Каждый выданный фрагмент укладывается в *limit* UTF-16
    code units: если один code point Unicode (например, некоторые эмодзи,
    занимающие 2 UTF-16 code units) сам по себе превышает *limit*, это
    запрос, который в принципе невозможно удовлетворить без разбиения
    code point на невалидную суррогатную половину, поэтому вместо превышающего
    лимит или пустого фрагмента выбрасывается ``ValueError``.
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
    """True для обычных текстовых сообщений, не являющихся slash-командами.

    Структурно исключает команды из обработчика обычного текста по
    содержимому (ведущий "/"), независимо от порядка регистрации хендлеров.
    """
    return isinstance(message.text, str) and not _is_command_text(message.text)


def is_non_text_message(message: Message) -> bool:
    """True только когда у сообщения действительно нет текста (фото, стикер и т.д.).

    Намеренно оставлен структурно узким: это единственный фильтр для
    :func:`handle_non_text_message`, поэтому он никогда не должен совпадать ни
    с некорректным slash-текстом (его раньше поглощает
    :func:`ignore_malformed_command`), ни с любым другим текстовым
    сообщением.
    """
    return message.text is None


async def cmd_start(message: Message) -> None:
    """Отправить короткое приветствие. Не обращается ни к одному сервису."""
    await message.answer(_START_MESSAGE)


async def cmd_help(message: Message) -> None:
    """Перечислить четыре команды и коротко объяснить обычные сообщения.

    Не обращается ни к одному сервису.
    """
    await message.answer(_HELP_MESSAGE)


async def cmd_memory(message: Message, memory_service: MemoryService) -> None:
    """Ответить общим числом сохранённых воспоминаний этого пользователя."""
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
    """Удалить все сохранённые воспоминания этого пользователя."""
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
    """Ответить на любую slash-команду, не совпавшую с известным хендлером.

    Не обращается ни к одному сервису.
    """
    await message.answer(_UNKNOWN_COMMAND_REPLY)


async def ignore_foreign_command(message: Message) -> None:
    """Молча поглотить командо-подобное сообщение, адресованное другому боту.

    Достигается только когда суффикс ``@mention`` команды не совпал с
    username этого бота -- фильтр ``Command`` aiogram уже отклонил его для
    всех более ранних хендлеров. Не отправляет ответ и не обращается ни к
    одному сервису, так что обновление не проваливается ни в обработчик
    обычного текста, ни в catch-all для нетекстовых сообщений.
    """
    return None


async def ignore_malformed_command(message: Message) -> None:
    """Молча поглотить slash-текст, не являющийся синтаксически корректной командой.

    Достигается только для текста, начинающегося с "/", который не совпал ни
    с известной командой, ни с catch-all для неизвестной команды, ни с
    catch-all для чужого бота (например, "/", "//help", "/foo-bar"). Не
    отправляет ответ и не обращается ни к одному сервису, так что не может
    провалиться ни в обработку обычного текста, ни в обработчик нетекстовых
    сообщений.
    """
    return None


async def handle_non_text_message(message: Message) -> None:
    """Ответить на фото, стикеры, голосовые, документы и любое другое нетекстовое обновление.

    Не обращается ни к одному сервису -- никогда не вызывает recall,
    генерацию ответа или remember.
    """
    await message.answer(_NON_TEXT_REPLY)


async def handle_text_message(
    message: Message,
    memory_service: MemoryService,
    reply_service: HaystackAgentService,
) -> None:
    """Поток обработки обычного текста: recall -> generate -> send -> remember.

    ``reply_service`` генерирует ответ; в production это
    ``HaystackAgentService``, который может сам вызвать инструмент погоды,
    валюты, текущего времени, праздников или факта о числе, прежде чем
    ответить.
    ``remember`` выполняется только после того, как все фрагменты ответа
    успешно отправлены. Сбой отправки выбрасывает исключение повторно (после
    безопасного логирования), чтобы его обработал собственный error boundary
    aiogram, и ``remember`` при этом полностью пропускается, просто никогда
    не доходя до этой строки.
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
        reply = await reply_service.generate_reply(user_text=user_text, memories=memories)
    except HaystackAgentServiceError as exc:
        logger.warning("event=chat_generation_failed error_type=%s", type(exc).__name__)
        await message.answer(_CHAT_FAILURE_REPLY)
        return

    chunks = split_telegram_text(reply)
    try:
        for chunk in chunks:
            await message.answer(chunk)
    except Exception as exc:
        # Широкий except намеренный и узко ограничен отправкой в Telegram:
        # неудачная отправка никогда не должна сопровождаться remember(), а
        # обрабатывает её собственный error boundary aiogram -- а не второе
        # сообщение от нас.
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
    reply_service: HaystackAgentService,
) -> Dispatcher:
    """Построить Dispatcher Stage 5 с обоими сервисами как контекстными данными.

    Хендлеры получают *memory_service* / *reply_service* по имени параметра
    через workflow data aiogram -- без модульного изменяемого singleton.
    """
    dispatcher = Dispatcher(memory_service=memory_service, reply_service=reply_service)
    dispatcher.include_router(_build_router())
    return dispatcher


async def run_bot() -> None:
    """Сконструировать все сервисы и запустить long polling. Никогда не выполняется при импорте."""
    settings = get_settings()
    manager = PineconeManager(settings)
    memory_service = MemoryService(manager=manager, settings=settings)
    reply_service = HaystackAgentService(settings=settings)

    bot = Bot(token=settings.TELEGRAM_BOT_TOKEN.get_secret_value())
    dispatcher = create_dispatcher(memory_service=memory_service, reply_service=reply_service)

    await dispatcher.start_polling(bot)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
