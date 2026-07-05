"""Unit tests for telegram_vector_memory_bot.bot.

Most handlers are exercised directly as plain async functions against small
duck-typed fakes (FakeMessage, FakeUser, FakeMemoryService, FakeChatService).
A separate "Dispatcher-fed routing" section instead drives the real,
constructed Dispatcher end-to-end via ``feed_raw_update`` against a real
``aiogram.Bot`` bound to an offline fake session (FakeTelegramSession) that
intercepts only ``GetMe``/``SendMessage`` and never performs network I/O --
this is what actually exercises aiogram's own Command bot-mention semantics,
which a direct handler call cannot.

Async methods are exercised via ``asyncio.run`` directly rather than a
pytest-asyncio plugin, since none is a project dependency.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import time
from datetime import UTC, datetime
from typing import Any

import aiogram
import openai
import pinecone
import pytest
from aiogram.client.session.base import BaseSession
from aiogram.methods import GetMe, SendMessage
from aiogram.types import Chat as AiogramChat
from aiogram.types import Message as AiogramMessage
from aiogram.types import User as AiogramUser

from telegram_vector_memory_bot import bot as bot_module
from telegram_vector_memory_bot.chat_service import ChatServiceError
from telegram_vector_memory_bot.config import Settings
from telegram_vector_memory_bot.models import (
    MemoryAction,
    MemoryReason,
    MemoryWriteResult,
    RecalledMemory,
)
from telegram_vector_memory_bot.pinecone_manager import VectorQueryError, VectorStorageError

FAKE_TOKEN = "123456:fake-injected-telegram-token-ABCDEF"
FAKE_API_KEY = "sk-FAKE-INJECTED-SECRET-VALUE"
_CURRENT_BOT_USERNAME = "current_bot_test"
_CURRENT_BOT_ID = 999

# Expected fixed reply texts, copied literally rather than imported from
# bot_module -- these are private production constants, so a change to the
# wording there must independently fail this suite instead of silently
# changing the test expectation along with it.
_START_TEXT = (
    "Привет! Я запоминаю то, что вы мне пишете, и использую эти воспоминания "
    "как контекст в разговоре.\n"
    "Команды: /help, /memory, /forget_me."
)
_HELP_TEXT = (
    "Доступные команды:\n"
    "/start — приветствие\n"
    "/help — эта справка\n"
    "/memory — сколько у вас сохранено воспоминаний\n"
    "/forget_me — удалить всю вашу память\n\n"
    "Обычные текстовые сообщения: я отвечаю с учётом того, что вы рассказывали раньше."
)
_MEMORY_COUNT_FAILURE_TEXT = "Не удалось получить количество воспоминаний. Попробуйте позже."
_FORGET_ME_SUCCESS_TEXT = "Готово: вся ваша память удалена."
_FORGET_ME_FAILURE_TEXT = "Не удалось удалить память. Попробуйте позже."
_UNKNOWN_COMMAND_TEXT = "Неизвестная команда. Наберите /help, чтобы увидеть список команд."
_CHAT_FAILURE_TEXT = "Извините, не получилось сформировать ответ. Попробуйте ещё раз."
_NON_TEXT_TEXT = "Пока я понимаю только текстовые сообщения."


def _build_settings(**overrides: Any) -> Settings:
    data: dict[str, Any] = {
        "PINECONE_API_KEY": FAKE_API_KEY,
        "PINECONE_INDEX_NAME": "test-index",
        "OPENAI_API_KEY": FAKE_API_KEY,
        "OPENAI_CHAT_MODEL": "gpt-4o-mini",
        "TELEGRAM_BOT_TOKEN": FAKE_TOKEN,
    }
    data.update(overrides)
    return Settings(_env_file=None, **data)


def _default_write_result() -> MemoryWriteResult:
    return MemoryWriteResult(
        action=MemoryAction.INSERTED,
        reason=MemoryReason.NEW_MEMORY,
        memory_id="mem-1",
        existing_id=None,
        similarity_score=None,
    )


def _recalled_memory(text: str = "previous fact") -> RecalledMemory:
    return RecalledMemory(
        memory_id="mem-1",
        text=text,
        score=0.9,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        source="telegram",
        content_hash="abc123",
    )


class FakeUser:
    def __init__(
        self,
        *,
        id: int = 1,
        username: str | None = "jdoe",
        first_name: str | None = "Jane",
        last_name: str | None = "Doe",
    ) -> None:
        self.id = id
        self.username = username
        self.first_name = first_name
        self.last_name = last_name


class FakeMessage:
    """Duck-typed stand-in for aiogram's Message -- no real Bot/session involved."""

    def __init__(
        self,
        *,
        text: str | None,
        from_user: FakeUser | None,
        events: list[str] | None = None,
    ) -> None:
        self.text = text
        self.from_user = from_user
        self.answer_calls: list[str] = []
        self.fail_on_answer_call: int | None = None
        self.answer_exception: Exception = RuntimeError("telegram send failed")
        self.events = events if events is not None else []

    async def answer(self, text: str) -> None:
        index = len(self.answer_calls)
        self.answer_calls.append(text)
        self.events.append("send")
        if self.fail_on_answer_call is not None and index == self.fail_on_answer_call:
            raise self.answer_exception

    def __repr__(self) -> str:
        return "FAKE_MESSAGE_REPR_MARKER_DO_NOT_LOG"


class FakeMemoryService:
    """Duck-typed stand-in for MemoryService's public API used by bot.py."""

    def __init__(self, *, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []
        self.recall_calls: list[dict[str, Any]] = []
        self.remember_calls: list[dict[str, Any]] = []
        self.forget_user_calls: list[dict[str, Any]] = []
        self.get_memory_count_calls: list[dict[str, Any]] = []

        self.recall_response: list[RecalledMemory] = []
        self.remember_response: MemoryWriteResult = _default_write_result()
        self.get_memory_count_response: int = 0

        self.raise_on_recall: Exception | None = None
        self.raise_on_remember: Exception | None = None
        self.raise_on_forget_user: Exception | None = None
        self.raise_on_get_memory_count: Exception | None = None

    def recall(
        self, *, user_id: int, query: str, top_k: int | None = None
    ) -> list[RecalledMemory]:
        self.events.append("recall")
        self.recall_calls.append({"user_id": user_id, "query": query, "top_k": top_k})
        if self.raise_on_recall is not None:
            raise self.raise_on_recall
        return self.recall_response

    def remember(
        self,
        *,
        user_id: int,
        text: str,
        username: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
    ) -> MemoryWriteResult:
        self.events.append("remember")
        self.remember_calls.append(
            {
                "user_id": user_id,
                "text": text,
                "username": username,
                "first_name": first_name,
                "last_name": last_name,
            }
        )
        if self.raise_on_remember is not None:
            raise self.raise_on_remember
        return self.remember_response

    def forget_user(self, *, user_id: int) -> None:
        self.forget_user_calls.append({"user_id": user_id})
        if self.raise_on_forget_user is not None:
            raise self.raise_on_forget_user

    def get_memory_count(self, *, user_id: int) -> int:
        self.get_memory_count_calls.append({"user_id": user_id})
        if self.raise_on_get_memory_count is not None:
            raise self.raise_on_get_memory_count
        return self.get_memory_count_response


class FakeChatService:
    """Duck-typed stand-in for ChatService's public API used by bot.py."""

    def __init__(self, *, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []
        self.generate_reply_calls: list[dict[str, Any]] = []
        self.response: str = "generated reply"
        self.exception: Exception | None = None

    async def generate_reply(self, *, user_text: str, memories: Any) -> str:
        self.events.append("generate")
        self.generate_reply_calls.append({"user_text": user_text, "memories": list(memories)})
        if self.exception is not None:
            raise self.exception
        return self.response


class FakeTelegramSession(BaseSession):
    """Offline aiogram session: intercepts only GetMe/SendMessage, no network I/O.

    Bound to a real ``aiogram.Bot``, this lets tests drive the real,
    constructed ``Dispatcher`` via ``feed_raw_update`` -- including aiogram's
    own ``Command`` bot-mention validation, which calls ``bot.me()`` -- while
    guaranteeing no Telegram API request is ever actually made.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sent_messages: list[dict[str, Any]] = []
        self._next_message_id = 1

    async def make_request(self, bot: Any, method: Any, timeout: float | None = None) -> Any:
        if isinstance(method, GetMe):
            return AiogramUser(
                id=_CURRENT_BOT_ID,
                is_bot=True,
                first_name="Test Bot",
                username=_CURRENT_BOT_USERNAME,
            )
        if isinstance(method, SendMessage):
            message_id = self._next_message_id
            self._next_message_id += 1
            self.sent_messages.append({"chat_id": method.chat_id, "text": method.text})
            return AiogramMessage(
                message_id=message_id,
                date=datetime.now(UTC),
                chat=AiogramChat(id=method.chat_id, type="private"),
                text=method.text,
            )
        raise AssertionError(
            f"unexpected Telegram API method in offline test: {type(method).__name__}"
        )

    async def close(self) -> None:
        return None

    async def stream_content(
        self,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: int = 30,
        chunk_size: int = 65536,
        raise_for_status: bool = True,
    ) -> Any:
        raise NotImplementedError("not used in offline tests")
        yield b""  # pragma: no cover -- keeps this an async generator function


def _make_update(
    *, text: str | None = None, photo: bool = False, user_id: int = 123
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "message_id": 1,
        "date": int(time.time()),
        "chat": {"id": user_id, "type": "private"},
        "from": {"id": user_id, "is_bot": False, "first_name": "Jane"},
    }
    if text is not None:
        message["text"] = text
    if photo:
        message["photo"] = [
            {"file_id": "abc", "file_unique_id": "abc-unique", "width": 90, "height": 90}
        ]
    return {"update_id": 1, "message": message}


def _build_harness() -> tuple[
    aiogram.Bot, aiogram.Dispatcher, FakeTelegramSession, FakeMemoryService, FakeChatService
]:
    session = FakeTelegramSession()
    telegram_bot = aiogram.Bot(token=FAKE_TOKEN, session=session)
    memory_service = FakeMemoryService()
    chat_service = FakeChatService()
    dispatcher = bot_module.create_dispatcher(
        memory_service=memory_service, chat_service=chat_service
    )
    return telegram_bot, dispatcher, session, memory_service, chat_service


# ---------------------------------------------------------------------------
# Dispatcher / startup wiring
# ---------------------------------------------------------------------------


def test_create_dispatcher_includes_stage5_router() -> None:
    memory_service = FakeMemoryService()
    chat_service = FakeChatService()

    dispatcher = bot_module.create_dispatcher(
        memory_service=memory_service, chat_service=chat_service
    )

    assert any(router.name == "stage5_router" for router in dispatcher.sub_routers)


def test_create_dispatcher_exposes_services_as_workflow_data() -> None:
    memory_service = FakeMemoryService()
    chat_service = FakeChatService()

    dispatcher = bot_module.create_dispatcher(
        memory_service=memory_service, chat_service=chat_service
    )

    assert dispatcher.workflow_data["memory_service"] is memory_service
    assert dispatcher.workflow_data["chat_service"] is chat_service


def test_importing_bot_module_constructs_no_external_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(self: Any, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("must not be constructed merely by importing bot.py")

    monkeypatch.setattr(pinecone.Pinecone, "__init__", _explode)
    monkeypatch.setattr(openai.OpenAI, "__init__", _explode)
    monkeypatch.setattr(openai.AsyncOpenAI, "__init__", _explode)
    monkeypatch.setattr(Settings, "__init__", _explode)
    monkeypatch.setattr(aiogram.Bot, "__init__", _explode)

    reloaded = importlib.reload(bot_module)

    assert reloaded.create_dispatcher is not None


def test_run_bot_wires_services_in_order_and_polls_with_constructed_bot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_settings = _build_settings()
    events: list[str] = []
    constructed_bots: list[Any] = []
    polling_calls: list[Any] = []

    class RecordingBot:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            events.append("Bot")
            constructed_bots.append(self)

    class RecordingDispatcher:
        async def start_polling(self, bot: Any) -> None:
            events.append("start_polling")
            polling_calls.append(bot)

    def fake_pinecone_manager(settings: Settings) -> str:
        events.append("PineconeManager")
        return "manager-marker"

    def fake_memory_service(*, manager: Any, settings: Settings) -> str:
        events.append("MemoryService")
        assert manager == "manager-marker"
        return "memory-service-marker"

    def fake_chat_service(*, settings: Settings) -> str:
        events.append("ChatService")
        return "chat-service-marker"

    dispatcher_instance = RecordingDispatcher()

    def fake_create_dispatcher(*, memory_service: Any, chat_service: Any) -> RecordingDispatcher:
        events.append("create_dispatcher")
        assert memory_service == "memory-service-marker"
        assert chat_service == "chat-service-marker"
        return dispatcher_instance

    monkeypatch.setattr(bot_module, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(bot_module, "PineconeManager", fake_pinecone_manager)
    monkeypatch.setattr(bot_module, "MemoryService", fake_memory_service)
    monkeypatch.setattr(bot_module, "ChatService", fake_chat_service)
    monkeypatch.setattr(bot_module, "Bot", RecordingBot)
    monkeypatch.setattr(bot_module, "create_dispatcher", fake_create_dispatcher)

    asyncio.run(bot_module.run_bot())

    assert events == [
        "PineconeManager",
        "MemoryService",
        "ChatService",
        "Bot",
        "create_dispatcher",
        "start_polling",
    ]
    assert constructed_bots[0].kwargs["token"] == FAKE_TOKEN
    assert polling_calls == [constructed_bots[0]]


def test_main_runs_run_bot_via_asyncio_run(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_run_bot() -> None:
        calls.append("run_bot")

    monkeypatch.setattr(bot_module, "run_bot", fake_run_bot)

    bot_module.main()

    assert calls == ["run_bot"]


def test_run_bot_does_not_log_the_telegram_token(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_settings = _build_settings()

    class RecordingBot:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class RecordingDispatcher:
        async def start_polling(self, bot: Any) -> None:
            return None

    monkeypatch.setattr(bot_module, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(bot_module, "PineconeManager", lambda settings: "manager-marker")
    monkeypatch.setattr(
        bot_module, "MemoryService", lambda *, manager, settings: "memory-service-marker"
    )
    monkeypatch.setattr(bot_module, "ChatService", lambda *, settings: "chat-service-marker")
    monkeypatch.setattr(bot_module, "Bot", RecordingBot)
    monkeypatch.setattr(
        bot_module,
        "create_dispatcher",
        lambda *, memory_service, chat_service: RecordingDispatcher(),
    )

    with caplog.at_level(logging.DEBUG):
        asyncio.run(bot_module.run_bot())

    assert FAKE_TOKEN not in caplog.text


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def test_start_sends_expected_text_and_touches_no_services() -> None:
    message = FakeMessage(text="/start", from_user=FakeUser())

    asyncio.run(bot_module.cmd_start(message))

    assert message.answer_calls == [_START_TEXT]


def test_help_sends_expected_text_and_touches_no_services() -> None:
    message = FakeMessage(text="/help", from_user=FakeUser())

    asyncio.run(bot_module.cmd_help(message))

    assert message.answer_calls == [_HELP_TEXT]


def test_memory_command_calls_get_memory_count_and_replies_with_count() -> None:
    memory_service = FakeMemoryService()
    memory_service.get_memory_count_response = 3
    message = FakeMessage(text="/memory", from_user=FakeUser(id=42))

    asyncio.run(bot_module.cmd_memory(message, memory_service))

    assert memory_service.get_memory_count_calls == [{"user_id": 42}]
    assert message.answer_calls == ["В памяти сохранено: 3 записи."]


@pytest.mark.parametrize(
    ("count", "expected_word"),
    [
        (0, "записей"),
        (1, "запись"),
        (2, "записи"),
        (5, "записей"),
        (11, "записей"),
        (21, "запись"),
        (22, "записи"),
        (25, "записей"),
    ],
)
def test_memory_command_uses_correct_russian_plural_form(
    count: int, expected_word: str
) -> None:
    memory_service = FakeMemoryService()
    memory_service.get_memory_count_response = count
    message = FakeMessage(text="/memory", from_user=FakeUser())

    asyncio.run(bot_module.cmd_memory(message, memory_service))

    assert message.answer_calls == [f"В памяти сохранено: {count} {expected_word}."]


def test_memory_command_missing_from_user_does_not_call_service_or_reply() -> None:
    memory_service = FakeMemoryService()
    message = FakeMessage(text="/memory", from_user=None)

    asyncio.run(bot_module.cmd_memory(message, memory_service))

    assert memory_service.get_memory_count_calls == []
    assert message.answer_calls == []


def test_forget_me_missing_from_user_does_not_call_service_or_reply() -> None:
    memory_service = FakeMemoryService()
    message = FakeMessage(text="/forget_me", from_user=None)

    asyncio.run(bot_module.cmd_forget_me(message, memory_service))

    assert memory_service.forget_user_calls == []
    assert message.answer_calls == []


def test_memory_command_failure_sends_fixed_safe_reply() -> None:
    memory_service = FakeMemoryService()
    memory_service.raise_on_get_memory_count = VectorQueryError("stats failed: secret-detail")
    message = FakeMessage(text="/memory", from_user=FakeUser())

    asyncio.run(bot_module.cmd_memory(message, memory_service))

    assert message.answer_calls == [_MEMORY_COUNT_FAILURE_TEXT]


def test_forget_me_calls_only_forget_user_for_that_user() -> None:
    memory_service = FakeMemoryService()
    message = FakeMessage(text="/forget_me", from_user=FakeUser(id=7))

    asyncio.run(bot_module.cmd_forget_me(message, memory_service))

    assert memory_service.forget_user_calls == [{"user_id": 7}]
    assert memory_service.recall_calls == []
    assert memory_service.remember_calls == []
    assert message.answer_calls == [_FORGET_ME_SUCCESS_TEXT]


def test_forget_me_already_empty_namespace_is_still_success() -> None:
    # forget_user is idempotent at the MemoryService/PineconeManager layer;
    # a no-op success looks identical to any other success from here -- no
    # exception is raised in either case.
    memory_service = FakeMemoryService()
    message = FakeMessage(text="/forget_me", from_user=FakeUser())

    asyncio.run(bot_module.cmd_forget_me(message, memory_service))

    assert message.answer_calls == [_FORGET_ME_SUCCESS_TEXT]


def test_forget_me_failure_does_not_claim_deletion() -> None:
    memory_service = FakeMemoryService()
    memory_service.raise_on_forget_user = VectorStorageError("delete failed: secret-detail")
    message = FakeMessage(text="/forget_me", from_user=FakeUser())

    asyncio.run(bot_module.cmd_forget_me(message, memory_service))

    assert message.answer_calls == [_FORGET_ME_FAILURE_TEXT]
    assert _FORGET_ME_SUCCESS_TEXT not in message.answer_calls


def test_unknown_command_replies_and_touches_no_services() -> None:
    message = FakeMessage(text="/unknown_command", from_user=FakeUser())

    asyncio.run(bot_module.handle_unknown_command(message))

    assert message.answer_calls == [_UNKNOWN_COMMAND_TEXT]


def test_commands_never_call_remember() -> None:
    memory_service = FakeMemoryService()

    asyncio.run(bot_module.cmd_start(FakeMessage(text="/start", from_user=FakeUser())))
    asyncio.run(bot_module.cmd_help(FakeMessage(text="/help", from_user=FakeUser())))
    asyncio.run(
        bot_module.cmd_memory(FakeMessage(text="/memory", from_user=FakeUser()), memory_service)
    )
    asyncio.run(
        bot_module.cmd_forget_me(
            FakeMessage(text="/forget_me", from_user=FakeUser()), memory_service
        )
    )
    asyncio.run(
        bot_module.handle_unknown_command(
            FakeMessage(text="/nonexistent", from_user=FakeUser())
        )
    )

    assert memory_service.remember_calls == []


# ---------------------------------------------------------------------------
# Message flow and order
# ---------------------------------------------------------------------------


def test_message_flow_exact_order_recall_generate_send_remember() -> None:
    events: list[str] = []
    memory_service = FakeMemoryService(events=events)
    chat_service = FakeChatService(events=events)
    message = FakeMessage(text="hello", from_user=FakeUser(), events=events)

    asyncio.run(bot_module.handle_text_message(message, memory_service, chat_service))

    assert events == ["recall", "generate", "send", "remember"]


def test_chat_service_receives_exact_memories_from_recall() -> None:
    memory_service = FakeMemoryService()
    recalled = [_recalled_memory("fact one"), _recalled_memory("fact two")]
    memory_service.recall_response = recalled
    chat_service = FakeChatService()
    message = FakeMessage(text="hi", from_user=FakeUser())

    asyncio.run(bot_module.handle_text_message(message, memory_service, chat_service))

    assert chat_service.generate_reply_calls[0]["memories"] == recalled


def test_recall_failure_degrades_to_empty_list_and_generation_still_runs() -> None:
    memory_service = FakeMemoryService()
    memory_service.raise_on_recall = VectorQueryError("query failed")
    chat_service = FakeChatService()
    message = FakeMessage(text="hi", from_user=FakeUser())

    asyncio.run(bot_module.handle_text_message(message, memory_service, chat_service))

    assert chat_service.generate_reply_calls[0]["memories"] == []
    assert message.answer_calls == [chat_service.response]
    assert len(memory_service.remember_calls) == 1


def test_chat_failure_sends_fallback_and_does_not_call_remember() -> None:
    memory_service = FakeMemoryService()
    chat_service = FakeChatService()
    chat_service.exception = ChatServiceError("chat completion request failed")
    message = FakeMessage(text="hi", from_user=FakeUser())

    asyncio.run(bot_module.handle_text_message(message, memory_service, chat_service))

    assert message.answer_calls == [_CHAT_FAILURE_TEXT]
    assert memory_service.remember_calls == []


def test_send_failure_does_not_call_remember() -> None:
    memory_service = FakeMemoryService()
    chat_service = FakeChatService()
    chat_service.response = "short reply"
    message = FakeMessage(text="hi", from_user=FakeUser())
    message.fail_on_answer_call = 0

    with pytest.raises(RuntimeError):
        asyncio.run(bot_module.handle_text_message(message, memory_service, chat_service))

    assert memory_service.remember_calls == []


def test_failure_on_second_chunk_after_first_succeeded_still_does_not_call_remember() -> None:
    memory_service = FakeMemoryService()
    chat_service = FakeChatService()
    chat_service.response = "x" * 5000  # splits into two chunks under the 4096 default limit
    message = FakeMessage(text="hi", from_user=FakeUser())
    message.fail_on_answer_call = 1  # second chunk (0-indexed)

    with pytest.raises(RuntimeError):
        asyncio.run(bot_module.handle_text_message(message, memory_service, chat_service))

    assert len(message.answer_calls) == 2
    assert memory_service.remember_calls == []


def test_remember_failure_after_reply_sent_is_swallowed_safely() -> None:
    memory_service = FakeMemoryService()
    memory_service.raise_on_remember = VectorStorageError("upsert failed")
    chat_service = FakeChatService()
    message = FakeMessage(text="hi", from_user=FakeUser())

    # Must not raise -- the user already has their reply.
    asyncio.run(bot_module.handle_text_message(message, memory_service, chat_service))

    assert message.answer_calls == [chat_service.response]


def test_successful_memory_write_logged_with_safe_fields_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    memory_service = FakeMemoryService()
    memory_service.remember_response = MemoryWriteResult(
        action=MemoryAction.SKIPPED,
        reason=MemoryReason.SEMANTIC_DUPLICATE,
        memory_id=None,
        existing_id="mem-existing",
        similarity_score=0.87,
    )
    chat_service = FakeChatService()
    message = FakeMessage(text="hi", from_user=FakeUser())

    with caplog.at_level(logging.INFO):
        asyncio.run(bot_module.handle_text_message(message, memory_service, chat_service))

    assert "action=skipped" in caplog.text
    assert "reason=semantic_duplicate" in caplog.text
    assert "0.87" in caplog.text
    assert "mem-existing" not in caplog.text


def test_optional_user_fields_none_forwarded_safely() -> None:
    memory_service = FakeMemoryService()
    chat_service = FakeChatService()
    message = FakeMessage(
        text="hi",
        from_user=FakeUser(username=None, first_name=None, last_name=None),
    )

    asyncio.run(bot_module.handle_text_message(message, memory_service, chat_service))

    call = memory_service.remember_calls[0]
    assert call["username"] is None
    assert call["first_name"] is None
    assert call["last_name"] is None


def test_bot_reply_text_never_passed_to_remember() -> None:
    memory_service = FakeMemoryService()
    chat_service = FakeChatService()
    chat_service.response = "this is the generated reply"
    message = FakeMessage(text="original user message", from_user=FakeUser())

    asyncio.run(bot_module.handle_text_message(message, memory_service, chat_service))

    assert memory_service.remember_calls[0]["text"] == "original user message"
    assert chat_service.response not in memory_service.remember_calls[0]["text"]


def test_user_ids_isolated_across_two_message_executions() -> None:
    memory_service = FakeMemoryService()
    chat_service = FakeChatService()

    message_1 = FakeMessage(text="hi", from_user=FakeUser(id=1))
    message_2 = FakeMessage(text="hi", from_user=FakeUser(id=2))

    asyncio.run(bot_module.handle_text_message(message_1, memory_service, chat_service))
    asyncio.run(bot_module.handle_text_message(message_2, memory_service, chat_service))

    assert memory_service.recall_calls[0]["user_id"] == 1
    assert memory_service.recall_calls[1]["user_id"] == 2
    assert memory_service.remember_calls[0]["user_id"] == 1
    assert memory_service.remember_calls[1]["user_id"] == 2


def test_empty_text_does_not_call_external_services() -> None:
    memory_service = FakeMemoryService()
    chat_service = FakeChatService()
    message = FakeMessage(text="   ", from_user=FakeUser())

    asyncio.run(bot_module.handle_text_message(message, memory_service, chat_service))

    assert memory_service.recall_calls == []
    assert chat_service.generate_reply_calls == []
    assert memory_service.remember_calls == []
    assert message.answer_calls == []


def test_missing_from_user_does_not_call_external_services() -> None:
    memory_service = FakeMemoryService()
    chat_service = FakeChatService()
    message = FakeMessage(text="hi", from_user=None)

    asyncio.run(bot_module.handle_text_message(message, memory_service, chat_service))

    assert memory_service.recall_calls == []
    assert chat_service.generate_reply_calls == []
    assert memory_service.remember_calls == []
    assert message.answer_calls == []


# ---------------------------------------------------------------------------
# Routing / filter behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command_text", ["/start", "/help", "/memory", "/forget_me"])
def test_known_commands_cannot_match_plain_text_handler(command_text: str) -> None:
    message = FakeMessage(text=command_text, from_user=FakeUser())

    assert bot_module.is_ordinary_text_message(message) is False


def test_unknown_slash_command_cannot_match_plain_text_handler() -> None:
    message = FakeMessage(text="/totally_unknown_command", from_user=FakeUser())

    assert bot_module.is_ordinary_text_message(message) is False


def test_ordinary_text_matches_plain_text_handler() -> None:
    message = FakeMessage(text="just a normal message", from_user=FakeUser())

    assert bot_module.is_ordinary_text_message(message) is True


@pytest.mark.parametrize("text", ["just a normal message", "/", "//help", "/foo-bar", ""])
def test_is_non_text_message_false_for_any_text_message(text: str) -> None:
    message = FakeMessage(text=text, from_user=FakeUser())

    assert bot_module.is_non_text_message(message) is False


def test_is_non_text_message_true_only_when_text_is_none() -> None:
    message = FakeMessage(text=None, from_user=FakeUser())

    assert bot_module.is_non_text_message(message) is True


def test_non_text_message_never_calls_memory_or_chat_services() -> None:
    message = FakeMessage(text=None, from_user=FakeUser())

    asyncio.run(bot_module.handle_non_text_message(message))

    assert message.answer_calls == [_NON_TEXT_TEXT]


# ---------------------------------------------------------------------------
# Dispatcher-fed routing (real Dispatcher.feed_raw_update, fake Bot session)
# ---------------------------------------------------------------------------


def test_dispatcher_start_reaches_only_start_handler() -> None:
    telegram_bot, dispatcher, session, memory_service, chat_service = _build_harness()

    asyncio.run(dispatcher.feed_raw_update(telegram_bot, _make_update(text="/start")))

    assert session.sent_messages == [{"chat_id": 123, "text": _START_TEXT}]
    assert memory_service.recall_calls == []
    assert memory_service.remember_calls == []
    assert memory_service.get_memory_count_calls == []
    assert memory_service.forget_user_calls == []
    assert chat_service.generate_reply_calls == []


def test_dispatcher_help_addressed_to_current_bot_reaches_only_help_handler() -> None:
    telegram_bot, dispatcher, session, memory_service, chat_service = _build_harness()

    asyncio.run(
        dispatcher.feed_raw_update(
            telegram_bot, _make_update(text=f"/help@{_CURRENT_BOT_USERNAME}")
        )
    )

    assert session.sent_messages == [{"chat_id": 123, "text": _HELP_TEXT}]
    assert memory_service.recall_calls == []
    assert memory_service.remember_calls == []
    assert chat_service.generate_reply_calls == []


def test_dispatcher_help_addressed_to_another_bot_is_ignored() -> None:
    telegram_bot, dispatcher, session, memory_service, chat_service = _build_harness()

    asyncio.run(
        dispatcher.feed_raw_update(telegram_bot, _make_update(text="/help@some_other_bot"))
    )

    assert session.sent_messages == []
    assert memory_service.recall_calls == []
    assert memory_service.remember_calls == []
    assert memory_service.get_memory_count_calls == []
    assert memory_service.forget_user_calls == []
    assert chat_service.generate_reply_calls == []


def test_dispatcher_memory_reaches_only_memory_handler() -> None:
    telegram_bot, dispatcher, session, memory_service, chat_service = _build_harness()
    memory_service.get_memory_count_response = 4

    asyncio.run(dispatcher.feed_raw_update(telegram_bot, _make_update(text="/memory")))

    assert session.sent_messages == [{"chat_id": 123, "text": "В памяти сохранено: 4 записи."}]
    assert memory_service.get_memory_count_calls == [{"user_id": 123}]
    assert memory_service.recall_calls == []
    assert memory_service.remember_calls == []
    assert memory_service.forget_user_calls == []
    assert chat_service.generate_reply_calls == []


def test_dispatcher_forget_me_reaches_only_forget_handler() -> None:
    telegram_bot, dispatcher, session, memory_service, chat_service = _build_harness()

    asyncio.run(dispatcher.feed_raw_update(telegram_bot, _make_update(text="/forget_me")))

    assert session.sent_messages == [{"chat_id": 123, "text": _FORGET_ME_SUCCESS_TEXT}]
    assert memory_service.forget_user_calls == [{"user_id": 123}]
    assert memory_service.recall_calls == []
    assert memory_service.remember_calls == []
    assert memory_service.get_memory_count_calls == []
    assert chat_service.generate_reply_calls == []


def test_dispatcher_unknown_command_reaches_only_unknown_handler() -> None:
    telegram_bot, dispatcher, session, memory_service, chat_service = _build_harness()

    asyncio.run(dispatcher.feed_raw_update(telegram_bot, _make_update(text="/unknown")))

    assert session.sent_messages == [{"chat_id": 123, "text": _UNKNOWN_COMMAND_TEXT}]
    assert memory_service.recall_calls == []
    assert memory_service.remember_calls == []
    assert chat_service.generate_reply_calls == []


def test_dispatcher_unknown_addressed_to_current_bot_reaches_only_unknown_handler() -> None:
    telegram_bot, dispatcher, session, memory_service, chat_service = _build_harness()

    asyncio.run(
        dispatcher.feed_raw_update(
            telegram_bot, _make_update(text=f"/unknown@{_CURRENT_BOT_USERNAME}")
        )
    )

    assert session.sent_messages == [{"chat_id": 123, "text": _UNKNOWN_COMMAND_TEXT}]
    assert memory_service.recall_calls == []
    assert memory_service.remember_calls == []
    assert chat_service.generate_reply_calls == []


def test_dispatcher_unknown_command_addressed_to_another_bot_is_ignored() -> None:
    telegram_bot, dispatcher, session, memory_service, chat_service = _build_harness()

    asyncio.run(
        dispatcher.feed_raw_update(telegram_bot, _make_update(text="/unknown@some_other_bot"))
    )

    assert session.sent_messages == []
    assert memory_service.recall_calls == []
    assert memory_service.remember_calls == []
    assert chat_service.generate_reply_calls == []


def test_dispatcher_ordinary_text_reaches_only_message_orchestration() -> None:
    telegram_bot, dispatcher, session, memory_service, chat_service = _build_harness()
    chat_service.response = "a generated reply"

    asyncio.run(dispatcher.feed_raw_update(telegram_bot, _make_update(text="hello there")))

    assert session.sent_messages == [{"chat_id": 123, "text": "a generated reply"}]
    assert memory_service.recall_calls == [
        {"user_id": 123, "query": "hello there", "top_k": None}
    ]
    assert len(memory_service.remember_calls) == 1
    assert chat_service.generate_reply_calls[0]["user_text"] == "hello there"


def test_dispatcher_non_text_reaches_only_unsupported_message_handler() -> None:
    telegram_bot, dispatcher, session, memory_service, chat_service = _build_harness()

    asyncio.run(dispatcher.feed_raw_update(telegram_bot, _make_update(photo=True)))

    assert session.sent_messages == [{"chat_id": 123, "text": _NON_TEXT_TEXT}]
    assert memory_service.recall_calls == []
    assert memory_service.remember_calls == []
    assert chat_service.generate_reply_calls == []


def _assert_fully_absorbed(
    session: FakeTelegramSession,
    memory_service: FakeMemoryService,
    chat_service: FakeChatService,
) -> None:
    assert session.sent_messages == []
    assert memory_service.recall_calls == []
    assert memory_service.remember_calls == []
    assert memory_service.get_memory_count_calls == []
    assert memory_service.forget_user_calls == []
    assert chat_service.generate_reply_calls == []


@pytest.mark.parametrize("malformed_text", ["/", "//help", "/foo-bar"])
def test_dispatcher_malformed_slash_text_is_silently_absorbed(malformed_text: str) -> None:
    telegram_bot, dispatcher, session, memory_service, chat_service = _build_harness()

    asyncio.run(dispatcher.feed_raw_update(telegram_bot, _make_update(text=malformed_text)))

    _assert_fully_absorbed(session, memory_service, chat_service)


def test_dispatcher_photo_update_still_reaches_only_non_text_handler() -> None:
    telegram_bot, dispatcher, session, memory_service, chat_service = _build_harness()

    asyncio.run(dispatcher.feed_raw_update(telegram_bot, _make_update(photo=True)))

    assert session.sent_messages == [{"chat_id": 123, "text": _NON_TEXT_TEXT}]
    assert memory_service.recall_calls == []
    assert memory_service.remember_calls == []
    assert memory_service.get_memory_count_calls == []
    assert memory_service.forget_user_calls == []
    assert chat_service.generate_reply_calls == []


def test_dispatcher_ordinary_text_still_reaches_message_orchestration() -> None:
    telegram_bot, dispatcher, session, memory_service, chat_service = _build_harness()
    chat_service.response = "a generated reply"

    asyncio.run(dispatcher.feed_raw_update(telegram_bot, _make_update(text="hello there")))

    assert session.sent_messages == [{"chat_id": 123, "text": "a generated reply"}]
    assert memory_service.recall_calls == [
        {"user_id": 123, "query": "hello there", "top_k": None}
    ]
    assert len(memory_service.remember_calls) == 1
    assert chat_service.generate_reply_calls[0]["user_text"] == "hello there"


def test_dispatcher_slash_later_in_text_remains_ordinary_text() -> None:
    telegram_bot, dispatcher, session, memory_service, chat_service = _build_harness()
    chat_service.response = "a generated reply"
    text = "расскажи про /help"

    asyncio.run(dispatcher.feed_raw_update(telegram_bot, _make_update(text=text)))

    assert session.sent_messages == [{"chat_id": 123, "text": "a generated reply"}]
    assert memory_service.recall_calls == [{"user_id": 123, "query": text, "top_k": None}]
    assert len(memory_service.remember_calls) == 1
    assert chat_service.generate_reply_calls[0]["user_text"] == text


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def test_split_short_reply_returns_one_unchanged_chunk() -> None:
    text = "short reply"

    assert bot_module.split_telegram_text(text) == [text]


def test_split_exact_limit_returns_one_chunk() -> None:
    text = "a" * 4096

    assert bot_module.split_telegram_text(text) == [text]


def test_split_ascii_text_above_limit_is_split_safely() -> None:
    text = "a" * 5000

    chunks = bot_module.split_telegram_text(text)

    assert len(chunks) == 2
    assert len(chunks[0]) == 4096
    assert len(chunks[1]) == 904


def test_split_counts_non_bmp_characters_in_utf16_units() -> None:
    emoji = "\U0001f600"  # single Python character, 2 UTF-16 code units
    text = emoji * 3000

    chunks = bot_module.split_telegram_text(text)

    assert len(chunks[0]) == 2048  # 2048 * 2 units == 4096, the limit
    for chunk in chunks:
        assert len(chunk.encode("utf-16-le")) // 2 <= 4096


def test_split_emoji_with_limit_one_raises() -> None:
    # A single non-BMP character occupies 2 UTF-16 code units and can never
    # fit in a limit of 1 without splitting it into an invalid surrogate
    # half, so this must raise rather than silently emit an over-limit
    # chunk (the Stage 5B.1 regression case).
    with pytest.raises(ValueError):
        bot_module.split_telegram_text("\U0001f600", limit=1)


def test_split_emoji_with_limit_two_succeeds() -> None:
    emoji = "\U0001f600"

    assert bot_module.split_telegram_text(emoji, limit=2) == [emoji]


def test_split_mixed_bmp_and_non_bmp_text_at_tight_limit() -> None:
    emoji = "\U0001f600"

    assert bot_module.split_telegram_text(f"A{emoji}B", limit=2) == ["A", emoji, "B"]
    assert bot_module.split_telegram_text(emoji * 2, limit=3) == [emoji, emoji]


def test_split_makes_progress_and_terminates_for_long_mixed_text() -> None:
    # Regression guard: a splitter that fails to make progress on some
    # character could loop indefinitely instead of returning. This mixes
    # BMP and non-BMP runs at a limit that forces many chunk boundaries.
    text = ("a" * 50) + ("\U0001f600" * 50) + ("b" * 50)

    chunks = bot_module.split_telegram_text(text, limit=7)

    assert chunks  # terminated and produced output
    assert "".join(chunks) == text
    assert all(chunk for chunk in chunks)
    for chunk in chunks:
        utf16_units = len(chunk.encode("utf-16-le")) // 2
        assert utf16_units <= 7


def test_split_concatenation_reproduces_original_text_exactly() -> None:
    text = ("a" * 3000) + ("\U0001f600" * 1000) + ("b" * 3000)

    chunks = bot_module.split_telegram_text(text)

    assert "".join(chunks) == text


def test_split_preserves_original_order() -> None:
    text = ("1" * 3000) + ("2" * 3000)

    chunks = bot_module.split_telegram_text(text)

    assert "".join(chunks) == text
    assert chunks[0].startswith("1")
    assert chunks[-1].endswith("2")


def test_split_no_chunk_exceeds_limit() -> None:
    text = ("a" * 3000) + ("\U0001f600" * 1000) + ("b" * 3000)

    chunks = bot_module.split_telegram_text(text)

    for chunk in chunks:
        utf16_units = len(chunk.encode("utf-16-le")) // 2
        assert utf16_units <= 4096


def test_split_no_empty_chunks_produced() -> None:
    text = "a" * 10000

    chunks = bot_module.split_telegram_text(text)

    assert all(chunk for chunk in chunks)


@pytest.mark.parametrize("invalid_text", ["", "   ", "\n\t"])
def test_split_invalid_empty_input_fails_safely(invalid_text: str) -> None:
    with pytest.raises(ValueError):
        bot_module.split_telegram_text(invalid_text)


@pytest.mark.parametrize("invalid_limit", [0, -1])
def test_split_non_positive_limit_rejected(invalid_limit: int) -> None:
    with pytest.raises(ValueError):
        bot_module.split_telegram_text("hello", limit=invalid_limit)


# ---------------------------------------------------------------------------
# Logging / privacy
# ---------------------------------------------------------------------------


def test_recall_failure_logs_only_safe_event_and_error_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    memory_service = FakeMemoryService()
    memory_service.raise_on_recall = VectorQueryError("query failed with sensitive detail XYZ")
    chat_service = FakeChatService()
    message = FakeMessage(text="hi", from_user=FakeUser())

    with caplog.at_level(logging.WARNING):
        asyncio.run(bot_module.handle_text_message(message, memory_service, chat_service))

    assert "event=recall_failed" in caplog.text
    assert "error_type=VectorQueryError" in caplog.text
    assert "sensitive detail XYZ" not in caplog.text


def test_chat_generation_failure_logs_only_safe_event_and_error_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    memory_service = FakeMemoryService()
    chat_service = FakeChatService()
    chat_service.exception = ChatServiceError("chat completion request failed")
    message = FakeMessage(text="hi", from_user=FakeUser())

    with caplog.at_level(logging.WARNING):
        asyncio.run(bot_module.handle_text_message(message, memory_service, chat_service))

    assert "event=chat_generation_failed" in caplog.text
    assert "error_type=ChatServiceError" in caplog.text


def test_send_failure_logs_only_safe_event_and_error_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    memory_service = FakeMemoryService()
    chat_service = FakeChatService()
    message = FakeMessage(text="hi", from_user=FakeUser())
    message.fail_on_answer_call = 0

    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError):
        asyncio.run(bot_module.handle_text_message(message, memory_service, chat_service))

    assert "event=send_failed" in caplog.text
    assert "error_type=RuntimeError" in caplog.text


def test_remember_failure_logs_only_safe_event_and_error_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    memory_service = FakeMemoryService()
    memory_service.raise_on_remember = VectorStorageError("upsert failed with secret XYZ")
    chat_service = FakeChatService()
    message = FakeMessage(text="hi", from_user=FakeUser())

    with caplog.at_level(logging.WARNING):
        asyncio.run(bot_module.handle_text_message(message, memory_service, chat_service))

    assert "event=remember_failed" in caplog.text
    assert "error_type=VectorStorageError" in caplog.text
    assert "secret XYZ" not in caplog.text


def test_memory_count_failure_logs_only_safe_event_and_error_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    memory_service = FakeMemoryService()
    memory_service.raise_on_get_memory_count = VectorQueryError("stats failed with secret XYZ")
    message = FakeMessage(text="/memory", from_user=FakeUser())

    with caplog.at_level(logging.WARNING):
        asyncio.run(bot_module.cmd_memory(message, memory_service))

    assert "event=memory_count_failed" in caplog.text
    assert "error_type=VectorQueryError" in caplog.text
    assert "secret XYZ" not in caplog.text


def test_forget_me_failure_logs_only_safe_event_and_error_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    memory_service = FakeMemoryService()
    memory_service.raise_on_forget_user = VectorStorageError("delete failed with secret XYZ")
    message = FakeMessage(text="/forget_me", from_user=FakeUser())

    with caplog.at_level(logging.WARNING):
        asyncio.run(bot_module.cmd_forget_me(message, memory_service))

    assert "event=forget_me_failed" in caplog.text
    assert "error_type=VectorStorageError" in caplog.text
    assert "secret XYZ" not in caplog.text


def test_logs_do_not_contain_secrets_or_raw_text(caplog: pytest.LogCaptureFixture) -> None:
    memory_service = FakeMemoryService()
    memory_service.raise_on_recall = VectorQueryError(f"query failed for key {FAKE_API_KEY}")
    memory_service.raise_on_remember = VectorStorageError("upsert failed: raw-secret-body")
    chat_service = FakeChatService()
    chat_service.response = "the generated reply text"

    secret_user_text = "my password is hunter2 and my name is Alice Wonderland"
    message = FakeMessage(
        text=secret_user_text,
        from_user=FakeUser(
            username="secret-username", first_name="Alice", last_name="Wonderland"
        ),
    )

    with caplog.at_level(logging.DEBUG):
        asyncio.run(bot_module.handle_text_message(message, memory_service, chat_service))

    log_text = caplog.text
    for leaked in (
        FAKE_TOKEN,
        FAKE_API_KEY,
        secret_user_text,
        "the generated reply text",
        "secret-username",
        "Alice",
        "Wonderland",
        "raw-secret-body",
        "query failed for key",
    ):
        assert leaked not in log_text


def test_logs_never_contain_full_message_repr(caplog: pytest.LogCaptureFixture) -> None:
    memory_service = FakeMemoryService()
    chat_service = FakeChatService()
    message = FakeMessage(text="hi", from_user=FakeUser())

    with caplog.at_level(logging.DEBUG):
        asyncio.run(bot_module.handle_text_message(message, memory_service, chat_service))

    assert "FAKE_MESSAGE_REPR_MARKER_DO_NOT_LOG" not in caplog.text
