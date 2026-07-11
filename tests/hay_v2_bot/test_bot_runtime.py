"""Offline tests for hay_v2_bot.bot.runtime and entry-point wiring."""

from __future__ import annotations

import asyncio
import importlib
import runpy
import sys
from typing import Any

import aiogram
import openai
import pinecone
import pytest
from hay_v2_bot.bot import handlers as handlers_module
from hay_v2_bot.bot import runtime as runtime_module
from hay_v2_bot.config import DocumentProcessingSettings, DocumentRagSettings
from hay_v2_bot.storage import document_namespace_for_user

from telegram_vector_memory_bot.config import Settings

FAKE_TOKEN = "123456:fake-injected-telegram-token-ABCDEF"
FAKE_API_KEY = "sk-FAKE-INJECTED-SECRET-VALUE"


def _settings(**overrides: Any) -> Settings:
    data: dict[str, Any] = {
        "PINECONE_API_KEY": FAKE_API_KEY,
        "PINECONE_INDEX_NAME": "test-index",
        "OPENAI_API_KEY": FAKE_API_KEY,
        "OPENAI_CHAT_MODEL": "gpt-4o-mini",
        "TELEGRAM_BOT_TOKEN": FAKE_TOKEN,
    }
    data.update(overrides)
    return Settings(_env_file=None, **data)


def _processing_settings(**overrides: Any) -> DocumentProcessingSettings:
    data = {"max_file_bytes": 20 * 1024 * 1024, "max_chunks_per_document": 2000}
    data.update(overrides)
    return DocumentProcessingSettings(_env_file=None, **data)


def _rag_settings(**overrides: Any) -> DocumentRagSettings:
    data: dict[str, Any] = {
        "PINECONE_API_KEY": "pinecone-key",
        "PINECONE_INDEX_NAME": "test-index",
        "OPENAI_API_KEY": "openai-key",
        "OPENAI_CHAT_MODEL": "gpt-4o-mini",
    }
    data.update(overrides)
    return DocumentRagSettings(_env_file=None, **data)


def test_runtime_construction_uses_existing_telegram_token_alias() -> None:
    constructed_bots: list[dict[str, Any]] = []

    class RecordingBot:
        def __init__(self, **kwargs: Any) -> None:
            constructed_bots.append(kwargs)

    runtime = runtime_module.create_runtime(
        settings_factory=lambda: _settings(),
        processing_settings_factory=lambda: _processing_settings(),
        rag_settings_factory=lambda: _rag_settings(),
        pinecone_manager_factory=lambda settings: "manager",
        memory_service_factory=lambda **kwargs: "memory",
        reply_service_factory=lambda **kwargs: "reply",
        document_rag_service_factory=lambda **kwargs: "documents",
        bot_factory=RecordingBot,
        dispatcher_factory=lambda **kwargs: "dispatcher",
    )

    assert constructed_bots == [{"token": FAKE_TOKEN}]
    assert runtime.settings.TELEGRAM_BOT_TOKEN.get_secret_value() == FAKE_TOKEN


def test_importing_runtime_module_constructs_no_external_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("must not be constructed during import")

    monkeypatch.setattr(pinecone.Pinecone, "__init__", explode)
    monkeypatch.setattr(openai.OpenAI, "__init__", explode)
    monkeypatch.setattr(aiogram.Bot, "__init__", explode)
    monkeypatch.setattr(Settings, "__init__", explode)
    monkeypatch.setattr(DocumentProcessingSettings, "__init__", explode)
    monkeypatch.setattr(DocumentRagSettings, "__init__", explode)

    reloaded = importlib.reload(runtime_module)

    assert reloaded.create_runtime is not None


def test_document_and_memory_services_receive_separate_namespace_policies() -> None:
    runtime = runtime_module.create_runtime(
        settings_factory=lambda: _settings(MEMORY_NAMESPACE_PREFIX="telegram-user"),
        processing_settings_factory=lambda: _processing_settings(),
        rag_settings_factory=lambda: _rag_settings(),
        pinecone_manager_factory=lambda settings: "manager",
        memory_service_factory=lambda **kwargs: type(
            "FakeMemoryService",
            (),
            {
                "namespace_for_user": lambda self, user_id: (
                    f"{kwargs['settings'].MEMORY_NAMESPACE_PREFIX}-{user_id}"
                )
            },
        )(),
        reply_service_factory=lambda **kwargs: "reply",
        document_rag_service_factory=lambda **kwargs: "documents",
        bot_factory=lambda **kwargs: "bot",
        dispatcher_factory=lambda **kwargs: "dispatcher",
    )

    assert runtime.memory_service.namespace_for_user(7) == "telegram-user-7"
    assert document_namespace_for_user(7) == "telegram-documents-user-7"


def test_create_runtime_is_testable_with_injected_factories() -> None:
    calls: list[str] = []

    def fake_pinecone_manager(settings: Settings) -> str:
        calls.append("PineconeManager")
        return "manager"

    def fake_memory_service(*, manager: Any, settings: Settings) -> str:
        calls.append("MemoryService")
        assert manager == "manager"
        return "memory"

    def fake_reply_service(*, settings: Settings) -> str:
        calls.append("HaystackAgentService")
        return "reply"

    def fake_document_service(*, processing_settings: Any, rag_settings: Any) -> str:
        calls.append("DocumentRagService")
        return "documents"

    def fake_bot(**kwargs: Any) -> str:
        calls.append("Bot")
        return "bot"

    def fake_dispatcher(**kwargs: Any) -> str:
        calls.append("Dispatcher")
        return "dispatcher"

    runtime = runtime_module.create_runtime(
        settings_factory=lambda: _settings(),
        processing_settings_factory=lambda: _processing_settings(),
        rag_settings_factory=lambda: _rag_settings(),
        pinecone_manager_factory=fake_pinecone_manager,
        memory_service_factory=fake_memory_service,
        reply_service_factory=fake_reply_service,
        document_rag_service_factory=fake_document_service,
        bot_factory=fake_bot,
        dispatcher_factory=fake_dispatcher,
    )

    assert calls == [
        "PineconeManager",
        "MemoryService",
        "HaystackAgentService",
        "DocumentRagService",
        "Bot",
        "Dispatcher",
    ]
    assert runtime.dispatcher == "dispatcher"


def test_handlers_register_once() -> None:
    dispatcher = aiogram.Dispatcher()

    handlers_module.register_handlers(dispatcher)
    handlers_module.register_handlers(dispatcher)

    matching = [router for router in dispatcher.sub_routers if router.name == "hay_v2_bot_router"]
    assert len(matching) == 1


def test_python_module_entry_points_delegate_to_same_main_function() -> None:
    main_module = importlib.import_module("hay_v2_bot.main")
    package_main_module = importlib.import_module("hay_v2_bot.__main__")

    assert main_module.main is runtime_module.main
    assert package_main_module.main is runtime_module.main


def test_no_polling_starts_merely_by_importing_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[str] = []

    async def fake_run_bot(*args: Any, **kwargs: Any) -> None:
        started.append("run_bot")

    monkeypatch.setattr(runtime_module, "run_bot", fake_run_bot)

    importlib.reload(importlib.import_module("hay_v2_bot.main"))
    importlib.reload(importlib.import_module("hay_v2_bot.__main__"))

    assert started == []


def test_run_bot_uses_runtime_factory_and_starts_polling() -> None:
    polling_calls: list[Any] = []

    class RecordingDispatcher:
        async def start_polling(self, bot: Any) -> None:
            polling_calls.append(bot)

    runtime = runtime_module.BotRuntime(
        settings=_settings(),
        processing_settings=_processing_settings(),
        rag_settings=_rag_settings(),
        pinecone_manager="manager",  # type: ignore[arg-type]
        memory_service="memory",  # type: ignore[arg-type]
        reply_service="reply",  # type: ignore[arg-type]
        document_rag_service="documents",  # type: ignore[arg-type]
        bot="bot",  # type: ignore[arg-type]
        dispatcher=RecordingDispatcher(),  # type: ignore[arg-type]
    )

    asyncio.run(runtime_module.run_bot(runtime_factory=lambda **kwargs: runtime))

    assert polling_calls == ["bot"]


def test_module_execution_paths_share_same_runtime_main(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_main() -> None:
        calls.append("main")

    monkeypatch.setattr(runtime_module, "main", fake_main)
    sys.modules.pop("hay_v2_bot.main", None)
    sys.modules.pop("hay_v2_bot.__main__", None)

    runpy.run_module("hay_v2_bot.main", run_name="__main__")
    runpy.run_module("hay_v2_bot.__main__", run_name="__main__")

    assert calls == ["main", "main"]
