"""Runtime factory and startup entry points for the v2 Telegram bot."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aiogram import Bot, Dispatcher

from hay_v2_bot.bot.handlers import create_dispatcher
from hay_v2_bot.config import DocumentProcessingSettings, DocumentRagSettings
from hay_v2_bot.services import DocumentRagService
from telegram_vector_memory_bot.config import Settings, get_settings
from telegram_vector_memory_bot.haystack_agent import HaystackAgentService
from telegram_vector_memory_bot.memory_service import MemoryService
from telegram_vector_memory_bot.pinecone_manager import PineconeManager


@dataclass(frozen=True)
class BotRuntime:
    """Fully constructed runtime objects for the v2 Telegram bot."""

    settings: Settings
    processing_settings: DocumentProcessingSettings
    rag_settings: DocumentRagSettings
    pinecone_manager: PineconeManager
    memory_service: MemoryService
    reply_service: HaystackAgentService
    document_rag_service: DocumentRagService
    bot: Bot
    dispatcher: Dispatcher


def create_runtime(
    *,
    settings_factory: Callable[[], Settings] = get_settings,
    processing_settings_factory: Callable[
        [], DocumentProcessingSettings
    ] = DocumentProcessingSettings,
    rag_settings_factory: Callable[[], DocumentRagSettings] = DocumentRagSettings,
    pinecone_manager_factory: Callable[[Settings], PineconeManager] = PineconeManager,
    memory_service_factory: Callable[..., MemoryService] = MemoryService,
    reply_service_factory: Callable[..., HaystackAgentService] = HaystackAgentService,
    document_rag_service_factory: Callable[..., DocumentRagService] | None = None,
    bot_factory: Callable[..., Bot] = Bot,
    dispatcher_factory: Callable[..., Dispatcher] = create_dispatcher,
) -> BotRuntime:
    """Construct every runtime dependency without starting polling."""
    settings = settings_factory()
    pinecone_manager = pinecone_manager_factory(settings)
    memory_service = memory_service_factory(manager=pinecone_manager, settings=settings)
    reply_service = reply_service_factory(settings=settings)
    processing_settings = processing_settings_factory()
    rag_settings = rag_settings_factory()

    if document_rag_service_factory is None:
        document_rag_service = DocumentRagService(processing_settings, rag_settings)
    else:
        document_rag_service = document_rag_service_factory(
            processing_settings=processing_settings,
            rag_settings=rag_settings,
        )

    bot = bot_factory(token=settings.TELEGRAM_BOT_TOKEN.get_secret_value())
    dispatcher = dispatcher_factory(
        memory_service=memory_service,
        reply_service=reply_service,
        document_rag_service=document_rag_service,
        processing_settings=processing_settings,
    )

    return BotRuntime(
        settings=settings,
        processing_settings=processing_settings,
        rag_settings=rag_settings,
        pinecone_manager=pinecone_manager,
        memory_service=memory_service,
        reply_service=reply_service,
        document_rag_service=document_rag_service,
        bot=bot,
        dispatcher=dispatcher,
    )


async def run_bot(
    *,
    runtime_factory: Callable[..., BotRuntime] = create_runtime,
    runtime_factory_kwargs: dict[str, Any] | None = None,
) -> None:
    """Create the v2 runtime and start aiogram long polling."""
    runtime = runtime_factory(**(runtime_factory_kwargs or {}))
    await runtime.dispatcher.start_polling(runtime.bot)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(run_bot())


__all__ = ["BotRuntime", "create_runtime", "main", "run_bot"]
