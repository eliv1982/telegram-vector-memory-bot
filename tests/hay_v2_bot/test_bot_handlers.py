"""Offline tests for hay_v2_bot.bot.handlers."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiogram
import pytest
from aiogram.client.session.base import BaseSession
from aiogram.methods import GetMe, SendMessage
from aiogram.types import Chat as AiogramChat
from aiogram.types import Message as AiogramMessage
from aiogram.types import User as AiogramUser
from hay_v2_bot.bot import handlers, messages
from hay_v2_bot.config import DocumentProcessingSettings, DocumentRagSettings
from hay_v2_bot.models import (
    DOCX_CONTENT_TYPE,
    PDF_CONTENT_TYPE,
    DocumentAnswer,
    DocumentIngestionOutcome,
    DocumentSource,
)
from hay_v2_bot.services import (
    DocumentIngestionError,
    DocumentQuestionError,
    DocumentRagService,
    DocumentRagServiceError,
)
from hay_v2_bot.storage import PineconeDocumentStoreFactory

from telegram_vector_memory_bot.haystack_agent import HaystackAgentServiceError
from telegram_vector_memory_bot.models import (
    MemoryAction,
    MemoryReason,
    MemoryWriteResult,
    RecalledMemory,
)
from telegram_vector_memory_bot.pinecone_manager import VectorQueryError, VectorStorageError

FAKE_TOKEN = "123456:fake-injected-telegram-token-ABCDEF"
_CURRENT_BOT_USERNAME = "current_bot_test"
_CURRENT_BOT_ID = 999
UPLOAD_TIME = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)


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
        id: int = 123,
        username: str | None = "jdoe",
        first_name: str | None = "Jane",
        last_name: str | None = "Doe",
    ) -> None:
        self.id = id
        self.username = username
        self.first_name = first_name
        self.last_name = last_name


class FakeDocument:
    def __init__(
        self,
        *,
        file_name: str | None = "docuscope_smoke.pdf",
        mime_type: str | None = PDF_CONTENT_TYPE,
        file_size: int | None = 1024,
    ) -> None:
        self.file_name = file_name
        self.mime_type = mime_type
        self.file_size = file_size


class FakeTelegramBot:
    def __init__(
        self, payload: bytes = b"%PDF-1.7\ncontent", *, exception: BaseException | None = None
    ) -> None:
        self.payload = payload
        self.exception = exception
        self.download_calls: list[dict[str, object]] = []

    async def download(self, document: object, destination: Path) -> None:
        self.download_calls.append({"document": document, "destination": Path(destination)})
        if self.exception is not None:
            raise self.exception
        Path(destination).write_bytes(self.payload)


class FakeMessage:
    def __init__(
        self,
        *,
        text: str | None = None,
        document: FakeDocument | None = None,
        from_user: FakeUser | None = None,
        bot: FakeTelegramBot | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.text = text
        self.document = document
        self.from_user = from_user
        self.bot = bot or FakeTelegramBot()
        self.answer_calls: list[str] = []
        self.events = events if events is not None else []
        self.fail_on_answer_call: int | None = None
        self.answer_exception: Exception = RuntimeError("telegram send failed")

    async def answer(self, text: str) -> None:
        index = len(self.answer_calls)
        self.answer_calls.append(text)
        self.events.append(f"answer:{text}")
        if self.fail_on_answer_call is not None and index == self.fail_on_answer_call:
            raise self.answer_exception


class FakeMemoryService:
    def __init__(self) -> None:
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

    def recall(self, *, user_id: int, query: str, top_k: int | None = None) -> list[RecalledMemory]:
        self.recall_calls.append({"user_id": user_id, "query": query, "top_k": top_k})
        if self.raise_on_recall is not None:
            raise self.raise_on_recall
        return self.recall_response

    def remember(self, *, user_id: int, text: str) -> MemoryWriteResult:
        self.remember_calls.append({"user_id": user_id, "text": text})
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


class FakeReplyService:
    def __init__(self) -> None:
        self.generate_reply_calls: list[dict[str, Any]] = []
        self.response = "generated reply"
        self.exception: Exception | None = None

    async def generate_reply(self, *, user_text: str, memories: list[RecalledMemory]) -> str:
        self.generate_reply_calls.append({"user_text": user_text, "memories": list(memories)})
        if self.exception is not None:
            raise self.exception
        return self.response


class FakeDocumentRagService:
    def __init__(
        self,
        *,
        ingestion_result: DocumentIngestionOutcome | None = None,
        answer_result: DocumentAnswer | None = None,
        ingestion_exception: BaseException | None = None,
        answer_exception: BaseException | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.ingestion_result = ingestion_result or _ingestion_outcome()
        self.answer_result = answer_result or _fallback_answer()
        self.ingestion_exception = ingestion_exception
        self.answer_exception = answer_exception
        self.events = events if events is not None else []
        self.ingest_calls: list[Any] = []
        self.answer_calls: list[dict[str, Any]] = []
        self.delete_user_documents_calls: list[int] = []
        self.delete_user_documents_exception: BaseException | None = None
        self.ingest_file_existed = False
        self.ingest_file_bytes: bytes | None = None

    def ingest_and_summarize(self, request: Any) -> DocumentIngestionOutcome:
        self.events.append("ingest")
        self.ingest_calls.append(request)
        self.ingest_file_existed = request.local_path.exists()
        if self.ingest_file_existed:
            self.ingest_file_bytes = request.local_path.read_bytes()
        if self.ingestion_exception is not None:
            raise self.ingestion_exception
        return self.ingestion_result

    def answer_question(self, user_id: int, question: str) -> DocumentAnswer:
        self.answer_calls.append({"user_id": user_id, "question": question})
        if self.answer_exception is not None:
            raise self.answer_exception
        return self.answer_result

    def delete_user_documents(self, user_id: int) -> None:
        self.delete_user_documents_calls.append(user_id)
        if self.delete_user_documents_exception is not None:
            raise self.delete_user_documents_exception


class TrackingTemporaryDirectory(AbstractContextManager[str]):
    def __init__(self, path: Path, tracker: list[Path]) -> None:
        self.path = path
        self.tracker = tracker

    def __enter__(self) -> str:
        self.path.mkdir(parents=True, exist_ok=False)
        self.tracker.append(self.path)
        return str(self.path)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        shutil.rmtree(self.path, ignore_errors=True)
        return None


class FakeTelegramSession(BaseSession):
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
        raise AssertionError(f"unexpected Telegram API method: {type(method).__name__}")

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
        yield b""


def _processing_settings(**overrides: Any) -> DocumentProcessingSettings:
    defaults = {
        "max_file_bytes": 20 * 1024 * 1024,
        "max_chunks_per_document": 2000,
    }
    defaults.update(overrides)
    return DocumentProcessingSettings(_env_file=None, **defaults)


def _ingestion_outcome(
    *,
    file_name: str = "docuscope_smoke.pdf",
    content_type: str = PDF_CONTENT_TYPE,
    summary: str = "В документе описан бюджет пилота Orion.",
) -> DocumentIngestionOutcome:
    return DocumentIngestionOutcome(
        file_hash="a" * 64,
        file_name=file_name,
        content_type=content_type,
        chunk_count=1,
        documents_written=1,
        document_ids=("doc-1",),
        summary=summary,
    )


def _grounded_answer() -> DocumentAnswer:
    return DocumentAnswer(
        answer="Одобренный бюджет Orion составляет 4.2 million euros.",
        sources=(
            DocumentSource(
                document_id="doc-1",
                file_name="docuscope_smoke.pdf",
                chunk_index=0,
                page_number=1,
                score=0.99,
            ),
        ),
        used_document_count=1,
        fallback_used=False,
    )


def _fallback_answer() -> DocumentAnswer:
    return DocumentAnswer(
        answer="В загруженных документах недостаточно информации для ответа.",
        sources=(),
        used_document_count=0,
        fallback_used=True,
    )


_GROUP_CHAT_ID = -1001234567890


def _make_update(
    *,
    text: str | None = None,
    photo: bool = False,
    document: bool = False,
    user_id: int = 123,
    chat_type: str = "private",
) -> dict[str, Any]:
    chat_id = user_id if chat_type == "private" else _GROUP_CHAT_ID
    message: dict[str, Any] = {
        "message_id": 1,
        "date": int(time.time()),
        "chat": {"id": chat_id, "type": chat_type},
        "from": {"id": user_id, "is_bot": False, "first_name": "Jane"},
    }
    if text is not None:
        message["text"] = text
    if photo:
        message["photo"] = [
            {"file_id": "abc", "file_unique_id": "abc-unique", "width": 90, "height": 90}
        ]
    if document:
        message["document"] = {
            "file_id": "doc-file",
            "file_unique_id": "doc-unique",
            "file_name": "budget.pdf",
            "mime_type": PDF_CONTENT_TYPE,
            "file_size": 1024,
        }
    return {"update_id": 1, "message": message}


def _build_dispatcher_harness() -> tuple[
    aiogram.Bot,
    aiogram.Dispatcher,
    FakeTelegramSession,
    FakeMemoryService,
    FakeReplyService,
    FakeDocumentRagService,
]:
    session = FakeTelegramSession()
    telegram_bot = aiogram.Bot(token=FAKE_TOKEN, session=session)
    memory_service = FakeMemoryService()
    reply_service = FakeReplyService()
    document_rag_service = FakeDocumentRagService()
    dispatcher = handlers.create_dispatcher(
        memory_service=memory_service,
        reply_service=reply_service,
        document_rag_service=document_rag_service,
        processing_settings=_processing_settings(),
    )
    return telegram_bot, dispatcher, session, memory_service, reply_service, document_rag_service


def test_valid_pdf_is_downloaded_ingested_summarized_and_cleaned_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temp_dirs: list[Path] = []
    events: list[str] = []

    def fake_tempdir() -> TrackingTemporaryDirectory:
        path = tmp_path / f"upload-{len(temp_dirs)}"
        return TrackingTemporaryDirectory(path, temp_dirs)

    monkeypatch.setattr(handlers, "TemporaryDirectory", fake_tempdir)
    document_service = FakeDocumentRagService(events=events)
    message = FakeMessage(
        document=FakeDocument(file_name="nested/path/docuscope_smoke.pdf"),
        from_user=FakeUser(),
        events=events,
    )

    asyncio.run(
        handlers.handle_document_upload(
            message,
            document_service,
            _processing_settings(),
        )
    )

    assert message.answer_calls == [
        messages.UPLOAD_STARTED_MESSAGE,
        messages.UPLOAD_COMPLETED_MESSAGE,
        "В документе описан бюджет пилота Orion.",
    ]
    assert events[0] == f"answer:{messages.UPLOAD_STARTED_MESSAGE}"
    assert "ingest" in events
    request = document_service.ingest_calls[0]
    assert request.user_id == 123
    assert request.file_name == "docuscope_smoke.pdf"
    assert request.content_type == PDF_CONTENT_TYPE
    assert document_service.ingest_file_existed is True
    assert document_service.ingest_file_bytes == b"%PDF-1.7\ncontent"
    assert temp_dirs and not temp_dirs[0].exists()


def test_valid_docx_is_accepted() -> None:
    document_service = FakeDocumentRagService(
        ingestion_result=_ingestion_outcome(
            file_name="contract.docx",
            content_type=DOCX_CONTENT_TYPE,
            summary="В документе описан порядок эскалации.",
        )
    )
    message = FakeMessage(
        document=FakeDocument(file_name="contract.docx", mime_type=DOCX_CONTENT_TYPE),
        from_user=FakeUser(),
    )

    asyncio.run(handlers.handle_document_upload(message, document_service, _processing_settings()))

    assert document_service.ingest_calls[0].content_type == DOCX_CONTENT_TYPE
    assert message.answer_calls[-1] == "В документе описан порядок эскалации."


def test_missing_mime_with_valid_suffix_is_accepted() -> None:
    document_service = FakeDocumentRagService(
        ingestion_result=_ingestion_outcome(
            file_name="Contract.DOCX", content_type=DOCX_CONTENT_TYPE
        )
    )
    message = FakeMessage(
        document=FakeDocument(file_name="Contract.DOCX", mime_type=None),
        from_user=FakeUser(),
    )

    asyncio.run(handlers.handle_document_upload(message, document_service, _processing_settings()))

    assert document_service.ingest_calls[0].content_type == DOCX_CONTENT_TYPE
    assert document_service.ingest_calls[0].file_name == "Contract.DOCX"


def test_unsupported_suffix_is_rejected_before_download() -> None:
    bot = FakeTelegramBot()
    document_service = FakeDocumentRagService()
    message = FakeMessage(
        document=FakeDocument(file_name="notes.txt", mime_type="text/plain"),
        from_user=FakeUser(),
        bot=bot,
    )

    asyncio.run(handlers.handle_document_upload(message, document_service, _processing_settings()))

    assert message.answer_calls == [messages.UNSUPPORTED_DOCUMENT_MESSAGE]
    assert bot.download_calls == []
    assert document_service.ingest_calls == []


def test_conflicting_mime_and_suffix_is_rejected() -> None:
    bot = FakeTelegramBot()
    document_service = FakeDocumentRagService()
    message = FakeMessage(
        document=FakeDocument(file_name="budget.pdf", mime_type=DOCX_CONTENT_TYPE),
        from_user=FakeUser(),
        bot=bot,
    )

    asyncio.run(handlers.handle_document_upload(message, document_service, _processing_settings()))

    assert message.answer_calls == [messages.UNSUPPORTED_DOCUMENT_MESSAGE]
    assert bot.download_calls == []
    assert document_service.ingest_calls == []


def test_oversized_telegram_file_is_rejected_before_download() -> None:
    bot = FakeTelegramBot()
    document_service = FakeDocumentRagService()
    message = FakeMessage(
        document=FakeDocument(file_name="big.pdf", file_size=25 * 1024 * 1024),
        from_user=FakeUser(),
        bot=bot,
    )

    asyncio.run(handlers.handle_document_upload(message, document_service, _processing_settings()))

    assert message.answer_calls == [messages.format_file_too_large_message(20 * 1024 * 1024)]
    assert bot.download_calls == []
    assert document_service.ingest_calls == []


def test_progress_message_is_sent_before_processing() -> None:
    events: list[str] = []
    document_service = FakeDocumentRagService(events=events)
    message = FakeMessage(
        document=FakeDocument(),
        from_user=FakeUser(),
        events=events,
    )

    asyncio.run(handlers.handle_document_upload(message, document_service, _processing_settings()))

    assert events.index(f"answer:{messages.UPLOAD_STARTED_MESSAGE}") < events.index("ingest")


def test_completion_and_summary_are_sent_separately() -> None:
    document_service = FakeDocumentRagService()
    message = FakeMessage(document=FakeDocument(), from_user=FakeUser())

    asyncio.run(handlers.handle_document_upload(message, document_service, _processing_settings()))

    assert message.answer_calls[-2:] == [
        messages.UPLOAD_COMPLETED_MESSAGE,
        "В документе описан бюджет пилота Orion.",
    ]


def test_temporary_directory_cleanup_occurs_on_service_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temp_dirs: list[Path] = []

    def fake_tempdir() -> TrackingTemporaryDirectory:
        path = tmp_path / f"upload-{len(temp_dirs)}"
        return TrackingTemporaryDirectory(path, temp_dirs)

    monkeypatch.setattr(handlers, "TemporaryDirectory", fake_tempdir)
    document_service = FakeDocumentRagService(
        ingestion_exception=DocumentIngestionError(r"C:\secret\folder\file.pdf")
    )
    message = FakeMessage(document=FakeDocument(), from_user=FakeUser())

    asyncio.run(handlers.handle_document_upload(message, document_service, _processing_settings()))

    assert message.answer_calls[-1] == messages.PROCESSING_FAILURE_MESSAGE
    assert temp_dirs and not temp_dirs[0].exists()


def test_known_failures_produce_safe_text_without_traceback_or_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    document_service = FakeDocumentRagService(
        ingestion_exception=DocumentIngestionError(r"Traceback: C:\secret\folder\file.pdf")
    )
    message = FakeMessage(document=FakeDocument(), from_user=FakeUser())

    with caplog.at_level(logging.WARNING):
        asyncio.run(
            handlers.handle_document_upload(message, document_service, _processing_settings())
        )

    assert message.answer_calls[-1] == messages.PROCESSING_FAILURE_MESSAGE
    assert "Traceback" not in "\n".join(message.answer_calls)
    assert r"C:\secret\folder\file.pdf" not in "\n".join(message.answer_calls)
    assert "event=document_upload_failed" in caplog.text
    assert "error_type=DocumentIngestionError" in caplog.text


def test_asyncio_to_thread_is_used_for_synchronous_document_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_service = FakeDocumentRagService()
    message = FakeMessage(document=FakeDocument(), from_user=FakeUser())
    to_thread_calls: list[str] = []

    async def fake_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        to_thread_calls.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr(handlers.asyncio, "to_thread", fake_to_thread)

    asyncio.run(handlers.handle_document_upload(message, document_service, _processing_settings()))

    assert to_thread_calls == ["ingest_and_summarize"]


def test_successful_grounded_document_answer_is_returned_with_sources() -> None:
    memory_service = FakeMemoryService()
    memory_service.recall_response = [_recalled_memory()]
    reply_service = FakeReplyService()
    document_service = FakeDocumentRagService(answer_result=_grounded_answer())
    message = FakeMessage(text="Какой бюджет Orion?", from_user=FakeUser())

    asyncio.run(
        handlers.handle_text_message(
            message,
            memory_service,
            reply_service,
            document_service,
        )
    )

    assert message.answer_calls == [
        "Одобренный бюджет Orion составляет 4.2 million euros.",
        "Источники:\n• docuscope_smoke.pdf, стр. 1",
    ]
    assert reply_service.generate_reply_calls == []
    assert memory_service.recall_calls == [
        {"user_id": 123, "query": "Какой бюджет Orion?", "top_k": None}
    ]
    assert memory_service.remember_calls[0]["text"] == "Какой бюджет Orion?"


def test_fallback_document_answer_delegates_once_to_v1_agent() -> None:
    memory_service = FakeMemoryService()
    reply_service = FakeReplyService()
    document_service = FakeDocumentRagService(answer_result=_fallback_answer())
    message = FakeMessage(text="О чем документ?", from_user=FakeUser())

    asyncio.run(
        handlers.handle_text_message(message, memory_service, reply_service, document_service)
    )

    assert reply_service.generate_reply_calls == [{"user_text": "О чем документ?", "memories": []}]
    assert message.answer_calls == ["generated reply"]


def test_no_source_answer_delegates_once_to_v1_agent() -> None:
    memory_service = FakeMemoryService()
    reply_service = FakeReplyService()
    document_service = FakeDocumentRagService(
        answer_result=DocumentAnswer(
            answer="Локальный ответ без источников.",
            sources=(),
            used_document_count=0,
            fallback_used=False,
        )
    )
    message = FakeMessage(text="Что внутри?", from_user=FakeUser())

    asyncio.run(
        handlers.handle_text_message(message, memory_service, reply_service, document_service)
    )

    assert reply_service.generate_reply_calls == [{"user_text": "Что внутри?", "memories": []}]
    assert message.answer_calls == ["generated reply"]


def test_controlled_document_rag_error_delegates_once_to_v1_agent() -> None:
    memory_service = FakeMemoryService()
    reply_service = FakeReplyService()
    document_service = FakeDocumentRagService(
        answer_exception=DocumentQuestionError("Question answering failed")
    )
    message = FakeMessage(text="Что сказано про инциденты?", from_user=FakeUser())

    asyncio.run(
        handlers.handle_text_message(message, memory_service, reply_service, document_service)
    )

    assert len(reply_service.generate_reply_calls) == 1
    assert message.answer_calls == ["generated reply"]


def test_successful_document_answer_does_not_call_v1_agent() -> None:
    memory_service = FakeMemoryService()
    reply_service = FakeReplyService()
    document_service = FakeDocumentRagService(answer_result=_grounded_answer())
    message = FakeMessage(text="Бюджет?", from_user=FakeUser())

    asyncio.run(
        handlers.handle_text_message(message, memory_service, reply_service, document_service)
    )

    assert reply_service.generate_reply_calls == []


def test_v1_memory_write_behavior_remains_active_for_ordinary_text() -> None:
    memory_service = FakeMemoryService()
    reply_service = FakeReplyService()
    document_service = FakeDocumentRagService(answer_result=_fallback_answer())
    message = FakeMessage(text="Запомни, что я люблю горы.", from_user=FakeUser())

    asyncio.run(
        handlers.handle_text_message(message, memory_service, reply_service, document_service)
    )

    # FakeUser carries a username/first/last name; none of it may reach memory.
    assert memory_service.remember_calls == [{"user_id": 123, "text": "Запомни, что я люблю горы."}]


@pytest.mark.parametrize("command_text", ["/start", "/help", "/memory", "/forget_me", "/unknown"])
def test_commands_are_not_captured_by_ordinary_text_handler(command_text: str) -> None:
    message = FakeMessage(text=command_text, from_user=FakeUser())

    assert handlers.is_ordinary_text_message(message) is False


def test_document_content_is_never_written_as_v1_memory() -> None:
    memory_service = FakeMemoryService()
    document_service = FakeDocumentRagService()
    message = FakeMessage(document=FakeDocument(), from_user=FakeUser())

    asyncio.run(handlers.handle_document_upload(message, document_service, _processing_settings()))

    assert memory_service.remember_calls == []


def test_missing_from_user_is_handled_safely() -> None:
    memory_service = FakeMemoryService()
    reply_service = FakeReplyService()
    document_service = FakeDocumentRagService(answer_result=_grounded_answer())
    message = FakeMessage(text="Привет", from_user=None)

    asyncio.run(
        handlers.handle_text_message(message, memory_service, reply_service, document_service)
    )

    assert message.answer_calls == []
    assert memory_service.recall_calls == []
    assert reply_service.generate_reply_calls == []


def test_dispatcher_help_addressed_to_another_bot_is_ignored() -> None:
    telegram_bot, dispatcher, session, memory_service, reply_service, document_service = (
        _build_dispatcher_harness()
    )

    asyncio.run(
        dispatcher.feed_raw_update(
            telegram_bot,
            _make_update(text="/help@some_other_bot"),
        )
    )

    assert session.sent_messages == []
    assert memory_service.recall_calls == []
    assert reply_service.generate_reply_calls == []
    assert document_service.answer_calls == []


@pytest.mark.parametrize("malformed_text", ["/", "//help", "/foo-bar"])
def test_dispatcher_malformed_slash_text_is_silently_absorbed(malformed_text: str) -> None:
    telegram_bot, dispatcher, session, memory_service, reply_service, document_service = (
        _build_dispatcher_harness()
    )

    asyncio.run(dispatcher.feed_raw_update(telegram_bot, _make_update(text=malformed_text)))

    assert session.sent_messages == []
    assert memory_service.recall_calls == []
    assert reply_service.generate_reply_calls == []
    assert document_service.answer_calls == []


def test_non_text_message_fallback_is_preserved() -> None:
    telegram_bot, dispatcher, session, memory_service, reply_service, document_service = (
        _build_dispatcher_harness()
    )

    asyncio.run(dispatcher.feed_raw_update(telegram_bot, _make_update(photo=True)))

    assert session.sent_messages == [{"chat_id": 123, "text": messages.NON_TEXT_MESSAGE}]
    assert memory_service.recall_calls == []
    assert reply_service.generate_reply_calls == []
    assert document_service.answer_calls == []


def test_send_failure_still_prevents_memory_write() -> None:
    memory_service = FakeMemoryService()
    reply_service = FakeReplyService()
    document_service = FakeDocumentRagService(answer_result=_grounded_answer())
    message = FakeMessage(text="Бюджет?", from_user=FakeUser())
    message.fail_on_answer_call = 1

    with pytest.raises(RuntimeError):
        asyncio.run(
            handlers.handle_text_message(message, memory_service, reply_service, document_service)
        )

    assert memory_service.remember_calls == []


def test_fallback_chat_failure_returns_safe_message() -> None:
    memory_service = FakeMemoryService()
    reply_service = FakeReplyService()
    reply_service.exception = HaystackAgentServiceError("secret body")
    document_service = FakeDocumentRagService(answer_result=_fallback_answer())
    message = FakeMessage(text="Привет", from_user=FakeUser())

    asyncio.run(
        handlers.handle_text_message(message, memory_service, reply_service, document_service)
    )

    assert message.answer_calls == [messages.CHAT_FAILURE_MESSAGE]


def test_recall_failure_does_not_prevent_document_answer() -> None:
    memory_service = FakeMemoryService()
    memory_service.raise_on_recall = VectorQueryError("query failed")
    reply_service = FakeReplyService()
    document_service = FakeDocumentRagService(answer_result=_grounded_answer())
    message = FakeMessage(text="Бюджет?", from_user=FakeUser())

    asyncio.run(
        handlers.handle_text_message(message, memory_service, reply_service, document_service)
    )

    assert message.answer_calls[0] == "Одобренный бюджет Orion составляет 4.2 million euros."
    assert reply_service.generate_reply_calls == []


def test_remember_failure_is_logged_safely(
    caplog: pytest.LogCaptureFixture,
) -> None:
    memory_service = FakeMemoryService()
    memory_service.raise_on_remember = VectorStorageError("upsert failed with secret XYZ")
    reply_service = FakeReplyService()
    document_service = FakeDocumentRagService(answer_result=_grounded_answer())
    message = FakeMessage(text="Бюджет?", from_user=FakeUser())

    with caplog.at_level(logging.WARNING):
        asyncio.run(
            handlers.handle_text_message(message, memory_service, reply_service, document_service)
        )

    assert "event=remember_failed" in caplog.text
    assert "secret XYZ" not in caplog.text


# ---------------------------------------------------------------------------
# /forget_me deletes BOTH user namespaces
# ---------------------------------------------------------------------------


def _run_forget_me(
    memory_service: FakeMemoryService,
    document_service: FakeDocumentRagService,
) -> FakeMessage:
    message = FakeMessage(text="/forget_me", from_user=FakeUser(id=123))
    asyncio.run(handlers.cmd_forget_me(message, memory_service, document_service))
    return message


def test_forget_me_deletes_both_namespaces_and_reports_full_success() -> None:
    memory_service = FakeMemoryService()
    document_service = FakeDocumentRagService()

    message = _run_forget_me(memory_service, document_service)

    assert memory_service.forget_user_calls == [{"user_id": 123}]
    assert document_service.delete_user_documents_calls == [123]
    assert message.answer_calls == [messages.FORGET_ME_SUCCESS_MESSAGE]


@pytest.mark.parametrize(
    ("memory_exception", "documents_exception"),
    [
        pytest.param(None, DocumentRagServiceError("Document cleanup failed"), id="docs-fail"),
        pytest.param(VectorStorageError("delete failed"), None, id="memory-fails"),
        pytest.param(
            VectorStorageError("delete failed"),
            DocumentRagServiceError("Document cleanup failed"),
            id="both-fail",
        ),
    ],
)
def test_forget_me_never_reports_success_when_either_deletion_fails(
    memory_exception: Exception | None, documents_exception: Exception | None
) -> None:
    memory_service = FakeMemoryService()
    memory_service.raise_on_forget_user = memory_exception
    document_service = FakeDocumentRagService()
    document_service.delete_user_documents_exception = documents_exception

    message = _run_forget_me(memory_service, document_service)

    assert message.answer_calls == [messages.FORGET_ME_FAILURE_MESSAGE]
    assert messages.FORGET_ME_SUCCESS_MESSAGE not in message.answer_calls


@pytest.mark.parametrize(
    ("memory_exception", "documents_exception"),
    [
        pytest.param(None, DocumentRagServiceError("x"), id="docs-fail"),
        pytest.param(VectorStorageError("x"), None, id="memory-fails"),
        pytest.param(VectorStorageError("x"), DocumentRagServiceError("x"), id="both-fail"),
        pytest.param(RuntimeError("unexpected"), None, id="memory-unexpected-error"),
        pytest.param(None, RuntimeError("unexpected"), id="docs-unexpected-error"),
    ],
)
def test_forget_me_attempts_both_deletions_even_when_one_fails(
    memory_exception: Exception | None, documents_exception: Exception | None
) -> None:
    memory_service = FakeMemoryService()
    memory_service.raise_on_forget_user = memory_exception
    document_service = FakeDocumentRagService()
    document_service.delete_user_documents_exception = documents_exception

    _run_forget_me(memory_service, document_service)

    assert memory_service.forget_user_calls == [{"user_id": 123}]
    assert document_service.delete_user_documents_calls == [123]


def test_forget_me_failure_reply_and_logs_expose_no_error_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "ns=telegram-documents-user-123 key=sk-SECRET host=pinecone.internal"
    memory_service = FakeMemoryService()
    memory_service.raise_on_forget_user = VectorStorageError(f"memory delete failed: {secret}")
    document_service = FakeDocumentRagService()
    document_service.delete_user_documents_exception = DocumentRagServiceError(
        f"docs delete failed: {secret}"
    )

    with caplog.at_level(logging.DEBUG):
        message = _run_forget_me(memory_service, document_service)

    assert message.answer_calls == [messages.FORGET_ME_FAILURE_MESSAGE]
    sent = "\n".join(message.answer_calls)
    for leaked in ("telegram-documents-user", "sk-SECRET", "pinecone.internal", "123"):
        assert leaked not in sent
    for leaked in ("telegram-documents-user", "sk-SECRET", "pinecone.internal", "delete failed"):
        assert leaked not in caplog.text
    assert "event=forget_me_memory_failed" in caplog.text
    assert "error_type=VectorStorageError" in caplog.text
    assert "event=forget_me_documents_failed" in caplog.text
    assert "error_type=DocumentRagServiceError" in caplog.text


def test_forget_me_failure_message_is_fixed_and_distinct_from_success() -> None:
    assert messages.FORGET_ME_FAILURE_MESSAGE != messages.FORGET_ME_SUCCESS_MESSAGE
    assert "/forget_me" in messages.FORGET_ME_FAILURE_MESSAGE
    assert not any(ch.isdigit() for ch in messages.FORGET_ME_FAILURE_MESSAGE)


def test_forget_me_without_from_user_calls_nothing() -> None:
    memory_service = FakeMemoryService()
    document_service = FakeDocumentRagService()
    message = FakeMessage(text="/forget_me", from_user=None)

    asyncio.run(handlers.cmd_forget_me(message, memory_service, document_service))

    assert message.answer_calls == []
    assert memory_service.forget_user_calls == []
    assert document_service.delete_user_documents_calls == []


def test_forget_me_runs_both_blocking_deletions_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    to_thread_calls: list[str] = []

    async def fake_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        to_thread_calls.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr(handlers.asyncio, "to_thread", fake_to_thread)

    _run_forget_me(FakeMemoryService(), FakeDocumentRagService())

    assert to_thread_calls == ["forget_user", "delete_user_documents"]


class _AlreadyMissing(Exception):
    status_code = 404


class _MissingNamespaceIndex:
    def __init__(self) -> None:
        self.delete_calls: list[dict[str, Any]] = []

    def delete(self, **kwargs: Any) -> None:
        self.delete_calls.append(kwargs)
        raise _AlreadyMissing("namespace not found")


class _MissingNamespaceClient:
    def __init__(self, index: _MissingNamespaceIndex) -> None:
        self._index = index

    def describe_index(self, name: str) -> dict[str, Any]:
        return {
            "name": name,
            "host": "https://pinecone.invalid/index",
            "dimension": 1536,
            "metric": "cosine",
            "status": {"ready": True, "state": "Ready"},
        }

    def Index(self, *, host: str = "", **_: Any) -> _MissingNamespaceIndex:
        return self._index


def test_forget_me_is_idempotent_when_document_namespace_is_already_missing() -> None:
    # Real service + real store factory over a fake Pinecone that reports not-found.
    index = _MissingNamespaceIndex()
    rag_settings = DocumentRagSettings(
        _env_file=None,
        PINECONE_API_KEY="pinecone-key",
        PINECONE_INDEX_NAME="document-index",
        OPENAI_API_KEY="openai-key",
        OPENAI_BASE_URL="https://example.invalid/v1",
        OPENAI_EMBEDDING_MODEL="embedding-model",
        OPENAI_CHAT_MODEL="chat-model",
    )
    factory = PineconeDocumentStoreFactory(
        rag_settings, pinecone_client=_MissingNamespaceClient(index)
    )
    document_service = DocumentRagService(
        _processing_settings(), rag_settings, adapter=object(), document_store_factory=factory
    )
    memory_service = FakeMemoryService()  # v1 manager already treats a missing namespace as done

    message = FakeMessage(text="/forget_me", from_user=FakeUser(id=123))
    asyncio.run(handlers.cmd_forget_me(message, memory_service, document_service))

    assert index.delete_calls == [{"delete_all": True, "namespace": "telegram-documents-user-123"}]
    assert message.answer_calls == [messages.FORGET_ME_SUCCESS_MESSAGE]


def test_dispatcher_forget_me_in_private_chat_deletes_both_namespaces() -> None:
    telegram_bot, dispatcher, session, memory_service, reply_service, document_service = (
        _build_dispatcher_harness()
    )

    asyncio.run(dispatcher.feed_raw_update(telegram_bot, _make_update(text="/forget_me")))

    assert memory_service.forget_user_calls == [{"user_id": 123}]
    assert document_service.delete_user_documents_calls == [123]
    assert session.sent_messages == [{"chat_id": 123, "text": messages.FORGET_ME_SUCCESS_MESSAGE}]


# ---------------------------------------------------------------------------
# Private-chat-only routing (groups would expose a sender's private context)
# ---------------------------------------------------------------------------

_NON_PRIVATE_CHAT_TYPES = ["group", "supergroup", "channel"]


def _assert_no_handler_or_service_activity(
    session: FakeTelegramSession,
    memory_service: FakeMemoryService,
    reply_service: FakeReplyService,
    document_service: FakeDocumentRagService,
) -> None:
    assert session.sent_messages == []
    assert memory_service.recall_calls == []
    assert memory_service.remember_calls == []
    assert memory_service.forget_user_calls == []
    assert memory_service.get_memory_count_calls == []
    assert reply_service.generate_reply_calls == []
    assert document_service.answer_calls == []
    assert document_service.ingest_calls == []
    assert document_service.delete_user_documents_calls == []


@pytest.mark.parametrize("chat_type", _NON_PRIVATE_CHAT_TYPES)
@pytest.mark.parametrize(
    "text",
    [
        "Какой бюджет Orion?",
        "/start",
        "/help",
        "/memory",
        "/forget_me",
        "/unknown",
        f"/help@{_CURRENT_BOT_USERNAME}",
        f"/memory@{_CURRENT_BOT_USERNAME}",
        f"/forget_me@{_CURRENT_BOT_USERNAME}",
        "расскажи про /help",
    ],
)
def test_dispatcher_non_private_text_and_commands_reach_no_handler_or_service(
    chat_type: str, text: str
) -> None:
    telegram_bot, dispatcher, session, memory_service, reply_service, document_service = (
        _build_dispatcher_harness()
    )
    document_service.answer_result = _grounded_answer()  # would be served if routing leaked

    asyncio.run(
        dispatcher.feed_raw_update(telegram_bot, _make_update(text=text, chat_type=chat_type))
    )

    _assert_no_handler_or_service_activity(session, memory_service, reply_service, document_service)


@pytest.mark.parametrize("chat_type", _NON_PRIVATE_CHAT_TYPES)
def test_dispatcher_non_private_non_text_message_reaches_no_handler(chat_type: str) -> None:
    telegram_bot, dispatcher, session, memory_service, reply_service, document_service = (
        _build_dispatcher_harness()
    )

    asyncio.run(
        dispatcher.feed_raw_update(telegram_bot, _make_update(photo=True, chat_type=chat_type))
    )

    _assert_no_handler_or_service_activity(session, memory_service, reply_service, document_service)


@pytest.mark.parametrize("chat_type", _NON_PRIVATE_CHAT_TYPES)
def test_dispatcher_non_private_document_upload_reaches_no_handler_or_service(
    chat_type: str,
) -> None:
    # An upload in a group must not be downloaded, ingested, or stored under the sender.
    telegram_bot, dispatcher, session, memory_service, reply_service, document_service = (
        _build_dispatcher_harness()
    )

    asyncio.run(
        dispatcher.feed_raw_update(telegram_bot, _make_update(document=True, chat_type=chat_type))
    )

    _assert_no_handler_or_service_activity(session, memory_service, reply_service, document_service)


def test_dispatcher_private_chat_is_still_served_after_group_rejection() -> None:
    telegram_bot, dispatcher, session, memory_service, reply_service, document_service = (
        _build_dispatcher_harness()
    )

    asyncio.run(
        dispatcher.feed_raw_update(telegram_bot, _make_update(text="hi", chat_type="supergroup"))
    )
    asyncio.run(dispatcher.feed_raw_update(telegram_bot, _make_update(text="hi")))

    assert session.sent_messages == [{"chat_id": 123, "text": "generated reply"}]
    assert memory_service.recall_calls == [{"user_id": 123, "query": "hi", "top_k": None}]
    assert document_service.answer_calls == [{"user_id": 123, "question": "hi"}]
    assert len(memory_service.remember_calls) == 1
