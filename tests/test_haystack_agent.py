"""Модульные тесты для telegram_vector_memory_bot.haystack_agent.

Ни один реальный клиент OpenAI и ни один реальный HTTP-вызов никогда не
выполняются. Chat model заменена на ``FakeChatGenerator`` -- минимальный
Haystack ``@component``, возвращающий заскриптованные ответы ``ChatMessage``
-- включая tool-call сообщения -- так что реальный
``haystack.components.agents.Agent`` можно прогнать целиком (выбор
инструмента, вызов инструмента, извлечение итогового ответа) полностью
offline. Используются настоящие объекты ``Tool`` из ``tools.py``,
оборачивающие фейковые функции, так что сетевого I/O не происходит.

Тесты выбора инструмента используют ``PromptRoutingFakeChatGenerator``,
который читает реальный текст входящего сообщения пользователя и выбирает
свой заскриптованный tool call на основании этого содержимого -- со всеми
пятью реальными инструментами, зарегистрированными на одном и том же
``Agent`` одновременно. Это намеренно строже, чем генератор, который просто
заскриптован всегда возвращать один и тот же tool call независимо от
промпта (это доказывало бы только то, что работает механика вызова, а не
то, что для данного вопроса действительно был "выбран" нужный инструмент):
здесь другой вопрос реально приводит к другому tool call, а неверное
совпадение по ключевому слову вызвало бы не тот инструмент и провалило бы
проверки того, какая фейковая функция инструмента сработала.

Асинхронные методы прогоняются напрямую через ``asyncio.run``, а не через
плагин pytest-asyncio, поскольку он не является зависимостью проекта.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime
from typing import Any

import pytest
from haystack import component
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool

from telegram_vector_memory_bot.config import Settings
from telegram_vector_memory_bot.haystack_agent import (
    _SYSTEM_PROMPT,
    HaystackAgentService,
    HaystackAgentServiceError,
    build_context_message,
)
from telegram_vector_memory_bot.models import RecalledMemory

FAKE_API_KEY = "sk-FAKE-INJECTED-SECRET-VALUE"


def _build_settings(**overrides: Any) -> Settings:
    data: dict[str, Any] = {
        "PINECONE_API_KEY": "test-pinecone-key",
        "PINECONE_INDEX_NAME": "test-index",
        "OPENAI_API_KEY": FAKE_API_KEY,
        "OPENAI_CHAT_MODEL": "gpt-4o-mini",
        "TELEGRAM_BOT_TOKEN": "test-telegram-token",
    }
    data.update(overrides)
    return Settings(_env_file=None, **data)


def _memory(
    *,
    memory_id: str = "mem-1",
    text: str = "Я предпочитаю короткие ответы.",
    score: float = 0.9,
) -> RecalledMemory:
    return RecalledMemory(
        memory_id=memory_id,
        text=text,
        score=score,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        source="telegram",
        content_hash="abc123",
    )


@component
class FakeChatGenerator:
    """Минимальный компонент chat-generator Haystack с заскриптованными ответами.

    Каждый вызов ``run_async`` (или ``run``) возвращает следующий
    заскриптованный словарь ``{"replies": [...]}``, позволяя реальному
    ``Agent`` провести полный цикл tool-call без какого-либо сетевого
    доступа.
    """

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.run_calls: list[dict[str, Any]] = []

    @component.output_types(replies=list)
    def run(self, messages: list[ChatMessage], tools: Any = None, **kwargs: Any) -> dict[str, Any]:
        self.run_calls.append({"messages": messages, "tools": tools})
        return self._responses.pop(0)

    @component.output_types(replies=list)
    async def run_async(
        self, messages: list[ChatMessage], tools: Any = None, **kwargs: Any
    ) -> dict[str, Any]:
        return self.run(messages, tools=tools, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "FakeChatGenerator", "data": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FakeChatGenerator:
        return cls([])


def _tool_call_reply(
    tool_name: str, arguments: dict[str, Any], call_id: str = "call_1"
) -> dict[str, Any]:
    message = ChatMessage.from_assistant(
        text=None, tool_calls=[ToolCall(id=call_id, tool_name=tool_name, arguments=arguments)]
    )
    return {"replies": [message]}


def _final_reply(text: str) -> dict[str, Any]:
    return {"replies": [ChatMessage.from_assistant(text)]}


# (подстрока-триггер, имя инструмента, аргументы инструмента, закрывающий
# ответ после результата инструмента). Сопоставляется без учёта регистра с
# реальным текстом сообщения пользователя -- см. PromptRoutingFakeChatGenerator
# ниже.
_TOOL_SELECTION_TRIGGERS: list[tuple[str, str, dict[str, Any], str]] = [
    (
        "погода",
        "get_current_weather",
        {"city": "Helsinki"},
        "Сейчас в Хельсинки 5°C, ветер 10 км/ч.",
    ),
    (
        "евро в долларах",
        "convert_currency",
        {"amount": 100, "from_currency": "EUR", "to_currency": "USD"},
        "100 евро — это 110 долларов.",
    ),
    (
        "который час",
        "get_current_time",
        {"location": "Tokyo"},
        "Сейчас в Токио 21:15, понедельник.",
    ),
    (
        "праздники в дании",
        "get_public_holidays",
        {"country": "Дания", "year": 2026},
        "1 января 2026 — Новый год в Дании.",
    ),
    (
        "последняя версия haystack-ai",
        "get_pypi_package_info",
        {"package_name": "haystack-ai"},
        "Последняя версия haystack-ai — 2.4.0.",
    ),
]


@component
class PromptRoutingFakeChatGenerator:
    """Выбирает заскриптованный tool call (или обычный ответ) из текста промпта пользователя.

    При первом вызове проверяет последнее сообщение пользователя и
    сопоставляет его с ``_TOOL_SELECTION_TRIGGERS`` (сравнение подстрок без
    учёта регистра), возвращая tool-call ответ для того триггера, который
    совпал -- или обычный ответ "нет подходящего инструмента", если ни один
    не совпал. При втором вызове (после того как ``ToolInvoker`` агента
    добавил результат инструмента в историю сообщений) возвращает закрывающий
    текст ответа для того же триггера. Это значит, что поведение фейкового
    генератора реально зависит от того, что спросил пользователь, а не от
    авторского предположения о том, что сделала бы LLM -- так что эти тесты
    действительно проверяют *выбор* инструмента, а не только механику его
    *вызова*.
    """

    def __init__(self) -> None:
        self._call_count = 0

    @staticmethod
    def _latest_user_text(messages: list[ChatMessage]) -> str:
        user_messages = [m for m in messages if m.is_from("user")]
        if not user_messages:
            return ""
        return user_messages[-1].text or ""

    @component.output_types(replies=list)
    def run(self, messages: list[ChatMessage], tools: Any = None, **kwargs: Any) -> dict[str, Any]:
        self._call_count += 1
        user_text = self._latest_user_text(messages).lower()

        if self._call_count == 1:
            for trigger, tool_name, arguments, _final_text in _TOOL_SELECTION_TRIGGERS:
                if trigger in user_text:
                    return _tool_call_reply(tool_name, arguments)
            return _final_reply("I don't have a tool for that.")

        for trigger, _tool_name, _arguments, final_text in _TOOL_SELECTION_TRIGGERS:
            if trigger in user_text:
                return _final_reply(final_text)
        return _final_reply("ok")

    @component.output_types(replies=list)
    async def run_async(
        self, messages: list[ChatMessage], tools: Any = None, **kwargs: Any
    ) -> dict[str, Any]:
        return self.run(messages, tools=tools, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "PromptRoutingFakeChatGenerator", "data": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptRoutingFakeChatGenerator:
        return cls()


def _build_agent_with_fake_generator(
    responses: list[dict[str, Any]], tools: list[Tool]
) -> tuple[Agent, FakeChatGenerator]:
    generator = FakeChatGenerator(responses)
    agent = Agent(
        chat_generator=generator, tools=tools, system_prompt="You are a helpful assistant."
    )
    return agent, generator


def _build_agent_with_prompt_routing(tools: list[Tool]) -> Agent:
    """Построить ``Agent``, чей фейковый LLM выбирает tool call из реального текста промпта.

    В отличие от ``_build_agent_with_fake_generator``, *tools* здесь обычно
    содержит сразу несколько (как правило, все пять) реальных инструментов,
    так что прошедший тест показывает, что среди настоящих альтернатив был
    выбран именно нужный -- а не просто что был вызван единственный
    доступный инструмент.
    """
    return Agent(
        chat_generator=PromptRoutingFakeChatGenerator(),
        tools=tools,
        system_prompt="You are a helpful assistant.",
    )


# ---------------------------------------------------------------------------
# build_context_message
# ---------------------------------------------------------------------------


def test_build_context_message_serializes_only_text() -> None:
    memories = [_memory(text="first"), _memory(text="second")]

    message = build_context_message(memories)

    assert "недоверенный" in message.lower()
    parsed = json.loads(message.split("\n", 1)[1])
    assert parsed == ["first", "second"]


def test_build_context_message_empty_list_is_empty_json_array() -> None:
    message = build_context_message([])

    parsed = json.loads(message.split("\n", 1)[1])
    assert parsed == []


# ---------------------------------------------------------------------------
# _SYSTEM_PROMPT
# ---------------------------------------------------------------------------


def test_system_prompt_is_in_russian() -> None:
    assert re.search(r"[Ѐ-ӿ]", _SYSTEM_PROMPT)


def test_system_prompt_defaults_to_russian_but_allows_user_language() -> None:
    lowered = _SYSTEM_PROMPT.lower()
    assert "по умолчанию отвечай на русском" in lowered
    assert "написано на другом" in lowered and "можно ответить на этом языке" in lowered


def test_system_prompt_treats_retrieved_context_as_untrusted() -> None:
    lowered = _SYSTEM_PROMPT.lower()
    assert "недоверенные" in lowered
    assert "не инструкции" in lowered
    assert "никогда не" in lowered and "выполняй" in lowered


def test_system_prompt_lists_all_five_tools() -> None:
    for tool_name in (
        "get_current_weather",
        "convert_currency",
        "get_current_time",
        "get_public_holidays",
        "get_pypi_package_info",
    ):
        assert tool_name in _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# generate_reply -- проверка входных данных
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("user_text", ["", "   ", "\n\t"])
def test_empty_user_text_rejected_without_agent_call(user_text: str) -> None:
    agent, generator = _build_agent_with_fake_generator([_final_reply("hi")], tools=[])
    service = HaystackAgentService(settings=_build_settings(), agent=agent)

    with pytest.raises(HaystackAgentServiceError) as exc_info:
        asyncio.run(service.generate_reply(user_text=user_text, memories=[]))

    assert str(exc_info.value) == "user_text must not be empty or whitespace-only"
    assert generator.run_calls == []


# ---------------------------------------------------------------------------
# generate_reply -- обычный успех (без tool call)
# ---------------------------------------------------------------------------


def test_plain_reply_without_tool_call() -> None:
    agent, _ = _build_agent_with_fake_generator([_final_reply("Sure, here is a reply.")], tools=[])
    service = HaystackAgentService(settings=_build_settings(), agent=agent)

    reply = asyncio.run(service.generate_reply(user_text="hi", memories=[]))

    assert reply == "Sure, here is a reply."


def test_generate_reply_passes_context_and_user_message_to_agent() -> None:
    agent, generator = _build_agent_with_fake_generator([_final_reply("ok")], tools=[])
    service = HaystackAgentService(settings=_build_settings(), agent=agent)
    memories = [_memory(text="likes coffee")]

    asyncio.run(service.generate_reply(user_text="what do I like?", memories=memories))

    sent_messages = generator.run_calls[0]["messages"]
    user_messages = [m for m in sent_messages if m.is_from("user")]
    assert len(user_messages) == 1
    assert user_messages[0].text == "what do I like?"

    system_messages = [m for m in sent_messages if m.is_from("system")]
    assert any("likes coffee" in (m.text or "") for m in system_messages)


# ---------------------------------------------------------------------------
# generate_reply -- выбор инструмента
# ---------------------------------------------------------------------------


@pytest.fixture
def _all_five_tools_with_fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Зарегистрировать все пять реальных инструментов на одном агенте, каждый поверх фейка.

    Возвращает словарь журналов вызовов по каждому инструменту (по имени
    инструмента), чтобы тест мог проверить, что заполнен журнал ровно
    одного инструмента -- доказывая, что фейковый LLM действительно *выбрал*
    этот инструмент среди пяти реальных альтернатив, а не просто вызвал
    единственный данный ему инструмент.
    """
    from telegram_vector_memory_bot import tools as tools_module

    calls: dict[str, list[Any]] = {
        "get_current_weather": [],
        "convert_currency": [],
        "get_current_time": [],
        "get_public_holidays": [],
        "get_pypi_package_info": [],
    }

    def fake_weather(city: str) -> str:
        calls["get_current_weather"].append({"city": city})
        return "Текущая погода в Хельсинки: температура 5.0°C, скорость ветра 10.0 км/ч."

    def fake_convert(amount: float, from_currency: str, to_currency: str) -> str:
        calls["convert_currency"].append(
            {"amount": amount, "from_currency": from_currency, "to_currency": to_currency}
        )
        return "100 EUR = 110.00 USD (курс: 1 EUR = 1.1 USD)."

    def fake_get_current_time(location: str) -> str:
        calls["get_current_time"].append({"location": location})
        return (
            "Текущее время для 'Tokyo' (Asia/Tokyo): 2026-07-06 21:15:00 JST, понедельник."
        )

    def fake_public_holidays(country: str, year: int | None = None) -> str:
        calls["get_public_holidays"].append({"country": country, "year": year})
        return "Праздники в DK (2026): 2026-01-01 — Nytårsdag (New Year's Day)."

    def fake_pypi_package_info(package_name: str) -> str:
        calls["get_pypi_package_info"].append({"package_name": package_name})
        return (
            "Пакет haystack-ai: последняя версия 2.4.0. "
            "Описание: LLM orchestration framework. "
            "Требуемая версия Python: >=3.9. "
            "Лицензия: Apache-2.0. "
            "Ссылка: https://haystack.deepset.ai/."
        )

    monkeypatch.setattr(tools_module.weather_tool, "function", fake_weather)
    monkeypatch.setattr(tools_module.currency_tool, "function", fake_convert)
    monkeypatch.setattr(tools_module.time_tool, "function", fake_get_current_time)
    monkeypatch.setattr(tools_module.public_holidays_tool, "function", fake_public_holidays)
    monkeypatch.setattr(
        tools_module.pypi_package_info_tool, "function", fake_pypi_package_info
    )

    return calls


def _all_five_tools() -> list[Tool]:
    from telegram_vector_memory_bot import tools as tools_module

    return [
        tools_module.weather_tool,
        tools_module.currency_tool,
        tools_module.time_tool,
        tools_module.public_holidays_tool,
        tools_module.pypi_package_info_tool,
    ]


def _assert_only_this_tool_was_called(calls: dict[str, list[Any]], tool_name: str) -> None:
    for name, recorded in calls.items():
        if name == tool_name:
            assert recorded, f"expected {tool_name} to have been called, but it was not"
        else:
            assert not recorded, f"expected {name} not to be called, but it was: {recorded}"


def test_weather_tool_is_selected_for_weather_question(
    _all_five_tools_with_fakes: dict[str, list[Any]],
) -> None:
    agent = _build_agent_with_prompt_routing(_all_five_tools())
    service = HaystackAgentService(settings=_build_settings(), agent=agent)

    reply = asyncio.run(
        service.generate_reply(user_text="Какая погода в Хельсинки?", memories=[])
    )

    _assert_only_this_tool_was_called(_all_five_tools_with_fakes, "get_current_weather")
    assert _all_five_tools_with_fakes["get_current_weather"] == [{"city": "Helsinki"}]
    assert "Хельсинки" in reply


def test_currency_tool_is_selected_for_currency_question(
    _all_five_tools_with_fakes: dict[str, list[Any]],
) -> None:
    agent = _build_agent_with_prompt_routing(_all_five_tools())
    service = HaystackAgentService(settings=_build_settings(), agent=agent)

    reply = asyncio.run(
        service.generate_reply(user_text="Сколько 100 евро в долларах?", memories=[])
    )

    _assert_only_this_tool_was_called(_all_five_tools_with_fakes, "convert_currency")
    assert _all_five_tools_with_fakes["convert_currency"] == [
        {"amount": 100, "from_currency": "EUR", "to_currency": "USD"}
    ]
    assert "110" in reply


def test_time_tool_is_selected_for_what_time_is_it_question(
    _all_five_tools_with_fakes: dict[str, list[Any]],
) -> None:
    agent = _build_agent_with_prompt_routing(_all_five_tools())
    service = HaystackAgentService(settings=_build_settings(), agent=agent)

    reply = asyncio.run(
        service.generate_reply(user_text="Который час в Токио?", memories=[])
    )

    _assert_only_this_tool_was_called(_all_five_tools_with_fakes, "get_current_time")
    assert _all_five_tools_with_fakes["get_current_time"] == [{"location": "Tokyo"}]
    assert "Токио" in reply


def test_public_holidays_tool_is_selected_for_holiday_question(
    _all_five_tools_with_fakes: dict[str, list[Any]],
) -> None:
    agent = _build_agent_with_prompt_routing(_all_five_tools())
    service = HaystackAgentService(settings=_build_settings(), agent=agent)

    reply = asyncio.run(
        service.generate_reply(
            user_text="Какие праздники в Дании в 2026 году?", memories=[]
        )
    )

    _assert_only_this_tool_was_called(_all_five_tools_with_fakes, "get_public_holidays")
    assert _all_five_tools_with_fakes["get_public_holidays"] == [{"country": "Дания", "year": 2026}]
    assert "Новый год" in reply


def test_pypi_tool_is_selected_for_package_version_question(
    _all_five_tools_with_fakes: dict[str, list[Any]],
) -> None:
    agent = _build_agent_with_prompt_routing(_all_five_tools())
    service = HaystackAgentService(settings=_build_settings(), agent=agent)

    reply = asyncio.run(
        service.generate_reply(user_text="Какая последняя версия haystack-ai?", memories=[])
    )

    _assert_only_this_tool_was_called(_all_five_tools_with_fakes, "get_pypi_package_info")
    assert _all_five_tools_with_fakes["get_pypi_package_info"] == [
        {"package_name": "haystack-ai"}
    ]
    assert "haystack-ai" in reply


def test_no_tool_is_selected_for_an_unrelated_question(
    _all_five_tools_with_fakes: dict[str, list[Any]],
) -> None:
    agent = _build_agent_with_prompt_routing(_all_five_tools())
    service = HaystackAgentService(settings=_build_settings(), agent=agent)

    reply = asyncio.run(
        service.generate_reply(user_text="Расскажи анекдот про программистов", memories=[])
    )

    assert all(recorded == [] for recorded in _all_five_tools_with_fakes.values())
    assert reply == "I don't have a tool for that."


# ---------------------------------------------------------------------------
# generate_reply -- обработка ошибок
# ---------------------------------------------------------------------------


def test_agent_run_exception_wrapped_with_cause_preserved() -> None:
    class ExplodingAgent:
        async def run_async(self, *, messages: list[ChatMessage]) -> dict[str, Any]:
            raise RuntimeError("connection reset")

    service = HaystackAgentService(settings=_build_settings(), agent=ExplodingAgent())

    with pytest.raises(HaystackAgentServiceError) as exc_info:
        asyncio.run(service.generate_reply(user_text="hi", memories=[]))

    assert str(exc_info.value) == "haystack agent run failed"
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "connection reset" not in str(exc_info.value)


def test_no_last_message_rejected() -> None:
    class EmptyResultAgent:
        async def run_async(self, *, messages: list[ChatMessage]) -> dict[str, Any]:
            return {"messages": []}

    service = HaystackAgentService(settings=_build_settings(), agent=EmptyResultAgent())

    with pytest.raises(HaystackAgentServiceError) as exc_info:
        asyncio.run(service.generate_reply(user_text="hi", memories=[]))

    assert str(exc_info.value) == "haystack agent run produced no reply message"


def test_blank_final_reply_rejected() -> None:
    class BlankReplyAgent:
        async def run_async(self, *, messages: list[ChatMessage]) -> dict[str, Any]:
            return {"last_message": ChatMessage.from_assistant("   ")}

    service = HaystackAgentService(settings=_build_settings(), agent=BlankReplyAgent())

    with pytest.raises(HaystackAgentServiceError) as exc_info:
        asyncio.run(service.generate_reply(user_text="hi", memories=[]))

    assert str(exc_info.value) == "haystack agent reply message must not be blank"


def test_error_message_does_not_expose_secrets_or_prompt_data() -> None:
    class ExplodingAgent:
        async def run_async(self, *, messages: list[ChatMessage]) -> dict[str, Any]:
            raise RuntimeError(f"upstream failure for key {FAKE_API_KEY}")

    settings = _build_settings()
    service = HaystackAgentService(settings=settings, agent=ExplodingAgent())
    secret_user_text = "my password is hunter2"
    memories = [_memory(text="my secret memory text")]

    with pytest.raises(HaystackAgentServiceError) as exc_info:
        asyncio.run(service.generate_reply(user_text=secret_user_text, memories=memories))

    rendered_error = str(exc_info.value)
    assert rendered_error == "haystack agent run failed"
    for leaked in (FAKE_API_KEY, secret_user_text, "my secret memory text", "upstream failure"):
        assert leaked not in rendered_error


# ---------------------------------------------------------------------------
# Конструирование
# ---------------------------------------------------------------------------


def test_default_tools_used_when_none_injected() -> None:
    from telegram_vector_memory_bot.tools import build_default_tools

    service = HaystackAgentService(settings=_build_settings())

    configured_tool_names = {tool.name for tool in service._agent.tools}
    assert configured_tool_names == {tool.name for tool in build_default_tools()}
    assert "get_current_time" in configured_tool_names


def test_injected_agent_is_used_instead_of_building_one() -> None:
    agent, _ = _build_agent_with_fake_generator([_final_reply("ok")], tools=[])

    service = HaystackAgentService(settings=_build_settings(), agent=agent)

    assert service._agent is agent


class _RecordingChatGenerator:
    """Записывает kwargs конструктора; выставляет минимальный ``run``, чтобы
    реальный ``Agent`` принял его как валидный chat generator без вызовов."""

    last_kwargs: dict[str, Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs

    def run(self, messages: list[ChatMessage], tools: Any = None, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("run must not be called merely by constructing the service")


def test_chat_generator_constructed_with_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from telegram_vector_memory_bot import haystack_agent as haystack_agent_module

    _RecordingChatGenerator.last_kwargs = None
    monkeypatch.setattr(haystack_agent_module, "OpenAIChatGenerator", _RecordingChatGenerator)
    settings = _build_settings(OPENAI_BASE_URL="https://proxy.example.com/v1")

    HaystackAgentService(settings=settings)

    assert _RecordingChatGenerator.last_kwargs is not None
    assert _RecordingChatGenerator.last_kwargs["api_base_url"] == "https://proxy.example.com/v1"


def test_chat_generator_constructed_without_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from telegram_vector_memory_bot import haystack_agent as haystack_agent_module

    _RecordingChatGenerator.last_kwargs = None
    monkeypatch.setattr(haystack_agent_module, "OpenAIChatGenerator", _RecordingChatGenerator)

    HaystackAgentService(settings=_build_settings())

    assert _RecordingChatGenerator.last_kwargs is not None
    assert "api_base_url" not in _RecordingChatGenerator.last_kwargs


def test_non_text_last_message_rejected() -> None:
    class NonTextReplyAgent:
        async def run_async(self, *, messages: list[ChatMessage]) -> dict[str, Any]:
            # Сообщение, несущее только tool call, без текстового содержимого --
            # проверяет ветку, где последнее сообщение агента не обычный текст.
            last_message = ChatMessage.from_assistant(
                text=None,
                tool_calls=[ToolCall(id="call_1", tool_name="noop", arguments={})],
            )
            return {"last_message": last_message}

    service = HaystackAgentService(settings=_build_settings(), agent=NonTextReplyAgent())

    with pytest.raises(HaystackAgentServiceError) as exc_info:
        asyncio.run(service.generate_reply(user_text="hi", memories=[]))

    assert str(exc_info.value) == "haystack agent reply message contained no text content"
