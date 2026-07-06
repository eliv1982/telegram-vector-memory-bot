"""Unit tests for telegram_vector_memory_bot.haystack_agent.

No real OpenAI client and no real HTTP call is ever made. The chat model is
replaced by ``FakeChatGenerator``, a minimal Haystack ``@component`` that
returns pre-scripted ``ChatMessage`` replies -- including tool-call messages
-- so a real ``haystack.components.agents.Agent`` can be exercised end to
end (tool selection, tool invocation, final reply extraction) entirely
offline. Real ``Tool`` objects from ``tools.py`` are used, wrapping fake
underlying functions so no network I/O occurs.

The tool-selection tests use ``PromptRoutingFakeChatGenerator``, which reads
the actual incoming user message text and picks its scripted tool call from
that content -- with all five real tools registered on the same ``Agent`` at
once. This is deliberately stronger than a generator that is merely
pre-scripted to always return one fixed tool call regardless of the prompt
(which would only prove the plumbing works, not that the right tool was
"selected" for a given question): here, a different question really does
route to a different tool call, and a wrong keyword match would call the
wrong tool and fail the assertions on which fake tool function ran.

Async methods are exercised via ``asyncio.run`` directly rather than a
pytest-asyncio plugin, since none is a project dependency.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

import pytest
from haystack import component
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool

from telegram_vector_memory_bot.config import Settings
from telegram_vector_memory_bot.haystack_agent import (
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
    """Minimal Haystack chat-generator component with pre-scripted replies.

    Each call to ``run_async`` (or ``run``) returns the next scripted
    ``{"replies": [...]}`` dict, letting a real ``Agent`` drive a full
    tool-call round trip without any network access.
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


# (trigger substring, tool name, tool arguments, closing reply after the tool result)
# Matched case-insensitively against the user's actual message text -- see
# PromptRoutingFakeChatGenerator below.
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
        "про финляндию",
        "get_country_info",
        {"country": "Finland"},
        "Финляндия — страна в Европе со столицей в Хельсинки.",
    ),
    (
        "alan turing",
        "get_wikipedia_summary",
        {"topic": "Alan Turing"},
        "Алан Тьюринг — английский математик и computer scientist.",
    ),
    (
        "который час",
        "get_current_time",
        {"location": "Helsinki"},
        "Сейчас в Хельсинки 21:15, понедельник.",
    ),
]


@component
class PromptRoutingFakeChatGenerator:
    """Chooses a scripted tool call (or plain reply) from the user's own prompt text.

    On its first call it inspects the latest user message and matches it
    against ``_TOOL_SELECTION_TRIGGERS`` (case-insensitive substring match),
    returning a tool-call reply for whichever trigger matched -- or a plain
    "no tool for that" reply if none did. On the second call (after the
    ``Agent``'s ``ToolInvoker`` has appended the tool result to the message
    history) it returns the closing reply text for that same trigger. This
    means the fake generator's behavior genuinely depends on what the user
    asked, rather than an author-scripted assumption about what the LLM
    would do -- so these tests actually exercise tool *selection*, not just
    tool *invocation* plumbing.
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
    """Build an ``Agent`` whose fake LLM picks its tool call from the real prompt text.

    Unlike ``_build_agent_with_fake_generator``, *tools* is meant to hold
    several (typically all five) real tools at once, so a passing test shows
    the right one was chosen among genuine alternatives -- not just that the
    only tool available happened to be invoked.
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

    assert "untrusted" in message.lower()
    parsed = json.loads(message.split("\n", 1)[1])
    assert parsed == ["first", "second"]


def test_build_context_message_empty_list_is_empty_json_array() -> None:
    message = build_context_message([])

    parsed = json.loads(message.split("\n", 1)[1])
    assert parsed == []


# ---------------------------------------------------------------------------
# generate_reply -- input validation
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
# generate_reply -- plain (no tool call) success
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
# generate_reply -- tool selection
# ---------------------------------------------------------------------------


@pytest.fixture
def _all_five_tools_with_fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Register all five real tools on one agent, each backed by a call-recording fake.

    Returns a dict of per-tool call logs (keyed by tool name) so a test can
    assert that exactly one tool's log was populated -- proving the fake LLM
    actually *selected* that tool among five real alternatives, rather than
    merely invoking the one tool it was given.
    """
    from telegram_vector_memory_bot import tools as tools_module

    calls: dict[str, list[Any]] = {
        "get_current_weather": [],
        "convert_currency": [],
        "get_country_info": [],
        "get_wikipedia_summary": [],
        "get_current_time": [],
    }

    def fake_weather(city: str) -> str:
        calls["get_current_weather"].append({"city": city})
        return "Current weather in Helsinki: temperature 5.0°C, wind speed 10.0 km/h."

    def fake_convert(amount: float, from_currency: str, to_currency: str) -> str:
        calls["convert_currency"].append(
            {"amount": amount, "from_currency": from_currency, "to_currency": to_currency}
        )
        return "100 EUR = 110.00 USD (rate: 1 EUR = 1.1 USD)."

    def fake_country_info(country: str) -> str:
        calls["get_country_info"].append({"country": country})
        return (
            "Finland: capital Helsinki, region Europe, population 5540720, "
            "currencies: Euro (EUR)."
        )

    def fake_wikipedia_summary(topic: str, language: str = "en") -> str:
        calls["get_wikipedia_summary"].append({"topic": topic})
        return (
            "Alan Turing: English mathematician and computer scientist. "
            "(source: https://en.wikipedia.org/wiki/Alan_Turing)"
        )

    def fake_get_current_time(location: str) -> str:
        calls["get_current_time"].append({"location": location})
        return "Current time for 'Helsinki' (Europe/Helsinki): 2026-07-06 21:15:00 EEST, Monday."

    monkeypatch.setattr(tools_module.weather_tool, "function", fake_weather)
    monkeypatch.setattr(tools_module.currency_tool, "function", fake_convert)
    monkeypatch.setattr(tools_module.country_info_tool, "function", fake_country_info)
    monkeypatch.setattr(tools_module.wikipedia_summary_tool, "function", fake_wikipedia_summary)
    monkeypatch.setattr(tools_module.time_tool, "function", fake_get_current_time)

    return calls


def _all_five_tools() -> list[Tool]:
    from telegram_vector_memory_bot import tools as tools_module

    return [
        tools_module.weather_tool,
        tools_module.currency_tool,
        tools_module.country_info_tool,
        tools_module.wikipedia_summary_tool,
        tools_module.time_tool,
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


def test_country_info_tool_is_selected_for_country_question(
    _all_five_tools_with_fakes: dict[str, list[Any]],
) -> None:
    agent = _build_agent_with_prompt_routing(_all_five_tools())
    service = HaystackAgentService(settings=_build_settings(), agent=agent)

    reply = asyncio.run(
        service.generate_reply(user_text="Расскажи кратко про Финляндию", memories=[])
    )

    _assert_only_this_tool_was_called(_all_five_tools_with_fakes, "get_country_info")
    assert _all_five_tools_with_fakes["get_country_info"] == [{"country": "Finland"}]
    assert "Хельсинки" in reply


def test_wikipedia_summary_tool_is_selected_for_who_is_question(
    _all_five_tools_with_fakes: dict[str, list[Any]],
) -> None:
    agent = _build_agent_with_prompt_routing(_all_five_tools())
    service = HaystackAgentService(settings=_build_settings(), agent=agent)

    reply = asyncio.run(service.generate_reply(user_text="Кто такой Alan Turing?", memories=[]))

    _assert_only_this_tool_was_called(_all_five_tools_with_fakes, "get_wikipedia_summary")
    assert _all_five_tools_with_fakes["get_wikipedia_summary"] == [{"topic": "Alan Turing"}]
    assert "Тьюринг" in reply


def test_time_tool_is_selected_for_what_time_is_it_question(
    _all_five_tools_with_fakes: dict[str, list[Any]],
) -> None:
    agent = _build_agent_with_prompt_routing(_all_five_tools())
    service = HaystackAgentService(settings=_build_settings(), agent=agent)

    reply = asyncio.run(
        service.generate_reply(user_text="Который час в Хельсинки?", memories=[])
    )

    _assert_only_this_tool_was_called(_all_five_tools_with_fakes, "get_current_time")
    assert _all_five_tools_with_fakes["get_current_time"] == [{"location": "Helsinki"}]
    assert "Хельсинки" in reply


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
# generate_reply -- error handling
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
# Construction
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
    """Records construction kwargs; exposes a bare-minimum ``run`` so a real
    ``Agent`` accepts it as a valid chat generator without making any call."""

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
            # A message carrying only a tool call, no text content -- exercises
            # the branch where the agent's last message is not plain text.
            last_message = ChatMessage.from_assistant(
                text=None,
                tool_calls=[ToolCall(id="call_1", tool_name="noop", arguments={})],
            )
            return {"last_message": last_message}

    service = HaystackAgentService(settings=_build_settings(), agent=NonTextReplyAgent())

    with pytest.raises(HaystackAgentServiceError) as exc_info:
        asyncio.run(service.generate_reply(user_text="hi", memories=[]))

    assert str(exc_info.value) == "haystack agent reply message contained no text content"
