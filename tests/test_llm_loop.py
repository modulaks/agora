"""Pętla tool-calling bez prawdziwej Ollamy: podstawiony klient zwraca obiekty w formacie ollama.Message."""

from typing import Annotated

import pytest
from ollama import ChatResponse, Message
from pydantic import Field

from agora.llm import OllamaChat
from agora.tools import Toolbox, tool


class StubClient:
    def __init__(self):
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        n = len(self.calls)
        if n == 1:
            msg = Message(role="assistant", content="", thinking="najpierw sprawdzę", tool_calls=[
                Message.ToolCall(function=Message.ToolCall.Function(name="add", arguments={"a": 2, "b": 3}))
            ])
        else:
            tool_msgs = [m for m in kwargs["messages"] if isinstance(m, dict) and m.get("role") == "tool"]
            assert tool_msgs and tool_msgs[-1]["content"] == "5" and tool_msgs[-1]["tool_name"] == "add"
            msg = Message(role="assistant", content="Wynik to 5.", thinking="")
        return ChatResponse(model="x", message=msg, done=True)


@pytest.mark.asyncio
async def test_tool_loop_roundtrip():
    @tool
    def add(a: Annotated[int, Field(description="a")], b: Annotated[int, Field(description="b")] = 0) -> str:
        """Dodaje."""
        return str(a + b)

    chat = OllamaChat("http://localhost:1", "fake", concurrency=1)
    chat._client = StubClient()
    events = []

    async def on_event(kind, payload):
        events.append((kind, payload))

    result = await chat.run(system="s", user="u", toolbox=Toolbox([add]), temperature=0.1, on_event=on_event, max_rounds=3)
    assert result.text == "Wynik to 5."
    assert result.tool_calls == 1 and result.rounds == 2
    kinds = [k for k, _ in events]
    assert kinds == ["thinking", "tool_call", "tool_result"]
    schema = Toolbox([add]).schemas()[0]
    assert schema["function"]["parameters"]["required"] == ["a"]


class SlowClient:
    async def chat(self, **kwargs):
        import asyncio

        await asyncio.sleep(0.35)
        return ChatResponse(model="x", message=Message(role="assistant", content="gotowe"), done=True)


@pytest.mark.asyncio
async def test_heartbeat_while_waiting(monkeypatch):
    import agora.llm as llm

    monkeypatch.setattr(llm, "HEARTBEAT_SECONDS", 0.1)
    chat = OllamaChat("http://localhost:1", "fake", concurrency=1)
    chat._client = SlowClient()
    events = []

    async def on_event(kind, payload):
        events.append((kind, payload))

    result = await chat.run(system="s", user="u", toolbox=Toolbox([]), temperature=0.1, on_event=on_event, max_rounds=1)
    assert result.text == "gotowe"
    waits = [p for k, p in events if k == "waiting"]
    assert len(waits) >= 2 and all(p["round"] == 1 for p in waits)


# ── backend Claude na podstawionym kliencie (bez sieci) ──────────────────

from types import SimpleNamespace as NS

from agora.claude_chat import AnthropicChat, to_anthropic_tools


class StubAnthropic:
    def __init__(self):
        self.calls = []
        self.beta = NS(messages=NS(create=self.create))

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        usage = NS(input_tokens=100, output_tokens=20, cache_read_input_tokens=80, cache_creation_input_tokens=0)
        if len(self.calls) == 1:
            content = [
                NS(type="thinking", thinking="dodam liczby"),
                NS(type="tool_use", id="tu_1", name="add", input={"a": 2, "b": 3}),
                NS(type="tool_use", id="tu_2", name="add", input={"a": "x"}),
            ]
            return NS(content=content, stop_reason="tool_use", usage=usage)
        results = kwargs["messages"][-1]["content"]
        assert [r["tool_use_id"] for r in results] == ["tu_1", "tu_2"]
        assert results[0]["content"] == "5" and not results[0]["is_error"]
        assert results[1]["is_error"]
        return NS(content=[NS(type="text", text="Wynik to 5.")], stop_reason="end_turn", usage=usage)


@pytest.mark.asyncio
async def test_anthropic_tool_loop():
    @tool
    def add(a: Annotated[int, Field(description="a")], b: Annotated[int, Field(description="b")] = 0) -> str:
        """Dodaje."""
        return str(a + b)

    stub = StubAnthropic()
    chat = AnthropicChat("claude-sonnet-5-5", effort="medium", client=stub)
    events = []

    async def on_event(kind, payload):
        events.append((kind, payload))

    result = await chat.run(system="s", user="u", toolbox=Toolbox([add]), temperature=0.7, on_event=on_event, max_rounds=3)
    assert result.text == "Wynik to 5." and result.error is None
    assert result.tool_calls == 2 and result.rounds == 2
    assert result.usage == {"in": 200, "out": 40, "cache_read": 160}
    first = stub.calls[0]
    assert "temperature" not in first
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first["fallbacks"] == "default" and first["betas"] == ["server-side-fallback-2026-07-01"]
    assert first["tools"][0]["input_schema"]["required"] == ["a"]
    assert [k for k, _ in events].count("tool_call") == 2 and events[0][0] == "thinking"


def test_anthropic_tool_schema_and_haiku_without_fallback():
    schema = to_anthropic_tools([{"type": "function", "function": {"name": "x", "description": "d", "parameters": {"type": "object"}}}])
    assert schema == [{"name": "x", "description": "d", "input_schema": {"type": "object"}}]
    assert AnthropicChat("claude-haiku-5-5", client=object())._extra() == {}
