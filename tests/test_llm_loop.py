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
