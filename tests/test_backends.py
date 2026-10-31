"""Wybór modelu z komendy, .env innego projektu i pętla OpenAI na podstawionym kliencie (bez sieci)."""

from types import SimpleNamespace as NS
from typing import Annotated

import pytest
from pydantic import Field

from agora.openai_chat import OpenAIChat
from agora.settings import load_settings, parse_model_choice
from agora.tools import Toolbox, tool


@pytest.mark.parametrize("choice, expected", [
    ("llama", ("ollama", None)),
    ("lama", ("ollama", None)),
    ("openai", ("openai", None)),
    ("sonnet", ("anthropic", "claude-sonnet-5-5")),
    ("Haiku", ("anthropic", "claude-haiku-5-5")),
    ("openai:gpt-5", ("openai", "gpt-5")),
    ("ollama:qwen3:8b", ("ollama", "qwen3:8b")),
    ("qwen3.8:27b", ("ollama", "qwen3.8:27b")),
    ("claude-opus-5-5", ("anthropic", "claude-opus-5-5")),
    ("gpt-4.1", ("openai", "gpt-4.1")),
])
def test_parse_model_choice(choice, expected):
    assert parse_model_choice(choice) == expected


KEYS = ("OPENAI_API_KEY", "OPENAI_MODEL", "ANTHROPIC_API_KEY", "AGORA_ENV_FILE", "AGORA_BACKEND", "ANTHROPIC_MODEL", "OLLAMA_MODEL")


def test_env_file_from_other_project_and_model_flag(tmp_path, monkeypatch):
    for k in KEYS:  # setenv+delenv: monkeypatch przywróci stan sprzed testu, także dla kluczy dopisanych przez .env
        monkeypatch.setenv(k, "x")
        monkeypatch.delenv(k)
    other = tmp_path / "inny-projekt"
    other.mkdir()
    (other / ".env").write_text("﻿OPENAI_API_KEY=sk-test-123\nOPENAI_MODEL=gpt-5\n", encoding="utf-8")
    monkeypatch.setenv("AGORA_DATA_DIR", str(tmp_path / "data"))

    s = load_settings(model_choice="openai", env_file=other)  # katalog też wystarczy
    assert s.backend == "openai" and s.openai_model == "gpt-5" and s.env_file == other / ".env"
    import os
    assert os.environ["OPENAI_API_KEY"] == "sk-test-123"

    s2 = load_settings(model_choice="sonnet")
    assert s2.backend == "anthropic" and s2.anthropic_model == "claude-sonnet-5-5"


def test_missing_env_file_is_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("AGORA_ENV_FILE", raising=False)
    with pytest.raises(SystemExit, match="Nie ma pliku .env"):
        load_settings(env_file=tmp_path / "nie-ma")


class StubOpenAI:
    def __init__(self):
        self.calls = []
        self.chat = NS(completions=NS(create=self.create))

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        usage = NS(prompt_tokens=50, completion_tokens=10, prompt_tokens_details=NS(cached_tokens=40))
        if len(self.calls) == 1:
            calls = [
                NS(id="c1", function=NS(name="add", arguments='{"a": 2, "b": 3}')),
                NS(id="c2", function=NS(name="add", arguments="{zły json")),
            ]
            msg = NS(content=None, tool_calls=calls, refusal=None)
            return NS(choices=[NS(message=msg, finish_reason="tool_calls")], usage=usage)
        tool_msgs = [m for m in kwargs["messages"] if m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]
        assert tool_msgs[0]["content"] == "5" and tool_msgs[1]["content"].startswith("BŁĄD")
        assistant = [m for m in kwargs["messages"] if m.get("role") == "assistant"][0]
        assert assistant["tool_calls"][0]["function"]["name"] == "add"
        msg = NS(content="Wynik to 5.", tool_calls=None, refusal=None)
        return NS(choices=[NS(message=msg, finish_reason="stop")], usage=usage)


@pytest.mark.asyncio
async def test_openai_tool_loop():
    @tool
    def add(a: Annotated[int, Field(description="a")], b: Annotated[int, Field(description="b")] = 0) -> str:
        """Dodaje."""
        return str(a + b)

    stub = StubOpenAI()
    chat = OpenAIChat("gpt-5", client=stub)
    events = []

    async def on_event(kind, payload):
        events.append(kind)

    result = await chat.run(system="s", user="u", toolbox=Toolbox([add]), temperature=0.7, on_event=on_event, max_rounds=3)
    assert result.text == "Wynik to 5." and result.error is None
    assert result.tool_calls == 2 and result.rounds == 2
    assert result.usage == {"in": 100, "out": 20, "cache_read": 80}
    assert "temperature" not in stub.calls[0] and stub.calls[0]["tool_choice"] == "auto"
    assert stub.calls[0]["tools"][0]["type"] == "function"
    assert events.count("tool_call") == 2
