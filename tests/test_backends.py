"""Trzy modele z komendy, odporne wczytywanie .env i pętla OpenAI na podstawionym kliencie (bez sieci)."""

import os
from types import SimpleNamespace as NS
from typing import Annotated

import pytest
from pydantic import Field

from agora.__main__ import main
from agora.openai_chat import OpenAIChat
from agora.settings import load_dotenv, load_settings
from agora.tools import Toolbox, tool


@pytest.mark.parametrize("model, backend, label", [
    ("qwen", "ollama", "qwen3.8:27b @ "),
    ("luna", "openai", "luna (OpenAI API)"),
    ("sonnet", "anthropic", "claude-sonnet-5-5 (Claude API"),
])
def test_three_models(model, backend, label):
    s = load_settings(model=model)
    assert s.backend == backend and s.model_label.startswith(label)


def test_default_is_qwen_and_other_names_rejected():
    assert load_settings().backend == "ollama"
    with pytest.raises(SystemExit):
        main(["--model", "gpt-4o", "--dry-run", "--ticks", "0"])


def test_dotenv_copied_from_other_project(tmp_path, monkeypatch):
    for k in ("A_KEY", "B_KEY", "EMPTY_KEY"):
        monkeypatch.setenv(k, "x")
        monkeypatch.delenv(k)
    env = tmp_path / ".env"
    # BOM, komentarz w cp1250 i pusta wartość: nic z tego nie może wywrócić startu
    env.write_bytes("\ufeffA_KEY=sk-1\n".encode("utf-8") + "# za\u017c\u00f3\u0142\u0107\n".encode("cp1250") + b"\xef\xbb\xbfB_KEY=sk-2\nEMPTY_KEY=\n")
    load_dotenv(env)
    assert os.environ["A_KEY"] == "sk-1" and os.environ["B_KEY"] == "sk-2"
    assert "EMPTY_KEY" not in os.environ


def test_dotenv_wins_over_old_windows_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("C_KEY", "sk-stary-z-setx")
    env = tmp_path / ".env"
    env.write_text("C_KEY=sk-nowy-z-env\n", encoding="ascii")
    load_dotenv(env)
    assert os.environ["C_KEY"] == "sk-nowy-z-env"


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
