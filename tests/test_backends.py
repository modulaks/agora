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
    ("luna", "openai", "gpt-6-luna (OpenAI API)"),
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
    monkeypatch.setenv("C_API_KEY", "sk-stary-z-setx")
    monkeypatch.setenv("C_DATA_DIR", "z-testu")
    env = tmp_path / ".env"
    env.write_text("C_API_KEY=sk-nowy-z-env\nC_DATA_DIR=z-env\n", encoding="ascii")
    load_dotenv(env)
    assert os.environ["C_API_KEY"] == "sk-nowy-z-env"  # klucz z .env wygrywa
    assert os.environ["C_DATA_DIR"] == "z-testu"  # reszta: środowisko wygrywa


def _resp(rid, output, status="completed"):
    usage = NS(input_tokens=50, output_tokens=10, input_tokens_details=NS(cached_tokens=40))
    return NS(id=rid, status=status, output=output, usage=usage, error=None, incomplete_details=None)


class StubResponses:
    """Podstawiony klient Responses API: runda 1 woła dwa narzędzia, runda 2 kończy tekstem."""

    def __init__(self):
        self.calls = []
        self.responses = NS(create=self.create)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            assert kwargs["input"] == [{"role": "user", "content": "u"}] and "previous_response_id" not in kwargs
            return _resp("r1", [
                NS(type="reasoning", summary=[NS(type="summary_text", text="dodam liczby")]),
                NS(type="function_call", call_id="c1", name="add", arguments='{"a": 2, "b": 3}'),
                NS(type="function_call", call_id="c2", name="add", arguments="{zły json"),
            ])
        assert kwargs["previous_response_id"] == "r1" and kwargs["instructions"] == "s"
        outs = kwargs["input"]
        assert [o["call_id"] for o in outs] == ["c1", "c2"] and all(o["type"] == "function_call_output" for o in outs)
        assert outs[0]["output"] == "5" and outs[1]["output"].startswith("BŁĄD")
        return _resp("r2", [NS(type="message", content=[NS(type="output_text", text="Wynik to 5.")])])


@pytest.mark.asyncio
async def test_openai_responses_tool_loop():
    @tool
    def add(a: Annotated[int, Field(description="a")], b: Annotated[int, Field(description="b")] = 0) -> str:
        """Dodaje."""
        return str(a + b)

    stub = StubResponses()
    chat = OpenAIChat("gpt-6-luna", client=stub)
    events = []

    async def on_event(kind, payload):
        events.append(kind)

    result = await chat.run(system="s", user="u", toolbox=Toolbox([add]), temperature=0.7, on_event=on_event, max_rounds=3)
    assert result.text == "Wynik to 5." and result.error is None
    assert result.tool_calls == 2 and result.rounds == 2
    assert result.usage == {"in": 100, "out": 20, "cache_read": 80}
    first = stub.calls[0]
    assert "temperature" not in first and first["tool_choice"] == "auto" and first["store"] is True
    assert first["reasoning"] == {"effort": "medium", "summary": "auto"}  # rozumowanie razem z narzędziami
    assert first["tools"][0] == {"type": "function", "name": "add", "description": "Dodaje.", "parameters": first["tools"][0]["parameters"], "strict": False}
    assert events[0] == "thinking" and events.count("tool_call") == 2


@pytest.mark.asyncio
async def test_reasoning_summary_dropped_when_account_cannot_use_it():
    import httpx2
    import openai

    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if "summary" in (kwargs.get("reasoning") or {}):
            raise openai.BadRequestError("Your organization must be verified to generate reasoning summaries.", response=httpx2.Response(400, request=httpx2.Request("POST", "https://x")), body=None)
        return _resp("r1", [NS(type="message", content=[NS(type="output_text", text="ok")])])

    chat = OpenAIChat("gpt-6-luna", client=NS(responses=NS(create=create)))

    async def on_event(kind, payload):
        pass

    for _ in range(2):
        result = await chat.run(system="s", user="u", toolbox=Toolbox([]), temperature=0.5, on_event=on_event, max_rounds=1)
        assert result.text == "ok" and result.error is None
    assert [c["reasoning"] for c in calls] == [{"effort": "medium", "summary": "auto"}, {"effort": "medium"}, {"effort": "medium"}]


@pytest.mark.asyncio
async def test_incomplete_response_is_reported():
    async def create(**kwargs):
        r = _resp("r1", [], status="incomplete")
        r.incomplete_details = NS(reason="max_output_tokens")
        return r

    chat = OpenAIChat("gpt-6-luna", client=NS(responses=NS(create=create)))

    async def on_event(kind, payload):
        pass

    result = await chat.run(system="s", user="u", toolbox=Toolbox([]), temperature=0.5, on_event=on_event, max_rounds=1)
    assert result.error == "odpowiedź niepełna (max_output_tokens)"


def test_key_copied_from_chat_is_reported(monkeypatch):
    from agora.llm import broken_key_message
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-????????????????")
    assert "uszkodzony" in broken_key_message("OPENAI_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-9DFrlFStb12g83Lf5ao7vfiOlzqw-PyESYa3mJ_EZk58")
    assert broken_key_message("OPENAI_API_KEY") is None


class _Models:
    def __init__(self, ids):
        self.ids = ids

    async def retrieve(self, model):
        if model not in self.ids:
            import httpx2
            raise __import__("openai").NotFoundError("brak", response=httpx2.Response(404, request=httpx2.Request("GET", "https://x")), body=None)
        return NS(id=model)

    def list(self):
        ids = self.ids

        async def gen():
            for i in ids:
                yield NS(id=i)
        return gen()


@pytest.mark.asyncio
@pytest.mark.parametrize("ids, expected", [
    (["gpt-6-luna", "gpt-5.6-luna"], "gpt-6-luna"),
    (["gpt-4o", "gpt-5.6-luna-fast", "gpt-5.6-luna"], "gpt-5.6-luna"),
    (["gpt-4o", "gpt-7-luna-preview"], "gpt-7-luna-preview"),
])
async def test_luna_resolved_from_account(ids, expected, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-poprawny-klucz-testowy")
    chat = OpenAIChat("gpt-6-luna", client=NS(models=_Models(ids)))
    await chat.check()
    assert chat.model == expected


@pytest.mark.asyncio
async def test_no_luna_on_account(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-poprawny-klucz-testowy")
    chat = OpenAIChat("gpt-6-luna", client=NS(models=_Models(["gpt-4o"])))
    with pytest.raises(SystemExit, match="Luny"):
        await chat.check()
