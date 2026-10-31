"""Pętla tool-calling na natywnym API Ollamy. Każdy krok (myślenie, wywołanie, wynik) jest raportowany."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

import httpx
import ollama

from .tools import Toolbox

EventSink = Callable[[str, dict[str, Any]], Awaitable[None] | None]

HEARTBEAT_SECONDS = 30.0


@dataclass(slots=True)
class ChatResult:
    text: str
    rounds: int = 0
    tool_calls: int = 0
    seconds: float = 0.0
    error: str | None = None
    transcript: list[dict[str, Any]] = field(default_factory=list)


class ChatBackend(Protocol):
    async def run(
        self,
        *,
        system: str,
        user: str,
        toolbox: Toolbox,
        temperature: float,
        on_event: EventSink,
        max_rounds: int,
        agent: str = "",
    ) -> ChatResult: ...

    async def complete(self, *, system: str, user: str, temperature: float = 0.2) -> str: ...


async def _emit(sink: EventSink, kind: str, payload: dict[str, Any]) -> None:
    out = sink(kind, payload)
    if asyncio.iscoroutine(out):
        await out


class OllamaChat:
    """Jeden współdzielony klient; semafor ogranicza równoległe wywołania."""

    def __init__(
        self,
        host: str,
        model: str,
        *,
        num_ctx: int = 16384,
        think: bool = True,
        concurrency: int = 2,
        timeout: float = 600.0,
    ):
        self.host = host
        self.model = model
        self.num_ctx = num_ctx
        self.think = think
        self.timeout = timeout
        self._client = ollama.AsyncClient(host=host, timeout=timeout)
        self._sem = asyncio.Semaphore(max(1, concurrency))

    async def _chat(self, messages: list[Any], *, tools: list[dict[str, Any]] | None, temperature: float, think: bool | None = None):
        options = {"temperature": temperature, "num_ctx": self.num_ctx}
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "options": options}
        if tools:
            kwargs["tools"] = tools
        use_think = self.think if think is None else think
        if use_think:
            kwargs["think"] = True
        async with self._sem:
            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    return await self._client.chat(**kwargs)
                except (httpx.HTTPError, ollama.ResponseError, asyncio.TimeoutError) as e:
                    last_exc = e
                    msg = str(e)
                    # Model bez wsparcia think/tools: spróbuj bez.
                    if "think" in msg.lower() and "think" in kwargs:
                        kwargs.pop("think")
                        continue
                    if "does not support tools" in msg.lower() and "tools" in kwargs:
                        kwargs.pop("tools")
                        continue
                    await asyncio.sleep(2.0 * (attempt + 1))
            assert last_exc is not None
            raise last_exc

    async def _chat_with_heartbeat(self, messages: list[Any], *, tools: list[dict[str, Any]] | None, temperature: float, on_event: EventSink, round_no: int):
        """Wywołanie modelu bez streamingu trwa minutami; co HEARTBEAT_SECONDS raportujemy, że agent czeka."""
        start = time.monotonic()
        task = asyncio.ensure_future(self._chat(messages, tools=tools, temperature=temperature))
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=HEARTBEAT_SECONDS)
                if done:
                    return task.result()
                await _emit(on_event, "waiting", {"seconds": round(time.monotonic() - start), "round": round_no})
        finally:
            if not task.done():
                task.cancel()

    async def run(
        self,
        *,
        system: str,
        user: str,
        toolbox: Toolbox,
        temperature: float,
        on_event: EventSink,
        max_rounds: int = 6,
        agent: str = "",
    ) -> ChatResult:
        start = time.monotonic()
        messages: list[Any] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        transcript: list[dict[str, Any]] = []
        result = ChatResult(text="")
        schemas = toolbox.schemas()
        for round_no in range(1, max_rounds + 2):
            # Ostatnia runda bez narzędzi: wymuszamy zamknięcie tury tekstem.
            tools = schemas if round_no <= max_rounds else None
            try:
                response = await self._chat_with_heartbeat(messages, tools=tools, temperature=temperature, on_event=on_event, round_no=round_no)
            except Exception as e:  # noqa: BLE001
                result.error = f"{type(e).__name__}: {e}"
                await _emit(on_event, "llm_error", {"error": result.error, "round": round_no})
                break
            msg = response.message
            result.rounds = round_no
            thinking = (getattr(msg, "thinking", None) or "").strip()
            content = (msg.content or "").strip()
            calls = list(msg.tool_calls or [])
            if thinking:
                await _emit(on_event, "thinking", {"text": thinking, "round": round_no})
                transcript.append({"role": "thinking", "content": thinking})
            if content:
                transcript.append({"role": "assistant", "content": content})
            messages.append(msg)
            if not calls:
                result.text = content
                break
            for call in calls:
                name = call.function.name
                args = call.function.arguments or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                await _emit(on_event, "tool_call", {"tool": name, "args": args, "round": round_no})
                out = await toolbox.call(name, args)
                result.tool_calls += 1
                await _emit(on_event, "tool_result", {"tool": name, "result": out, "round": round_no})
                transcript.append({"role": "tool", "name": name, "args": args, "result": out})
                messages.append({"role": "tool", "content": out, "tool_name": name})
            if content:
                # Model mówił i wołał narzędzia naraz; zapamiętujemy treść jako ostatnią wypowiedź.
                result.text = content
        result.seconds = round(time.monotonic() - start, 1)
        result.transcript = transcript
        return result

    async def complete(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        response = await self._chat(messages, tools=None, temperature=temperature, think=False)
        return (response.message.content or "").strip()
