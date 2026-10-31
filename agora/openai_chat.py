"""Backend OpenAI: zwykłe chat completions z narzędziami. Ta sama pętla i te same zdarzenia co OllamaChat.

Wybór: --model luna, klucz OPENAI_API_KEY w .env.
OPENAI_BASE_URL pozwala wskazać dowolny serwer zgodny z API OpenAI.
"""

from __future__ import annotations

import os
import time
from typing import Any

import openai

from .llm import ChatResult, EventSink, _emit, add_usage, broken_key_message, mask_key, with_heartbeat
from .tools import Toolbox


# Znane nazwy GPT Luna w API, od najnowszej.
LUNA_NAMES = ("gpt-6-luna", "gpt-5.6-luna")


class OpenAIChat:
    """Temperatura z ról nie jest wysyłana: modele rozumujące (gpt-5, o-seria) przyjmują tylko domyślną."""

    def __init__(self, model: str, *, timeout: float = 600.0, client: Any | None = None):
        self.model = model
        if client is None:
            if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPENAI_BASE_URL"):
                raise SystemExit("OpenAI: brak klucza. Skopiuj .env z projektu, w którym jest OPENAI_API_KEY, do folderu agora.")
            # Serwer lokalny (OPENAI_BASE_URL) zwykle nie wymaga klucza, ale SDK chce niepustej wartości.
            client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "brak", timeout=timeout)
        self._client = client
        # GPT Luna w chat completions przyjmuje narzędzia tylko z reasoning_effort="none".
        # Model, który tego pola nie zna, dostanie zapytanie bez niego (patrz _create).
        self._reasoning_none = True

    async def _create(self, **kwargs: Any) -> Any:
        if self._reasoning_none:
            try:
                return await self._client.chat.completions.create(**kwargs, reasoning_effort="none")
            except openai.BadRequestError as e:
                if "reasoning_effort" not in str(e):
                    raise
                self._reasoning_none = False
        return await self._client.chat.completions.create(**kwargs)

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
        result = ChatResult(text="")
        transcript: list[dict[str, Any]] = []
        tools = toolbox.schemas()  # już w formacie {"type": "function", "function": {...}}
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for round_no in range(1, max_rounds + 2):
            # Ostatnia runda bez narzędzi: wymuszamy zamknięcie tury tekstem.
            last_round = round_no > max_rounds
            try:
                response = await with_heartbeat(
                    self._create(
                        model=self.model,
                        messages=messages,
                        tools=tools,
                        tool_choice="none" if last_round else "auto",
                    ),
                    on_event,
                    round_no,
                )
            except openai.AuthenticationError:
                result.error = "zły klucz OPENAI_API_KEY"
            except openai.NotFoundError:
                result.error = f"model {self.model!r} nie istnieje albo klucz nie ma do niego dostępu"
            except openai.APIError as e:
                result.error = f"{type(e).__name__}: {e}"
            if result.error:
                await _emit(on_event, "llm_error", {"error": result.error, "round": round_no})
                break

            result.rounds = round_no
            u = response.usage
            if u is not None:
                details = getattr(u, "prompt_tokens_details", None)
                add_usage(result.usage, **{"in": u.prompt_tokens, "out": u.completion_tokens, "cache_read": getattr(details, "cached_tokens", None)})
            choice = response.choices[0]
            msg = choice.message
            if getattr(msg, "refusal", None):
                result.error = f"model odmówił: {msg.refusal}"
                await _emit(on_event, "llm_error", {"error": result.error, "round": round_no})
                break
            content = (msg.content or "").strip()
            calls = list(msg.tool_calls or [])
            if content:
                transcript.append({"role": "assistant", "content": content})
                result.text = content
            assistant: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if calls:
                assistant["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]
            messages.append(assistant)
            if choice.finish_reason == "length":
                result.error = "odpowiedź ucięta na limicie długości"
                await _emit(on_event, "llm_error", {"error": result.error, "round": round_no})
                break
            if not calls:
                break
            for call in calls:
                name = call.function.name
                raw = call.function.arguments or "{}"
                await _emit(on_event, "tool_call", {"tool": name, "args": raw, "round": round_no})
                out = await toolbox.call(name, raw)  # Toolbox sam parsuje JSON i zwraca BŁĄD przy złym
                result.tool_calls += 1
                await _emit(on_event, "tool_result", {"tool": name, "result": out, "round": round_no})
                transcript.append({"role": "tool", "name": name, "args": raw, "result": out})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
        result.seconds = round(time.monotonic() - start, 1)
        result.transcript = transcript
        return result

    async def complete(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (response.choices[0].message.content or "").strip()

    async def check(self) -> None:
        """Sprawdza klucz i model przed startem (bez kosztu tokenów)."""
        if (msg := broken_key_message("OPENAI_API_KEY")):
            raise SystemExit(msg)
        try:
            await self._client.models.retrieve(self.model)
            return
        except openai.NotFoundError:
            pass  # niżej: szukamy tej rodziny modeli na koncie
        except openai.AuthenticationError:
            raise SystemExit(f"OpenAI odrzucił klucz {mask_key(os.environ.get('OPENAI_API_KEY'))} (unieważniony, wygasły albo z innego konta). Nowy klucz: https://platform.openai.com/api-keys")
        except openai.APIConnectionError as e:
            raise SystemExit(f"OpenAI: brak połączenia ({type(e).__name__}). Sprawdź internet / OPENAI_BASE_URL.")
        found = await self._find_on_account()
        if found is None:
            raise SystemExit(f"OpenAI: na tym koncie nie ma modelu {self.model!r} ani żadnej innej wersji Luny.")
        self.model = found

    async def _find_on_account(self) -> str | None:
        """Model z tej samej rodziny dostępny na koncie: najpierw znane nazwy, potem dowolna Luna (bez wariantu -fast, droższego 2x)."""
        ids = [m.id async for m in self._client.models.list()]
        for name in LUNA_NAMES:
            if name in ids:
                return name
        lunas = sorted((i for i in ids if "luna" in i.lower()), key=lambda i: (i.lower().endswith("-fast"), i), reverse=False)
        return lunas[0] if lunas else None
