"""Backend OpenAI na Responses API (/v1/responses): rozumowanie modelu działa razem z narzędziami.

Ta sama pętla i te same zdarzenia co OllamaChat. Wybór: --model luna, klucz OPENAI_API_KEY w .env.
Kolejne rundy tury łączy previous_response_id, więc wysyłamy tylko nowe elementy (wyniki narzędzi).
"""

from __future__ import annotations

import os
import time
from typing import Any

import openai

from .llm import ChatResult, EventSink, _emit, add_usage, broken_key_message, mask_key, with_heartbeat
from .tools import Toolbox

MAX_OUTPUT_TOKENS = 16000  # obejmuje też tokeny rozumowania
# Rozumowanie dla wieloetapowej pracy z narzędziami; streszczenie myślenia trafia do konsoli jako 💭.
REASONING = {"effort": "medium", "summary": "auto"}

# Znane nazwy GPT Luna w API, od najnowszej.
LUNA_NAMES = ("gpt-6-luna", "gpt-5.6-luna")


def to_responses_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Schemat w formacie chat completions ({"type": "function", "function": {...}}) → format Responses API."""
    out = []
    for s in schemas:
        fn = s.get("function", s)
        out.append({
            "type": "function",
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            "strict": False,  # pola opcjonalne w schematach narzędzi; strict wymagałby wszystkich pól
        })
    return out


class OpenAIChat:
    """Temperatura z ról nie jest wysyłana: modele rozumujące przyjmują tylko domyślną."""

    def __init__(self, model: str, *, timeout: float = 600.0, client: Any | None = None):
        self.model = model
        if client is None:
            if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPENAI_BASE_URL"):
                raise SystemExit("OpenAI: brak klucza. Skopiuj .env z projektu, w którym jest OPENAI_API_KEY, do folderu agora.")
            client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "brak", timeout=timeout)
        self._client = client
        self._reasoning: dict[str, str] | None = dict(REASONING)

    async def _create(self, **kwargs: Any) -> Any:
        """responses.create; gdy konto lub model nie przyjmuje streszczeń albo ustawień rozumowania, ponawia bez nich."""
        while True:
            extra = {"reasoning": self._reasoning} if self._reasoning else {}
            try:
                return await self._client.responses.create(**kwargs, **extra)
            except openai.BadRequestError as e:
                msg = str(e).lower()
                if self._reasoning and "summary" in self._reasoning and ("summar" in msg or "verif" in msg):
                    self._reasoning = {k: v for k, v in self._reasoning.items() if k != "summary"}
                    continue
                if self._reasoning and "reasoning" in msg:
                    self._reasoning = None
                    continue
                raise

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
        tools = to_responses_tools(toolbox.schemas())
        new_input: list[dict[str, Any]] = [{"role": "user", "content": user}]
        previous_id: str | None = None
        for round_no in range(1, max_rounds + 2):
            # Ostatnia runda bez narzędzi: wymuszamy zamknięcie tury tekstem.
            last_round = round_no > max_rounds
            kwargs: dict[str, Any] = {
                "model": self.model,
                "instructions": system,  # przy previous_response_id instrukcje nie przechodzą same, podajemy je co rundę
                "input": new_input,
                "tools": tools,
                "tool_choice": "none" if last_round else "auto",
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "store": True,  # potrzebne do previous_response_id
            }
            if previous_id:
                kwargs["previous_response_id"] = previous_id
            try:
                response = await with_heartbeat(self._create(**kwargs), on_event, round_no)
            except openai.AuthenticationError:
                result.error = "OpenAI odrzucił klucz OPENAI_API_KEY"
            except openai.NotFoundError:
                result.error = f"model {self.model!r} nie istnieje albo klucz nie ma do niego dostępu"
            except openai.APIError as e:
                result.error = f"{type(e).__name__}: {e}"
            if result.error:
                await _emit(on_event, "llm_error", {"error": result.error, "round": round_no})
                break

            previous_id = response.id
            result.rounds = round_no
            u = response.usage
            if u is not None:
                details = getattr(u, "input_tokens_details", None)
                add_usage(result.usage, **{"in": u.input_tokens, "out": u.output_tokens, "cache_read": getattr(details, "cached_tokens", None)})
            if getattr(response, "error", None):
                result.error = f"OpenAI: {getattr(response.error, 'message', response.error)}"
                await _emit(on_event, "llm_error", {"error": result.error, "round": round_no})
                break

            texts: list[str] = []
            calls: list[Any] = []
            refusal = ""
            for item in response.output or []:
                if item.type == "reasoning":
                    summary = "\n".join((s.text or "").strip() for s in (item.summary or []) if getattr(s, "text", None)).strip()
                    if summary:
                        await _emit(on_event, "thinking", {"text": summary, "round": round_no})
                        transcript.append({"role": "thinking", "content": summary})
                elif item.type == "message":
                    for part in item.content or []:
                        if part.type == "output_text" and (part.text or "").strip():
                            texts.append(part.text.strip())
                        elif part.type == "refusal":
                            refusal = getattr(part, "refusal", "") or "odmowa"
                elif item.type == "function_call":
                    calls.append(item)
            if refusal:
                result.error = f"model odmówił: {refusal}"
                await _emit(on_event, "llm_error", {"error": result.error, "round": round_no})
                break
            content = "\n".join(texts)
            if content:
                transcript.append({"role": "assistant", "content": content})
                result.text = content
            if response.status == "incomplete":
                reason = getattr(getattr(response, "incomplete_details", None), "reason", None) or "?"
                result.error = f"odpowiedź niepełna ({reason})"
                await _emit(on_event, "llm_error", {"error": result.error, "round": round_no})
                break
            if not calls:
                break
            new_input = []
            for call in calls:
                raw = call.arguments or "{}"
                await _emit(on_event, "tool_call", {"tool": call.name, "args": raw, "round": round_no})
                out = await toolbox.call(call.name, raw)  # Toolbox sam parsuje JSON i zwraca BŁĄD przy złym
                result.tool_calls += 1
                await _emit(on_event, "tool_result", {"tool": call.name, "result": out, "round": round_no})
                transcript.append({"role": "tool", "name": call.name, "args": raw, "result": out})
                new_input.append({"type": "function_call_output", "call_id": call.call_id, "output": out})
        result.seconds = round(time.monotonic() - start, 1)
        result.transcript = transcript
        return result

    async def complete(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        response = await self._client.responses.create(model=self.model, instructions=system, input=user, store=False, max_output_tokens=4000)
        return (response.output_text or "").strip()

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
