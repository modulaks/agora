"""Backend Claude (Anthropic API): ta sama pętla tool-calling co OllamaChat, raportuje te same zdarzenia.

Wybór: --model sonnet, klucz ANTHROPIC_API_KEY w .env. Płatne za tokeny.
"""

from __future__ import annotations

import os
import time
from typing import Any

import anthropic

from .llm import ChatResult, EventSink, _emit, add_usage, mask_key, with_heartbeat
from .tools import Toolbox

MAX_TOKENS = 16000
# Modele z serwerowym fallbackiem przy odmowie (Haiku go nie ma).
FALLBACK_MODELS = ("claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1")


def to_anthropic_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Schemat w formacie Ollamy/OpenAI → format Anthropic (input_schema)."""
    out = []
    for s in schemas:
        fn = s.get("function", s)
        out.append({"name": fn["name"], "description": fn.get("description", ""), "input_schema": fn.get("parameters") or {"type": "object", "properties": {}}})
    return out


class AnthropicChat:
    """Temperatura z ról jest ignorowana: obecne modele Claude jej nie przyjmują. Głębię myślenia ustawia effort."""

    def __init__(self, model: str, *, effort: str = "medium", timeout: float = 600.0, client: Any | None = None):
        self.model = model
        self.effort = effort
        self._client = client or anthropic.AsyncAnthropic(timeout=timeout)

    def _extra(self) -> dict[str, Any]:
        if self.model.startswith(FALLBACK_MODELS):
            return {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        return {}

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
        tools = to_anthropic_tools(toolbox.schemas())
        # System i narzędzia są stałe dla agenta: znacznik cache obejmuje je między turami.
        system_blocks = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        for round_no in range(1, max_rounds + 2):
            # Ostatnia runda bez narzędzi (tool_choice none): wymuszamy zamknięcie tury tekstem.
            last_round = round_no > max_rounds
            try:
                response = await with_heartbeat(
                    self._client.beta.messages.create(
                        model=self.model,
                        max_tokens=MAX_TOKENS,
                        system=system_blocks,
                        messages=messages,
                        tools=tools,
                        tool_choice={"type": "none"} if last_round else {"type": "auto"},
                        thinking={"type": "adaptive", "display": "summarized"},
                        output_config={"effort": self.effort},
                        cache_control={"type": "ephemeral"},
                        **self._extra(),
                    ),
                    on_event,
                    round_no,
                )
            except anthropic.AuthenticationError:
                result.error = "zły lub brakujący ANTHROPIC_API_KEY w .env"
            except anthropic.NotFoundError:
                result.error = f"model {self.model!r} nie istnieje albo klucz nie ma do niego dostępu"
            except anthropic.APIError as e:
                result.error = f"{type(e).__name__}: {e}"
            if result.error:
                await _emit(on_event, "llm_error", {"error": result.error, "round": round_no})
                break

            result.rounds = round_no
            u = response.usage
            add_usage(result.usage, **{
                "in": u.input_tokens, "out": u.output_tokens,
                "cache_read": getattr(u, "cache_read_input_tokens", None), "cache_write": getattr(u, "cache_creation_input_tokens", None),
            })
            if response.stop_reason == "refusal":
                result.error = "model odmówił odpowiedzi (stop_reason=refusal)"
                await _emit(on_event, "llm_error", {"error": result.error, "round": round_no})
                break

            texts: list[str] = []
            calls = []
            for block in response.content:
                if block.type == "thinking" and (block.thinking or "").strip():
                    await _emit(on_event, "thinking", {"text": block.thinking.strip(), "round": round_no})
                    transcript.append({"role": "thinking", "content": block.thinking.strip()})
                elif block.type == "text" and block.text.strip():
                    texts.append(block.text.strip())
                elif block.type == "tool_use":
                    calls.append(block)
            content = "\n".join(texts)
            if content:
                transcript.append({"role": "assistant", "content": content})
                result.text = content
            # Cała odpowiedź wraca do historii bez zmian (bloki thinking muszą zostać nietknięte).
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "max_tokens":
                result.error = f"odpowiedź ucięta na limicie {MAX_TOKENS} tokenów"
                await _emit(on_event, "llm_error", {"error": result.error, "round": round_no})
                break
            if not calls:
                break
            tool_results = []
            for call in calls:
                args = call.input if isinstance(call.input, dict) else {}
                await _emit(on_event, "tool_call", {"tool": call.name, "args": args, "round": round_no})
                out = await toolbox.call(call.name, args)
                result.tool_calls += 1
                await _emit(on_event, "tool_result", {"tool": call.name, "result": out, "round": round_no})
                transcript.append({"role": "tool", "name": call.name, "args": args, "result": out})
                tool_results.append({"type": "tool_result", "tool_use_id": call.id, "content": out, "is_error": out.startswith("BŁĄD")})
            # Wszystkie wyniki w jednej wiadomości.
            messages.append({"role": "user", "content": tool_results})
        result.seconds = round(time.monotonic() - start, 1)
        result.transcript = transcript
        return result

    async def complete(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        response = await self._client.beta.messages.create(
            model=self.model,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low"},
            **self._extra(),
        )
        return "\n".join(b.text for b in response.content if b.type == "text").strip()

    async def check(self) -> None:
        """Sprawdza klucz i model przed startem społeczności (bez kosztu tokenów)."""
        try:
            await self._client.models.retrieve(self.model)
        except TypeError:  # SDK nie znalazł żadnego klucza
            raise SystemExit("Claude API: brak klucza. Wpisz ANTHROPIC_API_KEY=... w pliku .env (klucz z https://console.anthropic.com/settings/keys).")
        except anthropic.AuthenticationError:
            raise SystemExit(f"Anthropic odrzucił klucz {mask_key(os.environ.get('ANTHROPIC_API_KEY'))} (unieważniony, wygasły albo z innego konta). Nowy klucz: https://console.anthropic.com/settings/keys")
        except anthropic.NotFoundError:
            raise SystemExit(f"Claude API: model {self.model!r} nie istnieje albo klucz nie ma do niego dostępu.")
        except anthropic.APIConnectionError as e:
            raise SystemExit(f"Claude API: brak połączenia z api.anthropic.com ({type(e).__name__}). Sprawdź internet / proxy.")
