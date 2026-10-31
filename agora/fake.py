"""Tryb --dry-run: skryptowany "model", który wywołuje narzędzia bez Ollamy i bez sieci.

Sprawdza okablowanie całej społeczności (kanały, graf, DM-y, pamięć, podgląd). Treści są umowne.
"""

from __future__ import annotations

import re
import time
from typing import Any

from .llm import ChatResult, EventSink, _emit
from .tools import Toolbox

SRC = "https://example.org/dry-run/"


def _plan(agent: str, n: int, user: str) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    """Zwraca (myśl, lista wywołań narzędzi) dla roli i numeru tury."""
    dm_ids = re.findall(r"DM #(\d+) od (\w+)", user.split("[ODPOWIEDZI NA TWOJE DM-y]")[0])
    calls: list[tuple[str, dict[str, Any]]] = []
    thought = f"({agent}, tura {n}) patrzę na graf i kanały, decyduję co ma sens."
    for dm_id, sender in dm_ids[:1]:
        calls.append(("dm_reply", {"dm_id": int(dm_id), "text": f"Odpowiadam {sender}: patrz węzeł C1 i zarzut O1, rozstrzygnę po odparciu."}))
    if agent == "dekompozytor":
        if n == 1:
            calls += [
                ("kb_decompose", {"question_id": "Q1", "subquestions": [
                    "Jakie kryteria pozwalają uznać wynik za nową wiedzę naukową, a nie rekombinację?",
                    "Czy udokumentowane odkrycia z udziałem LLM spełniają te kryteria?",
                ]}),
                ("open_thread", {"title": "Kryteria nowej wiedzy", "first_post": "Proponuję zacząć od Q4: bez kryteriów nie rozstrzygniemy Q1. Metodolog, podasz definicję operacyjną?", "question_id": "Q4"}),
            ]
        else:
            calls.append(("idle", {"reason": "drzewo pytań jest w porządku, czekam na tezy."}))
    elif agent == "badacz_a":
        if n == 1:
            calls += [
                ("kb_add_claim", {"text": "Istnieją opublikowane przypadki, w których model językowy zaproponował hipotezę później potwierdzoną eksperymentalnie.", "answers": "Q3", "confidence": 0.6}),
                ("kb_add_evidence", {"claim_id": "C1", "text": "Raport opisuje hipotezę wygenerowaną przez model i jej eksperymentalne potwierdzenie przez zespół badawczy.", "url": SRC + "raport-1", "kind": "source"}),
                ("post", {"channel": "zrodla", "text": "Dodałem C1 z dowodem E1 (raport o hipotezie potwierdzonej eksperymentalnie). Pytanie do sceptyka: czy wkład modelu był tam istotny, czy rutynowy?"}),
            ]
        else:
            calls.append(("post", {"channel": "ogolny", "text": "Szukam dalszych źródeł do C1; na razie jedno. Czy ktoś zna przypadek przeciwny do C1?"}))
    elif agent == "badacz_b":
        if n == 1:
            calls += [
                ("kb_add_claim", {"text": "W większości opisywanych przypadków wkład modelu polegał na przeszukaniu i połączeniu istniejącej literatury, a nie na nowym eksperymencie.", "answers": "Q3", "confidence": 0.5}),
                ("kb_add_evidence", {"claim_id": "C2", "text": "Analiza przypadków wskazuje, że modele proponowały kombinacje znanych wyników bez nowych danych.", "url": SRC + "analiza-2", "kind": "source"}),
            ]
        else:
            calls.append(("idle", {"reason": "brak nowych przypadków do dodania."}))
    elif agent == "metodolog":
        if n == 1:
            calls += [
                ("kb_define", {"term": "nowa wiedza naukowa", "definition": "Twierdzenie, które nie wynika trywialnie z literatury, jest sprawdzalne i zostało potwierdzone niezależnym eksperymentem lub danymi."}),
                ("post", {"channel": "t-kryteria-nowej-wiedzy", "text": "Dodałem D1 jako definicję operacyjną nowej wiedzy. Jeśli C1 ma się bronić, dowód E1 musi pokazać niezależne potwierdzenie, nie tylko sformułowanie hipotezy."}),
            ]
        else:
            calls.append(("idle", {"reason": "definicje są na miejscu."}))
    elif agent == "sceptyk":
        if n == 1:
            calls += [
                ("kb_object", {"target_id": "C1", "text": "Pojedynczy raport nie pokazuje, że hipoteza nie była obecna w literaturze; brak sprawdzenia nowości względem wcześniejszych prac."}),
                ("post", {"channel": "ogolny", "text": "Podniosłem O1 wobec C1: raport nie weryfikuje nowości hipotezy względem literatury. Badacz_a, masz źródło, które to sprawdza?"}),
            ]
        else:
            calls.append(("idle", {"reason": "czekam na odparcie O1."}))
    elif agent == "obronca":
        if n == 1:
            calls += [
                ("kb_rebut", {"objection_id": "O1", "text": "Zarzut dotyczy nowości, a C1 mówi tylko o potwierdzeniu hipotezy; nowość rozstrzyga osobne pytanie Q4, więc O1 trafia obok tezy."}),
                ("post", {"channel": "ogolny", "text": "Odparłem O1 przez R1: C1 nie twierdzi o nowości, to jest przedmiot Q4. Sędzio, czy rozstrzygniesz O1?"}),
            ]
        else:
            calls.append(("idle", {"reason": "nie ma nowych zarzutów do odparcia."}))
    elif agent == "logik":
        if n == 1:
            calls.append(("kb_infer", {"premise_ids": ["C1", "C2"], "rule": "induction", "conclusion": "Modele językowe współtworzą odkrycia głównie przez syntezę literatury, a potwierdzenie eksperymentalne pozostaje po stronie ludzi.", "answers": "Q1"}))
        else:
            calls.append(("kb_assume", {"text": "Zakłada się, że opisane przypadki są reprezentatywne dla wszystkich zastosowań modeli w nauce.", "used_by": "I1"}))
    elif agent == "empiryk":
        if n == 1:
            calls.append(("post", {"channel": "ogolny", "text": "Dla C2 brakuje warunku falsyfikacji: jaki przypadek by ją obalił? Proponuję, żeby badacz_b wskazał, co by musiało wystąpić w danych."}))
        else:
            calls.append(("remember", {"text": "C2 wymaga warunku falsyfikacji; wrócić do tego, gdy pojawią się nowe dowody."}))
    elif agent == "eksplorator":
        if n == 1:
            calls += [
                ("dm", {"to": "sedzia", "text": "Czy O1 wobec C1 jest już gotowe do rozstrzygnięcia po odparciu R1?"}),
                ("kb_add_question", {"text": "Jaki test odróżniłby hipotezę wygenerowaną z literatury od hipotezy wymagającej nowych danych?", "parent_id": "Q4"}),
            ]
        else:
            calls.append(("idle", {"reason": "kruks jest zadany, czekam na odpowiedzi."}))
    elif agent == "sedzia":
        if n >= 2:
            calls += [
                ("kb_rule", {"objection_id": "O1", "decision": "overruled", "reason": "Zarzut O1 dotyczy nowości, podczas gdy C1 orzeka wyłącznie o potwierdzeniu; R1 trafnie to pokazuje."}),
                ("post", {"channel": "ogolny", "text": "Rozstrzygnąłem O1: overruled. Stan: C1 accepted z E1, C2 accepted z E2, I1 ważne. Pytania otwarte: Q4, Q5, Q6."}),
            ]
        else:
            calls.append(("idle", {"reason": "spór wokół O1 jeszcze trwa."}))
    else:
        calls.append(("idle", {"reason": "rola bez skryptu w trybie dry-run."}))
    return thought, calls


class FakeChat:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    async def run(self, *, system: str, user: str, toolbox: Toolbox, temperature: float, on_event: EventSink, max_rounds: int = 6, agent: str = "") -> ChatResult:
        start = time.monotonic()
        n = self.counts[agent] = self.counts.get(agent, 0) + 1
        thought, calls = _plan(agent, n, user)
        await _emit(on_event, "thinking", {"text": thought, "round": 1})
        result = ChatResult(text="")
        transcript: list[dict[str, Any]] = [{"role": "thinking", "content": thought}]
        for name, args in calls:
            if not toolbox.has(name):
                continue
            await _emit(on_event, "tool_call", {"tool": name, "args": args, "round": 1})
            out = await toolbox.call(name, args)
            result.tool_calls += 1
            await _emit(on_event, "tool_result", {"tool": name, "result": out, "round": 1})
            transcript.append({"role": "tool", "name": name, "args": args, "result": out})
        result.rounds = 1
        result.text = f"Tura {n}: wykonałem {result.tool_calls} operacji zgodnie z rolą ({agent})."
        result.seconds = round(time.monotonic() - start, 2)
        result.transcript = transcript
        return result

    async def complete(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        return "Streszczenie (dry run): notatki zwinięte."
