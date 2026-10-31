"""Narzędzia grafu wiedzy dla jednego agenta (autor wpisywany automatycznie)."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .graph import ArgumentGraph
from .tools import ToolSpec, tool


class TickScratch:
    """Co agent zrobił w bieżącej turze (do polityki postów i logów)."""

    def __init__(self) -> None:
        self.graph_ops: list[str] = []
        self.posts: int = 0
        self.web_calls: int = 0

    def reset(self) -> None:
        self.graph_ops.clear()
        self.posts = 0
        self.web_calls = 0


def make_kb_tools(graph: ArgumentGraph, author: str, scratch: TickScratch) -> list[ToolSpec]:
    @tool
    def kb_digest() -> str:
        """Skrót stanu grafu wiedzy: pytania otwarte, tezy bez dowodu, tezy bez zarzutów, zarzuty bez odpowiedzi, ostatnie zmiany."""
        return graph.digest()

    @tool
    def kb_read(node_id: Annotated[str, Field(description="ID węzła, np. Q3, C12, O4")]) -> str:
        """Pełny widok węzła grafu: treść, status, dowody, zarzuty, odparcia, powiązania."""
        return graph.node_view(node_id)

    @tool
    def kb_search(query: Annotated[str, Field(description="słowa kluczowe")]) -> str:
        """Szuka węzłów grafu po słowach kluczowych. Użyj, zanim dodasz coś, co może już istnieć."""
        rows = graph.db.search_nodes(query)
        if not rows:
            return "brak pasujących węzłów"
        return "\n".join(f"{r['id']} [{r['type']}, {r['status']}] {r['text'][:200]}" for r in rows)

    @tool
    def kb_add_question(
        text: Annotated[str, Field(description="pełne pytanie ze znakiem zapytania")],
        parent_id: Annotated[str, Field(description="ID pytania-rodzica (Q...), jeśli to podpytanie lub kruks")] = "",
    ) -> str:
        """Dodaje nowe pytanie do drzewa. Podpytanie musi być głębsze niż rodzic, nie jego parafrazą."""
        qid = graph.add_question(text, author, parent_id=parent_id or None)
        scratch.graph_ops.append(qid)
        return f"dodano pytanie {qid}"

    @tool
    def kb_decompose(
        question_id: Annotated[str, Field(description="ID pytania (Q...) do rozbicia")],
        subquestions: Annotated[list[str], Field(description="2 do 6 podpytań, konkretnych i rozstrzygalnych, w kolejności: co ustalić najpierw")],
    ) -> str:
        """Rozbija pytanie na podpytania (dzieci w drzewie pytań)."""
        ids = graph.decompose(question_id, subquestions, author)
        scratch.graph_ops.extend(ids)
        return f"dodano podpytania: {', '.join(ids)}"

    @tool
    def kb_add_claim(
        text: Annotated[str, Field(description="teza jako zdanie oznajmujące, sprawdzalne")],
        answers: Annotated[str, Field(description="ID pytania (Q...), na które teza odpowiada; pusty, jeśli pomocnicza")] = "",
        confidence: Annotated[float, Field(description="twoja pewność 0..1 (szczerze; bez dowodu i tak zostanie obcięta do 0.3)")] = 0.5,
    ) -> str:
        """Dodaje tezę. Status (proposed/accepted/contested/refuted) wylicza graf, nie ty."""
        cid = graph.add_claim(text, author, answers=answers or None, confidence=confidence)
        scratch.graph_ops.append(cid)
        return f"dodano tezę {cid} (status: proposed, dopóki nie ma dowodu)"

    @tool
    def kb_add_evidence(
        claim_id: Annotated[str, Field(description="ID tezy (C...)")],
        text: Annotated[str, Field(description="co źródło/dane faktycznie mówią, wiernie, 1-3 zdania")],
        url: Annotated[str, Field(description="URL źródła (wymagany dla kind=source)")] = "",
        kind: Annotated[str, Field(description="source | data | observation")] = "source",
    ) -> str:
        """Dodaje dowód wspierający tezę. Nie naciągaj: jeśli źródło nie potwierdza tezy, użyj kb_object."""
        eid = graph.add_evidence(claim_id, text, author, url=url, kind=kind)
        scratch.graph_ops.append(eid)
        node = graph.db.get_node(claim_id.strip().upper())
        return f"dodano dowód {eid}; teza {claim_id.upper()} ma teraz status {node['status'] if node else '?'}"

    @tool
    def kb_infer(
        premise_ids: Annotated[list[str], Field(description="ID tez-przesłanek (C...), 1 do 6")],
        rule: Annotated[str, Field(description="deduction | induction | abduction | analogy")],
        conclusion: Annotated[str, Field(description="wniosek jako zdanie oznajmujące")],
        answers: Annotated[str, Field(description="ID pytania (Q...), na które wniosek odpowiada, opcjonalnie")] = "",
    ) -> str:
        """Buduje jawne wnioskowanie: przesłanki + reguła → wniosek (wniosek staje się tezą, jeśli jej nie ma)."""
        iid, cid = graph.infer(premise_ids, rule, conclusion, author, answers=answers or None)
        scratch.graph_ops.extend([iid, cid])
        node = graph.db.get_node(iid)
        return f"dodano wnioskowanie {iid} (status {node['status'] if node else '?'}) z wnioskiem {cid}. Wnioskowanie jest ważne dopiero, gdy wszystkie przesłanki są accepted."

    @tool
    def kb_object(
        target_id: Annotated[str, Field(description="ID tezy (C...) lub wnioskowania (I...)")],
        text: Annotated[str, Field(description="konkretny zarzut: kontrprzykład, błąd, słabość źródła, ukryte założenie")],
    ) -> str:
        """Podnosi zarzut wobec tezy albo wobec ogniwa wnioskowania. Zarzut bez odparcia czyni cel spornym."""
        oid = graph.object(target_id, text, author)
        scratch.graph_ops.append(oid)
        return f"dodano zarzut {oid} wobec {target_id.upper()}"

    @tool
    def kb_rebut(
        objection_id: Annotated[str, Field(description="ID zarzutu (O...)")],
        text: Annotated[str, Field(description="odparcie: dlaczego zarzut nie trafia albo został usunięty")],
    ) -> str:
        """Odpiera zarzut. Po odparciu cel wraca do accepted, a sędzia może rozstrzygnąć spór."""
        rid = graph.rebut(objection_id, text, author)
        scratch.graph_ops.append(rid)
        return f"dodano odparcie {rid}"

    @tool
    def kb_define(
        term: Annotated[str, Field(description="termin")],
        definition: Annotated[str, Field(description="definicja operacyjna, 1-3 zdania")],
    ) -> str:
        """Definiuje termin używany w dyskusji (albo dodaje alternatywną definicję, jeśli jest spór słowny)."""
        did = graph.define(term, definition, author)
        scratch.graph_ops.append(did)
        return f"dodano definicję {did}"

    @tool
    def kb_assume(
        text: Annotated[str, Field(description="założenie, które jest przyjmowane milcząco")],
        used_by: Annotated[str, Field(description="ID tezy (C...) lub wnioskowania (I...), które z niego korzysta")],
    ) -> str:
        """Nazywa ukryte założenie stojące za tezą lub wnioskowaniem."""
        aid = graph.assume(text, used_by, author)
        scratch.graph_ops.append(aid)
        return f"dodano założenie {aid}"

    @tool
    def kb_rule(
        objection_id: Annotated[str, Field(description="ID zarzutu (O...)")],
        decision: Annotated[str, Field(description="sustained (zarzut podtrzymany, cel obalony) | overruled (zarzut oddalony)")],
        reason: Annotated[str, Field(description="uzasadnienie odwołujące się do konkretnych węzłów")],
    ) -> str:
        """Rozstrzyga zarzut (tylko dla roli sędziego). Podtrzymany zarzut obala cel, oddalony przywraca go."""
        if author != "sedzia":
            return "BŁĄD: rozstrzyganie zarzutów należy do roli sedzia. Możesz napisać do niego DM z prośbą o rozstrzygnięcie."
        oid = graph.rule(objection_id, decision, reason, author)
        scratch.graph_ops.append(oid)
        return f"rozstrzygnięto {oid}: {decision}"

    @tool
    def kb_mark_question(
        question_id: Annotated[str, Field(description="ID pytania (Q...)")],
        status: Annotated[str, Field(description="answered | undecidable | open")],
        reason: Annotated[str, Field(description="uzasadnienie")],
    ) -> str:
        """Oznacza pytanie jako odpowiedziane (wymaga przyjętej tezy i zamkniętych podpytań) albo nierozstrzygalne."""
        if author != "sedzia":
            return "BŁĄD: oznaczanie statusu pytań należy do roli sedzia. Napisz do niego DM."
        qid = graph.mark_question(question_id, status, reason, author)
        scratch.graph_ops.append(qid)
        return f"pytanie {qid}: {status}"

    return [
        kb_digest, kb_read, kb_search, kb_add_question, kb_decompose, kb_add_claim, kb_add_evidence,
        kb_infer, kb_object, kb_rebut, kb_define, kb_assume, kb_rule, kb_mark_question,
    ]
