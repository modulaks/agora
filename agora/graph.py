"""Graf argumentacji: węzły typowane, krawędzie, liczone statusy i pewność, wykrywanie powtórek.

Typy węzłów:
  Q pytanie       C teza        E dowód (supports → C)    I wnioskowanie (C premise_of → I concludes → C)
  O zarzut (objects → C|I)      R odparcie (rebuts → O)    D definicja     A założenie (assumes → C|I)

Status tezy nie jest deklarowany przez agenta, tylko wyliczany z grafu:
  refuted    - ma zarzut podtrzymany przez sędziego
  contested  - ma zarzut bez odparcia i bez rozstrzygnięcia
  accepted   - ma dowód albo ważne wnioskowanie i nie ma otwartych zarzutów
  proposed   - nic z powyższych
Zmiana jednego węzła przelicza wszystko (recompute).
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Callable

from .db import DB
from .tools import ToolError

RULE_WEIGHTS = {"deduction": 0.95, "induction": 0.70, "abduction": 0.50, "analogy": 0.40}
RULE_ALIASES = {
    "dedukcja": "deduction", "deduction": "deduction",
    "indukcja": "induction", "induction": "induction",
    "abdukcja": "abduction", "abduction": "abduction",
    "analogia": "analogy", "analogy": "analogy",
}
TYPE_NAMES = {"Q": "pytanie", "C": "teza", "E": "dowód", "I": "wnioskowanie", "O": "zarzut", "R": "odparcie", "D": "definicja", "A": "założenie"}
NODE_ID_RE = re.compile(r"\b([QCEIORDA])(\d+)\b")

ChangeHook = Callable[[str, dict[str, Any]], None]


def _norm_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 2]


def similarity(a: str, b: str) -> float:
    ta, tb = _norm_tokens(a), _norm_tokens(b)
    if not ta or not tb:
        return 0.0
    jaccard = len(set(ta) & set(tb)) / len(set(ta) | set(tb))
    ratio = difflib.SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
    return max(jaccard, ratio)


class ArgumentGraph:
    def __init__(self, db: DB, *, dup_threshold: float = 0.72, on_change: ChangeHook | None = None):
        self.db = db
        self.dup_threshold = dup_threshold
        self.on_change = on_change
        self._pending_changes: list[tuple[str, dict[str, Any]]] = []

    # ── pomocnicze ───────────────────────────────────────────────────────

    def _require(self, node_id: str, types: str) -> dict[str, Any]:
        node_id = node_id.strip().upper()
        node = self.db.get_node(node_id)
        if node is None:
            raise ToolError(f"węzeł {node_id} nie istnieje. Użyj kb_search albo kb_digest, żeby znaleźć właściwe ID.")
        if node["type"] not in types:
            expected = " lub ".join(TYPE_NAMES[t] for t in types)
            raise ToolError(f"{node_id} to {TYPE_NAMES[node['type']]}, a tu potrzebne jest {expected}.")
        return node

    def _check_duplicate(self, node_type: str, text: str, *, extra_types: str = "") -> None:
        candidates = []
        for t in node_type + extra_types:
            candidates.extend(self.db.nodes(t))
        for cand in candidates:
            s = similarity(cand["text"], text)
            if s >= self.dup_threshold:
                raise ToolError(
                    f"to powtórzenie istniejącego węzła {cand['id']} (podobieństwo {s:.2f}): \"{cand['text'][:160]}\". "
                    f"Nie dodawaj tego samego innymi słowami. Odnieś się do {cand['id']} albo wnieś coś nowego."
                )

    @staticmethod
    def _clean(text: str, *, min_len: int = 12, field: str = "text") -> str:
        text = " ".join((text or "").split())
        if len(text) < min_len:
            raise ToolError(f"pole {field} jest za krótkie ({len(text)} znaków). Napisz pełne, konkretne zdanie.")
        return text

    def _record(self, kind: str, payload: dict[str, Any]) -> None:
        """Zmiana w grafie: od razu do hooka (dziennik #graf + konsola), synchronicznie."""
        self._pending_changes.append((kind, payload))
        if self.on_change:
            try:
                self.on_change(kind, payload)
            except Exception:  # noqa: BLE001 - hook obserwacyjny nie może psuć operacji
                pass
        if len(self._pending_changes) > 500:
            del self._pending_changes[:-200]

    # ── operacje ─────────────────────────────────────────────────────────

    def add_question(self, text: str, author: str, *, parent_id: str | None = None) -> str:
        text = self._clean(text, min_len=15)
        if "?" not in text:
            raise ToolError("pytanie musi być pytaniem (ze znakiem zapytania), nie tezą.")
        parent = None
        if parent_id:
            parent = self._require(parent_id, "Q")
            if similarity(parent["text"], text) >= 0.6:
                raise ToolError(f"to przeformułowanie pytania-rodzica {parent['id']}, a nie pytanie głębsze. Zapytaj o coś, co trzeba ustalić, żeby odpowiedzieć na rodzica.")
        self._check_duplicate("Q", text)
        qid = self.db.next_id("Q")
        depth = (int(parent["data"].get("depth", 0)) + 1) if parent else 0
        self.db.add_node(qid, "Q", text, author, data={"parent": parent["id"] if parent else None, "depth": depth, "reason": ""}, status="open")
        if parent:
            self.db.add_edge(qid, parent["id"], "child_of")
        self._record("question", {"id": qid, "text": text, "author": author, "parent": parent["id"] if parent else None})
        self.recompute()
        return qid

    def decompose(self, question_id: str, subquestions: list[str], author: str) -> list[str]:
        parent = self._require(question_id, "Q")
        if not (2 <= len(subquestions) <= 6):
            raise ToolError("podaj od 2 do 6 podpytań.")
        ids = []
        for sq in subquestions:
            ids.append(self.add_question(sq, author, parent_id=parent["id"]))
        return ids

    def add_claim(self, text: str, author: str, *, answers: str | None = None, confidence: float = 0.5) -> str:
        text = self._clean(text, min_len=15)
        if text.endswith("?"):
            raise ToolError("teza musi być zdaniem oznajmującym, które może być prawdziwe lub fałszywe. Na pytanie użyj kb_add_question.")
        self._check_duplicate("C", text)
        q = self._require(answers, "Q") if answers else None
        cid = self.db.next_id("C")
        conf = max(0.0, min(1.0, float(confidence)))
        self.db.add_node(cid, "C", text, author, data={"answers": q["id"] if q else None, "author_confidence": conf}, status="proposed", confidence=conf)
        if q:
            self.db.add_edge(cid, q["id"], "answers")
        self._record("claim", {"id": cid, "text": text, "author": author, "answers": q["id"] if q else None})
        self.recompute()
        return cid

    def add_evidence(self, claim_id: str, text: str, author: str, *, url: str = "", kind: str = "source") -> str:
        claim = self._require(claim_id, "C")
        text = self._clean(text, min_len=20)
        kind = (kind or "source").lower()
        if kind not in {"source", "data", "observation"}:
            raise ToolError("kind musi być jednym z: source, data, observation.")
        url = (url or "").strip()
        if kind == "source" and not url.startswith("http"):
            raise ToolError("dowód typu source wymaga URL (http...). Najpierw znajdź źródło (web_search / web_fetch).")
        for e in self.db.nodes("E"):
            if e["data"].get("claim") == claim["id"] and (url and e["data"].get("url") == url or similarity(e["text"], text) >= self.dup_threshold):
                raise ToolError(f"ten dowód już jest w grafie jako {e['id']}.")
        eid = self.db.next_id("E")
        self.db.add_node(eid, "E", text, author, data={"claim": claim["id"], "url": url, "kind": kind}, status="active", confidence=0.0)
        self.db.add_edge(eid, claim["id"], "supports")
        self._record("evidence", {"id": eid, "text": text, "author": author, "claim": claim["id"], "url": url, "kind": kind})
        self.recompute()
        return eid

    def infer(self, premise_ids: list[str], rule: str, conclusion: str, author: str, *, answers: str | None = None) -> tuple[str, str]:
        rule_key = RULE_ALIASES.get((rule or "").strip().lower())
        if not rule_key:
            raise ToolError("rule musi być jedną z: deduction, induction, abduction, analogy.")
        if not premise_ids or len(premise_ids) > 6:
            raise ToolError("podaj od 1 do 6 przesłanek (ID tez).")
        premises = [self._require(p, "C") for p in premise_ids]
        for p in premises:
            if p["status"] == "refuted":
                raise ToolError(f"przesłanka {p['id']} jest obalona, nie można na niej budować.")
        conclusion = self._clean(conclusion, min_len=15)
        existing = None
        for c in self.db.nodes("C"):
            if similarity(c["text"], conclusion) >= self.dup_threshold:
                existing = c
                break
        if existing is None:
            cid = self.add_claim(conclusion, author, answers=answers, confidence=0.5)
        else:
            cid = existing["id"]
        for i in self.db.nodes("I"):
            if i["data"].get("conclusion") == cid and set(i["data"].get("premises", [])) == {p["id"] for p in premises}:
                raise ToolError(f"to wnioskowanie już istnieje jako {i['id']}.")
        iid = self.db.next_id("I")
        text = f"{' + '.join(p['id'] for p in premises)} ⇒ {cid} ({rule_key})"
        self.db.add_node(iid, "I", text, author, data={"rule": rule_key, "premises": [p["id"] for p in premises], "conclusion": cid}, status="pending")
        for p in premises:
            self.db.add_edge(p["id"], iid, "premise_of")
        self.db.add_edge(iid, cid, "concludes")
        self._record("inference", {"id": iid, "text": text, "author": author, "conclusion": cid, "rule": rule_key, "premises": [p["id"] for p in premises]})
        self.recompute()
        return iid, cid

    def object(self, target_id: str, text: str, author: str) -> str:
        target = self._require(target_id, "CI")
        text = self._clean(text, min_len=20)
        for o in self.db.nodes("O"):
            if o["data"].get("target") == target["id"] and similarity(o["text"], text) >= self.dup_threshold:
                raise ToolError(f"ten zarzut już został podniesiony jako {o['id']} (status: {self._objection_state(o)}). Podnieś inny albo odnieś się do odparcia.")
        if target["author"] == author and target["type"] == "C":
            # Dozwolone (samokrytyka), ale odnotowane.
            pass
        oid = self.db.next_id("O")
        self.db.add_node(oid, "O", text, author, data={"target": target["id"], "ruling": "", "ruling_reason": ""}, status="open")
        self.db.add_edge(oid, target["id"], "objects")
        self._record("objection", {"id": oid, "text": text, "author": author, "target": target["id"]})
        self.recompute()
        return oid

    def rebut(self, objection_id: str, text: str, author: str) -> str:
        obj = self._require(objection_id, "O")
        if obj["data"].get("ruling") == "sustained":
            raise ToolError(f"zarzut {obj['id']} został podtrzymany przez sędziego. Zamiast odpierać, dodaj osłabioną wersję tezy.")
        text = self._clean(text, min_len=20)
        if obj["author"] == author:
            raise ToolError("nie odpieraj własnego zarzutu; jeśli go wycofujesz, napisz to na kanale.")
        for r in self.db.nodes("R"):
            if r["data"].get("objection") == obj["id"] and similarity(r["text"], text) >= self.dup_threshold:
                raise ToolError(f"to odparcie już istnieje jako {r['id']}.")
        rid = self.db.next_id("R")
        self.db.add_node(rid, "R", text, author, data={"objection": obj["id"]}, status="active")
        self.db.add_edge(rid, obj["id"], "rebuts")
        self._record("rebuttal", {"id": rid, "text": text, "author": author, "objection": obj["id"], "target": obj["data"].get("target")})
        self.recompute()
        return rid

    def define(self, term: str, definition: str, author: str) -> str:
        term = " ".join((term or "").split()).strip(" '\"")
        if len(term) < 2:
            raise ToolError("podaj termin.")
        definition = self._clean(definition, min_len=20, field="definition")
        for d in self.db.nodes("D"):
            if d["data"].get("term", "").lower() == term.lower():
                if similarity(d["text"], definition) >= self.dup_threshold:
                    raise ToolError(f"termin '{term}' ma już taką definicję ({d['id']}).")
        did = self.db.next_id("D")
        self.db.add_node(did, "D", definition, author, data={"term": term}, status="active")
        self._record("definition", {"id": did, "text": definition, "author": author, "term": term})
        return did

    def assume(self, text: str, used_by: str, author: str) -> str:
        target = self._require(used_by, "CI")
        text = self._clean(text, min_len=15)
        for a in self.db.nodes("A"):
            if a["data"].get("used_by") == target["id"] and similarity(a["text"], text) >= self.dup_threshold:
                raise ToolError(f"to założenie już jest nazwane jako {a['id']}.")
        aid = self.db.next_id("A")
        self.db.add_node(aid, "A", text, author, data={"used_by": target["id"]}, status="active")
        self.db.add_edge(aid, target["id"], "assumes")
        self._record("assumption", {"id": aid, "text": text, "author": author, "used_by": target["id"]})
        return aid

    def rule(self, objection_id: str, decision: str, reason: str, author: str) -> str:
        obj = self._require(objection_id, "O")
        decision = (decision or "").strip().lower()
        if decision not in {"sustained", "overruled"}:
            raise ToolError("decision musi być sustained (zarzut podtrzymany) albo overruled (zarzut oddalony).")
        reason = self._clean(reason, min_len=20, field="reason")
        if obj["data"].get("ruling"):
            raise ToolError(f"zarzut {obj['id']} został już rozstrzygnięty ({obj['data']['ruling']}).")
        rebuttals = [r for r in self.db.nodes("R") if r["data"].get("objection") == obj["id"]]
        if decision == "sustained" and not rebuttals:
            # Dopuszczalne, ale wymaga uzasadnienia, że obrona miała szansę.
            pass
        data = dict(obj["data"])
        data["ruling"] = decision
        data["ruling_reason"] = reason
        data["ruled_by"] = author
        self.db.update_node(obj["id"], data=data)
        self._record("ruling", {"id": obj["id"], "decision": decision, "reason": reason, "text": reason, "author": author, "target": obj["data"].get("target")})
        self.recompute()
        return obj["id"]

    def mark_question(self, question_id: str, status: str, reason: str, author: str) -> str:
        q = self._require(question_id, "Q")
        status = (status or "").strip().lower()
        if status not in {"answered", "undecidable", "open"}:
            raise ToolError("status musi być answered, undecidable albo open.")
        reason = self._clean(reason, min_len=15, field="reason")
        if status == "answered":
            answering = [c for c in self.db.nodes("C") if c["data"].get("answers") == q["id"] and c["status"] == "accepted"]
            if not answering:
                raise ToolError(f"pytanie {q['id']} nie ma przyjętej (accepted) tezy, która na nie odpowiada. Najpierw musi istnieć taka teza z dowodem i bez otwartych zarzutów.")
            open_children = [c for c in self.db.nodes("Q") if c["data"].get("parent") == q["id"] and c["status"] in {"open", "in_progress"}]
            if open_children:
                raise ToolError(f"pytanie {q['id']} ma otwarte podpytania: {', '.join(c['id'] for c in open_children)}.")
        data = dict(q["data"])
        data["manual_status"] = status
        data["reason"] = reason
        data["marked_by"] = author
        self.db.update_node(q["id"], data=data)
        self._record("question_status", {"id": q["id"], "status": status, "reason": reason, "text": reason, "author": author})
        self.recompute()
        return q["id"]

    # ── statusy ──────────────────────────────────────────────────────────

    def _objection_state(self, o: dict[str, Any]) -> str:
        ruling = o["data"].get("ruling") or ""
        if ruling:
            return ruling
        rebutted = any(r["data"].get("objection") == o["id"] for r in self.db.nodes("R"))
        return "rebutted" if rebutted else "open"

    def recompute(self) -> None:
        nodes = {n["id"]: n for n in self.db.nodes()}
        objections = [n for n in nodes.values() if n["type"] == "O"]
        rebuttals = [n for n in nodes.values() if n["type"] == "R"]
        evidence = [n for n in nodes.values() if n["type"] == "E"]
        inferences = [n for n in nodes.values() if n["type"] == "I"]
        claims = [n for n in nodes.values() if n["type"] == "C"]
        questions = [n for n in nodes.values() if n["type"] == "Q"]

        rebutted_ids = {r["data"].get("objection") for r in rebuttals}
        obj_state: dict[str, str] = {}
        for o in objections:
            ruling = o["data"].get("ruling") or ""
            state = ruling if ruling else ("rebutted" if o["id"] in rebutted_ids else "open")
            obj_state[o["id"]] = state

        def objections_on(target: str) -> list[str]:
            return [o["id"] for o in objections if o["data"].get("target") == target]

        status: dict[str, str] = {c["id"]: "proposed" for c in claims}
        conf: dict[str, float] = {c["id"]: min(float(c["data"].get("author_confidence", 0.5)), 0.3) for c in claims}
        inf_status: dict[str, str] = {i["id"]: "pending" for i in inferences}

        for _ in range(12):
            changed = False
            for i in inferences:
                prem = i["data"].get("premises", [])
                st_o = [obj_state.get(o, "open") for o in objections_on(i["id"])]
                if any(s == "sustained" for s in st_o) or any(status.get(p) == "refuted" for p in prem):
                    new = "invalid"
                elif any(s == "open" for s in st_o):
                    new = "contested"
                elif prem and all(status.get(p) == "accepted" for p in prem):
                    new = "valid"
                else:
                    new = "pending"
                if inf_status[i["id"]] != new:
                    inf_status[i["id"]] = new
                    changed = True
            for c in claims:
                cid = c["id"]
                st_o = [obj_state.get(o, "open") for o in objections_on(cid)]
                ev = [e for e in evidence if e["data"].get("claim") == cid]
                valid_inf = [i for i in inferences if i["data"].get("conclusion") == cid and inf_status[i["id"]] == "valid"]
                if any(s == "sustained" for s in st_o):
                    new, new_conf = "refuted", 0.05
                else:
                    ev_conf = 0.0
                    if ev:
                        ev_conf = min(0.85, 0.55 + 0.1 * (len(ev) - 1) + (0.05 if any(e["data"].get("kind") == "data" for e in ev) else 0.0))
                    inf_conf = 0.0
                    for i in valid_inf:
                        prem = i["data"].get("premises", [])
                        base = min(conf.get(p, 0.0) for p in prem) if prem else 0.0
                        inf_conf = max(inf_conf, base * RULE_WEIGHTS.get(i["data"].get("rule", "abduction"), 0.5))
                    supported = bool(ev) or bool(valid_inf)
                    if any(s == "open" for s in st_o):
                        new = "contested"
                        new_conf = max(0.1, max(ev_conf, inf_conf, conf[cid]) * 0.5)
                    elif supported:
                        new = "accepted"
                        new_conf = max(ev_conf, inf_conf)
                    else:
                        new = "proposed"
                        new_conf = min(float(c["data"].get("author_confidence", 0.5)), 0.3)
                new_conf = round(new_conf, 2)
                if status[cid] != new or abs(conf[cid] - new_conf) > 1e-9:
                    status[cid], conf[cid] = new, new_conf
                    changed = True
            if not changed:
                break

        for c in claims:
            if c["status"] != status[c["id"]] or abs(float(c["confidence"]) - conf[c["id"]]) > 1e-9:
                old = c["status"]
                self.db.update_node(c["id"], status=status[c["id"]], confidence=conf[c["id"]])
                if old and old != status[c["id"]]:
                    self._record("status", {"id": c["id"], "old": old, "new": status[c["id"]], "confidence": conf[c["id"]], "text": c["text"]})
        for i in inferences:
            if i["status"] != inf_status[i["id"]]:
                self.db.update_node(i["id"], status=inf_status[i["id"]])
        for o in objections:
            st = obj_state[o["id"]]
            if o["status"] != st:
                self.db.update_node(o["id"], status=st)

        # pytania
        children: dict[str, list[dict[str, Any]]] = {}
        for q in questions:
            children.setdefault(q["data"].get("parent") or "", []).append(q)
        q_status: dict[str, str] = {}
        for q in sorted(questions, key=lambda x: -int(x["data"].get("depth", 0))):
            manual = q["data"].get("manual_status") or ""
            answering = [c for c in claims if c["data"].get("answers") == q["id"]]
            accepted = [c for c in answering if status[c["id"]] == "accepted"]
            kids = children.get(q["id"], [])
            kids_done = all(q_status.get(k["id"], k["status"]) in {"answered", "undecidable"} for k in kids)
            if manual == "undecidable":
                st = "undecidable"
            elif manual == "answered" and accepted and kids_done:
                st = "answered"
            elif answering or kids:
                st = "in_progress"
            else:
                st = "open"
            q_status[q["id"]] = st
            if q["status"] != st:
                self.db.update_node(q["id"], status=st)
                if q["status"] == "answered" and st != "answered":
                    self._record("status", {"id": q["id"], "old": "answered", "new": st, "text": q["text"]})

    # ── widoki ───────────────────────────────────────────────────────────

    def node_view(self, node_id: str) -> str:
        node = self.db.get_node(node_id.strip().upper())
        if node is None:
            raise ToolError(f"węzeł {node_id} nie istnieje.")
        lines = [f"{node['id']} [{TYPE_NAMES[node['type']]}, status: {node['status'] or '-'}, pewność: {node['confidence']:.2f}, autor: {node['author']}]", node["text"]]
        d = node["data"]
        if node["type"] == "C":
            if d.get("answers"):
                lines.append(f"odpowiada na: {d['answers']}")
            ev = [e for e in self.db.nodes("E") if e["data"].get("claim") == node["id"]]
            for e in ev:
                lines.append(f"  dowód {e['id']} ({e['data'].get('kind')}): {e['text'][:200]} {e['data'].get('url', '')}".rstrip())
            for i in self.db.nodes("I"):
                if i["data"].get("conclusion") == node["id"]:
                    lines.append(f"  wynika z {i['id']} [{i['status']}]: {i['text']}")
                if node["id"] in i["data"].get("premises", []):
                    lines.append(f"  jest przesłanką w {i['id']} [{i['status']}]")
        if node["type"] == "I":
            for p in d.get("premises", []):
                pn = self.db.get_node(p)
                if pn:
                    lines.append(f"  przesłanka {p} [{pn['status']}]: {pn['text'][:160]}")
            cn = self.db.get_node(d.get("conclusion", ""))
            if cn:
                lines.append(f"  wniosek {cn['id']} [{cn['status']}]: {cn['text'][:160]}")
        if node["type"] in "CI":
            for o in self.db.nodes("O"):
                if o["data"].get("target") == node["id"]:
                    lines.append(f"  zarzut {o['id']} [{o['status']}] ({o['author']}): {o['text'][:200]}")
                    for r in self.db.nodes("R"):
                        if r["data"].get("objection") == o["id"]:
                            lines.append(f"    odparcie {r['id']} ({r['author']}): {r['text'][:200]}")
                    if o["data"].get("ruling"):
                        lines.append(f"    rozstrzygnięcie: {o['data']['ruling']} ({o['data'].get('ruled_by')}): {o['data'].get('ruling_reason', '')[:200]}")
            for a in self.db.nodes("A"):
                if a["data"].get("used_by") == node["id"]:
                    lines.append(f"  założenie {a['id']}: {a['text'][:160]}")
        if node["type"] == "O":
            t = self.db.get_node(d.get("target", ""))
            if t:
                lines.append(f"dotyczy {t['id']} [{t['status']}]: {t['text'][:160]}")
            for r in self.db.nodes("R"):
                if r["data"].get("objection") == node["id"]:
                    lines.append(f"  odparcie {r['id']} ({r['author']}): {r['text'][:200]}")
            if d.get("ruling"):
                lines.append(f"rozstrzygnięcie: {d['ruling']}: {d.get('ruling_reason', '')[:200]}")
        if node["type"] == "Q":
            if d.get("parent"):
                lines.append(f"rodzic: {d['parent']} (głębokość {d.get('depth', 0)})")
            kids = [q for q in self.db.nodes("Q") if q["data"].get("parent") == node["id"]]
            for k in kids:
                lines.append(f"  podpytanie {k['id']} [{k['status']}]: {k['text'][:160]}")
            for c in self.db.nodes("C"):
                if c["data"].get("answers") == node["id"]:
                    lines.append(f"  odpowiedź {c['id']} [{c['status']}, {c['confidence']:.2f}]: {c['text'][:160]}")
            if d.get("reason"):
                lines.append(f"uwaga: {d['reason']}")
        if node["type"] == "D":
            lines.insert(1, f"termin: {d.get('term')}")
        return "\n".join(lines)

    def digest(self, *, max_items: int = 6) -> str:
        qs = self.db.nodes("Q")
        cs = self.db.nodes("C")
        os_ = self.db.nodes("O")
        es = self.db.nodes("E")
        ds = self.db.nodes("D")
        by_status = {}
        for c in cs:
            by_status[c["status"]] = by_status.get(c["status"], 0) + 1
        lines = [
            f"GRAF WIEDZY: pytań {len(qs)} (otwarte/w toku: {sum(1 for q in qs if q['status'] in ('open', 'in_progress'))}), "
            f"tez {len(cs)} ({', '.join(f'{k} {v}' for k, v in sorted(by_status.items())) or 'brak'}), dowodów {len(es)}, zarzutów {len(os_)}, definicji {len(ds)}."
        ]
        open_q = [q for q in qs if q["status"] in ("open", "in_progress")]
        leafs = [q for q in open_q if not any(k["data"].get("parent") == q["id"] for k in qs)]
        leafs.sort(key=lambda q: (-int(q["data"].get("depth", 0)), q["id"]))
        if leafs:
            lines.append("Pytania otwarte bez podpytań (najgłębsze pierwsze):")
            for q in leafs[:max_items]:
                n_ans = sum(1 for c in cs if c["data"].get("answers") == q["id"])
                lines.append(f"  {q['id']} (gł. {q['data'].get('depth', 0)}, odpowiedzi: {n_ans}) {q['text'][:170]}")
        unattacked = [c for c in cs if c["status"] == "accepted" and not any(o["data"].get("target") == c["id"] for o in os_)]
        if unattacked:
            lines.append("Tezy przyjęte, których nikt nie podważył:")
            for c in unattacked[:max_items]:
                lines.append(f"  {c['id']} ({c['confidence']:.2f}) {c['text'][:170]}")
        unsupported = [c for c in cs if c["status"] == "proposed"]
        if unsupported:
            lines.append("Tezy bez dowodu ani wnioskowania:")
            for c in unsupported[:max_items]:
                lines.append(f"  {c['id']} ({c['author']}) {c['text'][:170]}")
        open_obj = [o for o in os_ if o["status"] == "open"]
        if open_obj:
            lines.append("Zarzuty bez odparcia i bez rozstrzygnięcia:")
            for o in open_obj[:max_items]:
                lines.append(f"  {o['id']} → {o['data'].get('target')} ({o['author']}) {o['text'][:170]}")
        rebutted = [o for o in os_ if o["status"] == "rebutted"]
        if rebutted:
            lines.append("Zarzuty odparte, czekające na rozstrzygnięcie sędziego:")
            for o in rebutted[:max_items]:
                lines.append(f"  {o['id']} → {o['data'].get('target')}: {o['text'][:120]}")
        contested = [c for c in cs if c["status"] == "contested"]
        if contested:
            lines.append("Tezy sporne:")
            for c in contested[:max_items]:
                lines.append(f"  {c['id']} {c['text'][:170]}")
        if ds:
            lines.append("Zdefiniowane terminy: " + ", ".join(f"{d['data'].get('term')} ({d['id']})" for d in ds[-12:]))
        recent = self.db.recent_nodes(6)
        if recent:
            lines.append("Ostatnio dodane: " + "; ".join(f"{n['id']} ({n['author']}) {n['text'][:70]}" for n in recent))
        return "\n".join(lines)

    def codex(self) -> str:
        qs = self.db.nodes("Q")
        cs = self.db.nodes("C")
        es = self.db.nodes("E")
        os_ = self.db.nodes("O")
        rs = self.db.nodes("R")
        is_ = self.db.nodes("I")
        ds = self.db.nodes("D")
        out = ["# CODEX: stan wiedzy społeczności", ""]
        out.append(f"Węzły: pytania {len(qs)}, tezy {len(cs)}, dowody {len(es)}, wnioskowania {len(is_)}, zarzuty {len(os_)}, odparcia {len(rs)}, definicje {len(ds)}.")
        out.append("")

        def render_q(q: dict[str, Any], indent: int) -> None:
            pad = "  " * indent
            out.append(f"{pad}- **{q['id']}** [{q['status']}] {q['text']}")
            for c in cs:
                if c["data"].get("answers") == q["id"]:
                    out.append(f"{pad}  - {c['id']} [{c['status']}, {c['confidence']:.2f}] {c['text']}")
            for k in qs:
                if k["data"].get("parent") == q["id"]:
                    render_q(k, indent + 1)

        out.append("## Drzewo pytań")
        for q in qs:
            if not q["data"].get("parent"):
                render_q(q, 0)
        out.append("")

        def claim_block(c: dict[str, Any]) -> None:
            out.append(f"### {c['id']} [{c['status']}, pewność {c['confidence']:.2f}] {c['text']}")
            out.append(f"autor: {c['author']}" + (f", odpowiada na {c['data'].get('answers')}" if c["data"].get("answers") else ""))
            for e in es:
                if e["data"].get("claim") == c["id"]:
                    out.append(f"- dowód {e['id']} ({e['data'].get('kind')}): {e['text']} {e['data'].get('url', '')}".rstrip())
            for i in is_:
                if i["data"].get("conclusion") == c["id"]:
                    out.append(f"- wnioskowanie {i['id']} [{i['status']}]: {i['text']}")
            for o in os_:
                if o["data"].get("target") == c["id"]:
                    out.append(f"- zarzut {o['id']} [{o['status']}] ({o['author']}): {o['text']}")
                    for r in rs:
                        if r["data"].get("objection") == o["id"]:
                            out.append(f"  - odparcie {r['id']} ({r['author']}): {r['text']}")
                    if o["data"].get("ruling"):
                        out.append(f"  - rozstrzygnięcie ({o['data'].get('ruled_by')}): {o['data']['ruling']}: {o['data'].get('ruling_reason', '')}")
            out.append("")

        for title, st in (("## Tezy przyjęte", "accepted"), ("## Tezy sporne", "contested"), ("## Tezy obalone", "refuted"), ("## Tezy zgłoszone bez poparcia", "proposed")):
            group = [c for c in cs if c["status"] == st]
            if group:
                out.append(title)
                out.append("")
                for c in group:
                    claim_block(c)
        if ds:
            out.append("## Definicje")
            for d in ds:
                out.append(f"- **{d['data'].get('term')}** ({d['id']}, {d['author']}): {d['text']}")
            out.append("")
        return "\n".join(out)
