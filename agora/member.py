"""Członek społeczności: własna pętla życia, kontekst tury, pamięć, narzędzia."""

from __future__ import annotations

import asyncio
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from .db import DB, now_iso
from .graph import ArgumentGraph
from .kb_tools import TickScratch, make_kb_tools
from .llm import ChatBackend
from .observe import Observer
from .roles import system_prompt
from .settings import RoleSpec, Settings
from .social_tools import make_social_tools, render_message
from .tools import Toolbox
from .web_tools import make_web_tools

MAX_UNREAD_PER_CHANNEL = 12
FIRST_TICK_BACKLOG = 8
MEMORY_MAX_LINES = 24
MEMORY_KEEP_LINES = 8
JOURNAL_MAX = 12


class Member:
    def __init__(
        self,
        role: RoleSpec,
        *,
        settings: Settings,
        db: DB,
        graph: ArgumentGraph,
        chat: ChatBackend,
        observer: Observer,
        members: list[str],
        cache_dir: Path,
    ):
        self.role = role
        self.name = role.name
        self.settings = settings
        self.db = db
        self.graph = graph
        self.chat = chat
        self.observer = observer
        self.members = members
        self.scratch = TickScratch()
        self.system = system_prompt(role, all_names=members, topic=settings.topic)
        self._stop = asyncio.Event()
        self.ticks_done = 0

        social, self._reset_social = make_social_tools(
            db, self.name, self.scratch,
            members=members, post_policy=settings.post_policy, emit=self.emit, remember_fn=self.remember,
        )
        specs = make_kb_tools(graph, self.name, self.scratch)
        if settings.web_enabled:
            specs += make_web_tools(db, self.name, self.scratch, cache_dir=cache_dir, max_results=settings.web_max_results, fetch_max_chars=settings.fetch_max_chars)
        specs += social
        self.toolbox = Toolbox(specs)

    # ── obserwacja i pamięć ──────────────────────────────────────────────

    def emit(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        self.observer.event(self.name, kind, payload or {}, channel=(payload or {}).get("channel"))

    def _memory_lines(self) -> list[str]:
        state = self.db.get_state(self.name)
        return [ln for ln in (state.get("memory") or "").splitlines() if ln.strip()]

    def remember(self, text: str) -> str:
        if len(text) < 8:
            return "BŁĄD: notatka jest za krótka."
        lines = self._memory_lines()
        stamp = datetime.now().strftime("%m-%d %H:%M")
        lines.append(f"- [{stamp}] {text[:400]}")
        self.db.set_state(self.name, memory="\n".join(lines))
        self.emit("memory", {"what": "notatka", "text": text})
        return f"zapisano notatkę ({len(lines)} notatek w pamięci)"

    async def _maybe_compact_memory(self) -> None:
        lines = self._memory_lines()
        if len(lines) <= MEMORY_MAX_LINES:
            return
        state = self.db.get_state(self.name)
        old_summary = state.get("summary") or ""
        to_fold, keep = lines[:-MEMORY_KEEP_LINES], lines[-MEMORY_KEEP_LINES:]
        prompt = (
            "Masz dotychczasowe streszczenie pamięci agenta i nowe notatki. Napisz jedno zwięzłe streszczenie (do 150 słów), "
            "zachowując konkrety: ID węzłów, ustalenia, otwarte sprawy, plany. Nic nie dodawaj od siebie.\n\n"
            f"STRESZCZENIE DOTYCHCZASOWE:\n{old_summary or '(brak)'}\n\nNOTATKI DO WŁĄCZENIA:\n" + "\n".join(to_fold)
        )
        try:
            summary = await self.chat.complete(system="Streszczasz notatki po polsku, wiernie i zwięźle.", user=prompt, temperature=0.1)
        except Exception as e:  # noqa: BLE001
            self.emit("error", {"error": f"kompakcja pamięci nie powiodła się: {e}"})
            return
        self.db.set_state(self.name, summary=summary, memory="\n".join(keep))
        self.emit("memory", {"what": f"kompakcja ({len(to_fold)} notatek → streszczenie)", "text": summary})

    # ── kontekst tury ────────────────────────────────────────────────────

    def _collect_unread(self, cursors: dict[str, int], first_tick: bool) -> tuple[list[str], dict[str, int], int]:
        blocks: list[str] = []
        new_cursors: dict[str, int] = dict(cursors)
        total = 0
        for ch in self.db.list_channels():
            name = ch["name"]
            cursor = int(cursors.get(name, 0))
            if cursor == 0 and first_tick:
                rows = self.db.last_messages(ch["id"], FIRST_TICK_BACKLOG)
                skipped = max(0, int(ch["n_messages"] or 0) - len(rows))
            else:
                rows = self.db.messages(ch["id"], after_id=cursor)
                skipped = 0
            last_id = int(ch["last_message_id"] or 0)
            new_cursors[name] = max(cursor, last_id)
            rows = [m for m in rows if m["author"] != self.name]
            if not rows:
                continue
            if len(rows) > MAX_UNREAD_PER_CHANNEL:
                skipped += len(rows) - MAX_UNREAD_PER_CHANNEL
                rows = rows[-MAX_UNREAD_PER_CHANNEL:]
            total += len(rows)
            head = f"#{name} ({ch['title']})" + (f" [pominięto {skipped} starszych; read_channel pokaże więcej]" if skipped else "")
            blocks.append(head + "\n" + "\n".join(render_message(m) for m in rows))
        return blocks, new_cursors, total

    def _build_prompt(self, tick: int, unread_blocks: list[str], pending_dms: list[dict[str, Any]], replies: list[dict[str, Any]], state: dict[str, Any]) -> str:
        parts = [f"TURA {tick}. Czas: {datetime.now().strftime('%Y-%m-%d %H:%M')}."]
        parts.append("[NOWE WIADOMOŚCI W KANAŁACH]")
        parts.append("\n\n".join(unread_blocks) if unread_blocks else "(brak nowych wiadomości)")
        if pending_dms:
            parts.append("[DM-y DO CIEBIE BEZ ODPOWIEDZI] (odpowiedz przez dm_reply albo świadomie zostaw)")
            parts.extend(f"DM #{d['id']} od {d['sender']}: {d['text']}" for d in pending_dms)
        if replies:
            parts.append("[ODPOWIEDZI NA TWOJE DM-y]")
            parts.extend(f"DM #{d['id']} od {d['sender']}: {d['text']}" for d in replies)
        parts.append("[SKRÓT GRAFU WIEDZY]")
        parts.append(self.graph.digest())
        summary = state.get("summary") or ""
        memory = state.get("memory") or ""
        if summary or memory:
            parts.append("[TWOJA PAMIĘĆ]")
            if summary:
                parts.append("streszczenie: " + summary)
            if memory:
                parts.append(memory)
        journal = state.get("journal") or []
        if journal:
            parts.append("[TWÓJ DZIENNIK, ostatnie tury]")
            parts.extend(journal[-6:])
        parts.append(
            "[CO TERAZ]\nZdecyduj sam, co robisz, zgodnie ze swoją rolą. Najpierw sprawdź graf (kb_read/kb_search), jeśli chcesz coś dodać. "
            "Każda wypowiedź w kanale ma odnosić się do ID węzłów. Jeśli nie masz nic nowego, wywołaj idle. "
            "Na końcu napisz jedno zdanie: co zrobiłeś i dlaczego."
        )
        return "\n\n".join(parts)

    # ── pętla życia ──────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop.set()

    async def live(self, *, start_delay: float = 0.0) -> None:
        if start_delay > 0:
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=start_delay)
                return
            except asyncio.TimeoutError:
                pass
        state = self.db.get_state(self.name)
        tick = int(state.get("tick") or 0)
        while not self._stop.is_set():
            tick += 1
            try:
                await self.tick(tick)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self.emit("error", {"error": f"tura {tick}: {type(e).__name__}: {e}"})
            self.ticks_done += 1
            if self.settings.max_ticks is not None and self.ticks_done >= self.settings.max_ticks:
                break
            delay = max(1.0, self.settings.tick_seconds + random.uniform(-self.settings.tick_jitter, self.settings.tick_jitter))
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
                break
            except asyncio.TimeoutError:
                continue

    async def tick(self, tick: int) -> None:
        self.scratch.reset()
        self._reset_social()
        state = self.db.get_state(self.name)
        cursors: dict[str, int] = dict(state.get("cursors") or {})
        first_tick = int(state.get("tick") or 0) == 0
        unread_blocks, new_cursors, n_unread = self._collect_unread(cursors, first_tick)
        pending = self.db.pending_dms(self.name)
        dm_cursor = int(cursors.get("__dm__", 0))
        replies = [d for d in self.db.dms_for(self.name, after_id=dm_cursor) if d["recipient"] == self.name and d.get("reply_to")]
        all_dms = self.db.dms_for(self.name, after_id=0)
        new_cursors["__dm__"] = max([dm_cursor] + [int(d["id"]) for d in all_dms])

        self.emit("tick_start", {"tick": tick, "unread": n_unread, "dms": len(pending) + len(replies)})
        prompt = self._build_prompt(tick, unread_blocks, pending, replies, state)

        async def on_event(kind: str, payload: dict[str, Any]) -> None:
            self.emit(kind, payload)

        result = await self.chat.run(
            system=self.system,
            user=prompt,
            toolbox=self.toolbox,
            temperature=self.role.temperature,
            on_event=on_event,
            max_rounds=self.settings.max_tool_rounds,
            agent=self.name,
        )
        summary = (result.text or "").strip()
        if result.error and not summary:
            summary = f"(błąd modelu: {result.error})"
        journal = list(state.get("journal") or [])
        journal.append(f"- tura {tick}: {summary[:300] if summary else '(bez podsumowania)'}")
        journal = journal[-JOURNAL_MAX:]
        self.db.set_state(self.name, cursors=new_cursors, tick=tick, last_seen=now_iso(), journal=journal)
        self.emit("tick_end", {
            "tick": tick, "summary": summary, "seconds": result.seconds, "rounds": result.rounds,
            "tool_calls": result.tool_calls, "posts": self.scratch.posts, "graph_ops": list(self.scratch.graph_ops), "web_calls": self.scratch.web_calls,
            "transcript": result.transcript,
        })
        await self._maybe_compact_memory()
