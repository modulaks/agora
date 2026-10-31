"""Obserwacja: wszystko, co się dzieje, trafia do konsoli (rich) i do tabeli events w SQLite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape

from .db import DB

PALETTE = ["cyan", "green", "yellow", "magenta", "blue", "bright_red", "bright_green", "bright_yellow", "bright_magenta", "bright_cyan", "white"]

GRAPH_LABELS = {
    "question": "pytanie",
    "claim": "teza",
    "evidence": "dowód",
    "inference": "wnioskowanie",
    "objection": "zarzut",
    "rebuttal": "odparcie",
    "definition": "definicja",
    "assumption": "założenie",
    "ruling": "rozstrzygnięcie",
    "question_status": "status pytania",
    "status": "zmiana statusu",
}


def _short(text: Any, n: int) -> str:
    s = str(text)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


class Observer:
    def __init__(self, db: DB, *, agents: list[str], show_thinking: bool = True, thinking_chars: int = 600, quiet: bool = False, log_path: Path | None = None):
        self.db = db
        self.console = Console(highlight=False, soft_wrap=True)
        # Kopia konsoli do pliku (bez kolorów, UTF-8), niezależnie od --quiet.
        self.log_console: Console | None = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_console = Console(file=log_path.open("a", encoding="utf-8"), highlight=False, soft_wrap=True, no_color=True, force_terminal=False, width=200)
        self.show_thinking = show_thinking
        self.thinking_chars = thinking_chars
        self.quiet = quiet
        self.colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(agents)}
        self.colors["system"] = "bright_black"

    def _c(self, agent: str) -> str:
        return self.colors.get(agent, "white")

    def _print(self, text: str) -> None:
        if not self.quiet:
            self.console.print(text)
        if self.log_console is not None:
            self.log_console.print(text)
            self.log_console.file.flush()

    def event(self, agent: str, kind: str, payload: dict[str, Any] | None = None, *, channel: str | None = None) -> int:
        payload = payload or {}
        event_id = self.db.add_event(agent, kind, payload, channel=channel)
        self.render(agent, kind, payload, channel)
        return event_id

    def render(self, agent: str, kind: str, p: dict[str, Any], channel: str | None) -> None:
        c = self._c(agent)
        a = f"[bold {c}]{escape(agent)}[/]"
        if kind == "post":
            reply = f" ↩#{p['reply_to']}" if p.get("reply_to") else ""
            self._print(f"[bold white on {self._bg(c)}] #{escape(p.get('channel', channel or '?'))} [/] {a} [dim]#{p.get('message_id')}{reply}[/]\n  {escape(p.get('text', ''))}")
        elif kind == "thread":
            self._print(f"[bold]{a} założył wątek [bold white]#{escape(p.get('channel', ''))}[/]: {escape(p.get('title', ''))}[/]" + (f" [dim](pytanie {p.get('question_id')})[/]" if p.get("question_id") else ""))
        elif kind == "dm":
            arrow = "odpowiada" if p.get("reply_to") else "pisze do"
            self._print(f"[bold]✉ DM[/] {a} {arrow} [bold {self._c(p.get('to', ''))}]{escape(p.get('to', ''))}[/] [dim]#{p.get('dm_id')}[/]\n  {escape(p.get('text', ''))}")
        elif kind == "thinking":
            if self.show_thinking:
                self._print(f"[dim italic]💭 {escape(agent)} myśli: {escape(_short(p.get('text', ''), self.thinking_chars))}[/]")
        elif kind == "tool_call":
            args = p.get("args", {})
            try:
                args_s = json.dumps(args, ensure_ascii=False)
            except TypeError:
                args_s = str(args)
            self._print(f"  [{c}]⚙ {escape(agent)}[/] → [bold]{escape(p.get('tool', ''))}[/]({escape(_short(args_s, 260))})")
        elif kind == "tool_result":
            res = str(p.get("result", ""))
            style = "red" if res.startswith("BŁĄD") else "dim"
            self._print(f"    [{style}]↳ {escape(_short(res, 300))}[/]")
        elif kind == "graph":
            label = GRAPH_LABELS.get(p.get("op", ""), p.get("op", ""))
            extra = ""
            if p.get("op") == "status":
                extra = f" {p.get('old')} → {p.get('new')}"
                self._print(f"[bold magenta]◆ graf[/] {label} [bold]{p.get('id')}[/]{extra} [dim]{escape(_short(p.get('text', ''), 120))}[/]")
            else:
                rel = ""
                for key in ("parent", "answers", "claim", "target", "objection", "conclusion", "used_by"):
                    if p.get(key):
                        rel = f" → {p[key]}"
                        break
                self._print(f"[bold magenta]◆ graf[/] {label} [bold]{p.get('id')}[/]{rel} [dim]({escape(str(p.get('author', '')))})[/] {escape(_short(p.get('text', ''), 220))}")
        elif kind == "idle":
            self._print(f"  [dim]{escape(agent)}: nic nie robi. {escape(p.get('reason', ''))}[/]")
        elif kind == "tick_start":
            self._print(f"[dim]── {escape(agent)} tura {p.get('tick')} (nowych wiad.: {p.get('unread', 0)}, DM: {p.get('dms', 0)})[/]")
        elif kind == "tick_end":
            self._print(f"  [{c}]■ {escape(agent)}[/] [dim]({p.get('seconds')}s, rund {p.get('rounds')}, narzędzi {p.get('tool_calls')})[/] {escape(_short(p.get('summary', ''), 300))}")
        elif kind == "waiting":
            self._print(f"  [dim]⏳ {escape(agent)} czeka na model ({p.get('seconds')} s, runda {p.get('round')})[/]")
        elif kind == "memory":
            self._print(f"  [dim]🧠 {escape(agent)} {escape(p.get('what', ''))}: {escape(_short(p.get('text', ''), 200))}[/]")
        elif kind in ("llm_error", "error"):
            self._print(f"[bold red]✖ {escape(agent)}: {escape(_short(p.get('error', ''), 400))}[/]")
        elif kind == "system":
            self._print(f"[bright_black]{escape(p.get('text', ''))}[/]")
        else:
            self._print(f"[dim]{escape(agent)} {kind} {escape(_short(json.dumps(p, ensure_ascii=False, default=str), 200))}[/]")

    @staticmethod
    def _bg(color: str) -> str:
        return {"cyan": "dark_cyan", "green": "dark_green", "yellow": "dark_orange3", "magenta": "dark_magenta", "blue": "dark_blue"}.get(color, "grey37")

    def banner(self, text: str) -> None:
        self._print(f"[bold white on grey23] {escape(text)} [/]")
