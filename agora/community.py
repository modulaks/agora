"""Społeczność: baza, kanały startowe, graf, członkowie i ich pętle życia."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .db import DB
from .graph import ArgumentGraph
from .llm import ChatBackend, OllamaChat
from .member import Member
from .observe import Observer
from .settings import Settings

CODEX_INTERVAL = 60.0


class Community:
    def __init__(self, settings: Settings, *, chat: ChatBackend | None = None, quiet: bool = False):
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.db = DB(settings.db_path)
        names = [r.name for r in settings.roles]
        self.observer = Observer(
            self.db, agents=names, show_thinking=settings.show_thinking, thinking_chars=settings.thinking_chars, quiet=quiet,
            log_path=settings.data_dir / "agora-log.txt",
        )
        self.graph = ArgumentGraph(self.db, on_change=self._on_graph_change)
        self.chat: ChatBackend = chat or self._make_chat(settings)
        cache_dir = settings.data_dir / "cache"
        self.members = [
            Member(role, settings=settings, db=self.db, graph=self.graph, chat=self.chat, observer=self.observer, members=names, cache_dir=cache_dir)
            for role in settings.roles
        ]
        self._tasks: list[asyncio.Task[Any]] = []

    @staticmethod
    def _make_chat(settings: Settings) -> ChatBackend:
        if settings.backend == "anthropic":
            from .claude_chat import AnthropicChat

            return AnthropicChat(settings.anthropic_model, effort=settings.anthropic_effort, timeout=settings.llm_timeout)
        if settings.backend == "openai":
            from .openai_chat import OpenAIChat

            return OpenAIChat(settings.openai_model, timeout=settings.llm_timeout)
        return OllamaChat(
            settings.ollama_host, settings.ollama_model,
            num_ctx=settings.num_ctx, think=settings.think, concurrency=settings.concurrency, timeout=settings.llm_timeout,
        )

    # ── hook grafu: dziennik #graf + konsola ─────────────────────────────

    def _on_graph_change(self, kind: str, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload["op"] = kind
        author = str(payload.get("author") or "system")
        self.observer.event(author, "graph", payload, channel="graf")
        ch = self.db.get_channel("graf")
        if ch is None:
            return
        node_id = payload.get("id", "?")
        text = str(payload.get("text", ""))[:300]
        if kind == "status":
            line = f"{node_id}: {payload.get('old')} → {payload.get('new')}" + (f" (pewność {payload.get('confidence')})" if payload.get("confidence") is not None else "")
        elif kind == "ruling":
            line = f"{author} rozstrzygnął {node_id}: {payload.get('decision')} ({text})"
        elif kind == "question_status":
            line = f"{author} oznaczył {node_id} jako {payload.get('status')} ({text})"
        else:
            rel = ""
            for key in ("parent", "answers", "claim", "target", "objection", "conclusion", "used_by"):
                if payload.get(key):
                    rel = f" → {payload[key]}"
                    break
            label = {"question": "pytanie", "claim": "teza", "evidence": "dowód", "inference": "wnioskowanie", "objection": "zarzut", "rebuttal": "odparcie", "definition": "definicja", "assumption": "założenie"}.get(kind, kind)
            line = f"{author} dodał {label} {node_id}{rel}: {text}"
        self.db.add_message(ch["id"], "system", line, kind="system")

    # ── bootstrap ────────────────────────────────────────────────────────

    def bootstrap(self) -> None:
        s = self.settings
        for spec in s.channels:
            if self.db.get_channel(spec.name) is None:
                self.db.create_channel(spec.name, spec.title, "system")
        general = self.db.get_channel("ogolny")
        assert general is not None
        if not self.db.nodes("Q") and s.seed_questions:
            root = self.graph.add_question(s.seed_questions[0], "system")
            for q in s.seed_questions[1:]:
                try:
                    self.graph.add_question(q, "system", parent_id=root)
                except Exception as e:  # noqa: BLE001
                    self.observer.event("system", "error", {"error": f"pytanie startowe pominięte: {e}"})
            intro = (
                f"Temat społeczności: {s.topic}\nPytanie główne: {root}. Zacznijcie od dekompozycji i definicji. "
                f"Członkowie: {', '.join(r.name for r in s.roles)}. Człowiek obserwuje i nie uczestniczy."
            )
            self.db.add_message(general["id"], "system", intro, kind="system")
            self.observer.event("system", "system", {"text": intro}, channel="ogolny")
        self.write_codex()

    # ── CODEX ────────────────────────────────────────────────────────────

    def write_codex(self) -> Path:
        path = self.settings.data_dir / "CODEX.md"
        path.write_text(self.graph.codex(), encoding="utf-8")
        return path

    async def _codex_loop(self) -> None:
        while True:
            await asyncio.sleep(CODEX_INTERVAL)
            try:
                self.write_codex()
            except Exception as e:  # noqa: BLE001
                self.observer.event("system", "error", {"error": f"CODEX: {e}"})

    # ── uruchomienie ─────────────────────────────────────────────────────

    async def check_model(self) -> None:
        if self.settings.dry_run:
            return
        if self.settings.backend in ("anthropic", "openai"):
            self.observer.event("system", "system", {"text": f"Sprawdzam klucz i model: {self.settings.model_label} ..."})
            check = getattr(self.chat, "check", None)
            if check is not None:
                await check()
            return
        import ollama

        host = self.settings.ollama_host
        self.observer.event("system", "system", {"text": f"Łączę się z Ollamą pod {host} ..."})
        client = ollama.AsyncClient(host=host, timeout=20.0)
        try:
            listing = await client.list()
        except Exception as e:  # noqa: BLE001
            reason = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            raise SystemExit(
                f"Nie mogę połączyć się z Ollamą pod {host} ({reason}).\n"
                f"Sprawdź, czy Ollama działa i nasłuchuje w sieci (OLLAMA_HOST=0.0.0.0 po stronie serwera), "
                f"czy adres w .env jest poprawny i czy firewall przepuszcza port. Test: curl {host}/api/tags"
            )
        names = [m.model for m in (listing.models or [])]
        if self.settings.ollama_model not in names and f"{self.settings.ollama_model}:latest" not in names:
            self.observer.event("system", "error", {"error": f"model {self.settings.ollama_model!r} nie jest na liście Ollamy ({', '.join(names) or 'pusto'}). Pobierz go na serwerze: ollama pull {self.settings.ollama_model}"})

    async def run(self) -> None:
        s = self.settings
        self.bootstrap()
        await self.check_model()
        self.observer.banner(
            f"AGORA · {len(self.members)} członków · model {s.model_label} · tura co ~{s.tick_seconds:.0f}s · dane: {s.data_dir}"
            + (" · DRY RUN" if s.dry_run else "")
        )
        self.observer.event("system", "system", {"text": "Start. Ctrl+C zatrzymuje; ponowne uruchomienie wznawia z bazy."})
        stagger = s.tick_seconds / max(1, len(self.members))
        for i, m in enumerate(self.members):
            self._tasks.append(asyncio.create_task(m.live(start_delay=i * stagger * 0.5), name=f"member-{m.name}"))
        codex_task = asyncio.create_task(self._codex_loop(), name="codex")
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass
        finally:
            codex_task.cancel()
            for m in self.members:
                m.stop()
            for t in self._tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self.write_codex()
            self.observer.event("system", "system", {"text": f"Stop. Stan: {self.db.stats()}. CODEX: {self.settings.data_dir / 'CODEX.md'}"})

    def stop(self) -> None:
        for m in self.members:
            m.stop()
