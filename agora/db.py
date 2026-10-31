"""SQLite jako serwer społeczności: kanały, wiadomości, DM-y, graf, zdarzenia, stan agentów.

Jeden plik, tryb WAL, prosty wrapper synchroniczny (operacje są milisekundowe, nie ma sensu
komplikować tego asynchronicznie). Podgląd w przeglądarce czyta ten sam plik.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS channels (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT 'channel',   -- channel | thread
    parent_id INTEGER,
    question_id TEXT,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    channel_id INTEGER NOT NULL,
    author TEXT NOT NULL,
    reply_to INTEGER,
    kind TEXT NOT NULL DEFAULT 'post',       -- post | system
    text TEXT NOT NULL,
    ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_messages_channel ON messages(channel_id, id);
CREATE TABLE IF NOT EXISTS dms (
    id INTEGER PRIMARY KEY,
    sender TEXT NOT NULL,
    recipient TEXT NOT NULL,
    reply_to INTEGER,
    text TEXT NOT NULL,
    answered INTEGER NOT NULL DEFAULT 0,
    ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,                      -- Q C E I O R D A
    text TEXT NOT NULL,
    author TEXT NOT NULL,
    data TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS edges (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    rel TEXT NOT NULL,
    PRIMARY KEY (src, dst, rel)
);
CREATE TABLE IF NOT EXISTS counters (
    prefix TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    agent TEXT NOT NULL,
    kind TEXT NOT NULL,
    channel TEXT,
    payload TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_events_agent ON events(agent, id);
CREATE TABLE IF NOT EXISTS agent_state (
    agent TEXT PRIMARY KEY,
    memory TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    journal TEXT NOT NULL DEFAULT '[]',
    cursors TEXT NOT NULL DEFAULT '{}',
    tick INTEGER NOT NULL DEFAULT 0,
    last_seen TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS sources (
    url TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    found_by TEXT NOT NULL DEFAULT '',
    chars INTEGER NOT NULL DEFAULT 0,
    fetched_at TEXT NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DB:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)

    # ── low level ────────────────────────────────────────────────────────

    def _exec(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, tuple(params))

    def _rows(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self._exec(sql, params).fetchall()]

    def _row(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        r = self._exec(sql, params).fetchone()
        return dict(r) if r else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── channels ─────────────────────────────────────────────────────────

    def create_channel(
        self,
        name: str,
        title: str,
        created_by: str,
        *,
        kind: str = "channel",
        parent_id: int | None = None,
        question_id: str | None = None,
    ) -> int:
        cur = self._exec(
            "INSERT INTO channels(name, title, kind, parent_id, question_id, created_by, created_at) VALUES (?,?,?,?,?,?,?)",
            (name, title, kind, parent_id, question_id, created_by, now_iso()),
        )
        return int(cur.lastrowid)

    def get_channel(self, name: str) -> dict[str, Any] | None:
        return self._row("SELECT * FROM channels WHERE name=?", (name,))

    def get_channel_by_id(self, channel_id: int) -> dict[str, Any] | None:
        return self._row("SELECT * FROM channels WHERE id=?", (channel_id,))

    def list_channels(self) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.channel_id=c.id) AS n_messages,"
            " (SELECT MAX(id) FROM messages m WHERE m.channel_id=c.id) AS last_message_id"
            " FROM channels c ORDER BY c.id"
        )

    # ── messages ─────────────────────────────────────────────────────────

    def add_message(self, channel_id: int, author: str, text: str, *, reply_to: int | None = None, kind: str = "post") -> int:
        cur = self._exec(
            "INSERT INTO messages(channel_id, author, reply_to, kind, text, ts) VALUES (?,?,?,?,?,?)",
            (channel_id, author, reply_to, kind, text, now_iso()),
        )
        return int(cur.lastrowid)

    def messages(self, channel_id: int, *, after_id: int = 0, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM messages WHERE channel_id=? AND id>? ORDER BY id"
        params: list[Any] = [channel_id, after_id]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return self._rows(sql, params)

    def last_messages(self, channel_id: int, n: int) -> list[dict[str, Any]]:
        rows = self._rows("SELECT * FROM messages WHERE channel_id=? ORDER BY id DESC LIMIT ?", (channel_id, n))
        return list(reversed(rows))

    def count_messages_after(self, channel_id: int, after_id: int) -> int:
        row = self._row("SELECT COUNT(*) AS n FROM messages WHERE channel_id=? AND id>?", (channel_id, after_id))
        return int(row["n"]) if row else 0

    def get_message(self, message_id: int) -> dict[str, Any] | None:
        return self._row("SELECT * FROM messages WHERE id=?", (message_id,))

    def all_messages_after(self, after_id: int, limit: int = 500) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT m.*, c.name AS channel FROM messages m JOIN channels c ON c.id=m.channel_id WHERE m.id>? ORDER BY m.id LIMIT ?",
            (after_id, limit),
        )

    def messages_by_author_after(self, author: str, after_id: int) -> int:
        row = self._row("SELECT COUNT(*) AS n FROM messages WHERE author=? AND id>?", (author, after_id))
        return int(row["n"]) if row else 0

    # ── dms ──────────────────────────────────────────────────────────────

    def add_dm(self, sender: str, recipient: str, text: str, *, reply_to: int | None = None) -> int:
        # Odpowiedź (reply_to) sama nie czeka na odpowiedź, więc od razu answered=1.
        cur = self._exec(
            "INSERT INTO dms(sender, recipient, reply_to, text, answered, ts) VALUES (?,?,?,?,?,?)",
            (sender, recipient, reply_to, text, 1 if reply_to else 0, now_iso()),
        )
        if reply_to:
            self._exec("UPDATE dms SET answered=1 WHERE id=?", (reply_to,))
        return int(cur.lastrowid)

    def pending_dms(self, recipient: str) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM dms WHERE recipient=? AND answered=0 ORDER BY id", (recipient,))

    def get_dm(self, dm_id: int) -> dict[str, Any] | None:
        return self._row("SELECT * FROM dms WHERE id=?", (dm_id,))

    def dms_for(self, agent: str, after_id: int = 0) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT * FROM dms WHERE (sender=? OR recipient=?) AND id>? ORDER BY id",
            (agent, agent, after_id),
        )

    def all_dms(self, after_id: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM dms WHERE id>? ORDER BY id LIMIT ?", (after_id, limit))

    # ── graph ────────────────────────────────────────────────────────────

    def next_id(self, prefix: str) -> str:
        with self._lock:
            row = self._row("SELECT value FROM counters WHERE prefix=?", (prefix,))
            value = (int(row["value"]) if row else 0) + 1
            self._exec(
                "INSERT INTO counters(prefix, value) VALUES (?,?) ON CONFLICT(prefix) DO UPDATE SET value=excluded.value",
                (prefix, value),
            )
            return f"{prefix}{value}"

    def add_node(
        self,
        node_id: str,
        node_type: str,
        text: str,
        author: str,
        *,
        data: dict[str, Any] | None = None,
        status: str = "",
        confidence: float = 0.0,
    ) -> None:
        ts = now_iso()
        self._exec(
            "INSERT INTO nodes(id, type, text, author, data, status, confidence, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (node_id, node_type, text, author, json.dumps(data or {}, ensure_ascii=False), status, confidence, ts, ts),
        )

    def update_node(self, node_id: str, **fields: Any) -> None:
        if not fields:
            return
        if "data" in fields and isinstance(fields["data"], dict):
            fields["data"] = json.dumps(fields["data"], ensure_ascii=False)
        fields["updated_at"] = now_iso()
        cols = ", ".join(f"{k}=?" for k in fields)
        self._exec(f"UPDATE nodes SET {cols} WHERE id=?", (*fields.values(), node_id))

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        row = self._row("SELECT * FROM nodes WHERE id=?", (node_id,))
        return self._decode_node(row) if row else None

    def nodes(self, node_type: str | None = None) -> list[dict[str, Any]]:
        if node_type:
            rows = self._rows("SELECT * FROM nodes WHERE type=? ORDER BY created_at, id", (node_type,))
        else:
            rows = self._rows("SELECT * FROM nodes ORDER BY created_at, id")
        return [self._decode_node(r) for r in rows]

    def recent_nodes(self, n: int) -> list[dict[str, Any]]:
        rows = self._rows("SELECT * FROM nodes ORDER BY created_at DESC, id DESC LIMIT ?", (n,))
        return [self._decode_node(r) for r in reversed(rows)]

    @staticmethod
    def _decode_node(row: dict[str, Any]) -> dict[str, Any]:
        row = dict(row)
        try:
            row["data"] = json.loads(row.get("data") or "{}")
        except json.JSONDecodeError:
            row["data"] = {}
        return row

    def add_edge(self, src: str, dst: str, rel: str) -> None:
        self._exec("INSERT OR IGNORE INTO edges(src, dst, rel) VALUES (?,?,?)", (src, dst, rel))

    def edges(self, *, src: str | None = None, dst: str | None = None, rel: str | None = None) -> list[dict[str, Any]]:
        clauses, params = [], []
        if src:
            clauses.append("src=?"); params.append(src)
        if dst:
            clauses.append("dst=?"); params.append(dst)
        if rel:
            clauses.append("rel=?"); params.append(rel)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return self._rows(f"SELECT * FROM edges{where}", params)

    def all_edges(self) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM edges")

    def search_nodes(self, query: str, limit: int = 12) -> list[dict[str, Any]]:
        words = [w.lower() for w in query.split() if len(w) > 2][:6]
        if not words:
            return []
        clauses = " AND ".join("LOWER(text) LIKE ?" for _ in words)
        rows = self._rows(f"SELECT * FROM nodes WHERE {clauses} ORDER BY created_at DESC LIMIT ?", [f"%{w}%" for w in words] + [limit])
        if not rows:
            clauses = " OR ".join("LOWER(text) LIKE ?" for _ in words)
            rows = self._rows(f"SELECT * FROM nodes WHERE {clauses} ORDER BY created_at DESC LIMIT ?", [f"%{w}%" for w in words] + [limit])
        return [self._decode_node(r) for r in rows]

    # ── events ───────────────────────────────────────────────────────────

    def add_event(self, agent: str, kind: str, payload: dict[str, Any] | None = None, *, channel: str | None = None) -> int:
        cur = self._exec(
            "INSERT INTO events(ts, agent, kind, channel, payload) VALUES (?,?,?,?,?)",
            (now_iso(), agent, kind, channel, json.dumps(payload or {}, ensure_ascii=False, default=str)),
        )
        return int(cur.lastrowid)

    def events(self, *, after_id: int = 0, limit: int = 200, agent: str | None = None, kinds: Iterable[str] | None = None) -> list[dict[str, Any]]:
        clauses, params = ["id>?"], [after_id]
        if agent:
            clauses.append("agent=?"); params.append(agent)
        kinds = list(kinds or [])
        if kinds:
            clauses.append("kind IN (%s)" % ",".join("?" * len(kinds))); params.extend(kinds)
        rows = self._rows(f"SELECT * FROM events WHERE {' AND '.join(clauses)} ORDER BY id LIMIT ?", [*params, limit])
        for r in rows:
            try:
                r["payload"] = json.loads(r.get("payload") or "{}")
            except json.JSONDecodeError:
                r["payload"] = {}
        return rows

    def last_event_id(self) -> int:
        row = self._row("SELECT MAX(id) AS m FROM events")
        return int(row["m"] or 0) if row else 0

    # ── agent state ──────────────────────────────────────────────────────

    def get_state(self, agent: str) -> dict[str, Any]:
        row = self._row("SELECT * FROM agent_state WHERE agent=?", (agent,))
        if not row:
            self._exec("INSERT INTO agent_state(agent) VALUES (?)", (agent,))
            row = self._row("SELECT * FROM agent_state WHERE agent=?", (agent,)) or {}
        row = dict(row)
        try:
            row["cursors"] = json.loads(row.get("cursors") or "{}")
        except json.JSONDecodeError:
            row["cursors"] = {}
        try:
            row["journal"] = json.loads(row.get("journal") or "[]")
        except json.JSONDecodeError:
            row["journal"] = []
        return row

    def set_state(self, agent: str, **fields: Any) -> None:
        if not fields:
            return
        self.get_state(agent)
        for key in ("cursors", "journal"):
            if key in fields and not isinstance(fields[key], str):
                fields[key] = json.dumps(fields[key], ensure_ascii=False)
        cols = ", ".join(f"{k}=?" for k in fields)
        self._exec(f"UPDATE agent_state SET {cols} WHERE agent=?", (*fields.values(), agent))

    def all_states(self) -> list[dict[str, Any]]:
        return [self.get_state(r["agent"]) for r in self._rows("SELECT agent FROM agent_state ORDER BY agent")]

    # ── sources ──────────────────────────────────────────────────────────

    def upsert_source(self, url: str, title: str, found_by: str, chars: int) -> None:
        self._exec(
            "INSERT INTO sources(url, title, found_by, chars, fetched_at) VALUES (?,?,?,?,?)"
            " ON CONFLICT(url) DO UPDATE SET title=excluded.title, chars=excluded.chars, fetched_at=excluded.fetched_at",
            (url, title, found_by, chars, now_iso()),
        )

    def sources(self) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM sources ORDER BY fetched_at DESC")

    # ── misc ─────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for table in ("channels", "messages", "dms", "nodes", "edges", "events", "sources"):
            row = self._row(f"SELECT COUNT(*) AS n FROM {table}")
            out[table] = int(row["n"]) if row else 0
        return out


def elapsed_s(start: float) -> float:
    return round(time.monotonic() - start, 2)
