"""python -m agora.export → zrzut wszystkiego do czytelnych plików Markdown w community/export/."""

from __future__ import annotations

import json
from pathlib import Path

from .db import DB
from .graph import ArgumentGraph
from .settings import load_settings


def export(db: DB, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ch in db.list_channels():
        rows = db.messages(ch["id"])
        lines = [f"# #{ch['name']} ({ch['title']})", ""]
        for m in rows:
            who = "system" if m["kind"] == "system" else m["author"]
            reply = f" ↩#{m['reply_to']}" if m.get("reply_to") else ""
            lines.append(f"**{who}** #{m['id']}{reply} _{m['ts']}_\n\n{m['text']}\n")
        p = out_dir / f"channel-{ch['name']}.md"
        p.write_text("\n".join(lines), encoding="utf-8")
        written.append(p)
    dms = db.all_dms(limit=100000)
    lines = ["# Wiadomości prywatne", ""]
    for d in dms:
        reply = f" (odp. na #{d['reply_to']})" if d.get("reply_to") else ""
        lines.append(f"**{d['sender']} → {d['recipient']}** #{d['id']}{reply} _{d['ts']}_\n\n{d['text']}\n")
    p = out_dir / "dms.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    written.append(p)
    for st in db.all_states():
        lines = [f"# {st['agent']}", "", f"tura: {st.get('tick', 0)}", "", "## Streszczenie pamięci", "", st.get("summary") or "(brak)", "", "## Notatki", "", st.get("memory") or "(brak)", "", "## Dziennik", ""]
        lines.extend(st.get("journal") or [])
        lines += ["", "## Myślenie (ostatnie 30 zdarzeń)", ""]
        evs = db.events(after_id=0, limit=100000, agent=st["agent"], kinds=["thinking"])
        for e in evs[-30:]:
            lines.append(f"- _{e['ts']}_ {e['payload'].get('text', '')}")
        p = out_dir / f"agent-{st['agent']}.md"
        p.write_text("\n".join(lines), encoding="utf-8")
        written.append(p)
    graph = ArgumentGraph(db)
    p = out_dir / "CODEX.md"
    p.write_text(graph.codex(), encoding="utf-8")
    written.append(p)
    p = out_dir / "graph.json"
    p.write_text(json.dumps({"nodes": db.nodes(), "edges": db.all_edges()}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    written.append(p)
    return written


def main() -> int:
    settings = load_settings()
    db = DB(settings.db_path)
    files = export(db, settings.data_dir / "export")
    print("\n".join(str(f) for f in files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
