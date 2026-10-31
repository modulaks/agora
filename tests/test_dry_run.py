"""Cała społeczność w trybie dry-run: 2 tury, bez Ollamy i bez sieci."""

import asyncio

import pytest

from agora.community import Community
from agora.fake import FakeChat
from agora.settings import load_settings


@pytest.mark.asyncio
async def test_dry_run_two_ticks(tmp_path, monkeypatch):
    monkeypatch.setenv("AGORA_DATA_DIR", str(tmp_path / "community"))
    settings = load_settings(dry_run=True, max_ticks=2)
    settings.tick_seconds = 0.2
    settings.tick_jitter = 0.05
    settings.web_enabled = False
    community = Community(settings, chat=FakeChat(), quiet=True)
    await asyncio.wait_for(community.run(), timeout=60)
    stats = community.db.stats()
    assert stats["nodes"] >= 12 and stats["messages"] >= 15 and stats["dms"] >= 2
    statuses = {n["id"]: n["status"] for n in community.db.nodes("C")}
    assert statuses["C1"] == "accepted"
    assert (settings.data_dir / "CODEX.md").exists()
    errors = [e for e in community.db.events(after_id=0, limit=5000, kinds=["error", "llm_error"])]
    assert errors == []
    # Restart wznawia stan: tury liczą się dalej, cursory nie zalewają agentów starymi wiadomościami.
    community2 = Community(settings, chat=FakeChat(), quiet=True)
    assert community2.db.get_state("sedzia")["tick"] == 2
