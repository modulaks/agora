"""Ustawienia: .env (infrastruktura) + config/community.yaml (temat, role)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path) -> None:
    """Minimalny loader .env: nie nadpisuje zmiennych już ustawionych w środowisku."""
    if not path.exists():
        return
    # utf-8-sig: PowerShell 5.1 zapisuje UTF-8 z BOM.
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if value:  # puste "KLUCZ=" traktujemy jak brak ustawienia
            os.environ.setdefault(key, value)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class RoleSpec:
    name: str
    title: str
    prompt: str
    temperature: float = 0.6


@dataclass(slots=True)
class ChannelSpec:
    name: str
    title: str


@dataclass(slots=True)
class Settings:
    # model
    backend: str  # "ollama" | "anthropic"
    anthropic_model: str
    anthropic_effort: str
    ollama_host: str
    ollama_model: str
    num_ctx: int
    concurrency: int
    think: bool
    llm_timeout: float
    # rytm
    tick_seconds: float
    tick_jitter: float
    max_tool_rounds: int
    post_policy: str
    # web
    web_enabled: bool
    web_max_results: int
    fetch_max_chars: int
    # obserwacja
    show_thinking: bool
    thinking_chars: int
    log_level: str
    data_dir: Path
    viewer_port: int
    # społeczność
    topic: str
    seed_questions: list[str]
    channels: list[ChannelSpec]
    roles: list[RoleSpec]
    # tryby
    dry_run: bool = False
    max_ticks: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "agora.sqlite3"

    @property
    def model_label(self) -> str:
        if self.backend == "anthropic":
            return f"{self.anthropic_model} (Claude API, effort {self.anthropic_effort})"
        return f"{self.ollama_model} @ {self.ollama_host}"


def load_settings(config_path: Path | None = None, *, dry_run: bool = False, max_ticks: int | None = None) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    config_path = config_path or (PROJECT_ROOT / "config" / "community.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    roles = [
        RoleSpec(
            name=str(r["name"]),
            title=str(r.get("title", r["name"])),
            prompt=str(r.get("prompt", "")).strip(),
            temperature=float(r.get("temperature", 0.6)),
        )
        for r in raw.get("roles", [])
    ]
    if not roles:
        raise ValueError("config/community.yaml: brak sekcji roles")

    channels = [ChannelSpec(name=str(c["name"]), title=str(c.get("title", c["name"]))) for c in raw.get("channels", [])]
    if not any(c.name == "ogolny" for c in channels):
        channels.insert(0, ChannelSpec(name="ogolny", title="Dyskusja ogólna"))
    if not any(c.name == "graf" for c in channels):
        channels.append(ChannelSpec(name="graf", title="Dziennik zmian w grafie wiedzy"))

    data_dir = Path(_env("AGORA_DATA_DIR", "./community"))
    if not data_dir.is_absolute():
        data_dir = (PROJECT_ROOT / data_dir).resolve()

    backend = _env("AGORA_BACKEND", "ollama").strip().lower()
    if backend not in {"ollama", "anthropic"}:
        raise ValueError(f".env: AGORA_BACKEND={backend!r}; dozwolone: ollama, anthropic")
    effort = _env("ANTHROPIC_EFFORT", "medium").strip().lower()
    if effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise ValueError(f".env: ANTHROPIC_EFFORT={effort!r}; dozwolone: low, medium, high, xhigh, max")

    return Settings(
        backend=backend,
        anthropic_model=_env("ANTHROPIC_MODEL", "claude-sonnet-5-5").strip(),
        anthropic_effort=effort,
        ollama_host=_env("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=_env("OLLAMA_MODEL", "qwen3:8b"),
        num_ctx=_env_int("OLLAMA_NUM_CTX", 16384),
        concurrency=max(1, _env_int("OLLAMA_CONCURRENCY", 2)),
        think=_env_bool("OLLAMA_THINK", True),
        llm_timeout=_env_float("OLLAMA_TIMEOUT", 600.0),
        tick_seconds=_env_float("AGORA_TICK_SECONDS", 20.0),
        tick_jitter=_env_float("AGORA_TICK_JITTER", 10.0),
        max_tool_rounds=_env_int("AGORA_MAX_TOOL_ROUNDS", 6),
        post_policy=_env("AGORA_POST_POLICY", "strict").strip().lower(),
        web_enabled=_env_bool("AGORA_WEB", True),
        web_max_results=_env_int("AGORA_WEB_MAX_RESULTS", 5),
        fetch_max_chars=_env_int("AGORA_FETCH_MAX_CHARS", 5000),
        show_thinking=_env_bool("AGORA_SHOW_THINKING", True),
        thinking_chars=_env_int("AGORA_THINKING_CHARS", 600),
        log_level=_env("AGORA_LOG_LEVEL", "INFO").upper(),
        data_dir=data_dir,
        viewer_port=_env_int("AGORA_VIEWER_PORT", 8765),
        topic=str(raw.get("topic", "")).strip(),
        seed_questions=[str(q).strip() for q in raw.get("seed_questions", []) if str(q).strip()],
        channels=channels,
        roles=roles,
        dry_run=dry_run,
        max_ticks=max_ticks,
    )
