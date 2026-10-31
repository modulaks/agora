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
    backend: str  # "ollama" | "openai" | "anthropic"
    anthropic_model: str
    anthropic_effort: str
    openai_model: str
    env_file: Path | None  # .env innego projektu, jeśli użyty
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
        if self.backend == "openai":
            return f"{self.openai_model} (OpenAI API)"
        return f"{self.ollama_model} @ {self.ollama_host}"


BACKENDS = ("ollama", "openai", "anthropic")

# Skróty dla --model. None = model z .env danego backendu.
MODEL_ALIASES: dict[str, tuple[str, str | None]] = {
    "ollama": ("ollama", None), "llama": ("ollama", None), "lama": ("ollama", None),
    "openai": ("openai", None), "gpt": ("openai", None),
    "anthropic": ("anthropic", None), "claude": ("anthropic", None),
    "sonnet": ("anthropic", "claude-sonnet-5-5"),
    "haiku": ("anthropic", "claude-haiku-5-5"),
    "opus": ("anthropic", "claude-opus-5-5"),
}


def parse_model_choice(choice: str) -> tuple[str, str | None]:
    """--model: skrót (llama, openai, sonnet, haiku), dostawca:model (openai:gpt-5) albo sama nazwa modelu."""
    c = choice.strip()
    if c.lower() in MODEL_ALIASES:
        return MODEL_ALIASES[c.lower()]
    provider, sep, model = c.partition(":")
    if sep and provider.lower() in BACKENDS + ("claude", "llama", "lama", "gpt"):
        backend = MODEL_ALIASES[provider.lower()][0]
        return backend, model.strip() or None
    if c.startswith("claude-"):
        return "anthropic", c
    if c.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai", c
    return "ollama", c  # np. qwen3.8:27b


def load_env_files(env_file: Path | None = None) -> Path | None:
    """Najpierw .env tego projektu (ma pierwszeństwo), potem .env innego projektu: z niego bierzemy to, czego brakuje (np. klucze)."""
    load_dotenv(PROJECT_ROOT / ".env")
    extra = env_file or (Path(os.environ["AGORA_ENV_FILE"]) if os.environ.get("AGORA_ENV_FILE") else None)
    if extra is None:
        return None
    extra = extra.expanduser()
    if extra.is_dir():
        extra = extra / ".env"
    if not extra.is_file():
        raise SystemExit(f"Nie ma pliku .env pod {extra} (podany w --env albo AGORA_ENV_FILE).")
    load_dotenv(extra)
    return extra


def load_settings(
    config_path: Path | None = None,
    *,
    dry_run: bool = False,
    max_ticks: int | None = None,
    model_choice: str | None = None,
    env_file: Path | None = None,
) -> Settings:
    used_env = load_env_files(env_file)
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
    models = {
        "ollama": _env("OLLAMA_MODEL", "qwen3:8b").strip(),
        "openai": _env("OPENAI_MODEL", "gpt-5-mini").strip(),
        "anthropic": _env("ANTHROPIC_MODEL", "claude-sonnet-5-5").strip(),
    }
    if model_choice:
        backend, chosen = parse_model_choice(model_choice)
        if chosen:
            models[backend] = chosen
    if backend not in BACKENDS:
        raise SystemExit(f".env: AGORA_BACKEND={backend!r}; dozwolone: {', '.join(BACKENDS)}")
    effort = _env("ANTHROPIC_EFFORT", "medium").strip().lower()
    if effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise SystemExit(f".env: ANTHROPIC_EFFORT={effort!r}; dozwolone: low, medium, high, xhigh, max")

    return Settings(
        backend=backend,
        anthropic_model=models["anthropic"],
        anthropic_effort=effort,
        openai_model=models["openai"],
        env_file=used_env,
        ollama_host=_env("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=models["ollama"],
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
