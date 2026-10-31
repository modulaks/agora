"""python -m agora  → uruchamia społeczność (Ctrl+C zatrzymuje).

Opcje:
  --dry-run        skryptowany model, bez Ollamy i bez sieci (test okablowania)
  --ticks N        każdy agent kończy po N turach (domyślnie bez końca)
  --viewer         uruchom też podgląd w przeglądarce (http://localhost:PORT)
  --config PATH    inny plik community.yaml
  --quiet          bez wypisywania na konsolę (zostaje baza i podgląd)
  --model M        llama | openai | sonnet | haiku | opus, albo dostawca:model
  --env PATH       .env innego projektu (klucze API)
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import threading
from pathlib import Path

from .community import Community
from .settings import load_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agora", description="Autonomiczna społeczność agentów LLM. Ty tylko obserwujesz.")
    parser.add_argument("--dry-run", action="store_true", help="skryptowany model, bez Ollamy i bez sieci")
    parser.add_argument("--ticks", type=int, default=None, help="liczba tur na agenta (domyślnie bez końca)")
    parser.add_argument("--viewer", action="store_true", help="uruchom podgląd w przeglądarce w tym samym procesie")
    parser.add_argument("--config", type=Path, default=None, help="ścieżka do community.yaml")
    parser.add_argument("--quiet", action="store_true", help="nie wypisuj zdarzeń na konsolę")
    parser.add_argument(
        "--model", default=None, metavar="MODEL",
        help="llama | openai | sonnet | haiku | opus, albo dostawca:model (openai:gpt-5, ollama:qwen3:8b). Domyślnie AGORA_BACKEND z .env",
    )
    parser.add_argument("--env", type=Path, default=None, metavar="ŚCIEŻKA", help=".env innego projektu (np. z kluczami API); to samo co AGORA_ENV_FILE")
    args = parser.parse_args(argv)

    settings = load_settings(args.config, dry_run=args.dry_run, max_ticks=args.ticks, model_choice=args.model, env_file=args.env)
    if args.dry_run:
        settings.tick_seconds = min(settings.tick_seconds, 1.0)
        settings.tick_jitter = 0.2
        settings.web_enabled = False
        from .fake import FakeChat

        community = Community(settings, chat=FakeChat(), quiet=args.quiet)
    else:
        community = Community(settings, quiet=args.quiet)

    if args.viewer:
        from .viewer import serve

        threading.Thread(target=serve, args=(settings.db_path, settings.data_dir, settings.viewer_port, settings.model_label), daemon=True).start()
        print(f"Podgląd: http://localhost:{settings.viewer_port}")

    async def runner() -> None:
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(community.run())

        def _stop() -> None:
            community.stop()
            if not task.done():
                task.cancel()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _stop)
            except (NotImplementedError, RuntimeError):
                pass
        try:
            await task
        except asyncio.CancelledError:
            pass

    try:
        asyncio.run(runner())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
