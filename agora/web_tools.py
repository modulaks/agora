"""Darmowy dostęp do sieci: DuckDuckGo (ddgs, bez klucza), pobieranie stron (httpx + trafilatura), Wikipedia API."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Annotated, Any

import httpx
from pydantic import Field

from .db import DB
from .kb_tools import TickScratch
from .tools import ToolError, ToolSpec, tool

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AgoraResearchBot/0.1 (+local research community)"


def _search_sync(query: str, max_results: int, region: str) -> list[dict[str, str]]:
    from ddgs import DDGS  # import leniwy, żeby brak pakietu nie wywracał importu modułu

    with DDGS() as ddg:
        raw = ddg.text(query, region=region, safesearch="moderate", max_results=max_results) or []
    out = []
    for r in raw:
        out.append({"title": r.get("title", ""), "url": r.get("href", r.get("url", "")), "snippet": r.get("body", "")})
    return out


def _extract_sync(html: str, url: str) -> str:
    import trafilatura

    text = trafilatura.extract(html, url=url, include_comments=False, include_tables=True, favor_recall=True) or ""
    return text


def make_web_tools(db: DB, author: str, scratch: TickScratch, *, cache_dir: Path, max_results: int = 5, fetch_max_chars: int = 5000) -> list[ToolSpec]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    @tool
    async def web_search(
        query: Annotated[str, Field(description="zapytanie, 2-8 słów; po polsku lub angielsku")],
        lang: Annotated[str, Field(description="pl | en (region wyników)")] = "en",
    ) -> str:
        """Szuka w sieci (DuckDuckGo). Zwraca tytuły, URL-e i fragmenty. Potem przeczytaj stronę przez web_fetch."""
        query = " ".join(query.split())
        if len(query) < 3:
            raise ToolError("zapytanie jest puste.")
        region = "pl-pl" if lang.lower().startswith("pl") else "us-en"
        scratch.web_calls += 1
        try:
            rows = await asyncio.to_thread(_search_sync, query, max_results, region)
        except Exception as e:  # noqa: BLE001
            raise ToolError(f"wyszukiwanie nie powiodło się ({type(e).__name__}: {e}). Spróbuj innego zapytania za chwilę.")
        if not rows:
            return "brak wyników; spróbuj inaczej sformułować zapytanie lub zmienić lang"
        lines = [f"{i + 1}. {r['title']}\n   {r['url']}\n   {r['snippet'][:240]}" for i, r in enumerate(rows)]
        return "\n".join(lines)

    @tool
    async def web_fetch(
        url: Annotated[str, Field(description="pełny URL strony do przeczytania")],
        max_chars: Annotated[int, Field(description="ile znaków treści zwrócić (domyślnie z konfiguracji)")] = 0,
    ) -> str:
        """Pobiera stronę i zwraca jej treść jako czysty tekst (bez nawigacji i reklam). Czytaj zanim zacytujesz."""
        url = url.strip()
        if not url.startswith("http"):
            raise ToolError("podaj pełny URL zaczynający się od http.")
        limit = max_chars if max_chars and max_chars > 0 else fetch_max_chars
        limit = min(limit, fetch_max_chars * 3)
        key = hashlib.sha1(url.encode("utf-8")).hexdigest()
        cache_file = cache_dir / f"{key}.json"
        scratch.web_calls += 1
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            text, title = cached.get("text", ""), cached.get("title", "")
        else:
            try:
                async with httpx.AsyncClient(follow_redirects=True, timeout=30.0, headers={"User-Agent": USER_AGENT}) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    ctype = resp.headers.get("content-type", "")
                    if "pdf" in ctype:
                        raise ToolError("to PDF; ten projekt czyta tylko strony HTML. Poszukaj wersji HTML albo streszczenia.")
                    html = resp.text
            except ToolError:
                raise
            except Exception as e:  # noqa: BLE001
                raise ToolError(f"nie udało się pobrać strony ({type(e).__name__}: {e}).")
            text = await asyncio.to_thread(_extract_sync, html, url)
            title = ""
            try:
                import re

                m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
                title = " ".join(m.group(1).split())[:200] if m else ""
            except Exception:  # noqa: BLE001
                title = ""
            if not text.strip():
                raise ToolError("strona nie ma czytelnej treści tekstowej (może wymagać JavaScriptu). Spróbuj innego źródła.")
            cache_file.write_text(json.dumps({"url": url, "title": title, "text": text}, ensure_ascii=False), encoding="utf-8")
        db.upsert_source(url, title, author, len(text))
        body = text[:limit]
        more = f"\n[... ucięto, strona ma {len(text)} znaków; wywołaj ponownie z większym max_chars, jeśli trzeba]" if len(text) > limit else ""
        return f"TYTUŁ: {title or '(brak)'}\nURL: {url}\n\n{body}{more}"

    @tool
    async def wiki_lookup(
        query: Annotated[str, Field(description="hasło lub temat")],
        lang: Annotated[str, Field(description="pl | en")] = "en",
    ) -> str:
        """Szuka hasła w Wikipedii i zwraca streszczenie najlepszego trafienia z URL."""
        lang = "pl" if lang.lower().startswith("pl") else "en"
        scratch.web_calls += 1
        base = f"https://{lang}.wikipedia.org"
        try:
            async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": USER_AGENT}) as client:
                r = await client.get(f"{base}/w/api.php", params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 3})
                r.raise_for_status()
                hits = r.json().get("query", {}).get("search", [])
                if not hits:
                    return "brak hasła"
                out = []
                for h in hits[:2]:
                    title = h["title"]
                    s = await client.get(f"{base}/api/rest_v1/page/summary/{title.replace(' ', '_')}")
                    if s.status_code != 200:
                        continue
                    js: dict[str, Any] = s.json()
                    url = js.get("content_urls", {}).get("desktop", {}).get("page", f"{base}/wiki/{title.replace(' ', '_')}")
                    extract = js.get("extract", "")
                    db.upsert_source(url, title, author, len(extract))
                    out.append(f"{title}\n{url}\n{extract[:1200]}")
                return "\n\n".join(out) if out else "brak streszczenia"
        except Exception as e:  # noqa: BLE001
            raise ToolError(f"Wikipedia nie odpowiedziała ({type(e).__name__}: {e}).")

    return [web_search, web_fetch, wiki_lookup]
