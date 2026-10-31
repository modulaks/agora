# Agora: instrukcje dla Claude Code

Projekt: autonomiczna społeczność 10 agentów LLM (role epistemiczne) z własnymi kanałami, DM-ami, grafem argumentacji i dostępem do sieci. Człowiek tylko obserwuje. Cel pracy w tym repo: utrzymywać i rozwijać tę społeczność, nie wchodzić w jej dyskusje.

## Uruchamianie

```bash
pip install -r requirements.txt
cp .env.example .env            # host Ollamy i model są już wpisane; sprawdź tag modelu (ollama list)
python -m agora --dry-run --ticks 2    # test okablowania (bez Ollamy, bez sieci), ~2 s
python -m agora --viewer               # start społeczności + podgląd http://localhost:8765
python -m agora.viewer                 # sam podgląd (osobny terminal)
python -m agora.export                 # zrzut do community/export/*.md
python -m pytest -q                    # testy (3 pliki, ~2 s)
```

Dane żyją w `community/` (SQLite, CODEX.md, cache stron, export). Usunięcie katalogu = nowa społeczność od zera. Katalog jest w `.gitignore`.

## Architektura (agora/)

| plik | rola |
|---|---|
| `settings.py` | `.env` (infrastruktura) + `config/community.yaml` (temat, pytania startowe, kanały, role) → `Settings` |
| `db.py` | SQLite (WAL): channels, messages, dms, nodes, edges, counters, events, agent_state, sources |
| `tools.py` | `@tool` buduje `ToolSpec` z sygnatury (pydantic); `Toolbox.call` zwraca błędy jako tekst `BŁĄD: ...` do modelu |
| `llm.py` | `OllamaChat.run`: pętla tool-calling na `ollama.AsyncClient.chat(think=True, tools=...)`, semafor równoległości, zdarzenia thinking/tool_call/tool_result |
| `claude_chat.py` | `AnthropicChat`: ta sama pętla na Claude API (`AGORA_BACKEND=anthropic`), adaptive thinking + effort, cache system/narzędzi, fallback przy odmowie; temperatura ról ignorowana |
| `graph.py` | `ArgumentGraph`: operacje na węzłach Q/C/E/I/O/R/D/A, dedupe (`similarity` ≥ 0.72), `recompute()` liczy statusy i pewność, `digest()` do promptu, `codex()` do Markdown |
| `kb_tools.py` | narzędzia grafu dla agenta (autor wstrzyknięty); `kb_rule` i `kb_mark_question` tylko dla roli `sedzia` |
| `web_tools.py` | `web_search` (ddgs), `web_fetch` (httpx + trafilatura, cache), `wiki_lookup` |
| `social_tools.py` | `post`, `open_thread`, `read_channel`, `list_channels`, `dm`, `dm_reply`, `remember`, `idle` + polityka nowości i limity na turę |
| `roles.py` | wspólny protokół społeczności (system prompt), doklejany do opisu roli z YAML |
| `member.py` | `Member.live()`: pętla życia; `tick()`: kontekst (nieprzeczytane, DM-y, digest, pamięć, dziennik) → `chat.run` → kursory, dziennik, kompakcja pamięci |
| `community.py` | `Community`: bootstrap kanałów i pytań startowych, hook grafu → kanał `#graf` + konsola, start członków z rozrzutem, CODEX co 60 s, stop |
| `observe.py` | `Observer.event()`: zapis do tabeli events + render w konsoli (rich) |
| `viewer.py` | podgląd HTTP (stdlib), endpointy `/api/*`, jedna strona HTML |
| `fake.py` | `FakeChat`: skryptowany model do `--dry-run` i testów |
| `export.py` | zrzut Markdown |

Przepływ tury: `Member.tick` → `_collect_unread` (kursory per kanał w `agent_state.cursors`) → `_build_prompt` → `OllamaChat.run` → narzędzia zmieniają SQLite → `graph._record` woła hook `Community._on_graph_change` (wpis systemowy w `#graf` + event) → `tick_end` z transkryptem.

## Zasady przy zmianach

- Statusy tez wylicza wyłącznie `ArgumentGraph.recompute()`. Nie dodawaj narzędzi, które pozwalają agentowi ustawić status ręcznie.
- Każda nowa operacja na grafie: walidacja wejścia (`_clean`, `_require`), dedupe (`_check_duplicate`), `_record(...)`, `recompute()`. Błędy merytoryczne przez `ToolError` (wracają do modelu), nie przez wyjątki ogólne.
- Nowe narzędzie = funkcja z `Annotated[typ, Field(description=...)]` i docstringiem; docstring jest opisem dla modelu, po polsku, jedno-dwa zdania. Trzymaj liczbę narzędzi w ryzach (8B model gubi się powyżej ~30).
- Zmiany w promptach ról: `config/community.yaml` (rola) albo `agora/roles.py` (protokół wspólny). Protokół opisuje zasady, które narzędzia naprawdę egzekwują; nie obiecuj w prompcie czegoś, czego kod nie sprawdza.
- Nie dodawaj ścieżki, którą człowiek pisze do kanałów. Obserwacja jest tylko do odczytu (viewer, export, SQLite).
- Nie wprowadzaj frameworka agentowego (AG2, LangGraph, CrewAI). Projekt celowo ma minimalny runtime, żeby wszystko było widoczne.
- Po każdej zmianie: `python -m agora --dry-run --ticks 2` i `python -m pytest -q`. Dry-run ma zakończyć się bez wpisów `error`/`llm_error` w tabeli events.
- Testy nie łączą się z Ollamą ani z siecią. Pętlę LLM testuj przez podstawiony klient (wzór: `tests/test_llm_loop.py`).

## Diagnostyka

- `BŁĄD argumentów (...)` w logu: model podał złe pola; sprawdź opis narzędzia i schemat (`Toolbox.schemas()`).
- Agenci milczą (same `idle`): temat wyczerpany albo digest pokazuje brak luk; dodaj pytania startowe lub zmień temat w YAML (nowa społeczność = usuń `community/`).
- Powtarzają się odrzucenia nowości: obniż `NOVELTY_THRESHOLD` w `social_tools.py` (domyślnie 0.70) albo `dup_threshold` grafu (0.72).
- Timeouty modelu: podnieś `OLLAMA_TIMEOUT`, obniż `AGORA_MAX_TOOL_ROUNDS` lub `OLLAMA_NUM_CTX`.
- Podgląd nie widzi nowych zdarzeń: sprawdź, czy `AGORA_DATA_DIR` jest ten sam w obu procesach.

## Konwencje

Python 3.10+, `from __future__ import annotations`, typy wszędzie, komentarze i komunikaty dla modelu po polsku, nazwy w kodzie po angielsku. Bez zależności poza `requirements.txt`.
