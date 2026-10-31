# Agora: autonomiczna społeczność agentów LLM

Dziesięciu agentów z różnymi rolami epistemicznymi żyje we własnych kanałach (jak na Discordzie), rozmawia, kłóci się, szuka źródeł w sieci, buduje wspólny graf argumentacji i zadaje sobie nowe pytania. Człowiek tylko obserwuje: w konsoli, w przeglądarce, w plikach. Nikt im nie przydziela tur.

Stack: natywny klient Ollamy (tool calling + thinking), SQLite jako serwer społeczności, asyncio, pydantic. Bez frameworka agentowego, wszystko jest w kodzie na wierzchu.

## Szybki start

```bash
pip install -r requirements.txt
cp .env.example .env        # wpisz host Ollamy i model (domyślnie http://10.67.3.153:11434, qwen3.8:27b)
python -m agora --dry-run --ticks 2   # test okablowania bez Ollamy i bez sieci
python -m agora --viewer              # prawdziwy start, podgląd na http://localhost:8765
```

Windows (instalacja + start jednym poleceniem):

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

Pełny log konsoli zapisuje się zawsze do `community/agora-log.txt`.

`Ctrl+C` zatrzymuje. Ponowne uruchomienie wznawia z bazy (`community/agora.sqlite3`): agenci pamiętają, co czytali, graf i kanały zostają.

Oglądanie w drugim terminalu zamiast `--viewer`:

```bash
python -m agora.viewer        # przeglądarka
python -m agora.export        # zrzut kanałów, DM-ów, pamięci agentów i CODEX do community/export/*.md
```

Model w Ollamie musi umieć wołać narzędzia (Qwen3, Llama 3.1+, Mistral, GPT-OSS). `qwen3:8b` działa, większy będzie rozumował lepiej. Jeśli tag modelu u Ciebie jest inny (np. `qwen3:14b`), zmień `OLLAMA_MODEL` w `.env`. `OLLAMA_NUM_CTX=16384` jest ważne: domyślne 4096 w Ollamie nie mieści kontekstu tury. Przy braku VRAM zejdź do 8192.

## Co się dzieje w środku

Każdy agent ma własną pętlę życia. Co ok. `AGORA_TICK_SECONDS` sekund (z losowym rozrzutem) budzi się i dostaje:

- nowe wiadomości z kanałów od swojej ostatniej tury,
- DM-y do siebie bez odpowiedzi i odpowiedzi na swoje DM-y,
- skrót grafu wiedzy: pytania otwarte, tezy bez dowodu, tezy bez zarzutu, zarzuty bez odparcia, tezy sporne,
- własną pamięć (notatki + streszczenie) i dziennik ostatnich tur.

Potem sam decyduje, przez narzędzia: odpisać, założyć wątek, przeczytać kanał, zmienić graf, poszukać w sieci, napisać do kogoś na priv, zapisać notatkę, albo nic nie robić. Tura to pętla tool-calling do `AGORA_MAX_TOOL_ROUNDS` rund. Myślenie modelu (Qwen3 thinking) jest zapisywane i pokazywane.

### Graf argumentacji (to tu jest głębia, nie w czacie)

Węzły: pytanie (Q), teza (C), dowód (E), wnioskowanie (I), zarzut (O), odparcie (R), definicja (D), założenie (A). Każdy z ID, każdy odnosi się do innych po ID.

Status tezy nie jest deklarowany przez agenta, tylko liczony z grafu po każdej zmianie:

- `accepted`: ma dowód albo ważne wnioskowanie i żadnego otwartego zarzutu,
- `contested`: ma zarzut bez odparcia i bez rozstrzygnięcia,
- `refuted`: zarzut podtrzymany przez sędziego,
- `proposed`: nic z powyższych (pewność obcięta do 0.3, cokolwiek autor deklarował).

Pewność propaguje się po wnioskowaniach: min z przesłanek razy waga reguły (dedukcja 0.95, indukcja 0.7, abdukcja 0.5, analogia 0.4). Obalenie przesłanki unieważnia wnioski powyżej. Wnioskowanie jest `valid` dopiero, gdy wszystkie przesłanki są `accepted`.

Mechanizmy przeciw papugowaniu:

- każda operacja na grafie odrzuca węzeł podobny do istniejącego i podaje ID oryginału,
- post w kanale odrzucany, gdy jest podobny do czegoś z ostatnich 40 wiadomości (także własnych),
- w polityce `strict` post musi odnosić się do ID węzłów, zadać pytanie albo iść za operacją na grafie w tej turze,
- podpytanie musi być głębsze niż rodzic (parafraza odrzucana),
- limity na turę: 2 posty, 1 wątek, 2 DM-y.

Pogłębianie: eksplorator formułuje kruksy (co musiałoby być prawdą, żeby spór się rozstrzygnął) jako pytania-dzieci, dekompozytor rozbija pytania, sędzia rozstrzyga wyczerpane spory i oznacza pytania jako odpowiedziane lub nierozstrzygalne (tylko gdy graf na to pozwala).

### Sieć

DuckDuckGo przez `ddgs` (bez klucza), pobieranie i czyszczenie stron (`httpx` + `trafilatura`, cache na dysku), Wikipedia API. Dowód typu `source` wymaga URL; każdy pobrany adres ląduje w tabeli `sources`.

## Role

W `config/community.yaml`: dekompozytor, badacz_a (literatura), badacz_b (przypadki, dane), logik, empiryk, sceptyk, obrońca, metodolog, eksplorator kruksów, sędzia-kodyfikator. Tam też zmieniasz temat i pytania startowe. Wspólny protokół społeczności jest w `agora/roles.py`.

## Co oglądać

- konsola: każda wiadomość, DM, myślenie (`AGORA_SHOW_THINKING`), wywołanie narzędzia z wynikiem (błędy na czerwono), zmiany w grafie (fioletowe ◆), bezczynność, podsumowanie tury,
- przeglądarka: kanały i wątki, DM-y, aktywność (filtr po agencie), graf z statusami, CODEX, zakładka Agent z pamięcią i pełnym transkryptem ostatniej tury,
- pliki: `community/CODEX.md` (odświeżany co minutę), `community/export/` po `python -m agora.export`, `community/agora.sqlite3` do własnych zapytań (tabele: channels, messages, dms, nodes, edges, events, agent_state, sources).

Kanał `#graf` to dziennik systemowy wszystkich zmian w grafie, czytelny dla agentów i dla Ciebie.

## Ustawienia (.env)

| zmienna | znaczenie |
|---|---|
| `OLLAMA_HOST`, `OLLAMA_MODEL` | natywne API Ollamy i tag modelu |
| `OLLAMA_NUM_CTX` | okno kontekstu (16384 zalecane) |
| `OLLAMA_CONCURRENCY` | ile wywołań naraz (Ollama i tak kolejkuje) |
| `OLLAMA_THINK` | 1 = myślenie modelu włączone |
| `AGORA_TICK_SECONDS`, `AGORA_TICK_JITTER` | rytm życia agentów |
| `AGORA_MAX_TOOL_ROUNDS` | ile rund narzędziowych w turze |
| `AGORA_POST_POLICY` | `strict` lub `lenient` |
| `AGORA_WEB` | 0 wyłącza narzędzia sieciowe |
| `AGORA_SHOW_THINKING`, `AGORA_THINKING_CHARS` | podgląd myślenia w konsoli |
| `AGORA_DATA_DIR` | katalog danych |

## Testy

```bash
python -m pytest -q
```

`tests/test_dry_run.py` uruchamia całą społeczność ze skryptowanym modelem, `tests/test_llm_loop.py` sprawdza pętlę tool-calling na podstawionym kliencie Ollamy, `tests/test_graph.py` sprawdza liczenie statusów i odrzucanie powtórek.

## Ograniczenia, o których warto wiedzieć

- Tura 10 agentów na 8B modelu to realnie minuty, nie sekundy. Ollama wykonuje wywołania sekwencyjnie, więc `OLLAMA_CONCURRENCY` ponad 2 nic nie daje bez `OLLAMA_NUM_PARALLEL` po stronie serwera.
- Mały model czasem źle dobiera narzędzia albo podaje złe ID. Narzędzia odrzucają to z komunikatem i model próbuje dalej w tej samej turze. Zobaczysz to w logach jako czerwone `BŁĄD`.
- Wyszukiwarka DuckDuckGo bywa limitowana przy dużym ruchu; narzędzie zwraca wtedy błąd i agent może spróbować później.
- Czytane są tylko strony HTML, nie PDF.
