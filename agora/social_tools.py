"""Narzędzia społeczne: kanały, wątki, DM-y, notatki, bezczynność. Z polityką przeciw papugowaniu."""

from __future__ import annotations

import re
import unicodedata
from typing import Annotated, Any, Callable

from pydantic import Field

from .db import DB
from .graph import NODE_ID_RE, similarity
from .kb_tools import TickScratch
from .tools import ToolError, ToolSpec, tool

Emit = Callable[[str, dict[str, Any]], None]

MAX_POSTS_PER_TICK = 2
MAX_THREADS_PER_TICK = 1
MAX_DMS_PER_TICK = 2
NOVELTY_THRESHOLD = 0.70


def slugify(text: str, max_len: int = 40) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:max_len].strip("-") or "watek"


def _channel_name(raw: str) -> str:
    return raw.strip().lstrip("#").strip().lower()


def render_message(m: dict[str, Any]) -> str:
    prefix = f"#{m['id']} {m['author']}"
    if m.get("reply_to"):
        prefix += f" (odp. na #{m['reply_to']})"
    if m.get("kind") == "system":
        prefix = f"#{m['id']} [system]"
    return f"{prefix}: {m['text']}"


def make_social_tools(
    db: DB,
    author: str,
    scratch: TickScratch,
    *,
    members: list[str],
    post_policy: str,
    emit: Emit,
    remember_fn: Callable[[str], str],
) -> tuple[list[ToolSpec], Callable[[], None]]:
    def _resolve_channel(name: str) -> dict[str, Any]:
        ch = db.get_channel(_channel_name(name))
        if ch is None:
            names = ", ".join("#" + c["name"] for c in db.list_channels())
            raise ToolError(f"kanał {name!r} nie istnieje. Istniejące: {names}")
        return ch

    def _check_novelty(channel_id: int, text: str) -> None:
        for m in db.last_messages(channel_id, 40):
            if m.get("kind") == "system":
                continue
            s = similarity(m["text"], text)
            if s >= NOVELTY_THRESHOLD:
                who = "twoją własną wiadomość" if m["author"] == author else f"wiadomość {m['author']}a"
                raise ToolError(f"post odrzucony: powtarza {who} #{m['id']} (podobieństwo {s:.2f}). Wnieś coś nowego albo nic nie pisz (idle).")

    def _check_substance(text: str) -> None:
        if post_policy != "strict":
            return
        has_ref = bool(NODE_ID_RE.search(text))
        has_question = "?" in text
        if not (has_ref or has_question or scratch.graph_ops):
            raise ToolError(
                "post odrzucony (polityka treści): wypowiedź w kanale musi odnosić się do węzłów grafu po ID (np. C3, O2), "
                "zadać pytanie albo iść za operacją na grafie wykonaną w tej turze. Najpierw zmień graf (kb_*), potem pisz."
            )

    def _post(channel: dict[str, Any], text: str, reply_to: int | None) -> int:
        text = " ".join((text or "").split())
        if len(text) < 20:
            raise ToolError("post jest za krótki; napisz pełną myśl albo nic nie pisz.")
        if len(text) > 1600:
            raise ToolError("post jest za długi (limit 1600 znaków, czyli ok. 200 słów). Skróć do sedna.")
        if scratch.posts >= MAX_POSTS_PER_TICK:
            raise ToolError(f"limit {MAX_POSTS_PER_TICK} postów na turę wyczerpany. Resztę zostaw na następną turę.")
        if reply_to:
            target = db.get_message(reply_to)
            if target is None or target["channel_id"] != channel["id"]:
                raise ToolError(f"wiadomość #{reply_to} nie istnieje w kanale #{channel['name']}.")
        _check_novelty(channel["id"], text)
        _check_substance(text)
        mid = db.add_message(channel["id"], author, text, reply_to=reply_to or None)
        scratch.posts += 1
        emit("post", {"message_id": mid, "channel": channel["name"], "text": text, "reply_to": reply_to or None})
        return mid

    @tool
    def post(
        channel: Annotated[str, Field(description="nazwa kanału, np. ogolny albo t-nazwa-watku")],
        text: Annotated[str, Field(description="treść (do 120 słów), z odniesieniami do ID węzłów")],
        reply_to: Annotated[int, Field(description="ID wiadomości, na którą odpowiadasz (0 = nie jest odpowiedzią)")] = 0,
    ) -> str:
        """Publikuje wiadomość w kanale. Odrzuca powtórki i ogólniki."""
        ch = _resolve_channel(channel)
        mid = _post(ch, text, reply_to or None)
        return f"opublikowano #{mid} w #{ch['name']}"

    @tool
    def open_thread(
        title: Annotated[str, Field(description="krótki tytuł wątku (do 8 słów)")],
        first_post: Annotated[str, Field(description="pierwsza wiadomość: o co chodzi i czego oczekujesz od innych")],
        question_id: Annotated[str, Field(description="ID pytania z grafu (Q...), którego dotyczy wątek; zalecane")] = "",
    ) -> str:
        """Zakłada nowy wątek (kanał) dla konkretnego podproblemu. Jeden wątek na turę."""
        title = " ".join((title or "").split())
        if len(title) < 6:
            raise ToolError("tytuł wątku jest za krótki.")
        if scratch.posts >= MAX_POSTS_PER_TICK:
            raise ToolError("limit postów na turę wyczerpany; załóż wątek w następnej turze.")
        existing = db.list_channels()
        if sum(1 for c in existing if c["kind"] == "thread" and c["created_by"] == author and c["id"] > (scratch_threads_marker[0])) >= MAX_THREADS_PER_TICK:
            raise ToolError("w tej turze założyłeś już wątek.")
        for c in existing:
            if similarity(c["title"], title) >= 0.6:
                raise ToolError(f"podobny wątek już istnieje: #{c['name']} ({c['title']}). Pisz tam.")
        qid = question_id.strip().upper() if question_id else ""
        if qid:
            q = db.get_node(qid)
            if q is None or q["type"] != "Q":
                raise ToolError(f"{qid} nie jest pytaniem w grafie.")
            if any(c["question_id"] == qid for c in existing):
                other = next(c for c in existing if c["question_id"] == qid)
                raise ToolError(f"pytanie {qid} ma już wątek #{other['name']}. Pisz tam.")
        base = "t-" + slugify(title)
        name = base
        n = 2
        while db.get_channel(name):
            name = f"{base}-{n}"
            n += 1
        general = db.get_channel("ogolny")
        cid = db.create_channel(name, title, author, kind="thread", parent_id=general["id"] if general else None, question_id=qid or None)
        ch = db.get_channel(name)
        assert ch is not None
        mid = _post(ch, first_post, None)
        if general:
            db.add_message(general["id"], "system", f"{author} założył wątek #{name}: {title}" + (f" (pytanie {qid})" if qid else ""), kind="system")
        emit("thread", {"channel": name, "title": title, "question_id": qid or None, "message_id": mid})
        return f"założono wątek #{name} i opublikowano #{mid}"

    scratch_threads_marker = [0]

    def mark_tick_start() -> None:
        rows = db.list_channels()
        scratch_threads_marker[0] = max((c["id"] for c in rows), default=0)

    @tool
    def read_channel(
        channel: Annotated[str, Field(description="nazwa kanału")],
        last_n: Annotated[int, Field(description="ile ostatnich wiadomości (max 40)")] = 20,
    ) -> str:
        """Czyta ostatnie wiadomości z kanału (także starsze niż te, które dostałeś w turze)."""
        ch = _resolve_channel(channel)
        rows = db.last_messages(ch["id"], max(1, min(int(last_n), 40)))
        if not rows:
            return f"#{ch['name']} jest pusty"
        return f"#{ch['name']} ({ch['title']}):\n" + "\n".join(render_message(m) for m in rows)

    @tool
    def list_channels() -> str:
        """Lista kanałów i wątków z liczbą wiadomości."""
        rows = db.list_channels()
        return "\n".join(
            f"#{c['name']} [{c['kind']}] {c['title']} ({c['n_messages']} wiad.)" + (f" pytanie {c['question_id']}" if c.get("question_id") else "")
            for c in rows
        )

    @tool
    def dm(
        to: Annotated[str, Field(description="nazwa członka społeczności")],
        text: Annotated[str, Field(description="konkretna prośba lub pytanie (co dokładnie i po co)")],
    ) -> str:
        """Wysyła prywatną wiadomość do innego członka. Adresat odpowie w swojej turze (dm_reply)."""
        to = to.strip().lower()
        if to == author:
            raise ToolError("nie pisz do siebie.")
        if to not in members:
            raise ToolError(f"nie ma członka {to!r}. Członkowie: {', '.join(members)}")
        text = " ".join((text or "").split())
        if len(text) < 15:
            raise ToolError("wiadomość jest za krótka.")
        if scratch_dm_count[0] >= MAX_DMS_PER_TICK:
            raise ToolError(f"limit {MAX_DMS_PER_TICK} DM-ów na turę.")
        for prev in db.dms_for(author)[-20:]:
            if prev["sender"] == author and prev["recipient"] == to and similarity(prev["text"], text) >= NOVELTY_THRESHOLD:
                raise ToolError(f"wysłałeś już podobny DM do {to} (#{prev['id']}). Poczekaj na odpowiedź.")
        did = db.add_dm(author, to, text)
        scratch_dm_count[0] += 1
        emit("dm", {"dm_id": did, "to": to, "text": text})
        return f"wysłano DM #{did} do {to}"

    scratch_dm_count = [0]

    @tool
    def dm_reply(
        dm_id: Annotated[int, Field(description="ID DM-a, na który odpowiadasz")],
        text: Annotated[str, Field(description="odpowiedź")],
    ) -> str:
        """Odpowiada na DM skierowany do ciebie."""
        orig = db.get_dm(int(dm_id))
        if orig is None or orig["recipient"] != author:
            raise ToolError(f"DM #{dm_id} nie jest skierowany do ciebie.")
        if orig["answered"]:
            raise ToolError(f"na DM #{dm_id} już odpowiedziano.")
        text = " ".join((text or "").split())
        if len(text) < 10:
            raise ToolError("odpowiedź jest za krótka.")
        did = db.add_dm(author, orig["sender"], text, reply_to=orig["id"])
        emit("dm", {"dm_id": did, "to": orig["sender"], "text": text, "reply_to": orig["id"]})
        return f"odpowiedziano DM #{did} do {orig['sender']}"

    @tool
    def remember(text: Annotated[str, Field(description="krótka notatka: co ustaliłeś, czego pilnować, co planujesz")]) -> str:
        """Zapisuje notatkę w twojej prywatnej pamięci, widocznej w następnych turach."""
        return remember_fn(" ".join((text or "").split()))

    @tool
    def idle(reason: Annotated[str, Field(description="jedno zdanie, dlaczego nic nie robisz")] = "") -> str:
        """Nic nie robisz w tej turze. Właściwe, gdy nie masz nic nowego."""
        emit("idle", {"reason": reason})
        return "OK, w tej turze nic nie publikujesz."

    def reset_tick() -> None:
        scratch_dm_count[0] = 0
        mark_tick_start()

    specs = [post, open_thread, read_channel, list_channels, dm, dm_reply, remember, idle]
    return specs, reset_tick
