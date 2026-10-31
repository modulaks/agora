"""Podgląd w przeglądarce (tylko do oglądania): kanały, wiadomości, DM-y, myślenie agentów, graf, CODEX.

python -m agora.viewer  → http://localhost:8765  (czyta ten sam plik SQLite, nic nie zmienia)
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .db import DB
from .graph import ArgumentGraph

HTML = r"""<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><title>Agora · podgląd</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#141618;--panel:#1c1f22;--line:#2b3035;--fg:#e6e6e6;--dim:#9aa3ad;--acc:#7cc4ff;--warn:#ffb86b;--bad:#ff6b6b;--ok:#8be28b;--mag:#d58cff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif;height:100vh;display:grid;grid-template-columns:230px 1fr 420px;grid-template-rows:auto 1fr}
header{grid-column:1/4;padding:8px 14px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:center;background:var(--panel)}
header b{color:var(--acc)}header span{color:var(--dim)}
aside,main,section{overflow:auto;padding:10px}aside{border-right:1px solid var(--line)}section{border-left:1px solid var(--line)}
h3{margin:8px 0 6px;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim)}
.ch,.ag{padding:4px 8px;border-radius:6px;cursor:pointer;display:flex;justify-content:space-between;gap:6px}
.ch:hover,.ag:hover{background:#23282d}.ch.sel,.ag.sel{background:#2a3238}
.ch small,.ag small{color:var(--dim)}
.msg{padding:8px 10px;border-bottom:1px solid var(--line)}.msg .who{font-weight:600}.msg .meta{color:var(--dim);font-size:12px;margin-left:6px}
.msg.system{color:var(--dim);font-style:italic}.msg.system .who{color:var(--dim)}
.msg .txt{white-space:pre-wrap;margin-top:2px}
.tabs{display:flex;gap:6px;margin-bottom:8px}.tabs button{background:#23282d;border:1px solid var(--line);color:var(--fg);padding:4px 10px;border-radius:6px;cursor:pointer}.tabs button.sel{background:#2f3a44;border-color:var(--acc)}
.ev{padding:6px 8px;border-bottom:1px solid var(--line);font-size:13px}.ev .k{color:var(--dim);font-size:11px;text-transform:uppercase;margin-right:6px}
.ev.thinking{color:#c9c2a6;font-style:italic;white-space:pre-wrap}.ev.tool_call{color:var(--acc)}.ev.tool_result{color:var(--dim);white-space:pre-wrap}.ev.tool_result.err{color:var(--bad)}
.ev.graph{color:var(--mag)}.ev.idle{color:var(--dim)}.ev.tick_end{color:var(--ok)}.ev.error,.ev.llm_error{color:var(--bad)}.ev.dm{color:var(--warn)}.ev.memory{color:#b7a9ff}
.node{padding:6px 8px;border-bottom:1px solid var(--line);font-size:13px}.node b{color:var(--acc)}.st{font-size:11px;padding:1px 6px;border-radius:10px;background:#2a3238;margin-left:6px}
.st.accepted{color:var(--ok)}.st.contested{color:var(--warn)}.st.refuted{color:var(--bad)}.st.proposed{color:var(--dim)}.st.open,.st.in_progress{color:var(--acc)}.st.answered{color:var(--ok)}
pre{white-space:pre-wrap;font:13px/1.4 ui-monospace,Consolas,monospace;color:#d8d8d8}
label.sw{color:var(--dim);font-size:12px;margin-left:auto}input[type=checkbox]{vertical-align:middle}
.who[data-a]{color:var(--acc)}
</style></head><body>
<header><b>AGORA</b><span id="stats">…</span><label class="sw"><input type="checkbox" id="think" checked> pokazuj myślenie</label><label class="sw"><input type="checkbox" id="follow" checked> przewijaj</label></header>
<aside>
<h3>Kanały</h3><div id="channels"></div>
<h3>Członkowie</h3><div id="agents"></div>
</aside>
<main id="feed"></main>
<section>
<div class="tabs"><button data-t="events" class="sel">Aktywność</button><button data-t="graph">Graf</button><button data-t="codex">CODEX</button><button data-t="agent">Agent</button></div>
<div id="right"></div>
</section>
<script>
const S={channel:'ogolny',agent:null,tab:'events',lastMsg:0,lastEv:0,events:[],colors:{}};
const PAL=['#7cc4ff','#8be28b','#ffd86b','#d58cff','#6bd3ff','#ff8c8c','#a6ff8b','#ffe08b','#ff9ce6','#8bf0ff','#ffffff'];
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const col=a=>{if(!(a in S.colors))S.colors[a]=PAL[Object.keys(S.colors).length%PAL.length];return S.colors[a]};
async function j(u){const r=await fetch(u);return r.json()}
function renderLeft(o){
  const ch=document.getElementById('channels');ch.innerHTML='';
  const items=[...o.channels,{name:'__dm__',title:'Wiadomości prywatne',n_messages:o.stats.dms,kind:'dm'}];
  for(const c of items){const d=document.createElement('div');d.className='ch'+(S.channel===c.name?' sel':'');
    d.innerHTML=`<span>#${esc(c.name==='__dm__'?'dm':c.name)}</span><small>${c.n_messages||0}</small>`;d.title=c.title||'';
    d.onclick=()=>{S.channel=c.name;S.lastMsg=0;document.getElementById('feed').innerHTML='';renderLeft(o);tick()};ch.appendChild(d)}
  const ag=document.getElementById('agents');ag.innerHTML='';
  const all=document.createElement('div');all.className='ag'+(S.agent?'':' sel');all.innerHTML='<span>wszyscy</span>';all.onclick=()=>{S.agent=null;S.lastEv=0;S.events=[];renderLeft(o);tick()};ag.appendChild(all);
  for(const a of o.agents){const d=document.createElement('div');d.className='ag'+(S.agent===a.name?' sel':'');
    d.innerHTML=`<span style="color:${col(a.name)}">${esc(a.name)}</span><small>t${a.tick||0}</small>`;d.title=a.last||'';
    d.onclick=()=>{S.agent=a.name;S.lastEv=0;S.events=[];renderLeft(o);tick()};ag.appendChild(d)}
  document.getElementById('stats').textContent=`wiad. ${o.stats.messages} · DM ${o.stats.dms} · węzły ${o.stats.nodes} · zdarzenia ${o.stats.events} · ${o.model||''}`;
}
function addMsgs(rows){const f=document.getElementById('feed');
  for(const m of rows){const d=document.createElement('div');d.className='msg'+(m.kind==='system'?' system':'');
    const who=m.kind==='system'?'system':m.author;const reply=m.reply_to?` ↩#${m.reply_to}`:'';
    const chan=S.channel==='__dm__'?` → <span style="color:${col(m.recipient)}">${esc(m.recipient)}</span>`:'';
    d.innerHTML=`<span class="who" style="color:${m.kind==='system'?'':col(who)}">${esc(who)}</span>${chan}<span class="meta">#${m.id}${reply} · ${esc((m.ts||'').slice(11,19))}</span><div class="txt">${esc(m.text)}</div>`;
    f.appendChild(d);S.lastMsg=Math.max(S.lastMsg,m.id)}
  if(rows.length&&document.getElementById('follow').checked)f.scrollTop=f.scrollHeight}
function renderEvents(){const r=document.getElementById('right');if(S.tab!=='events')return;const showT=document.getElementById('think').checked;
  r.innerHTML='';for(const e of S.events.slice(-300)){if(e.kind==='thinking'&&!showT)continue;const p=e.payload||{};let txt='';
    switch(e.kind){case 'thinking':txt=p.text;break;case 'tool_call':txt=`${p.tool}(${JSON.stringify(p.args)})`;break;case 'tool_result':txt=`↳ ${p.result}`;break;
      case 'graph':txt=`${p.op} ${p.id}${p.target?' → '+p.target:p.claim?' → '+p.claim:p.parent?' → '+p.parent:''}${p.old?' '+p.old+' → '+p.new:''}: ${p.text||''}`;break;
      case 'idle':txt=`nic nie robi. ${p.reason||''}`;break;case 'tick_start':txt=`tura ${p.tick} (nowych ${p.unread}, DM ${p.dms})`;break;
      case 'tick_end':txt=`■ ${p.summary||''} (${p.seconds}s, ${p.tool_calls} narz.)`;break;case 'post':txt=`#${p.channel}: ${p.text}`;break;case 'dm':txt=`→ ${p.to}: ${p.text}`;break;
      case 'thread':txt=`wątek #${p.channel}: ${p.title}`;break;case 'memory':txt=`${p.what}: ${p.text}`;break;default:txt=p.error||p.text||JSON.stringify(p)}
    const d=document.createElement('div');d.className='ev '+e.kind+((e.kind==='tool_result'&&String(p.result||'').startsWith('BŁĄD'))?' err':'');
    d.innerHTML=`<span class="k">${esc(e.kind)}</span><span style="color:${col(e.agent)}">${esc(e.agent)}</span> ${esc(txt).slice(0,1500)}`;r.appendChild(d)}
  if(document.getElementById('follow').checked)r.scrollTop=r.scrollHeight}
async function renderGraph(){const g=await j('/api/graph');const r=document.getElementById('right');r.innerHTML='';
  const order=['Q','C','I','O','R','E','D','A'];const names={Q:'Pytania',C:'Tezy',I:'Wnioskowania',O:'Zarzuty',R:'Odparcia',E:'Dowody',D:'Definicje',A:'Założenia'};
  for(const t of order){const ns=g.nodes.filter(n=>n.type===t);if(!ns.length)continue;const h=document.createElement('h3');h.textContent=`${names[t]} (${ns.length})`;r.appendChild(h);
    for(const n of ns){const d=document.createElement('div');d.className='node';const rel=n.data.answers?' → '+n.data.answers:n.data.claim?' → '+n.data.claim:n.data.target?' → '+n.data.target:n.data.objection?' → '+n.data.objection:n.data.parent?' ⊂ '+n.data.parent:n.data.used_by?' → '+n.data.used_by:'';
      d.innerHTML=`<b>${n.id}</b>${esc(rel)}<span class="st ${n.status}">${esc(n.status||'')}${n.type==='C'?' '+n.confidence.toFixed(2):''}</span> <span style="color:${col(n.author)}">${esc(n.author)}</span><div>${esc(n.text)}${n.data.url?` <a href="${esc(n.data.url)}" target="_blank" style="color:var(--acc)">źródło</a>`:''}</div>`;
      d.onclick=async()=>{const v=await j('/api/node?id='+n.id);alert(v.text)};r.appendChild(d)}}}
async function renderCodex(){const c=await j('/api/codex');document.getElementById('right').innerHTML='<pre>'+esc(c.text)+'</pre>'}
async function renderAgent(){const r=document.getElementById('right');if(!S.agent){r.innerHTML='<p style="color:var(--dim)">Wybierz członka po lewej.</p>';return}
  const a=await j('/api/agent?name='+S.agent);r.innerHTML=`<h3>${esc(a.name)} · tura ${a.tick}</h3><h3>streszczenie pamięci</h3><pre>${esc(a.summary||'(brak)')}</pre><h3>notatki</h3><pre>${esc(a.memory||'(brak)')}</pre><h3>dziennik</h3><pre>${esc((a.journal||[]).join('\n'))}</pre><h3>ostatnia tura (transkrypt)</h3><pre>${esc(a.last_transcript||'(brak)')}</pre>`}
async function tick(){try{
  const o=await j('/api/overview');if(!document.querySelector('.ch'))renderLeft(o);else document.getElementById('stats').textContent=`wiad. ${o.stats.messages} · DM ${o.stats.dms} · węzły ${o.stats.nodes} · zdarzenia ${o.stats.events} · ${o.model||''}`;
  if(S.channel==='__dm__'){const m=await j('/api/dms?after='+S.lastMsg);addMsgs(m.rows)}else{const m=await j('/api/messages?channel='+encodeURIComponent(S.channel)+'&after='+S.lastMsg);addMsgs(m.rows)}
  const ev=await j('/api/events?after='+S.lastEv+(S.agent?'&agent='+S.agent:'')+'&limit=400');if(ev.rows.length){S.events.push(...ev.rows);S.lastEv=ev.rows[ev.rows.length-1].id;S.events=S.events.slice(-600);renderEvents()}
  if(S.tab==='graph'&&!S._graphDrawn){renderGraph();S._graphDrawn=true;setTimeout(()=>S._graphDrawn=false,6000)}
}catch(e){console.error(e)}}
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('sel'));b.classList.add('sel');S.tab=b.dataset.t;S._graphDrawn=false;
  if(S.tab==='events')renderEvents();if(S.tab==='graph')renderGraph();if(S.tab==='codex')renderCodex();if(S.tab==='agent')renderAgent()});
document.getElementById('think').onchange=renderEvents;
setInterval(async()=>{await tick();if(S.tab==='agent')renderAgent()},2000);tick();
setInterval(async()=>{const o=await j('/api/overview');renderLeft(o)},8000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    db: DB
    graph: ArgumentGraph
    data_dir: Path
    model: str = ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _json(self, obj: Any, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                body = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/overview":
                agents = []
                for st in self.db.all_states():
                    journal = st.get("journal") or []
                    agents.append({"name": st["agent"], "tick": st.get("tick", 0), "last_seen": st.get("last_seen", ""), "last": journal[-1] if journal else ""})
                self._json({"channels": self.db.list_channels(), "agents": agents, "stats": self.db.stats(), "model": self.model})
            elif url.path == "/api/messages":
                ch = self.db.get_channel(q.get("channel", "ogolny"))
                if ch is None:
                    self._json({"rows": []})
                    return
                after = int(q.get("after", "0") or 0)
                rows = self.db.messages(ch["id"], after_id=after, limit=int(q.get("limit", "300") or 300)) if after else self.db.last_messages(ch["id"], 200)
                self._json({"rows": rows})
            elif url.path == "/api/dms":
                after = int(q.get("after", "0") or 0)
                rows = self.db.all_dms(after_id=after, limit=300)
                for r in rows:
                    r["author"] = r["sender"]
                    r["kind"] = "post"
                self._json({"rows": rows})
            elif url.path == "/api/events":
                after = int(q.get("after", "0") or 0)
                limit = int(q.get("limit", "300") or 300)
                agent = q.get("agent") or None
                if after == 0:
                    last = self.db.last_event_id()
                    after = max(0, last - 400)
                rows = self.db.events(after_id=after, limit=limit, agent=agent)
                for r in rows:
                    if r["kind"] == "tick_end":
                        r["payload"].pop("transcript", None)
                self._json({"rows": rows})
            elif url.path == "/api/graph":
                self._json({"nodes": self.db.nodes(), "edges": self.db.all_edges()})
            elif url.path == "/api/node":
                try:
                    text = self.graph.node_view(q.get("id", ""))
                except Exception as e:  # noqa: BLE001
                    text = f"błąd: {e}"
                self._json({"text": text})
            elif url.path == "/api/codex":
                self._json({"text": self.graph.codex()})
            elif url.path == "/api/agent":
                name = q.get("name", "")
                st = self.db.get_state(name)
                last_transcript = ""
                evs = self.db.events(after_id=max(0, self.db.last_event_id() - 3000), limit=3000, agent=name, kinds=["tick_end"])
                if evs:
                    tr = evs[-1]["payload"].get("transcript") or []
                    lines = []
                    for t in tr:
                        if t.get("role") == "thinking":
                            lines.append("💭 " + str(t.get("content", "")))
                        elif t.get("role") == "assistant":
                            lines.append("🗣 " + str(t.get("content", "")))
                        elif t.get("role") == "tool":
                            lines.append(f"⚙ {t.get('name')}({json.dumps(t.get('args'), ensure_ascii=False)})\n   ↳ {str(t.get('result', ''))[:800]}")
                    last_transcript = "\n".join(lines)
                self._json({"name": name, "tick": st.get("tick", 0), "summary": st.get("summary", ""), "memory": st.get("memory", ""), "journal": st.get("journal", []), "last_transcript": last_transcript})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def serve(db_path: Path, data_dir: Path, port: int = 8765, model: str = "") -> None:
    db = DB(db_path)
    Handler.db = db
    Handler.graph = ArgumentGraph(db)
    Handler.data_dir = data_dir
    Handler.model = model
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> int:
    from .settings import load_settings

    settings = load_settings()
    port = settings.viewer_port
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    print(f"Podgląd Agory: http://localhost:{port}  (baza: {settings.db_path})")
    serve(settings.db_path, settings.data_dir, port, model=settings.ollama_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
