"""Web 入口:在浏览器里看四个角色怎么接力(零依赖 http.server + SSE)。

页面结构很简单——左边是编排时间线,右边是边写边长的终稿。
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import mock, roles
from .llm import chat, default_config
from .orchestrator import Orchestrator, list_runs
from .utils import run_stamp

RUNS: dict[str, "Run"] = {}
RUNS_LOCK = threading.Lock()

HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>格物 GeWu · 多 Agent 深度研究助手</title>
<style>
:root{
  --bg:#0b0f14; --panel:#121822; --panel2:#161d29; --line:#232c3b;
  --fg:#dce3ee; --dim:#7d8aa0; --accent:#7aa2f7;
  --planner:#7aa2f7; --researcher:#9ece6a; --writer:#e0af68; --critic:#f7768e;
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);
  font:14px/1.7 "Segoe UI","Microsoft YaHei",system-ui,sans-serif;}
a{color:var(--accent)}
header{padding:18px 26px 12px;border-bottom:1px solid var(--line);
  display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
.brand{font-size:22px;font-weight:700;letter-spacing:.08em}
.brand em{font-style:normal;color:var(--dim);font-size:13px;letter-spacing:.14em;margin-left:8px}
.sub{color:var(--dim);font-size:12.5px}
.layout{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);gap:0;height:calc(100vh - 68px)}
.left{padding:18px 22px;overflow:auto;border-right:1px solid var(--line)}
.right{padding:18px 22px;overflow:auto;background:var(--panel)}
.roles{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.pill{font-size:12px;padding:3px 9px;border-radius:999px;border:1px solid var(--line);background:var(--panel2);color:var(--dim)}
textarea{width:100%;min-height:74px;resize:vertical;background:var(--panel2);color:var(--fg);
  border:1px solid var(--line);border-radius:10px;padding:11px 13px;font:inherit;outline:none}
textarea:focus{border-color:#33456a}
.controls{display:flex;gap:14px;align-items:center;margin-top:10px;flex-wrap:wrap}
label{color:var(--dim);font-size:12.5px;display:flex;gap:6px;align-items:center}
select,input[type=checkbox]{accent-color:var(--accent)}
select{background:var(--panel2);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:4px 8px;font:inherit}
button{background:linear-gradient(180deg,#2a4a86,#213c6e);color:#eaf1ff;border:1px solid #31538f;
  border-radius:9px;padding:8px 18px;font:inherit;font-weight:600;cursor:pointer}
button:disabled{opacity:.5;cursor:not-allowed}
.hint{color:var(--dim);font-size:12.5px;margin:10px 0 0;min-height:18px}
.timeline{margin-top:16px;display:flex;flex-direction:column;gap:10px}
.card{border:1px solid var(--line);border-left:3px solid var(--line);border-radius:10px;
  background:var(--panel2);padding:10px 13px;animation:fade .25s ease}
@keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.card .top{display:flex;gap:8px;align-items:center}
.card .name{font-weight:600}
.card .role{font-size:11.5px;color:var(--dim)}
.card .status{margin-left:auto;font-size:12px;color:var(--dim)}
.card .title{color:var(--fg);margin-top:3px}
.card .logs{margin-top:6px;color:var(--dim);font-size:12.5px}
.card .logs div{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card .foot{margin-top:6px;display:flex;gap:10px;font-size:12px;color:var(--dim)}
.card .foot b{color:var(--fg);font-weight:600}
.spin{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--accent);
  animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:.35}50%{opacity:1}}
.right h2{font-size:13px;color:var(--dim);font-weight:600;letter-spacing:.1em;margin:0 0 12px}
#report h1{font-size:22px;margin:6px 0 10px}
#report h2{font-size:16px;margin:26px 0 10px;color:#c8d4e8;border-left:3px solid #2b3b57;padding-left:10px}
#report h3{font-size:14.5px;margin:18px 0 8px}
#report p{margin:9px 0}
#report ul,#report ol{margin:8px 0 8px 20px}
#report li{margin:5px 0}
#report blockquote{border-left:3px solid #2b3b57;margin:10px 0;padding:4px 12px;color:var(--dim)}
#report table{border-collapse:collapse;width:100%;font-size:12.5px;margin:12px 0}
#report th,#report td{border:1px solid var(--line);padding:6px 8px;text-align:left}
#report th{background:#171f2c;color:var(--dim);font-weight:600}
#report code{background:#1b2431;padding:1px 5px;border-radius:4px;font-size:12.5px}
#report a{text-decoration:none;border-bottom:1px dashed #37507a}
.empty{color:var(--dim);font-size:13px;line-height:1.9}
.hist{margin-top:18px;border-top:1px solid var(--line);padding-top:12px}
.hist .row{display:flex;gap:10px;font-size:12.5px;color:var(--dim);cursor:pointer;padding:3px 0}
.hist .row:hover{color:var(--fg)}
</style>
</head>
<body>
<header>
  <div class="brand">格物 GeWu<em>MULTI-AGENT RESEARCH</em></div>
  <div class="sub">规划者拆题 → 研究者求证 → 写作者成文 → 审稿者挑刺 → 修订终稿</div>
</header>
<div class="layout">
  <div class="left">
    <div class="roles">
      <span class="pill" style="color:var(--planner)">🧭 规划者</span>
      <span class="pill" style="color:var(--researcher)">🔍 研究者 ×N</span>
      <span class="pill" style="color:var(--writer)">✍️ 写作者</span>
      <span class="pill" style="color:var(--critic)">🧐 审稿者</span>
      <span class="pill">每个子问题 = 一条独立任务</span>
    </div>
    <textarea id="topic" placeholder="输入一个课题,例如:大模型会对就业产生什么影响">大模型会对就业产生什么影响</textarea>
    <div class="controls">
      <label>子问题 <select id="depth">
        <option value="3">3</option><option value="4" selected>4</option>
        <option value="5">5</option><option value="6">6</option>
      </select></label>
      <label><input type="checkbox" id="search" checked> 联网检索(DuckDuckGo)</label>
      <button id="go">开始研究</button>
    </div>
    <div class="hint" id="hint"></div>
    <div class="timeline" id="timeline"></div>
    <div class="hist" id="hist"></div>
  </div>
  <div class="right">
    <h2>REPORT · 终稿</h2>
    <div id="report"><div class="empty">
      报告会在写作者第一次交稿后开始出现。<br>
      左边每个角色都是一条独立任务:拆题 → 若干条研究 → 成文 → 挑刺 → 修订。
    </div></div>
  </div>
</div>
<script>
const $ = s => document.querySelector(s);
const COLORS = {planner:'--planner', researcher:'--researcher', writer:'--writer', critic:'--critic'};
const NAMES = {planner:'🧭 规划者', researcher:'🔍 研究者', writer:'✍️ 写作者', critic:'🧐 审稿者'};
let source = null, current = null;

function esc(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function inline(s){
  return esc(s)
    .replace(/\*\*(.+?)\*\*/g,'<b>$1</b>')
    .replace(/`(.+?)`/g,'<code>$1</code>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
}

function md(text){
  const lines = text.split('\n');
  let out = [], list = null, table = null;
  const closeList = () => { if(list){ out.push(`</${list}>`); list = null; } };
  const closeTable = () => {
    if(!table) return;
    out.push('<table>');
    table.forEach((row,i)=>{
      if(i===1 && row.every(c=>/^:?-{2,}/.test(c.trim()))) return;
      out.push('<tr>' + row.map(c=>`<${i===0?'th':'td'}>${inline(c.trim())}</${i===0?'th':'td'}>`).join('') + '</tr>');
    });
    out.push('</table>');
    table = null;
  };
  for(const raw of lines){
    const line = raw.replace(/\s+$/,'');
    if(/^\s*\|.*\|\s*$/.test(line)){
      if(list){ closeList(); }
      const cells = line.trim().replace(/^\|/,'').replace(/\|$/,'').split('|');
      if(!table) table = [];
      table.push(cells);
      continue;
    } else { closeTable(); }
    if(!line.trim()){
      closeList(); out.push(''); continue;
    }
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if(h){ closeList(); const lv = h[1].length; out.push(`<h${lv}>${inline(h[2])}</h${lv}>`); continue; }
    if(/^>\s?/.test(line)){ closeList(); out.push(`<blockquote>${inline(line.replace(/^>\s?/,''))}</blockquote>`); continue; }
    if(/^(-{3,}|\*{3,})$/.test(line.trim())){ closeList(); out.push('<hr>'); continue; }
    const ul = line.match(/^\s*[-*+]\s+(.*)$/);
    if(ul){ if(list!=='ul'){ closeList(); out.push('<ul>'); list='ul'; } out.push(`<li>${inline(ul[1])}</li>`); continue; }
    const ol = line.match(/^\s*\d+\.\s+(.*)$/);
    if(ol){ if(list!=='ol'){ closeList(); out.push('<ol>'); list='ol'; } out.push(`<li>${inline(ol[1])}</li>`); continue; }
    out.push(`<p>${inline(line)}</p>`);
  }
  closeList(); closeTable();
  return out.join('\n');
}

function card(ev){
  let el = document.getElementById('t-'+ev.task.id);
  if(!el){
    el = document.createElement('div');
    el.className = 'card';
    el.id = 't-'+ev.task.id;
    el.style.borderLeftColor = getComputedStyle(document.documentElement).getPropertyValue(COLORS[ev.task.role]||'--line');
    el.innerHTML = `<div class="top"><span class="name">${NAMES[ev.task.role]||ev.task.role}</span>
      <span class="role">${ev.task.id}</span><span class="status"><span class="spin"></span></span></div>
      <div class="title">${esc(ev.task.title)}</div><div class="logs"></div><div class="foot"></div>`;
    $('#timeline').appendChild(el);
    el.scrollIntoView({block:'nearest', behavior:'smooth'});
  }
  return el;
}

function onEvent(ev){
  if(ev.type === 'run_start'){
    $('#timeline').innerHTML = ''; $('#report').innerHTML = '';
    $('#hint').textContent = `已启动 · ${ev.topic} · ${ev.depth} 个子问题 · ${ev.search?'联网检索':'纯模型知识'}`;
  }
  else if(ev.type === 'task_start'){
    const el = card(ev);
    $('#hint').textContent = `${NAMES[ev.task.role]||''} 正在处理 ${ev.task.id}…`;
  }
  else if(ev.type === 'task_log'){
    const el = document.getElementById('t-'+ev.task_id);
    if(el){ const logs = el.querySelector('.logs'); const d = document.createElement('div'); d.textContent='· '+ev.text; logs.appendChild(d); }
  }
  else if(ev.type === 'task_end'){
    const el = document.getElementById('t-'+ev.task.id);
    if(el){
      const st = el.querySelector('.status');
      st.textContent = ev.task.status === 'failed' ? '⚠️ 失败' : (ev.task.status === 'reused' ? '♻️ 复用' : '✓ 完成');
      st.style.color = ev.task.status === 'failed' ? 'var(--critic)' : 'var(--researcher)';
      el.querySelector('.foot').innerHTML = `<span>${esc(ev.summary||'')}</span><span>${ev.task.elapsed||0}s</span>` +
        (ev.artifact?`<a href="/api/artifact/${current}/${ev.artifact.split('/').pop()}" target="_blank">产物</a>`:'');
    }
  }
  else if(ev.type === 'log'){
    $('#hint').textContent = ev.text;
  }
  else if(ev.type === 'final'){
    $('#hint').textContent = `终稿已生成 · ${ev.chars} 字`;
    loadReport();
  }
  else if(ev.type === 'run_end'){
    $('#hint').textContent = `完成 · 耗时 ${ev.elapsed}s · ${ev.tasks} 条独立任务`;
    loadReport(); $('#go').disabled = false; $("#hint").textContent += ' · 可继续下一个课题';
    loadHistory();
  }
  else if(ev.type === 'error'){
    $('#hint').textContent = '✗ ' + ev.error; $('#go').disabled = false;
  }
}

async function loadReport(){
  if(!current) return;
  try{
    const r = await fetch('/api/report/'+current);
    if(!r.ok) return;
    const text = await r.text();
    $('#report').innerHTML = md(text);
  }catch(e){}
}

function connect(runId){
  current = runId;
  if(source) source.close();
  source = new EventSource('/api/events/'+runId);
  source.onmessage = e => { try{ onEvent(JSON.parse(e.data)); }catch(err){} };
  source.onerror = () => { if(source) source.close(); };
}

async function start(){
  const topic = $('#topic').value.trim();
  if(!topic){ $('#hint').textContent = '先写一个课题'; return; }
  $('#go').disabled = true;
  $('#hint').textContent = '正在拉起四个角色…';
  try{
    const r = await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({topic, depth: +$('#depth').value, search: $('#search').checked})});
    const data = await r.json();
    if(!r.ok || data.error){ $('#hint').textContent = '✗ ' + (data.error||'启动失败'); $('#go').disabled=false; return; }
    connect(data.run_id);
  }catch(e){ $('#hint').textContent = '✗ '+e.message; $('#go').disabled=false; }
}

async function loadHistory(){
  try{
    const r = await fetch('/api/runs'); const runs = await r.json();
    const box = $('#hist');
    if(!runs.length){ box.innerHTML = ''; return; }
    box.innerHTML = '<div class="row" style="color:var(--dim)">历史 run</div>' + runs.slice(0,6).map(x =>
      `<div class="row" data-id="${x.run_id}"><span>${x.run_id}</span><span>${x.chars||0} 字</span><span>${x.sources||0} 来源</span><span>${esc((x.topic||'').slice(0,28))}</span></div>`).join('');
    box.querySelectorAll('.row[data-id]').forEach(row=>row.onclick=()=>{ current = row.dataset.id; loadReport(); });
  }catch(e){}
}

$('#go').onclick = start;
loadHistory();
</script>
</body>
</html>
"""


class Run:
    """一次网页侧的运行:事件队列 + 终稿 + 生命周期。"""

    def __init__(self, run_id: str, topic: str, depth: int, search: bool, model: str):
        self.run_id = run_id
        self.topic = topic
        self.depth = depth
        self.search = search
        self.model = model
        self.events: list[dict] = []
        self.done = False
        self.report = ""
        self.report_path = ""
        self.run_dir = ""
        self.error = ""
        self.cond = threading.Condition()

    def push(self, ev: dict):
        with self.cond:
            self.events.append(ev)
            self.cond.notify_all()

    def finish(self):
        with self.cond:
            self.done = True
            self.cond.notify_all()


class Handler(BaseHTTPRequestHandler):
    server_version = "gewu/0.1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # 不刷屏
        pass

    # ---- 响应辅助
    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _text(self, text: str, code=200, ctype="text/plain; charset=utf-8"):
        self._send(code, text.encode("utf-8"), ctype)

    # ---- 路由
    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path in ("/", "/index.html"):
            return self._text(HTML, ctype="text/html; charset=utf-8")
        if path == "/api/runs":
            return self._json(list_runs())
        if path.startswith("/api/events/"):
            return self._events(path.rsplit("/", 1)[-1])
        if path.startswith("/api/report/"):
            rid = path.rsplit("/", 1)[-1]
            run = RUNS.get(rid)
            if run and run.report:
                return self._text(run.report, ctype="text/markdown; charset=utf-8")
            if run and run.report_path and os.path.isfile(run.report_path):
                with open(run.report_path, encoding="utf-8") as f:
                    return self._text(f.read(), ctype="text/markdown; charset=utf-8")
            return self._text("还没有报告", code=404)
        if path.startswith("/api/artifact/"):
            parts = path.split("/")  # ['', 'api', 'artifact', rid, name]
            if len(parts) == 5:
                run = RUNS.get(parts[3])
                if run and run.run_dir:
                    full = os.path.join(run.run_dir, "tasks", os.path.basename(parts[4]))
                    if os.path.isfile(full):
                        with open(full, encoding="utf-8") as f:
                            content = f.read()
                        ctype = "application/json; charset=utf-8" if full.endswith(".json") else "text/plain; charset=utf-8"
                        return self._text(content, ctype=ctype)
            return self._text("not found", code=404)
        return self._text("not found", code=404)

    def do_POST(self):
        path = urllib.parse.urlsplit(self.path).path
        if path != "/api/run":
            return self._text("not found", code=404)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except (ValueError, json.JSONDecodeError):
            return self._json({"error": "请求体不是合法 JSON"}, code=400)

        topic = str(payload.get("topic") or "").strip()
        if not topic:
            return self._json({"error": "课题不能为空"}, code=400)
        try:
            depth = max(1, min(6, int(payload.get("depth") or 4)))
        except (TypeError, ValueError):
            depth = 4
        search = bool(payload.get("search", True))

        rid = run_stamp()
        with RUNS_LOCK:
            base = rid
            n = 2
            while rid in RUNS:
                rid = f"{base}-{n}"
                n += 1
            run = Run(rid, topic, depth, search, self.server.model_label)  # type: ignore[attr-defined]
            RUNS[rid] = run

        threading.Thread(target=_run_in_thread, args=(run, self.server), daemon=True).start()
        return self._json({"run_id": rid, "model": run.model})

    def _events(self, rid: str):
        run = RUNS.get(rid)
        if not run:
            return self._text("run not found", code=404)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        idx = 0
        last_ping = time.time()
        try:
            while True:
                with run.cond:
                    if idx >= len(run.events) and not run.done:
                        run.cond.wait(timeout=5)
                batch = run.events[idx:]
                if batch:
                    for ev in batch:
                        chunk = f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                        self.wfile.write(chunk.encode("utf-8"))
                    idx += len(batch)
                    self.wfile.flush()
                    last_ping = time.time()
                elif run.done:
                    self.wfile.write(b"event: done\ndata: {}\n\n")
                    self.wfile.flush()
                    break
                elif time.time() - last_ping > 10:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    last_ping = time.time()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


def _run_in_thread(run: Run, server):
    try:
        orch = Orchestrator(
            run.topic,
            depth=run.depth,
            use_search=run.search,
            llm=server.llm_fn,
            on_event=run.push,
            run_id=run.run_id,
            model_label=run.model,
        )
        result = orch.run()
        run.report = result.report
        run.report_path = result.report_path
        run.run_dir = result.run_dir
    except Exception as e:  # noqa: BLE001 - 兜底,别让线程静默死掉
        run.error = str(e)
        run.push({"type": "error", "error": str(e), "run_id": run.run_id})
    finally:
        run.finish()


def serve(host: str = "127.0.0.1", port: int = 8000, mock_mode: bool = False):
    if mock_mode:
        llm_fn, label = mock.make_mock_llm(), "mock(离线演示)"
    else:
        cfg = default_config()
        if not cfg.get("api_key"):
            print("⚠️ 未配置 LLM_API_KEY:网页能打开,但点击研究会报错。可用 `python -m gewu web --mock`。")
        llm_fn = lambda messages, **kw: chat(messages, config=cfg, **kw)  # noqa: E731
        label = cfg.get("model") or "unknown"

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.llm_fn = llm_fn  # type: ignore[attr-defined]
    server.model_label = label  # type: ignore[attr-defined]

    print(f"格物 GeWu · Web 界面 → http://{host}:{port}  (模型:{label},Ctrl+C 结束)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()
