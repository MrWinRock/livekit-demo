"""Real-time teleprompter: local web UI that feeds text directly to the agent TTS.

Open http://localhost:7860 while the agent is running, type text in the
textarea, and press Ctrl+Enter (or the Speak button) — the agent speaks it
verbatim through its TTS pipeline, bypassing the LLM.

Port is configurable via the PROMPTER_PORT env var.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    from livekit.agents import AgentSession

logger = logging.getLogger("prompter")

_session: AgentSession | None = None
_started = False
_bg_tasks: set[asyncio.Task] = set()

_HTML = """\
<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Prompter</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0f0f1a;
    color: #e2e2f0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    height: 100vh;
    display: flex;
    flex-direction: column;
    padding: 20px;
    gap: 14px;
  }
  header { display: flex; align-items: center; gap: 10px; }
  h1 { font-size: 1.1rem; font-weight: 700; color: #a78bfa;
       letter-spacing: .05em; margin-left: auto; }
  #dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: #4b5563; flex-shrink: 0;
    transition: background .3s, box-shadow .3s;
  }
  #dot.on  { background: #34d399; box-shadow: 0 0 8px #34d39988; }
  #dot.off { background: #f87171; }
  #status { font-size: .78rem; color: #6b7280; }
  textarea {
    flex: 1;
    background: #1a1a2e;
    border: 1.5px solid #2e2e4e;
    border-radius: 10px;
    color: #e2e2f0;
    font-size: 1.05rem;
    line-height: 1.65;
    padding: 16px;
    resize: none;
    outline: none;
    font-family: inherit;
    transition: border-color .2s;
  }
  textarea:focus { border-color: #7c3aed; }
  .row { display: flex; gap: 10px; align-items: center; }
  #speakBtn {
    background: #7c3aed; border: none; border-radius: 8px;
    color: #fff; cursor: pointer; font-size: .95rem; font-weight: 600;
    padding: 10px 28px; transition: background .15s, opacity .15s;
  }
  #speakBtn:hover:not(:disabled) { background: #6d28d9; }
  #speakBtn:disabled { opacity: .35; cursor: default; }
  #clearBtn {
    background: transparent; border: 1.5px solid #2e2e4e;
    border-radius: 8px; color: #9ca3af; cursor: pointer;
    font-size: .85rem; padding: 10px 14px;
    transition: border-color .15s, color .15s;
  }
  #clearBtn:hover { border-color: #7c3aed; color: #e2e2f0; }
  .hint { font-size: .73rem; color: #4b5563; margin-left: auto; }
  #log {
    background: #1a1a2e; border: 1.5px solid #2e2e4e;
    border-radius: 10px; max-height: 150px; overflow-y: auto;
    padding: 10px 14px; display: flex; flex-direction: column; gap: 5px;
  }
  .entry {
    font-size: .82rem; padding: 6px 10px; border-radius: 6px;
    border-left: 3px solid #7c3aed; background: #252538;
    white-space: pre-wrap; word-break: break-word;
  }
  .ts { color: #6b7280; font-size: .72rem; margin-right: 8px; }
  .empty { color: #374151; font-size: .82rem; font-style: italic;
           text-align: center; padding: 8px 0; }
</style>
</head>
<body>
<header>
  <div id="dot"></div>
  <span id="status">Connecting...</span>
  <h1>Agent Prompter</h1>
</header>

<textarea id="inp"
  placeholder="Type text for the agent to speak...&#10;&#10;Supports multiple lines and paragraphs.&#10;Press Ctrl+Enter or click Speak."
  spellcheck="false"></textarea>

<div class="row">
  <button id="speakBtn" onclick="speak()">Speak</button>
  <button id="clearBtn" onclick="clearInp()">Clear</button>
  <span class="hint">Ctrl + Enter to speak</span>
</div>

<div id="log"><div class="empty">Sent messages appear here.</div></div>

<script>
const inp      = document.getElementById('inp');
const btn      = document.getElementById('speakBtn');
const dot      = document.getElementById('dot');
const statusEl = document.getElementById('status');
const log      = document.getElementById('log');
let alive = false;

function ts() {
  return new Date().toLocaleTimeString('th-TH',
    {hour:'2-digit', minute:'2-digit', second:'2-digit'});
}
function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function poll() {
  try {
    const r = await fetch('/health');
    const j = await r.json();
    alive = j.session_active;
  } catch { alive = false; }
  dot.className = alive ? 'on' : 'off';
  statusEl.textContent = alive ? 'Session active' : 'No active session';
  btn.disabled = !alive;
}

async function speak() {
  const text = inp.value.trim();
  if (!text || !alive) return;
  btn.disabled = true;
  try {
    const r = await fetch('/speak', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    if (r.ok) {
      pushLog(text);
      inp.value = '';
      inp.focus();
    } else {
      const j = await r.json().catch(() => ({}));
      alert('Error: ' + (j.error || r.status));
    }
  } catch(e) { alert('Network error: ' + e.message); }
  finally { if (alive) btn.disabled = false; }
}

function clearInp() { inp.value = ''; inp.focus(); }

function pushLog(text) {
  log.querySelectorAll('.empty').forEach(e => e.remove());
  const div = document.createElement('div');
  div.className = 'entry';
  const preview = text.length > 140 ? text.slice(0, 140) + '\\u2026' : text;
  div.innerHTML = '<span class="ts">' + ts() + '</span>' + esc(preview);
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

inp.addEventListener('keydown', e => {
  if (e.key === 'Enter' && e.ctrlKey) { e.preventDefault(); speak(); }
});

poll();
setInterval(poll, 3000);
inp.focus();
</script>
</body>
</html>
"""


def set_session(session: AgentSession) -> None:
    global _session
    _session = session


def clear_session() -> None:
    global _session
    _session = None


async def _say(text: str) -> None:
    if _session is None:
        return
    try:
        await _session.say(text)
    except Exception as exc:
        logger.warning("say() failed: %s", exc)


async def _index(request: web.Request) -> web.Response:
    return web.Response(text=_HTML, content_type="text/html")


async def _health(request: web.Request) -> web.Response:
    return web.json_response({"session_active": _session is not None})


async def _speak(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)
    text = (data.get("text") or "").strip()
    if not text:
        return web.json_response({"error": "text is empty"}, status=400)
    if _session is None:
        return web.json_response({"error": "no active session"}, status=503)
    t = asyncio.create_task(_say(text))
    t.add_done_callback(_bg_tasks.discard)
    _bg_tasks.add(t)
    return web.json_response({"ok": True})


async def start() -> None:
    """Start the prompter HTTP server. Safe to call multiple times — only starts once."""
    global _started
    if _started:
        return
    _started = True
    port = int(os.getenv("PROMPTER_PORT", "7860"))
    host = os.getenv("PROMPTER_HOST", "localhost")
    app = web.Application()
    app.router.add_get("/", _index)
    app.router.add_get("/health", _health)
    app.router.add_post("/speak", _speak)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info("Prompter UI → http://%s:%d", host, port)
