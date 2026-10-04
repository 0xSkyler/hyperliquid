"""Read-only dashboard: /health, /api/state, and a small auto-refreshing page at /."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from aiohttp import web

_PAGE = """<!doctype html><meta charset=utf-8><title>HL trader</title>
<style>body{font:14px system-ui;margin:16px;background:#111;color:#ddd}
.g{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}
.c{background:#1c1c1c;padding:10px;border-radius:6px}.k{color:#888;font-size:12px}
.v{font-size:18px}pre{background:#1c1c1c;padding:10px;border-radius:6px;overflow:auto}
h3{margin:18px 0 6px}</style>
<h2 id=t>loading</h2><div class=g id=g></div>
<h3>Last decision</h3><pre id=d></pre><h3>Calibration (out-of-sample)</h3><pre id=m></pre>
<h3>Recent orders</h3><pre id=r></pre><h3>News (context only)</h3><pre id=n></pre>
<script>
const f=(x,n=2)=>x==null?'-':Number(x).toFixed(n);
async function go(){try{const s=await (await fetch('/api/state')).json();const e=s.engine,j=e.journal,l=e.last||{};
document.getElementById('t').textContent=`${e.coin} | ${e.mode.toUpperCase()} | ${e.state}`;
const cards=[['Equity',f(j.equity)],['Net return %',f(j.net_return_pct,3)],['Max drawdown %',f(j.max_drawdown_pct,3)],
['Mid',f(e.mid,1)],['Exposure (x equity)',f(l.f_current,2)],['Edge bps (calibrated)',f(l.expected_edge_bps,3)],
['Trusted beta',f(l.extra&&l.extra.beta,3)],['OOS IC',f(e.model.oos_ic,4)],['Fills',j.fills],['Fees',f(j.fees,4)],
['Faults',(l.faults||[]).join(', ')||'none'],['Feed age s',f(s.health.feed_age_s,2)]];
document.getElementById('g').innerHTML=cards.map(([k,v])=>`<div class=c><div class=k>${k}</div><div class=v>${v}</div></div>`).join('');
const pre=(id,o)=>document.getElementById(id).textContent=JSON.stringify(o,null,1);
pre('d',l);pre('m',e.model.calibration);pre('r',e.recent_decisions.slice(-5));pre('n',(s.news.latest||[]).map(x=>`${x.source} [${x.confirmations}] ${x.title}`));
}catch(err){document.getElementById('t').textContent='disconnected: '+err}}
go();setInterval(go,2000);</script>"""


def make_app(state_fn: Callable[[], dict[str, Any]]) -> web.Application:
    async def index(_: web.Request) -> web.Response:
        return web.Response(text=_PAGE, content_type="text/html")

    async def state(_: web.Request) -> web.Response:
        return web.Response(text=json.dumps(state_fn(), default=float), content_type="application/json")

    async def health(_: web.Request) -> web.Response:
        h = state_fn()["health"]
        return web.json_response(h, status=200 if h["ok"] else 503)

    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/api/state", state), web.get("/health", health)])
    return app


async def serve(state_fn: Callable[[], dict[str, Any]], host: str, port: int) -> web.AppRunner:
    runner = web.AppRunner(make_app(state_fn), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    return runner
