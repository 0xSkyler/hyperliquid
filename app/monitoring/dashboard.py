"""Dashboard and control panel.

Read-only: /health, /api/state and the page at /.
Control (token required): /api/control and its POST actions. Three checks protect them:
the X-Control-Token header must match `<data_dir>/control_token`; the request must be
addressed to this machine by a local name (blocks DNS rebinding); and a browser request
must originate from this page (blocks other websites open in the same browser).
"""

from __future__ import annotations

import hmac
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from aiohttp import web

from app.control import ControlError

_LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]")

_PAGE = r"""<!doctype html><meta charset=utf-8><title>HL trader</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
body{font:14px system-ui,sans-serif;margin:0;background:#111;color:#ddd}
main{max-width:1100px;margin:0 auto;padding:16px}
#bar{padding:14px 16px;font-size:20px;font-weight:600;background:#1c3a2a;display:flex;gap:16px;flex-wrap:wrap;align-items:center}
#bar.live{background:#7a1616}#bar.testnet{background:#5a4a12}#bar.paused{outline:3px solid #e0a020;outline-offset:-3px}
#bar small{font-weight:400;font-size:13px;opacity:.85}
.g{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}
.c,.panel{background:#1c1c1c;padding:10px 12px;border-radius:6px}.k{color:#888;font-size:12px}.v{font-size:18px}
.panel{margin-top:10px}.panel h4{margin:0 0 8px;font-size:13px;color:#aaa;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:6px 0}
button{font:inherit;padding:8px 14px;border-radius:5px;border:1px solid #444;background:#2a2a2a;color:#eee;cursor:pointer}
button:hover{background:#383838}button.on{background:#2d6a4f;border-color:#3d8a66}
button.danger{background:#7a1616;border-color:#a32222}button.warn{background:#6b5310;border-color:#927218}
button:disabled{opacity:.45;cursor:not-allowed}
input{font:inherit;padding:7px 9px;border-radius:5px;border:1px solid #444;background:#151515;color:#eee;min-width:0}
input.wide{flex:1 1 320px}label{color:#aaa;font-size:13px}
#msg{min-height:20px;margin:8px 0;font-size:13px}#msg.err{color:#ff7b7b}#msg.ok{color:#7bd88f}
#err{background:#4a1a1a;padding:10px 12px;border-radius:6px;margin-top:10px;display:none}
pre{background:#1c1c1c;padding:10px;border-radius:6px;overflow:auto;margin:0}h3{margin:18px 0 6px;font-size:15px}
.hint{color:#888;font-size:12px;margin:4px 0}
</style>
<div id=bar><span id=mode>loading</span><small id=sub></small></div>
<main>
<div id=err></div>
<h3>Control</h3>
<div id=locked class=panel><h4>Unlock controls</h4>
 <div class=row><input id=tok class=wide type=password placeholder="control token"><button id=tokbtn>Unlock</button></div>
 <div class=hint>On the server run: <code>sudo cat /opt/hltrader/app/data/control_token</code> and paste the result here. It is remembered in this browser.</div></div>
<div id=ctl style="display:none">
 <div id=msg></div>
 <div class=panel><h4>Trading mode</h4>
  <div class=row>
   <button data-mode=paper>Paper</button><button data-mode=shadow>Shadow</button>
   <button data-mode=testnet class=warn>Testnet</button><button data-mode=live class=danger>LIVE (real money)</button></div>
  <div class=hint>Paper: real prices, simulated money. Shadow: decides but sends nothing. Testnet: real orders with test money. Live: real orders with real money. Changing mode restarts the engine (a few seconds).</div></div>
 <div class=panel><h4>Trading switch</h4>
  <div class=row><button id=pause></button><button id=flat class=danger>Close position and pause</button></div>
  <div class=hint>Paused: the engine keeps watching and learning but sends no orders. "Close position" cancels resting orders, closes the whole position at market and pauses.</div></div>
 <div class=panel><h4>Hyperliquid account (needed for Testnet and Live)</h4>
  <div class=row><input id=addr class=wide placeholder="account address 0x... (your main wallet, public)"></div>
  <div class=row><input id=key class=wide type=password autocomplete=off placeholder="API wallet private key (never your main wallet key)"></div>
  <div class=row><button id=savecred>Save credentials</button><button id=clearcred>Remove credentials</button><span id=credstate class=hint></span></div>
  <div class=hint>Create an API wallet in Hyperliquid under More &rarr; API. It can trade but cannot withdraw. The key is stored on this server only and is never shown again.</div></div>
 <div class=panel><h4>Risk preferences</h4>
  <div class=row><label>Risk aversion <input id=ra type=number step=0.5 min=1 max=100 style="width:90px"></label>
   <label>Max leverage <input id=ml type=number step=1 min=1 max=100 style="width:90px"></label>
   <label>Paper balance $ <input id=pe type=number step=100 min=10 style="width:120px"></label>
   <button id=saveprefs>Save and restart</button></div>
  <div class=hint>Risk aversion: 1 = most aggressive, 4 = default, higher = smaller positions. Max leverage caps exposure below the venue's limit.</div></div>
</div>
<h3>Status</h3><div class=g id=g></div>
<h3>Model arena</h3><pre id=a></pre><h3>Last decision</h3><pre id=d></pre>
<h3>Calibration (out-of-sample)</h3><pre id=m></pre><h3>Recent orders</h3><pre id=r></pre>
<h3>News (context only)</h3><pre id=n></pre>
</main>
<script>
const $=id=>document.getElementById(id),f=(x,n=2)=>x==null?'-':Number(x).toFixed(n);
let token=localStorage.getItem('hl_token')||'',ctl=null;
function say(t,ok){const m=$('msg');m.textContent=t;m.className=ok?'ok':'err'}
async function api(path,body){
  const r=await fetch(path,{method:body?'POST':'GET',headers:{'X-Control-Token':token,'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});
  const j=await r.json().catch(()=>({}));
  if(r.status===401){token='';localStorage.removeItem('hl_token');ctl=null;throw new Error('wrong or missing control token')}
  if(!r.ok)throw new Error(j.error||('request failed ('+r.status+')'));return j}
async function act(path,body,done){try{const j=await api(path,body||{});say(j.message||done||'done',true);await loadCtl()}catch(e){say(e.message,false)}}
async function loadCtl(){
  if(!token){$('locked').style.display='';$('ctl').style.display='none';return}
  try{ctl=await api('/api/control')}catch(e){$('locked').style.display='';$('ctl').style.display='none';return}
  $('locked').style.display='none';$('ctl').style.display='';
  document.querySelectorAll('[data-mode]').forEach(b=>{b.classList.toggle('on',b.dataset.mode===ctl.mode);
    b.disabled=(b.dataset.mode==='testnet'||b.dataset.mode==='live')&&!ctl.has_key});
  $('pause').textContent=ctl.paused?'Resume trading':'Pause trading';$('pause').className=ctl.paused?'on':'warn';
  $('credstate').textContent=ctl.has_key?'key saved for '+(ctl.account_address||'(address from environment)'):'no key saved';
  if(document.activeElement!==$('addr'))$('addr').value=ctl.account_address||'';
  for(const [id,k] of [['ra','risk_aversion'],['ml','max_leverage'],['pe','paper_equity']])
    if(document.activeElement!==$(id))$(id).value=ctl.effective[k]??'';
}
$('tokbtn').onclick=async()=>{token=$('tok').value.trim();localStorage.setItem('hl_token',token);await loadCtl();if(!ctl)alert('That token was not accepted.')};
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{
  const mode=b.dataset.mode;let confirmText='';
  if(mode==='live'){confirmText=prompt('LIVE mode places real orders with real money.\n\nTo confirm, type exactly:\n'+ctl.live_confirm_phrase)||'';if(!confirmText)return}
  else if(mode==='testnet'&&!confirm('Switch to TESTNET? It will place real orders with test money.'))return;
  act('/api/control/mode',{mode,confirm:confirmText},'switching to '+mode+'; the engine is restarting')});
$('pause').onclick=()=>act('/api/control/pause',{paused:!ctl.paused});
$('flat').onclick=()=>{if(confirm('Cancel all resting orders, close the whole position at market, and pause trading?'))act('/api/control/flatten')};
$('savecred').onclick=async()=>{await act('/api/control/credentials',{account_address:$('addr').value,api_secret_key:$('key').value},'credentials saved');$('key').value=''};
$('clearcred').onclick=()=>{if(confirm('Remove the saved key and address? Testnet/Live will switch back to paper.'))act('/api/control/credentials/clear')};
$('saveprefs').onclick=()=>act('/api/control/preferences',{risk_aversion:$('ra').value,max_leverage:$('ml').value,paper_equity:$('pe').value},'saved; the engine is restarting');
async function go(){try{const s=await (await fetch('/api/state')).json();const e=s.engine,j=e.journal,l=e.last||{},h=s.health;
const bar=$('bar');bar.className=e.mode+(h.paused?' paused':'');
$('mode').textContent=e.coin+' | '+e.mode.toUpperCase()+(e.mode==='live'?' - REAL MONEY':'')+' | '+(h.paused?'PAUSED':e.state);
$('sub').textContent='equity '+f(j.equity)+'   position '+f(l.f_current,2)+'x   faults: '+((h.faults||[]).join(', ')||'none');
const er=$('err');er.style.display=h.startup_error?'':'none';er.textContent=h.startup_error||'';
const cards=[['Equity',f(j.equity)],['Net return %',f(j.net_return_pct,3)],['Max drawdown %',f(j.max_drawdown_pct,3)],
['Mid',f(e.mid,1)],['Exposure (x equity)',f(l.f_current,2)],['Edge bps (calibrated)',f(l.expected_edge_bps,3)],
['Trusted beta',f(l.extra&&l.extra.beta,3)],['Champion',e.model.champion],['Fills',j.fills],['Fees',f(j.fees,4)],
['Faults',(h.faults||[]).join(', ')||'none'],['Feed age s',f(h.feed_age_s,2)]];
const g=$('g');g.replaceChildren(...cards.map(([k,v])=>{const c=document.createElement('div');c.className='c';
 const a=document.createElement('div');a.className='k';a.textContent=k;const b=document.createElement('div');b.className='v';b.textContent=v;c.append(a,b);return c}));
const pre=(id,o)=>$(id).textContent=JSON.stringify(o,null,1);
pre('a',e.model.arena.map(x=>`${x.champion?'CHAMPION ':'          '}${x.name.padEnd(16)} ic ${f(x.oos_ic,4)}  trusted beta ${f(x.max_trusted_beta,3)}  vs champion t ${f(x.vs_champion_t,2)}  resolved ${x.resolved}`));
pre('d',l);pre('m',e.model.calibration);pre('r',e.recent_decisions.slice(-5));pre('n',(s.news.latest||[]).map(x=>`${x.source} [${x.confirmations}] ${x.title}`));
}catch(err){$('mode').textContent='engine not responding (restarting?)';$('sub').textContent=''}}
go();loadCtl();setInterval(go,2000);setInterval(loadCtl,5000);</script>"""

Handler = Callable[[dict[str, Any]], dict[str, Any]]


def make_app(
    state_fn: Callable[[], dict[str, Any]], token: str = "", control: Mapping[str, Handler] | None = None, port: int = 8787
) -> web.Application:
    control = control or {}

    def authorised(req: web.Request) -> web.Response | None:
        host = (req.headers.get("Host") or "").rsplit(":", 1)[0]
        origin = req.headers.get("Origin")
        local_origins = {f"http://{h}:{port}" for h in _LOCAL_HOSTS}
        if host not in _LOCAL_HOSTS or (origin is not None and origin not in local_origins):
            return web.json_response({"error": "controls are only available from this machine"}, status=403)
        if not token or not hmac.compare_digest(req.headers.get("X-Control-Token", ""), token):
            return web.json_response({"error": "wrong or missing control token"}, status=401)
        return None

    async def index(_: web.Request) -> web.Response:
        return web.Response(text=_PAGE, content_type="text/html", headers={"Cache-Control": "no-store"})

    async def state(_: web.Request) -> web.Response:
        return web.Response(text=json.dumps(state_fn(), default=float), content_type="application/json")

    async def health(_: web.Request) -> web.Response:
        h = state_fn()["health"]
        return web.json_response(h, status=200 if h["ok"] else 503)

    def action(name: str) -> Callable[[web.Request], Awaitable[web.Response]]:
        async def handle(req: web.Request) -> web.Response:
            if (denied := authorised(req)) is not None:
                return denied
            body: dict[str, Any] = {}
            if req.method == "POST":
                try:
                    parsed = await req.json()
                except ValueError:
                    return web.json_response({"error": "invalid request"}, status=400)
                body = parsed if isinstance(parsed, dict) else {}
            try:
                out = control[name](body)
            except ControlError as e:
                return web.json_response({"error": str(e)}, status=400)
            return web.json_response(out)

        return handle

    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/api/state", state), web.get("/health", health)])
    for name in control:
        route = "/api/control" if name == "get" else "/api/control/" + name.replace("_", "/")
        app.add_routes([web.get(route, action(name)) if name == "get" else web.post(route, action(name))])
    return app


async def serve(
    state_fn: Callable[[], dict[str, Any]], host: str, port: int, token: str = "", control: Mapping[str, Handler] | None = None
) -> web.AppRunner:
    runner = web.AppRunner(make_app(state_fn, token, control, port), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    return runner
