"""Dashboard and control panel.

Read-only: /health, /api/state and the page at /.
Control (token required): /api/control and its POST actions. Three checks protect them:
the X-Control-Token header must match `<data_dir>/control_token`; the request must be
addressed to this machine by a local name (blocks DNS rebinding); and a browser request
must originate from this page (blocks other websites open in the same browser).
"""

from __future__ import annotations

import hmac
import inspect
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
#bar{padding:14px 16px;font-size:20px;font-weight:600;background:#333;display:flex;gap:16px;flex-wrap:wrap;align-items:center}
#bar.running{background:#7a1616}#bar.stopped{background:#3a3a1c}
#bar small{font-weight:400;font-size:13px;opacity:.9}
.g{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}
.c,.panel{background:#1c1c1c;padding:10px 12px;border-radius:6px}.k{color:#888;font-size:12px}.v{font-size:18px}
.panel{margin-top:10px}.panel h4{margin:0 0 8px;font-size:13px;color:#aaa;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:6px 0}
button{font:inherit;padding:8px 14px;border-radius:5px;border:1px solid #444;background:#2a2a2a;color:#eee;cursor:pointer}
button:hover{background:#383838}button.go{background:#2d6a4f;border-color:#3d8a66;font-size:17px;padding:12px 28px;font-weight:600}
button.stop{background:#7a1616;border-color:#a32222;font-size:17px;padding:12px 28px;font-weight:600}
button.danger{background:#5a1a1a;border-color:#8a2a2a}button:disabled{opacity:.45;cursor:not-allowed}
input{font:inherit;padding:7px 9px;border-radius:5px;border:1px solid #444;background:#151515;color:#eee;min-width:0}
input.wide{flex:1 1 320px}label{color:#aaa;font-size:13px}
#msg{min-height:20px;margin:8px 0;font-size:14px}#msg.err{color:#ff7b7b}#msg.ok{color:#7bd88f}
#err{background:#4a1a1a;padding:10px 12px;border-radius:6px;margin-top:10px;display:none}
#why{font-size:14px;margin:8px 0 2px}.big{font-size:26px;font-weight:600}
pre{background:#1c1c1c;padding:10px;border-radius:6px;overflow:auto;margin:0}h3{margin:18px 0 6px;font-size:15px}
.hint{color:#888;font-size:12px;margin:4px 0}.warn{color:#e0b040}
</style>
<div id=bar><span id=mode>loading</span><small id=sub></small></div>
<main>
<div id=err></div>
<div id=locked class=panel><h4>Unlock controls</h4>
 <div class=row><input id=tok class=wide type=password placeholder="control token"><button id=tokbtn>Unlock</button></div>
 <div class=hint>On the server run: <code>sudo cat /opt/hltrader/app/data/control_token</code> and paste the result here. It is remembered in this browser.</div></div>
<div id=ctl style="display:none">
 <div id=msg></div>
 <div class=panel id=connectbox><h4>1. Connect your Hyperliquid account</h4>
  <div class=row><input id=key class=wide type=password autocomplete=off placeholder="API wallet private key"></div>
  <div class=row><input id=addr class=wide placeholder="wallet address 0x... (optional - found automatically from the key)"></div>
  <div class=row><button id=connect>Connect and fetch balance</button></div>
  <div class=hint>In Hyperliquid open More &rarr; API, create an API wallet, press Authorize, and paste the private key it shows. An API wallet can trade but cannot withdraw. The key is checked with Hyperliquid, stored on this server only, and never shown again.</div></div>
 <div class=panel id=acctbox style="display:none"><h4>Account</h4>
  <div class=row><span class=big id=bal>-</span><span id=acct class=hint></span></div>
  <div id=balnote class="hint warn"></div>
  <div class=row><button id=refresh>Refresh balance</button><button id=disconnect>Disconnect</button></div></div>
 <div class=panel id=runbox style="display:none"><h4>2. Trading</h4>
  <div class=row><button id=start class=go>Start trading</button><button id=stop class=stop>Stop trading</button>
   <button id=flat class=danger>Close position and stop</button></div>
  <div id=why></div>
  <div class=hint>Started: the engine sends real orders whenever it finds a trade it trusts. Stopped: it keeps watching and learning but sends nothing. Stopping does not close an open position; use "Close position and stop" for that.</div></div>
 <div class=panel id=scalpbox style="display:none"><h4>Scalper</h4>
  <div class=g id=sg></div>
  <div class=hint>It rests a buy below and a sell above the price and earns the gap when both fill. "Edge per fill" = spread captured plus where the price went 5 s later; it has to stay above the fee for the scalper to make money.</div></div>
 <div class=panel id=mktbox><h4>Market</h4>
  <div class=row><span id=mkt class=big></span><button id=scan>Scan all markets</button></div>
  <div id=mktlist></div>
  <div class=hint>A passive scalper only gets paid where the bid-ask spread is wider than the fee on both legs. "Margin" is spread minus those fees, before any losses to fast price moves. Wide spreads usually mean thin, jumpy markets.</div></div>
 <div class=panel id=prefbox style="display:none"><h4>Risk</h4>
  <div class=row><label>Max leverage <input id=ml type=number step=1 min=1 max=100 style="width:90px"></label>
   <label>Risk aversion <input id=ra type=number step=0.5 min=1 max=100 style="width:90px"></label>
   <button id=saveprefs>Save</button></div>
  <div class=hint>Max leverage caps how large a position can be relative to the balance. Risk aversion: 1 = most aggressive, 4 = default, higher = smaller positions. Saving restarts the engine (a few seconds).</div></div>
</div>
<h3>Status</h3><div class=g id=g></div>
<h3>Model arena</h3><pre id=a></pre><h3>Last decision</h3><pre id=d></pre>
<h3>Calibration (out-of-sample)</h3><pre id=m></pre><h3>Recent orders</h3><pre id=r></pre>
<h3>News (context only)</h3><pre id=n></pre>
</main>
<script>
const $=id=>document.getElementById(id),f=(x,n=2)=>x==null?'-':Number(x).toFixed(n);
let token=localStorage.getItem('hl_token')||'',ctl=null,busy=false;
function say(t,ok){const m=$('msg');m.textContent=t;m.className=ok?'ok':'err'}
function show(id,on){$(id).style.display=on?'':'none'}
async function api(path,body){
  const r=await fetch(path,{method:body?'POST':'GET',headers:{'X-Control-Token':token,'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});
  const j=await r.json().catch(()=>({}));
  if(r.status===401){token='';localStorage.removeItem('hl_token');ctl=null;throw new Error('wrong or missing control token')}
  if(!r.ok)throw new Error(j.error||('request failed ('+r.status+')'));return j}
async function act(path,body,working){if(busy)return;busy=true;say(working||'working...',true);
  try{const j=await api(path,body||{});say(j.message||'done',true)}catch(e){say(e.message,false)}
  busy=false;await loadCtl()}
function money(x){return x==null?'-':'$'+Number(x).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}
async function loadCtl(){
  if(!token){show('locked',1);show('ctl',0);return}
  try{ctl=await api('/api/control')}catch(e){if(!token){show('locked',1);show('ctl',0)}return}
  show('locked',0);show('ctl',1);
  const c=ctl.connected,a=ctl.account||{};
  show('connectbox',!c);show('acctbox',c);show('runbox',c);show('prefbox',1);
  if(c){$('bal').textContent=a.equity==null?'reading balance...':money(a.equity);
    $('acct').textContent=(ctl.account_address||'')+(a.abstraction?'  |  account mode: '+a.abstraction:'')+
      (a.position?'  |  position: '+a.position+' '+ctl.coin:'  |  no open position');
    let note='';
    if(a.equity!=null&&!a.unified&&a.perp_account_value===0&&a.spot_usdc>0)
      note='Your '+money(a.spot_usdc)+' USDC is in the Spot balance. In this account mode it has to be moved to Perps before it can be traded: in Hyperliquid use Portfolio > Transfer.';
    else if(a.equity===0)note='This account has no USDC available for trading. Deposit to Hyperliquid, then press Refresh balance.';
    else if(a.equity!=null&&a.equity<11)note='Hyperliquid\'s minimum order is $10. With a balance this small the engine can only take very few position sizes.';
    $('balnote').textContent=note;
    $('start').disabled=ctl.running||!a.equity;$('stop').disabled=!ctl.running;
    $('why').textContent=ctl.running?('Running. '+(ctl.reason||'')):'Stopped. No orders will be sent until you press Start trading.'}
  for(const [id,k] of [['ra','risk_aversion'],['ml','max_leverage']])if(document.activeElement!==$(id))$(id).value=ctl.effective[k]??'';
  $('mkt').textContent=ctl.coin;
}
function tiles(el,cards){el.replaceChildren(...cards.map(([k,v])=>{const c=document.createElement('div');c.className='c';
 const a=document.createElement('div');a.className='k';a.textContent=k;const b=document.createElement('div');b.className='v';b.textContent=v;c.append(a,b);return c}))}
$('scan').onclick=async()=>{say('scanning markets...',true);try{const j=await api('/api/control/scan',{});say(j.message,true);
  const box=$('mktlist');box.replaceChildren();
  for(const m of j.markets){const row=document.createElement('div');row.className='row';
    const txt=document.createElement('span');txt.style.cssText='font-family:monospace;white-space:pre';
    txt.textContent=m.coin.padEnd(10)+' spread '+f(m.spread_bps,2).padStart(6)+' bps   margin '+f(m.margin_bps,2).padStart(6)+' bps   volume $'+(m.volume_usd/1e6).toFixed(1).padStart(7)+'M   size at touch $'+Math.round(m.touch_depth_usd);
    const b=document.createElement('button');b.textContent=m.coin===j.current?'current':'Use';b.disabled=m.coin===j.current;
    b.onclick=()=>{if(confirm('Switch the engine to '+m.coin+'? Trading stops until you press Start again.'))act('/api/control/market',{coin:m.coin},'switching market...')};
    row.append(b,txt);box.append(row)}}catch(e){say(e.message,false)}};
$('tokbtn').onclick=async()=>{token=$('tok').value.trim();localStorage.setItem('hl_token',token);await loadCtl();if(!ctl)alert('That token was not accepted.')};
$('connect').onclick=async()=>{await act('/api/control/connect',{api_secret_key:$('key').value,account_address:$('addr').value},'checking the key with Hyperliquid...');$('key').value=''};
$('refresh').onclick=()=>act('/api/control/refresh',{},'reading balance...');
$('disconnect').onclick=()=>{if(confirm('Stop trading and remove the saved key from this server? An open position is NOT closed by this.'))act('/api/control/disconnect')};
$('start').onclick=()=>act('/api/control/start');
$('stop').onclick=()=>act('/api/control/stop');
$('flat').onclick=()=>{if(confirm('Cancel all resting orders, close the whole position at market, and stop trading?'))act('/api/control/flatten')};
$('saveprefs').onclick=()=>act('/api/control/preferences',{risk_aversion:$('ra').value,max_leverage:$('ml').value},'saving...');
async function go(){try{const s=await (await fetch('/api/state')).json();const e=s.engine,j=e.journal,l=e.last||{},h=s.health;
const status=!h.connected?'NOT CONNECTED':(h.running?'LIVE - TRADING':'LIVE - STOPPED');
$('bar').className=!h.connected?'':(h.running?'running':'stopped');
$('mode').textContent=e.coin+' | '+status;
$('sub').textContent=(h.connected?'balance '+f(j.equity)+'   position '+f(l.f_current,2)+'x   ':'watching the market, no account connected   ')+'faults: '+((h.faults||[]).join(', ')||'none');
const er=$('err');er.style.display=h.startup_error?'':'none';er.textContent=h.startup_error||'';
const cards=[['Balance',h.connected?f(j.equity):'-'],['Net return %',h.connected?f(j.net_return_pct,3):'-'],['Max drawdown %',h.connected?f(j.max_drawdown_pct,3):'-'],
['BTC price',f(e.mid,1)],['Position (x balance)',h.connected?f(l.f_current,2):'-'],['Expected edge bps',f(l.expected_edge_bps,3)],
['Trust in forecasts',f(l.extra&&l.extra.beta,3)],['Champion model',e.model.champion],['Fills',h.connected?j.fills:'-'],['Fees paid',h.connected?f(j.fees,4):'-'],
['Faults',(h.faults||[]).join(', ')||'none'],['Feed age s',f(h.feed_age_s,2)]];
tiles($('g'),cards);
const sc=e.scalper||{},q=sc.quotes||{},w=q.working||{},mv=j.move_after_fill_bps||{};
show('scalpbox',sc.enabled&&h.connected);
if(sc.enabled)tiles($('sg'),[['Our bid',w.bid??'-'],['Our ask',w.ask??'-'],['Market spread bps',f(q.half_spread_bps==null?null:q.half_spread_bps*2,2)],
 ['Inventory (x balance)',f(q.inventory_x,2)+' / '+f(q.inventory_limit_x,1)],['Fills',j.fills],['Spread captured bps',f(j.spread_capture_bps,2)],
 ['Price move 5s after fill bps',f(mv['5s'],2)],['Edge per fill bps',f((j.spread_capture_bps||0)+(mv['5s']||0),2)],
 ['Fees paid',f(j.fees,4)],['Time with quotes up %',f(sc.quote_uptime_pct,0)],['Orders placed / cancelled',(sc.orders_placed||0)+' / '+(sc.orders_cancelled||0)]]);
const pre=(id,o)=>$(id).textContent=JSON.stringify(o,null,1);
pre('a',e.model.arena.map(x=>`${x.champion?'CHAMPION ':'          '}${x.name.padEnd(16)} ic ${f(x.oos_ic,4)}  trusted beta ${f(x.max_trusted_beta,3)}  vs champion t ${f(x.vs_champion_t,2)}  resolved ${x.resolved}`));
pre('d',l);pre('m',e.model.calibration);pre('r',e.recent_decisions.slice(-5));pre('n',(s.news.latest||[]).map(x=>`${x.source} [${x.confirmations}] ${x.title}`));
}catch(err){$('mode').textContent='engine not responding (restarting?)';$('sub').textContent=''}}
go();loadCtl();setInterval(go,2000);setInterval(()=>{if(!busy)loadCtl()},4000);</script>"""

Handler = Callable[[dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]]


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
                if inspect.isawaitable(out):
                    out = await out
            except ControlError as e:
                return web.json_response({"error": str(e)}, status=400)
            return web.Response(text=json.dumps(out, default=float), content_type="application/json")

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
