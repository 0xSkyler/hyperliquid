"""Hyperliquid adapter: public market data (REST + WebSocket) and, for TESTNET/LIVE, order placement.

Market data needs no credentials. Order placement goes through the official
`hyperliquid-python-sdk`, imported lazily so paper mode has no signing dependency.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import aiohttp
import numpy as np

from app.exchange.base import AccountState, AssetCtx, AssetMeta, Book, Fill, OrderIntent, Trade

log = logging.getLogger(__name__)


def parse_book(d: dict[str, Any], recv_ts: float) -> Book:
    def side(levels: list[dict[str, str]]) -> np.ndarray:
        if not levels:
            return np.empty((0, 2))
        return np.array([(float(x["px"]), float(x["sz"])) for x in levels], dtype=float)

    return Book(d["coin"], recv_ts, d["time"] / 1000.0, side(d["levels"][0]), side(d["levels"][1]))


def parse_trades(rows: list[dict[str, Any]], recv_ts: float) -> list[Trade]:
    return [Trade(recv_ts, float(r["px"]), float(r["sz"]), r["side"] == "B") for r in rows]


def parse_ctx(d: dict[str, Any], recv_ts: float) -> AssetCtx:
    c = d["ctx"]
    return AssetCtx(
        recv_ts,
        float(c["funding"]),
        float(c.get("premium") or 0.0),
        float(c["oraclePx"]),
        float(c["markPx"]),
        float(c["openInterest"]),
    )


def parse_event(channel: str, data: Any, recv_ts: float) -> tuple[str, Any] | None:
    if channel == "l2Book":
        return "book", parse_book(data, recv_ts)
    if channel == "bbo":
        b, a = data["bbo"]
        if b is None or a is None:
            return None
        return "bbo", (float(b["px"]), float(b["sz"]), float(a["px"]), float(a["sz"]), data["time"] / 1000.0)
    if channel == "trades":
        return "trades", parse_trades(data, recv_ts)
    if channel == "chart":  # also written by our recorder
        return "chart", (str(data.get("tf", "5m")), float(data["score"]))
    if channel == "news":  # written by our own recorder so replays see the same news feature
        return "news", float(data["score"])
    if channel == "activeAssetCtx":
        return "ctx", parse_ctx(data, recv_ts)
    return None


UNIFIED_MODES = ("unifiedAccount", "portfolioMargin")


def parse_account(abstraction: str, perp: dict[str, Any], spot: dict[str, Any], coin: str) -> dict[str, Any]:
    """One view of an account from its abstraction mode, perp state and spot state.

    Hyperliquid's default for new accounts is the *unified* mode, where collateral lives in the
    spot USDC balance and the perp account value is "not meaningful" (it reads 0). Reading only
    the perp state would report an empty account. In unified modes equity is taken as spot USDC
    plus the unrealised PnL of open perp positions; in the classic modes it is the perp account value.
    """
    perp_value = float(perp.get("marginSummary", {}).get("accountValue", 0.0))
    usdc: dict[str, Any] = next((b for b in spot.get("balances", []) if b.get("coin") == "USDC"), {})
    spot_usdc, spot_hold = float(usdc.get("total", 0.0)), float(usdc.get("hold", 0.0))
    pos, entry, upnl = 0.0, 0.0, 0.0
    for ap in perp.get("assetPositions", []):
        q = ap["position"]
        upnl += float(q.get("unrealizedPnl") or 0.0)
        if q["coin"] == coin:
            pos, entry = float(q["szi"]), float(q.get("entryPx") or 0.0)
    unified = abstraction in UNIFIED_MODES
    return {
        "abstraction": abstraction, "unified": unified,
        "equity": spot_usdc + upnl if unified else perp_value,
        "perp_account_value": perp_value, "spot_usdc": spot_usdc, "spot_usdc_hold": spot_hold,
        "unrealized_pnl": upnl, "position": pos, "entry_px": entry,
    }  # fmt: skip


def agent_address(secret_key: str) -> str:
    """The address an API wallet key signs as."""
    import eth_account

    return str(eth_account.Account.from_key(secret_key).address)


class HyperliquidData:
    def __init__(self, api_url: str) -> None:
        self.api_url = api_url
        self.ws_url = api_url.replace("http", "ws", 1) + "/ws"  # https -> wss, http -> ws
        self.reconnects = 0

    async def info(self, payload: dict[str, Any]) -> Any:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(self.api_url + "/info", json=payload) as r:
                r.raise_for_status()
                return await r.json()

    async def meta(self, coin: str) -> AssetMeta:
        m = await self.info({"type": "meta"})
        for a in m["universe"]:
            if a["name"] == coin and not a.get("isDelisted"):
                return AssetMeta(coin, int(a["szDecimals"]), float(a["maxLeverage"]))
        raise ValueError(f"{coin} is not a listed Hyperliquid perp")

    async def candles(self, coin: str, interval: str = "5m", step_s: int = 300, n: int = 1100) -> np.ndarray:
        """Last `n` *closed* candles as (n, 6): ts (s), open, high, low, close, volume."""
        now_ms = int(time.time() * 1000)
        req = {"coin": coin, "interval": interval, "startTime": now_ms - n * step_s * 1000, "endTime": now_ms}
        rows = await self.info({"type": "candleSnapshot", "req": req})
        closed = [r for r in rows if r["T"] < now_ms]
        return np.array([[r["t"] / 1000, float(r["o"]), float(r["h"]), float(r["l"]), float(r["c"]), float(r["v"])]
                         for r in closed])  # fmt: skip

    async def role(self, address: str) -> dict[str, Any]:
        """{"role": "user" | "agent" | "vault" | "subAccount" | "missing", "data": {...}}"""
        r = await self.info({"type": "userRole", "user": address})
        return r if isinstance(r, dict) else {"role": "missing"}

    async def account(self, address: str, coin: str) -> dict[str, Any]:
        abstraction = await self.info({"type": "userAbstraction", "user": address})
        perp = await self.info({"type": "clearinghouseState", "user": address})
        spot = await self.info({"type": "spotClearinghouseState", "user": address})
        return parse_account(str(abstraction), perp, spot, coin) | {"address": address}

    async def fees(self, address: str) -> tuple[float, float] | None:
        """(taker, maker) perp fee rates for this account, or None if unavailable."""
        if not address:
            return None
        f = await self.info({"type": "userFees", "user": address})
        return float(f["userCrossRate"]), float(f["userAddRate"])

    async def raw_stream(self, coin: str) -> AsyncIterator[tuple[float, str, Any]]:
        """Yield (recv_ts, channel, data); reconnects forever with backoff."""
        subs = [
            {"type": "l2Book", "coin": coin},  # 20-level snapshot, only every few seconds
            {"type": "bbo", "coin": coin},  # top of book on every change
            {"type": "trades", "coin": coin},
            {"type": "activeAssetCtx", "coin": coin},
        ]
        backoff = 0.5
        while True:
            try:
                async with aiohttp.ClientSession() as s, s.ws_connect(self.ws_url, heartbeat=None) as ws:
                    for sub in subs:
                        await ws.send_json({"method": "subscribe", "subscription": sub})
                    backoff = 0.5
                    last_ping = time.time()
                    while True:
                        msg = await ws.receive(timeout=20)
                        now = time.time()
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            raise ConnectionError(f"websocket closed: {msg.type}")
                        m = msg.json()
                        ch = m.get("channel")
                        if ch in ("l2Book", "bbo", "trades", "activeAssetCtx"):
                            yield now, ch, m["data"]
                        if now - last_ping > 30:
                            await ws.send_json({"method": "ping"})
                            last_ping = now
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - any transport failure means reconnect
                self.reconnects += 1
                log.warning("market stream error (%s); reconnecting in %.1fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 15.0)


class HyperliquidLive:
    """Real order placement (TESTNET or LIVE). Implements the Venue protocol.

    All network calls run on a single worker thread so the engine never blocks. The engine
    reads a cached account snapshot; if that snapshot is old the safety kernel halts trading.

    NOTE: this class has not been exercised against a funded account by the author of this
    repository. Validate on TESTNET before LIVE (docs/LIVE.md).
    """

    def __init__(self, api_url: str, address: str, secret_key: str, meta: AssetMeta) -> None:
        import eth_account
        from hyperliquid.exchange import Exchange
        from hyperliquid.info import Info

        wallet = eth_account.Account.from_key(secret_key)
        self._ex = Exchange(wallet, api_url, account_address=address, timeout=10)
        self._info = Info(api_url, skip_ws=True, timeout=10)
        self.snapshot: dict[str, Any] = {}  # last parsed account view, for the control panel
        self._abstraction = "default"
        self._abstraction_ts = 0.0
        self._heavy_ts = 0.0  # open orders and fills cost 10x the rate-limit weight of account state
        self._last_pos: float | None = None
        self._last_submit = 0.0
        self.heavy_interval_s = 10.0  # the scalper lowers this: it needs a fresher view of its resting orders
        self._working: tuple[str, ...] = ()
        self._working_ts = 0.0
        self._address = address
        self._meta = meta
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hl-orders")
        self._acct = AccountState(0.0, known=False)
        self._inflight: dict[int, float] = {}
        self._seq = 0
        self._fills: list[Fill] = []
        self._seen_fills: set[str] = set()
        self._fills_since_ms = int(time.time() * 1000)
        self._resting: dict[int, float] = {}  # oid -> expiry ts
        self.errors = 0

    # --- Venue protocol -------------------------------------------------
    def submit(self, intent: OrderIntent, now: float) -> None:
        self._seq += 1
        seq = self._seq
        self._inflight[seq] = now
        self._last_submit = time.time()
        self._pool.submit(self._place, seq, intent, now)

    def cancel_all(self, now: float) -> None:
        self._pool.submit(self._cancel_all)

    def cancel(self, client_id: str, now: float) -> None:
        self._last_submit = time.time()
        self._pool.submit(self._cancel_one, client_id)

    def account(self, now: float) -> AccountState:
        a = self._acct
        oldest = min(self._inflight.values(), default=0.0)
        return AccountState(
            a.ts, a.known, a.equity, a.position, a.entry_px, a.open_orders, len(self._inflight), oldest,
            self._working, self._working_ts,
        )  # fmt: skip

    def drain_fills(self) -> list[Fill]:
        out, self._fills = self._fills, []
        return out

    def on_book(self, book: Book) -> None: ...
    def on_trades(self, trades: list[Trade]) -> None: ...
    def on_ctx(self, ctx: AssetCtx) -> None: ...

    # --- worker-thread side --------------------------------------------
    def refresh(self) -> None:
        """Reconcile from the exchange. Call via asyncio.to_thread every couple of seconds.

        Account state is cheap (rate-limit weight 2) and read every call. Open orders and fills
        cost weight 20 each, so they are read every 10 seconds, or immediately when the position
        changed or an order was sent in the last few seconds.
        """
        try:
            now = time.time()
            if now - self._abstraction_ts > 300:
                self._abstraction = str(self._info.query_user_abstraction_state(self._address))
                self._abstraction_ts = now
            perp = self._info.user_state(self._address)
            spot = self._info.spot_user_state(self._address)
            snap = parse_account(self._abstraction, perp, spot, self._meta.coin)
            pos = snap["position"]
            heavy = now - self._heavy_ts > self.heavy_interval_s or pos != self._last_pos or now - self._last_submit < 5
            open_n = self._acct.open_orders
            if heavy:
                self._heavy_ts = now
                oo = [o for o in self._info.open_orders(self._address) if o["coin"] == self._meta.coin]
                open_n = len(oo)
                self._working = tuple(str(o["cloid"]) for o in oo if o.get("cloid"))
                self._working_ts = now
                for o in oo:
                    if self._resting.get(o["oid"], now + 1) <= now:
                        self._ex.cancel(self._meta.coin, o["oid"])
                for f in self._info.user_fills_by_time(self._address, self._fills_since_ms):
                    key = f"{f['tid']}"
                    if f["coin"] != self._meta.coin or key in self._seen_fills:
                        continue
                    self._seen_fills.add(key)
                    self._fills.append(
                        Fill(f["time"] / 1000.0, f["coin"], f["side"] == "B", float(f["px"]), float(f["sz"]),
                             float(f["fee"]), not f.get("crossed", True), str(f.get("cloid") or ""))
                    )  # fmt: skip
            self._last_pos = pos
            self.snapshot = snap | {"address": self._address, "ts": now}
            self._acct = AccountState(now, True, snap["equity"], pos, snap["entry_px"], open_n)
        except Exception:  # noqa: BLE001
            self.errors += 1
            log.exception("account refresh failed; account state is now UNKNOWN")
            self._acct = AccountState(self._acct.ts, known=False)

    def requests_left(self) -> int | None:
        """Order actions left in the account's Hyperliquid budget, or None if it cannot be read."""
        try:
            r = self._info.post("/info", {"type": "userRateLimit", "user": self._address})
            left = int(r["nRequestsCap"]) - int(r["nRequestsUsed"])
            self.snapshot = self.snapshot | {"requests_left": left}
            return left
        except Exception:  # noqa: BLE001 - informational
            return None

    def set_max_cross_leverage(self) -> None:
        self._ex.update_leverage(int(self._meta.max_leverage), self._meta.coin, is_cross=True)

    def _place(self, seq: int, it: OrderIntent, now: float) -> None:
        try:
            cloid = None
            if it.client_id.startswith("0x") and len(it.client_id) == 34:
                from hyperliquid.utils.types import Cloid

                cloid = Cloid.from_str(it.client_id)
            r = self._ex.order(
                it.coin, it.is_buy, it.sz, it.limit_px, {"limit": {"tif": it.tif}}, reduce_only=it.reduce_only, cloid=cloid
            )
            if r.get("status") != "ok":
                raise RuntimeError(f"order rejected: {r}")
            for s in r["response"]["data"]["statuses"]:
                if "error" in s:
                    log.warning("order error: %s", s["error"])
                elif "resting" in s:
                    self._resting[s["resting"]["oid"]] = now + it.ttl_s
            self._inflight.pop(seq, None)
        except Exception:  # noqa: BLE001
            # Leave the entry in _inflight: an unacknowledged order is an instrumentation fault
            # and the kernel blocks trading until an operator restarts and reconciles.
            self.errors += 1
            log.exception("order placement failed or unacknowledged")

    def _cancel_one(self, client_id: str) -> None:
        try:
            from hyperliquid.utils.types import Cloid

            self._ex.cancel_by_cloid(self._meta.coin, Cloid.from_str(client_id))
        except Exception:  # noqa: BLE001 - usually "already filled or cancelled"; the next refresh shows the truth
            self.errors += 1
            log.warning("cancel of %s failed", client_id, exc_info=True)

    def _cancel_all(self) -> None:
        try:
            for o in self._info.open_orders(self._address):
                if o["coin"] == self._meta.coin:
                    self._ex.cancel(self._meta.coin, o["oid"])
        except Exception:  # noqa: BLE001
            self.errors += 1
            log.exception("cancel_all failed")
