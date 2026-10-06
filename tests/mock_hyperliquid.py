"""A stand-in Hyperliquid for end-to-end tests: REST /info and /exchange plus the market-data WebSocket.

It models one *unified* account (collateral in spot USDC, perp account value reading 0), which
is Hyperliquid's default for new accounts. Orders sent through the real SDK are recorded and
filled in full. Signatures are not verified: this checks that the app talks to the exchange
correctly, not that Hyperliquid would accept the signature.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from aiohttp import WSMsgType, web

OWNER = "0x" + "ab" * 20


class MockHyperliquid:
    def __init__(self, usdc: float = 10.0, position: float = 0.0, mid: float = 85000.0) -> None:
        self.usdc, self.position, self.mid = usdc, position, mid
        self.orders: list[dict[str, Any]] = []
        self.actions: list[str] = []
        self.fills: list[dict[str, Any]] = []
        self.role: dict[str, Any] = {"role": "agent", "data": {"user": OWNER}}
        self.info_requests: list[str] = []
        self.resting: list[dict[str, Any]] = []  # post-only orders waiting in the book
        self.walk = 0.0  # set > 0 to make the mid move a little each update, so volatility is not zero
        self._tid = 0

    def hit(self, is_buy: bool) -> dict[str, Any]:
        """The market trades into one of our resting quotes: fill it in full."""
        o = next(r for r in self.resting if r["b"] is is_buy)
        self.resting.remove(o)
        sz = float(o["s"])
        self.position = round(self.position + (sz if is_buy else -sz), 8)
        self._tid += 1
        self.fills.append({"tid": self._tid, "coin": "BTC", "side": "B" if is_buy else "A", "px": o["p"], "sz": o["s"],
                           "fee": "0.0", "time": int(time.time() * 1000), "crossed": False, "cloid": o.get("c")})  # fmt: skip
        return o

    def app(self) -> web.Application:
        app = web.Application()
        app.add_routes([web.post("/info", self.info), web.post("/exchange", self.exchange), web.get("/ws", self.ws)])
        return app

    async def info(self, req: web.Request) -> web.Response:
        body = await req.json()
        kind = body.get("type")
        self.info_requests.append(kind)
        out: Any
        if kind == "meta":
            out = {"universe": [{"name": "BTC", "szDecimals": 5, "maxLeverage": 40}]}
        elif kind == "spotMeta":
            out = {"universe": [], "tokens": []}
        elif kind == "perpDexs":
            out = [None]
        elif kind == "userRole":
            out = self.role
        elif kind == "userAbstraction":
            out = "unifiedAccount"
        elif kind == "clearinghouseState":
            pos = [{"position": {"coin": "BTC", "szi": str(self.position), "entryPx": str(self.mid), "unrealizedPnl": "0.0"}}]
            out = {"marginSummary": {"accountValue": "0.0"}, "withdrawable": "0.0", "assetPositions": pos if self.position else []}
        elif kind == "spotClearinghouseState":
            out = {"balances": [{"coin": "USDC", "total": str(self.usdc), "hold": "0.0"}]}
        elif kind in ("openOrders", "frontendOpenOrders"):
            out = [{"coin": "BTC", "oid": r["oid"], "cloid": r.get("c"), "side": "B" if r["b"] else "A", "limitPx": r["p"],
                    "sz": r["s"]} for r in self.resting]  # fmt: skip
        elif kind == "userFillsByTime":
            out = self.fills
        elif kind == "userFees":
            out = {"userCrossRate": "0.00045", "userAddRate": "0.00015"}
        elif kind == "userRateLimit":
            out = {"cumVlm": "0.0", "nRequestsUsed": len(self.actions), "nRequestsCap": 10000}
        elif kind == "candleSnapshot":
            out = []
        else:
            out = {}
        return web.json_response(out)

    async def exchange(self, req: web.Request) -> web.Response:
        body = await req.json()
        action = body["action"]
        self.actions.append(action["type"])
        assert body.get("signature") and body.get("nonce"), "every exchange action must be signed"
        if action["type"] == "cancelByCloid":
            gone = {c["cloid"] for c in action["cancels"]}
            self.resting = [r for r in self.resting if r.get("c") not in gone]
            return web.json_response({"status": "ok", "response": {"type": "cancel", "data": {"statuses": ["success"] * len(gone)}}})
        if action["type"] != "order":
            return web.json_response({"status": "ok", "response": {"type": "default"}})
        statuses = []
        for o in action["orders"]:
            self.orders.append(o)
            sz, buy = float(o["s"]), bool(o["b"])
            if o["t"] == {"limit": {"tif": "Alo"}}:
                crosses = float(o["p"]) >= self.mid + 0.5 if buy else float(o["p"]) <= self.mid - 0.5
                if crosses:
                    statuses.append({"error": "Post only order would have immediately matched."})
                    continue
                self._tid += 1
                self.resting.append(o | {"oid": self._tid})
                statuses.append({"resting": {"oid": self._tid, "cloid": o.get("c")}})
                continue
            if o["r"] and (self.position == 0 or (self.position > 0) == buy):
                statuses.append({"error": "Reduce only order would increase position."})
                continue
            self.position = round(self.position + (sz if buy else -sz), 8)
            self._tid += 1
            self.fills.append({"tid": self._tid, "coin": "BTC", "side": "B" if buy else "A", "px": str(self.mid), "sz": o["s"],
                               "fee": f"{sz * self.mid * 0.00045:.6f}", "time": int(time.time() * 1000), "crossed": True})  # fmt: skip
            statuses.append({"filled": {"totalSz": o["s"], "avgPx": str(self.mid), "oid": self._tid}})
        return web.json_response({"status": "ok", "response": {"type": "order", "data": {"statuses": statuses}}})

    async def ws(self, req: web.Request) -> web.WebSocketResponse:
        sock = web.WebSocketResponse()
        await sock.prepare(req)

        async def pump() -> None:
            n = 0
            while not sock.closed:
                n += 1
                if self.walk:
                    self.mid = round(self.mid + self.walk * (1 if (n * 7919) % 5 < 2 else -1 if (n * 7919) % 5 < 4 else 0))
                ms = int(time.time() * 1000)
                bid, ask = self.mid - 0.5, self.mid + 0.5
                levels = [[{"px": str(bid - i), "sz": "2.0", "n": 3} for i in range(20)],
                          [{"px": str(ask + i), "sz": "2.0", "n": 3} for i in range(20)]]  # fmt: skip
                await sock.send_json({"channel": "l2Book", "data": {"coin": "BTC", "time": ms, "levels": levels}})
                await sock.send_json({"channel": "bbo", "data": {"coin": "BTC", "time": ms, "bbo": [levels[0][0], levels[1][0]]}})
                ctx = {"funding": "0.0000125", "openInterest": "1000", "oraclePx": str(self.mid), "markPx": str(self.mid),
                       "premium": "0.0"}  # fmt: skip
                await sock.send_json({"channel": "activeAssetCtx", "data": {"coin": "BTC", "ctx": ctx}})
                trade = {"coin": "BTC", "side": "B" if n % 2 else "A", "px": str(ask if n % 2 else bid), "sz": "0.01", "time": ms}
                await sock.send_json({"channel": "trades", "data": [trade]})
                await asyncio.sleep(0.2)

        task = asyncio.create_task(pump())
        try:
            async for msg in sock:
                if msg.type == WSMsgType.TEXT:
                    await sock.send_json({"channel": "subscriptionResponse", "data": {}})
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, ConnectionError):
                await task
        return sock
