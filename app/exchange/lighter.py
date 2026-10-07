"""Lighter adapter: public market data only (order book, trades, market stats).

Lighter is a zero-fee perpetual exchange: its default "Standard" account pays no maker or taker
fee, and in exchange its orders are delayed (about 300 ms for taker orders at the time of
writing). This module reads Lighter's public WebSocket and re-shapes every message into the
same records the Hyperliquid adapter produces, so recordings of Lighter replay through the
engine, the simulator and the research tools unchanged. Markets are named "lighter:<SYMBOL>".

There is no order placement here. Whether to build that is decided by what the recordings show.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import aiohttp

from app.exchange.base import AssetMeta

log = logging.getLogger(__name__)
API = "https://mainnet.zklighter.elliot.ai"
WS = "wss://mainnet.zklighter.elliot.ai/stream"
PREFIX = "lighter:"


async def markets() -> dict[str, dict[str, Any]]:
    """Active perpetual markets by symbol, with ids, decimals, fees and leverage limits."""
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as s, s.get(API + "/api/v1/orderBookDetails") as r:
        r.raise_for_status()
        rows = (await r.json())["order_book_details"]
    return {m["symbol"]: m for m in rows if m.get("status") == "active" and m.get("market_type") == "perp"}


def asset_meta(symbol: str, m: dict[str, Any]) -> AssetMeta:
    """Lighter prices sit on a fixed decimal grid (BTC: $0.1), unlike Hyperliquid's significant-figure rule."""
    max_lev = 10_000.0 / float(m.get("min_initial_margin_fraction") or 500)
    return AssetMeta(PREFIX + symbol, int(m["size_decimals"]), max_lev, float(m.get("min_quote_amount") or 10.0),
                     price_tick=10.0 ** -int(m["price_decimals"]))  # fmt: skip


class LocalBook:
    """The full book, kept current from a snapshot plus incremental updates (size 0 removes a level)."""

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}

    def apply(self, ob: dict[str, Any], snapshot: bool) -> None:
        if snapshot:
            self.bids.clear()
            self.asks.clear()
        for side, book in (("bids", self.bids), ("asks", self.asks)):
            for lv in ob.get(side, []):
                px, sz = float(lv["price"]), float(lv["size"])
                if sz <= 0:
                    book.pop(px, None)
                else:
                    book[px] = sz

    def top(self, n: int) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        bids = [(p, self.bids[p]) for p in heapq.nlargest(n, self.bids)]
        asks = [(p, self.asks[p]) for p in heapq.nsmallest(n, self.asks)]
        return bids, asks

    def crossed(self) -> bool:
        return bool(self.bids and self.asks and max(self.bids) >= min(self.asks))


def _levels(side: list[tuple[float, float]]) -> list[dict[str, Any]]:
    return [{"px": repr(p), "sz": repr(s), "n": 1} for p, s in side]


async def raw_stream(symbols: list[str], depth: int = 10, min_book_interval_s: float = 0.1) -> AsyncIterator[tuple[float, str, Any]]:
    """Yield (recv_ts, channel, data) in the Hyperliquid record shape: l2Book, bbo, trades, activeAssetCtx.

    Book snapshots are emitted at most every `min_book_interval_s`; a top-of-book record is emitted on every
    change of the best bid or ask. Reconnects forever; a crossed local book forces a fresh snapshot.
    """
    info = await markets()
    ids = {int(info[s]["market_id"]): s for s in symbols if s in info}
    missing = [s for s in symbols if s not in info]
    if missing:
        raise ValueError(f"not active Lighter perpetuals: {missing}")
    backoff = 0.5
    while True:
        books = {mid: LocalBook() for mid in ids}
        last_emit = dict.fromkeys(ids, 0.0)
        last_bbo: dict[int, tuple[float, float, float, float]] = {}
        try:
            async with aiohttp.ClientSession() as s, s.ws_connect(WS, heartbeat=None, max_msg_size=0) as ws:
                for mid in ids:
                    for ch in ("order_book", "trade", "market_stats"):
                        await ws.send_json({"type": "subscribe", "channel": f"{ch}/{mid}"})
                backoff = 0.5
                last_ping = time.time()
                while True:
                    msg = await ws.receive(timeout=20)
                    now = time.time()
                    if now - last_ping > 30:  # the server drops connections that never speak
                        await ws.send_json({"type": "ping"})
                        last_ping = now
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        raise ConnectionError(f"websocket closed: {msg.type}")
                    m = msg.json()
                    kind = m.get("type", "")
                    if kind == "ping":
                        await ws.send_json({"type": "pong"})
                        continue
                    channel = str(m.get("channel", ""))
                    if ":" not in channel:
                        continue
                    name, _, raw_id = channel.partition(":")
                    mid = int(raw_id)
                    if mid not in ids:
                        continue
                    coin = PREFIX + ids[mid]
                    if name == "order_book":
                        book = books[mid]
                        book.apply(m["order_book"], snapshot=kind.startswith("subscribed"))
                        if book.crossed():
                            raise ConnectionError("local order book crossed; resubscribing for a fresh snapshot")
                        bids, asks = book.top(depth)
                        if not bids or not asks:
                            continue
                        ms = int(m.get("timestamp") or now * 1000)
                        bbo = (bids[0][0], bids[0][1], asks[0][0], asks[0][1])
                        if now - last_emit[mid] >= min_book_interval_s:
                            last_emit[mid] = now
                            last_bbo[mid] = bbo
                            yield now, "l2Book", {"coin": coin, "time": ms, "levels": [_levels(bids), _levels(asks)]}
                        elif last_bbo.get(mid) != bbo:
                            last_bbo[mid] = bbo
                            yield now, "bbo", {"coin": coin, "time": ms, "bbo": _levels(bids[:1]) + _levels(asks[:1])}
                    elif name == "trade" and kind.startswith("update"):
                        rows = [{"coin": coin, "side": "B" if t.get("is_maker_ask") else "A", "px": str(t["price"]), "sz": str(t["size"]),
                                 "time": int(t.get("timestamp") or now * 1000)} for t in m.get("trades", [])]  # fmt: skip
                        if rows:
                            yield now, "trades", rows
                    elif name == "market_stats":
                        st = m.get("market_stats", {})
                        idx, mark = float(st.get("index_price") or 0.0), float(st.get("mark_price") or 0.0)
                        if idx > 0 and mark > 0:
                            ctx = {
                                "funding": str(float(st.get("current_funding_rate") or 0.0) / 100.0),  # quoted in percent per hour
                                "premium": str(mark / idx - 1.0), "oraclePx": str(idx), "markPx": str(mark),
                                "openInterest": str(st.get("open_interest") or 0.0),
                            }  # fmt: skip
                            yield now, "activeAssetCtx", {"coin": coin, "ctx": ctx}
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - any transport or consistency failure means reconnect
            log.warning("lighter stream error (%s); reconnecting in %.1fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15.0)
