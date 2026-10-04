"""Measure latency to Hyperliquid from this machine: python scripts/latency.py [--testnet]"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time

import aiohttp

from app.config.settings import MAINNET_API, TESTNET_API


async def main() -> None:
    api = TESTNET_API if "--testnet" in sys.argv else MAINNET_API
    rest: list[float] = []
    age: list[float] = []
    async with aiohttp.ClientSession() as s:
        for _ in range(20):
            t = time.perf_counter()
            async with s.post(api + "/info", json={"type": "l2Book", "coin": "BTC"}) as r:
                await r.read()
            rest.append((time.perf_counter() - t) * 1000)
        async with s.ws_connect(api.replace("https", "wss") + "/ws") as ws:
            await ws.send_json({"method": "subscribe", "subscription": {"type": "bbo", "coin": "BTC"}})
            while len(age) < 40:
                m = (await ws.receive()).json()
                if m.get("channel") == "bbo":
                    age.append(time.time() * 1000 - m["data"]["time"])

    def show(name: str, v: list[float]) -> None:
        q = statistics.quantiles(v, n=20)
        print(f"{name:34s} p50 {statistics.median(v):7.1f} ms   p95 {q[18]:7.1f} ms   min {min(v):7.1f} ms")

    show("REST /info round trip", rest[1:])
    show("WS bbo age  (needs synced clock)", age)


asyncio.run(main())
