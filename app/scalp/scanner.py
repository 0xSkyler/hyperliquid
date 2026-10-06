"""Market scanner: where on Hyperliquid can a passive scalper get paid at all?

A maker earns the bid-ask spread and pays the maker fee on both legs, so the first-order test
for a market is simply: spread > 2 x maker fee. This ranks every perpetual by that margin.
It is a snapshot of one moment, and a wide spread is usually wide for a reason (fast, thin,
jumpy markets), so treat the list as "worth recording and simulating", not "safe to quote".
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol


class InfoSource(Protocol):
    async def info(self, payload: dict[str, Any]) -> Any: ...


async def scan(data: InfoSource, maker_fee: float, min_volume_usd: float = 2e6, limit: int = 80) -> list[dict[str, Any]]:
    meta, ctxs = await data.info({"type": "metaAndAssetCtxs"})
    coins = [(u["name"], float(c.get("dayNtlVlm") or 0.0), float(u.get("maxLeverage", 1)), float(c.get("funding") or 0.0))
             for u, c in zip(meta["universe"], ctxs, strict=False) if not u.get("isDelisted")]  # fmt: skip
    coins = sorted((c for c in coins if c[1] >= min_volume_usd), key=lambda c: -c[1])[:limit]
    sem = asyncio.Semaphore(8)
    fee_bps = maker_fee * 1e4

    async def one(name: str, vol: float, lev: float, funding: float) -> dict[str, Any] | None:
        async with sem:
            try:
                b = await data.info({"type": "l2Book", "coin": name})
                bid, ask = float(b["levels"][0][0]["px"]), float(b["levels"][1][0]["px"])
                bq, aq = float(b["levels"][0][0]["sz"]), float(b["levels"][1][0]["sz"])
            except (KeyError, IndexError, TypeError, ValueError):
                return None
        mid = (bid + ask) / 2
        spread = (ask - bid) / mid * 1e4
        return {
            "coin": name, "volume_usd": vol, "spread_bps": spread, "maker_round_trip_bps": 2 * fee_bps,
            "margin_bps": spread - 2 * fee_bps,  # what one full round trip at the touch earns before adverse selection
            "touch_depth_usd": min(bq, aq) * mid, "max_leverage": lev, "funding_hourly_bps": funding * 1e4,
        }  # fmt: skip

    rows = [r for r in await asyncio.gather(*(one(*c) for c in coins)) if r is not None]
    return sorted(rows, key=lambda r: -r["margin_bps"])
