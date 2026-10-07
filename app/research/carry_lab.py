"""Carry lab: does collecting funding pay after fees?

    python -m app.research.carry_lab

On a perpetual, longs pay shorts a funding rate every hour when the perp trades above spot
(and the reverse when below). Holding the coin on spot and shorting the same amount on the perp
leaves no exposure to the price, and collects that funding. This is not scalping and it is
not fast; it is the one thing measured in this project whose income is larger than its fees.

For each coin this takes a year of real Hyperliquid funding payments and works out, per $1 of
position:

- always_in: open once, hold all year, close once;
- rule: hold only while the trailing 7-day funding is positive, paying fees on every switch.

Costs charged: spot taker + perp taker to open, and again to close (23 bps per round trip at
base fees). Return on capital assumes the short is margined at 3x, so $1 of position ties up
about $1.33.

What this does NOT model, and what can lose money: the spot and perp prices drifting apart
while you hold or when you close; a fast rally liquidating the short if it is under-margined;
funding turning negative for longer than the past year shows; and the bridged spot asset
(for example UBTC for BTC) losing its peg. Writes models/carry_lab.json.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np

from app.config.settings import MAINNET_API, Settings

SPOT_TAKER, PERP_TAKER = 0.0007, 0.00045
ROUND_TRIP = 2 * (SPOT_TAKER + PERP_TAKER)
CAPITAL_PER_NOTIONAL = 1 + 1 / 3  # spot leg plus 3x-margined short
HOURS_YEAR = 24 * 365


async def funding_history(coin: str, days: int = 365) -> tuple[np.ndarray, np.ndarray]:
    now = int(time.time() * 1000)
    start = now - days * 86400 * 1000
    ts: list[int] = []
    rates: list[float] = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
        while start < now:
            async with s.post(MAINNET_API + "/info", json={"type": "fundingHistory", "coin": coin, "startTime": start, "endTime": now}) as r:
                page = await r.json()
            if not page:
                break
            ts += [int(p["time"]) for p in page]
            rates += [float(p["fundingRate"]) for p in page]
            nxt = int(page[-1]["time"]) + 1
            if nxt <= start or len(page) < 2:
                break
            start = nxt
            await asyncio.sleep(0.25)
    return np.array(ts), np.array(rates)


def evaluate(rates: np.ndarray, window_h: int = 168) -> dict[str, Any]:
    n = len(rates)
    years = n / HOURS_YEAR
    cum = np.cumsum(rates)
    worst = float((np.maximum.accumulate(cum) - cum).max())
    always = float(rates.sum() - ROUND_TRIP)

    # Rule: in while the trailing week's funding is positive. Decided from past hours only.
    trail = np.convolve(rates, np.ones(window_h), mode="full")[:n]
    held = np.zeros(n, dtype=bool)
    held[window_h:] = trail[window_h - 1 : -1] > 0
    switches = int(np.abs(np.diff(held.astype(int), prepend=0)).sum()) + int(held[-1])  # includes the final close
    rule = float(rates[held].sum() - switches * ROUND_TRIP / 2)
    return {
        "hours": n, "hours_funding_positive_pct": float((rates > 0).mean() * 100),
        "funding_apr_pct": float(rates.mean() * HOURS_YEAR * 100),
        "worst_run_of_negative_funding_pct": worst * 100,
        "always_in": {"net_pct_of_position_per_year": always / years * 100,
                      "net_pct_of_capital_per_year": always / years / CAPITAL_PER_NOTIONAL * 100, "fees_pct": ROUND_TRIP * 100},
        "rule_trailing_week_positive": {
            "net_pct_of_position_per_year": rule / years * 100, "net_pct_of_capital_per_year": rule / years / CAPITAL_PER_NOTIONAL * 100,
            "time_held_pct": float(held.mean() * 100), "opens_and_closes": switches, "fees_pct": switches * ROUND_TRIP / 2 * 100,
        },
        "by_quarter_apr_pct": [float(q.mean() * HOURS_YEAR * 100) for q in np.array_split(rates, 4)],
    }  # fmt: skip


async def run(coins: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"round_trip_fees_bps": ROUND_TRIP * 1e4, "capital_per_dollar_of_position": CAPITAL_PER_NOTIONAL,
                           "not_modelled": ["spot-perp basis", "liquidation of the short", "spot asset peg", "funding regime change"],
                           "coins": {}}  # fmt: skip
    for c in coins:
        _, rates = await funding_history(c)
        if len(rates) > 24 * 30:
            out["coins"][c] = evaluate(rates)
    return out


def main() -> None:
    s = Settings.from_env()
    r = asyncio.run(run(["BTC", "ETH", "SOL", "HYPE"]))
    print(f"{'coin':5s} {'funding APR':>11s} {'hours +':>8s} {'worst run':>10s} | {'hold all year: net of capital':>30s} | {'rule: net of capital':>21s} {'held':>6s} {'switches':>8s}")
    for c, v in r["coins"].items():
        a, b = v["always_in"], v["rule_trailing_week_positive"]
        print(f"{c:5s} {v['funding_apr_pct']:10.1f}% {v['hours_funding_positive_pct']:7.1f}% {v['worst_run_of_negative_funding_pct']:9.2f}% | "
              f"{a['net_pct_of_capital_per_year']:29.1f}% | {b['net_pct_of_capital_per_year']:20.1f}% {b['time_held_pct']:5.0f}% {b['opens_and_closes']:8d}")  # fmt: skip
    out = Path(s.chart_model_dir) / "carry_lab.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
