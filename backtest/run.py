"""Replay recorded market data through the *same* Engine and PaperVenue used in paper mode.

    python -m backtest.run data/raw.jsonl            # recorded with HL_RECORD_RAW=1
    python -m backtest.run --synthetic 20000         # self-test on generated data (not evidence of edge)

The model learns online during the replay (predict, then update), so every forecast and
every trade in the report is out-of-sample with respect to the data that came before it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from app.brain.engine import Engine
from app.config.settings import Mode, Settings
from app.exchange.base import AssetMeta, Book, Trade
from app.exchange.hyperliquid import parse_event
from app.exchange.paper import PaperVenue

Event = tuple[float, str, Any]


def read_events(path: str) -> Iterator[Event]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ev = parse_event(r["ch"], r["d"], r["t"])
            if ev:
                yield r["t"], ev[0], ev[1]


def synthetic_events(
    seconds: int, signal: float = 0.0, seed: int = 0, px0: float = 85000.0, vol_bps: float = 0.6
) -> Iterator[Event]:
    """Random-walk market. With signal > 0, touch imbalance predicts the next 1s return (bps per unit)."""
    rng = np.random.default_rng(seed)
    mid, imb = px0, 0.0
    for t in range(seconds):
        ts = 1_000_000.0 + t
        mid *= float(np.exp((signal * imb + vol_bps * rng.standard_normal()) * 1e-4))
        imb = 0.7 * imb + 0.5 * float(rng.standard_normal())
        skew = float(np.tanh(imb))
        b0 = round(mid - 0.5)
        lv = np.arange(20.0)
        bids = np.column_stack([b0 - lv, np.full(20, 2.0)])
        asks = np.column_stack([b0 + 1 + lv, np.full(20, 2.0)])
        bids[0, 1], asks[0, 1] = 2.0 * (1 + 0.9 * skew), 2.0 * (1 - 0.9 * skew)
        yield ts, "book", Book("BTC", ts, ts, bids, asks)
        buy = bool(rng.random() < 0.5)
        yield ts, "trades", [Trade(ts, b0 + 1 if buy else b0, float(rng.exponential(0.05)), buy)]


def run_backtest(events: Iterable[Event], s: Settings, meta: AssetMeta) -> dict[str, Any]:
    s = dataclasses.replace(s, mode=Mode.BACKTEST)
    venue = PaperVenue(s.paper_equity, meta, s.taker_fee, s.maker_fee, s.latency_ms / 1000)
    eng = Engine(s, venue, meta)
    next_tick: float | None = None
    actions = 0
    for ts, kind, payload in events:
        if next_tick is None:
            next_tick = ts + s.decision_interval_s
        while ts > next_tick:  # events stamped exactly at the tick are visible to it
            d = eng.on_tick(next_tick)
            actions += d is not None and d.order is not None
            next_tick += s.decision_interval_s
        if kind == "book":
            eng.on_book(payload)
        elif kind == "bbo":
            eng.on_bbo(ts, payload)
        elif kind == "trades":
            eng.on_trades(payload)
        else:
            eng.on_ctx(payload)
    snap = eng.snapshot()
    return {
        "ticks": eng.ticks,
        "orders": actions,
        "rejects": venue.rejects,
        "funding_paid": venue.funding_paid,
        **snap["journal"],
        "oos_ic": snap["model"]["oos_ic"],
        "resolved_forecasts": snap["model"]["resolved_forecasts"],
        "calibration": snap["model"]["calibration"],
        "weights": snap["model"]["weights"],
        "recent_decisions": snap["recent_decisions"][-3:],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--synthetic", type=int, default=0, metavar="SECONDS")
    ap.add_argument("--signal", type=float, default=0.0, help="planted edge for --synthetic, bps per unit imbalance")
    ap.add_argument("--horizon", type=float, default=None)
    ap.add_argument("--max-leverage", type=float, default=40.0)
    args = ap.parse_args()
    s = Settings.from_env()
    if args.horizon:
        s = dataclasses.replace(s, horizon_s=args.horizon)
    meta = AssetMeta(s.coin, 5, args.max_leverage)
    if args.synthetic:
        events: Iterable[Event] = synthetic_events(args.synthetic, args.signal)
    elif args.path:
        events = read_events(args.path)
    else:
        ap.error("give a recorded file or --synthetic N")
    print(json.dumps(run_backtest(events, s, meta), indent=1, default=float))


if __name__ == "__main__":
    main()
