"""Scalp lab: replay recorded markets through the scalper and measure its skill, per fill.

    python scripts/record.py --coins BTC,ETH,ENA --minutes 60     # record first
    python -m app.research.scalp_lab "data/rec/*.jsonl"

Each recording is replayed twice through the real engine in maker mode against the simulator
(order and cancel latency, post-only rejects, queue position, fills only when the market trades
at or through our price):

- GROSS (fees set to zero): the scalper's raw skill. Spread captured per fill, and where the
  price went after the fill. Capture minus adverse movement is what the quoting itself earns.
- NET (real maker and taker fees): what would have reached the account.

With fees on, the scalper refuses quotes that cannot pay for themselves, so on tight markets
the NET run may simply not trade. That is its fee discipline, and it is the right behaviour.

Simulation limits: we are assumed last in the queue at our price, other people's cancels ahead
of us are ignored (pessimistic), and our own orders do not move the market (optimistic for
size). A short recording is a sample of one market mood, not proof.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

from app.brain.engine import Engine
from app.config.settings import Mode, Settings
from app.exchange.base import AssetMeta
from app.exchange.paper import PaperVenue
from app.research.dataset import expand_paths, read_events

# szDecimals / max leverage for replay when offline; unknown coins fall back to a generic small-cap spec.
SPECS = {"BTC": (5, 40), "ETH": (4, 25), "SOL": (2, 20), "HYPE": (2, 10), "ENA": (0, 10), "ZRO": (1, 5), "XPL": (0, 10)}


def replay(path: str, s: Settings, meta: AssetMeta, equity: float) -> dict[str, Any]:
    s = dataclasses.replace(s, mode=Mode.BACKTEST, strategy="maker", coin=meta.coin, models=("ridge",))
    venue = PaperVenue(equity, meta, s.taker_fee, s.maker_fee, s.latency_ms / 1000)
    eng = Engine(s, venue, meta)
    next_tick: float | None = None
    first = last = 0.0
    spreads: list[float] = []
    for ts, kind, payload in read_events([path]):
        first = first or ts
        last = ts
        if next_tick is None:
            next_tick = ts + s.decision_interval_s
        while ts > next_tick:
            eng.on_tick(next_tick)
            next_tick += s.decision_interval_s
        if kind == "book":
            eng.on_book(payload)
            if payload.valid() and len(spreads) < 100_000:
                spreads.append((payload.best_ask - payload.best_bid) / payload.mid * 1e4)
        elif kind == "bbo":
            eng.on_bbo(ts, payload)
        elif kind == "trades":
            eng.on_trades(payload)
        elif kind == "ctx":
            eng.on_ctx(payload)
    j = eng.journal.summary()
    hours = max(last - first, 1.0) / 3600
    move = j["move_after_fill_bps"]
    return {
        "hours": hours, "median_spread_bps": sorted(spreads)[len(spreads) // 2] if spreads else None,
        "fills": j["fills"], "fills_per_hour": j["fills"] / hours, "maker_ratio": j["maker_ratio"],
        "volume_usd": j["volume"], "spread_capture_bps": j["spread_capture_bps"],
        "move_after_fill_bps": move, "edge_per_fill_bps_before_fees": j["spread_capture_bps"] + move["5s"],
        "fees_usd": j["fees"], "pnl_usd": j["equity"] - equity, "pnl_pct": (j["equity"] / equity - 1) * 100,
        "max_drawdown_pct": j["max_drawdown_pct"], "max_inventory_x": eng.max_abs_exposure,
        "quote_uptime_pct": eng.snapshot()["scalper"]["quote_uptime_pct"],
        "orders_placed": eng.quotes.placed, "orders_cancelled": eng.quotes.cancelled, "liquidations": j["liquidations"],
        "fast_alpha": eng.fast_alpha.snapshot(), "takes": eng.takes,
        "markout_5s_vs_fill_bps": j["markout_5s_bps"],
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="recordings (globs allowed); the coin is the file name before the first '-'")
    ap.add_argument("--equity", type=float, default=1000.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    s = Settings.from_env()
    rows = []
    print(f"{'market':7s} {'hours':>5s} {'spread':>7s} | {'GROSS fills':>11s} {'capture':>8s} {'move 5s':>8s} {'edge/fill':>9s} {'pnl %':>7s} | "
          f"{'NET fills':>9s} {'pnl %':>7s} {'uptime %':>8s}")
    for path in expand_paths(args.paths):
        coin = Path(path).name.split("-")[0].upper()
        if Path(path).stat().st_size == 0:
            continue
        dec, lev = SPECS.get(coin, (1, 5))
        meta = AssetMeta(coin, dec, float(lev))
        gross = replay(path, dataclasses.replace(s, maker_fee=0.0, taker_fee=0.0), meta, args.equity)
        net = replay(path, s, meta, args.equity)
        rows.append({"market": coin, "file": Path(path).name, "gross": gross, "net": net})
        sp = gross["median_spread_bps"]
        print(f"{coin:7s} {gross['hours']:5.2f} {sp if sp is None else round(sp, 2)!s:>7s} | {gross['fills']:11d} "
              f"{gross['spread_capture_bps']:8.2f} {gross['move_after_fill_bps']['5s']:8.2f} "
              f"{gross['edge_per_fill_bps_before_fees']:9.2f} {gross['pnl_pct']:7.3f} | {net['fills']:9d} {net['pnl_pct']:7.3f} "
              f"{net['quote_uptime_pct']:8.1f}")  # fmt: skip
    out = Path(args.out or Path(s.chart_model_dir) / "scalp_lab.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    note = {"maker_fee": s.maker_fee, "taker_fee": s.taker_fee, "equity": args.equity,
            "caveat": "short recordings; simulated fills (see module docstring)"}  # fmt: skip
    out.write_text(json.dumps({"settings": note, "results": rows}, indent=1, default=lambda x: None if math.isnan(x) else x), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
