"""Stress lab: run the real engine through hostile market environments.

    python -m app.research.stress

Each environment is a synthetic market with a genuine planted edge, so the engine has learned
to trust its forecasts and is carrying leveraged positions when the event hits. That is the
worst case: an engine that never trades survives everything trivially.

Reported per environment: net return, worst drawdown, largest exposure taken, fills,
liquidations, and how many ticks the safety kernel blocked trading. Writes
models/stress_report.json. These are simulations with simplified liquidity; they show how
the decision logic behaves, not what a real venue would do to your orders.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from app.config.settings import Settings
from app.exchange.base import AssetMeta, Book, Trade
from backtest.run import Event, run_backtest

EVENT_AT = 0.7  # the shock arrives 70% of the way through
ENVIRONMENTS = (
    "calm", "trending", "choppy", "vol_spike", "flash_crash", "gap_down", "thin_liquidity",
    "feed_outage", "crossed_book", "edge_disappears", "edge_reverses",
)  # fmt: skip


def environment(name: str, seconds: int, seed: int = 0, signal: float = 6.0, px0: float = 85000.0) -> Iterator[Event]:
    """Random-walk market where touch imbalance predicts the next 1s return, plus the named disturbance."""
    if name not in ENVIRONMENTS:
        raise ValueError(name)
    rng = np.random.default_rng(seed)
    t0 = int(seconds * EVENT_AT)
    mid, imb = px0, 0.0
    for t in range(seconds):
        ts = 1_000_000.0 + t
        vol, drift, depth, sig = 0.6, 0.0, 2.0, signal
        after = t >= t0
        if name == "trending":
            drift = 0.15
        elif name == "choppy":
            vol = 3.0
        elif name == "vol_spike" and t0 <= t < t0 + 300:
            vol = 5.0
        elif name == "flash_crash":
            if t0 <= t < t0 + 20:
                drift = -40.0  # about -8% in 20 seconds
            elif t0 + 20 <= t < t0 + 80:
                drift = 6.5  # partial rebound
        elif name == "gap_down" and t == t0:
            drift = -300.0  # -3% between two ticks
        elif name == "thin_liquidity" and after:
            depth = 0.02
        elif name == "edge_disappears" and after:
            sig = 0.0
        elif name == "edge_reverses" and after:
            sig = -signal

        mid *= float(np.exp((sig * imb + drift + vol * rng.standard_normal()) * 1e-4))
        imb = 0.7 * imb + 0.5 * float(rng.standard_normal())
        if name == "feed_outage" and t0 <= t < t0 + 120:
            mid *= float(np.exp(-0.8e-4))  # the market keeps moving (about -1%) while we are blind
            continue
        skew = float(np.tanh(imb))
        b0 = round(mid - 0.5)
        lv = np.arange(20.0)
        bids = np.column_stack([b0 - lv, np.full(20, depth)])
        asks = np.column_stack([b0 + 1 + lv, np.full(20, depth)])
        bids[0, 1], asks[0, 1] = depth * (1 + 0.9 * skew), depth * (1 - 0.9 * skew)
        if name == "crossed_book" and t0 <= t < t0 + 30:
            bids[0, 0] = asks[0, 0] + 5  # corrupt data: bid above ask
        yield ts, "book", Book("BTC", ts, ts, bids, asks)
        buy = bool(rng.random() < 0.5)
        yield ts, "trades", [Trade(ts, b0 + 1 if buy else b0, float(rng.exponential(0.05)), buy)]


def run_environment(name: str, s: Settings, meta: AssetMeta, seconds: int = 6000, seed: int = 0) -> dict[str, Any]:
    r = run_backtest(environment(name, seconds, seed), s, meta, keep_engine=True)
    eng = r["engine"]
    return {
        "environment": name, "net_return_pct": r["net_return_pct"], "max_drawdown_pct": r["max_drawdown_pct"],
        "max_exposure_x": eng.max_abs_exposure, "fills": r["fills"], "liquidations": r["liquidations"],
        "kernel_blocked_ticks": eng.halted_ticks, "final_equity": r["equity"],
    }  # fmt: skip


def main() -> None:
    s = dataclasses.replace(Settings.from_env(), horizon_s=5.0, models=("ridge",))
    meta = AssetMeta(s.coin, 5, 40.0)
    rows = [run_environment(name, s, meta) for name in ENVIRONMENTS]
    print(f"{'environment':18s} {'net %':>10s} {'max DD %':>9s} {'max lev':>8s} {'fills':>6s} {'liq':>4s} {'blocked':>8s}")
    for r in rows:
        print(f"{r['environment']:18s} {r['net_return_pct']:10.1f} {r['max_drawdown_pct']:9.1f} "
              f"{r['max_exposure_x']:8.1f} {r['fills']:6d} {r['liquidations']:4d} {r['kernel_blocked_ticks']:8d}")  # fmt: skip
    out = Path(s.chart_model_dir) / "stress_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    note = (f"synthetic markets with a planted edge; risk_aversion={s.risk_aversion}, "
            f"jump scenario {s.jump_size} @ {s.jump_prob}")  # fmt: skip
    out.write_text(json.dumps({"note": note, "results": rows}, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
