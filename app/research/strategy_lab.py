"""Strategy lab: every strategy family x parameter grid x timeframe x market regime, net of costs.

    python -m app.research.strategy_lab

For each timeframe (5m, 1h, 4h, 1d) it backtests every variant in app/strategies/library.py
over the full candle history, charging a taker fee on every change of position, and reports:

- full-period results per variant (this is in-sample: with ~40 variants x 4 timeframes, the
  best of them looks good by luck alone, so do not read these as evidence);
- results by market regime (bull/bear/range x high/low volatility);
- a walk-forward test: each year, pick the variant that did best on *earlier* years only and
  record what it then did that year. This is the honest number.

Funding payments are not modelled. Writes models/strategy_lab.json.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.config.settings import Settings
from app.research.history import history_path, load
from app.research.regimes import REGIME_NAMES, TIMEFRAMES, bar_regimes, resample
from app.strategies.library import VARIANTS, all_signals

YEAR_S = 365.25 * 86400


def net_returns(pos: np.ndarray, close: np.ndarray, cost: float) -> np.ndarray:
    """Per-bar net log-return of holding pos[t] from close t to close t+1, paying `cost` per unit turned over."""
    fwd = np.append(np.diff(np.log(close)), 0.0)
    turn = np.abs(np.diff(pos, axis=0, prepend=np.zeros((1, pos.shape[1]))))
    return pos * fwd[:, None] - cost * turn


def sharpe(r: np.ndarray, bars_per_year: float) -> np.ndarray:
    sd = r.std(axis=0)
    return np.where(sd > 0, r.mean(axis=0) / np.where(sd > 0, sd, 1) * math.sqrt(bars_per_year), 0.0)


def study(bars: np.ndarray, regimes: np.ndarray, step: int, cost: float) -> dict[str, Any]:
    pos = all_signals(bars)
    r = net_returns(pos, bars[:, 4], cost)
    bpy = YEAR_S / step
    year = bars[:, 0].astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
    turn = np.abs(np.diff(pos, axis=0, prepend=np.zeros((1, pos.shape[1])))).sum(axis=0)
    full = sharpe(r, bpy)
    variants: list[dict[str, Any]] = [{
        "name": v.name, "family": v.family, "net_sharpe": float(full[j]),
        "net_return_pct_per_year": float(r[:, j].mean() * bpy * 100),
        "turns_per_year": float(turn[j] / (len(bars) / bpy)),
        "by_regime": {name: float(sharpe(r[regimes == k, j : j + 1], bpy)[0]) if (regimes == k).sum() > 50 else None
                      for k, name in enumerate(REGIME_NAMES)},
    } for j, v in enumerate(VARIANTS)]  # fmt: skip

    # Walk-forward: choose on the past, score on the following year.
    years = sorted(set(year.tolist()))
    wf: list[dict[str, Any]] = []
    names = [v.name for v in VARIANTS]
    for y in years[2:]:
        past, now = year < y, year == y
        if now.sum() < 30:
            continue
        pick = int(np.argmax(sharpe(r[past], bpy)))
        wf.append({"year": int(y), "picked": VARIANTS[pick].name,
                   "picked_past_sharpe": float(sharpe(r[past], bpy)[pick]),
                   "net_return_pct": float(r[now, pick].sum() * 100),
                   "net_sharpe": float(sharpe(r[now, pick : pick + 1], bpy)[0])})  # fmt: skip
    oos = np.concatenate([r[year == w["year"], names.index(w["picked"])] for w in wf]) if wf else r[:0, 0]
    hold = np.append(np.diff(np.log(bars[:, 4])), 0.0)
    best = sorted(variants, key=lambda v: -v["net_sharpe"])
    return {
        "bars": int(len(bars)), "variants_tested": len(VARIANTS),
        "buy_and_hold_sharpe": float(sharpe(hold[:, None], bpy)[0]),
        "variants_with_positive_net_sharpe": int((full > 0).sum()),
        "best_in_sample": [{k: b[k] for k in ("name", "net_sharpe", "net_return_pct_per_year")} for b in best[:5]],
        "best_by_regime": {name: max(((v["by_regime"][name], v["name"]) for v in variants
                                      if v["by_regime"][name] is not None), default=(None, None))
                           for name in REGIME_NAMES},
        "walk_forward": wf,
        "walk_forward_net_sharpe": float(sharpe(oos[:, None], bpy)[0]) if len(oos) else None,
        "walk_forward_net_return_pct_per_year": float(oos.mean() * bpy * 100) if len(oos) else None,
        "walk_forward_profitable_years": f"{sum(w['net_return_pct'] > 0 for w in wf)}/{len(wf)}",
        "variants": variants,
    }  # fmt: skip


def main() -> None:
    s = Settings.from_env()
    base = load(history_path(s.data_dir, 300))
    if len(base) < 100_000:
        raise SystemExit("not enough history; run: python -m app.research.history --years 10")
    daily = resample(base, 86400)
    report: dict[str, Any] = {
        "source": "Bitstamp BTC/USD", "cost_per_unit_turn": s.taker_fee, "funding_modelled": False,
        "from": time.strftime("%Y-%m-%d", time.gmtime(base[0, 0])), "to": time.strftime("%Y-%m-%d", time.gmtime(base[-1, 0])),
        "timeframes": {},
    }  # fmt: skip
    for tf, step in TIMEFRAMES.items():
        bars = base if step == 300 else resample(base, step)
        res = study(bars, bar_regimes(bars, daily), step, s.taker_fee)
        report["timeframes"][tf] = res
        print(f"\n== {tf}: {res['bars']:,} bars, {res['variants_with_positive_net_sharpe']}/{res['variants_tested']} "
              f"variants net-positive in sample, buy-and-hold Sharpe {res['buy_and_hold_sharpe']:.2f}")  # fmt: skip
        for b in res["best_in_sample"][:3]:
            print(f"   in-sample best: {b['name']:32s} Sharpe {b['net_sharpe']:5.2f}  {b['net_return_pct_per_year']:7.1f}%/yr")
        print(f"   WALK-FORWARD (honest): Sharpe {res['walk_forward_net_sharpe']:.2f}, "
              f"{res['walk_forward_net_return_pct_per_year']:.1f}%/yr, profitable years {res['walk_forward_profitable_years']}")  # fmt: skip
    out = Path(s.chart_model_dir) / "strategy_lab.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
