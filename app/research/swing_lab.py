"""Swing lab: would a slower lane on the 1-hour chart signal, executed with maker orders, pay?

    python -m app.research.swing_lab

Every hour the 1h chart model forecasts the next hour. Two policies are tested:

- flat_when_weak: long or short one unit while the forecast is strong, flat otherwise. This
  is the pure 1-hour signal.
- hold_until_flip: take a position on a strong forecast and keep it until a strong forecast
  in the other direction. Far fewer trades, but it then mostly holds BTC for days, so its
  result is largely market exposure rather than the signal (see `avg_position`).

Everything is walk-forward: the forecast for each year comes from a model trained only on
earlier years, and the "strong" threshold for each year is chosen from results on earlier
years only (the first forecast year only sets thresholds and is not reported).

Execution is simulated on the 5-minute bars inside each hour:

- maker: a limit order rests at the hour's closing price for up to 15 minutes and fills only
  if the market trades clearly *through* it (by 1 bp), which is exactly when price is moving
  against the order, so adverse selection is built in. An unfilled entry is abandoned; an
  unfilled exit is forced out at market.
- taker (for comparison): always fills, pays the taker fee plus 1 bp slippage.

Not modelled: funding payments, queue position, and that Hyperliquid's book is not
Bitstamp's. Returns are per unit of notional, summed without compounding. Writes
models/swing_lab.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from app.config.settings import Settings
from app.research.chart_train import make_xy, walk_forward_oos, year_of
from app.research.history import history_path, load
from app.research.regimes import resample

HOURS_PER_YEAR = 8766.0
QUANTILES = (0.0, 0.5, 0.7, 0.9, 0.97)
POLICIES = ("flat_when_weak", "hold_until_flip")
TTL_BARS = 3  # 15 minutes
THROUGH = 1e-4  # a resting order fills only if price trades 1 bp through it
SLIP = 1e-4  # taker slippage


def targets(p: np.ndarray, thr: float, policy: str) -> np.ndarray:
    strong = np.where(np.abs(p) >= thr, np.sign(p), np.nan)
    if policy == "flat_when_weak":
        return np.nan_to_num(strong)
    idx = np.where(np.isnan(strong), 0, np.arange(len(strong)))
    np.maximum.accumulate(idx, out=idx)
    return np.nan_to_num(strong[idx])


def simulate(
    target: np.ndarray, close: np.ndarray, sub_hi: np.ndarray, sub_lo: np.ndarray, sub_close: np.ndarray,
    maker: bool, maker_fee: float, taker_fee: float,
) -> dict[str, Any]:  # fmt: skip
    """Hour by hour. target[i] is decided at close[i]; sub_* are the 5m bars of the following hour (n, 12)."""
    n = len(target) - 1
    ret = np.zeros(n)
    held = np.zeros(n)
    attempts = fills = 0
    pos = 0.0
    for i in range(n):
        r0, r1 = close[i], close[i + 1]
        delta = target[i] - pos
        new, px, fee = pos, r0, 0.0
        if delta != 0:
            side = 1.0 if delta > 0 else -1.0
            if maker:
                attempts += 1
                if side > 0:
                    hit = bool((sub_lo[i, :TTL_BARS] < r0 * (1 - THROUGH)).any())
                else:
                    hit = bool((sub_hi[i, :TTL_BARS] > r0 * (1 + THROUGH)).any())
                if hit:
                    fills += 1
                    new, fee = target[i], maker_fee * abs(delta)
                elif pos != 0 and side != np.sign(pos):  # unfilled exit: get out at market after the wait
                    qty = min(abs(delta), abs(pos))
                    new, fee = pos + side * qty, taker_fee * qty
                    px = sub_close[i, TTL_BARS - 1] * (1 + side * SLIP)
            else:
                new, fee, px = target[i], taker_fee * abs(delta), r0 * (1 + side * SLIP)
        ret[i] = pos * (px - r0) / r0 + new * (r1 - px) / r0 - fee
        held[i] = new
        pos = new
    return {"ret": ret, "pos": held, "attempts": attempts, "fills": fills}


def stats(ret: np.ndarray, pos: np.ndarray) -> dict[str, float]:
    if len(ret) == 0:
        return {}
    eq = np.cumsum(ret)
    sd = ret.std()
    return {
        "net_return_pct_per_year": float(ret.mean() * HOURS_PER_YEAR * 100),
        "sharpe": float(ret.mean() / sd * math.sqrt(HOURS_PER_YEAR)) if sd > 0 else 0.0,
        "max_drawdown_pct": float((np.maximum.accumulate(eq) - eq).max() * 100),
        "hours_in_market_pct": float((pos != 0).mean() * 100),
        "avg_position": float(pos.mean()),
        "position_changes_per_year": float(np.abs(np.diff(pos, prepend=0)).sum() / len(ret) * HOURS_PER_YEAR),
    }


def run(base: np.ndarray, maker_fee: float, taker_fee: float) -> dict[str, Any]:
    bars = resample(base, 3600)
    X, y, _, rows = make_xy(bars, 1)
    folds = walk_forward_oos(X, y, year_of(bars[rows, 0]), 1)
    P = np.full(len(bars), np.nan)
    for _, test, p in folds:
        P[rows[test]] = p

    # 5-minute bars of the hour after each hourly close.
    start = np.searchsorted(base[:, 0], bars[:, 0] + 3600)
    sub = np.minimum(start[:, None] + np.arange(12)[None, :], len(base) - 1)
    hi, lo, cl = base[sub, 2], base[sub, 3], base[sub, 4]
    close, year = bars[:, 4], year_of(bars[:, 0])
    valid = np.isfinite(P) & (start + 12 <= len(base))
    years = sorted(set(year[valid].tolist()))

    def sim_year(yr: int, thr: float, policy: str, maker: bool) -> dict[str, Any]:
        m = np.flatnonzero(valid & (year == yr))
        sl = slice(int(m[0]), int(m[-1]) + 1)
        return simulate(targets(P[sl], thr, policy), close[sl], hi[sl], lo[sl], cl[sl], maker, maker_fee, taker_fee)

    out: dict[str, Any] = {}
    for policy in POLICIES:
        per_year: list[dict[str, Any]] = []
        acc: dict[str, list[np.ndarray]] = {"mr": [], "mp": [], "tr": [], "tp": []}
        attempts = fills = 0
        history: dict[float, list[np.ndarray]] = {q: [] for q in QUANTILES}
        for k, yr in enumerate(years):
            ref = valid & ((year == yr) if k == 0 else (year < yr))
            thr = {q: float(np.quantile(np.abs(P[ref]), q)) for q in QUANTILES}
            results = {q: sim_year(yr, thr[q], policy, True) for q in QUANTILES}
            if k > 0:  # choose the threshold from earlier years only

                def past_sharpe(q: float, history: dict[float, list[np.ndarray]] = history) -> float:
                    r = np.concatenate(history[q])
                    return float(r.mean() / r.std()) if r.std() > 0 else -1e9

                q = max(QUANTILES, key=past_sharpe)
                res, tk = results[q], sim_year(yr, thr[q], policy, False)
                acc["mr"].append(res["ret"])
                acc["mp"].append(res["pos"])
                acc["tr"].append(tk["ret"])
                acc["tp"].append(tk["pos"])
                attempts += res["attempts"]
                fills += res["fills"]
                c = close[valid & (year == yr)]
                per_year.append({"year": int(yr), "threshold_quantile": q,
                                 "maker_net_return_pct": float(res["ret"].sum() * 100),
                                 "taker_net_return_pct": float(tk["ret"].sum() * 100),
                                 "buy_and_hold_pct": float(c[-1] / c[0] - 1) * 100})  # fmt: skip
            for q2 in QUANTILES:
                history[q2].append(results[q2]["ret"])
        mr, mp, tr, tp = (np.concatenate(acc[k]) for k in ("mr", "mp", "tr", "tp"))
        recent = [p["maker_net_return_pct"] for p in per_year[-3:]]
        out[policy] = {
            "maker": stats(mr, mp) | {"fill_rate": fills / max(attempts, 1),
                                      "profitable_years": f"{sum(p['maker_net_return_pct'] > 0 for p in per_year)}/{len(per_year)}",
                                      "last_three_years_pct": recent},
            "taker": stats(tr, tp) | {"profitable_years": f"{sum(p['taker_net_return_pct'] > 0 for p in per_year)}/{len(per_year)}"},
            "per_year": per_year,
        }  # fmt: skip
    hold_r = np.diff(np.log(close[valid & (year > years[0])]))
    return {
        "signal": "chart_1h, walk-forward", "maker_fee": maker_fee, "taker_fee": taker_fee, "funding_modelled": False,
        "buy_and_hold": {"sharpe": float(hold_r.mean() / hold_r.std() * math.sqrt(HOURS_PER_YEAR)),
                         "net_return_pct_per_year": float(hold_r.mean() * HOURS_PER_YEAR * 100)},
        "policies": out,
    }  # fmt: skip


def main() -> None:
    s = Settings.from_env()
    base = load(history_path(s.data_dir, 300))
    if len(base) < 100_000:
        raise SystemExit("not enough history; run: python -m app.research.history --years 10")
    r = run(base, s.maker_fee, s.taker_fee)
    for policy, res in r["policies"].items():
        print(f"\n== {policy}")
        for kind in ("maker", "taker"):
            m = res[kind]
            print(f"   {kind:5s}: {m['net_return_pct_per_year']:7.1f}%/yr  Sharpe {m['sharpe']:5.2f}  max DD {m['max_drawdown_pct']:6.1f}%  "
                  f"avg position {m['avg_position']:5.2f}  changes/yr {m['position_changes_per_year']:5.0f}  "
                  f"profitable years {m['profitable_years']}" + (f"  fill rate {m['fill_rate']:.2f}" if "fill_rate" in m else ""))  # fmt: skip
        for p in res["per_year"]:
            print(f"      {p['year']}: maker {p['maker_net_return_pct']:7.1f}%   taker {p['taker_net_return_pct']:7.1f}%   "
                  f"hold {p['buy_and_hold_pct']:7.1f}%   (threshold q={p['threshold_quantile']})")  # fmt: skip
    b = r["buy_and_hold"]
    print(f"\nbuy & hold: {b['net_return_pct_per_year']:.1f}%/yr  Sharpe {b['sharpe']:.2f}")
    out = Path(s.chart_model_dir) / "swing_lab.json"
    out.write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
