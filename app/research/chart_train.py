"""Train the chart models on historical candles, with an honest walk-forward evaluation first.

    python -m app.research.history --years 10     # download candles (once)
    python -m app.research.chart_train            # all timeframes: evaluate, then train and save

For each timeframe (5m, 1h, 4h, 1d) and each calendar year, a model is trained only on
earlier years and scored on that year. Model size is chosen on a held-out slice of the
training data, never on the test year. The report gives, per year and per market regime
(bull/bear/range x high/low volatility):

- ic: correlation between forecast and outcome;
- top_decile_gross_bps: what the strongest 10% of forecasts captured before costs;
- top_decile_net_bps: the same after a taker round trip. This is the number that matters.

The saved models are then trained on all the data and used live only as input features.
Writes models/chart_<timeframe>.txt/.json.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.config.settings import Settings
from app.models.chart import FEATURES, MIN_BARS, TIMEFRAMES, chart_features
from app.research.history import history_path, load
from app.research.regimes import REGIME_NAMES, bar_regimes, resample

BASE = {
    "objective": "huber", "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "num_threads": 4, "seed": 0, "deterministic": True,
}  # fmt: skip
LEAVES = (7, 31)


def make_xy(candles: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rows with complete features and a known outcome: (X, y in vol units, y in bps, row index into candles)."""
    X, vol = chart_features(candles)
    lc = np.log(candles[:, 4])
    fwd = np.full(len(lc), np.nan)
    fwd[:-horizon] = lc[horizon:] - lc[:-horizon]
    y = np.clip(fwd / (vol * math.sqrt(horizon)), -10, 10)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    ok[:MIN_BARS] = False
    return X[ok], y[ok], fwd[ok] * 1e4, np.flatnonzero(ok)


def fit(X: np.ndarray, y: np.ndarray, purge: int) -> tuple[Any, dict[str, Any]]:
    """Choose tree size and count on the last 10% of the training data; returns (booster, chosen settings)."""
    import lightgbm as lgb

    cut = int(len(y) * 0.9)
    train = lgb.Dataset(X[: max(cut - purge, 1)], y[: max(cut - purge, 1)])
    best: tuple[float, Any, dict[str, Any]] | None = None
    for leaves in LEAVES:
        params = BASE | {"num_leaves": leaves, "min_data_in_leaf": int(min(max(len(y) / 2000, 20), 500))}
        b = lgb.train(params, train, 400, valid_sets=[lgb.Dataset(X[cut:], y[cut:], reference=train)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])  # fmt: skip
        loss = float(np.mean((y[cut:] - b.predict(X[cut:])) ** 2))
        if best is None or loss < best[0]:
            best = (loss, b, params | {"rounds": max(int(b.best_iteration or b.num_trees()), 10)})
    assert best is not None
    return best[1], best[2]


def _score(p: np.ndarray, y: np.ndarray, y_bps: np.ndarray, horizon: int, cost_bps: float) -> dict[str, Any]:
    if len(p) < 50 or p.std() == 0:
        return {"bars": int(len(p)), "ic": 0.0, "t_stat": 0.0, "top_decile_gross_bps": 0.0,
                "top_decile_net_bps": -cost_bps}  # fmt: skip
    ic = float(np.corrcoef(p, y)[0, 1])
    top = np.abs(p) >= np.quantile(np.abs(p), 0.9)
    gross = float(np.mean(np.sign(p[top]) * y_bps[top]))
    return {"bars": int(len(p)), "ic": ic, "t_stat": ic * math.sqrt(len(p) / horizon),
            "top_decile_gross_bps": gross, "top_decile_net_bps": gross - cost_bps}  # fmt: skip


def train_timeframe(tf: str, bars: np.ndarray, daily: np.ndarray, horizon: int, cost_bps: float) -> tuple[Any, dict[str, Any]]:
    X, y, y_bps, rows = make_xy(bars, horizon)
    ts = bars[rows, 0]
    regime = bar_regimes(bars[rows], daily)
    year = ts.astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
    purge = horizon + 5
    wf, oos_p, oos_idx = [], [], []
    for yr in sorted(set(year.tolist())):
        test = np.flatnonzero(year == yr)
        train_end = int(test[0]) - purge
        if len(test) < 50 or train_end < 500:
            continue
        booster, _ = fit(X[:train_end], y[:train_end], purge)
        p = booster.predict(X[test])
        wf.append({"year": int(yr)} | _score(p, y[test], y_bps[test], horizon, cost_bps))
        oos_p.append(p)
        oos_idx.append(test)
    P, idx = (np.concatenate(oos_p), np.concatenate(oos_idx)) if wf else (np.empty(0), np.empty(0, dtype=int))
    overall = _score(P, y[idx], y_bps[idx], horizon, cost_bps)
    by_regime = {name: _score(P[regime[idx] == k], y[idx][regime[idx] == k], y_bps[idx][regime[idx] == k], horizon, cost_bps)
                 for k, name in enumerate(REGIME_NAMES)}  # fmt: skip

    _, chosen = fit(X, y, purge)
    import lightgbm as lgb

    final = lgb.train({k: v for k, v in chosen.items() if k != "rounds"}, lgb.Dataset(X, y), chosen["rounds"])
    gain = final.feature_importance("gain")
    day = "%Y-%m-%d"
    meta = {
        "timeframe": tf, "features": list(FEATURES), "horizon_bars": horizon, "source": "Bitstamp BTC/USD",
        "trained_from": time.strftime(day, time.gmtime(ts[0])), "trained_to": time.strftime(day, time.gmtime(ts[-1])),
        "bars": int(len(y)), "trees": int(final.num_trees()), "num_leaves": chosen["num_leaves"],
        "cost_bps_round_trip": cost_bps, "out_of_sample": overall, "by_regime": by_regime, "walk_forward": wf,
        "profitable_years_after_costs": f"{sum(w['top_decile_net_bps'] > 0 for w in wf)}/{len(wf)}",
        "top_features": dict(sorted(zip(FEATURES, (int(g) for g in gain), strict=True), key=lambda kv: -kv[1])[:12]),
    }  # fmt: skip
    return final, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeframes", default=",".join(TIMEFRAMES))
    ap.add_argument("--horizon", type=int, default=1, help="bars ahead to predict")
    args = ap.parse_args()
    s = Settings.from_env()
    base = load(history_path(s.data_dir, 300))
    if len(base) < 100_000:
        raise SystemExit("not enough history; run: python -m app.research.history --years 10")
    daily = resample(base, 86400)
    out_dir = Path(s.chart_model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cost_bps = 2 * s.taker_fee * 1e4
    for tf in args.timeframes.split(","):
        step = TIMEFRAMES[tf]
        bars = base if step == 300 else resample(base, step)
        booster, meta = train_timeframe(tf, bars, daily, args.horizon, cost_bps)
        booster.save_model(str(out_dir / f"chart_{tf}.txt"))
        (out_dir / f"chart_{tf}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        o = meta["out_of_sample"]
        print(f"\n== {tf}: {meta['bars']:,} bars, {meta['trees']} trees. Out of sample: IC {o['ic']:.4f} "
              f"(t {o['t_stat']:.1f}), top decile {o['top_decile_gross_bps']:.1f} bps gross, "
              f"{o['top_decile_net_bps']:.1f} net; years net-positive {meta['profitable_years_after_costs']}")  # fmt: skip
        for w in meta["walk_forward"]:
            print(f"   {w['year']}: IC {w['ic']:7.4f}   top decile gross {w['top_decile_gross_bps']:7.1f} bps   "
                  f"net {w['top_decile_net_bps']:7.1f} bps")  # fmt: skip
        for name, r in meta["by_regime"].items():
            print(f"   {name:15s} IC {r['ic']:7.4f}   net {r['top_decile_net_bps']:7.1f} bps   ({r['bars']:,} bars)")


if __name__ == "__main__":
    main()
