"""Train the trade-flow model on historical tick trades, walk-forward first.

    python -m app.research.ticks --days 90      # download and featurise (cached per day)
    python -m app.research.flow_train

Evaluation: after an initial 30 days, each following week is scored by a model trained only
on the days before it. Per week and overall the report gives:

- ic: correlation between the forecast and the realised 60-second return;
- top_decile_gross_bps: what the strongest 10% of forecasts captured before costs;
- share_above_taker_cost: how often the forecast itself exceeded a taker round trip, and
  realised_when_above: what those moments actually returned. If the model rarely predicts
  more than costs, or is wrong when it does, it cannot be traded with taker orders.

The saved model (models/flow_<horizon>s.txt) is trained on all days and joins the live arena
as a challenger with zero trust.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from app.config.settings import Settings
from app.models.flow import PORTABLE
from app.research.ticks import load_features

PARAMS = {
    "objective": "huber", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 2000,
    "feature_fraction": 0.9, "bagging_fraction": 0.3, "bagging_freq": 1, "lambda_l2": 10.0,
    "verbose": -1, "num_threads": 8, "seed": 0, "deterministic": True,
}  # fmt: skip
MIN_TRAIN_DAYS = 30
TEST_DAYS = 7


def fit(X: np.ndarray, y: np.ndarray, purge: int, rounds: int | None = None) -> Any:
    import lightgbm as lgb

    yc = np.clip(y, -100, 100)
    if rounds is not None:
        return lgb.train(PARAMS, lgb.Dataset(X, yc), rounds)
    cut = int(len(y) * 0.9)
    train = lgb.Dataset(X[: cut - purge], yc[: cut - purge])
    valid = lgb.Dataset(X[cut:], yc[cut:], reference=train)
    return lgb.train(PARAMS, train, 300, valid_sets=[valid], callbacks=[lgb.early_stopping(20, verbose=False)])


def score(p: np.ndarray, y: np.ndarray, horizon: int, taker_bps: float, maker_bps: float) -> dict[str, Any]:
    ic = float(np.corrcoef(p, y)[0, 1]) if p.std() > 0 else 0.0
    top = np.abs(p) >= np.quantile(np.abs(p), 0.9)
    gross = float(np.mean(np.sign(p[top]) * y[top]))
    above = np.abs(p) > taker_bps
    return {
        "rows": int(len(p)), "ic": ic, "t_stat": ic * math.sqrt(len(p) / horizon),
        "top_decile_gross_bps": gross, "top_decile_net_taker_bps": gross - taker_bps,
        "top_decile_net_maker_bps": gross - maker_bps,
        "forecast_p99_abs_bps": float(np.quantile(np.abs(p), 0.99)),
        "share_above_taker_cost": float(above.mean()),
        "realised_when_above_bps": float(np.mean(np.sign(p[above]) * y[above])) if above.any() else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, default=None, help="seconds; default HL_HORIZON_S")
    args = ap.parse_args()
    s = Settings.from_env()
    horizon = args.horizon or round(s.horizon_s)
    X, y, day, names = load_features(s.data_dir, horizon)
    n_days = len(names)
    if n_days < MIN_TRAIN_DAYS + TEST_DAYS:
        raise SystemExit(f"only {n_days} days of tick features; run: python -m app.research.ticks --days 90")
    taker, maker = 2 * s.taker_fee * 1e4, 2 * s.maker_fee * 1e4
    print(f"{len(y):,} rows over {n_days} days ({names[0]} to {names[-1]}), {len(PORTABLE)} features")

    weeks, oos_p, oos_y = [], [], []
    for d0 in range(MIN_TRAIN_DAYS, n_days, TEST_DAYS):
        tr = day < d0
        te = (day >= d0) & (day < d0 + TEST_DAYS)
        booster = fit(X[tr], y[tr], purge=2 * horizon)
        p = booster.predict(X[te])
        oos_p.append(p)
        oos_y.append(y[te])
        w = {"from": names[d0], "to": names[min(d0 + TEST_DAYS, n_days) - 1]} | score(p, y[te], horizon, taker, maker)
        weeks.append(w)
        print(f"   {w['from']}..{w['to']}: IC {w['ic']:7.4f}  top decile gross {w['top_decile_gross_bps']:5.2f} bps  "
              f"99th pct forecast {w['forecast_p99_abs_bps']:5.2f} bps")  # fmt: skip
    overall = score(np.concatenate(oos_p), np.concatenate(oos_y), horizon, taker, maker)

    probe = fit(X, y, purge=2 * horizon)
    final = fit(X, y, 0, rounds=max(int(probe.best_iteration or probe.num_trees()), 20))
    out = Path(s.chart_model_dir) / f"flow_{horizon}s.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(str(out))
    gain = final.feature_importance("gain")
    meta = {
        "features": list(PORTABLE), "horizon_s": horizon, "source": "Binance USD-M BTCUSDT aggTrades",
        "days": n_days, "from": names[0], "to": names[-1], "rows": int(len(y)), "trees": int(final.num_trees()),
        "cost_bps_round_trip": {"taker": taker, "maker": maker}, "out_of_sample": overall,
        "weeks_with_positive_ic": f"{sum(w['ic'] > 0 for w in weeks)}/{len(weeks)}", "walk_forward": weeks,
        "importance": dict(sorted(zip(PORTABLE, (int(g) for g in gain), strict=True), key=lambda kv: -kv[1])),
    }  # fmt: skip
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    o = overall
    print(f"\nOut of sample: IC {o['ic']:.4f} (t {o['t_stat']:.1f}), weeks positive {meta['weeks_with_positive_ic']}\n"
          f"  strongest 10%: {o['top_decile_gross_bps']:.2f} bps gross, {o['top_decile_net_taker_bps']:.2f} after taker costs, "
          f"{o['top_decile_net_maker_bps']:.2f} after maker costs\n"
          f"  forecast exceeded taker cost {o['share_above_taker_cost'] * 100:.3f}% of the time; realised then: "
          f"{o['realised_when_above_bps']} bps\nsaved {out} ({meta['trees']} trees)")  # fmt: skip


if __name__ == "__main__":
    main()
