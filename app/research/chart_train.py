"""Train the chart model on historical candles, with an honest walk-forward evaluation first.

    python -m app.research.history --years 10     # download candles (once)
    python -m app.research.chart_train            # evaluate year by year, then train and save

Evaluation: for each calendar year (after the first two), a model is trained only on earlier
years and scored on that year. The report shows, per year, the correlation between forecast
and outcome, and how many basis points the strongest 10% of forecasts actually captured
before costs - the number to compare against the ~9 bps a taker round trip costs.

The saved model is then trained on all the data. It is used live only as an input feature.
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
from app.models.chart import FEATURES, VOL_WINDOW, chart_features
from app.research.history import history_path, load

PARAMS = {
    "objective": "huber", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 500,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0,
    "verbose": -1, "num_threads": 4, "seed": 0, "deterministic": True,
}  # fmt: skip


def make_xy(candles: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rows with complete features and a known outcome: (X, y in vol units, y in bps, timestamps)."""
    X, vol = chart_features(candles)
    lc = np.log(candles[:, 4])
    fwd = np.full(len(lc), np.nan)
    fwd[:-horizon] = lc[horizon:] - lc[:-horizon]
    y = np.clip(fwd / (vol * math.sqrt(horizon)), -10, 10)
    step = np.diff(candles[:, 0], prepend=candles[0, 0])
    contiguous = np.convolve(step != np.median(step[1:]), np.ones(VOL_WINDOW), mode="full")[: len(step)] <= 1
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y) & contiguous
    return X[ok], y[ok], fwd[ok] * 1e4, candles[ok, 0]


def fit(X: np.ndarray, y: np.ndarray, purge: int, rounds: int | None = None) -> Any:
    import lightgbm as lgb

    if rounds is not None:
        return lgb.train(PARAMS, lgb.Dataset(X, y), num_boost_round=rounds)
    cut = int(len(y) * 0.9)
    train = lgb.Dataset(X[: cut - purge], y[: cut - purge])
    valid = lgb.Dataset(X[cut:], y[cut:], reference=train)
    return lgb.train(PARAMS, train, 400, valid_sets=[valid], callbacks=[lgb.early_stopping(30, verbose=False)])


def walk_forward(X: np.ndarray, y: np.ndarray, y_bps: np.ndarray, ts: np.ndarray, horizon: int) -> list[dict[str, Any]]:
    rows = []
    year_of = ts.astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
    for yr in sorted(set(year_of.tolist()))[2:]:
        test = year_of == yr
        train_end = int(np.argmax(test)) - horizon - VOL_WINDOW
        if test.sum() < 1000 or train_end < 50_000:
            continue
        booster = fit(X[:train_end], y[:train_end], horizon + VOL_WINDOW)
        p = booster.predict(X[test])
        ic = float(np.corrcoef(p, y[test])[0, 1])
        top = np.abs(p) >= np.quantile(np.abs(p), 0.9)
        rows.append({
            "year": int(yr), "bars": int(test.sum()), "ic": ic, "t_stat": ic * math.sqrt(test.sum() / horizon),
            "hit_rate": float(np.mean(np.sign(p) == np.sign(y[test]))),
            "top_decile_gross_bps": float(np.mean(np.sign(p[top]) * y_bps[test][top])),
            "trees": int(booster.num_trees()),
        })  # fmt: skip
        print(json.dumps(rows[-1]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=1, help="bars ahead to predict")
    args = ap.parse_args()
    s = Settings.from_env()
    candles = load(history_path(s.data_dir, args.step))
    if len(candles) < 100_000:
        raise SystemExit("not enough history; run: python -m app.research.history --years 10")
    X, y, y_bps, ts = make_xy(candles, args.horizon)
    print(f"{len(y):,} usable bars, {len(FEATURES)} features")
    wf = walk_forward(X, y, y_bps, ts, args.horizon)
    final = fit(X, y, args.horizon + VOL_WINDOW)
    final = fit(X, y, 0, rounds=max(final.num_trees(), 20))  # refit on everything with the chosen size
    out = Path(s.chart_model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(str(out))
    day = "%Y-%m-%d"
    meta = {
        "features": list(FEATURES), "step_s": args.step, "horizon_bars": args.horizon, "source": "Bitstamp BTC/USD",
        "trained_from": time.strftime(day, time.gmtime(ts[0])), "trained_to": time.strftime(day, time.gmtime(ts[-1])),
        "bars": int(len(y)), "trees": int(final.num_trees()), "walk_forward": wf,
        "mean_oos_ic": float(np.mean([r["ic"] for r in wf])) if wf else None,
        "mean_top_decile_gross_bps": float(np.mean([r["top_decile_gross_bps"] for r in wf])) if wf else None,
        "importance": dict(sorted(zip(FEATURES, (int(g) for g in final.feature_importance("gain")), strict=True),
                                  key=lambda kv: -kv[1])),
    }  # fmt: skip
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"saved {out} ({meta['trees']} trees); mean out-of-sample IC {meta['mean_oos_ic']}")


if __name__ == "__main__":
    main()
