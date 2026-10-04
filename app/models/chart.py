"""Chart models: what ten years of BTC candles say about the next bar, per timeframe.

One LightGBM model per timeframe (5m, 1h, 4h, 1d), trained offline by
`python -m app.research.chart_train`. Inputs are scale-free chart features from the
indicator library plus the current position of every classic strategy variant
(app/strategies/library.py), so the model can learn in which conditions each strategy has
worked. Live, each model scores its latest closed candle and the result reaches the trading
models as one feature per timeframe ("chart_5m" ... "chart_1d"). None of them has authority
of its own; like every feature they matter only if they prove predictive out of sample.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from app.indicators import library as ind
from app.strategies.library import NAMES as STRATEGY_NAMES
from app.strategies.library import all_signals

TIMEFRAMES = {"5m": 300, "1h": 3600, "4h": 14400, "1d": 86400}
VOL_WINDOW = 288
MIN_BARS = 3 * VOL_WINDOW  # history needed before a score is meaningful
RET_LAGS = (1, 3, 12, 48, 288)
CHART_FEATURES = (
    *(f"ret_{k}" for k in RET_LAGS),
    "rsi_14", "rsi_48", "macd_hist", "boll_20", "boll_96", "donch_48", "donch_288", "adx_14", "atr_14",
    "eff_48", "stoch_14", "rel_volume", "vol_ratio", "body", "upper_wick", "lower_wick", "hour_sin", "hour_cos",
)  # fmt: skip
FEATURES = CHART_FEATURES + STRATEGY_NAMES


def _shift_diff(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    out[k:] = x[k:] - x[:-k]
    return out


def chart_features(candles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(features, per-bar volatility). Row t uses only bars up to and including t."""
    ts, o, h, lo, c, v = candles.T
    lc = np.log(c)
    lr = np.diff(lc, prepend=lc[0])
    vol = np.maximum(ind.rolling_std(lr, VOL_WINDOW), 1e-6)
    rng = np.maximum(h - lo, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        cols = [_shift_diff(lc, k) / (vol * math.sqrt(k)) for k in RET_LAGS]
        cols += [(ind.rsi(c, 14) - 50) / 50, (ind.rsi(c, 48) - 50) / 50]
        cols.append(ind.macd(c)[2] / c / vol)
        for n in (20, 96):
            cols.append(ind.zscore(c, n))
        for n in (48, VOL_WINDOW):
            dl, dh = ind.donchian(h, lo, n)
            cols.append(np.where(dh > dl, (c - dl) / (dh - dl) - 0.5, 0.0))
        cols.append(ind.adx(h, lo, c, 14) / 100)
        cols.append(ind.atr(h, lo, c, 14) / c / vol)
        cols.append(np.abs(_shift_diff(c, 48)) / np.maximum(ind.sma(np.abs(np.diff(c, prepend=c[0])), 48) * 48, 1e-12))
        cols.append(ind.stochastic(h, lo, c, 14) / 100 - 0.5)
        rel_v = v / np.maximum(ind.sma(v, VOL_WINDOW), 1e-12)  # relative to its own mean: venue-independent
        cols.append(np.log(rel_v + 0.01))
        cols.append(ind.rolling_std(lr, 12) / vol)
        cols += [(c - o) / rng, (h - np.maximum(o, c)) / rng, (np.minimum(o, c) - lo) / rng]
        hour = (ts % 86400) / 86400 * 2 * math.pi
        cols += [np.sin(hour), np.cos(hour)]
    return np.column_stack([np.column_stack(cols), all_signals(candles)]), vol


class ChartModel:
    def __init__(self, booster: Any, meta: dict[str, Any]) -> None:
        self.booster = booster
        self.meta = meta

    @classmethod
    def load(cls, model_dir: str, timeframe: str) -> ChartModel | None:
        p = Path(model_dir) / f"chart_{timeframe}.txt"
        meta_p = p.with_suffix(".json")
        if not p.is_file() or not meta_p.is_file():
            return None
        import lightgbm as lgb

        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if tuple(meta.get("features", ())) != FEATURES:
            return None  # trained against a different feature set
        return cls(lgb.Booster(model_file=str(p)), meta)

    def score(self, candles: np.ndarray) -> float | None:
        """Forecast for the bar after the last one, in volatility units, clipped to [-5, 5]."""
        if len(candles) < MIN_BARS:
            return None
        X, _ = chart_features(candles[-(MIN_BARS + 200) :])
        if not np.isfinite(X[-1]).all():
            return None
        return float(np.clip(self.booster.predict(X[-1:])[0], -5.0, 5.0))
