"""Market-condition labels and timeframe resampling for historical research.

Every label for a bar is computed from days that had already closed before that bar, so
conditioning on a regime never uses the future.
"""

from __future__ import annotations

import numpy as np

TIMEFRAMES = {"5m": 300, "1h": 3600, "4h": 14400, "1d": 86400}
TRENDS = ("bull", "bear", "range")
VOLS = ("high_vol", "low_vol")
REGIME_NAMES = tuple(f"{t}/{v}" for t in TRENDS for v in VOLS)


def resample(candles: np.ndarray, step: int) -> np.ndarray:
    """Aggregate (ts, o, h, l, c, v) candles into bars of `step` seconds. Drops a partial last bar."""
    ts = candles[:, 0]
    bucket = (ts // step).astype(np.int64)
    starts = np.flatnonzero(np.diff(bucket, prepend=bucket[0] - 1))
    base = float(np.median(np.diff(ts[:1000]))) if len(ts) > 1 else step
    full = np.diff(np.append(starts, len(ts))) >= round(step / base)
    out = np.column_stack([
        bucket[starts] * step,
        candles[starts, 1],
        np.maximum.reduceat(candles[:, 2], starts),
        np.minimum.reduceat(candles[:, 3], starts),
        candles[np.append(starts[1:], len(ts)) - 1, 4],
        np.add.reduceat(candles[:, 5], starts),
    ])  # fmt: skip
    return out[full]


def daily_regimes(daily: np.ndarray, trend_days: int = 30, trend_pct: float = 0.10, vol_days: int = 7) -> np.ndarray:
    """Regime index (into REGIME_NAMES) for each *daily* bar, from that day's close and earlier."""
    c = daily[:, 4]
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    n = len(c)
    trend = np.full(n, 2)  # range
    ret = np.full(n, np.nan)
    ret[trend_days:] = c[trend_days:] / c[:-trend_days] - 1
    trend[ret > trend_pct] = 0
    trend[ret < -trend_pct] = 1
    win = np.lib.stride_tricks.sliding_window_view(lr, vol_days)
    rv = np.concatenate([np.full(vol_days - 1, np.nan), win.std(axis=1)])
    vol = np.ones(n, dtype=int)  # low
    for i in range(vol_days, n):  # compare with the median of the trailing year, known at day i
        past = rv[max(vol_days, i - 365) : i]
        if len(past) and rv[i] > np.nanmedian(past):
            vol[i] = 0
    return trend * 2 + vol


def bar_regimes(bars: np.ndarray, daily: np.ndarray) -> np.ndarray:
    """Regime for each bar = the label of the last day that had fully closed before the bar opened."""
    labels = daily_regimes(daily)
    day_close = daily[:, 0] + 86400
    idx = np.searchsorted(day_close, bars[:, 0], side="right") - 1
    return np.where(idx >= 0, labels[np.maximum(idx, 0)], REGIME_NAMES.index("range/low_vol"))
