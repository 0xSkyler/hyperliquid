"""Technical indicators on NumPy arrays. Each returns an array aligned with its input
(NaN where undefined). They are inputs for feature research, never trade rules on their own."""

from __future__ import annotations

import numpy as np

Arr = np.ndarray


def sma(x: Arr, n: int) -> Arr:
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1 :] = (c[n:] - c[:-n]) / n
    return out


def ema(x: Arr, n: int) -> Arr:
    out = np.empty(len(x))
    a = 2.0 / (n + 1.0)
    acc = x[0]
    for i, v in enumerate(x):
        acc = a * v + (1 - a) * acc
        out[i] = acc
    return out


def wma(x: Arr, n: int) -> Arr:
    out = np.full(len(x), np.nan)
    w = np.arange(1, n + 1, dtype=float)
    if len(x) >= n:
        out[n - 1 :] = np.convolve(x, w[::-1], mode="valid") / w.sum()
    return out


def hma(x: Arr, n: int) -> Arr:
    raw = 2 * wma(x, max(n // 2, 1)) - wma(x, n)
    k = max(int(np.sqrt(n)), 1)
    out = np.full(len(x), np.nan)
    ok = ~np.isnan(raw)
    out[ok] = wma(raw[ok], k)
    return out


def _wilder(x: Arr, n: int) -> Arr:
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    acc = x[:n].mean()
    out[n - 1] = acc
    for i in range(n, len(x)):
        acc = (acc * (n - 1) + x[i]) / n
        out[i] = acc
    return out


def rsi(close: Arr, n: int = 14) -> Arr:
    d = np.diff(close)
    up, dn = _wilder(np.maximum(d, 0), n), _wilder(np.maximum(-d, 0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + up / dn)
    r[(dn == 0) & (up > 0)] = 100.0
    r[(dn == 0) & (up == 0)] = 50.0
    return np.insert(r, 0, np.nan)


def macd(close: Arr, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[Arr, Arr, Arr]:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def true_range(high: Arr, low: Arr, close: Arr) -> Arr:
    prev = np.insert(close[:-1], 0, close[0])
    return np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))


def atr(high: Arr, low: Arr, close: Arr, n: int = 14) -> Arr:
    return _wilder(true_range(high, low, close), n)


def rolling_std(x: Arr, n: int) -> Arr:
    m = sma(x, n)
    m2 = sma(x * x, n)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def bollinger(close: Arr, n: int = 20, k: float = 2.0) -> tuple[Arr, Arr, Arr]:
    m, s = sma(close, n), rolling_std(close, n)
    return m - k * s, m, m + k * s


def zscore(x: Arr, n: int) -> Arr:
    s = rolling_std(x, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s > 0, (x - sma(x, n)) / s, 0.0)


def roc(x: Arr, n: int) -> Arr:
    out = np.full(len(x), np.nan)
    out[n:] = x[n:] / x[:-n] - 1.0
    return out


def _rolling(x: Arr, n: int, fn) -> Arr:
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1 :] = fn(np.lib.stride_tricks.sliding_window_view(x, n), axis=1)
    return out


def donchian(high: Arr, low: Arr, n: int = 20) -> tuple[Arr, Arr]:
    return _rolling(low, n, np.min), _rolling(high, n, np.max)


def stochastic(high: Arr, low: Arr, close: Arr, n: int = 14) -> Arr:
    lo, hi = donchian(high, low, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(hi > lo, 100 * (close - lo) / (hi - lo), 50.0)


def williams_r(high: Arr, low: Arr, close: Arr, n: int = 14) -> Arr:
    return stochastic(high, low, close, n) - 100.0


def cci(high: Arr, low: Arr, close: Arr, n: int = 20) -> Arr:
    tp = (high + low + close) / 3
    md = _rolling(tp, n, lambda w, axis: np.mean(np.abs(w - w.mean(axis=axis, keepdims=True)), axis=axis))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(md > 0, (tp - sma(tp, n)) / (0.015 * md), 0.0)


def adx(high: Arr, low: Arr, close: Arr, n: int = 14) -> Arr:
    up, dn = np.diff(high, prepend=high[0]), -np.diff(low, prepend=low[0])
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = _wilder(true_range(high, low, close), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi, mdi = 100 * _wilder(plus, n) / tr, 100 * _wilder(minus, n) / tr
        dx = np.where(pdi + mdi > 0, 100 * np.abs(pdi - mdi) / (pdi + mdi), 0.0)
    out = np.full(len(close), np.nan)
    ok = ~np.isnan(dx)
    out[ok] = _wilder(dx[ok], n)
    return out


def obv(close: Arr, volume: Arr) -> Arr:
    return np.cumsum(np.sign(np.diff(close, prepend=close[0])) * volume)


def vwap(price: Arr, volume: Arr) -> Arr:
    v = np.cumsum(volume)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, np.cumsum(price * volume) / v, price)


def mfi(high: Arr, low: Arr, close: Arr, volume: Arr, n: int = 14) -> Arr:
    tp = (high + low + close) / 3
    d = np.diff(tp, prepend=tp[0])
    pos, neg = sma(np.where(d > 0, tp * volume, 0.0), n), sma(np.where(d < 0, tp * volume, 0.0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(neg > 0, 100 - 100 / (1 + pos / neg), 100.0)


def efficiency_ratio(x: Arr) -> float:
    """Kaufman efficiency: |net move| / path length. ~1 trending, ~0 noise."""
    path = float(np.abs(np.diff(x)).sum())
    return abs(float(x[-1] - x[0])) / path if path > 0 else 0.0
