"""Classic strategy families as position signals on candles.

Each variant maps candles to a target position in [-1, 1] per bar, decided at that bar's
close using only data up to it. They are studied in `app.research.strategy_lab` and fed to
the chart models as inputs, so the models can learn when (if ever) each one works. None of
them trades directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.indicators import library as ind

Arr = np.ndarray


@dataclass(frozen=True)
class Variant:
    family: str
    name: str
    fn: Callable[[Arr, Arr, Arr, Arr, Arr], Arr]  # (open, high, low, close, volume) -> position


def _hold(raw: Arr) -> Arr:
    """Forward-fill: NaN means 'keep the previous position'; starts flat."""
    idx = np.where(np.isnan(raw), 0, np.arange(len(raw)))
    np.maximum.accumulate(idx, out=idx)
    out = raw[idx]
    return np.where(np.isnan(out), 0.0, out)


def _prior(x: Arr) -> Arr:
    return np.concatenate([[np.nan], x[:-1]])


def ma_cross(fast: int, slow: int) -> Callable[..., Arr]:
    return lambda o, h, lo, c, v: np.sign(ind.ema(c, fast) - ind.ema(c, slow))


def tsmom(look: int) -> Callable[..., Arr]:
    return lambda o, h, lo, c, v: np.nan_to_num(np.sign(ind.roc(c, look)))


def macd_trend(fast: int, slow: int) -> Callable[..., Arr]:
    return lambda o, h, lo, c, v: np.sign(ind.macd(c, fast, slow, 9)[2])


def donchian_breakout(n: int) -> Callable[..., Arr]:
    def f(o: Arr, h: Arr, lo: Arr, c: Arr, v: Arr) -> Arr:
        dl, dh = ind.donchian(h, lo, n)
        raw = np.where(c > _prior(dh), 1.0, np.where(c < _prior(dl), -1.0, np.nan))
        return _hold(raw)

    return f


def breakout_fade(n: int, bars: int = 3) -> Callable[..., Arr]:
    """False-breakout: fade a fresh n-bar breakout for a few bars."""

    def f(o: Arr, h: Arr, lo: Arr, c: Arr, v: Arr) -> Arr:
        dl, dh = ind.donchian(h, lo, n)
        brk = np.where(c > _prior(dh), -1.0, np.where(c < _prior(dl), 1.0, 0.0))
        out = np.zeros(len(c))
        for k in range(bars):
            out[k:] += brk[: len(c) - k] if k else brk
        return np.clip(out, -1, 1)

    return f


def keltner_breakout(n: int, k: float) -> Callable[..., Arr]:
    """Volatility breakout: outside an ATR channel around the mean."""

    def f(o: Arr, h: Arr, lo: Arr, c: Arr, v: Arr) -> Arr:
        mid, a = ind.ema(c, n), ind.atr(h, lo, c, n)
        raw = np.where(c > mid + k * a, 1.0, np.where(c < mid - k * a, -1.0, np.nan))
        return _hold(np.where(np.isnan(a), 0.0, raw))

    return f


def bollinger_reversion(n: int, k: float) -> Callable[..., Arr]:
    def f(o: Arr, h: Arr, lo: Arr, c: Arr, v: Arr) -> Arr:
        z = ind.zscore(c, n)
        raw = np.where(z > k, -1.0, np.where(z < -k, 1.0, np.nan))
        raw = np.where(np.sign(z) != np.sign(_prior(z)), 0.0, raw)  # exit when price crosses the mean
        return _hold(np.where(np.isnan(z), 0.0, raw))

    return f


def rsi_reversion(n: int, band: float) -> Callable[..., Arr]:
    def f(o: Arr, h: Arr, lo: Arr, c: Arr, v: Arr) -> Arr:
        r = ind.rsi(c, n)
        raw = np.where(r < 50 - band, 1.0, np.where(r > 50 + band, -1.0, np.nan))
        raw = np.where((r - 50) * (_prior(r) - 50) < 0, 0.0, raw)  # exit on crossing 50
        return _hold(np.where(np.isnan(r), 0.0, raw))

    return f


def stochastic_reversion(n: int) -> Callable[..., Arr]:
    def f(o: Arr, h: Arr, lo: Arr, c: Arr, v: Arr) -> Arr:
        s = ind.stochastic(h, lo, c, n)
        return np.nan_to_num(np.where(s < 20, 1.0, np.where(s > 80, -1.0, 0.0)))

    return f


def vwap_reversion(n: int, k: float) -> Callable[..., Arr]:
    def f(o: Arr, h: Arr, lo: Arr, c: Arr, v: Arr) -> Arr:
        vw = ind.sma(c * v, n) / np.maximum(ind.sma(v, n), 1e-12)
        d = (c - vw) / np.maximum(ind.rolling_std(c, n), 1e-12)
        return np.nan_to_num(np.where(d > k, -1.0, np.where(d < -k, 1.0, 0.0)))

    return f


def trend_pullback(fast: int, slow: int) -> Callable[..., Arr]:
    """Buy dips in an uptrend, sell rallies in a downtrend."""

    def f(o: Arr, h: Arr, lo: Arr, c: Arr, v: Arr) -> Arr:
        up = ind.ema(c, fast) > ind.ema(c, slow)
        r = ind.rsi(c, 14)
        return np.nan_to_num(np.where(up & (r < 40), 1.0, np.where(~up & (r > 60), -1.0, 0.0)))

    return f


def adx_trend(n: int, threshold: float) -> Callable[..., Arr]:
    """Follow the trend only when ADX says there is one."""

    def f(o: Arr, h: Arr, lo: Arr, c: Arr, v: Arr) -> Arr:
        strong = np.nan_to_num(ind.adx(h, lo, c, n)) > threshold
        return np.where(strong, np.sign(ind.ema(c, n) - ind.ema(c, 3 * n)), 0.0)

    return f


def range_trade(n: int, k: float) -> Callable[..., Arr]:
    """Mean reversion only when ADX says the market is ranging."""

    def f(o: Arr, h: Arr, lo: Arr, c: Arr, v: Arr) -> Arr:
        quiet = np.nan_to_num(ind.adx(h, lo, c, 14), nan=100.0) < 20
        return np.where(quiet, bollinger_reversion(n, k)(o, h, lo, c, v), 0.0)

    return f


def _grid() -> list[Variant]:
    out: list[Variant] = []

    def add(family: str, label: str, fn: Callable[..., Arr]) -> None:
        out.append(Variant(family, f"{family}({label})", fn))

    for fast, slow in ((5, 20), (10, 50), (20, 100), (50, 200)):
        add("ma_cross", f"{fast},{slow}", ma_cross(fast, slow))
    for look in (6, 24, 96, 288):
        add("tsmom", str(look), tsmom(look))
    for fast, slow in ((12, 26), (24, 52)):
        add("macd", f"{fast},{slow}", macd_trend(fast, slow))
    for n in (20, 55, 200):
        add("donchian_breakout", str(n), donchian_breakout(n))
        add("breakout_fade", str(n), breakout_fade(n))
    for n, k in ((20, 1.5), (20, 2.5), (50, 2.0)):
        add("keltner_breakout", f"{n},{k}", keltner_breakout(n, k))
    for n, k in ((20, 1.5), (20, 2.5), (96, 2.0)):
        add("bollinger_reversion", f"{n},{k}", bollinger_reversion(n, k))
        add("range_trade", f"{n},{k}", range_trade(n, k))
    for n, band in ((14, 20), (14, 30), (48, 15)):
        add("rsi_reversion", f"{n},{band}", rsi_reversion(n, band))
    for n in (14, 48):
        add("stochastic_reversion", str(n), stochastic_reversion(n))
    for n, k in ((48, 1.5), (288, 2.0)):
        add("vwap_reversion", f"{n},{k}", vwap_reversion(n, k))
    for fast, slow in ((20, 100), (50, 200)):
        add("trend_pullback", f"{fast},{slow}", trend_pullback(fast, slow))
    for n, th in ((14, 25), (28, 20)):
        add("adx_trend", f"{n},{th}", adx_trend(n, th))
    return out


VARIANTS: tuple[Variant, ...] = tuple(_grid())
NAMES: tuple[str, ...] = tuple(v.name for v in VARIANTS)
LONGEST_LOOKBACK = 3 * 288  # bars of history the slowest variant needs to be settled


def all_signals(candles: Arr) -> Arr:
    """(n_bars, n_variants) matrix of positions."""
    _, o, h, lo, c, v = candles.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.column_stack([np.clip(np.nan_to_num(x.fn(o, h, lo, c, v)), -1, 1) for x in VARIANTS])
