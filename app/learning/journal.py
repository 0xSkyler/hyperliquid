"""Post-trade measurement: equity path, drawdown, fees, and fill markouts (adverse selection)."""

from __future__ import annotations

from collections import deque
from typing import Any

from app.exchange.base import Fill

MARKOUT_S = 5.0


class Journal:
    def __init__(self) -> None:
        self.start_equity: float | None = None
        self.equity = 0.0
        self.peak = 0.0
        self.max_dd = 0.0
        self.n_fills = 0
        self.maker_fills = 0
        self.fees = 0.0
        self.volume = 0.0
        self.liquidations = 0
        self._pending: deque[tuple[float, bool, float, bool]] = deque()
        self._markout = {True: [0.0, 0], False: [0.0, 0]}  # maker? -> [mean bps, n]

    def on_fill(self, f: Fill) -> None:
        self.n_fills += 1
        self.maker_fills += f.maker
        self.fees += f.fee
        self.volume += f.px * f.sz
        self.liquidations += f.liquidation
        self._pending.append((f.ts + MARKOUT_S, f.is_buy, f.px, f.maker))

    def on_tick(self, now: float, mid: float, equity: float) -> None:
        if self.start_equity is None:
            self.start_equity = equity
        self.equity = equity
        self.peak = max(self.peak, equity)
        if self.peak > 0:
            self.max_dd = max(self.max_dd, 1 - equity / self.peak)
        while self._pending and self._pending[0][0] <= now:
            _, is_buy, px, maker = self._pending.popleft()
            m = (mid - px) / px * 1e4 * (1 if is_buy else -1)
            acc = self._markout[maker]
            acc[1] += 1
            acc[0] += (m - acc[0]) / min(acc[1], 200)

    @property
    def drawdown(self) -> float:
        return 1 - self.equity / self.peak if self.peak > 0 else 0.0

    def maker_adverse_bps(self) -> float:
        """Measured cost of being picked off as a maker; a 1 bp prior until there is evidence."""
        mean, n = self._markout[True]
        return max(-mean, 0.0) if n >= 10 else 1.0

    def summary(self) -> dict[str, Any]:
        s0 = self.start_equity or 0.0
        return {
            "start_equity": s0,
            "equity": self.equity,
            "net_return_pct": (self.equity / s0 - 1) * 100 if s0 else 0.0,
            "max_drawdown_pct": self.max_dd * 100,
            "fills": self.n_fills,
            "maker_ratio": self.maker_fills / self.n_fills if self.n_fills else 0.0,
            "fees": self.fees,
            "volume": self.volume,
            "liquidations": self.liquidations,
            "markout_5s_bps": {"maker": self._markout[True][0], "taker": self._markout[False][0]},
        }
