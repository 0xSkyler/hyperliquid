"""Post-trade measurement: equity path, drawdown, fees, and what each fill was really worth.

For a scalper the numbers that matter are per fill:
- spread capture: how far from the mid we traded, in our favour, at the moment of the fill;
- markout: where the mid went 1, 5 and 30 seconds later. If markout eats the spread capture,
  our fills are being picked off (adverse selection);
- adverse selection per side, which the quoter uses to decide how far from fair value to rest.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from app.exchange.base import Fill

MARKOUT_S = 5.0
HORIZONS = (1.0, 5.0, 30.0)
_PRIOR_N = 20.0  # weight of the prior adverse-selection estimate, in fills


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
        self.capture_usd = 0.0  # sum over fills of (mid - price) in our favour: gross spread earned
        self._mid = 0.0
        # (due, horizon, is_buy, mid at fill, maker)
        self._pending: deque[tuple[float, float, bool, float, bool]] = deque()
        self._markout = {True: [0.0, 0], False: [0.0, 0]}  # maker? -> [mean bps at 5s vs fill px, n]
        self._px_pending: deque[tuple[float, bool, float, bool]] = deque()
        self._move: dict[float, list[float]] = {h: [0.0, 0.0] for h in HORIZONS}  # horizon -> [mean bps in our favour, n]
        self._adverse = {True: [0.0, 0.0], False: [0.0, 0.0]}  # is_buy -> [mean adverse bps at 5s, n] (maker fills)
        self._capture_bps = [0.0, 0.0]

    def on_fill(self, f: Fill, mid: float | None = None) -> None:
        self.n_fills += 1
        self.maker_fills += f.maker
        self.fees += f.fee
        self.volume += f.px * f.sz
        self.liquidations += f.liquidation
        m = f.mid or mid or self._mid or f.px
        side = 1.0 if f.is_buy else -1.0
        self.capture_usd += side * (m - f.px) * f.sz
        self._capture_bps[1] += 1
        self._capture_bps[0] += (side * (m - f.px) / m * 1e4 - self._capture_bps[0]) / min(self._capture_bps[1], 500)
        self._px_pending.append((f.ts + MARKOUT_S, f.is_buy, f.px, f.maker))
        for h in HORIZONS:
            self._pending.append((f.ts + h, h, f.is_buy, m, f.maker))

    def on_tick(self, now: float, mid: float, equity: float) -> None:
        if self.start_equity is None:
            self.start_equity = equity
        self._mid = mid
        self.equity = equity
        self.peak = max(self.peak, equity)
        if self.peak > 0:
            self.max_dd = max(self.max_dd, 1 - equity / self.peak)
        while self._px_pending and self._px_pending[0][0] <= now:
            _, is_buy, px, maker = self._px_pending.popleft()
            acc = self._markout[maker]
            acc[1] += 1
            acc[0] += ((mid - px) / px * 1e4 * (1 if is_buy else -1) - acc[0]) / min(acc[1], 200)
        keep: deque[tuple[float, float, bool, float, bool]] = deque()
        for item in self._pending:  # horizons differ, so the queue is not sorted by due time
            due, h, is_buy, m0, maker = item
            if due > now:
                keep.append(item)
                continue
            move = (mid - m0) / m0 * 1e4 * (1 if is_buy else -1)  # positive = price went our way after the fill
            mv = self._move[h]
            mv[1] += 1
            mv[0] += (move - mv[0]) / min(mv[1], 500)
            if h == MARKOUT_S and maker:
                ad = self._adverse[is_buy]
                ad[1] += 1
                ad[0] += (-move - ad[0]) / min(ad[1], 200)
        self._pending = keep

    @property
    def drawdown(self) -> float:
        return 1 - self.equity / self.peak if self.peak > 0 else 0.0

    def maker_adverse_bps(self) -> float:
        """Measured cost of being picked off as a maker (vs fill price); a 1 bp prior until there is evidence."""
        mean, n = self._markout[True]
        return max(-mean, 0.0) if n >= 10 else 1.0

    def adverse_bps(self, is_buy: bool, prior_bps: float) -> float:
        """How far the mid moves against a resting order on this side in the 5 s after it fills.

        Starts at `prior_bps` (the market's own half-spread: a touch quote roughly breaks even
        before fees) and moves to what our own fills show as they accumulate.
        """
        mean, n = self._adverse[is_buy]
        return max((mean * n + prior_bps * _PRIOR_N) / (n + _PRIOR_N), 0.0)

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
            "spread_capture_usd": self.capture_usd,
            "spread_capture_bps": self._capture_bps[0],
            "move_after_fill_bps": {f"{h:g}s": self._move[h][0] for h in HORIZONS},
            "adverse_selection_bps": {"bid_fills": self._adverse[True][0], "ask_fills": self._adverse[False][0]},
        }
