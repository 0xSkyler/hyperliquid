"""Lessons: the scalper's memory of what worked and what did not.

Two ledgers score, every second, the trades the scalper *could* have made, whether or not it
made them, and what each would have been worth a few seconds later. Each lesson is filed under
the situation it happened in. Before it acts for real, the scalper looks up that situation:

- QuoteLedger: "if I rest a quote this far behind the best price, on this side, while my
  forecast points with / against / nowhere, how often is it hit, and what is a hit worth?"
- TakeLedger: "if I cross the spread on a forecast this strong, what do I actually make?"

A situation whose record is a loss is not traded again while the record says so. The memory
fades slowly (old lessons count for less), so when the market changes it re-learns. Every
figure is used at its lower confidence bound: a few lucky results do not earn trust.

What this cannot do is promise that a mistake never recurs. Markets change faster than any
record; this guarantees only that the scalper stops repeating what its own evidence says loses.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

DISTANCES_BPS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)  # how far behind the best price a quote rests
CONTEXTS = ("forecast against", "neutral", "forecast with")
TAKE_EDGES_BPS = (0.1, 0.2, 0.4, 0.8, 1.6, 3.2)  # forecast-strength buckets for taking
_DECAY = 0.99995  # per second: a lesson's weight halves in about four hours
_Z = 2.0


def _context(side_alpha_bps: float, scale_bps: float) -> int:
    """0: the forecast says price will move against this quote; 2: in its favour; 1: neither."""
    thr = 0.5 * max(scale_bps, 1e-6)
    return 2 if side_alpha_bps > thr else 0 if side_alpha_bps < -thr else 1


class QuoteLedger:
    def __init__(self, ttl_s: float = 10.0, markout_s: float = 5.0, min_fills: float = 30.0) -> None:
        self.ttl_s, self.markout_s, self.min_fills = ttl_s, markout_s, min_fills
        shape = (2, len(CONTEXTS), len(DISTANCES_BPS))
        self.quotes = np.zeros(shape)  # hypothetical quotes placed
        self.fills = np.zeros(shape)  # ...that would have been hit
        self.sum = np.zeros(shape)  # sum of what each hit was worth (bps vs fill price, 5 s later)
        self.sum2 = np.zeros(shape)
        self.alpha_scale = 0.2  # running typical size of the fast forecast, bps
        self._open: deque[list[float]] = deque()  # [expires, side, ctx, d, px, queue ahead of us]
        self._filled: deque[tuple[float, int, int, int, float]] = deque()  # due, side, ctx, d, px
        self._last = 0.0

    def on_tick(
        self, now: float, bid: float, ask: float, bid_sz: float, ask_sz: float, tick: float, alpha_bps: float,
        trades: list[tuple[float, float, bool]],
    ) -> None:  # fmt: skip
        """`trades`: (price, size, aggressor is buyer) since the last call. A hypothetical quote is hit the way a
        real one at the back of the queue would be: when the market trades through its price, or when enough has
        traded at its price to clear the size that was resting there ahead of it."""
        mid = 0.5 * (bid + ask)
        dt = min(max(now - self._last, 0.0), 60.0) if self._last else 0.0
        self._last = now
        if dt:
            k = _DECAY**dt
            for a in (self.quotes, self.fills, self.sum, self.sum2):
                a *= k
        self.alpha_scale += 0.001 * (abs(alpha_bps) - self.alpha_scale)

        still: deque[list[float]] = deque()
        for q in self._open:
            exp, side, ctx, d, px = q[0], int(q[1]), int(q[2]), int(q[3]), q[4]
            hit = False
            for tpx, tsz, tbuy in trades:
                if tbuy == (side == 0):
                    continue  # a buyer cannot hit our bid, nor a seller our ask
                if (tpx < px - 1e-12) if side == 0 else (tpx > px + 1e-12):
                    hit = True
                    break
                if abs(tpx - px) <= 1e-12:
                    q[5] -= tsz
                    if q[5] < 0:
                        hit = True
                        break
            if hit:
                self.fills[side, ctx, d] += 1
                self._filled.append((now + self.markout_s, side, ctx, d, px))
            elif exp > now:
                still.append(q)
        self._open = still
        while self._filled and self._filled[0][0] <= now:
            _, side, ctx, d, px = self._filled.popleft()
            worth = (mid - px) / px * 1e4 * (1 if side == 0 else -1)
            self.sum[side, ctx, d] += worth
            self.sum2[side, ctx, d] += worth * worth

        for side in (0, 1):  # 0 = bid, 1 = ask
            ctx = _context(alpha_bps if side == 0 else -alpha_bps, self.alpha_scale)
            for d, dist in enumerate(DISTANCES_BPS):
                if side == 0:
                    px = math.floor(bid * (1 - dist * 1e-4) / tick + 1e-9) * tick
                else:
                    px = math.ceil(ask * (1 + dist * 1e-4) / tick - 1e-9) * tick
                self.quotes[side, ctx, d] += 1
                self._open.append([now + self.ttl_s, side, ctx, d, px, bid_sz if side == 0 else ask_sz])

    def _value(self, side: int, ctx: int, d: int, fee_bps: float) -> tuple[float, float, float, float]:
        """(fill probability, mean worth of a fill, lower bound of that, expected value per quote at the lower bound)."""
        n, q = self.fills[side, ctx, d], self.quotes[side, ctx, d]
        resolved = max(n - sum(1 for f in self._filled if f[1:4] == (side, ctx, d)), 0.0)
        if resolved < 1 or q < 1:
            return 0.0, 0.0, -math.inf, -math.inf
        mean = self.sum[side, ctx, d] / resolved
        var = max(self.sum2[side, ctx, d] / resolved - mean * mean, 0.0)
        lcb = mean - _Z * math.sqrt(var / resolved)
        p = min(n / q, 1.0)
        return p, mean, lcb, (p * (lcb - fee_bps) if resolved >= self.min_fills else -math.inf)

    def informed(self, is_buy: bool, alpha_bps: float) -> bool:
        """Has this situation been seen often enough for the record to decide?"""
        side = 0 if is_buy else 1
        ctx = _context(alpha_bps if is_buy else -alpha_bps, self.alpha_scale)
        return bool(self.fills[side, ctx].max() >= self.min_fills)

    def best_distance(self, is_buy: bool, alpha_bps: float, fee_bps: float) -> float | None:
        """Where to rest a quote on this side right now, in bps behind the best price; None = the record says do not."""
        side = 0 if is_buy else 1
        ctx = _context(alpha_bps if is_buy else -alpha_bps, self.alpha_scale)
        best, best_ev = None, 0.0
        for d, dist in enumerate(DISTANCES_BPS):
            ev = self._value(side, ctx, d, fee_bps)[3]
            if ev > best_ev:
                best, best_ev = dist, ev
        return best

    def table(self, fee_bps: float) -> list[dict[str, Any]]:
        rows = []
        for side, name in enumerate(("bid", "ask")):
            for ctx, cname in enumerate(CONTEXTS):
                for d, dist in enumerate(DISTANCES_BPS):
                    p, mean, lcb, ev = self._value(side, ctx, d, fee_bps)
                    if self.quotes[side, ctx, d] < 1:
                        continue
                    rows.append({
                        "side": name, "situation": cname, "bps_behind_touch": dist, "quotes": round(float(self.quotes[side, ctx, d])),
                        "hits": round(float(self.fills[side, ctx, d])), "hit_rate": p, "worth_per_hit_bps": mean,
                        "worth_lower_bound_bps": None if lcb == -math.inf else lcb,
                        "verdict": "not enough hits yet" if ev == -math.inf else "quote here" if ev > 0 else "avoid: loses after the fee",
                    })  # fmt: skip
        return rows


class TakeLedger:
    """What crossing the spread on a forecast of a given strength has actually been worth."""

    def __init__(self, markout_s: float = 5.0, min_n: float = 50.0) -> None:
        self.markout_s, self.min_n = markout_s, min_n
        n = len(TAKE_EDGES_BPS) + 1
        self.n = np.zeros(n)
        self.sum = np.zeros(n)
        self.sum2 = np.zeros(n)
        self._pending: deque[tuple[float, int, float, float]] = deque()  # due, bucket, side, price paid
        self._last = 0.0

    @staticmethod
    def bucket(alpha_bps: float) -> int:
        return int(np.searchsorted(TAKE_EDGES_BPS, abs(alpha_bps), side="right"))

    def on_tick(self, now: float, bid: float, ask: float, alpha_bps: float) -> None:
        mid = 0.5 * (bid + ask)
        dt = min(max(now - self._last, 0.0), 60.0) if self._last else 0.0
        self._last = now
        if dt:
            k = _DECAY**dt
            for a in (self.n, self.sum, self.sum2):
                a *= k
        while self._pending and self._pending[0][0] <= now:
            _, b, side, px = self._pending.popleft()
            worth = side * (mid - px) / px * 1e4  # after paying the spread, before the fee
            self.n[b] += 1
            self.sum[b] += worth
            self.sum2[b] += worth * worth
        if alpha_bps != 0.0:
            side = 1.0 if alpha_bps > 0 else -1.0
            self._pending.append((now + self.markout_s, self.bucket(alpha_bps), side, ask if side > 0 else bid))

    def worth(self, alpha_bps: float) -> tuple[float, float, float]:
        """(mean, lower bound, sample count) of what a take at this forecast strength has been worth, in bps."""
        b = self.bucket(alpha_bps)
        n = self.n[b]
        if n < 1:
            return 0.0, -math.inf, 0.0
        mean = self.sum[b] / n
        var = max(self.sum2[b] / n - mean * mean, 0.0)
        return float(mean), float(mean - _Z * math.sqrt(var / n)), float(n)

    def allows(self, alpha_bps: float, fee_bps: float, margin_bps: float) -> bool:
        """Take only where the record, at its lower bound, beats the fee. Until the record exists, do not."""
        _, lcb, n = self.worth(alpha_bps)
        return n >= self.min_n and lcb > fee_bps + margin_bps

    def table(self, fee_bps: float) -> list[dict[str, Any]]:
        rows = []
        edges = (0.0, *TAKE_EDGES_BPS, math.inf)
        for b in range(len(self.n)):
            if self.n[b] < 1:
                continue
            mean = float(self.sum[b] / self.n[b])
            lcb = mean - _Z * math.sqrt(max(float(self.sum2[b] / self.n[b]) - mean * mean, 0.0) / float(self.n[b]))
            hi = "+" if edges[b + 1] == math.inf else f"-{edges[b + 1]:g}"
            rows.append({
                "forecast_bps": f"{edges[b]:g}{hi}", "times": round(float(self.n[b])), "worth_bps": mean, "worth_lower_bound_bps": lcb,
                "verdict": "not enough yet" if self.n[b] < self.min_n else "take" if lcb > fee_bps else "avoid: less than the fee",
            })  # fmt: skip
        return rows
