"""In-memory market world model: book, trade flow, order-flow imbalance and sampled mids,
plus the feature vector and soft regime probabilities derived from them."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from app.exchange.base import AssetCtx, Book, Trade
from app.indicators.library import efficiency_ratio, rsi

FEATURE_NAMES = (
    "imb_l1", "imb_l5", "microprice_bps",
    "ofi_1s", "ofi_5s", "ofi_30s",
    "tfi_5s", "tfi_30s",
    "ret_5s", "ret_30s", "ret_300s",
    "rsi_15s", "z_300s", "premium_bps", "spread_bps", "news_llm", "chart_ctx",
    "bias",
)  # fmt: skip
REGIMES = ("trend", "range", "chaos")
LOOKBACK_S = 300.0
_FLOW_KEEP_S = 30.0


@dataclass(slots=True)
class Features:
    values: np.ndarray  # raw, without the bias term
    sigma_tick: float  # std of per-tick log returns
    regime: np.ndarray  # probabilities over REGIMES
    sell_rate: float  # aggressor-sell volume per second (fills resting bids)
    buy_rate: float


class MarketState:
    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self.n_look = round(LOOKBACK_S / interval_s)
        self.book: Book | None = None
        self.ctx: AssetCtx | None = None
        self.chart_score = 0.0  # chart model's forecast for the current 5-minute bar; 0 if no model
        self.news_score = 0.0  # decayed LLM-scored news pressure; 0 unless HL_LLM_NEWS is enabled
        self.feed_ts = 0.0  # last message of any kind: liveness of the market-data connection
        self.mids: deque[float] = deque(maxlen=self.n_look + 1)
        self._ofi: deque[tuple[float, float]] = deque()
        self._trades: deque[tuple[float, float]] = deque()  # (ts, signed size)

    def on_book(self, book: Book) -> None:
        prev = self.book
        if prev is not None and prev.valid() and book.valid():
            bp, bq = book.bids[0]
            pbp, pbq = prev.bids[0]
            ap, aq = book.asks[0]
            pap, paq = prev.asks[0]
            # Cont-Kukanov-Stoikov order-flow imbalance at the touch.
            e = (bq if bp >= pbp else 0.0) - (pbq if bp <= pbp else 0.0)
            e += (paq if ap >= pap else 0.0) - (aq if ap <= pap else 0.0)
            self._ofi.append((book.ts, float(e)))
        self.book = book
        self.feed_ts = max(self.feed_ts, book.ts)

    def on_trades(self, trades: list[Trade]) -> None:
        if trades:
            self.feed_ts = max(self.feed_ts, trades[-1].ts)
        for t in trades:
            self._trades.append((t.ts, t.sz if t.is_buy else -t.sz))

    def on_ctx(self, ctx: AssetCtx) -> None:
        self.ctx = ctx
        self.feed_ts = max(self.feed_ts, ctx.ts)

    def sample(self, now: float) -> None:
        """Called once per decision tick."""
        if self.book is not None and self.book.valid():
            self.mids.append(self.book.mid)
        for dq in (self._ofi, self._trades):
            while dq and dq[0][0] < now - _FLOW_KEEP_S:
                dq.popleft()

    def _window(self, dq: deque[tuple[float, float]], now: float, w: float) -> list[float]:
        out = []
        for ts, v in reversed(dq):
            if ts < now - w:
                break
            out.append(v)
        return out

    def features(self, now: float) -> Features | None:
        b = self.book
        if b is None or not b.valid() or len(self.mids) <= self.n_look:
            return None
        arr = np.fromiter(self.mids, dtype=float)
        lr = np.diff(np.log(arr))
        sig = float(lr.std())
        if sig <= 0:
            return None
        mid = b.mid
        k = min(5, len(b.bids), len(b.asks))
        bq, aq = b.bids[:k, 1], b.asks[:k, 1]
        w = 1.0 / np.arange(1, k + 1)
        micro = (b.best_ask * bq[0] + b.best_bid * aq[0]) / (bq[0] + aq[0])
        depth = float(bq.sum() + aq.sum()) / 2.0

        def ofi(win: float) -> float:
            return sum(self._window(self._ofi, now, win)) / depth

        def tfi(win: float) -> tuple[float, float, float]:
            v = self._window(self._trades, now, win)
            buy = sum(x for x in v if x > 0)
            sell = -sum(x for x in v if x < 0)
            tot = buy + sell
            return ((buy - sell) / tot if tot > 0 else 0.0), buy, sell

        def ret(secs: float) -> float:
            n = max(1, round(secs / self.interval_s))
            return math.log(arr[-1] / arr[-1 - n]) / (sig * math.sqrt(n))

        tfi5, _, _ = tfi(5)
        tfi30, buy30, sell30 = tfi(30)
        step = max(1, round(15 / self.interval_s))
        r = rsi(arr[::-1][::step][::-1], 14)[-1]
        sd = float(arr.std())
        vals = np.array([
            (bq[0] - aq[0]) / (bq[0] + aq[0]),
            float(((bq - aq) * w).sum() / ((bq + aq) * w).sum()),
            (micro - mid) / mid * 1e4,
            ofi(1), ofi(5), ofi(30),
            tfi5, tfi30,
            ret(5), ret(30), ret(LOOKBACK_S),
            (r - 50.0) / 50.0 if not math.isnan(r) else 0.0,
            (mid - float(arr.mean())) / sd if sd > 0 else 0.0,
            self.ctx.premium * 1e4 if self.ctx else 0.0,
            (b.best_ask - b.best_bid) / mid * 1e4,
            self.news_score,
            self.chart_score,
        ])  # fmt: skip

        # Soft regime: heuristics expressed as probabilities, never as a single hard label.
        er = efficiency_ratio(arr)
        n60 = max(2, round(60 / self.interval_s))
        vol_ratio = float(lr[-n60:].std()) / sig
        logits = np.array([6.0 * (er - 0.25), 6.0 * (0.25 - er), 3.0 * (vol_ratio - 1.5)])
        p = np.exp(logits - logits.max())
        return Features(vals, sig, p / p.sum(), sell30 / _FLOW_KEEP_S, buy30 / _FLOW_KEEP_S)
