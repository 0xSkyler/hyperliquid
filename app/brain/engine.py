"""The trading engine: perception -> forecast -> calibration -> utility decision -> execution.

Synchronous and event-driven so the exact same code runs live, in paper mode and in backtest.
Nothing here touches disk or network; persistence goes through a non-blocking sink.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Protocol

import numpy as np

from app.brain.decision import Decision, Forecast, decide
from app.config.settings import Mode, Settings
from app.exchange.base import AssetCtx, AssetMeta, Book, Trade, Venue, merge_bbo
from app.learning.journal import Journal
from app.market.state import FEATURE_NAMES, REGIMES, MarketState
from app.models.online import EdgeCalibrator, HalfLife, OnlineRidge, Standardizer
from app.risk.kernel import SafetyKernel


class Sink(Protocol):
    def put(self, stream: str, obj: Any) -> None: ...


class Engine:
    def __init__(self, s: Settings, venue: Venue, meta: AssetMeta, sink: Sink | None = None) -> None:
        self.s, self.venue, self.meta, self.sink = s, venue, meta, sink
        n = len(FEATURE_NAMES)
        self.market = MarketState(s.decision_interval_s)
        self.std = Standardizer(n - 1)
        self.model = OnlineRidge(n, s.forgetting)
        self.cal = EdgeCalibrator(len(REGIMES), s.cal_forgetting, s.cal_z, s.min_indep_samples, s.horizon_ticks)
        self.half_life = HalfLife(s.decision_interval_s)
        self.kernel = SafetyKernel(s)
        self.journal = Journal()
        # [due_ts, x, reference mid (None until known), raw forecast, regime, reference ts]
        self._pending: deque[list[Any]] = deque()
        self._unref: deque[list[Any]] = deque()
        self.decisions: deque[dict[str, Any]] = deque(maxlen=100)
        self.ticks = 0
        self.orders_sent = 0
        self.state = "OBSERVING"
        self.last: dict[str, Any] = {}

    # --- market events (hot path: memory only) --------------------------
    def on_book(self, book: Book) -> None:
        self.market.on_book(book)
        self.venue.on_book(book)
        # Forecasts are scored from the first price we could actually have traded at
        # (decision time + latency), not from the price we were looking at when deciding.
        while self._unref and self._unref[0][5] <= book.ts:
            item = self._unref.popleft()
            if book.valid():
                item[2] = book.mid

    def on_bbo(self, ts: float, bbo: tuple[float, float, float, float, float]) -> None:
        self.on_book(merge_bbo(self.market.book, self.meta.coin, ts, bbo))

    def on_trades(self, trades: list[Trade]) -> None:
        self.market.on_trades(trades)
        self.venue.on_trades(trades)

    def on_ctx(self, ctx: AssetCtx) -> None:
        self.market.on_ctx(ctx)
        self.venue.on_ctx(ctx)

    # --- decision tick ---------------------------------------------------
    def on_tick(self, now: float) -> Decision | None:
        s, mkt = self.s, self.market
        self.ticks += 1
        mkt.sample(now)
        for f in self.venue.drain_fills():
            self.journal.on_fill(f)
            self.kernel.on_fill(f.sz if f.is_buy else -f.sz)
            if self.sink:
                self.sink.put("fills", {"ts": f.ts, "is_buy": f.is_buy, "px": f.px, "sz": f.sz, "fee": f.fee,
                                        "maker": f.maker, "liquidation": f.liquidation})  # fmt: skip

        acct = self.venue.account(now)
        faults = self.kernel.check(now, mkt.book, acct, 10.0**-self.meta.sz_decimals, mkt.feed_ts)
        book = mkt.book
        if book is not None and book.valid() and acct.known:
            self.journal.on_tick(now, book.mid, acct.equity)
            # Learn from forecasts whose horizon has elapsed (calibrate first: out-of-sample).
            while self._pending and self._pending[0][0] <= now:
                _, x, mid0, mu_raw, regime, _ = self._pending.popleft()
                if mid0 is None:
                    continue
                y = math.log(book.mid / mid0) * 1e4
                self.cal.update(mu_raw, y, regime)
                self.model.update(x, y)

        feats = mkt.features(now) if not faults else None
        if faults or feats is None or book is None:
            self.state = "HALTED_INSTRUMENTATION" if faults else "OBSERVING"
            self.last = {"ts": now, "state": self.state, "faults": faults, "warmup": feats is None}
            if faults:
                self._unref.clear()
                self._pending.clear()  # forecasts spanning a data gap would be scored against garbage
                self.half_life.prev = None
            return None

        x = np.append(self.std.transform(feats.values), 1.0)
        mu_raw, pvar = self.model.predict(x)
        item = [now + s.horizon_s, x, None, mu_raw, feats.regime, now + s.latency_ms / 1000]
        self._pending.append(item)
        self._unref.append(item)
        beta = self.cal.beta(feats.regime)
        hl = self.half_life.update(mu_raw)
        sigma_h = max(feats.sigma_tick * math.sqrt(s.horizon_ticks) * 1e4, math.sqrt(self.model.resid_var))
        fc = Forecast(
            mu_bps=beta * mu_raw, mu_raw_bps=mu_raw, sigma_bps=sigma_h, param_sigma_bps=math.sqrt(pvar),
            beta=beta, half_life_s=hl, horizon_s=s.horizon_s, sell_rate=feats.sell_rate,
            buy_rate=feats.buy_rate, maker_adverse_bps=self.journal.maker_adverse_bps(),
        )  # fmt: skip
        d = decide(now, fc, acct, book, self.meta, s)

        contrib = self.model.w[:-1] * x[:-1]
        order = np.argsort(-contrib * np.sign(mu_raw if mu_raw else 1.0))
        d.extra = {
            "regime": {k: round(float(v), 3) for k, v in zip(REGIMES, feats.regime, strict=True)},
            "mu_raw_bps": mu_raw, "sigma_bps": sigma_h, "param_sigma_bps": fc.param_sigma_bps,
            "beta": beta, "half_life_s": hl,
            "supporting": [(FEATURE_NAMES[i], round(float(contrib[i]), 3)) for i in order[:4]],
            "contradicting": [(FEATURE_NAMES[i], round(float(contrib[i]), 3)) for i in order[-3:][::-1]],
        }  # fmt: skip

        if d.order is not None:
            if acct.inflight:
                d.action, d.reason, d.order = "HOLD", "waiting for a previous order to be acknowledged", None
            elif s.mode is Mode.SHADOW:
                d.extra["shadow"] = True
            else:
                if acct.open_orders:
                    self.venue.cancel_all(now)
                d.order.client_id = f"t{self.ticks}"
                self.venue.submit(d.order, now)
                self.orders_sent += 1

        self.state = self._label(d, feats.regime, beta)
        rec = d.to_dict() | {"state": self.state}
        self.last = rec | {"faults": [], "warmup": False}
        if d.order is not None:
            self.decisions.append(rec)
        if self.sink and (d.order is not None or self.ticks % 60 == 0):
            self.sink.put("decisions", rec)
        return d

    def _label(self, d: Decision, regime: np.ndarray, beta: float) -> str:
        """Descriptive only: nothing reads this to make a trading decision."""
        aggression = abs(d.f_target) / self.meta.max_leverage
        if aggression > 0.5:
            return "AGGRESSIVE"
        if aggression > 0.15:
            return "OPPORTUNISTIC"
        if aggression > 0:
            return "NORMAL"
        if self.journal.drawdown > 0.10:
            return "CAPITAL_PRESERVATION"
        if self.journal.drawdown > 0.03:
            return "RECOVERY"
        if regime[2] > 0.5:
            return "VOLATILITY_EXPANSION"
        return "OBSERVING" if beta == 0 else "NORMAL"

    def snapshot(self) -> dict[str, Any]:
        slope, lcb, n = self.cal.per_regime()
        b = self.market.book
        return {
            "mode": self.s.mode.value, "coin": self.meta.coin, "state": self.state, "ticks": self.ticks,
            "mid": b.mid if b is not None and b.valid() else None,
            "last": self.last, "journal": self.journal.summary(), "orders_sent": self.orders_sent,
            "model": {
                "resolved_forecasts": self.model.n_obs, "oos_ic": self.cal.ic(),
                "resid_std_bps": math.sqrt(self.model.resid_var),
                "calibration": {r: {"slope": float(slope[i]), "trusted_beta": float(lcb[i]),
                                    "indep_samples": float(n[i])} for i, r in enumerate(REGIMES)},
                "weights": dict(zip(FEATURE_NAMES, (round(float(w), 4) for w in self.model.w), strict=True)),
            },
            "recent_decisions": list(self.decisions)[-20:],
        }  # fmt: skip
