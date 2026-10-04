"""The trading engine: perception -> forecast -> calibration -> utility decision -> execution.

Synchronous and event-driven so the exact same code runs live, in paper mode and in backtest.
Nothing here touches disk or network; persistence goes through a non-blocking sink.
"""

from __future__ import annotations

import math
import pickle  # noqa: S403 - only ever loads the state file this process wrote itself
from collections import deque
from typing import Any, Protocol

import numpy as np

from app.brain.decision import Decision, Forecast, decide
from app.config.settings import Mode, Settings
from app.ensemble.arena import build_arena
from app.exchange.base import AssetCtx, AssetMeta, Book, OrderIntent, Trade, Venue, merge_bbo
from app.learning.journal import Journal
from app.market.state import FEATURE_NAMES, REGIMES, MarketState
from app.models.online import HalfLife, Standardizer
from app.risk.kernel import SafetyKernel


class Sink(Protocol):
    def put(self, stream: str, obj: Any) -> None: ...


class Engine:
    def __init__(self, s: Settings, venue: Venue, meta: AssetMeta, sink: Sink | None = None) -> None:
        self.s, self.venue, self.meta, self.sink = s, venue, meta, sink
        n = len(FEATURE_NAMES)
        self.market = MarketState(s.decision_interval_s)
        self.std = Standardizer(n - 1)
        self.arena = build_arena(s, FEATURE_NAMES[:-1], len(REGIMES), async_fit=s.mode is not Mode.BACKTEST)
        self.arena.on_promotion = self._on_promotion
        self.half_life = HalfLife(s.decision_interval_s)
        self.kernel = SafetyKernel(s)
        self.journal = Journal()
        # [due_ts, per-model inputs, reference mid (None until known), per-model raw forecasts, regime,
        #  reference ts, per-model trusted betas]
        self._pending: deque[list[Any]] = deque()
        self._unref: deque[list[Any]] = deque()
        self.decisions: deque[dict[str, Any]] = deque(maxlen=100)
        self.ticks = 0
        self.orders_sent = 0
        self.paused = False  # operator switch: keep watching and learning, send no orders
        self.halted_ticks = 0  # ticks on which the safety kernel blocked trading
        self.max_abs_exposure = 0.0  # largest |notional / equity| actually held
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

    def on_chart(self, tf_score: tuple[str, float]) -> None:
        tf, score = tf_score
        if tf in self.market.chart_scores:
            self.market.chart_scores[tf] = score

    # --- learned state: survives restarts --------------------------------
    def _signature(self) -> dict[str, Any]:
        return {
            "features": list(FEATURE_NAMES), "horizon_s": self.s.horizon_s, "interval_s": self.s.decision_interval_s,
            "models": [(e.model.name, e.feature_names) for e in self.arena.entries],
        }  # fmt: skip

    def dump_state(self) -> bytes:
        return pickle.dumps({
            "sig": self._signature(), "std": self.std, "entries": self.arena.entries,
            "champion": self.arena.champion, "promotions": self.arena.promotions, "half_life": self.half_life,
        })  # fmt: skip

    def load_state(self, blob: bytes) -> str:
        """Restore learned state. Returns '' on success, otherwise why it was not used."""
        try:
            st = pickle.loads(blob)  # noqa: S301
        except Exception as ex:  # noqa: BLE001
            return f"unreadable state file ({type(ex).__name__})"
        if st.get("sig") != self._signature():
            return "state was saved with different features, models or horizon"
        self.std, self.half_life = st["std"], st["half_life"]
        self.half_life.prev = None
        self.arena.entries, self.arena.champion, self.arena.promotions = st["entries"], st["champion"], st["promotions"]
        for e in self.arena.entries:
            if hasattr(e.model, "set_async"):
                e.model.set_async(self.s.mode is not Mode.BACKTEST)
        return ""

    def flatten(self, now: float) -> str:
        """Operator emergency stop: pause, cancel resting orders, close the whole position at market."""
        self.paused = True
        acct, book = self.venue.account(now), self.market.book
        self.venue.cancel_all(now)
        if not acct.known or book is None or not book.valid():
            return "paused; the position is unknown right now, so nothing was sent - check the exchange directly"
        if acct.position == 0:
            return "paused; there was no position to close"
        is_buy = acct.position < 0
        px = self.meta.round_px(book.best_ask * 1.01 if is_buy else book.best_bid * 0.99)
        self.venue.submit(OrderIntent(self.meta.coin, is_buy, abs(acct.position), px, "Ioc", True, client_id="flatten"), now)
        self.orders_sent += 1
        return f"paused; closing {abs(acct.position):g} {self.meta.coin} at market"

    def on_news(self, score: float) -> None:
        self.market.news_score = score

    def _on_promotion(self, event: dict[str, Any]) -> None:
        self.half_life.prev = None
        if self.sink:
            self.sink.put("promotions", event)

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
                _, xs, mid0, mus, regime, _, betas = self._pending.popleft()
                if mid0 is None:
                    continue
                self.arena.resolve(now, xs, mus, betas, math.log(book.mid / mid0) * 1e4, regime)

        feats = mkt.features(now) if not faults else None
        if faults or feats is None or book is None:
            self.state = "HALTED_INSTRUMENTATION" if faults else "OBSERVING"
            self.last = {"ts": now, "state": self.state, "faults": faults, "warmup": feats is None}
            if faults:
                self.halted_ticks += 1
                self._unref.clear()
                self._pending.clear()  # forecasts spanning a data gap would be scored against garbage
                self.half_life.prev = None
            return None

        xs, mus, pvars, betas = self.arena.predict(self.std.transform(feats.values), feats.regime, feats.values)
        item = [now + s.horizon_s, xs, None, mus, feats.regime, now + s.latency_ms / 1000, betas]
        self._pending.append(item)
        self._unref.append(item)
        c = self.arena.champion  # only the champion's forecast is traded
        champ = self.arena.champ
        x, mu_raw, pvar, beta = xs[c], mus[c], pvars[c], betas[c]
        hl = self.half_life.update(mu_raw)
        sigma_h = max(feats.sigma_tick * math.sqrt(s.horizon_ticks) * 1e4, math.sqrt(champ.model.resid_var))
        fc = Forecast(
            mu_bps=beta * mu_raw, mu_raw_bps=mu_raw, sigma_bps=sigma_h, param_sigma_bps=math.sqrt(pvar),
            beta=beta, half_life_s=hl, horizon_s=s.horizon_s, sell_rate=feats.sell_rate,
            buy_rate=feats.buy_rate, maker_adverse_bps=self.journal.maker_adverse_bps(),
        )  # fmt: skip
        d = decide(now, fc, acct, book, self.meta, s)
        self.max_abs_exposure = max(self.max_abs_exposure, abs(d.f_current))

        w = getattr(champ.model, "w", None)  # linear models can explain themselves; others cannot
        contrib = w[:-1] * x[:-1] if w is not None else np.zeros(0)
        order = np.argsort(-contrib * np.sign(mu_raw if mu_raw else 1.0))
        names = champ.feature_names
        d.extra = {
            "champion": champ.model.name,
            "regime": {k: round(float(v), 3) for k, v in zip(REGIMES, feats.regime, strict=True)},
            "mu_raw_bps": mu_raw, "sigma_bps": sigma_h, "param_sigma_bps": fc.param_sigma_bps,
            "beta": beta, "half_life_s": hl,
            "supporting": [(names[i], round(float(contrib[i]), 3)) for i in order[:4]],
            "contradicting": [(names[i], round(float(contrib[i]), 3)) for i in order[-3:][::-1]],
        }  # fmt: skip

        if d.order is not None:
            if acct.inflight:
                d.action, d.reason, d.order = "HOLD", "waiting for a previous order to be acknowledged", None
            elif s.mode is Mode.SHADOW:
                d.extra["shadow"] = True
            elif self.paused:
                d.extra["paused"] = True  # decided, deliberately not sent
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
        champ = self.arena.champ
        slope, lcb, n = champ.cal.per_regime()
        w = getattr(champ.model, "w", None)
        b = self.market.book
        return {
            "mode": self.s.mode.value, "coin": self.meta.coin, "state": self.state, "ticks": self.ticks,
            "mid": b.mid if b is not None and b.valid() else None,
            "last": self.last, "journal": self.journal.summary(), "orders_sent": self.orders_sent,
            "model": {
                "champion": champ.model.name, "arena": self.arena.snapshot(), "promotions": self.arena.promotions[-10:],
                "resolved_forecasts": champ.model.n_obs, "oos_ic": champ.cal.ic(),
                "resid_std_bps": math.sqrt(champ.model.resid_var),
                "calibration": {r: {"slope": float(slope[i]), "trusted_beta": float(lcb[i]),
                                    "indep_samples": float(n[i])} for i, r in enumerate(REGIMES)},
                "weights": dict(zip(champ.feature_names, (round(float(v), 4) for v in w[:-1]), strict=True))
                if w is not None else {},
            },
            "recent_decisions": list(self.decisions)[-20:],
        }  # fmt: skip
