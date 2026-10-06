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
from app.exchange.paper import PaperVenue
from app.learning.journal import Journal
from app.market.state import FEATURE_NAMES, REGIMES, MarketState
from app.models.online import HalfLife, Standardizer
from app.models.tree import TreeForecaster
from app.risk.kernel import SafetyKernel
from app.scalp.alpha import FastAlpha
from app.scalp.lessons import QuoteLedger, TakeLedger
from app.scalp.quoter import Desired, QuoteInputs, QuoteManager, desired_quotes


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
        # Scalper (strategy == "maker"): two-sided quoting, re-evaluated several times a second.
        self.maker = s.strategy == "maker"
        self.quotes = QuoteManager(meta.coin, s.scalp_min_requote_s, actions_per_min=s.scalp_actions_per_min)
        self._quote_inputs: QuoteInputs | None = None
        self.fast_alpha = FastAlpha(interval_s=s.decision_interval_s)
        self._quote_ok = False  # set by the 1-second tick: instruments healthy and inputs fresh
        self._last_quote_ts = 0.0
        self.quote_cycles = self.quoting_cycles = 0
        self.takes = 0  # times the scalper crossed the spread on a strong fast forecast
        self.making_off_until = 0.0  # passive quoting is rested while its own fills show it losing
        self.making_timeouts = 0
        # Practice book: the same quoting logic runs against a simulator on the live feed, with pretend
        # money, all the time. Real passive quotes are allowed only while this practice shows that resting
        # quotes are worth more than their fee. The scalper earns the right to quote before it risks a cent.
        self.practice_venue = PaperVenue(1000.0, meta, s.taker_fee, s.maker_fee, s.latency_ms / 1000)
        self.practice_quotes = QuoteManager(meta.coin, s.scalp_min_requote_s, actions_per_min=s.scalp_actions_per_min)
        self.practice = Journal()
        self.making_allowed = False
        # Lessons: every second, score the quotes and takes it could have made, filed by situation.
        self.quote_lessons = QuoteLedger()
        self.take_lessons = TakeLedger()
        self._raw_alpha = 0.0
        self._tick_trades: list[tuple[float, float, bool]] = []  # trades since the last tick, for the ledgers
        self._last_take_ts = 0.0
        self.last_quotes: dict[str, Any] = {}
        self.halted_ticks = 0  # ticks on which the safety kernel blocked trading
        self.max_abs_exposure = 0.0  # largest |notional / equity| actually held
        self.state = "OBSERVING"
        self.last: dict[str, Any] = {}

    # --- market events (hot path: memory only) --------------------------
    def on_book(self, book: Book) -> None:
        self.market.on_book(book)
        self.venue.on_book(book)
        if self.maker:
            self.practice_venue.on_book(book)
        # Forecasts are scored from the first price we could actually have traded at
        # (decision time + latency), not from the price we were looking at when deciding.
        while self._unref and self._unref[0][5] <= book.ts:
            item = self._unref.popleft()
            if book.valid():
                item[2] = book.mid
        if self.maker and book.ts - self._last_quote_ts >= self.s.scalp_quote_interval_s:
            self._last_quote_ts = book.ts
            self._quote_cycle(book.ts)

    # --- scalper -----------------------------------------------------------
    def _take_fills(self) -> None:
        book = self.market.book
        mid = book.mid if book is not None and book.valid() else None
        for f in self.venue.drain_fills():
            self.journal.on_fill(f, mid)
            self.kernel.on_fill(f.sz if f.is_buy else -f.sz)
            self.quotes.on_fill(f, 10.0**-self.meta.sz_decimals)
            if self.sink:
                self.sink.put("fills", {"ts": f.ts, "is_buy": f.is_buy, "px": f.px, "sz": f.sz, "fee": f.fee,
                                        "maker": f.maker, "liquidation": f.liquidation})  # fmt: skip

    def _apply(self, actions: list[tuple[str, Any]], now: float) -> None:
        for kind, arg in actions:
            if kind == "cancel":
                self.venue.cancel(arg, now)
            else:
                self.venue.submit(arg, now)
                self.orders_sent += 1

    def pull_quotes(self, now: float) -> None:
        self._apply(self.quotes.pull_all(), now)

    def _apply_lessons(self, want: Desired, acct: Any, book: Book) -> None:
        """Where the record is long enough to decide, it decides: quote where it says a hit pays, nowhere it says it loses."""
        if not self.s.scalp_lessons:
            return
        fee, tick = self.s.maker_fee * 1e4, self.meta.tick(book.mid)
        for is_buy in (True, False):
            cur = want.bid if is_buy else want.ask
            name = "bid" if is_buy else "ask"
            if not self.quote_lessons.informed(is_buy, self._raw_alpha):
                want.info[f"lesson_{name}"] = "no record yet"
                continue
            dist = self.quote_lessons.best_distance(is_buy, self._raw_alpha, fee)
            reducing = acct.position < 0 if is_buy else acct.position > 0
            if dist is None:
                want.info[f"lesson_{name}"] = "avoid"
                if not reducing:  # a quote that only reduces inventory is kept: getting flat is not a bet
                    if is_buy:
                        want.bid = None
                    else:
                        want.ask = None
                continue
            want.info[f"lesson_{name}"] = f"{dist:g} bps behind"
            if cur is None:
                continue  # inventory limit or size already ruled this side out
            if is_buy:
                cur.px = self.meta.round_px(min(math.floor(book.best_bid * (1 - dist * 1e-4) / tick + 1e-9) * tick, book.best_ask - tick))
            else:
                cur.px = self.meta.round_px(max(math.ceil(book.best_ask * (1 + dist * 1e-4) / tick - 1e-9) * tick, book.best_bid + tick))

    def _practice_cycle(self, now: float, book: Book, q: QuoteInputs) -> None:
        """Quote against the simulator with pretend money and keep score. Runs whether or not real trading is on."""
        v, lot = self.practice_venue, 10.0**-self.meta.sz_decimals
        for f in v.drain_fills():
            self.practice.on_fill(f, book.mid)
            self.practice_quotes.on_fill(f, lot)
        acct = v.account(now)
        if not acct.known or acct.equity <= 0:
            return
        prior = (book.best_ask - book.best_bid) / 2 / book.mid * 1e4
        pq = QuoteInputs(q.sigma_tick, q.tick_s, q.flow, q.alpha_bps, q.horizon_s, self.practice.adverse_bps(True, prior),
                         self.practice.adverse_bps(False, prior), q.fast_alpha_bps)  # fmt: skip
        want = desired_quotes(book, acct, pq, self.meta, self.s)
        self._apply_lessons(want, acct, book)
        for kind, arg in self.practice_quotes.reconcile(want, acct, now, self.meta.tick(book.mid)):
            if kind == "cancel":
                v.cancel(arg, now)
            else:
                v.submit(arg, now)
        f, limit = want.info["inventory_x"], want.info["inventory_limit_x"]
        if abs(f) > 1.5 * limit and not acct.inflight:  # same inventory discipline as the real book
            sz = self.meta.round_sz(min((abs(f) - limit) * acct.equity / book.mid, abs(acct.position)))
            px = self.meta.round_px(book.best_ask * 1.01 if f < 0 else book.best_bid * 0.99)
            if sz * book.mid >= self.meta.min_notional:
                v.submit(OrderIntent(self.meta.coin, f < 0, sz, px, "Ioc", True, client_id="inventory"), now)
        edge, n = self.practice.maker_edge()
        gate = self.s.scalp_gate_fills  # <= 0 switches the requirement off (tests and experiments only)
        # Not just a positive average: positive after allowing for luck (lower confidence bound), so it does not flap.
        self.making_allowed = gate <= 0 or (n >= gate and self.practice.maker_edge_lcb() > self.s.maker_fee * 1e4)

    def _quote_cycle(self, now: float) -> None:
        """Decide both quotes and send the fewest order actions that get the book there."""
        self._take_fills()
        book, q, s = self.market.book, self._quote_inputs, self.s
        self.quote_cycles += 1
        if self._quote_ok and q is not None and book is not None and book.valid():
            m0 = self.market.micro(now)
            q.fast_alpha_bps = self.fast_alpha.predict(FastAlpha.vector(*m0)) if m0 is not None else 0.0
            self._practice_cycle(now, book, q)
        live = self._quote_ok and not self.paused and s.mode is not Mode.SHADOW
        if not live or q is None or book is None or not book.valid():
            if self.quotes.working:
                self.pull_quotes(now)
            return
        acct = self.venue.account(now)
        if not acct.known or acct.equity <= 0:
            self.pull_quotes(now)
            return
        prior = (book.best_ask - book.best_bid) / 2 / book.mid * 1e4
        # What practice has measured is the starting belief; real fills then move it.
        q.adverse_bid_bps = self.journal.adverse_bps(True, self.practice.adverse_bps(True, prior))
        q.adverse_ask_bps = self.journal.adverse_bps(False, self.practice.adverse_bps(False, prior))
        want: Desired = desired_quotes(book, acct, q, self.meta, s)
        self._apply_lessons(want, acct, book)
        # Evidence gate, the same rule the forecasts live under: once enough passive fills exist, resting
        # quotes must be worth more than their fee 5 s later, or passive quoting is rested for a while.
        edge, n_fills = self.journal.maker_edge()
        if 0 < s.scalp_gate_fills <= n_fills and edge < s.maker_fee * 1e4 and now >= self.making_off_until:
            self.making_off_until = now + s.scalp_gate_cooldown_s
            self.making_timeouts += 1
            self.journal.retest_maker()
        if now < self.making_off_until or not self.making_allowed:  # keep only a quote that reduces inventory
            if acct.position <= 0:
                want.ask = None
            if acct.position >= 0:
                want.bid = None
            want.info["making"] = ("resting: real passive fills have been losing" if now < self.making_off_until
                                   else "not yet: practice quotes have not shown a profit after fees")  # fmt: skip
        f = want.info["inventory_x"]
        limit = want.info["inventory_limit_x"]
        self.max_abs_exposure = max(self.max_abs_exposure, abs(f))
        if abs(f) > 1.5 * limit and not acct.inflight:
            # Inventory far past its limit (a burst of fills, or the balance fell): cut the excess now.
            excess = (abs(f) - limit) * acct.equity / book.mid
            is_buy = f < 0
            px = self.meta.round_px(book.best_ask * 1.01 if is_buy else book.best_bid * 0.99)
            sz = self.meta.round_sz(min(excess, abs(acct.position)))
            if sz * book.mid >= self.meta.min_notional:
                self.venue.submit(OrderIntent(self.meta.coin, is_buy, sz, px, "Ioc", True, client_id="inventory"), now)
                self.orders_sent += 1
        # Take liquidity when the fast forecast alone pays for the taker fee and the spread. With real
        # fees this is rare by construction; it is the scalper's other hand, not its habit.
        # The decision is the record's, not the forecast's: take only where takes at this forecast strength have,
        # at their lower confidence bound, been worth more than the fee after paying the spread.
        if s.scalp_lessons:
            fa = self._raw_alpha
            take = fa != 0.0 and self.take_lessons.allows(fa, s.taker_fee * 1e4, s.scalp_take_margin_bps)
        else:  # without the record: trust the calibrated forecast against fee and spread
            fa = q.fast_alpha_bps
            take = abs(fa) > s.taker_fee * 1e4 + want.info["half_spread_bps"] + s.scalp_take_margin_bps
        if take and not acct.inflight and now - self._last_take_ts >= 1.0:
            is_buy = fa > 0
            room = (limit - f if is_buy else limit + f) * acct.equity
            notional = min(max(self.meta.min_notional * 1.1, acct.equity * s.scalp_clip_x), max(room, 0.0))
            px = book.best_ask if is_buy else book.best_bid  # at the touch only: never chase
            sz = self.meta.round_sz(notional / px)
            if sz * px >= self.meta.min_notional:
                self.venue.submit(OrderIntent(self.meta.coin, is_buy, sz, px, "Ioc", False, client_id="take"), now)
                self.orders_sent += 1
                self.takes += 1
                self._last_take_ts = now
        self._apply(self.quotes.reconcile(want, acct, now, self.meta.tick(book.mid)), now)
        self.quoting_cycles += bool(self.quotes.working)
        self.last_quotes = want.info | {"working": {("bid" if k else "ask"): w.px for k, w in self.quotes.working.items()}}

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

    def experience(self) -> int:
        """Seconds of market it has learned from: used to prefer the more experienced of two saved states."""
        return int(self.arena.champ.model.n_obs + self.fast_alpha.model.n_obs)

    def dump_state(self, slim: bool = False) -> bytes:
        """Everything learned. `slim` drops the tree model's raw training rows (for a seed that ships with the code)."""
        TreeForecaster.slim_pickle = slim
        try:
            return pickle.dumps({
                "sig": self._signature(), "std": self.std, "entries": self.arena.entries,
                "champion": self.arena.champion, "promotions": self.arena.promotions, "half_life": self.half_life,
                "fast_alpha": self.fast_alpha, "practice": self.practice, "quote_lessons": self.quote_lessons,
                "take_lessons": self.take_lessons, "experience": self.experience(),
            })  # fmt: skip
        finally:
            TreeForecaster.slim_pickle = False

    def peek_experience(self, blob: bytes) -> int:
        """How experienced a saved state is, or -1 if it cannot be used by this engine."""
        try:
            st = pickle.loads(blob)  # noqa: S301
        except Exception:  # noqa: BLE001
            return -1
        if not isinstance(st, dict) or st.get("sig") != self._signature():
            return -1
        return int(st.get("experience", st["entries"][st["champion"]].model.n_obs))

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
        if isinstance(st.get("fast_alpha"), FastAlpha):
            self.fast_alpha = st["fast_alpha"]
            self.fast_alpha.reset()
        if isinstance(st.get("practice"), Journal):
            self.practice = st["practice"]
        if isinstance(st.get("quote_lessons"), QuoteLedger) and isinstance(st.get("take_lessons"), TakeLedger):
            self.quote_lessons, self.take_lessons = st["quote_lessons"], st["take_lessons"]
            self.quote_lessons._open.clear()  # hypothetical quotes from before the restart cannot be scored fairly
            self.quote_lessons._filled.clear()
            self.take_lessons._pending.clear()
        for e in self.arena.entries:
            if hasattr(e.model, "set_async"):
                e.model.set_async(self.s.mode is not Mode.BACKTEST)
        return ""

    def flatten(self, now: float) -> str:
        """Operator emergency stop: pause, cancel resting orders, close the whole position at market."""
        self.paused = True
        acct, book = self.venue.account(now), self.market.book
        self.quotes.pull_all()
        self.venue.cancel_all(now)
        if not acct.known or book is None or not book.valid():
            return "Stopped. The position could not be read just now, so nothing was sent - check Hyperliquid directly."
        if acct.position == 0:
            return "Stopped. There was no position to close."
        is_buy = acct.position < 0
        px = self.meta.round_px(book.best_ask * 1.01 if is_buy else book.best_bid * 0.99)
        self.venue.submit(OrderIntent(self.meta.coin, is_buy, abs(acct.position), px, "Ioc", True, client_id="flatten"), now)
        self.orders_sent += 1
        return f"Stopped. Closing {abs(acct.position):g} {self.meta.coin} at market."

    def on_news(self, score: float) -> None:
        self.market.news_score = score

    def _on_promotion(self, event: dict[str, Any]) -> None:
        self.half_life.prev = None
        if self.sink:
            self.sink.put("promotions", event)

    def on_trades(self, trades: list[Trade]) -> None:
        self.market.on_trades(trades)
        self.venue.on_trades(trades)
        if self.maker:
            self.practice_venue.on_trades(trades)
            self._tick_trades.extend((t.px, t.sz, t.is_buy) for t in trades)

    def on_ctx(self, ctx: AssetCtx) -> None:
        self.market.on_ctx(ctx)
        self.venue.on_ctx(ctx)

    # --- decision tick ---------------------------------------------------
    def on_tick(self, now: float) -> Decision | None:
        s, mkt = self.s, self.market
        self.ticks += 1
        mkt.sample(now)
        self._take_fills()

        acct = self.venue.account(now)
        faults = self.kernel.check(now, mkt.book, acct, 10.0**-self.meta.sz_decimals, mkt.feed_ts)
        book = mkt.book
        if book is not None and book.valid() and acct.known:
            self.journal.on_tick(now, book.mid, acct.equity)
            if self.maker:
                pa = self.practice_venue.account(now)
                if pa.known:
                    self.practice.on_tick(now, book.mid, pa.equity)
            if self.maker and not faults and (m := mkt.micro(now)) is not None:
                xv = FastAlpha.vector(*m)
                self.fast_alpha.on_tick(now, xv, book.mid)
                self._raw_alpha = self.fast_alpha.raw(xv)
                self.quote_lessons.on_tick(now, book.best_bid, book.best_ask, float(book.bids[0, 1]), float(book.asks[0, 1]),
                                           self.meta.tick(book.mid), self._raw_alpha, self._tick_trades)  # fmt: skip
                self.take_lessons.on_tick(now, book.best_bid, book.best_ask, self._raw_alpha)
            self._tick_trades = []
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
            self._quote_ok = False  # the scalper pulls its quotes on the next cycle
            if faults:
                self.fast_alpha.reset()
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
        if self.maker:
            # The scalper trades by quoting, not by this directional decision; the forecast only leans its quotes.
            tfi5 = float(feats.values[FEATURE_NAMES.index("tfi_5s")])
            self._quote_inputs = QuoteInputs(feats.sigma_tick, s.decision_interval_s, tfi5, fc.mu_bps, s.horizon_s, 0.0, 0.0)
            self._quote_ok = True
            d.action, d.order, d.f_target = "HOLD", None, d.f_current
            d.reason = "scalping: quoting both sides" if self.quotes.working else "scalping: no quote currently worth resting"
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
            "scalper": {
                "enabled": self.maker, "quotes": self.last_quotes, "orders_placed": self.quotes.placed,
                "orders_cancelled": self.quotes.cancelled, "actions_per_min": self.quotes.actions_per_min,
                "skipped_for_budget": self.quotes.skipped_for_budget, "fast_alpha": self.fast_alpha.snapshot(),
                "takes": self.takes, "making_timeouts": self.making_timeouts,
                "passive_edge_bps": self.journal.maker_edge()[0], "passive_fills_judged": self.journal.maker_edge()[1],
                "making_allowed": self.making_allowed,
                "lessons": {"quotes": self.quote_lessons.table(self.s.maker_fee * 1e4),
                            "takes": self.take_lessons.table(self.s.taker_fee * 1e4)},
                "practice": {
                    "fills": self.practice.n_fills, "edge_bps_5s": self.practice.maker_edge()[0],
                    "edge_lower_bound_bps": self.practice.maker_edge_lcb(),
                    "needed_bps": self.s.maker_fee * 1e4, "fills_needed": self.s.scalp_gate_fills,
                    "pnl_pct": self.practice.summary()["net_return_pct"],
                },
                "quote_uptime_pct": 100.0 * self.quoting_cycles / self.quote_cycles if self.quote_cycles else 0.0,
            },
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
