"""Simulated venue driven by real (or replayed) market data.

Conservative by construction: orders become active only after `latency_s`, taker orders walk
the visible book, and resting orders fill only after the queue that was ahead of them has
traded (or the market trades through their price).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.exchange.base import AccountState, AssetCtx, AssetMeta, Book, Fill, OrderIntent, Trade


@dataclass(slots=True)
class _Resting:
    intent: OrderIntent
    remaining: float
    queue_ahead: float
    expires: float


class PaperVenue:
    def __init__(
        self, equity: float, meta: AssetMeta, taker_fee: float, maker_fee: float, latency_s: float = 0.15
    ) -> None:
        self.meta = meta
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.latency_s = latency_s
        self.cash = equity  # realized equity
        self.pos = 0.0
        self.entry = 0.0
        self.book: Book | None = None
        self.fees_paid = 0.0
        self.funding_paid = 0.0
        self.rejects = 0
        self.liquidations = 0
        self._pending: list[tuple[float, OrderIntent]] = []
        self._pending_cancels: list[tuple[float, str]] = []
        self._resting: list[_Resting] = []
        self._fills: list[Fill] = []
        self._funding_hour: int | None = None
        self._ref_mid = 0.0

    # --- Venue protocol -------------------------------------------------
    def submit(self, intent: OrderIntent, now: float) -> None:
        self._pending.append((now + self.latency_s, intent))

    def cancel_all(self, now: float) -> None:
        self._resting.clear()
        self._pending_cancels.clear()

    def cancel(self, client_id: str, now: float) -> None:
        """A cancel travels as slowly as an order: the quote can still be hit before it lands."""
        self._pending_cancels.append((now + self.latency_s, client_id))

    def account(self, now: float) -> AccountState:
        if self.book is None or not self.book.valid():
            return AccountState(now, known=False)
        oldest = min((t - self.latency_s for t, _ in self._pending), default=0.0)
        working = tuple(o.intent.client_id for o in self._resting) + tuple(it.client_id for _, it in self._pending)
        return AccountState(
            now, True, self._equity(self.book.mid), self.pos, self.entry, len(self._resting),
            len(self._pending), oldest, working, now,
        )  # fmt: skip

    def drain_fills(self) -> list[Fill]:
        out, self._fills = self._fills, []
        return out

    def on_book(self, book: Book) -> None:
        if self.book is not None and self.book.valid():
            self._ref_mid = self.book.mid  # the mid before this update: the reference for fills it causes
        self.book = book
        if not book.valid():
            return
        if not self._ref_mid:
            self._ref_mid = book.mid
        ready = [it for t, it in self._pending if t <= book.ts]
        self._pending = [(t, it) for t, it in self._pending if t > book.ts]
        for it in ready:
            self._activate(it, book)
        due = {cid for t, cid in self._pending_cancels if t <= book.ts}
        if due:
            self._pending_cancels = [(t, cid) for t, cid in self._pending_cancels if t > book.ts]
            self._resting = [o for o in self._resting if o.intent.client_id not in due]
            self._pending = [(t, it) for t, it in self._pending if it.client_id not in due]
        for o in list(self._resting):
            crossed = book.best_ask <= o.intent.limit_px if o.intent.is_buy else book.best_bid >= o.intent.limit_px
            if crossed:
                self._fill(o.intent, o.intent.limit_px, o.remaining, True, book.ts)
                self._resting.remove(o)
            elif o.expires <= book.ts:
                self._resting.remove(o)
        self._check_liquidation(book)

    def on_trades(self, trades: list[Trade]) -> None:
        if self.book is not None and self.book.valid():
            self._ref_mid = self.book.mid
        for t in trades:
            for o in list(self._resting):
                if o.intent.is_buy == t.is_buy:
                    continue  # same-side aggressor cannot hit our order
                px = o.intent.limit_px
                through = t.px < px if o.intent.is_buy else t.px > px
                if through:
                    qty = o.remaining
                elif t.px == px:
                    o.queue_ahead -= t.sz
                    qty = min(-o.queue_ahead, o.remaining) if o.queue_ahead < 0 else 0.0
                    o.queue_ahead = max(o.queue_ahead, 0.0)
                else:
                    continue
                if qty > 0:
                    self._fill(o.intent, px, qty, True, t.ts)
                    o.remaining -= qty
                    if o.remaining <= 1e-12:
                        self._resting.remove(o)

    def on_ctx(self, ctx: AssetCtx) -> None:
        hour = int(ctx.ts // 3600)
        if self._funding_hour is not None and hour != self._funding_hour and self.pos:
            pay = self.pos * ctx.mark_px * ctx.funding  # longs pay positive funding
            self.cash -= pay
            self.funding_paid += pay
        self._funding_hour = hour

    # --- internals -------------------------------------------------------
    def _equity(self, mid: float) -> float:
        return self.cash + self.pos * (mid - self.entry)

    def _activate(self, it: OrderIntent, book: Book) -> None:
        sz = it.sz
        if it.reduce_only:
            if self.pos == 0 or (self.pos > 0) == it.is_buy:
                self.rejects += 1
                return
            sz = min(sz, abs(self.pos))
        else:
            new_pos = self.pos + (sz if it.is_buy else -sz)
            if abs(new_pos) > abs(self.pos) and (
                abs(new_pos) * book.mid > self._equity(book.mid) * self.meta.max_leverage
            ):
                self.rejects += 1  # insufficient margin
                return
        levels = book.asks if it.is_buy else book.bids
        crosses = it.limit_px >= book.best_ask if it.is_buy else it.limit_px <= book.best_bid
        if it.tif == "Alo":
            if crosses:
                self.rejects += 1  # post-only would have taken
                return
            same = book.bids if it.is_buy else book.asks
            at_level = same[same[:, 0] == it.limit_px]
            queue = float(at_level[0, 1]) if len(at_level) else 0.0
            self._resting.append(_Resting(it, sz, queue, book.ts + it.ttl_s))
            return
        remaining = sz
        for px, avail in levels:
            if remaining <= 1e-12 or (px > it.limit_px if it.is_buy else px < it.limit_px):
                break
            q = min(remaining, float(avail))
            self._fill(it, float(px), q, False, book.ts)
            remaining -= q
        if remaining > 1e-12 and it.tif == "Gtc":
            self._resting.append(_Resting(it, remaining, 0.0, book.ts + it.ttl_s))

    def _fill(self, it: OrderIntent, px: float, sz: float, maker: bool, ts: float, liq: bool = False) -> None:
        signed = sz if it.is_buy else -sz
        if self.pos == 0 or (self.pos > 0) == it.is_buy:
            self.entry = (abs(self.pos) * self.entry + sz * px) / (abs(self.pos) + sz)
            self.pos += signed
        else:
            closing = min(sz, abs(self.pos))
            self.cash += closing * (px - self.entry) * (1 if self.pos > 0 else -1)
            self.pos += signed
            if abs(self.pos) < 1e-12:
                self.pos, self.entry = 0.0, 0.0
            elif sz > closing:
                self.entry = px  # flipped through zero
        fee = sz * px * (self.maker_fee if maker else self.taker_fee)
        self.cash -= fee
        self.fees_paid += fee
        ref = self._ref_mid if maker else (self.book.mid if self.book is not None else px)
        self._fills.append(Fill(ts, it.coin, it.is_buy, px, sz, fee, maker, it.client_id, liq, ref))

    def _check_liquidation(self, book: Book) -> None:
        if self.pos == 0:
            return
        mid = book.mid
        if self._equity(mid) <= self.meta.maintenance_margin * abs(self.pos) * mid:
            it = OrderIntent(self.meta.coin, self.pos < 0, abs(self.pos), mid, "Ioc", True)
            self._fill(it, mid, abs(self.pos), False, book.ts, liq=True)
            self.cash = 0.0  # pessimistic: assume nothing is returned after liquidation
            self.liquidations += 1
            self._resting.clear()
            self._pending.clear()
