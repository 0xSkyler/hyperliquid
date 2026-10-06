"""Scalping core: two-sided passive quoting (market making) with inventory and flow control.

A scalper earns the spread, not the direction. Each cycle this module decides, for each side,
the best price it is willing to rest at and how much, from five things:

1. Fair value  - the size-weighted microprice, nudged by whatever calibrated forecast the
                 models have earned (zero until they earn any).
2. Inventory   - quotes are shifted against the position so the side that reduces it is more
                 likely to fill, and the side that adds to it stops at the inventory limit.
3. Required edge - a quote must sit at least (maker fee + measured adverse selection + flow
                 penalty) away from fair value. Adverse selection is *learned from our own
                 fills*: how far the price moves against us in the seconds after one.
4. Toxic flow  - one-sided aggressive trading against a quote widens that side.
5. Queue priority - join the touch; step one tick inside it only when the spread is wide.

`QuoteManager` then turns desired quotes into the fewest order actions: it requotes at once when
a resting quote has become too aggressive (we are exposed), and lazily when it has merely
become less competitive, because every action spends the account's request budget.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

from app.config.settings import Settings
from app.exchange.base import AccountState, AssetMeta, Book, Fill, OrderIntent


@dataclass(slots=True)
class Quote:
    px: float
    sz: float


@dataclass(slots=True)
class Desired:
    bid: Quote | None
    ask: Quote | None
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QuoteInputs:
    sigma_tick: float  # per-tick log-return volatility (risk estimate)
    tick_s: float
    flow: float  # recent aggressor imbalance in [-1, 1]; positive = buying pressure
    alpha_bps: float  # calibrated forecast over the horizon; 0 when no model is trusted
    horizon_s: float
    adverse_bid_bps: float  # learned: how far mid falls after our bid is filled
    adverse_ask_bps: float
    fast_alpha_bps: float = 0.0  # calibrated forecast of the next few seconds from the top of the book


def desired_quotes(book: Book, acct: AccountState, q: QuoteInputs, meta: AssetMeta, s: Settings) -> Desired:
    mid, bid, ask = book.mid, book.best_bid, book.best_ask
    tick = meta.tick(mid)
    bq, aq = float(book.bids[0, 1]), float(book.asks[0, 1])
    half_spread_bps = (ask - bid) / 2 / mid * 1e4
    micro = (ask * bq + bid * aq) / (bq + aq) if bq + aq > 0 else mid
    hold_s = s.scalp_hold_s
    alpha_bps = q.alpha_bps * min(1.0, hold_s / q.horizon_s) + q.fast_alpha_bps
    sigma_hold_bps = q.sigma_tick * math.sqrt(max(hold_s / q.tick_s, 1.0)) * 1e4

    f = acct.position * mid / acct.equity
    f_max = min(s.scalp_inventory_x, meta.max_leverage)
    unit = max(half_spread_bps, sigma_hold_bps)
    skew_bps = -(f / f_max) * unit * s.scalp_skew
    reservation = micro * (1 + (alpha_bps + skew_bps) * 1e-4)

    fee_bps = s.maker_fee * 1e4
    need_bid = fee_bps + q.adverse_bid_bps + max(0.0, -q.flow) * s.scalp_toxicity * sigma_hold_bps
    need_ask = fee_bps + q.adverse_ask_bps + max(0.0, q.flow) * s.scalp_toxicity * sigma_hold_bps

    spread_ticks = round((ask - bid) / tick)
    inside = tick if spread_ticks >= 3 else 0.0  # only step inside a wide spread; otherwise join the queue
    bid_px = min(reservation * (1 - need_bid * 1e-4), bid + inside, ask - tick)
    ask_px = max(reservation * (1 + need_ask * 1e-4), ask - inside, bid + tick)
    bid_px = math.floor(bid_px / tick + 1e-9) * tick
    ask_px = math.ceil(ask_px / tick - 1e-9) * tick

    clip = max(meta.min_notional * 1.1, acct.equity * s.scalp_clip_x)
    pos_notional = abs(acct.position) * mid

    def size(is_buy: bool, px: float) -> float:
        reducing = (acct.position < 0) if is_buy else (acct.position > 0)
        room = (f_max - f if is_buy else f_max + f) * acct.equity
        notional = min(clip, max(room, 0.0))
        if reducing:
            notional = max(notional, min(pos_notional, clip))
        sz = meta.round_sz(notional / px)
        return sz if sz * px >= meta.min_notional else 0.0

    max_away = s.scalp_max_distance_bps * 1e-4  # a quote that far from the market only wastes requests
    out_bid = out_ask = None
    if (sz := size(True, bid_px)) > 0 and bid_px >= bid * (1 - max_away):
        out_bid = Quote(meta.round_px(bid_px), sz)
    if (sz := size(False, ask_px)) > 0 and ask_px <= ask * (1 + max_away):
        out_ask = Quote(meta.round_px(ask_px), sz)
    info = {
        "fair": micro, "half_spread_bps": half_spread_bps, "alpha_bps": alpha_bps, "skew_bps": skew_bps,
        "need_bid_bps": need_bid, "need_ask_bps": need_ask, "inventory_x": f, "inventory_limit_x": f_max,
        "bid": out_bid.px if out_bid else None, "ask": out_ask.px if out_ask else None,
        "bid_behind_touch_bps": (bid - out_bid.px) / mid * 1e4 if out_bid else None,
        "ask_behind_touch_bps": (out_ask.px - ask) / mid * 1e4 if out_ask else None,
    }  # fmt: skip
    return Desired(out_bid, out_ask, info)


@dataclass(slots=True)
class _Working:
    cid: str
    px: float
    sz: float
    ts: float


Action = tuple[str, Any]  # ("cancel", client_id) | ("place", OrderIntent)
_GRACE_S = 3.0  # how long a just-sent order may be missing from the venue's list before we believe it is gone


class QuoteManager:
    def __init__(self, coin: str, min_requote_s: float, ttl_s: float = 60.0, actions_per_min: float = 30.0) -> None:
        self.coin, self.min_requote_s, self.ttl_s = coin, min_requote_s, ttl_s
        # Hyperliquid gives each account a budget of order actions (10,000 plus one per dollar traded; once
        # spent, one action per 10 s). A token bucket keeps the scalper inside whatever rate it is given.
        self.actions_per_min = actions_per_min
        self._tokens = actions_per_min
        self._tokens_ts: float | None = None
        self.skipped_for_budget = 0
        self.working: dict[bool, _Working] = {}
        self._last_action = {True: -1e18, False: -1e18}
        self._seq = itertools.count(1)
        self.placed = self.cancelled = 0

    def _cid(self, now: float) -> str:
        # 0x + 32 hex chars: valid as a Hyperliquid client order id.
        return f"0x{int(now * 1000) & 0xFFFFFFFFFFFF:012x}{next(self._seq) & 0xFFFFFFFFFFFFFFFFFFFF:020x}"

    def on_fill(self, f: Fill, lot: float) -> None:
        for side, w in list(self.working.items()):
            if w.cid == f.client_id:
                w.sz -= f.sz
                if w.sz <= lot:
                    del self.working[side]

    def _refill(self, now: float) -> None:
        if self._tokens_ts is not None:
            cap = 2.0 * self.actions_per_min
            self._tokens = min(cap, self._tokens + (now - self._tokens_ts) * self.actions_per_min / 60.0)
        self._tokens_ts = now

    def reconcile(self, desired: Desired, acct: AccountState, now: float, tick: float) -> list[Action]:
        self._refill(now)
        actions: list[Action] = []
        for is_buy, want in ((True, desired.bid), (False, desired.ask)):
            w = self.working.get(is_buy)
            if w is not None and acct.working_ts > w.ts + _GRACE_S and w.cid not in acct.working:
                del self.working[is_buy]  # filled, expired or rejected: the venue no longer has it
                w = None
            if want is None:
                if w is not None:  # pulling a quote is always allowed, budget or not
                    actions.append(("cancel", w.cid))
                    self._tokens -= 1
                    del self.working[is_buy]
                    self._last_action[is_buy] = now
                continue
            if w is not None:
                # The quote already sits `need` bps from fair value for fees and adverse selection, so a
                # small drift is not worth an order action: requote only once it has moved a quarter of that.
                need_px = want.px * float(desired.info.get("need_bid_bps" if is_buy else "need_ask_bps", 0.0)) * 1e-4
                over = (w.px - want.px) if is_buy else (want.px - w.px)  # > 0: our quote is more aggressive than wanted
                moved = abs(over) >= max(tick * 0.999, 0.25 * need_px) or abs(w.sz - want.sz) > 0.25 * want.sz
                if not moved:
                    continue
                exposed = over > max(tick * 0.5, 0.5 * need_px)  # half the safety margin is gone: act now
                if not exposed and (now - self._last_action[is_buy] < self.min_requote_s or self._tokens < 2):
                    self.skipped_for_budget += self._tokens < 2
                    continue  # merely less competitive: wait, requests are a budget
                actions.append(("cancel", w.cid))
                self._tokens -= 1
                del self.working[is_buy]
                self._last_action[is_buy] = now
            elif now - self._last_action[is_buy] < self.min_requote_s:
                continue
            if self._tokens < 1:
                self.skipped_for_budget += 1
                continue  # out of budget: better no quote than a stale one
            self._tokens -= 1
            cid = self._cid(now)
            actions.append(("place", OrderIntent(self.coin, is_buy, want.sz, want.px, "Alo", False, self.ttl_s, cid)))
            self.working[is_buy] = _Working(cid, want.px, want.sz, now)
            self._last_action[is_buy] = now
        self.placed += sum(a[0] == "place" for a in actions)
        self.cancelled += sum(a[0] == "cancel" for a in actions)
        return actions

    def pull_all(self) -> list[Action]:
        actions: list[Action] = [("cancel", w.cid) for w in self.working.values()]
        self.cancelled += len(actions)
        self.working.clear()
        return actions
