"""Venue-neutral types. Strategy code depends on these, never on Hyperliquid internals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(slots=True)
class Book:
    coin: str
    ts: float  # local receive time, seconds
    exch_ts: float
    bids: np.ndarray  # (n, 2) [px, sz], best first
    asks: np.ndarray

    def valid(self) -> bool:
        return len(self.bids) > 0 and len(self.asks) > 0 and self.bids[0, 0] < self.asks[0, 0]

    @property
    def best_bid(self) -> float:
        return float(self.bids[0, 0])

    @property
    def best_ask(self) -> float:
        return float(self.asks[0, 0])

    @property
    def mid(self) -> float:
        return 0.5 * (self.best_bid + self.best_ask)


@dataclass(slots=True)
class Trade:
    ts: float
    px: float
    sz: float
    is_buy: bool  # aggressor side


@dataclass(slots=True)
class AssetCtx:
    ts: float
    funding: float  # hourly rate
    premium: float
    oracle_px: float
    mark_px: float
    open_interest: float


@dataclass(slots=True)
class OrderIntent:
    coin: str
    is_buy: bool
    sz: float
    limit_px: float
    tif: str  # "Ioc" | "Alo" | "Gtc"
    reduce_only: bool = False
    ttl_s: float = 0.0  # resting lifetime for Alo/Gtc
    client_id: str = ""


@dataclass(slots=True)
class Fill:
    ts: float
    coin: str
    is_buy: bool
    px: float
    sz: float
    fee: float
    maker: bool
    client_id: str = ""
    liquidation: bool = False
    mid: float = 0.0  # mid just before the fill, when the venue knows it (0 = unknown)


@dataclass(slots=True)
class AccountState:
    ts: float
    known: bool  # False => the kernel must block trading
    equity: float = 0.0
    position: float = 0.0  # signed size in coin
    entry_px: float = 0.0
    open_orders: int = 0
    inflight: int = 0  # submitted, not yet acknowledged
    oldest_inflight_ts: float = 0.0
    working: tuple[str, ...] = ()  # client ids of our resting orders, as of working_ts
    working_ts: float = 0.0


@dataclass(frozen=True)
class AssetMeta:
    coin: str
    sz_decimals: int
    max_leverage: float
    min_notional: float = 10.0

    @property
    def maintenance_margin(self) -> float:
        # Hyperliquid: maintenance margin is half the initial margin at max leverage.
        return 1.0 / (2.0 * self.max_leverage)

    def tick(self, px: float) -> float:
        """Smallest price step at this price: 5 significant figures, at most (6 - szDecimals) decimals."""
        return max(min(10.0 ** (math.floor(math.log10(px)) - 4), 1.0), 10.0 ** -(6 - self.sz_decimals))  # integers always valid

    def round_sz(self, sz: float) -> float:
        q = 10**self.sz_decimals
        return math.floor(sz * q + 1e-9) / q

    def round_px(self, px: float) -> float:
        # Perp prices: at most 5 significant figures and (6 - szDecimals) decimals; integers always ok.
        if px >= 100_000:
            return float(round(px))
        return round(float(f"{px:.5g}"), 6 - self.sz_decimals)


class Venue(Protocol):
    def submit(self, intent: OrderIntent, now: float) -> None: ...
    def cancel_all(self, now: float) -> None: ...
    def cancel(self, client_id: str, now: float) -> None: ...
    def account(self, now: float) -> AccountState: ...
    def drain_fills(self) -> list[Fill]: ...
    def on_book(self, book: Book) -> None: ...
    def on_trades(self, trades: list[Trade]) -> None: ...
    def on_ctx(self, ctx: AssetCtx) -> None: ...


def merge_bbo(prev: Book | None, coin: str, ts: float, bbo: tuple[float, float, float, float, float]) -> Book:
    """Refresh the top of book from a BBO update, keeping deeper levels from the last snapshot."""
    bp, bq, ap, aq, exch_ts = bbo
    bids, asks = np.array([[bp, bq]]), np.array([[ap, aq]])
    if prev is not None:
        bids = np.vstack([bids, prev.bids[prev.bids[:, 0] < bp]])
        asks = np.vstack([asks, prev.asks[prev.asks[:, 0] > ap]])
    return Book(coin, ts, exch_ts, bids, asks)


def impact_frac(levels: np.ndarray, mid: float, notional: float) -> float:
    """Average execution price vs mid, as a fraction, for taking `notional` from one side."""
    px, sz = levels[:, 0], levels[:, 1]
    if notional <= 0:
        return abs(float(px[0]) - mid) / mid
    cum = np.cumsum(px * sz)
    if notional > cum[-1]:
        return 1.0  # visible depth is insufficient: prohibitive
    k = int(np.searchsorted(cum, notional))
    prev = float(cum[k - 1]) if k > 0 else 0.0
    qty = float(sz[:k].sum()) + (notional - prev) / float(px[k])
    return abs(notional / qty - mid) / mid
