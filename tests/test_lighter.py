from __future__ import annotations

import numpy as np

from app.exchange.base import AssetMeta, Book, OrderIntent
from app.exchange.hyperliquid import parse_event
from app.exchange.lighter import PREFIX, LocalBook, _levels, asset_meta
from app.exchange.paper import PaperVenue
from app.research.scalp_lab import coin_of

BTC = {"symbol": "BTC", "market_id": 1, "size_decimals": 5, "price_decimals": 1, "min_quote_amount": "10.000000",
       "min_initial_margin_fraction": 200}  # fmt: skip


def test_lighter_market_uses_a_fixed_price_grid_not_hyperliquids_rule() -> None:
    m = asset_meta("BTC", BTC)
    assert m.coin == "lighter:BTC" and m.max_leverage == 50.0 and m.min_notional == 10.0
    assert m.tick(84271.4) == 0.1 and m.round_px(84271.4321) == 84271.4 and m.round_px(84271.46) == 84271.5
    hl = AssetMeta("BTC", 5, 40.0)
    assert hl.tick(84271.4) == 1.0 and hl.round_px(84271.4321) == 84271.0  # unchanged for Hyperliquid


def test_local_book_applies_snapshot_then_updates_and_detects_corruption() -> None:
    b = LocalBook()
    b.apply({"bids": [{"price": "100.0", "size": "1"}, {"price": "99.9", "size": "2"}],
             "asks": [{"price": "100.2", "size": "3"}, {"price": "100.3", "size": "4"}]}, snapshot=True)  # fmt: skip
    b.apply({"bids": [{"price": "100.0", "size": "0"}, {"price": "100.1", "size": "5"}], "asks": [{"price": "100.2", "size": "0.5"}]},
            snapshot=False)  # fmt: skip
    bids, asks = b.top(10)
    assert bids == [(100.1, 5.0), (99.9, 2.0)] and asks == [(100.2, 0.5), (100.3, 4.0)]  # level removed, added, resized
    assert not b.crossed()
    b.apply({"bids": [{"price": "100.4", "size": "1"}]}, snapshot=False)  # a missed removal would look like this
    assert b.crossed()
    b.apply({"bids": [{"price": "50.0", "size": "1"}], "asks": [{"price": "51.0", "size": "1"}]}, snapshot=True)
    assert b.top(5) == ([(50.0, 1.0)], [(51.0, 1.0)])  # a fresh snapshot replaces everything


def test_lighter_records_replay_through_the_same_parser_as_hyperliquid() -> None:
    rec = {"coin": PREFIX + "BTC", "time": 1700000000000, "levels": [_levels([(100.1, 5.0)]), _levels([(100.2, 0.5)])]}
    kind, book = parse_event("l2Book", rec, 5.0)  # type: ignore[misc]
    assert kind == "book" and isinstance(book, Book) and book.valid() and abs(book.mid - 100.15) < 1e-9
    _, trades = parse_event("trades", [{"coin": PREFIX + "BTC", "side": "A", "px": "100.1", "sz": "0.5", "time": 1}], 5.0)  # type: ignore[misc]
    assert not trades[0].is_buy and trades[0].px == 100.1
    assert coin_of("data/rec_lighter/lighter_BTC-20261007.jsonl") == "lighter:BTC"
    assert coin_of("data/rec/BTC-20261006.jsonl") == "BTC" and coin_of("x/xyz_SP500-1.jsonl") == "xyz:SP500"


def test_simulator_delays_taker_orders_separately_from_resting_ones() -> None:
    m = asset_meta("BTC", BTC)
    v = PaperVenue(1000.0, m, 0.0, 0.0, latency_s=0.15, taker_latency_s=0.45)  # Lighter's zero-fee account: slow takes

    def bk(ts: float, bid: float = 100.0) -> Book:
        return Book(m.coin, ts, ts, np.array([[bid, 5.0]]), np.array([[bid + 0.1, 5.0]]))

    v.on_book(bk(10.0))
    v.submit(OrderIntent(m.coin, True, 0.2, 100.1, "Ioc", client_id="take"), 10.0)
    v.submit(OrderIntent(m.coin, True, 0.2, 100.0, "Alo", ttl_s=60, client_id="rest"), 10.0)
    v.on_book(bk(10.2))
    assert v.account(10.2).working == ("rest", "take") and v.pos == 0.0  # the quote is live; the take has not arrived yet
    assert v.account(10.2).open_orders == 1
    v.on_book(bk(10.3, bid=100.5))  # the price runs away during the speed bump
    v.on_book(bk(10.5, bid=100.5))
    assert v.pos == 0.0 and v.drain_fills() == []  # the take arrives too late and, never chasing, does not fill
    v2 = PaperVenue(1000.0, m, 0.0, 0.0, latency_s=0.15)
    assert v2.taker_latency_s == 0.15  # by default both are the same, as on Hyperliquid
