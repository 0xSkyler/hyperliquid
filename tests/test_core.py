from __future__ import annotations

import dataclasses

import numpy as np

from app.brain.decision import Forecast, decide
from app.config.settings import Settings
from app.exchange.base import AccountState, AssetMeta, Book, OrderIntent, Trade, impact_frac
from app.exchange.hyperliquid import parse_event
from app.exchange.paper import PaperVenue
from app.indicators import library as ind
from app.market.state import MarketState
from app.models.online import EdgeCalibrator, OnlineRidge
from app.news.monitor import NewsItem, NewsMonitor, parse_feed, sanitize
from app.risk.kernel import SafetyKernel
from backtest.run import run_backtest, synthetic_events

META = AssetMeta("BTC", 5, 40.0)
S = Settings()


def book(mid: float = 85000.0, ts: float = 100.0, bq: float = 2.0, aq: float = 2.0) -> Book:
    lv = np.arange(20.0)
    bids = np.column_stack([mid - 0.5 - lv, np.full(20, 2.0)])
    asks = np.column_stack([mid + 0.5 + lv, np.full(20, 2.0)])
    bids[0, 1], asks[0, 1] = bq, aq
    return Book("BTC", ts, ts, bids, asks)


def fc(mu: float, beta: float = 1.0, sigma: float = 5.0, hl: float = 0.5) -> Forecast:
    return Forecast(mu * beta, mu, sigma, 0.5, beta, hl, 60.0, sell_rate=0.01, buy_rate=0.01)


def acct(pos: float = 0.0, equity: float = 1000.0) -> AccountState:
    return AccountState(100.0, True, equity, pos, 85000.0 if pos else 0.0)


# --- exchange parsing / rounding ------------------------------------------
def test_parse_hyperliquid_messages() -> None:
    raw = {"coin": "BTC", "time": 1700000000000,
           "levels": [[{"px": "85119.0", "sz": "7.9", "n": 3}], [{"px": "85120.0", "sz": "1.5", "n": 1}]]}  # fmt: skip
    kind, b = parse_event("l2Book", raw, 5.0)  # type: ignore[misc]
    assert kind == "book" and b.valid() and b.mid == 85119.5 and b.exch_ts == 1.7e9
    _, tr = parse_event("trades", [{"px": "85120", "sz": "0.1", "side": "B", "time": 1}], 5.0)  # type: ignore[misc]
    assert tr[0].is_buy and tr[0].sz == 0.1
    assert parse_event("subscriptionResponse", {}, 5.0) is None


def test_rounding_and_impact() -> None:
    assert META.round_px(85119.44) == 85119.0
    assert META.round_px(123456.7) == 123457.0
    assert META.round_sz(0.123456789) == 0.12345
    b = book()
    assert impact_frac(b.asks, b.mid, 0) == 0.5 / 85000
    assert impact_frac(b.asks, b.mid, 1e9) == 1.0  # beyond visible depth
    deep = impact_frac(b.asks, b.mid, 85000.5 * 2 + 85001.5 * 2)
    assert abs(deep - 1.0 / 85000) < 1e-9


# --- indicators ---------------------------------------------------------------
def test_indicators() -> None:
    x = np.arange(1.0, 51.0)
    assert ind.sma(x, 5)[-1] == 48.0
    assert ind.rsi(x, 14)[-1] == 100.0
    assert ind.rsi(x[::-1].copy(), 14)[-1] == 0.0
    assert abs(ind.ema(np.full(30, 7.0), 10)[-1] - 7.0) < 1e-12
    assert ind.efficiency_ratio(x) == 1.0
    lo, mid_, hi = ind.bollinger(x, 20)
    assert lo[-1] < mid_[-1] < hi[-1]
    rng = np.random.default_rng(0)
    c = 100 + np.cumsum(rng.standard_normal(200))
    h, lw = c + 1, c - 1
    assert 0 <= ind.adx(h, lw, c)[-1] <= 100
    assert np.all((ind.stochastic(h, lw, c)[20:] >= 0) & (ind.stochastic(h, lw, c)[20:] <= 100))
    assert ind.atr(h, lw, c)[-1] > 0


# --- market state ---------------------------------------------------------------
def test_order_flow_imbalance_sign() -> None:
    m = MarketState()
    m.on_book(book(bq=2.0, aq=2.0, ts=1))
    m.on_book(book(bq=5.0, aq=1.0, ts=2))  # bids added, asks pulled => positive OFI
    assert m._ofi[-1][1] == (5.0 - 2.0) + (2.0 - 1.0)


# --- learning ---------------------------------------------------------------------
def test_online_ridge_learns_linear_relation() -> None:
    rng = np.random.default_rng(1)
    m = OnlineRidge(3, forgetting=1.0, prior_var=10.0)
    for _ in range(2000):
        x = np.append(rng.standard_normal(2), 1.0)
        m.update(x, 2.0 * x[0] - 1.0 * x[1] + 0.5 + 0.1 * rng.standard_normal())
    assert np.allclose(m.w, [2.0, -1.0, 0.5], atol=0.05)
    assert m.predict(np.array([1.0, 0.0, 1.0]))[1] < 0.01


def test_calibrator_rejects_noise_and_accepts_signal() -> None:
    rng = np.random.default_rng(2)
    reg = np.array([1.0, 0.0, 0.0])
    noise = EdgeCalibrator(3, 0.9999, 2.0, 30, overlap=1)
    good = EdgeCalibrator(3, 0.9999, 2.0, 30, overlap=1)
    assert good.beta(reg) == 0.0  # no evidence yet => no trust
    for _ in range(3000):
        p = rng.standard_normal()
        noise.update(p, rng.standard_normal(), reg)
        good.update(p, 0.8 * p + rng.standard_normal(), reg)
    assert noise.beta(reg) < 0.03
    assert 0.6 < good.beta(reg) < 0.8  # lower bound sits just under the true 0.8
    assert good.beta(np.array([0.0, 1.0, 0.0])) == 0.0  # untested regime => still no trust


# --- decision engine ---------------------------------------------------------------
def test_no_edge_no_trade() -> None:
    d = decide(100.0, fc(0.0), acct(), book(), META, S)
    assert d.order is None and d.action == "HOLD"
    d = decide(100.0, fc(50.0, beta=0.0), acct(), book(), META, S)  # big raw forecast, zero credibility
    assert d.order is None and "no validated edge" in d.reason


def test_edge_smaller_than_round_trip_cost_is_not_traded() -> None:
    d = decide(100.0, fc(6.0), acct(), book(), META, S)  # 6 bps < 2 x (4.5 fee + spread)
    assert d.order is None


def test_strong_edge_trades_in_the_right_direction_with_derived_size() -> None:
    up = decide(100.0, fc(30.0, sigma=30.0), acct(), book(), META, S)
    dn = decide(100.0, fc(-30.0, sigma=30.0), acct(), book(), META, S)
    assert up.order and up.order.is_buy and up.f_target > 0 and up.exec_style == "taker"
    assert dn.order and not dn.order.is_buy and dn.f_target < 0
    weak = decide(100.0, fc(12.0, sigma=30.0), acct(), book(), META, S)
    assert weak.order and 0 < weak.f_target < up.f_target  # size scales with edge
    cautious = decide(100.0, fc(30.0, sigma=30.0), acct(), book(), META, dataclasses.replace(S, risk_aversion=20.0))
    assert cautious.order and cautious.f_target < up.f_target


def test_jump_risk_bounds_leverage_below_ruin() -> None:
    d = decide(100.0, fc(200.0, sigma=2.0), acct(), book(), META, S)  # absurdly good forecast
    assert d.order is not None
    # A 1.5% gap must not liquidate the account at the chosen exposure.
    assert 1 - d.f_target * S.jump_size > META.maintenance_margin * d.f_target


def test_position_is_closed_when_edge_disappears_and_reversed_when_it_flips() -> None:
    pos = 5 * 1000 / 85000  # 5x long
    d = decide(100.0, fc(0.0), acct(pos), book(), META, S)
    assert d.order and not d.order.is_buy and d.order.reduce_only and d.f_target == 0.0
    assert d.order.sz == pos
    d = decide(100.0, fc(-40.0), acct(pos), book(), META, S)
    assert d.order and d.f_target < 0 and not d.order.reduce_only and d.order.sz > pos


def test_long_half_life_and_active_flow_prefers_maker() -> None:
    f = fc(30.0, hl=30.0)
    f.sell_rate = 5.0  # plenty of aggressor selling into the bid
    d = decide(100.0, f, acct(), book(), META, S)
    assert d.order and d.exec_style == "maker" and d.order.tif == "Alo" and d.order.limit_px == 84999.5


# --- paper venue -----------------------------------------------------------------------
def venue(latency: float = 0.0) -> PaperVenue:
    v = PaperVenue(1000.0, META, 0.00045, 0.00015, latency)
    v.on_book(book(ts=100.0))
    return v


def test_paper_taker_fill_fees_and_pnl() -> None:
    v = venue()
    v.submit(OrderIntent("BTC", True, 0.01, 85010.0, "Ioc"), 100.0)
    v.on_book(book(ts=100.1))
    (f,) = v.drain_fills()
    assert f.px == 85000.5 and not f.maker and abs(f.fee - 0.01 * 85000.5 * 0.00045) < 1e-12
    v.on_book(book(mid=85100.0, ts=101.0))
    eq = v.account(101.0).equity
    assert abs(eq - (1000 - f.fee + 0.01 * (85100.0 - 85000.5))) < 1e-9
    v.submit(OrderIntent("BTC", False, 0.01, 85000.0, "Ioc", reduce_only=True), 101.0)
    v.on_book(book(mid=85100.0, ts=101.1))
    assert v.pos == 0.0 and abs(v.cash - (1000 + 0.01 * 99.0 - v.fees_paid)) < 1e-9


def test_paper_latency_and_ioc_limit() -> None:
    v = venue(latency=0.5)
    v.submit(OrderIntent("BTC", True, 0.01, 85000.5, "Ioc"), 100.0)
    v.on_book(book(ts=100.2))
    assert v.account(100.2).inflight == 1 and not v.drain_fills()
    v.on_book(book(mid=85050.0, ts=100.6))  # market moved away before the order arrived
    assert not v.drain_fills() and v.pos == 0.0


def test_paper_maker_needs_queue_to_clear() -> None:
    v = venue()
    v.submit(OrderIntent("BTC", True, 0.01, 84999.5, "Alo", ttl_s=60), 100.0)
    v.on_book(book(ts=100.1))
    assert v.account(100.1).open_orders == 1
    v.on_trades([Trade(100.2, 84999.5, 1.5, False)])  # 2.0 queued ahead of us
    assert not v.drain_fills()
    v.on_trades([Trade(100.3, 84999.5, 0.6, False)])  # queue exhausted, 0.1 reaches us
    (f,) = v.drain_fills()
    assert f.maker and f.sz == 0.01 and f.px == 84999.5
    v.submit(OrderIntent("BTC", True, 0.01, 85000.5, "Alo"), 101.0)  # would cross => rejected
    v.on_book(book(ts=101.1))
    assert v.rejects == 1


def test_paper_margin_reject_and_liquidation() -> None:
    v = venue()
    v.submit(OrderIntent("BTC", True, 1.0, 86000.0, "Ioc"), 100.0)  # $85k notional on $1k = 85x
    v.on_book(book(ts=100.1))
    assert v.rejects == 1 and v.pos == 0.0
    v.submit(OrderIntent("BTC", True, 0.4, 86000.0, "Ioc"), 100.2)  # 34x
    v.on_book(book(ts=100.3))
    assert v.pos == 0.4
    v.on_book(book(mid=83000.0, ts=101.0))  # -2.35% at 34x
    assert v.liquidations == 1 and v.pos == 0.0 and v.account(101.0).equity == 0.0
    assert v.drain_fills()[-1].liquidation


# --- safety kernel -----------------------------------------------------------------------
def test_kernel_blocks_on_broken_instrumentation() -> None:
    k = SafetyKernel(S)
    ok = AccountState(100.0, True, 1000.0, 0.0)
    assert k.check(100.0, book(ts=100.0), ok, 1e-5) == []
    assert k.check(100.0, None, ok, 1e-5) == ["no_market_data"]
    assert k.check(110.0, book(ts=100.0), AccountState(110.0, True, 1000.0), 1e-5) == ["stale_market_data"]
    crossed = book()
    crossed.bids[0, 0] = crossed.asks[0, 0] + 1
    assert k.check(100.0, crossed, ok, 1e-5) == ["crossed_or_empty_book"]
    assert "account_state_unknown" in k.check(100.0, book(), AccountState(100.0, False), 1e-5)
    stuck = AccountState(100.0, True, 1000.0, inflight=1, oldest_inflight_ts=90.0)
    assert "unacknowledged_order" in k.check(100.0, book(), stuck, 1e-5)


def test_kernel_adopts_startup_position_and_flags_persistent_mismatch() -> None:
    k = SafetyKernel(S)
    assert k.check(100.0, book(), AccountState(100.0, True, 1000.0, 0.5), 1e-5) == []
    assert k.expected_pos == 0.5  # restart != flat
    k.on_fill(0.1)
    assert k.check(101.0, book(ts=101), AccountState(101.0, True, 1000.0, 0.6), 1e-5) == []
    surprise = AccountState(102.0, True, 1000.0, 0.9)  # position we cannot explain
    assert k.check(102.0, book(ts=102), surprise, 1e-5) == []  # brief disagreement tolerated
    late = AccountState(130.0, True, 1000.0, 0.9)
    assert "position_mismatch" in k.check(130.0, book(ts=130), late, 1e-5)


# --- news -----------------------------------------------------------------------------------
def test_news_dedup_and_sanitising() -> None:
    xml = b"""<rss><channel>
      <item><title>Fed holds   rates steady &lt;script&gt;</title><link>https://a/1</link>
        <pubDate>Mon, 05 Oct 2026 12:00:00 GMT</pubDate></item></channel></rss>"""
    (it,) = parse_feed(xml, "a.com", 0.9, 1.0)
    assert it.title == "Fed holds rates steady <script>" and sanitize("a\x07b\n c") == "ab c"
    assert it.url == "https://a/1" and it.published_ts > 1.7e9
    n = NewsMonitor([])
    n.add(NewsItem("a.com", 0.9, 10.0, 11.0, "SEC approves spot Bitcoin ETF options", "u1"))
    n.add(NewsItem("b.com", 0.5, 9.0, 12.0, "SEC approves options on spot Bitcoin ETF", "u2"))
    n.add(NewsItem("c.com", 0.5, 12.0, 13.0, "Ethereum upgrade scheduled for November", "u3"))
    snap = n.snapshot()
    assert snap["clusters"] == 2
    top = next(c for c in snap["latest"] if c["confirmations"] == 2)
    assert top["source"] == "b.com"  # earliest report wins


# --- end to end ----------------------------------------------------------------------------
def test_backtest_does_not_trade_pure_noise() -> None:
    s = dataclasses.replace(S, horizon_s=5.0)
    r = run_backtest(synthetic_events(6000, signal=0.0, seed=3), s, META)
    assert r["resolved_forecasts"] > 5000
    assert r["fills"] == 0 and r["equity"] == 1000.0


def test_backtest_finds_and_exploits_a_planted_edge() -> None:
    s = dataclasses.replace(S, horizon_s=5.0)
    r = run_backtest(synthetic_events(8000, signal=6.0, seed=4), s, META)
    assert r["oos_ic"] > 0.3
    assert r["fills"] > 0 and r["liquidations"] == 0
    assert r["equity"] > 1000.0


def test_bbo_merge_refreshes_touch_and_keeps_consistent_depth() -> None:
    from app.exchange.base import merge_bbo

    b = merge_bbo(book(), "BTC", 101.0, (85001.5, 3.0, 85002.5, 1.0, 101.0))  # market moved up $2
    assert b.valid() and b.best_bid == 85001.5 and b.best_ask == 85002.5 and b.ts == 101.0
    assert np.all(np.diff(b.bids[:, 0]) < 0) and np.all(np.diff(b.asks[:, 0]) > 0)
    assert merge_bbo(None, "BTC", 1.0, (10.0, 1.0, 11.0, 1.0, 1.0)).valid()
    kind, payload = parse_event("bbo", {"coin": "BTC", "time": 1000, "bbo": [{"px": "10", "sz": "1", "n": 1},
                                {"px": "11", "sz": "2", "n": 1}]}, 5.0)  # type: ignore[misc]  # fmt: skip
    assert kind == "bbo" and payload == (10.0, 1.0, 11.0, 2.0, 1.0)


def test_kernel_uses_connection_liveness_not_touch_changes() -> None:
    k = SafetyKernel(S)
    ok = AccountState(110.0, True, 1000.0)
    assert k.check(110.0, book(ts=100.0), ok, 1e-5, feed_ts=109.5) == []  # quiet touch, live feed
    assert k.check(110.0, book(ts=100.0), ok, 1e-5, feed_ts=100.0) == ["stale_market_data"]
    late = AccountState(125.0, True, 1000.0)
    assert k.check(125.0, book(ts=100.0), late, 1e-5, feed_ts=124.9) == ["stale_order_book"]
