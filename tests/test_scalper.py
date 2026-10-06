from __future__ import annotations

import dataclasses

import numpy as np

from app.brain.engine import Engine
from app.config.settings import Mode, Settings
from app.exchange.base import AccountState, AssetMeta, Book, Fill, OrderIntent, Trade
from app.exchange.paper import PaperVenue
from app.learning.journal import Journal
from app.scalp.quoter import Desired, Quote, QuoteInputs, QuoteManager, desired_quotes
from backtest.run import synthetic_events

BTC = AssetMeta("BTC", 5, 40.0)
ALT = AssetMeta("ENA", 0, 10.0)
FREE = dataclasses.replace(Settings(), maker_fee=0.0, taker_fee=0.0, scalp_lessons=False)  # mechanics only
UNGATED = dataclasses.replace(FREE, scalp_gate_fills=0)  # real quoting without first proving it in practice


def book(bid: float, ask: float, bq: float = 2.0, aq: float = 2.0, ts: float = 100.0) -> Book:
    step = BTC.tick(bid) if bid > 1000 else ALT.tick(bid)
    lv = np.arange(5.0)
    return Book("X", ts, ts, np.column_stack([bid - lv * step, np.full(5, bq)]), np.column_stack([ask + lv * step, np.full(5, aq)]))


def inputs(flow: float = 0.0, alpha: float = 0.0, adverse: float | None = None, sigma: float = 0.6e-4) -> QuoteInputs:
    a = 0.0 if adverse is None else adverse
    return QuoteInputs(sigma, 1.0, flow, alpha, 60.0, a, a)


def acct(pos: float = 0.0, equity: float = 1000.0) -> AccountState:
    return AccountState(100.0, True, equity, pos)


# --- where to quote -------------------------------------------------------------------------
def test_with_no_costs_it_joins_the_touch_on_both_sides() -> None:
    b = book(84999.0, 85000.0)
    half = 0.5 / 84999.5 * 1e4
    d = desired_quotes(b, acct(), inputs(adverse=half), BTC, FREE)
    assert d.bid is not None and d.ask is not None
    assert d.bid.px == 84999.0 and d.ask.px == 85000.0  # tight spread: join, never cross
    assert d.bid.sz * d.bid.px >= BTC.min_notional and abs(d.bid.sz * d.bid.px - 500.0) < 1.0  # half the balance per quote


def test_fees_and_adverse_selection_push_quotes_behind_the_touch() -> None:
    b = book(84999.0, 85000.0)
    half = 0.5 / 84999.5 * 1e4
    free = desired_quotes(b, acct(), inputs(adverse=half), BTC, FREE)
    paid = desired_quotes(b, acct(), inputs(adverse=half), BTC, Settings())  # 1.5 bps maker fee
    assert free.bid and paid.bid and paid.ask and free.ask
    assert paid.bid.px < free.bid.px and paid.ask.px > free.ask.px
    assert (85000.0 - 0.5 - paid.bid.px) / 84999.5 * 1e4 >= 1.5  # at least the fee away from fair value
    picked_off = desired_quotes(b, acct(), inputs(adverse=4.0), BTC, FREE)  # our fills have been losing 4 bps
    assert picked_off.bid and picked_off.bid.px < free.bid.px - 25


def test_inventory_leans_quotes_and_stops_at_the_limit() -> None:
    b = book(84999.0, 85000.0)
    half = 0.5 / 84999.5 * 1e4
    flat = desired_quotes(b, acct(), inputs(adverse=half), BTC, FREE)
    long_ = desired_quotes(b, acct(pos=1000 / 85000), inputs(adverse=half), BTC, FREE)  # 1x long of a 2x limit
    assert flat.bid and long_.bid and long_.ask
    assert long_.bid.px < flat.bid.px  # less keen to buy more
    assert long_.info["skew_bps"] < 0
    full = desired_quotes(b, acct(pos=2000 / 85000), inputs(adverse=half), BTC, FREE)
    assert full.bid is None and full.ask is not None  # at the limit: only the side that reduces inventory
    short = desired_quotes(b, acct(pos=-2000 / 85000), inputs(adverse=half), BTC, FREE)
    assert short.ask is None and short.bid is not None


def test_one_sided_aggressive_flow_widens_only_the_exposed_side() -> None:
    b = book(84999.0, 85000.0)
    half = 0.5 / 84999.5 * 1e4
    calm = desired_quotes(b, acct(), inputs(adverse=half), BTC, FREE)
    dumped = desired_quotes(b, acct(), inputs(flow=-1.0, adverse=half), BTC, FREE)  # heavy aggressive selling
    assert calm.bid and calm.ask and dumped.bid and dumped.ask
    assert dumped.bid.px < calm.bid.px and dumped.ask.px == calm.ask.px


def test_wide_spread_steps_one_tick_inside_and_a_trusted_forecast_leans_fair_value() -> None:
    b = book(0.5000, 0.5010)  # 100 ticks wide (20 bps) on a small-cap
    d = desired_quotes(b, acct(), inputs(adverse=0.0), ALT, FREE)
    assert d.bid and d.ask
    assert abs(d.bid.px - 0.50001) < 1e-9 and abs(d.ask.px - 0.50099) < 1e-9  # one tick better than the touch, no more
    up = desired_quotes(b, acct(), inputs(alpha=40.0, adverse=8.0), ALT, FREE)
    flat = desired_quotes(b, acct(), inputs(alpha=0.0, adverse=8.0), ALT, FREE)
    assert up.ask and flat.ask and up.ask.px > flat.ask.px  # expecting a rise: ask further away


def test_tiny_balance_still_meets_the_exchange_minimum_or_does_not_quote() -> None:
    b = book(84999.0, 85000.0)
    d = desired_quotes(b, acct(equity=10.0), inputs(adverse=0.0), BTC, FREE)
    assert d.bid and d.bid.sz * d.bid.px >= 10.0  # $10 account: one minimum-size clip
    assert desired_quotes(b, acct(equity=3.0), inputs(adverse=0.0), BTC, FREE).bid is None  # cannot afford the minimum


# --- order management ---------------------------------------------------------------------------
def test_quote_manager_sends_only_necessary_actions() -> None:
    qm = QuoteManager("BTC", min_requote_s=1.0)
    want = Desired(Quote(84999.0, 0.005), Quote(85000.0, 0.005))
    first = qm.reconcile(want, acct(), 100.0, 1.0)
    assert [a[0] for a in first] == ["place", "place"]
    cids = {a[1].is_buy: a[1].client_id for a in first}
    assert all(c.startswith("0x") and len(c) == 34 for c in cids.values()) and cids[True] != cids[False]
    assert first[0][1].tif == "Alo"  # post-only: a scalper never pays the spread by accident
    assert qm.reconcile(want, acct(), 100.2, 1.0) == []  # nothing changed, nothing sent

    better = Desired(Quote(85000.0, 0.005), Quote(85000.0, 0.005))  # we could bid higher: only less competitive now
    assert qm.reconcile(better, acct(), 100.3, 1.0) == []  # throttled: requests are a budget
    assert [a[0] for a in qm.reconcile(better, acct(), 101.5, 1.0)] == ["cancel", "place"]

    safer = Desired(Quote(84990.0, 0.005), Quote(85000.0, 0.005))  # our bid is now too high: we are exposed
    urgent = qm.reconcile(safer, acct(), 101.6, 1.0)
    assert [a[0] for a in urgent] == ["cancel", "place"] and urgent[1][1].limit_px == 84990.0  # immediately, no throttle

    one_side = qm.reconcile(Desired(None, Quote(85000.0, 0.005)), acct(), 101.7, 1.0)
    assert [a[0] for a in one_side] == ["cancel"] and True not in qm.working


def test_quote_manager_tracks_fills_and_orders_that_vanished() -> None:
    qm = QuoteManager("BTC", min_requote_s=0.0)
    want = Desired(Quote(84999.0, 0.005), None)
    cid = qm.reconcile(want, acct(), 100.0, 1.0)[0][1].client_id
    qm.on_fill(Fill(100.5, "BTC", True, 84999.0, 0.002, 0.0, True, cid), 1e-5)
    assert qm.working[True].sz == 0.003  # partial fill: still working
    qm.on_fill(Fill(100.6, "BTC", True, 84999.0, 0.003, 0.0, True, cid), 1e-5)
    assert True not in qm.working and [a[0] for a in qm.reconcile(want, acct(), 100.7, 1.0)] == ["place"]
    # The venue's list no longer shows our order (expired / rejected) once it is recent enough to trust.
    gone = AccountState(110.0, True, 1000.0, working=(), working_ts=110.0)
    assert [a[0] for a in qm.reconcile(want, gone, 110.0, 1.0)] == ["place"]
    assert [a[0] for a in qm.pull_all()] == ["cancel"] and not qm.working


# --- simulator realism ------------------------------------------------------------------------------
def test_a_cancel_is_as_slow_as_an_order_so_a_stale_quote_can_still_be_hit() -> None:
    v = PaperVenue(1000.0, BTC, 0.0, 0.0, latency_s=0.5)
    v.on_book(book(84999.0, 85000.0, ts=100.0))
    v.submit(OrderIntent("BTC", True, 0.005, 84999.0, "Alo", ttl_s=60, client_id="q1"), 100.0)
    v.on_book(book(84999.0, 85000.0, ts=100.6))
    assert v.account(100.6).working == ("q1",)
    v.cancel("q1", 101.0)
    v.on_trades([Trade(101.2, 84990.0, 5.0, False)])  # the market drops through our bid before the cancel lands
    assert v.pos == 0.005 and v.drain_fills()[0].maker
    v.submit(OrderIntent("BTC", True, 0.005, 84980.0, "Alo", ttl_s=60, client_id="q2"), 102.0)
    v.on_book(book(84989.0, 84990.0, ts=102.6))
    v.cancel("q2", 102.7)
    v.on_book(book(84989.0, 84990.0, ts=103.3))  # this time the cancel arrives first
    v.on_trades([Trade(103.4, 84970.0, 5.0, False)])
    assert v.pos == 0.005 and v.account(103.4).working == ()


def test_journal_measures_capture_and_learns_adverse_selection_per_side() -> None:
    j = Journal()
    j.on_tick(100.0, 100.0, 1000.0)
    assert j.adverse_bps(True, prior_bps=0.5) == 0.5  # no fills yet: the prior
    for k in range(40):  # every bid we get filled on is followed by a 3 bp drop
        t = 100.0 + 10 * k
        j.on_fill(Fill(t, "X", True, 99.99, 1.0, 0.0, True), mid=100.0)
        j.on_tick(t + 6.0, 99.97, 1000.0)
    s = j.summary()
    assert abs(s["spread_capture_bps"] - 1.0) < 1e-6 and abs(s["spread_capture_usd"] - 0.4) < 1e-9
    assert abs(s["move_after_fill_bps"]["5s"] + 3.0) < 1e-6  # price went 3 bps against us
    assert 2.0 < j.adverse_bps(True, prior_bps=0.5) < 3.0  # learned, pulled toward what the fills show
    assert j.adverse_bps(False, prior_bps=0.5) == 0.5  # the other side has no evidence yet


# --- the engine as a scalper ------------------------------------------------------------------------------
def _scalp(seconds: int, s: Settings, seed: int = 1, paused: bool = False, equity: float = 1000.0) -> tuple[Engine, PaperVenue]:
    s = dataclasses.replace(s, mode=Mode.BACKTEST, strategy="maker", horizon_s=5.0, models=("ridge",))
    venue = PaperVenue(equity, BTC, s.taker_fee, s.maker_fee, 0.15)
    eng = Engine(s, venue, BTC)
    eng.paused = paused
    for t, kind, payload in synthetic_events(seconds, seed=seed, vol_bps=0.3):
        if kind == "book":
            eng.on_tick(t)
            eng.on_book(payload)
        else:
            eng.on_trades(payload)
    return eng, venue


def test_scalper_quotes_both_sides_fills_as_maker_and_respects_its_inventory_limit() -> None:
    eng, venue = _scalp(2500, UNGATED)
    j = eng.journal.summary()
    assert j["fills"] > 20 and j["maker_ratio"] > 0.9  # it trades by being hit, not by crossing
    assert j["spread_capture_bps"] > 0  # every maker fill is on the right side of the mid
    assert eng.max_abs_exposure <= 1.5 * FREE.scalp_inventory_x + 0.6 and venue.liquidations == 0
    snap = eng.snapshot()["scalper"]
    assert snap["enabled"] and snap["quote_uptime_pct"] > 50 and snap["orders_placed"] > j["fills"]
    assert eng.last["reason"].startswith("scalping")


def test_scalper_sends_nothing_when_stopped_and_pulls_quotes_when_data_goes_bad() -> None:
    eng, venue = _scalp(800, UNGATED, paused=True)
    assert eng.orders_sent == 0 and venue.pos == 0.0 and eng.arena.champ.model.n_obs > 300  # still learning

    eng, venue = _scalp(700, UNGATED)
    assert eng.quotes.working  # quoting
    t = 1_000_000.0 + 700
    eng.on_tick(t + 30)  # the feed has been silent for 30 s: instruments cannot be trusted
    assert "stale_market_data" in eng.last["faults"]
    assert venue.book is not None
    stale = dataclasses.replace(venue.book, ts=t + 31)
    eng.on_book(stale)
    venue.on_book(dataclasses.replace(stale, ts=t + 32))
    assert not eng.quotes.working and venue.account(t + 32).working == ()  # every quote cancelled


def test_with_real_fees_on_a_tight_market_the_scalper_refuses_to_quote_at_a_loss() -> None:
    eng, _ = _scalp(1500, dataclasses.replace(Settings(), scalp_gate_fills=0, scalp_lessons=False))  # BTC-like 0.12 bps spread against a 1.5 bps maker fee
    q = eng.last_quotes
    assert eng.journal.n_fills == 0 or eng.journal.summary()["spread_capture_bps"] > 1.0
    assert q.get("bid_behind_touch_bps") is None or q["bid_behind_touch_bps"] >= 1.4  # never at the touch for free


def test_action_budget_bounds_requoting_but_never_blocks_pulling_a_quote() -> None:
    qm = QuoteManager("BTC", min_requote_s=0.0, actions_per_min=6.0)  # the rate Hyperliquid always allows
    px, sent = 85000.0, 0
    for k in range(600):  # ten minutes of a market that drops a tick every second: our bid is "exposed" every time
        px -= 1.0
        sent += len(qm.reconcile(Desired(Quote(px, 0.005), None), acct(), 100.0 + k, 1.0))
    assert sent <= 6 * 10 + 14 and qm.skipped_for_budget > 100  # about six actions a minute, not two a second
    qm._tokens = 5.0
    qm.working.clear()
    assert [a[0] for a in qm.reconcile(Desired(Quote(px, 0.005), None), acct(), 800.0, 1.0)] == ["place"]
    qm._tokens = -50.0  # budget completely exhausted
    assert [a[0] for a in qm.reconcile(Desired(None, None), acct(), 801.0, 1.0)] == ["cancel"]  # safety first


def test_a_quote_resting_far_from_fair_is_not_requoted_for_small_drifts() -> None:
    qm = QuoteManager("BTC", min_requote_s=0.0)
    info = {"need_bid_bps": 4.0, "need_ask_bps": 4.0}  # resting 4 bps (~$34) away for fees and adverse selection
    assert len(qm.reconcile(Desired(Quote(84966.0, 0.005), None, info), acct(), 100.0, 1.0)) == 1
    for k, px in enumerate((84965.0, 84962.0, 84960.0)):  # fair value drifts a few dollars: not worth an action
        assert qm.reconcile(Desired(Quote(px, 0.005), None, info), acct(), 101.0 + k, 1.0) == []
    moved = qm.reconcile(Desired(Quote(84940.0, 0.005), None, info), acct(), 105.0, 1.0)  # half the margin is gone
    assert [a[0] for a in moved] == ["cancel", "place"]


# --- fast alpha -------------------------------------------------------------------------------------------
def test_fast_alpha_earns_trust_on_a_predictive_book_and_none_on_noise() -> None:
    from app.scalp.alpha import FastAlpha

    def run(signal: float, seed: int) -> FastAlpha:
        rng = np.random.default_rng(seed)
        fa, mid, imb = FastAlpha(horizon_s=5.0, interval_s=1.0), 100.0, 0.0
        for t in range(3000):
            x = FastAlpha.vector(float(np.tanh(imb)), 0.0, 0.0, 0.0, 0.0)
            if t == 0:
                assert fa.predict(x) == 0.0  # nothing proven yet: no opinion
            fa.on_tick(float(t), x, mid)
            mid *= float(np.exp((signal * np.tanh(imb) + 0.5 * rng.standard_normal()) * 1e-4))
            imb = 0.8 * imb + 0.6 * float(rng.standard_normal())
        return fa

    good, noise = run(0.5, 1), run(0.0, 2)
    assert good.beta > 0.3 and good.snapshot()["oos_ic"] > 0.2
    up = good.predict(FastAlpha.vector(0.9, 0.0, 0.0, 0.0, 0.0))
    assert up > 0.2 and good.predict(FastAlpha.vector(-0.9, 0.0, 0.0, 0.0, 0.0)) < -0.2  # heavy bid queue => up
    assert noise.beta == 0.0 and noise.predict(FastAlpha.vector(0.9, 0.0, 0.0, 0.0, 0.0)) == 0.0
    assert FastAlpha.vector(5.0, 99.0, -99.0, 3.0, 1e6).tolist() == [1.0, 5.0, -3.0, 1.0, 10.0, 1.0]  # inputs are bounded


def test_fast_forecast_pulls_the_vulnerable_quote_and_takes_only_when_it_pays() -> None:
    b = book(84999.0, 85000.0)
    calm = desired_quotes(b, acct(), inputs(adverse=0.06), BTC, FREE)
    falling = inputs(adverse=0.06)
    falling.fast_alpha_bps = -1.0  # the book says the price is about to drop 1 bp
    d = desired_quotes(b, acct(), falling, BTC, FREE)
    assert calm.bid and d.bid and d.ask
    assert d.bid.px <= calm.bid.px - 8 and d.ask.px == 85000.0  # bid steps out of the way; ask stays at the touch

    def takes(s: Settings) -> int:
        s = dataclasses.replace(s, mode=Mode.BACKTEST, strategy="maker", horizon_s=5.0, models=("ridge",))
        venue = PaperVenue(1000.0, BTC, s.taker_fee, s.maker_fee, 0.15)
        eng = Engine(s, venue, BTC)
        for t, kind, payload in synthetic_events(2500, signal=0.3, seed=3, vol_bps=0.3):  # imbalance predicts ~1 bp ahead
            if kind == "book":
                eng.on_tick(t)
                eng.on_book(payload)
            else:
                eng.on_trades(payload)
        assert eng.fast_alpha.beta > 0.2 and venue.liquidations == 0
        return eng.takes

    assert takes(FREE) > 10  # free to trade: it acts on the forecast
    assert takes(dataclasses.replace(Settings(), scalp_lessons=False)) == 0  # 4.5 bps taker fee: a ~1 bp forecast never pays for it, so it never takes


# --- practice before real money ---------------------------------------------------------------------------
def test_real_quotes_wait_until_practice_quotes_have_shown_a_profit() -> None:
    eng, venue = _scalp(2500, FREE)  # a random-walk market runs over resting quotes: practice loses
    p = eng.snapshot()["scalper"]["practice"]
    assert p["fills"] >= FREE.scalp_gate_fills and p["edge_bps_5s"] < 0
    assert not eng.making_allowed and eng.quotes.placed == 0 and venue.pos == 0.0  # so not one real quote was sent
    assert eng.last_quotes["making"].startswith("not yet")


def test_when_practice_is_convincingly_profitable_real_quoting_switches_on() -> None:
    eng, venue = _scalp(700, FREE)
    assert not eng.making_allowed and eng.quotes.placed == 0

    def practice_history(move_bps: float, n: int) -> Journal:
        j = Journal()
        j.on_tick(0.0, 100.0, 1000.0)
        for k in range(n):  # each practice fill is followed by the price moving `move_bps` in our favour (plus noise)
            j.on_fill(Fill(10.0 * k, "BTC", True, 100.0, 1.0, 0.0, True), mid=100.0)
            j.on_tick(10.0 * k + 6.0, 100.0 * (1 + (move_bps + 0.3 * (-1) ** k) * 1e-4), 1000.0)
        return j

    def continue_for(seconds: int, offset: float) -> None:
        for t, kind, payload in synthetic_events(seconds, seed=9, vol_bps=0.3):
            t += offset
            if kind == "book":
                eng.on_tick(t)
                eng.on_book(dataclasses.replace(payload, ts=t))
            else:
                eng.on_trades([dataclasses.replace(x, ts=t) for x in payload])

    eng.practice = practice_history(move_bps=0.2, n=5)  # promising, but five fills prove nothing
    continue_for(5, 700)
    assert not eng.making_allowed and eng.quotes.placed == 0
    eng.practice = practice_history(move_bps=2.0, n=60)  # sixty fills, each clearly worth more than the (zero) fee
    continue_for(5, 710)
    assert eng.making_allowed and eng.quotes.placed >= 2 and eng.quotes.working  # now it rests real quotes
    assert eng.snapshot()["scalper"]["practice"]["edge_lower_bound_bps"] > 0


# --- lessons: learning from every trade it could have made -------------------------------------------------
def test_quote_lessons_avoid_situations_that_lose_and_quote_those_that_pay() -> None:
    from app.scalp.lessons import QuoteLedger

    led = QuoteLedger(min_fills=30)
    assert not led.informed(True, 0.0) and led.best_distance(True, 0.0, 0.0) is None  # no record: no opinion yet
    mid = 100.0
    for t in range(4000):
        # Whenever the forecast is up (alpha > 0) sellers hit the bid and the price then RISES (good fill);
        # when it is down, sellers hit the bid and the price keeps FALLING (picked off).
        up = (t // 20) % 2 == 0
        alpha = 0.5 if up else -0.5
        mid += (0.004 if up else -0.004)
        bid, ask = mid - 0.005, mid + 0.005
        led.on_tick(float(t), bid, ask, 1.0, 1.0, 0.01, alpha, [(bid - 0.01, 5.0, False)])  # a sell sweeps through the bid
    assert led.informed(True, 0.5) and led.informed(True, -0.5)
    assert led.best_distance(True, 0.5, fee_bps=0.0) is not None  # forecast with us: resting a bid has paid
    assert led.best_distance(True, -0.5, fee_bps=0.0) != 0.0  # forecast against us: a touch bid has lost, so never again
    assert led.best_distance(True, 0.5, fee_bps=50.0) is None  # and nothing pays a 50 bps fee
    verdicts = {(r["side"], r["situation"], r["bps_behind_touch"]): r["verdict"] for r in led.table(0.0)}
    assert verdicts[("bid", "forecast against", 0.0)].startswith("avoid") and verdicts[("bid", "forecast with", 0.0)] == "quote here"


def test_quote_lessons_respect_the_queue_at_the_touch() -> None:
    from app.scalp.lessons import QuoteLedger

    led = QuoteLedger()
    led.on_tick(0.0, 99.99, 100.01, 5.0, 5.0, 0.01, 0.0, [])  # 5.0 already resting at the best bid ahead of us
    led.on_tick(1.0, 99.99, 100.01, 5.0, 5.0, 0.01, 0.0, [(99.99, 3.0, False)])  # 3 trades at our price: not our turn
    assert led.fills[0].sum() == 0
    led.on_tick(2.0, 99.99, 100.01, 5.0, 5.0, 0.01, 0.0, [(99.99, 3.0, False)])  # 6 in total: the queue ahead has cleared
    assert 0.99 < led.fills[0, 1, 0] <= 1  # the quote placed at t=0 at the touch is hit; later and deeper ones are not
    led.on_tick(3.0, 99.99, 100.01, 5.0, 5.0, 0.01, 0.0, [(99.99, 9.0, True)])  # a BUY at that price cannot hit a bid
    assert led.fills[0].sum() <= 1


def test_take_lessons_only_allow_what_has_beaten_the_fee() -> None:
    from app.scalp.lessons import TakeLedger

    led = TakeLedger(min_n=50)
    assert not led.allows(1.0, fee_bps=0.0, margin_bps=0.0)  # no record: do not take
    mid = 100.0
    for t in range(600):
        strong = t < 300  # first a spell where strong forecasts come true, then one where weak forecasts do nothing
        led.on_tick(float(t), mid - 0.0005, mid + 0.0005, 1.0 if strong else 0.15)
        mid *= 1 + (0.4e-4 if strong else 0.0)
    mean, lcb, n = led.worth(1.0)
    assert n >= 50 and lcb > 0.5
    assert led.allows(1.0, fee_bps=0.2, margin_bps=0.1) and not led.allows(1.0, fee_bps=4.5, margin_bps=0.1)
    assert not led.allows(0.15, fee_bps=0.2, margin_bps=0.1)  # weak forecasts have not paid: avoided
    assert {r["forecast_bps"]: r["verdict"] for r in led.table(4.5)}["0.8-1.6"] == "avoid: less than the fee"


def test_lessons_survive_a_restart_and_a_slim_export_keeps_what_was_learned() -> None:
    s = dataclasses.replace(Settings(), maker_fee=0.0, taker_fee=0.0, mode=Mode.BACKTEST, strategy="maker", horizon_s=5.0,
                            tree_min_train=300, tree_refit_every=300)  # fmt: skip
    eng = Engine(s, PaperVenue(1000.0, BTC, 0.0, 0.0, 0.15), BTC)
    for t, kind, payload in synthetic_events(900, signal=0.3, seed=3, vol_bps=0.3):
        if kind == "book":
            eng.on_tick(t)
            eng.on_book(payload)
        else:
            eng.on_trades(payload)
    assert eng.quote_lessons.quotes.sum() > 1000 and eng.take_lessons.n.sum() > 100 and eng.experience() > 1000
    full, slim = eng.dump_state(), eng.dump_state(slim=True)
    assert len(slim) < len(full)  # the tree model's raw training rows are left out of the shipped seed
    fresh = Engine(s, PaperVenue(1000.0, BTC, 0.0, 0.0, 0.15), BTC)
    assert fresh.peek_experience(slim) == eng.experience() and fresh.peek_experience(b"junk") == -1
    assert fresh.load_state(slim) == ""
    assert np.allclose(fresh.quote_lessons.fills, eng.quote_lessons.fills) and np.allclose(fresh.take_lessons.sum, eng.take_lessons.sum)
    assert fresh.practice.n_fills == eng.practice.n_fills and fresh.fast_alpha.beta == eng.fast_alpha.beta
    tree_a = next(e.model for e in eng.arena.entries if e.model.name == "tree")
    tree_b = next(e.model for e in fresh.arena.entries if e.model.name == "tree")
    x = np.zeros(len(eng.arena.entries[0].model.w))
    assert tree_a.fits >= 1 and tree_b.predict(x) == tree_a.predict(x) and tree_b._count == 0  # model kept, rows dropped
