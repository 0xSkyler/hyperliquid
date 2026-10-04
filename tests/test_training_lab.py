from __future__ import annotations

import dataclasses

import numpy as np

from app.brain.decision import Forecast, decide
from app.config.settings import Settings
from app.exchange.base import AccountState, AssetMeta
from app.market.state import FEATURE_NAMES
from app.research.regimes import REGIME_NAMES, bar_regimes, daily_regimes, resample
from app.research.strategy_lab import net_returns, study
from app.research.stress import ENVIRONMENTS, run_environment
from app.strategies.library import NAMES, VARIANTS, all_signals
from backtest.run import run_backtest, synthetic_events
from tests.test_core import book
from tests.test_state_and_chart import candles

META = AssetMeta("BTC", 5, 40.0)
S = dataclasses.replace(Settings(), horizon_s=5.0, models=("ridge",))


# --- strategy library ------------------------------------------------------------------
def test_every_strategy_is_bounded_and_never_looks_ahead() -> None:
    c = candles(2500, seed=3)
    full = all_signals(c)
    assert full.shape == (2500, len(VARIANTS)) and len(set(NAMES)) == len(NAMES)
    assert np.isfinite(full).all() and full.min() >= -1 and full.max() <= 1
    past = all_signals(c[:2000])
    late = np.abs(full[1000:2000] - past[1000:2000]).max(axis=0)
    assert late.max() < 1e-9, [n for n, d in zip(NAMES, late, strict=True) if d > 1e-9]
    families = {v.family for v in VARIANTS}
    assert {"ma_cross", "tsmom", "donchian_breakout", "bollinger_reversion", "rsi_reversion", "range_trade",
            "breakout_fade", "trend_pullback", "keltner_breakout", "vwap_reversion"} <= families  # fmt: skip


def test_trend_and_reversion_strategies_point_the_right_way() -> None:
    n = 600
    up = candles(n, seed=1)
    up[:, 1:5] *= np.exp(np.linspace(0, 1.0, n))[:, None]  # strong steady uptrend
    sig = dict(zip(NAMES, all_signals(up)[-1], strict=True))
    assert sig["ma_cross(20,100)"] == 1 and sig["tsmom(96)"] == 1 and sig["donchian_breakout(55)"] == 1
    assert sig["rsi_reversion(14,20)"] <= 0 and sig["bollinger_reversion(20,1.5)"] <= 0  # fading the move


# --- timeframes and regimes ---------------------------------------------------------------
def test_resample_aggregates_ohlcv_correctly() -> None:
    c = candles(288 * 3 + 7)  # three days of 5-minute bars plus a partial hour
    h = resample(c, 3600)
    first = c[c[:, 0] // 3600 == h[0, 0] // 3600]
    assert len(first) == 12 and h[0, 1] == first[0, 1] and h[0, 4] == first[-1, 4]
    assert h[0, 2] == first[:, 2].max() and h[0, 3] == first[:, 3].min() and abs(h[0, 5] - first[:, 5].sum()) < 1e-9
    assert np.all(np.diff(h[:, 0]) == 3600)  # partial bars at the edges are dropped


def test_regimes_are_labelled_from_closed_days_only() -> None:
    days = 500
    ts = 1_600_000_000 // 86400 * 86400 + 86400 * np.arange(days)
    close = np.concatenate([np.full(200, 100.0), 100 * 1.01 ** np.arange(150), np.full(150, 100 * 1.01**149)])
    daily = np.column_stack([ts, close, close, close, close, np.ones(days)])
    labels = daily_regimes(daily)
    assert REGIME_NAMES[labels[100]].startswith("range") and REGIME_NAMES[labels[300]].startswith("bull")
    assert np.array_equal(daily_regimes(daily[:320])[:320], labels[:320])  # the future does not change the past
    bars = np.column_stack([ts[300:302] + 3600, close[300:302], close[300:302], close[300:302], close[300:302], [1, 1]])
    assert bar_regimes(bars, daily)[0] == labels[299]  # an intraday bar sees yesterday's label, not today's


# --- strategy lab ------------------------------------------------------------------------------
def test_backtest_charges_cost_on_every_change_of_position() -> None:
    close = np.array([100.0, 101.0, 101.0, 99.0])
    pos = np.array([[1.0], [1.0], [-1.0], [0.0]])
    r = net_returns(pos, close, cost=0.001)[:, 0]
    assert abs(r[0] - (np.log(1.01) - 0.001)) < 1e-12  # open 1 unit
    assert r[1] == 0.0  # hold through a flat bar
    assert abs(r[2] - (-np.log(99 / 101) - 0.002)) < 1e-12  # flip = 2 units of turnover
    assert abs(r[3] - (-0.001)) < 1e-12  # close


def test_strategy_lab_finds_no_edge_in_a_random_walk() -> None:
    c = candles(60_000, seed=7)
    c[:, 0] = 1_420_070_400 + 3600 * np.arange(len(c))  # hourly bars spanning several years
    res = study(c, bar_regimes(c, resample(c, 86400)), 3600, cost=0.00045)
    assert res["variants_tested"] == len(VARIANTS) and len(res["walk_forward"]) >= 3
    assert res["walk_forward_net_sharpe"] < 0.5  # costs with no edge: nothing to harvest
    assert set(res["best_by_regime"]) == set(REGIME_NAMES)


# --- execution and stress -------------------------------------------------------------------------
def test_risk_reducing_orders_accept_slippage_but_opening_orders_do_not() -> None:
    def fc(mu: float) -> Forecast:
        return Forecast(mu, mu, 5.0, 0.5, 1.0, 0.5, 60.0)

    long_ = AccountState(100.0, True, 1000.0, 5 * 1000 / 85000, 85000.0)
    out = decide(100.0, fc(0.0), long_, book(), META, Settings())
    assert out.order and out.order.reduce_only and out.order.limit_px <= 84999.5 * 0.99 + 1  # 1% through the bid
    flat = AccountState(100.0, True, 1000.0)
    opening = decide(100.0, fc(30.0), flat, book(), META, Settings())
    assert opening.order and opening.order.limit_px <= 85000.5 * 1.0031  # never chases beyond the edge


def test_engine_survives_a_flash_crash_and_blind_spots_without_liquidation() -> None:
    crash = run_environment("flash_crash", S, META, seconds=4000)
    assert crash["fills"] > 20 and crash["max_exposure_x"] > 10  # it really was leveraged going in
    assert crash["liquidations"] == 0 and crash["final_equity"] > 0
    outage = run_environment("feed_outage", S, META, seconds=4000)
    assert outage["kernel_blocked_ticks"] >= 100 and outage["liquidations"] == 0
    bad = run_environment("crossed_book", S, META, seconds=4000)
    assert bad["kernel_blocked_ticks"] >= 30 and bad["liquidations"] == 0
    assert len(ENVIRONMENTS) == 11


def test_chart_scores_reach_the_models_as_features() -> None:
    assert FEATURE_NAMES[-5:] == ("chart_5m", "chart_1h", "chart_4h", "chart_1d", "bias")
    eng = run_backtest(synthetic_events(400, seed=1), S, META, keep_engine=True)["engine"]
    eng.on_chart(("1h", 0.7))
    eng.on_chart(("nonsense", 9.0))  # unknown timeframe is ignored
    f = eng.market.features(1_000_000.0 + 399)
    assert f is not None and f.values[FEATURE_NAMES.index("chart_1h")] == 0.7
    assert f.values[FEATURE_NAMES.index("chart_5m")] == 0.0
