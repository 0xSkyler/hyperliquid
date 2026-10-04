from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np

from app.config.settings import Settings
from app.ensemble.arena import build_arena
from app.market.state import FEATURE_NAMES, REGIMES
from app.models.flow import PORTABLE, PORTABLE_IDX, PretrainedFlow
from app.research.dataset import build_dataset
from app.research.flow_train import fit, score
from app.research.swing_lab import simulate, targets
from app.research.ticks import second_events


# --- tick trades -> events ------------------------------------------------------------------
def test_second_events_aggregate_flow_and_never_show_the_future() -> None:
    ts = np.array([100.2, 100.7, 100.9, 102.1, 102.5])
    px = np.array([50.0, 50.1, 49.9, 50.2, 50.0])
    qty = np.array([1.0, 2.0, 0.5, 3.0, 1.0])
    buy = np.array([True, True, False, True, False])
    ev = list(second_events(ts, px, qty, buy))
    books = [(t, b) for t, kind, b in ev if kind == "book"]
    trades = {t: tr for t, kind, tr in ev if kind == "trades"}
    assert [t for t, _ in books] == [101.0, 102.0, 103.0]  # stamped at the END of each second
    b0 = books[0][1]
    assert b0.valid() and b0.best_ask == 50.1 and b0.best_bid == 49.9  # last buy / last sell in the second
    assert [(x.is_buy, x.sz) for x in trades[101.0]] == [(True, 3.0), (False, 0.5)]
    assert 102.0 not in trades and books[1][1].mid == b0.mid  # a silent second carries the book forward
    assert books[2][1].best_ask == 50.2 and books[2][1].best_bid == 50.0
    crossed = list(second_events(np.array([5.1, 5.2]), np.array([10.0, 10.3]), np.ones(2), np.array([True, False])))
    assert crossed[0][2].valid()  # bid above ask from trade prints is repaired, never emitted


def test_tick_replay_yields_portable_features_with_planted_signal() -> None:
    rng = np.random.default_rng(0)
    n = 4000 * 4
    ts = 1000.0 + np.sort(rng.uniform(0, 4000, n))
    buy = rng.random(n) < 0.5
    drift = np.cumsum((buy * 2 - 1) * 0.02 + rng.standard_normal(n) * 0.05)  # buying pressure moves price up
    ds = build_dataset(second_events(ts, 50000 + drift, rng.exponential(0.1, n), buy), 1.0, 5)
    X = ds.raw[:, PORTABLE_IDX]
    assert X.shape[1] == len(PORTABLE) == 7 and len(ds.y) > 3000
    assert set(PORTABLE) <= set(FEATURE_NAMES) and "imb_l1" not in PORTABLE and "ofi_5s" not in PORTABLE
    assert np.isfinite(X).all() and np.abs(X[:, :2]).max() <= 1.0  # trade-flow imbalance is bounded


# --- pre-trained flow model -----------------------------------------------------------------------
def _flow_data(seed: int, signal: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((40_000, len(PORTABLE))).astype(np.float32)
    return X, (signal * X[:, 0] * (X[:, 2] > 0) + rng.standard_normal(40_000)).astype(np.float32)


def test_flow_model_trains_scores_and_joins_the_arena_untrusted(tmp_path: Any) -> None:
    X, y = _flow_data(0, signal=0.5)
    booster = fit(X[:30_000], y[:30_000], purge=120)
    sc = score(booster.predict(X[30_000:]), y[30_000:], 60, taker_bps=9.0, maker_bps=3.0)
    assert sc["ic"] > 0.15 and sc["top_decile_gross_bps"] > 0.2
    assert sc["share_above_taker_cost"] == 0.0 and sc["realised_when_above_bps"] is None  # never predicts > costs
    noise_X, noise_y = _flow_data(1, signal=0.0)
    nb = fit(noise_X[:30_000], noise_y[:30_000], purge=120)
    assert abs(score(nb.predict(noise_X[30_000:]), noise_y[30_000:], 60, 9.0, 3.0)["ic"]) < 0.03

    booster.save_model(str(tmp_path / "flow_60s.txt"))
    assert PretrainedFlow.load(str(tmp_path), 60.0) is None  # no metadata
    (tmp_path / "flow_60s.json").write_text(json.dumps({"features": list(PORTABLE)}))
    assert PretrainedFlow.load(str(tmp_path), 5.0) is None  # trained for a different horizon
    s = dataclasses.replace(Settings(), chart_model_dir=str(tmp_path), discovered_path=str(tmp_path / "none.json"))
    arena = build_arena(s, FEATURE_NAMES[:-1], len(REGIMES), async_fit=False)
    assert [e.model.name for e in arena.entries] == ["ridge", "mlp", "tree", "flow_pretrained"]
    flow = arena.entries[3]
    raw = np.arange(len(FEATURE_NAMES) - 1, dtype=float)
    z = np.full(len(raw), 99.0)  # the standardised vector must be ignored by this model
    xs, mus, _, betas = arena.predict(z, np.array([1.0, 0.0, 0.0]), raw)
    assert xs[3].tolist() == [float(i) for i in PORTABLE_IDX] and betas[3] == 0.0  # zero trust to start
    assert mus[3] == float(booster.predict(xs[3][None, :])[0])
    flow.model.update(xs[3], 1.0)
    assert flow.model.n_obs == 1 and flow.model.predict(xs[3])[0] == mus[3]  # scores itself, never refits


# --- swing lane simulation ------------------------------------------------------------------------
def _hour(lo: float, hi: float, close: float = 100.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.full((1, 12), hi), np.full((1, 12), lo), np.full((1, 12), close)


def test_swing_targets_policies() -> None:
    p = np.array([0.1, 2.0, 0.2, -0.3, -2.5, 0.1])
    assert targets(p, 1.0, "flat_when_weak").tolist() == [0, 1, 0, 0, -1, 0]
    assert targets(p, 1.0, "hold_until_flip").tolist() == [0, 1, 1, 1, -1, -1]


def test_maker_order_fills_only_when_price_trades_through_it() -> None:
    close = np.array([100.0, 101.0])
    tgt = np.array([1.0, 1.0])
    hi, lo, cl = _hour(lo=99.995, hi=101.0)  # dips 0.5 bp: touches but does not trade through
    miss = simulate(tgt, close, hi, lo, cl, True, 0.00015, 0.00045)
    assert miss["fills"] == 0 and miss["ret"][0] == 0.0 and miss["pos"][0] == 0.0  # entry abandoned, no chase
    hi, lo, cl = _hour(lo=99.9, hi=101.0)  # trades 10 bps through
    hit = simulate(tgt, close, hi, lo, cl, True, 0.00015, 0.00045)
    assert hit["fills"] == 1 and abs(hit["ret"][0] - (0.01 - 0.00015)) < 1e-12
    taker = simulate(tgt, close, hi, lo, cl, False, 0.00015, 0.00045)
    assert abs(taker["ret"][0] - ((101 - 100.01) / 100 - 0.00045)) < 1e-12  # pays slippage and the taker fee


def test_unfilled_maker_exit_is_forced_out_at_market() -> None:
    close = np.array([100.0, 100.0, 98.0])
    tgt = np.array([1.0, 0.0, 0.0])  # enter long, then exit
    hi = np.vstack([np.full(12, 100.5), np.full(12, 100.0)])  # hour 2: price never trades above the ask
    lo = np.vstack([np.full(12, 99.5), np.full(12, 98.0)])
    cl = np.vstack([np.full(12, 100.0), np.full(12, 99.0)])
    r = simulate(tgt, close, hi, lo, cl, True, 0.00015, 0.00045)
    assert r["pos"].tolist() == [1.0, 0.0]  # out, not stuck in the position
    exit_px = 99.0 * (1 - 1e-4)
    assert abs(r["ret"][1] - ((exit_px - 100.0) / 100.0 - 0.00045)) < 1e-12  # sold at market after the wait
