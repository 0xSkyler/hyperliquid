from __future__ import annotations

import dataclasses
import pickle
from typing import Any

import numpy as np

from app.config.settings import Settings
from app.exchange.base import AssetMeta
from app.models.chart import FEATURES, MIN_BARS, ChartModel, chart_features
from app.models.tree import TreeForecaster
from app.research.chart_train import fit, make_xy
from backtest.run import run_backtest, synthetic_events

S = dataclasses.replace(Settings(), horizon_s=5.0, min_indep_samples=20.0, tree_min_train=1000, tree_refit_every=1000)
META = AssetMeta("BTC", 5, 40.0)


def candles(n: int, seed: int = 0, momentum: float = 0.0) -> np.ndarray:
    """Synthetic 5-minute candles; with momentum > 0 each bar's return leans on the previous one."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    eps = rng.standard_normal(n) * 0.001
    for i in range(1, n):
        r[i] = momentum * r[i - 1] + eps[i]
    c = 20000 * np.exp(np.cumsum(r))
    o = np.concatenate([[c[0]], c[:-1]])
    wick = np.abs(rng.standard_normal(n)) * 0.0003 * c
    ts = 1_600_000_200 + 300 * np.arange(n)
    return np.column_stack([ts, o, np.maximum(o, c) + wick, np.minimum(o, c) - wick, c, rng.exponential(5.0, n)])


# --- learned state survives restarts -----------------------------------------------------
def _run(seconds: int, seed: int, state: bytes | None = None) -> Any:
    return run_backtest(synthetic_events(seconds, signal=6.0, seed=seed), S, META, state, keep_engine=True)["engine"]


def test_state_round_trip_restores_everything_the_models_learned() -> None:
    first = _run(3000, seed=1)
    blob = first.dump_state()
    fresh = _run(400, seed=2)  # a new process that has seen almost nothing
    assert fresh.arena.champ.model.n_obs < 200
    assert fresh.load_state(blob) == ""
    for a, b in zip(first.arena.entries, fresh.arena.entries, strict=True):
        assert a.model.name == b.model.name and a.model.n_obs == b.model.n_obs
        assert np.allclose(a.cal.Spy, b.cal.Spy) and a.dm_n == b.dm_n
    ridge_a, ridge_b = first.arena.entries[0].model, fresh.arena.entries[0].model
    assert np.allclose(ridge_a.w, ridge_b.w) and np.allclose(ridge_a.P, ridge_b.P)
    tree_a, tree_b = first.arena.entries[2].model, fresh.arena.entries[2].model
    assert tree_a.fits >= 1 and tree_b._count == tree_a._count
    x = first.arena.entries[2].view(np.zeros(len(ridge_a.w) - 1))
    assert tree_a.predict(x) == tree_b.predict(x)
    assert np.allclose(first.std.mean, fresh.std.mean) and fresh.arena.champion == first.arena.champion


def test_restored_engine_carries_on_learning_and_trading() -> None:
    blob = _run(3000, seed=1).dump_state()
    resumed = _run(1500, seed=3, state=blob)
    cold = _run(1500, seed=3)
    assert resumed.arena.champ.model.n_obs > 3500 > cold.arena.champ.model.n_obs
    # The warm engine already trusts its forecasts, so it trades from the start; the cold one must wait.
    assert resumed.journal.n_fills > cold.journal.n_fills


def test_incompatible_or_corrupt_state_is_refused_not_half_loaded() -> None:
    eng = _run(400, seed=1)
    before = eng.arena.champ.model.n_obs
    other = run_backtest(synthetic_events(400, seed=1), dataclasses.replace(S, horizon_s=9.0), META, keep_engine=True)
    assert "different" in eng.load_state(other["engine"].dump_state())
    assert "unreadable" in eng.load_state(b"not a pickle")
    assert eng.arena.champ.model.n_obs == before


def test_full_tree_buffer_round_trips_in_time_order() -> None:
    t = TreeForecaster(2, min_train=10**9, max_buffer=50, async_fit=False)
    for i in range(70):  # wraps the ring buffer
        t.update(np.array([float(i), 0.0]), float(i))
    t2 = pickle.loads(pickle.dumps(t))  # noqa: S301
    X, y = t2._chronological()
    assert y.tolist() == [float(i) for i in range(20, 70)] and X[:, 0].tolist() == y.tolist()
    t2.update(np.array([70.0, 0.0]), 70.0)
    assert t2._chronological()[1].tolist() == [float(i) for i in range(21, 71)]


# --- chart model -----------------------------------------------------------------------------
def test_chart_features_never_look_ahead() -> None:
    c = candles(1500)
    full, _ = chart_features(c)
    part, _ = chart_features(c[:1200])
    assert full.shape == (1500, len(FEATURES))
    assert np.allclose(full[900:1200], part[900:1200], equal_nan=True)  # appending the future changes nothing
    assert np.isfinite(full[MIN_BARS:]).all()


def test_chart_features_do_not_depend_on_price_or_volume_scale() -> None:
    c = candles(1500)
    scaled = c.copy()
    scaled[:, 1:5] *= 37.0  # e.g. another venue, another era
    scaled[:, 5] *= 1000.0
    assert np.allclose(chart_features(c)[0][MIN_BARS:], chart_features(scaled)[0][MIN_BARS:], atol=1e-5)


def test_chart_model_learns_planted_momentum_and_round_trips_through_disk(tmp_path: Any) -> None:
    import json

    c = candles(30_000, seed=1, momentum=0.3)
    X, y, _, _ = make_xy(c, horizon=1)
    cut = 24_000
    booster, chosen = fit(X[:cut], y[:cut], purge=300)
    assert chosen["num_leaves"] in (7, 31) and chosen["rounds"] >= 10
    assert np.corrcoef(booster.predict(X[cut:]), y[cut:])[0, 1] > 0.2  # out of sample
    noise_X, noise_y, _, _ = make_xy(candles(30_000, seed=2), horizon=1)
    nb, _ = fit(noise_X[:cut], noise_y[:cut], purge=300)
    assert abs(np.corrcoef(nb.predict(noise_X[cut:]), noise_y[cut:])[0, 1]) < 0.05  # nothing to find in a random walk

    path = tmp_path / "chart_1h.txt"
    booster.save_model(str(path))
    assert ChartModel.load(str(tmp_path), "1h") is None  # no metadata => not trusted
    path.with_suffix(".json").write_text(json.dumps({"features": list(FEATURES), "trained_to": "x"}))
    m = ChartModel.load(str(tmp_path), "1h")
    assert m is not None and m.score(c[:100]) is None  # too little history for a meaningful score
    s = m.score(c[:5000])
    assert s is not None and abs(s - float(booster.predict(chart_features(c[:5000])[0][-1:])[0])) < 1e-6
    path.with_suffix(".json").write_text(json.dumps({"features": ["something_else"]}))
    assert ChartModel.load(str(tmp_path), "1h") is None  # trained on a different feature set
