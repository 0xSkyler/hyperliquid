"""Gradient-boosted trees (LightGBM), refit periodically on a rolling buffer of resolved forecasts.

Every refit holds out the most recent 20% (with a purge gap of one horizon so overlapping
targets cannot leak) for early stopping. If the fitted model cannot beat "predict zero" on
that holdout it is discarded and the forecaster outputs zero until the next refit.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

PARAMS = {
    "objective": "huber", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "verbose": -1, "num_threads": 2, "seed": 0, "deterministic": True,
}  # fmt: skip


def available() -> bool:
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return False
    return True


class TreeForecaster:
    name = "tree"

    def __init__(
        self, n: int, min_train: int = 3000, refit_every: int = 1500, max_buffer: int = 100_000,
        purge: int = 60, async_fit: bool = True,
    ) -> None:  # fmt: skip
        self.min_train, self.refit_every, self.purge = min_train, refit_every, purge
        self._X = np.empty((max_buffer, n))
        self._y = np.empty(max_buffer)
        self._count = 0
        self._since_fit = 0
        self._booster: Any = None
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="tree-fit") if async_fit else None
        self._job: Future[None] | None = None
        self.resid_var = 0.0
        self.n_obs = 0
        self.fits = 0
        self.rejected_fits = 0

    def predict(self, x: np.ndarray) -> tuple[float, float]:
        b = self._booster
        return (float(b.predict(x[None, :])[0]), 0.0) if b is not None else (0.0, 0.0)

    def update(self, x: np.ndarray, y: float) -> None:
        self.n_obs += 1
        err = y - self.predict(x)[0]
        self.resid_var += max(1e-4, 1.0 / self.n_obs) * (err * err - self.resid_var)
        i = self._count % len(self._y)
        self._X[i], self._y[i] = x, y
        self._count += 1
        self._since_fit += 1
        if self._count >= self.min_train and self._since_fit >= self.refit_every:
            if self._job is not None and not self._job.done():
                return  # previous fit still running
            self._since_fit = 0
            X, yv = self._chronological()
            if self._pool is None:
                self._fit(X, yv)
            else:
                self._job = self._pool.submit(self._fit, X, yv)

    def _chronological(self) -> tuple[np.ndarray, np.ndarray]:
        n = len(self._y)
        if self._count <= n:
            return self._X[: self._count].copy(), self._y[: self._count].copy()
        i = self._count % n
        return np.vstack([self._X[i:], self._X[:i]]), np.concatenate([self._y[i:], self._y[:i]])

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            import lightgbm as lgb

            cut = int(len(y) * 0.8)
            tr = slice(0, max(cut - self.purge, 1))
            va = slice(cut, None)
            train = lgb.Dataset(X[tr], y[tr])
            valid = lgb.Dataset(X[va], y[va], reference=train)
            booster = lgb.train(
                PARAMS, train, num_boost_round=300, valid_sets=[valid],
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )  # fmt: skip
            mse = float(np.mean((y[va] - booster.predict(X[va])) ** 2))
            self.fits += 1
            if mse < float(np.mean(y[va] ** 2)):
                self._booster = booster
            else:
                self._booster = None
                self.rejected_fits += 1
        except Exception:  # noqa: BLE001 - a failed research fit must never take the engine down
            log.exception("tree fit failed")
