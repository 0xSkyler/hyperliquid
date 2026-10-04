"""Online learners. Everything here is predict-then-update, so every statistic the engine
trusts is out-of-sample by construction."""

from __future__ import annotations

import math

import numpy as np


class Standardizer:
    """Exponentially weighted z-scoring with clipping."""

    def __init__(self, n: int, alpha: float = 1 / 3000, clip: float = 4.0) -> None:
        self.mean = np.zeros(n)
        self.var = np.ones(n)
        self.alpha = alpha
        self.clip = clip
        self.count = 0

    def transform(self, x: np.ndarray) -> np.ndarray:
        self.count += 1
        a = max(self.alpha, 1.0 / self.count)  # behaves like a plain average early on
        d = x - self.mean
        self.mean += a * d
        self.var = (1 - a) * (self.var + a * d * d)
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-12), -self.clip, self.clip)


class OnlineRidge:
    """Recursive least squares with forgetting: forecast mean plus parameter uncertainty."""

    name = "ridge"

    def __init__(self, n: int, forgetting: float = 0.9999, prior_var: float = 0.05) -> None:
        self.w = np.zeros(n)
        self.P = np.eye(n) * prior_var
        self.lam = forgetting
        self.resid_var = 0.0
        self.n_obs = 0
        self._p_cap = prior_var * n * 10

    def predict(self, x: np.ndarray) -> tuple[float, float]:
        """Returns (mean, variance of the mean due to parameter uncertainty)."""
        return float(self.w @ x), float(x @ self.P @ x) * self.resid_var

    def update(self, x: np.ndarray, y: float) -> None:
        err = y - float(self.w @ x)
        self.n_obs += 1
        a = max(1 - self.lam, 1.0 / self.n_obs)
        if self.n_obs > 20:  # robustify against outliers once a scale estimate exists
            # Floor of 0.5 bps: a quiet spell (targets all exactly 0) must not clip every later error to 0.
            lim = 5.0 * math.sqrt(max(self.resid_var, 0.25))
            err = max(-lim, min(lim, err))
        self.resid_var += a * (err * err - self.resid_var)
        Px = self.P @ x
        k = Px / (self.lam + float(x @ Px))
        self.w += k * err
        self.P = (self.P - np.outer(k, Px)) / self.lam
        tr = float(np.trace(self.P))
        if tr > self._p_cap:  # covariance wind-up under forgetting
            self.P *= self._p_cap / tr


class EdgeCalibrator:
    """How much of the raw forecast has actually shown up out of sample, per regime.

    Tracks the regression slope of realized return on forecast and returns its *lower
    confidence bound*, clipped to [0, 1]. Until enough independent forecasts have resolved,
    or if the forecast has no demonstrated relationship with outcomes, the answer is 0 and
    the engine sees zero edge. This is the system's guard against imaginary edge.
    """

    def __init__(self, n_regimes: int, forgetting: float, z: float, min_indep: float, overlap: float) -> None:
        self.lam = forgetting
        self.z = z
        self.min_indep = min_indep
        self.overlap = max(overlap, 1.0)  # forecasts overlapping one horizon are not independent
        self.W = np.zeros(n_regimes)
        self.Spp = np.zeros(n_regimes)
        self.Spy = np.zeros(n_regimes)
        self.Syy = np.zeros(n_regimes)

    def update(self, pred: float, y: float, regime: np.ndarray) -> None:
        for acc, v in ((self.W, 1.0), (self.Spp, pred * pred), (self.Spy, pred * y), (self.Syy, y * y)):
            acc *= self.lam
            acc += regime * v

    def per_regime(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(slope, lower bound clipped to [0,1], independent sample count) per regime."""
        n = self.W / self.overlap
        ok = (self.Spp > 1e-12) & (n >= self.min_indep)
        spp = np.where(ok, self.Spp, 1.0)
        slope = np.where(ok, self.Spy / spp, 0.0)
        se = np.sqrt(np.maximum(self.Syy / spp - slope**2, 0.0) / np.maximum(n, 1.0))
        return slope, np.where(ok, np.clip(slope - self.z * se, 0.0, 1.0), 0.0), n

    def beta(self, regime: np.ndarray) -> float:
        return float(self.per_regime()[1] @ regime)

    def ic(self) -> float:
        spp, spy, syy = self.Spp.sum(), self.Spy.sum(), self.Syy.sum()
        return float(spy / math.sqrt(spp * syy)) if spp > 0 and syy > 0 else 0.0


class HalfLife:
    """Signal half-life from the lag-1 autocorrelation of the raw forecast."""

    def __init__(self, interval_s: float, alpha: float = 0.002) -> None:
        self.interval_s = interval_s
        self.alpha = alpha
        self.prev: float | None = None
        self.cov = 0.0
        self.var = 1e-12

    def update(self, m: float) -> float:
        if self.prev is not None:
            self.cov += self.alpha * (m * self.prev - self.cov)
            self.var += self.alpha * (m * m - self.var)
        self.prev = m
        rho = min(max(self.cov / self.var, 0.01), 0.999)
        return self.interval_s * math.log(0.5) / math.log(rho)
