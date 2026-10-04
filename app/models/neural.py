"""A small online neural network (one tanh hidden layer, Adam, Huber loss) in plain NumPy.

Deliberately tiny: with the amount of data a single market produces, a larger network would
memorise noise. It competes in the arena like any other model and earns trust only through
out-of-sample calibration.
"""

from __future__ import annotations

import math

import numpy as np


class OnlineMLP:
    name = "mlp"

    def __init__(self, n: int, hidden: int = 16, lr: float = 3e-3, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.p = [rng.normal(0, 1 / math.sqrt(n), (hidden, n)), np.zeros(hidden), np.zeros(hidden), np.zeros(1)]
        self.m = [np.zeros_like(a) for a in self.p]
        self.v = [np.zeros_like(a) for a in self.p]
        self.lr = lr
        self.t = 0
        self.y_var = 1.0
        self.resid_var = 0.0
        self.n_obs = 0

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        h = np.tanh(self.p[0] @ x + self.p[1])
        return h, float(self.p[2] @ h + self.p[3][0])

    def predict(self, x: np.ndarray) -> tuple[float, float]:
        return self._forward(x)[1] * self._scale(), 0.0

    def _scale(self) -> float:
        return math.sqrt(max(self.y_var, 1e-8))  # a run of unchanged prices must not zero the scale

    def update(self, x: np.ndarray, y: float) -> None:
        self.n_obs += 1
        a = max(1e-4, 1.0 / self.n_obs)
        h, out = self._forward(x)
        scale = self._scale()
        err = y - out * scale
        self.resid_var += a * (err * err - self.resid_var)
        self.y_var += max(1e-4, 1.0 / (self.n_obs + 20)) * (y * y - self.y_var)
        g = max(-1.0, min(1.0, out - max(-5.0, min(5.0, y / scale))))  # Huber gradient on scaled target
        dh = g * self.p[2] * (1 - h * h)
        grads = [np.outer(dh, x), dh, g * h, np.array([g])]
        self.t += 1
        for p, m, v, gr in zip(self.p, self.m, self.v, grads, strict=True):
            m += 0.1 * (gr - m)
            v += 0.001 * (gr * gr - v)
            p -= self.lr * (m / (1 - 0.9**self.t)) / (np.sqrt(v / (1 - 0.999**self.t)) + 1e-8)
