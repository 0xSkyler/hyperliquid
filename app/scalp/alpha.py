"""Fast alpha: where is the mid likely to be a few seconds from now?

A passive quote is filled exactly when someone wants to trade through it, which is usually
when the price is about to move against it. The defence is a very short-horizon forecast
from the top of the book: queue imbalance, microprice, and the last second of order flow.
On large-tick markets these predict the direction of the next move well enough to matter.

The forecast is learned online and, like every model here, is multiplied by how much of it
has actually shown up out of sample (lower confidence bound; zero until proven). The quoter
adds it to fair value, which moves the vulnerable quote away before it is hit and the other
one closer.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from app.models.online import EdgeCalibrator, OnlineRidge

FEATURES = ("imbalance_l1", "microprice_bps", "ofi_1s", "tfi_5s", "ret_1s_bps", "bias")
_ONE = np.array([1.0])


class FastAlpha:
    def __init__(self, horizon_s: float = 5.0, interval_s: float = 1.0, min_indep: float = 30.0) -> None:
        self.horizon_s = horizon_s
        self.model = OnlineRidge(len(FEATURES), forgetting=0.9995, prior_var=1.0)
        self.model.name = "fast_alpha"
        overlap = max(1.0, horizon_s / interval_s)
        self.cal = EdgeCalibrator(1, 0.9995, 2.0, min_indep, overlap)
        self._pending: deque[tuple[float, np.ndarray, float]] = deque()

    @staticmethod
    def vector(imbalance: float, micro_bps: float, ofi_1s: float, tfi_5s: float, ret_1s_bps: float) -> np.ndarray:
        c = lambda v, lim: max(-lim, min(lim, v))  # noqa: E731 - bound every input: one bad tick must not dominate
        return np.array([c(imbalance, 1.0), c(micro_bps, 5.0), c(ofi_1s, 3.0), c(tfi_5s, 1.0), c(ret_1s_bps, 10.0), 1.0])

    def on_tick(self, now: float, x: np.ndarray, mid: float) -> None:
        """Score and learn from forecasts whose horizon has passed, then remember this one."""
        while self._pending and self._pending[0][0] <= now:
            _, x0, mid0 = self._pending.popleft()
            y = math.log(mid / mid0) * 1e4
            self.cal.update(self.model.predict(x0)[0], y, _ONE)  # calibrate before learning: out of sample
            self.model.update(x0, y)
        self._pending.append((now + self.horizon_s, x, mid))

    def reset(self) -> None:
        self._pending.clear()  # after a data gap, old forecasts would be scored against an unrelated price

    @property
    def beta(self) -> float:
        return self.cal.beta(_ONE)

    def raw(self, x: np.ndarray) -> float:
        """The model's own forecast in bps, before any trust is applied. Used to file lessons by situation."""
        lim = 5.0 * math.sqrt(self.model.resid_var) if self.model.resid_var > 0 else 0.0
        return max(-lim, min(lim, self.model.predict(x)[0]))

    def predict(self, x: np.ndarray) -> float:
        """Calibrated forecast of the mid's move over the horizon, in bps (0 until the model has earned trust)."""
        b = self.beta
        if b <= 0:
            return 0.0
        lim = 5.0 * math.sqrt(self.model.resid_var)
        return b * max(-lim, min(lim, self.model.predict(x)[0]))

    def snapshot(self) -> dict[str, float]:
        return {"trusted_beta": self.beta, "oos_ic": self.cal.ic(), "scored": float(self.model.n_obs),
                "weights": dict(zip(FEATURES, (round(float(w), 4) for w in self.model.w), strict=True))}  # type: ignore[dict-item]  # fmt: skip
