"""Champion / challenger arena.

Every model forecasts every tick and is scored on the same resolved outcomes. Only the
champion's forecast reaches the decision engine. A challenger replaces the champion when its
*calibrated* forecast (raw forecast x its own trusted beta) has significantly lower squared
error than the champion's calibrated forecast on the same out-of-sample observations
(a Diebold-Mariano style paired test, corrected for overlapping horizons).

Consequence: a model whose forecasts have earned no trust (beta = 0) is indistinguishable
from "predict zero" and can never be promoted on noise.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from app.config.settings import Settings
from app.models.expand import FeatureExpander, load_specs
from app.models.flow import PORTABLE, PORTABLE_IDX, PretrainedFlow
from app.models.neural import OnlineMLP
from app.models.online import EdgeCalibrator, OnlineRidge
from app.models.tree import TreeForecaster
from app.models.tree import available as tree_available

log = logging.getLogger(__name__)


class Forecaster(Protocol):
    name: str
    resid_var: float
    n_obs: int

    def predict(self, x: np.ndarray) -> tuple[float, float]: ...
    def update(self, x: np.ndarray, y: float) -> None: ...


@dataclass
class Entry:
    model: Forecaster
    cal: EdgeCalibrator
    expander: FeatureExpander | None = None
    feature_names: list[str] = field(default_factory=list)
    dm_mean: float = 0.0  # EW mean of (champion sq. error - this model's sq. error)
    dm_var: float = 0.0
    dm_n: float = 0.0
    raw_cols: tuple[int, ...] | None = None  # if set, the model reads these unstandardised features instead
    failed: str = ""  # set if the model raised; it then forecasts zero for the rest of the run

    def view(self, z: np.ndarray, raw: np.ndarray | None = None) -> np.ndarray:
        if self.raw_cols is not None:
            return raw[list(self.raw_cols)] if raw is not None else np.zeros(len(self.raw_cols))
        extra = self.expander(z) if self.expander is not None else ()
        return np.concatenate([z, extra, [1.0]])


class Arena:
    def __init__(self, entries: list[Entry], s: Settings) -> None:
        if not entries:
            raise ValueError("arena needs at least one model")
        self.entries = entries
        self.s = s
        self.champion = 0
        self.promotions: list[dict[str, Any]] = []
        self.on_promotion: Callable[[dict[str, Any]], None] | None = None

    @property
    def champ(self) -> Entry:
        return self.entries[self.champion]

    def predict(
        self, z: np.ndarray, regime: np.ndarray, raw: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], list[float], list[float], list[float]]:
        """Per model: its input vector, raw forecast, parameter variance, trusted beta.
        `z` is the standardised feature vector, `raw` the same features before standardising."""
        xs = [e.view(z, raw) for e in self.entries]
        preds = [self._guard(e, lambda e=e, x=x: e.model.predict(x)) or (0.0, 0.0)  # type: ignore[misc]
                 for e, x in zip(self.entries, xs, strict=True)]  # fmt: skip
        betas = [e.cal.beta(regime) for e in self.entries]
        return xs, [p[0] for p in preds], [p[1] for p in preds], betas

    def resolve(
        self, now: float, xs: list[np.ndarray], mus: list[float], betas: list[float], y: float, regime: np.ndarray
    ) -> None:
        lam = self.s.cal_forgetting
        e_champ = (y - betas[self.champion] * mus[self.champion]) ** 2
        for i, e in enumerate(self.entries):
            e.cal.update(mus[i], y, regime)  # calibrate first: the forecast was made before this outcome
            self._guard(e, lambda e=e, i=i: e.model.update(xs[i], y))  # type: ignore[misc]
            d = e_champ - (y - betas[i] * mus[i]) ** 2
            e.dm_n = e.dm_n * lam + 1.0
            a = 1.0 / e.dm_n
            dev = d - e.dm_mean
            e.dm_mean += a * dev
            e.dm_var += a * (dev * (d - e.dm_mean) - e.dm_var)
        self._maybe_promote(now)

    @staticmethod
    def _guard(e: Entry, fn: Callable[[], Any]) -> Any:
        """A model that raises is switched off (forecast 0 => never traded), not allowed to stop the engine."""
        if e.failed:
            return None
        try:
            return fn()
        except Exception as ex:  # noqa: BLE001
            e.failed = f"{type(ex).__name__}: {ex}"
            log.exception("model %s failed and has been disabled", e.model.name)
            return None

    def dm_t(self, i: int) -> float:
        e = self.entries[i]
        n = e.dm_n / self.s.horizon_ticks
        if i == self.champion or n < self.s.min_indep_samples or e.dm_var <= 0:
            return 0.0
        return e.dm_mean / math.sqrt(e.dm_var / n)

    def _maybe_promote(self, now: float) -> None:
        ts = [self.dm_t(i) for i in range(len(self.entries))]
        best = int(np.argmax(ts))
        if ts[best] <= self.s.promote_z:
            return
        event = {"ts": now, "from": self.champ.model.name, "to": self.entries[best].model.name, "t_stat": ts[best]}
        self.champion = best
        for e in self.entries:  # statistics were relative to the old champion
            e.dm_mean = e.dm_var = e.dm_n = 0.0
        self.promotions.append(event)
        if self.on_promotion:
            self.on_promotion(event)

    def snapshot(self) -> list[dict[str, Any]]:
        out = []
        for i, e in enumerate(self.entries):
            _, lcb, n = e.cal.per_regime()
            out.append({
                "name": e.model.name, "champion": i == self.champion, "resolved": e.model.n_obs,
                "oos_ic": e.cal.ic(), "max_trusted_beta": float(lcb.max()), "indep_samples": float(n.sum()),
                "vs_champion_t": self.dm_t(i), "resid_std_bps": math.sqrt(e.model.resid_var), "failed": e.failed,
            })  # fmt: skip
        return out


def build_arena(s: Settings, base_names: tuple[str, ...], n_regimes: int, async_fit: bool) -> Arena:
    """`base_names` excludes the bias term. The first available model in s.models starts as champion."""
    n = len(base_names) + 1

    def cal() -> EdgeCalibrator:
        return EdgeCalibrator(n_regimes, s.cal_forgetting, s.cal_z, s.min_indep_samples, s.horizon_ticks)

    entries: list[Entry] = []
    for name in s.models:
        if name == "ridge":
            entries.append(Entry(OnlineRidge(n, s.forgetting), cal(), None, list(base_names)))
        elif name == "mlp":
            entries.append(Entry(OnlineMLP(n), cal(), None, list(base_names)))
        elif name == "tree" and tree_available():
            tree = TreeForecaster(n, s.tree_min_train, s.tree_refit_every, purge=s.horizon_ticks, async_fit=async_fit)
            entries.append(Entry(tree, cal(), None, list(base_names)))
        elif name == "flow":
            flow = PretrainedFlow.load(s.chart_model_dir, s.horizon_s)
            if flow is not None:
                entries.append(Entry(flow, cal(), None, list(PORTABLE), raw_cols=PORTABLE_IDX))
        elif name == "ridge_disc":
            specs = load_specs(s.discovered_path)
            if specs:
                ex = FeatureExpander(specs, base_names)
                model = OnlineRidge(n + len(specs), s.forgetting)
                model.name = "ridge_disc"
                entries.append(Entry(model, cal(), ex, list(base_names) + ex.names))
    return Arena(entries, s)
