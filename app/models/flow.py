"""Trade-flow model pre-trained on historical tick trades (python -m app.research.flow_train).

A fixed LightGBM model over the venue-portable features (trade-flow imbalance, volatility-
normalised returns, RSI, z-score), read *unstandardised* so its inputs mean exactly what they
meant in training. It joins the arena as one more challenger: it does not learn live, and the
trust it earned on historical Binance data does not carry over. It starts with zero trust on
Hyperliquid and has to earn it, and win promotion, like every other model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.market.state import FEATURE_NAMES

# Features computed from trades and mid-price only, so they transfer between venues.
PORTABLE = ("tfi_5s", "tfi_30s", "ret_5s", "ret_30s", "ret_300s", "rsi_15s", "z_300s")
PORTABLE_IDX = tuple(FEATURE_NAMES.index(n) for n in PORTABLE)


class PretrainedFlow:
    name = "flow_pretrained"

    def __init__(self, booster: Any, meta: dict[str, Any]) -> None:
        self.booster = booster
        self.meta = meta
        self.resid_var = 0.0
        self.n_obs = 0

    @classmethod
    def load(cls, model_dir: str, horizon_s: float) -> PretrainedFlow | None:
        p = Path(model_dir) / f"flow_{round(horizon_s)}s.txt"
        meta_p = p.with_suffix(".json")
        if not p.is_file() or not meta_p.is_file():
            return None
        import lightgbm as lgb

        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if tuple(meta.get("features", ())) != PORTABLE:
            return None  # trained against a different feature set
        return cls(lgb.Booster(model_file=str(p)), meta)

    def predict(self, x: np.ndarray) -> tuple[float, float]:
        return float(self.booster.predict(x[None, :])[0]), 0.0

    def update(self, x: np.ndarray, y: float) -> None:
        """Scores itself; never refits."""
        self.n_obs += 1
        err = y - self.predict(x)[0]
        self.resid_var += max(1e-4, 1.0 / self.n_obs) * (err * err - self.resid_var)
