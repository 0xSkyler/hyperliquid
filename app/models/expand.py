"""Derived features built from the standardised base feature vector.

Specs are plain JSON, produced by `app.research.discovery` and loaded by the engine:
    {"op": "prod", "a": "imb_l1", "b": "ret_5s"}   z_a * z_b
    {"op": "sq",   "a": "ofi_5s"}                  z_a * |z_a|
    {"op": "ema",  "a": "tfi_5s", "n": 20}         exponential mean of z_a
    {"op": "diff", "a": "ret_30s", "n": 5}         z_a minus its exponential mean
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

Spec = dict[str, Any]


def spec_name(s: Spec) -> str:
    args = [s["a"]] + ([s["b"]] if "b" in s else []) + ([str(s["n"])] if "n" in s else [])
    return f"{s['op']}({','.join(args)})"


def load_specs(path: str) -> list[Spec]:
    p = Path(path)
    if not p.is_file():
        return []
    return list(json.loads(p.read_text(encoding="utf-8")).get("features", []))


class FeatureExpander:
    def __init__(self, specs: list[Spec], base_names: tuple[str, ...]) -> None:
        self.specs = specs
        self.names = [spec_name(s) for s in specs]
        idx = {n: i for i, n in enumerate(base_names)}
        self._a = [idx[s["a"]] for s in specs]
        self._b = [idx[s["b"]] if "b" in s else -1 for s in specs]
        self._alpha = [2.0 / (s["n"] + 1.0) if "n" in s else 0.0 for s in specs]
        self._ema: list[float | None] = [None] * len(specs)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """Stateful: call exactly once per tick, in time order."""
        out = np.empty(len(self.specs))
        for k, s in enumerate(self.specs):
            v = float(z[self._a[k]])
            op = s["op"]
            if op == "prod":
                out[k] = v * float(z[self._b[k]])
            elif op == "sq":
                out[k] = v * abs(v)
            else:
                prev = self._ema[k]
                e = v if prev is None else prev + self._alpha[k] * (v - prev)
                self._ema[k] = e
                out[k] = e if op == "ema" else v - e
        return out

    def matrix(self, Z: np.ndarray) -> np.ndarray:
        """Same computation over a whole (time-ordered) matrix, for research."""
        self._ema = [None] * len(self.specs)
        out = np.vstack([self(z) for z in Z]) if len(Z) else np.empty((0, len(self.specs)))
        self._ema = [None] * len(self.specs)
        return out
