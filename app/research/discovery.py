"""Automated feature discovery.

    python -m app.research.discovery "data/raw-*.jsonl" --out data/discovered_features.json

Generates candidate features (pairwise interactions, signed squares, smoothed and
de-trended versions of every base feature) and keeps only those that add information the
linear model does not already have, under four safeguards against fooling ourselves:

1. Walk-forward: a candidate is scored against the *residual* of a linear model fitted only
   on earlier data (with a one-horizon purge gap), never on data it could have seen.
2. Overlap-aware significance: t-statistics use independent sample counts (n / horizon).
3. Multiple testing: the threshold is Bonferroni-corrected for the number of candidates.
4. Stability: the sign of the relationship must be the same in every evaluation fold.

Survivors are written to a file. They are NOT deployed to the trading model: the engine
loads them into a separate challenger ("ridge_disc") that must then win promotion live.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from app.config.settings import Settings
from app.models.expand import FeatureExpander, Spec, spec_name
from app.research.dataset import BASE_NAMES, build_dataset, read_events


def candidate_specs(names: tuple[str, ...]) -> list[Spec]:
    specs: list[Spec] = [{"op": "prod", "a": a, "b": b} for a, b in itertools.combinations(names, 2)]
    specs += [{"op": "sq", "a": a} for a in names]
    specs += [{"op": op, "a": a, "n": n} for op in ("ema", "diff") for a in names for n in (5, 20)]
    return specs


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = a.std(), b.std()
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb)) if sa > 1e-12 and sb > 1e-12 else 0.0


def walk_forward_residual(Z: np.ndarray, y: np.ndarray, horizon: int, folds: int) -> tuple[np.ndarray, list[slice]]:
    """Residual of y after a linear model fitted on strictly earlier data; NaN in the first fold."""
    n = len(y)
    X = np.column_stack([Z, np.ones(n)])
    resid = np.full(n, np.nan)
    bounds = np.linspace(0, n, folds + 1).astype(int)
    evals = []
    for k in range(1, folds):
        tr = slice(0, max(bounds[k] - horizon, 1))
        te = slice(bounds[k], bounds[k + 1])
        A = X[tr].T @ X[tr] + 1e-3 * np.eye(X.shape[1])
        w = np.linalg.solve(A, X[tr].T @ y[tr])
        resid[te] = y[te] - X[te] @ w
        evals.append(te)
    return resid, evals


def evaluate(
    Z: np.ndarray, y: np.ndarray, horizon: int, names: tuple[str, ...] = BASE_NAMES, folds: int = 4,
    max_keep: int = 8, alpha: float = 0.05,
) -> dict[str, Any]:  # fmt: skip
    specs = candidate_specs(names)
    z_crit = NormalDist().inv_cdf(1 - alpha / (2 * len(specs)))
    report: dict[str, Any] = {"rows": len(y), "candidates": len(specs), "t_threshold": z_crit, "features": []}
    if len(y) < folds * max(10 * horizon, 200):
        report["note"] = "not enough data to evaluate anything; record more"
        return report
    resid, evals = walk_forward_residual(Z, y, horizon, folds)
    C = FeatureExpander(specs, names).matrix(Z)
    ok = ~np.isnan(resid)
    n_indep = ok.sum() / horizon
    scored = []
    for j, spec in enumerate(specs):
        ic = _corr(C[ok, j], resid[ok])
        t = ic * math.sqrt(n_indep)
        fold_ics = [_corr(C[te, j], resid[te]) for te in evals]
        if abs(t) > z_crit and all(f * ic > 0 for f in fold_ics):
            scored.append((abs(t), j, {"name": spec_name(spec), "ic": ic, "t": t, "fold_ics": fold_ics, **spec}))
    kept: list[int] = []
    for _, j, info in sorted(scored, key=lambda s: -s[0]):
        if len(kept) >= max_keep:
            break
        if all(abs(_corr(C[ok, j], C[ok, k])) < 0.9 for k in kept):  # drop near-duplicates
            kept.append(j)
            report["features"].append(info)
    report["passed_before_dedup"] = len(scored)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="recorded raw files (globs allowed)")
    ap.add_argument("--out", default=None, help="where to write survivors (default: HL_DISCOVERED_PATH)")
    args = ap.parse_args()
    s = Settings.from_env()
    ds = build_dataset(read_events(args.paths), s.decision_interval_s, s.horizon_ticks, s.coin)
    report = evaluate(ds.Z, ds.y, s.horizon_ticks)
    print(json.dumps(report, indent=1))
    if report["features"]:
        out = Path(args.out or s.discovered_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"wrote {len(report['features'])} features to {out}; restart the engine to add the ridge_disc challenger")
    else:
        print("nothing survived; no file written")


if __name__ == "__main__":
    main()
