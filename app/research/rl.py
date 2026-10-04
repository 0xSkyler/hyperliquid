"""Reinforcement-learning research: tabular Q-learning for position-taking, offline only.

    python -m app.research.rl "data/raw-*.jsonl"

The agent sees a coarse state (three features bucketed low/neutral/high, plus its current
position), chooses short / flat / long, and is rewarded with the next step's return minus
trading cost. It trains on the first 70% of the recording and is evaluated greedily on the
last 30%, which it never saw.

This is a research tool. Nothing here is connected to live trading: the cost model is a flat
per-turn fee with no queueing, latency or impact, so a policy that looks good here has only
earned a closer look, not capital.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from app.config.settings import Settings
from app.research.dataset import BASE_NAMES, build_dataset, read_events

ACTIONS = np.array([-1.0, 0.0, 1.0])


def _states(Z: np.ndarray, cols: list[int]) -> np.ndarray:
    b = np.digitize(Z[:, cols], [-0.5, 0.5])  # 0, 1, 2 per feature
    return (b * 3 ** np.arange(len(cols))).sum(axis=1)


def _run(q: np.ndarray, st: np.ndarray, ret: np.ndarray, cost: float, rng: np.random.Generator | None,
         alpha: float, gamma: float, eps: float) -> tuple[float, int]:  # fmt: skip
    """One pass. With rng: epsilon-greedy and learning. Without: greedy evaluation only."""
    pos, pnl, trades = 1, 0.0, 0  # index into ACTIONS; start flat
    for t in range(len(st) - 1):
        s = st[t] * 3 + pos
        a = int(rng.integers(3)) if rng is not None and rng.random() < eps else int(np.argmax(q[s]))
        r = ACTIONS[a] * ret[t] - cost * abs(ACTIONS[a] - ACTIONS[pos])
        if rng is not None:
            s2 = st[t + 1] * 3 + a
            q[s, a] += alpha * (r + gamma * q[s2].max() - q[s, a])
        pnl += r
        trades += a != pos
        pos = a
    return pnl, trades


def train_q(
    Z: np.ndarray, mid: np.ndarray, step: int, cost_bps: float, names: tuple[str, ...] = BASE_NAMES,
    epochs: int = 30, alpha: float = 0.05, gamma: float = 0.9, eps: float = 0.1, seed: int = 0,
) -> dict[str, Any]:  # fmt: skip
    Z, mid = Z[::step], mid[::step]
    if len(mid) < 400:
        return {"note": "not enough data to train anything; record more", "steps": len(mid)}
    ret = np.append(np.diff(np.log(mid)) * 1e4, 0.0)  # return earned over the step after each decision
    cut = int(len(mid) * 0.7)
    # Choose the state features on the training part only.
    ic = [abs(np.corrcoef(Z[:cut, j], ret[:cut])[0, 1]) if Z[:cut, j].std() > 1e-12 else 0.0 for j in range(Z.shape[1])]
    cols = [int(j) for j in np.argsort(ic)[-3:]]
    st = _states(Z, cols)
    q = np.zeros((3 ** len(cols) * 3, 3))
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        _run(q, st[:cut], ret[:cut], cost_bps, rng, alpha, gamma, eps)
    train_pnl, _ = _run(q, st[:cut], ret[:cut], cost_bps, None, alpha, gamma, eps)
    test_pnl, test_trades = _run(q, st[cut:], ret[cut:], cost_bps, None, alpha, gamma, eps)
    return {
        "steps_train": cut, "steps_test": len(mid) - cut, "state_features": [names[j] for j in cols],
        "cost_bps_per_unit_turn": cost_bps,
        "train_pnl_bps": train_pnl, "test_pnl_bps": test_pnl, "test_trades": int(test_trades),
        "buy_and_hold_test_bps": float(ret[cut:].sum()),
        "verdict": "worth a closer look" if test_pnl > 0 and test_trades > 0 else "no usable policy found",
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--step", type=int, default=10, help="ticks between decisions")
    args = ap.parse_args()
    s = Settings.from_env()
    ds = build_dataset(read_events(args.paths), s.decision_interval_s, s.horizon_ticks, s.coin)
    print(json.dumps(train_q(ds.Z, ds.mid, args.step, s.taker_fee * 1e4), indent=1))


if __name__ == "__main__":
    main()
