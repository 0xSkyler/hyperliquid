"""Warm start: teach the models from recorded market data before they run live.

    python -m app.research.warmstart "data/raw-*.jsonl"

Replays the recordings through the same engine used live (learning as it goes, exactly as it
would have at the time) and writes the resulting learned state to the engine's state file.
The next `python -m app.main` picks it up and continues from there instead of from zero.

By default it continues from an existing state file; pass --fresh to start from nothing.
Do not replay a recording the state has already learned from: that counts the same evidence twice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config.settings import Settings
from app.exchange.base import AssetMeta
from app.research.dataset import read_events
from backtest.run import run_backtest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="recorded raw files (globs allowed), oldest first")
    ap.add_argument("--fresh", action="store_true", help="ignore any existing state file")
    ap.add_argument("--max-leverage", type=float, default=40.0)
    args = ap.parse_args()
    s = Settings.from_env()
    state_file = Path(s.state_file)
    state_in = state_file.read_bytes() if state_file.is_file() and not args.fresh else None
    r = run_backtest(read_events(args.paths), s, AssetMeta(s.coin, 5, args.max_leverage), state_in, keep_engine=True)
    engine = r.pop("engine")
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_bytes(engine.dump_state())
    print(json.dumps({k: r[k] for k in ("ticks", "champion", "promotions", "arena")}, indent=1, default=float))
    print(f"learned state written to {state_file}")


if __name__ == "__main__":
    main()
