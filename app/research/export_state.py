"""Export a market's learned state as a compact seed that ships with the code.

    python -m app.research.export_state <data_dir> <COIN> <out.pkl>

Reads <data_dir>/state/engine-<COIN>.pkl, drops the tree model's raw training rows, and writes
the rest (models, calibration, practice record, lessons). The engine loads a seed from
models/state/ at start-up when it is more experienced than the state it already has.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

from app.brain.engine import Engine
from app.config.settings import Mode, Settings
from app.exchange.base import AssetMeta
from app.exchange.paper import PaperVenue


def export(data_dir: str, coin: str, out: str) -> int:
    s = dataclasses.replace(Settings.from_env(), mode=Mode.BACKTEST, coin=coin, data_dir=data_dir)
    src = Path(data_dir) / "state" / f"engine-{coin}.pkl"
    if not src.is_file():
        print(f"no saved state at {src}")
        return 1
    meta = AssetMeta(coin, 5, 40.0)  # only the learned state is touched; market details do not matter here
    eng = Engine(s, PaperVenue(1000.0, meta, s.taker_fee, s.maker_fee), meta)
    why = eng.load_state(src.read_bytes())
    if why:
        print(f"cannot export {src}: {why}")
        return 1
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob = eng.dump_state(slim=True)
    dest.write_bytes(blob)
    print(f"{coin}: {eng.experience():,} seconds of experience -> {dest} ({len(blob) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    raise SystemExit(export(sys.argv[1], sys.argv[2], sys.argv[3]))
