"""Download historical BTC/USD candles from Bitstamp's public API (no account needed).

    python -m app.research.history --years 10

Writes data/history/btcusd_300s.npz (columns: ts, open, high, low, close, volume). Resumable:
re-running continues from the last stored candle, so it also serves to top the file up.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import aiohttp
import numpy as np

URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
COLS = ("timestamp", "open", "high", "low", "close", "volume")


def history_path(data_dir: str, step: int) -> Path:
    return Path(data_dir) / "history" / f"btcusd_{step}s.npz"


def load(path: Path) -> np.ndarray:
    """(n, 6) float64 array sorted by time; empty if the file does not exist."""
    return np.load(path)["candles"] if path.is_file() else np.empty((0, 6))


async def download(path: Path, years: float, step: int, pause_s: float = 0.12) -> np.ndarray:
    have = load(path)
    end = int(time.time()) // step * step
    start = int(have[-1, 0]) + step if len(have) else end - int(years * 365.25 * 86400)
    chunks = [have]
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        while start < end:
            for attempt in range(6):
                try:
                    async with s.get(URL, params={"step": step, "limit": 1000, "start": start}) as r:
                        r.raise_for_status()
                        rows = (await r.json())["data"]["ohlc"]
                    break
                except (TimeoutError, aiohttp.ClientError, KeyError):
                    await asyncio.sleep(2.0 * (attempt + 1))
            else:
                raise RuntimeError(f"Bitstamp kept failing at start={start}")
            block = np.array([[float(x[c]) for c in COLS] for x in rows if int(x["timestamp"]) < end])
            if len(block) == 0:
                break
            chunks.append(block)
            start = int(block[-1, 0]) + step
            if len(chunks) % 100 == 0:
                print(f"  {time.strftime('%Y-%m-%d', time.gmtime(start))}  ({sum(len(c) for c in chunks):,} candles)")
                _save(path, chunks)
            await asyncio.sleep(pause_s)
    return _save(path, chunks)


def _save(path: Path, chunks: list[np.ndarray]) -> np.ndarray:
    allc = np.vstack([c for c in chunks if len(c)])
    _, idx = np.unique(allc[:, 0], return_index=True)
    allc = allc[idx]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, candles=allc)
    tmp.replace(path)
    return allc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=float, default=10.0)
    ap.add_argument("--step", type=int, default=300, help="candle size in seconds")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    path = history_path(args.data_dir, args.step)
    c = asyncio.run(download(path, args.years, args.step))
    first, last = (time.strftime("%Y-%m-%d", time.gmtime(c[i, 0])) for i in (0, -1))
    print(f"{len(c):,} candles from {first} to {last} in {path}")


if __name__ == "__main__":
    main()
