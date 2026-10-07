"""Record raw market data for several coins at once, in the format the research tools replay.

    python scripts/record.py --coins BTC,ETH,ENA --minutes 60

Writes data/rec/<COIN>-<UTC date>.jsonl (appending). Each file replays through
`python -m app.research.scalp_lab` and `python -m backtest.run`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp

WS = "wss://api.hyperliquid.xyz/ws"
CHANNELS = ("l2Book", "bbo", "trades", "activeAssetCtx")


async def record(coins: list[str], minutes: float, out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    day = time.strftime("%Y%m%d", time.gmtime())
    # Builder-deployed markets are named "dex:COIN"; a colon is not allowed in Windows file names.
    files = {c: open(out_dir / f"{c.replace(':', '_')}-{day}.jsonl", "a", encoding="utf-8") for c in coins}  # noqa: SIM115
    counts = dict.fromkeys(coins, 0)
    end = time.time() + minutes * 60
    try:
        while time.time() < end:
            try:
                async with aiohttp.ClientSession() as s, s.ws_connect(WS, heartbeat=None) as ws:
                    for c in coins:
                        for ch in CHANNELS:
                            await ws.send_json({"method": "subscribe", "subscription": {"type": ch, "coin": c}})
                    last_ping = time.time()
                    while time.time() < end:
                        msg = await ws.receive(timeout=30)
                        now = time.time()
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            raise ConnectionError(str(msg.type))
                        m = msg.json()
                        ch, d = m.get("channel"), m.get("data")
                        if ch in CHANNELS and d:
                            coin = d[0]["coin"] if isinstance(d, list) else d["coin"]
                            if coin in files:
                                files[coin].write(json.dumps({"t": now, "ch": ch, "d": d}, separators=(",", ":")) + "\n")
                                counts[coin] += 1
                        if now - last_ping > 30:
                            await ws.send_json({"method": "ping"})
                            last_ping = now
            except (TimeoutError, aiohttp.ClientError, ConnectionError) as e:
                print(f"reconnecting after {type(e).__name__}", flush=True)
                await asyncio.sleep(1)
    finally:
        for f in files.values():
            f.close()
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coins", default="BTC,ETH,SOL")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--out", default="data/rec")
    args = ap.parse_args()
    counts = asyncio.run(record([c.strip() for c in args.coins.split(",") if c.strip()], args.minutes, Path(args.out)))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
