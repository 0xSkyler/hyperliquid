"""Historical tick trades: download and turn into training rows for the trade-flow model.

    python -m app.research.ticks --days 90

Source: Binance USD-M futures BTCUSDT aggregated trades (public, no account). Each day is
downloaded once to data/ticks/, replayed second by second through the same MarketState the
live engine uses, and reduced to the features that mean the same thing on any venue (trade-
flow imbalance, volatility-normalised returns, RSI, z-score) plus the forward return that
followed. Results are cached per day in data/ticks/features-<date>.npz.

What this data cannot provide: order-book sizes. The book-imbalance and order-flow-imbalance
features stay untrained here and are learned live on Hyperliquid only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import socket
import time
import urllib.request
import zipfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from app.exchange.base import Book, Trade
from app.models.flow import PORTABLE, PORTABLE_IDX
from app.research.dataset import Event, build_dataset

URL = "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-{d}.zip"
HALF_TICK = 0.05


def load_day(zip_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(ts seconds, price, quantity, aggressor-is-buyer) for one day, time-ordered."""
    import polars as pl

    with zipfile.ZipFile(zip_path) as z, z.open(z.namelist()[0]) as f:
        head = f.readline()
        has_header = not head[:1].isdigit()
    with zipfile.ZipFile(zip_path) as z, z.open(z.namelist()[0]) as f:
        cols = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"]
        df = pl.read_csv(f.read(), has_header=has_header, new_columns=cols)
    ts = df["transact_time"].to_numpy() / 1000.0
    maker = df["is_buyer_maker"].cast(pl.Utf8).str.to_lowercase().to_numpy() == "true"
    return ts, df["price"].to_numpy().astype(float), df["quantity"].to_numpy().astype(float), ~maker


def second_events(ts: np.ndarray, px: np.ndarray, qty: np.ndarray, is_buy: np.ndarray) -> Iterator[Event]:
    """One synthetic top-of-book and the aggregated buy/sell flow per second, stamped at the END of
    the second so nothing a decision at time T sees happened after T."""
    sec = np.floor(ts).astype(np.int64)
    s0, s1 = int(sec[0]), int(sec[-1])
    n = s1 - s0 + 1
    k = sec - s0
    buy_vol = np.bincount(k, weights=qty * is_buy, minlength=n)
    sell_vol = np.bincount(k, weights=qty * ~is_buy, minlength=n)

    def last_price(mask: np.ndarray) -> np.ndarray:
        out = np.full(n, np.nan)
        out[k[mask]] = px[mask]  # later trades overwrite earlier ones within a second
        idx = np.where(np.isnan(out), 0, np.arange(n))
        np.maximum.accumulate(idx, out=idx)
        return out[idx]

    ask, bid, last = last_price(is_buy), last_price(~is_buy), last_price(np.ones(len(px), bool))
    bad = ~(ask > bid)  # includes NaN at the very start
    bid = np.where(bad, last - HALF_TICK, bid)
    ask = np.where(bad, last + HALF_TICK, ask)
    for i in range(n):
        t = float(s0 + i + 1)
        yield t, "book", Book("BTC", t, t, np.array([[bid[i], 1.0]]), np.array([[ask[i], 1.0]]))
        tr = []
        if buy_vol[i] > 0:
            tr.append(Trade(t, float(ask[i]), float(buy_vol[i]), True))
        if sell_vol[i] > 0:
            tr.append(Trade(t, float(bid[i]), float(sell_vol[i]), False))
        if tr:
            yield t, "trades", tr


def process_day(day: str, data_dir: str, horizon_ticks: int) -> tuple[str, int, str]:
    """Download (if needed) and featurise one day. Returns (day, rows, note)."""
    folder = Path(data_dir) / "ticks"
    out = folder / f"features-{day}-h{horizon_ticks}.npz"
    if out.is_file():
        return day, int(np.load(out)["y"].shape[0]), "cached"
    zp = folder / f"BTCUSDT-aggTrades-{day}.zip"
    try:
        if not zp.is_file():
            tmp = zp.with_suffix(".part")
            socket.setdefaulttimeout(60)  # a stalled connection must fail, not hang the whole run
            for attempt in range(4):
                try:
                    urllib.request.urlretrieve(URL.format(d=day), tmp)  # noqa: S310 - fixed https URL
                    break
                except OSError:
                    if attempt == 3:
                        raise
                    time.sleep(5 * (attempt + 1))
            tmp.replace(zp)
        ds = build_dataset(second_events(*load_day(zp)), 1.0, horizon_ticks)
    except Exception as e:  # noqa: BLE001 - one bad day must not stop the rest
        return day, 0, f"failed: {type(e).__name__}: {e}"
    np.savez_compressed(out, X=ds.raw[:, PORTABLE_IDX].astype(np.float32), y=ds.y.astype(np.float32),
                        mid=ds.mid.astype(np.float32))  # fmt: skip
    return day, len(ds.y), "ok"


def load_features(data_dir: str, horizon_ticks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """All cached days in date order: (X, y, day index per row, day names)."""
    files = sorted((Path(data_dir) / "ticks").glob(f"features-*-h{horizon_ticks}.npz"))
    Xs, ys, ds = [], [], []
    for i, f in enumerate(files):
        z = np.load(f)
        Xs.append(z["X"])
        ys.append(z["y"])
        ds.append(np.full(len(z["y"]), i))
    if not files:
        return np.empty((0, len(PORTABLE))), np.empty(0), np.empty(0, dtype=int), []
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ds), [f.name[9:19] for f in files]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--horizon", type=int, default=60, help="forward-return horizon in seconds")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    (Path(args.data_dir) / "ticks").mkdir(parents=True, exist_ok=True)
    last = dt.datetime.now(dt.UTC).date() - dt.timedelta(days=2)  # the archive lags by a day or so
    days = [(last - dt.timedelta(days=i)).isoformat() for i in range(args.days)][::-1]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(process_day, d, args.data_dir, args.horizon) for d in days]
        for done, f in enumerate(futs, 1):
            day, rows, note = f.result()
            print(f"[{done:3d}/{len(days)}] {day}  {rows:6d} rows  {note}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
