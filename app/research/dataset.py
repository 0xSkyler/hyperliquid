"""Turn recorded market events into a research matrix that mirrors what the live engine sees."""

from __future__ import annotations

import glob
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.exchange.base import merge_bbo
from app.exchange.hyperliquid import parse_event
from app.market.state import FEATURE_NAMES, MarketState
from app.models.online import Standardizer

Event = tuple[float, str, Any]
BASE_NAMES = FEATURE_NAMES[:-1]


@dataclass
class Dataset:
    Z: np.ndarray  # causally standardised base features, one row per tick
    y: np.ndarray  # forward return in bps from the next tick (first executable price) over the horizon
    mid: np.ndarray  # mid at each row's tick
    names: tuple[str, ...] = BASE_NAMES


def expand_paths(paths: list[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        out.extend(sorted(glob.glob(p)) or [p])
    return out


def read_events(paths: list[str]) -> Iterator[Event]:
    for path in expand_paths(paths):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                ev = parse_event(r["ch"], r["d"], r["t"])
                if ev:
                    yield r["t"], ev[0], ev[1]


def build_dataset(events: Iterable[Event], interval_s: float, horizon_ticks: int, coin: str = "BTC") -> Dataset:
    market = MarketState(interval_s)
    std = Standardizer(len(BASE_NAMES))
    rows: list[tuple[int, np.ndarray]] = []
    mids: list[float] = []
    next_tick: float | None = None

    def tick(now: float) -> None:
        market.sample(now)
        b = market.book
        fresh = b is not None and b.valid() and now - market.feed_ts <= 3.0
        mids.append(b.mid if fresh and b is not None else float("nan"))
        f = market.features(now) if fresh else None
        if f is not None:
            rows.append((len(mids) - 1, std.transform(f.values)))

    for ts, kind, payload in events:
        if next_tick is None:
            next_tick = ts + interval_s
        if ts - next_tick > 60:  # gap between recordings: start a fresh market state
            market = MarketState(interval_s)
            mids.extend([float("nan")] * (horizon_ticks + 2))
            next_tick = ts + interval_s
        while ts > next_tick:
            tick(next_tick)
            next_tick += interval_s
        if kind == "book":
            market.on_book(payload)
        elif kind == "bbo":
            market.on_book(merge_bbo(market.book, coin, ts, payload))
        elif kind == "trades":
            market.on_trades(payload)
        elif kind == "news":
            market.news_score = payload
        elif kind == "chart":
            market.chart_score = payload
        else:
            market.on_ctx(payload)

    m = np.array(mids)
    keep = [(i, z) for i, z in rows if i + 1 + horizon_ticks < len(m)]
    if not keep:
        return Dataset(np.empty((0, len(BASE_NAMES))), np.empty(0), np.empty(0))
    idx = np.array([i for i, _ in keep])
    with np.errstate(invalid="ignore"):
        y = np.log(m[idx + 1 + horizon_ticks] / m[idx + 1]) * 1e4
    ok = np.isfinite(y)
    return Dataset(np.vstack([z for _, z in keep])[ok], y[ok], m[idx][ok])
