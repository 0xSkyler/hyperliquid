"""Process entry point: `python -m app.main [--duration SECONDS]`. Mode comes from HL_MODE."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import os
import time
from pathlib import Path
from typing import Any

from app.brain.engine import Engine
from app.config.settings import Mode, Settings
from app.exchange.base import Venue
from app.exchange.hyperliquid import HyperliquidData, HyperliquidLive, parse_event
from app.exchange.paper import PaperVenue
from app.models.chart import TIMEFRAMES, ChartModel
from app.monitoring.dashboard import serve
from app.news.monitor import NewsMonitor, RssSource
from app.persistence.sink import JsonlSink, PostgresSink

log = logging.getLogger("app")


async def run(s: Settings, duration: float | None) -> Engine:
    if s.mode in (Mode.RESEARCH, Mode.BACKTEST):
        raise SystemExit("use `python -m backtest.run` for backtest/research")
    data = HyperliquidData(s.api_url)
    meta = await data.meta(s.coin)
    fees = await data.fees(s.account_address)
    if fees:
        s = dataclasses.replace(s, taker_fee=fees[0], maker_fee=fees[1])
    log.info("mode=%s coin=%s maxLeverage=%s taker=%.5f maker=%.5f", s.mode.value, s.coin,
             meta.max_leverage, s.taker_fee, s.maker_fee)  # fmt: skip

    venue: Venue
    live: HyperliquidLive | None = None
    if s.mode in (Mode.TESTNET, Mode.LIVE):
        live = HyperliquidLive(s.api_url, s.account_address, os.environ["HL_API_SECRET_KEY"], meta)
        await asyncio.to_thread(live.set_max_cross_leverage)
        await asyncio.to_thread(live.refresh)  # reconcile before the first decision
        venue = live
    else:
        venue = PaperVenue(s.paper_equity, meta, s.taker_fee, s.maker_fee, s.latency_ms / 1000)

    sink = PostgresSink(s.database_url) if s.database_url else JsonlSink(s.data_dir)
    engine = Engine(s, venue, meta, sink)
    state_file = Path(s.state_file)
    if state_file.is_file():
        why = engine.load_state(state_file.read_bytes())
        champ = engine.arena.champ
        log.info("learned state: %s", why or f"restored ({champ.model.n_obs} resolved forecasts, "
                                             f"champion {champ.model.name})")  # fmt: skip
    else:
        log.info("learned state: none found, starting from scratch")

    def save_state() -> None:
        blob = engine.dump_state()
        state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_file.with_suffix(".tmp")
        tmp.write_bytes(blob)
        tmp.replace(state_file)

    charts = {tf: m for tf in TIMEFRAMES if (m := ChartModel.load(s.chart_model_dir, tf)) is not None}
    log.info("chart models: %s", ", ".join(f"{tf} (to {m.meta['trained_to']})" for tf, m in charts.items()) or "none")
    analyst = None
    if s.llm_news:
        from app.news.llm import LlmAnalyst

        analyst = LlmAnalyst(s.llm_model)
        log.info("LLM news analysis enabled (model=%s, at most %d calls per poll)", s.llm_model, s.llm_max_per_poll)
    news = NewsMonitor([RssSource(u) for u in s.news_feeds], analyst=analyst, max_per_poll=s.llm_max_per_poll)

    def raw_stream_name(ts: float) -> str:
        return "raw-" + time.strftime("%Y%m%d", time.gmtime(ts))  # one file per UTC day
    loop_lag = {"max_ms": 0.0, "tick_ms": 0.0}

    def state() -> dict[str, Any]:
        b = engine.market.book
        age = time.time() - engine.market.feed_ts if b else None
        faults = engine.last.get("faults", ["starting"])
        chart_info = {"loaded": list(charts), "scores": engine.market.chart_scores}
        health = {"ok": not faults, "chart": chart_info, "faults": faults, "feed_age_s": age,
                  "ws_reconnects": data.reconnects,
                  "sink_dropped": sink.dropped, "loop_lag_ms": loop_lag}  # fmt: skip
        return {"engine": engine.snapshot(), "news": news.snapshot(), "health": health}

    async def stream() -> None:
        async for ts, ch, d in data.raw_stream(s.coin):
            if s.record_raw:
                sink.put(raw_stream_name(ts), {"t": ts, "ch": ch, "d": d})
            ev = parse_event(ch, d, ts)
            if ev is None:
                continue
            kind, payload = ev
            if kind == "book":
                engine.on_book(payload)
            elif kind == "bbo":
                engine.on_bbo(ts, payload)
            elif kind == "trades":
                engine.on_trades(payload)
            else:
                engine.on_ctx(payload)

    async def chart_loop() -> None:
        """Score each newly closed 5-minute bar. One REST call per bar, never on the decision path."""
        last_bar = dict.fromkeys(charts, 0.0)
        while charts:
            for tf, model in charts.items():
                step = TIMEFRAMES[tf]
                now = time.time()
                if now < last_bar[tf] + 2 * step + 3:
                    continue  # the bar after the one we scored has not closed yet
                try:
                    c = await data.candles(s.coin, tf, step)
                    if len(c) and c[-1, 0] != last_bar[tf]:
                        last_bar[tf] = c[-1, 0]
                        score = model.score(c) or 0.0
                        engine.on_chart((tf, score))
                        if s.record_raw:
                            sink.put(raw_stream_name(now), {"t": now, "ch": "chart", "d": {"tf": tf, "score": score}})
                    elif now - last_bar[tf] > 4 * step:
                        engine.on_chart((tf, 0.0))  # candles stopped updating: an old score is not context
                except Exception as e:  # noqa: BLE001 - context feature only; never stop trading for it
                    log.warning("chart update (%s) failed: %s", tf, e)
            await asyncio.sleep(20)

    async def ticker() -> None:
        nxt = time.time()
        n = 0
        while True:
            nxt += s.decision_interval_s
            await asyncio.sleep(max(0.0, nxt - time.time()))
            t0 = time.perf_counter()
            loop_lag["max_ms"] = max(loop_lag["max_ms"], (time.time() - nxt) * 1000)
            if live is not None and n % 2 == 0:
                await asyncio.to_thread(live.refresh)
            now = time.time()
            score = news.score(now) if analyst is not None else 0.0
            if abs(score - engine.market.news_score) > 1e-6:
                engine.on_news(score)
                if s.record_raw:
                    sink.put(raw_stream_name(now), {"t": now, "ch": "news", "d": {"score": score}})
            d = engine.on_tick(now)
            loop_lag["tick_ms"] = (time.perf_counter() - t0) * 1000
            if d is not None and d.order is not None:
                log.info("%s f %.2f -> %.2f | edge %.2f bps | %s | %s", d.action, d.f_current, d.f_target,
                         d.expected_edge_bps, d.exec_style, d.reason)  # fmt: skip
            n += 1
            if n % 300 == 0:
                save_state()
            if n % 60 == 0:
                j = engine.journal.summary()
                champ = engine.arena.champ
                log.info("state=%s equity=%.2f fills=%d champion=%s resolved=%d ic=%.4f beta=%s", engine.state,
                         j["equity"], j["fills"], champ.model.name, champ.model.n_obs, champ.cal.ic(),
                         engine.last.get("extra", {}).get("beta"))  # fmt: skip

    runner = await serve(state, s.dashboard_host, s.dashboard_port)
    log.info("dashboard: http://%s:%d", s.dashboard_host, s.dashboard_port)
    tasks = [asyncio.create_task(c) for c in (stream(), ticker(), news.run(), chart_loop())]
    try:
        done, _ = await asyncio.wait(tasks[:3], timeout=duration, return_when=asyncio.FIRST_EXCEPTION)
        for t in done:
            t.result()
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await runner.cleanup()
        save_state()
        sink.close()
    return engine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=None, help="exit after N seconds (default: run forever)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        import uvloop

        uvloop.install()
    except ImportError:
        pass
    asyncio.run(run(Settings.from_env(), args.duration))


if __name__ == "__main__":
    main()
