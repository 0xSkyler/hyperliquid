"""Process entry point: `python -m app.main [--duration SECONDS]`. Mode comes from HL_MODE."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import signal
import time
from pathlib import Path
from typing import Any

from app.brain.engine import Engine
from app.config.settings import Mode
from app.control import ControlStore, Startup, load_startup
from app.exchange.base import Venue
from app.exchange.hyperliquid import HyperliquidData, HyperliquidLive, parse_event
from app.exchange.paper import PaperVenue
from app.models.chart import TIMEFRAMES, ChartModel
from app.monitoring.dashboard import serve
from app.news.monitor import NewsMonitor, RssSource
from app.persistence.sink import JsonlSink, PostgresSink

log = logging.getLogger("app")


async def run(st: Startup, duration: float | None) -> bool:
    """Run the engine until stopped. Returns True if the control panel asked for a restart."""
    s = st.settings
    if s.mode in (Mode.RESEARCH, Mode.BACKTEST):
        raise SystemExit("use `python -m backtest.run` for backtest/research")
    store = ControlStore(s.data_dir)
    startup_error = st.error
    data = HyperliquidData(s.api_url)
    meta = await data.meta(s.coin)
    if s.max_leverage_cap > 0:
        meta = dataclasses.replace(meta, max_leverage=min(meta.max_leverage, s.max_leverage_cap))
    fees = await data.fees(s.account_address)
    if fees:
        s = dataclasses.replace(s, taker_fee=fees[0], maker_fee=fees[1])
    log.info("mode=%s coin=%s maxLeverage=%s taker=%.5f maker=%.5f", s.mode.value, s.coin,
             meta.max_leverage, s.taker_fee, s.maker_fee)  # fmt: skip

    venue: Venue
    live: HyperliquidLive | None = None
    if s.mode in (Mode.TESTNET, Mode.LIVE):
        try:
            live = HyperliquidLive(s.api_url, s.account_address, st.secret_key, meta)
            await asyncio.to_thread(live.set_max_cross_leverage)
            await asyncio.to_thread(live.refresh)  # reconcile before the first decision
            if not live.account(time.time()).known:
                raise RuntimeError("the exchange did not return this account's state")
        except Exception as e:  # noqa: BLE001 - never crash-loop on a bad key or missing library
            log.error("could not start %s mode (%s: %s); falling back to paper", s.mode.value, type(e).__name__, e)
            startup_error = (f"Could not start {s.mode.value} mode: {type(e).__name__}: {e}. "
                             "Running in paper mode instead.")  # fmt: skip
            live = None
            if s.mode is Mode.TESTNET:  # paper mode should watch the real market, not testnet
                s = dataclasses.replace(s, mode=Mode.PAPER)
                data = HyperliquidData(s.api_url)
            s = dataclasses.replace(s, mode=Mode.PAPER)
    if live is not None:
        venue = live
    else:
        venue = PaperVenue(s.paper_equity, meta, s.taker_fee, s.maker_fee, s.latency_ms / 1000)

    sink = PostgresSink(s.database_url) if s.database_url else JsonlSink(s.data_dir)
    engine = Engine(s, venue, meta, sink)
    engine.paused = st.paused
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
        health = {"ok": not faults, "paused": engine.paused, "startup_error": startup_error, "chart": chart_info,
                  "faults": faults, "feed_age_s": age,
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

    # --- control panel actions ---------------------------------------------
    restart = asyncio.Event()

    def restart_soon(message: str) -> dict[str, Any]:
        asyncio.get_running_loop().call_later(0.5, restart.set)  # let the reply go out first
        return {"message": message}

    def c_get(_: dict[str, Any]) -> dict[str, Any]:
        eff = {"risk_aversion": s.risk_aversion, "max_leverage": meta.max_leverage, "paper_equity": s.paper_equity}
        return store.public() | {"mode": s.mode.value, "paused": engine.paused, "effective": eff,
                                 "startup_error": startup_error}  # fmt: skip

    def c_mode(b: dict[str, Any]) -> dict[str, Any]:
        store.set_mode(str(b.get("mode", "")), str(b.get("confirm", "")))
        log.warning("control panel: mode change to %s requested", b.get("mode"))
        return restart_soon(f"switching to {b.get('mode')}; the engine is restarting")

    def c_pause(b: dict[str, Any]) -> dict[str, Any]:
        engine.paused = bool(b.get("paused"))
        store.set_paused(engine.paused)
        log.warning("control panel: trading %s", "paused" if engine.paused else "resumed")
        return {"message": "trading paused: no orders will be sent" if engine.paused else "trading resumed"}

    def c_flatten(_: dict[str, Any]) -> dict[str, Any]:
        message = engine.flatten(time.time())
        store.set_paused(True)
        log.warning("control panel: flatten requested -> %s", message)
        return {"message": message}

    def c_credentials(b: dict[str, Any]) -> dict[str, Any]:
        store.set_credentials(str(b.get("account_address", "")), str(b.get("api_secret_key", "")))
        log.warning("control panel: credentials saved")
        if s.mode in (Mode.TESTNET, Mode.LIVE):
            return restart_soon("credentials saved; the engine is restarting to use them")
        return {"message": "credentials saved; you can now choose Testnet or Live"}

    def c_credentials_clear(_: dict[str, Any]) -> dict[str, Any]:
        was_trading = s.mode in (Mode.TESTNET, Mode.LIVE)
        store.clear_credentials()
        log.warning("control panel: credentials removed")
        if was_trading:
            return restart_soon("credentials removed; restarting in paper mode")
        return {"message": "credentials removed"}

    def c_preferences(b: dict[str, Any]) -> dict[str, Any]:
        store.set_preferences(b.get("risk_aversion"), b.get("max_leverage"), b.get("paper_equity"))
        return restart_soon("preferences saved; the engine is restarting")

    control = {"get": c_get, "mode": c_mode, "pause": c_pause, "flatten": c_flatten, "credentials": c_credentials,
               "credentials_clear": c_credentials_clear, "preferences": c_preferences}  # fmt: skip
    runner = await serve(state, s.dashboard_host, s.dashboard_port, store.token(), control)
    log.info("dashboard and control panel: http://%s:%d (token in %s)", s.dashboard_host, s.dashboard_port, store.token_path)
    # systemd stops the service with SIGTERM: shut down cleanly so the learned state is saved.
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows: Ctrl-C still raises KeyboardInterrupt
            pass
    stopper = asyncio.create_task(stop.wait())
    restarter = asyncio.create_task(restart.wait())
    tasks = [asyncio.create_task(c) for c in (stream(), ticker(), news.run(), chart_loop())]
    try:
        waiting = [*tasks[:3], stopper, restarter]
        done, _ = await asyncio.wait(waiting, timeout=duration, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            if t is not stopper and t is not restarter:
                t.result()  # the worker loops never return, so a finished one means it raised
        if stopper in done:
            log.info("stop requested: shutting down")
    finally:
        stopper.cancel()
        restarter.cancel()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await runner.cleanup()
        save_state()
        sink.close()
    return restart.is_set() and not stop.is_set()


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
    while asyncio.run(run(load_startup(), args.duration)):
        log.info("restarting with the control panel's new settings")


if __name__ == "__main__":
    main()
