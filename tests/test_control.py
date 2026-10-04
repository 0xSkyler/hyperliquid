from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import stat
import sys
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer

from app.config.settings import LIVE_CONFIRM_PHRASE, Mode, Settings
from app.control import ControlError, ControlStore, load_startup
from app.exchange.base import AssetMeta
from app.monitoring.dashboard import make_app
from backtest.run import run_backtest, synthetic_events

ADDR = "0x" + "ab" * 20
KEY = "0x" + "12" * 32  # a made-up test value, not a real wallet
TOKEN = "t" * 24  # stand-in control token for the HTTP tests


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in [k for k in os.environ if k.startswith("HL_")]:
        monkeypatch.delenv(k)


# --- stored choices ---------------------------------------------------------------------
def test_defaults_to_paper_and_token_is_private_and_stable(tmp_path: Any) -> None:
    st = load_startup(str(tmp_path))
    assert st.settings.mode is Mode.PAPER and not st.paused and st.error == "" and st.secret_key == ""
    store = ControlStore(str(tmp_path))
    t = store.token()
    assert len(t) >= 30 and store.token() == t  # generated once, then reused
    if sys.platform != "win32":
        assert stat.S_IMODE(os.stat(store.token_path).st_mode) == 0o600


def test_credentials_are_validated_stored_privately_and_never_shown(tmp_path: Any) -> None:
    store = ControlStore(str(tmp_path))
    with pytest.raises(ControlError):
        store.set_credentials("not-an-address", KEY)
    with pytest.raises(ControlError):
        store.set_credentials(ADDR, "too-short")
    store.set_credentials(ADDR, KEY)
    pub = store.public()
    assert pub["has_key"] is True and pub["account_address"] == ADDR
    assert KEY not in json.dumps(pub) and "api_secret_key" not in pub
    if sys.platform != "win32":
        assert stat.S_IMODE(os.stat(store.path).st_mode) == 0o600
    store.clear_credentials()
    assert store.public()["has_key"] is False and "api_secret_key" not in store.load()


def test_live_needs_credentials_and_the_exact_phrase(tmp_path: Any) -> None:
    store = ControlStore(str(tmp_path))
    with pytest.raises(ControlError, match="credentials|address"):
        store.set_mode("testnet")
    with pytest.raises(ControlError):
        store.set_mode("backtest")  # not something the panel may select
    store.set_credentials(ADDR, KEY)
    with pytest.raises(ControlError, match="confirmation phrase"):
        store.set_mode("live", "yes please")
    assert load_startup(str(tmp_path)).settings.mode is Mode.PAPER
    store.set_mode("live", LIVE_CONFIRM_PHRASE)
    st = load_startup(str(tmp_path))
    assert st.settings.mode is Mode.LIVE and st.secret_key == KEY and st.settings.account_address == ADDR and not st.error
    store.set_mode("paper")
    assert "live_confirm" not in store.load()  # going back to live means confirming again
    store.set_mode("testnet")
    assert load_startup(str(tmp_path)).settings.mode is Mode.TESTNET
    store.clear_credentials()  # removing the key takes it out of any real-order mode
    assert load_startup(str(tmp_path)).settings.mode is Mode.PAPER


def test_unstartable_request_falls_back_to_paper_with_a_reason(tmp_path: Any) -> None:
    store = ControlStore(str(tmp_path))
    store.save({"mode": "live", "account_address": ADDR, "api_secret_key": KEY})  # e.g. file edited by hand
    st = load_startup(str(tmp_path))
    assert st.settings.mode is Mode.PAPER and "not been confirmed" in st.error
    store.save({"mode": "testnet"})
    st = load_startup(str(tmp_path))
    assert st.settings.mode is Mode.PAPER and "needs an account address" in st.error
    store.path.write_text("{ this is not json")
    assert load_startup(str(tmp_path)).settings.mode is Mode.PAPER  # a corrupt file is ignored, not fatal


def test_preferences_are_range_checked_and_applied(tmp_path: Any) -> None:
    store = ControlStore(str(tmp_path))
    with pytest.raises(ControlError):
        store.set_preferences(risk_aversion=0.2)
    with pytest.raises(ControlError):
        store.set_preferences(max_leverage="lots")
    store.set_preferences(risk_aversion="8", max_leverage=5, paper_equity="")
    store.set_paused(True)
    st = load_startup(str(tmp_path))
    assert st.settings.risk_aversion == 8.0 and st.settings.max_leverage_cap == 5.0 and st.paused
    assert st.settings.paper_equity == Settings().paper_equity  # blank field leaves the value alone


# --- engine switches ---------------------------------------------------------------------------
def test_paused_engine_keeps_learning_but_sends_nothing_and_flatten_closes() -> None:
    s = dataclasses.replace(Settings(), horizon_s=5.0, models=("ridge",))
    eng = run_backtest(synthetic_events(3000, signal=6.0, seed=4), s, AssetMeta("BTC", 5, 40.0), keep_engine=True)["engine"]
    assert eng.orders_sent > 0  # it trades when allowed to
    paused = run_backtest(synthetic_events(1500, signal=6.0, seed=5), s, AssetMeta("BTC", 5, 40.0), eng.dump_state(),
                          keep_engine=True)["engine"]  # fmt: skip
    assert paused.orders_sent > 0
    # Same again, but paused from the start: forecasts keep resolving, no orders leave.
    from app.brain.engine import Engine
    from app.exchange.paper import PaperVenue

    venue = PaperVenue(1000.0, AssetMeta("BTC", 5, 40.0), s.taker_fee, s.maker_fee, 0.0)
    e2 = Engine(dataclasses.replace(s, mode=Mode.BACKTEST), venue, AssetMeta("BTC", 5, 40.0))
    assert e2.load_state(eng.dump_state()) == ""
    e2.paused = True
    start = e2.arena.champ.model.n_obs
    t = 0.0
    for t, kind, payload in synthetic_events(1500, signal=6.0, seed=6):
        e2.on_tick(t) if kind == "book" else None
        (e2.on_book if kind == "book" else e2.on_trades)(payload)
    assert e2.orders_sent == 0 and venue.pos == 0.0 and e2.arena.champ.model.n_obs > start + 1000

    # Flatten: open a position by hand, then the emergency stop closes all of it and stays paused.
    from app.exchange.base import OrderIntent

    venue.submit(OrderIntent("BTC", True, 0.05, venue.book.best_ask * 1.01, "Ioc"), t)  # type: ignore[union-attr]
    venue.on_book(venue.book)  # type: ignore[arg-type]
    assert venue.pos == 0.05
    e2.paused = False
    msg = e2.flatten(t)
    venue.on_book(venue.book)  # type: ignore[arg-type]
    assert "closing 0.05 BTC" in msg and venue.pos == 0.0 and e2.paused
    assert "no position" in e2.flatten(t)


# --- HTTP control API ----------------------------------------------------------------------------
def _client_run(coro_fn: Any, control: dict[str, Any]) -> Any:
    async def go() -> Any:
        app = make_app(lambda: {"health": {"ok": True}}, token=TOKEN, control=control, port=8787)
        async with TestClient(TestServer(app)) as c:
            return await coro_fn(c)

    return asyncio.run(go())


def test_control_api_requires_token_local_host_and_same_origin() -> None:
    calls: list[dict[str, Any]] = []

    def pause(b: dict[str, Any]) -> dict[str, Any]:
        calls.append(b)
        return {"message": "ok"}

    def bad(_: dict[str, Any]) -> dict[str, Any]:
        raise ControlError("save your account address first")

    ok = {"X-Control-Token": TOKEN, "Host": "127.0.0.1:8787"}

    async def scenario(c: TestClient) -> None:  # type: ignore[type-arg]
        assert (await c.get("/api/state")).status == 200  # read-only status needs no token
        assert (await c.get("/")).status == 200
        r = await c.post("/api/control/pause", json={"paused": True}, headers={"Host": "127.0.0.1:8787"})
        assert r.status == 401
        r = await c.post("/api/control/pause", json={"paused": True}, headers=ok | {"X-Control-Token": "wrong"})
        assert r.status == 401
        r = await c.post("/api/control/pause", json={"paused": True}, headers=ok | {"Host": "evil.example:8787"})
        assert r.status == 403  # DNS rebinding
        r = await c.post("/api/control/pause", json={"paused": True}, headers=ok | {"Origin": "https://evil.example"})
        assert r.status == 403  # another website open in the same browser
        assert calls == []
        r = await c.post("/api/control/pause", json={"paused": True}, headers=ok | {"Origin": "http://127.0.0.1:8787"})
        assert r.status == 200 and calls == [{"paused": True}]
        r = await c.get("/api/control", headers=ok)
        assert r.status == 200 and (await r.json()) == {"mode": "paper"}
        r = await c.post("/api/control/mode", json={"mode": "testnet"}, headers=ok)
        assert r.status == 400 and "account address" in (await r.json())["error"]  # shown in the panel
        assert (await c.get("/api/control/pause", headers=ok)).status == 405  # actions are POST only

    _client_run(scenario, {"get": lambda _: {"mode": "paper"}, "pause": pause, "mode": bad})
