from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import secrets
import stat
import sys
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer

from app.config.settings import Mode, Settings
from app.control import ControlError, ControlStore, load_startup, verify_credentials
from app.exchange.base import AssetMeta, OrderIntent
from app.exchange.hyperliquid import HyperliquidLive, agent_address, parse_account
from app.monitoring.dashboard import make_app
from backtest.run import run_backtest, synthetic_events

OWNER = "0x" + "ab" * 20
KEY = "0x" + secrets.token_hex(32)  # a throwaway key generated for this test run, not a real wallet
SIGNER = agent_address(KEY)
TOKEN = "t" * 24  # stand-in control token for the HTTP tests
META = AssetMeta("BTC", 5, 40.0)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in [k for k in os.environ if k.startswith("HL_")]:
        monkeypatch.delenv(k)


# --- reading an account: the unified-account fix ---------------------------------------------
PERP_EMPTY: dict[str, Any] = {"marginSummary": {"accountValue": "0.0"}, "assetPositions": []}


def test_unified_account_balance_is_read_from_spot_usdc() -> None:
    spot = {"balances": [{"coin": "USDC", "total": "10.0", "hold": "0.0"}, {"coin": "HYPE", "total": "3.0", "hold": "0.0"}]}
    unified = parse_account("unifiedAccount", PERP_EMPTY, spot, "BTC")
    assert unified["equity"] == 10.0 and unified["unified"] and unified["perp_account_value"] == 0.0
    assert parse_account("portfolioMargin", PERP_EMPTY, spot, "BTC")["equity"] == 10.0
    # The same balances in the classic mode are NOT tradable for perps until transferred.
    classic = parse_account("default", PERP_EMPTY, spot, "BTC")
    assert classic["equity"] == 0.0 and classic["spot_usdc"] == 10.0 and not classic["unified"]


def test_classic_account_uses_perp_value_and_positions_are_parsed() -> None:
    perp = {"marginSummary": {"accountValue": "250.5"},
            "assetPositions": [{"position": {"coin": "ETH", "szi": "1.0", "entryPx": "3000", "unrealizedPnl": "5.0"}},
                               {"position": {"coin": "BTC", "szi": "-0.002", "entryPx": "85000.0", "unrealizedPnl": "-1.5"}}]}  # fmt: skip
    a = parse_account("default", perp, {"balances": []}, "BTC")
    assert a["equity"] == 250.5 and a["position"] == -0.002 and a["entry_px"] == 85000.0
    u = parse_account("unifiedAccount", perp, {"balances": [{"coin": "USDC", "total": "100", "hold": "20"}]}, "BTC")
    assert u["equity"] == 100 + 5.0 - 1.5 and u["spot_usdc_hold"] == 20.0  # cash plus unrealised PnL
    assert parse_account("default", {}, {}, "BTC")["equity"] == 0.0  # empty / unknown account does not raise


# --- connecting -----------------------------------------------------------------------------------
class FakeLookup:
    def __init__(self, role: dict[str, Any], equity: float = 10.0) -> None:
        self._role, self._equity, self.asked = role, equity, []  # type: ignore[var-annotated]

    async def role(self, address: str) -> dict[str, Any]:
        self.asked.append(address)
        return self._role

    async def account(self, address: str, coin: str) -> dict[str, Any]:
        return {"equity": self._equity, "abstraction": "unifiedAccount", "unified": True, "address": address}


def _verify(lookup: FakeLookup, key: str = KEY, address: str = "") -> dict[str, Any]:
    return asyncio.run(verify_credentials(lookup, key, address, "BTC"))


def test_connect_finds_the_account_from_the_key_alone_and_returns_its_balance() -> None:
    lookup = FakeLookup({"role": "agent", "data": {"user": OWNER}})
    acc = _verify(lookup)
    assert lookup.asked == [SIGNER]  # asked Hyperliquid who this API wallet trades for
    assert acc["address"] == OWNER and acc["equity"] == 10.0 and acc["key"] == KEY
    assert _verify(lookup, KEY[2:])["key"] == KEY  # a key pasted without 0x is accepted
    assert _verify(lookup, address=OWNER.upper().replace("0X", "0x"))["address"] == OWNER  # matching address, any case


def test_connect_refuses_wrong_or_dangerous_keys_with_a_useful_message() -> None:
    agent = FakeLookup({"role": "agent", "data": {"user": OWNER}})
    with pytest.raises(ControlError, match="not a private key"):
        _verify(agent, "hello")
    with pytest.raises(ControlError, match="wallet address should be"):
        _verify(agent, address="0x123")
    with pytest.raises(ControlError, match="authorised for account"):
        _verify(agent, address="0x" + "cd" * 20)
    with pytest.raises(ControlError, match="main wallet"):  # a key that could withdraw is never stored
        _verify(FakeLookup({"role": "user"}))
    with pytest.raises(ControlError, match="does not recognise this API wallet"):
        _verify(FakeLookup({"role": "missing"}))
    with pytest.raises(ControlError, match="not supported"):
        _verify(FakeLookup({"role": "vault"}))


# --- stored choices ---------------------------------------------------------------------------------
def test_without_an_account_nothing_can_trade(tmp_path: Any) -> None:
    st = load_startup(str(tmp_path))
    assert not st.connected and not st.running and st.secret_key == ""
    assert st.settings.mode is Mode.PAPER  # internal simulator only, held stopped by the runner
    ControlStore(str(tmp_path)).set_running(True)  # even a stray "running" flag cannot start it
    assert not load_startup(str(tmp_path)).running


def test_connecting_goes_live_but_trading_waits_for_start(tmp_path: Any) -> None:
    store = ControlStore(str(tmp_path))
    store.set_running(True)
    store.save_credentials(OWNER, KEY)
    st = load_startup(str(tmp_path))
    assert st.connected and st.settings.mode is Mode.LIVE and st.secret_key == KEY and st.settings.account_address == OWNER
    assert not st.running  # a newly connected account never starts trading by itself
    store.set_running(True)
    assert load_startup(str(tmp_path)).running  # ...and once started it stays started across restarts
    store.disconnect()
    st = load_startup(str(tmp_path))
    assert not st.connected and not st.running and "api_secret_key" not in store.load()


def test_key_is_private_on_disk_and_never_in_what_the_panel_can_read(tmp_path: Any) -> None:
    store = ControlStore(str(tmp_path))
    store.save_credentials(OWNER, KEY)
    pub = store.public()
    assert pub == {"account_address": OWNER, "has_key": True, "running": False}
    assert KEY not in json.dumps(pub)
    token = store.token()
    assert len(token) >= 30 and store.token() == token
    if sys.platform != "win32":
        assert stat.S_IMODE(os.stat(store.path).st_mode) == 0o600
        assert stat.S_IMODE(os.stat(store.token_path).st_mode) == 0o600
    store.path.write_text("{ this is not json")
    assert not load_startup(str(tmp_path)).connected  # a corrupt file is ignored, not fatal


def test_preferences_are_range_checked_and_applied(tmp_path: Any) -> None:
    store = ControlStore(str(tmp_path))
    with pytest.raises(ControlError, match="between"):
        store.set_preferences(risk_aversion=0.2)
    with pytest.raises(ControlError, match="number"):
        store.set_preferences(max_leverage="lots")
    store.set_preferences(risk_aversion="8", max_leverage=5)
    st = load_startup(str(tmp_path))
    assert st.settings.risk_aversion == 8.0 and st.settings.max_leverage_cap == 5.0
    store.set_preferences(risk_aversion="", max_leverage=None)  # blank fields change nothing
    assert load_startup(str(tmp_path)).settings.risk_aversion == 8.0


def test_developer_modes_still_work_from_the_environment(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HL_MODE", "paper")
    st = load_startup(str(tmp_path))
    assert st.settings.mode is Mode.PAPER and st.running and not st.connected
    assert Settings.from_env().mode is Mode.PAPER
    monkeypatch.delenv("HL_MODE")
    assert Settings.from_env().mode is Mode.LIVE  # the default, with no confirmation phrase anywhere


# --- engine switches ---------------------------------------------------------------------------------
def test_stopped_engine_keeps_learning_but_sends_nothing_and_close_position_works() -> None:
    from app.brain.engine import Engine
    from app.exchange.paper import PaperVenue

    s = dataclasses.replace(Settings(), horizon_s=5.0, models=("ridge",))
    trained = run_backtest(synthetic_events(3000, signal=6.0, seed=4), s, META, keep_engine=True)["engine"]
    assert trained.orders_sent > 0  # it trades when started

    venue = PaperVenue(1000.0, META, s.taker_fee, s.maker_fee, 0.0)
    eng = Engine(dataclasses.replace(s, mode=Mode.BACKTEST), venue, META)
    assert eng.load_state(trained.dump_state()) == ""
    eng.paused = True
    start = eng.arena.champ.model.n_obs
    t = 0.0
    for t, kind, payload in synthetic_events(1500, signal=6.0, seed=6):
        if kind == "book":
            eng.on_tick(t)
            eng.on_book(payload)
        else:
            eng.on_trades(payload)
    assert eng.orders_sent == 0 and venue.pos == 0.0 and eng.arena.champ.model.n_obs > start + 1000
    assert eng.last["extra"].get("paused") or eng.last["action"] == "HOLD"

    assert venue.book is not None
    venue.submit(OrderIntent("BTC", True, 0.05, venue.book.best_ask * 1.01, "Ioc"), t)
    venue.on_book(venue.book)
    assert venue.pos == 0.05
    eng.paused = False
    msg = eng.flatten(t)
    venue.on_book(venue.book)
    assert "Closing 0.05 BTC" in msg and venue.pos == 0.0 and eng.paused
    assert "no position" in eng.flatten(t)


# --- the live venue against a fake exchange ---------------------------------------------------------------
class FakeInfo:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.calls: list[str] = []
        self.spot_usdc = "10.0"
        self.position = "0.0"
        self.fills: list[dict[str, Any]] = []
        self.fail = False

    def _hit(self, name: str) -> None:
        self.calls.append(name)
        if self.fail:
            raise ConnectionError("exchange unreachable")

    def query_user_abstraction_state(self, user: str) -> str:
        self._hit("abstraction")
        return "unifiedAccount"

    def user_state(self, address: str) -> dict[str, Any]:
        self._hit("perp")
        pos = [{"position": {"coin": "BTC", "szi": self.position, "entryPx": "85000", "unrealizedPnl": "0.0"}}]
        return {"marginSummary": {"accountValue": "0.0"}, "assetPositions": pos if float(self.position) else []}

    def spot_user_state(self, address: str) -> dict[str, Any]:
        self._hit("spot")
        return {"balances": [{"coin": "USDC", "total": self.spot_usdc, "hold": "0.0"}]}

    def open_orders(self, address: str) -> list[dict[str, Any]]:
        self._hit("open_orders")
        return []

    def user_fills_by_time(self, address: str, start: int) -> list[dict[str, Any]]:
        self._hit("fills")
        return self.fills


class FakeExchange:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.orders: list[tuple[Any, ...]] = []

    def order(self, *a: Any, **kw: Any) -> dict[str, Any]:
        self.orders.append((*a, kw))
        return {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"filled": {"totalSz": "0.0002"}}]}}}

    def cancel(self, *a: Any) -> dict[str, Any]:
        return {"status": "ok"}

    def update_leverage(self, *a: Any, **kw: Any) -> dict[str, Any]:
        return {"status": "ok"}


@pytest.fixture
def live(monkeypatch: pytest.MonkeyPatch) -> HyperliquidLive:
    import hyperliquid.exchange
    import hyperliquid.info

    monkeypatch.setattr(hyperliquid.info, "Info", FakeInfo)
    monkeypatch.setattr(hyperliquid.exchange, "Exchange", FakeExchange)
    return HyperliquidLive("https://example.invalid", OWNER, KEY, META)


def test_live_venue_reports_unified_balance_and_becomes_unknown_on_failure(live: HyperliquidLive) -> None:
    info: FakeInfo = live._info
    assert not live.account(0.0).known  # nothing is assumed before the first reconciliation
    live.refresh()
    acct = live.account(0.0)
    assert acct.known and acct.equity == 10.0 and acct.position == 0.0  # the $10 that used to read as zero
    assert live.snapshot["abstraction"] == "unifiedAccount" and live.snapshot["address"] == OWNER
    info.fail = True
    live.refresh()
    assert not live.account(0.0).known and live.errors == 1  # the safety kernel will now block trading
    info.fail = False
    live.refresh()
    assert live.account(0.0).known


def test_live_venue_stays_within_rate_limits_and_reports_each_fill_once(live: HyperliquidLive) -> None:
    info: FakeInfo = live._info
    for _ in range(5):
        live.refresh()
    assert info.calls.count("perp") == 5 and info.calls.count("spot") == 5
    assert info.calls.count("open_orders") == 1 and info.calls.count("fills") == 1  # expensive calls are spaced out
    assert info.calls.count("abstraction") == 1

    live.submit(OrderIntent("BTC", True, 0.0002, 86000.0, "Ioc"), 100.0)
    live._pool.shutdown(wait=True)
    ex: FakeExchange = live._ex
    assert ex.orders[0][:4] == ("BTC", True, 0.0002, 86000.0) and ex.orders[0][4] == {"limit": {"tif": "Ioc"}}
    assert live.account(0.0).inflight == 0  # acknowledged

    info.position = "0.0002"
    info.fills = [{"tid": 7, "coin": "BTC", "side": "B", "px": "85010", "sz": "0.0002", "fee": "0.0077", "time": 1000, "crossed": True},
                  {"tid": 8, "coin": "ETH", "side": "B", "px": "3000", "sz": "1", "fee": "1", "time": 1000, "crossed": True}]  # fmt: skip
    live.refresh()  # position changed => fills are fetched immediately, not ten seconds later
    live.refresh()
    fills = live.drain_fills()
    assert len(fills) == 1 and fills[0].px == 85010.0 and fills[0].is_buy and not fills[0].maker
    assert live.account(0.0).position == 0.0002 and live.drain_fills() == []


# --- HTTP control API ------------------------------------------------------------------------------------
def test_control_api_requires_token_local_host_and_same_origin_and_runs_async_actions() -> None:
    calls: list[dict[str, Any]] = []

    def start(b: dict[str, Any]) -> dict[str, Any]:
        calls.append(b)
        return {"message": "ok"}

    async def connect(b: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(0)
        if not b.get("api_secret_key"):
            raise ControlError("That is not a private key")
        return {"message": "Connected", "balance": 10.0}

    ok = {"X-Control-Token": TOKEN, "Host": "127.0.0.1:8787"}

    async def scenario() -> None:
        control = {"get": lambda _: {"connected": False}, "start": start, "connect": connect}
        app = make_app(lambda: {"health": {"ok": True}}, token=TOKEN, control=control, port=8787)
        async with TestClient(TestServer(app)) as c:
            assert (await c.get("/api/state")).status == 200  # read-only status needs no token
            page = await (await c.get("/")).text()
            assert "Start trading" in page and "Connect and fetch balance" in page
            for gone in ("Paper", "Shadow", "Testnet", "I_UNDERSTAND"):
                assert gone not in page  # live is the only mode and there is no phrase to type
            assert (await c.post("/api/control/start", json={}, headers={"Host": "127.0.0.1:8787"})).status == 401
            assert (await c.post("/api/control/start", json={}, headers=ok | {"X-Control-Token": "wrong"})).status == 401
            assert (await c.post("/api/control/start", json={}, headers=ok | {"Host": "evil.example:8787"})).status == 403
            assert (await c.post("/api/control/start", json={}, headers=ok | {"Origin": "https://evil.example"})).status == 403
            assert calls == []
            assert (await c.post("/api/control/start", json={}, headers=ok | {"Origin": "http://127.0.0.1:8787"})).status == 200
            assert calls == [{}]
            r = await c.get("/api/control", headers=ok)
            assert r.status == 200 and (await r.json()) == {"connected": False}
            r = await c.post("/api/control/connect", json={"api_secret_key": ""}, headers=ok)
            assert r.status == 400 and "not a private key" in (await r.json())["error"]
            r = await c.post("/api/control/connect", json={"api_secret_key": "x"}, headers=ok)
            assert r.status == 200 and (await r.json())["balance"] == 10.0
            assert (await c.get("/api/control/start", headers=ok)).status == 405  # actions are POST only

    asyncio.run(scenario())
