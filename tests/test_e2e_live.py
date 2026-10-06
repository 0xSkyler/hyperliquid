"""End to end: the real app (runner, control panel API, engine, live venue, the real Hyperliquid
SDK and its signing) against a mock exchange. This is the operator's whole journey:

    not connected -> connect with an API key -> balance shown -> Start -> close position -> disconnect
"""

from __future__ import annotations

import asyncio
import json
import secrets
import socket
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp.test_utils import TestServer

from app.control import ControlStore, load_startup
from app.main import run
from tests.mock_hyperliquid import OWNER, MockHyperliquid

KEY = "0x" + secrets.token_hex(32)  # throwaway key generated for this run


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_operator_journey_connect_balance_start_close_disconnect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def journey() -> None:
        mock = MockHyperliquid(usdc=10.0, position=0.0002)  # $10 in a unified account, a small long already open
        server = TestServer(mock.app())
        await server.start_server()
        port = _free_port()
        for k, v in {"HL_API_URL": f"http://127.0.0.1:{server.port}", "HL_DATA_DIR": str(tmp_path), "HL_DASHBOARD_PORT": str(port),
                     "HL_NEWS_FEEDS": "", "HL_CHART_MODEL_DIR": str(tmp_path / "none"), "HL_MODELS": "ridge"}.items():  # fmt: skip
            monkeypatch.setenv(k, v)
        monkeypatch.delenv("HL_MODE", raising=False)
        stop = asyncio.Event()

        async def app_loop() -> None:
            while await run(load_startup(), None, stop):
                pass

        app_task = asyncio.create_task(app_loop())
        base = f"http://127.0.0.1:{port}"
        token = ""

        async with aiohttp.ClientSession() as http:

            async def state() -> dict[str, Any]:
                async with http.get(base + "/api/state") as r:
                    return dict(await r.json())

            async def call(path: str = "", body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
                headers = {"X-Control-Token": token}
                url = base + "/api/control" + path
                async with (http.post(url, json=body, headers=headers) if body is not None else http.get(url, headers=headers)) as r:
                    return r.status, dict(await r.json())

            async def until(pred: Any, what: str, timeout: float = 20.0) -> Any:
                deadline = asyncio.get_running_loop().time() + timeout
                last: Any = None
                while asyncio.get_running_loop().time() < deadline:
                    if app_task.done():
                        app_task.result()
                    try:
                        last = await pred()
                        if last:
                            return last
                    except (aiohttp.ClientError, KeyError, AssertionError):
                        pass
                    await asyncio.sleep(0.25)
                raise AssertionError(f"timed out waiting for: {what} (last={last!r})")

            # 1. Fresh install: watching the market, connected to nothing, nothing can be started.
            await until(lambda: state(), "dashboard up")
            token = ControlStore(str(tmp_path)).token()
            h = (await state())["health"]
            assert h["connected"] is False and h["running"] is False
            status, body = await call("/start", {})
            assert status == 400 and "Connect your Hyperliquid account first" in body["error"]
            assert (await call("", None))[1]["connected"] is False

            # 2. A key Hyperliquid does not know is refused and not stored.
            mock.role = {"role": "missing"}
            status, body = await call("/connect", {"api_secret_key": KEY})
            assert status == 400 and "does not recognise" in body["error"]
            mock.role = {"role": "user"}
            status, body = await call("/connect", {"api_secret_key": KEY})
            assert status == 400 and "main wallet" in body["error"]
            assert "api_secret_key" not in ControlStore(str(tmp_path)).load()

            # 3. Connect with only the API key: the account is found and its balance returned.
            mock.role = {"role": "agent", "data": {"user": OWNER}}
            status, body = await call("/connect", {"api_secret_key": KEY, "account_address": ""})
            assert status == 200 and OWNER in body["message"] and "$10.00" in body["message"]

            async def connected() -> Any:
                c = (await call())[1]
                return c if c.get("connected") and c["account"].get("equity") == 10.0 else None

            c = await until(connected, "engine restarted and connected with the unified balance")
            assert c["account"]["abstraction"] == "unifiedAccount" and c["account"]["position"] == 0.0002
            assert c["account_address"] == OWNER and c["has_key"] and KEY not in json.dumps(c)
            assert c["running"] is False  # connecting never starts trading by itself
            h = (await state())["health"]
            assert h["connected"] is True and h["running"] is False and not h["startup_error"]
            assert "updateLeverage" in mock.actions and mock.orders == []  # set up, but no order sent

            # 4. Start, and it stays started across an engine restart.
            status, body = await call("/start", {})
            assert status == 200 and "Trading started" in body["message"]
            assert (await state())["health"]["running"] is True
            assert (await call("/preferences", {"max_leverage": "3", "risk_aversion": "6"}))[0] == 200

            async def restarted_running() -> Any:
                c = (await call())[1]
                return c if c.get("connected") and c["effective"]["max_leverage"] == 3.0 and c["running"] else None

            c = await until(restarted_running, "restart with new risk settings, still running")
            assert c["effective"]["risk_aversion"] == 6.0 and c["reason"]

            # 5. Close position and stop: one reduce-only market sell of exactly the position, through the real SDK.
            await until(lambda: _has_book(state), "market data flowing after restart")
            status, body = await call("/flatten", {})
            assert status == 200 and "Closing 0.0002 BTC" in body["message"]
            await until(lambda: _true(len(mock.orders) == 1), "order reaches the exchange")
            o = mock.orders[0]
            assert o["b"] is False and float(o["s"]) == 0.0002 and o["r"] is True and o["t"] == {"limit": {"tif": "Ioc"}}
            assert float(o["p"]) < mock.mid and float(o["p"]) == int(float(o["p"]))  # marketable, valid BTC price tick

            async def flat() -> Any:
                c = (await call())[1]
                return c if c["account"].get("position") == 0.0 and not c["running"] else None

            await until(flat, "position closed and trading stopped")
            assert (await state())["engine"]["journal"]["fills"] == 1 and mock.position == 0.0
            assert (await state())["health"]["faults"] == []  # the fill was reconciled, no position mismatch

            # 6. Stop is honoured, and with no balance Start is refused.
            mock.usdc = 0.0
            assert (await call("/refresh", {}))[0] == 200
            status, body = await call("/start", {})
            assert status == 400 and "no balance" in body["error"]

            # 7. Disconnect removes the key and returns to "not connected".
            assert (await call("/disconnect", {}))[0] == 200
            await until(lambda: _is(state, False), "disconnected after restart")
            saved = ControlStore(str(tmp_path)).load()
            assert "api_secret_key" not in saved and saved.get("running") is False
            assert len(mock.orders) == 1  # nothing else was ever sent

        stop.set()
        await asyncio.wait_for(app_task, 20)
        await server.close()
        assert (tmp_path / "state" / "engine-BTC.pkl").is_file()  # learned state saved on the way out

    asyncio.run(journey())


async def _true(v: bool) -> bool:
    return v


async def _has_book(state: Any) -> bool:
    return (await state())["engine"]["mid"] is not None


async def _is(state: Any, connected: bool) -> bool:
    return (await state())["health"]["connected"] is connected
