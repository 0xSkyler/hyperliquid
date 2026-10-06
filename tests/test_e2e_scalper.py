"""End to end: the scalper quoting through the real Hyperliquid SDK against the mock exchange.

Connect -> Start -> two post-only quotes rest in the book -> one is hit -> the engine sees the
fill and its inventory -> Stop cancels everything.
"""

from __future__ import annotations

import asyncio
import secrets
import socket
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp.test_utils import TestServer

import app.market.state as market_state
from app.control import ControlStore, load_startup
from app.main import run
from tests.mock_hyperliquid import MockHyperliquid

KEY = "0x" + secrets.token_hex(32)  # throwaway key generated for this run


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_scalper_rests_two_quotes_handles_a_fill_and_cancels_on_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def journey() -> None:
        mock = MockHyperliquid(usdc=1000.0)
        mock.walk = 3.0  # a moving market, so volatility is measurable
        server = TestServer(mock.app())
        await server.start_server()
        port = _free_port()
        monkeypatch.setattr(market_state, "LOOKBACK_S", 8.0)  # warm up in seconds instead of five minutes
        for k, v in {"HL_API_URL": f"http://127.0.0.1:{server.port}", "HL_DATA_DIR": str(tmp_path), "HL_DASHBOARD_PORT": str(port),
                     "HL_NEWS_FEEDS": "", "HL_CHART_MODEL_DIR": str(tmp_path / "none"), "HL_MODELS": "ridge",
                     "HL_STRATEGY": "maker", "HL_SCALP_MIN_REQUOTE_S": "0.5"}.items():  # fmt: skip
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

            async def post(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
                async with http.post(base + "/api/control" + path, json=body, headers={"X-Control-Token": token}) as r:
                    return r.status, dict(await r.json())

            async def until(pred: Any, what: str, timeout: float = 40.0) -> Any:
                deadline = asyncio.get_running_loop().time() + timeout
                last: Any = None
                while asyncio.get_running_loop().time() < deadline:
                    if app_task.done():
                        app_task.result()
                    try:
                        last = await pred()
                        if last:
                            return last
                    except (aiohttp.ClientError, KeyError, AssertionError, StopIteration):
                        pass
                    await asyncio.sleep(0.25)
                raise AssertionError(f"timed out waiting for: {what} (last={last!r})")

            await until(state, "dashboard up")
            token = ControlStore(str(tmp_path)).token()
            assert (await post("/connect", {"api_secret_key": KEY}))[0] == 200

            async def connected() -> bool:
                return bool((await state())["health"]["connected"])

            await until(connected, "connected")
            assert mock.resting == [] and mock.orders == []  # connected but not started: nothing rests in the book
            assert (await post("/start", {}))[0] == 200

            # Two post-only quotes, one each side, on valid ticks, not crossing the market.
            async def two_quotes() -> bool:
                return len(mock.resting) == 2 and {r["b"] for r in mock.resting} == {True, False}

            await until(two_quotes, "a bid and an ask resting on the exchange")
            bid = next(r for r in mock.resting if r["b"])
            ask = next(r for r in mock.resting if not r["b"])
            for q in (bid, ask):
                assert q["t"] == {"limit": {"tif": "Alo"}} and q["r"] is False
                assert q["c"].startswith("0x") and len(q["c"]) == 34  # our client id travelled through the SDK
                assert float(q["p"]) == int(float(q["p"])) and float(q["s"]) * float(q["p"]) >= 10.0
            assert float(bid["p"]) < float(ask["p"])
            sc = (await state())["engine"]["scalper"]
            assert sc["enabled"] and sc["orders_placed"] >= 2

            # The market hits our bid: the engine must see the fill and the inventory, and keep its ask working.
            filled = mock.hit(True)

            async def sees_fill() -> bool:
                s = await state()
                return s["engine"]["journal"]["fills"] == 1 and s["engine"]["scalper"]["quotes"]["inventory_x"] > 0

            await until(sees_fill, "fill and inventory visible to the engine")
            s = await state()
            assert s["health"]["faults"] == [] and s["engine"]["journal"]["maker_ratio"] == 1.0
            assert abs(s["engine"]["scalper"]["quotes"]["inventory_x"] - float(filled["s"]) * mock.mid / 1000.0) < 0.05

            async def requoted_bid() -> bool:
                return any(r["b"] and r["c"] != filled["c"] for r in mock.resting)

            await until(requoted_bid, "a fresh bid after the fill")
            assert all(o["t"] == {"limit": {"tif": "Alo"}} for o in mock.orders if not o["r"])  # it never took liquidity

            # Stop: every resting quote is cancelled; the position is left alone.
            assert (await post("/stop", {}))[0] == 200

            async def book_empty() -> bool:
                return mock.resting == []

            await until(book_empty, "all quotes cancelled after Stop")
            await asyncio.sleep(1.5)
            assert mock.resting == [] and mock.position > 0 and "cancelByCloid" in mock.actions

        stop.set()
        await asyncio.wait_for(app_task, 20)
        await server.close()

    asyncio.run(journey())
