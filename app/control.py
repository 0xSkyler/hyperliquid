"""Operator control: connect a Hyperliquid account, start and stop live trading.

The control panel has one flow: paste an API wallet key (and optionally the account address),
see the balance, press Start. What it stores lives in `<data_dir>/control.json`, readable only
by the service user. The panel itself is protected by a random token in
`<data_dir>/control_token`.

Properties that are deliberately kept:
- the API key is write-only: it can be saved or removed, never read back through the panel;
- a key is checked against Hyperliquid before it is saved, and a *main wallet* key is refused,
  because unlike an API wallet key it can move funds;
- connecting does not start trading: orders are sent only after Start is pressed;
- if the saved key stops working, the engine stays up, sends nothing, and says why.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any, Protocol

from app.config.settings import Mode, Settings

_ADDRESS = re.compile(r"^0x[0-9a-fA-F]{40}$")
_KEY = re.compile(r"^(0x)?[0-9a-fA-F]{64}$")


class ControlError(ValueError):
    """Something the operator asked for cannot be done; the message is shown in the panel."""


class AccountLookup(Protocol):
    async def role(self, address: str) -> dict[str, Any]: ...
    async def account(self, address: str, coin: str) -> dict[str, Any]: ...


class ControlStore:
    def __init__(self, data_dir: str) -> None:
        self.dir = Path(data_dir)
        self.path = self.dir / "control.json"
        self.token_path = self.dir / "control_token"

    def _write_private(self, path: Path, text: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        tmp.replace(path)

    def load(self) -> dict[str, Any]:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def save(self, d: dict[str, Any]) -> None:
        self._write_private(self.path, json.dumps(d, indent=1))

    def token(self) -> str:
        try:
            t = self.token_path.read_text(encoding="utf-8").strip()
            if len(t) >= 20:
                return t
        except OSError:
            pass
        t = secrets.token_urlsafe(24)
        self._write_private(self.token_path, t + "\n")
        return t

    def credentials(self) -> tuple[str, str]:
        """(account address, API wallet key); environment variables are the fallback."""
        d = self.load()
        return (d.get("account_address") or os.environ.get("HL_ACCOUNT_ADDRESS", ""),
                d.get("api_secret_key") or os.environ.get("HL_API_SECRET_KEY", ""))  # fmt: skip

    def save_credentials(self, address: str, key: str) -> None:
        # A newly connected account never starts trading by itself.
        self.save(self.load() | {"account_address": address, "api_secret_key": key, "running": False})

    def disconnect(self) -> None:
        d = self.load()
        for k in ("account_address", "api_secret_key"):
            d.pop(k, None)
        self.save(d | {"running": False})

    def set_coin(self, coin: str) -> None:
        coin = coin.strip()
        if not re.fullmatch(r"[A-Za-z0-9]{1,12}", coin):
            raise ControlError("That is not a market name.")
        self.save(self.load() | {"coin": coin})

    def set_running(self, running: bool) -> None:
        self.save(self.load() | {"running": bool(running)})

    def set_preferences(self, risk_aversion: Any = None, max_leverage: Any = None) -> None:
        d = self.load()
        for name, val, lo, hi in (("risk_aversion", risk_aversion, 1.0, 100.0), ("max_leverage", max_leverage, 1.0, 100.0)):
            if val is None or val == "":
                continue
            try:
                x = float(val)
            except (TypeError, ValueError):
                raise ControlError(f"{name.replace('_', ' ')} must be a number") from None
            if not lo <= x <= hi:
                raise ControlError(f"{name.replace('_', ' ')} must be between {lo:g} and {hi:g}")
            d[name] = x
        self.save(d)

    def public(self) -> dict[str, Any]:
        """What the panel may display. Never includes the key."""
        address, key = self.credentials()
        d = self.load()
        return {"account_address": address, "has_key": bool(key), "running": bool(d.get("running", False))}


async def verify_credentials(lookup: AccountLookup, key: str, address: str, coin: str) -> dict[str, Any]:
    """Check an API wallet key against Hyperliquid and return the account it trades for, with balances.

    Raises ControlError with a message the operator can act on. Nothing is stored here.
    """
    key, address = key.strip(), address.strip()
    if not _KEY.match(key):
        raise ControlError("That is not a private key: it should be 64 hex characters, optionally starting with 0x.")
    if address and not _ADDRESS.match(address):
        raise ControlError("The wallet address should be 0x followed by 40 hex characters. You can also leave it empty.")
    if not key.startswith("0x"):
        key = "0x" + key
    try:
        from app.exchange.hyperliquid import agent_address

        signer = agent_address(key)
    except ImportError:
        raise ControlError("The order-signing libraries are not installed on this server. Re-run the installer.") from None
    except Exception:  # noqa: BLE001
        raise ControlError("That private key is not valid.") from None

    role = await lookup.role(signer)
    kind = role.get("role")
    if kind == "agent":
        owner = str(role.get("data", {}).get("user", ""))
        if address and address.lower() != owner.lower():
            raise ControlError(
                f"This API wallet is authorised for account {owner}, not {address}. "
                "Clear the address field (it is found automatically) or enter the matching one."
            )
    elif kind == "user":
        raise ControlError(
            "This is the private key of a main wallet, which can withdraw funds. For your safety it is not accepted. "
            "Create an API wallet in Hyperliquid (More > API) and use its key instead."
        )
    elif kind == "missing":
        raise ControlError(
            "Hyperliquid does not recognise this API wallet. Create it under More > API, press Authorize, "
            "and paste the private key shown there. If you just authorised it, wait a few seconds and try again."
        )
    else:
        raise ControlError(f"This key belongs to a {kind} account, which is not supported. Use an API wallet of your main account.")
    account = await lookup.account(owner, coin)
    return account | {"address": owner, "key": key, "signer": signer}


@dataclasses.dataclass
class Startup:
    settings: Settings
    secret_key: str = ""
    running: bool = False  # the operator's Start/Stop switch
    connected: bool = False  # credentials are present, so a live venue will be attempted
    error: str = ""


def load_startup(store_dir: str | None = None) -> Startup:
    """Environment settings with the control panel's choices applied on top.

    The product has one trading mode, live. Without credentials the engine still runs, watching
    the market and learning, but it is not connected to any account and can send nothing.
    (HL_MODE=paper/shadow/testnet remain as developer switches and are not offered in the panel.)
    """
    s = Settings.from_env()
    store = ControlStore(store_dir or s.data_dir)
    d = store.load()
    address, key = store.credentials()
    changes: dict[str, Any] = {"account_address": address}
    if isinstance(d.get("coin"), str) and d["coin"]:
        changes["coin"] = d["coin"]
    if isinstance(d.get("risk_aversion"), (int, float)):
        changes["risk_aversion"] = float(d["risk_aversion"])
    if isinstance(d.get("max_leverage"), (int, float)):
        changes["max_leverage_cap"] = float(d["max_leverage"])
    s = dataclasses.replace(s, **changes)
    needs_account = s.mode in (Mode.LIVE, Mode.TESTNET)
    connected = needs_account and bool(key and address)
    if needs_account and not connected:
        s = dataclasses.replace(s, mode=Mode.PAPER)  # internal simulator, held stopped: nothing is traded
    running = bool(d.get("running", False)) if needs_account else True
    return Startup(s, key, running and (connected or not needs_account), connected)
