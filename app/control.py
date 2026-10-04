"""Operator control: what the dashboard's control panel is allowed to change, and how it is stored.

Choices made in the control panel (mode, credentials, pause, risk preferences) are kept in
`<data_dir>/control.json`, readable only by the service user, and override the environment.
The panel is protected by a random token in `<data_dir>/control_token`.

Safety properties:
- the API key is write-only: it can be set or cleared, never read back through the dashboard;
- real-money mode needs credentials *and* the typed confirmation phrase;
- if the requested mode cannot be started (missing credentials, bad key, missing libraries),
  the process falls back to paper mode and reports why, instead of crash-looping.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any

from app.config.settings import LIVE_CONFIRM_PHRASE, Mode, Settings

SELECTABLE_MODES = (Mode.PAPER, Mode.SHADOW, Mode.TESTNET, Mode.LIVE)
_ADDRESS = re.compile(r"^0x[0-9a-fA-F]{40}$")
_KEY = re.compile(r"^(0x)?[0-9a-fA-F]{64}$")


class ControlError(ValueError):
    """A change the operator asked for is not valid; the message is shown in the panel."""


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

    # --- changes requested from the panel ---------------------------------
    def set_credentials(self, address: str, key: str) -> None:
        address, key = address.strip(), key.strip()
        if not _ADDRESS.match(address):
            raise ControlError("account address must be 0x followed by 40 hex characters")
        if not _KEY.match(key):
            raise ControlError("API wallet private key must be 64 hex characters (optionally starting 0x)")
        self.save(self.load() | {"account_address": address, "api_secret_key": key})

    def clear_credentials(self) -> None:
        d = self.load()
        for k in ("account_address", "api_secret_key", "live_confirm"):
            d.pop(k, None)
        if d.get("mode") in (Mode.TESTNET.value, Mode.LIVE.value):
            d["mode"] = Mode.PAPER.value
        self.save(d)

    def set_mode(self, mode: str, confirm: str = "") -> None:
        try:
            m = Mode(mode)
        except ValueError:
            raise ControlError(f"unknown mode {mode!r}") from None
        if m not in SELECTABLE_MODES:
            raise ControlError(f"mode {mode!r} cannot be selected here")
        d = self.load()
        if m in (Mode.TESTNET, Mode.LIVE) and not (d.get("api_secret_key") or os.environ.get("HL_API_SECRET_KEY")):
            raise ControlError("save your account address and API wallet key first")
        if m is Mode.LIVE:
            if confirm != LIVE_CONFIRM_PHRASE:
                raise ControlError(f"to trade real money, type the confirmation phrase exactly: {LIVE_CONFIRM_PHRASE}")
            d["live_confirm"] = confirm
        else:
            d.pop("live_confirm", None)  # leaving live mode always requires confirming again to return
        self.save(d | {"mode": m.value})

    def set_preferences(self, risk_aversion: Any = None, max_leverage: Any = None, paper_equity: Any = None) -> None:
        d = self.load()
        for name, val, lo, hi in (("risk_aversion", risk_aversion, 1.0, 100.0), ("max_leverage", max_leverage, 1.0, 100.0),
                                  ("paper_equity", paper_equity, 10.0, 1e9)):  # fmt: skip
            if val is None or val == "":
                continue
            try:
                x = float(val)
            except (TypeError, ValueError):
                raise ControlError(f"{name} must be a number") from None
            if not lo <= x <= hi:
                raise ControlError(f"{name} must be between {lo:g} and {hi:g}")
            d[name] = x
        self.save(d)

    def set_paused(self, paused: bool) -> None:
        self.save(self.load() | {"paused": bool(paused)})

    def public(self) -> dict[str, Any]:
        """What the panel may display. Never includes the key."""
        d = self.load()
        addr = d.get("account_address") or os.environ.get("HL_ACCOUNT_ADDRESS", "")
        return {
            "requested_mode": d.get("mode"), "account_address": addr,
            "has_key": bool(d.get("api_secret_key") or os.environ.get("HL_API_SECRET_KEY")),
            "paused": bool(d.get("paused", False)), "risk_aversion": d.get("risk_aversion"),
            "max_leverage": d.get("max_leverage"), "paper_equity": d.get("paper_equity"),
            "live_confirm_phrase": LIVE_CONFIRM_PHRASE,
        }  # fmt: skip


@dataclasses.dataclass
class Startup:
    settings: Settings
    secret_key: str = ""
    paused: bool = False
    error: str = ""  # why the requested mode was not started, if it was not


def load_startup(store_dir: str | None = None) -> Startup:
    """Environment settings with the control panel's choices applied on top, validated."""
    s = Settings.from_env(check_live=False)
    store = ControlStore(store_dir or s.data_dir)
    d = store.load()
    env_confirm = os.environ.get("HL_LIVE_CONFIRM") == LIVE_CONFIRM_PHRASE
    changes: dict[str, Any] = {}
    if d.get("mode") in {m.value for m in SELECTABLE_MODES}:
        changes["mode"] = Mode(d["mode"])
    if d.get("account_address"):
        changes["account_address"] = d["account_address"]
    for k in ("risk_aversion", "paper_equity"):
        if isinstance(d.get(k), (int, float)):
            changes[k] = float(d[k])
    if isinstance(d.get("max_leverage"), (int, float)):
        changes["max_leverage_cap"] = float(d["max_leverage"])
    s = dataclasses.replace(s, **changes)
    key = d.get("api_secret_key") or os.environ.get("HL_API_SECRET_KEY", "")
    out = Startup(s, key, bool(d.get("paused", False)))

    wanted = s.mode
    if wanted in (Mode.TESTNET, Mode.LIVE) and not (key and s.account_address):
        out.error = f"{wanted.value} mode needs an account address and API wallet key; running in paper mode instead"
    elif wanted is Mode.LIVE and not (env_confirm or d.get("live_confirm") == LIVE_CONFIRM_PHRASE):
        out.error = "live mode has not been confirmed; running in paper mode instead"
    if out.error:
        out.settings = dataclasses.replace(s, mode=Mode.PAPER)
    return out
