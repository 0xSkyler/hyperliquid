"""Unattended training run: learn from the live market for a fixed number of hours, then save and publish.

    python scripts/train.py --hours 24 --coins BTC,ETH,SOL --push

For each market it runs the real engine, connected to no account (so it cannot place an order),
on the live Hyperliquid feed. In that mode the engine:

- trains its forecasts and scores them out of sample;
- practises quoting on a simulator with pretend money;
- files a lesson every second for every quote and take it could have made (app/scalp/lessons.py);
- records the raw market data for later research.

The supervisor restarts a market's process if it dies, writes a progress line every 30 minutes,
and at the end exports each market's learned state to models/state/ (which the engine loads at
start-up if it is more experienced than what it already has), writes models/training/REPORT.md,
and with --push commits and pushes both.

The computer has to stay on and awake for the whole run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess  # noqa: S404 - runs this project's own modules and git
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BASE_PORT = 8801


def _env(coin: str, port: int, data_dir: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("HL_MODE", "HL_API_SECRET_KEY", "HL_ACCOUNT_ADDRESS")}
    env.update({"HL_COIN": coin, "HL_DATA_DIR": str(data_dir), "HL_DASHBOARD_PORT": str(port), "HL_RECORD_RAW": "1",
                "HL_NEWS_FEEDS": "", "HL_STRATEGY": "maker", "PYTHONPATH": str(ROOT)})  # fmt: skip
    return env


def _state(port: int) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=5) as r:  # noqa: S310 - localhost
            return dict(json.load(r))
    except Exception:  # noqa: BLE001
        return None


def _summary(coin: str, st: dict[str, Any] | None) -> dict[str, Any]:
    if st is None:
        return {"coin": coin, "up": False}
    e, sc = st["engine"], st["engine"]["scalper"]
    fa, pr = sc["fast_alpha"], sc["practice"]
    takes = sc["lessons"]["takes"]
    quotes = sc["lessons"]["quotes"]
    return {
        "coin": coin, "up": True, "seconds_learned": e["ticks"], "faults": st["health"]["faults"],
        "forecast_accuracy": fa["oos_ic"], "forecast_trust": fa["trusted_beta"], "forecasts_scored": fa["scored"],
        "practice_fills": pr["fills"], "practice_edge_bps": pr["edge_bps_5s"], "real_quoting_earned": sc["making_allowed"],
        "take_lessons": takes, "best_take_bps": max((t["worth_lower_bound_bps"] for t in takes if t["times"] >= 50), default=None),
        "quote_situations_judged": sum(q["verdict"] != "not enough hits yet" for q in quotes),
        "quote_situations_that_pay": sum(q["verdict"] == "quote here" for q in quotes),
        "champion": e["model"]["champion"], "promotions": len(e["model"]["promotions"]),
        "slow_models": [{k: m[k] for k in ("name", "oos_ic", "max_trusted_beta", "resolved")} for m in e["model"]["arena"]],
    }  # fmt: skip


def _stay_awake() -> None:
    """Ask Windows not to idle-sleep while this process runs. Changes no settings and ends with the process.
    A closed laptop lid or a manual Sleep still stops the run."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def _git(*args: str) -> int:
    ident = ["-c", "user.name=nahian", "-c", "user.email=nahian@ordnur.com"]
    return subprocess.run(["git", *ident, *args], cwd=ROOT, check=False).returncode  # noqa: S603, S607


def _report(final: list[dict[str, Any]], hours: float, restarts: dict[str, int], fees: tuple[float, float]) -> str:
    lines = [f"# Training run: {hours:g} hours on live Hyperliquid data", "",
             f"Finished {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. No account was connected; nothing was traded.",
             f"Fees assumed: maker {fees[0]:g} bps, taker {fees[1]:g} bps per trade.", ""]  # fmt: skip
    for s in final:
        lines.append(f"## {s['coin']}")
        if not s.get("up"):
            lines += ["The process was not running at the end; see data/train/ for its log.", ""]
            continue
        lines += [
            f"- Learned from {s['seconds_learned'] / 3600:.1f} hours of market data; process restarts: {restarts.get(s['coin'], 0)}.",
            f"- Fast forecast (next 5 s): accuracy {s['forecast_accuracy']:.2f} (0 = none, 1 = perfect), "
            f"trust {s['forecast_trust']:.2f}, {int(s['forecasts_scored']):,} forecasts scored.",
            f"- Practice quoting: {s['practice_fills']} fills, worth {s['practice_edge_bps']:.2f} bps each after 5 s "
            f"(needs more than the maker fee of {fees[0]:g}). Real quoting earned: {'YES' if s['real_quoting_earned'] else 'no'}.",
            f"- Quote lessons: {s['quote_situations_judged']} situations judged, {s['quote_situations_that_pay']} of them pay.",
            "- Take lessons (what crossing the spread was worth, by forecast strength):", "",
            "| Forecast (bps) | Times seen | Worth (bps) | Lower bound | Verdict |", "|---|---|---|---|---|",
        ]  # fmt: skip
        for t in s["take_lessons"]:
            lines.append(f"| {t['forecast_bps']} | {t['times']:,} | {t['worth_bps']:.2f} | {t['worth_lower_bound_bps']:.2f} | {t['verdict']} |")
        lines += ["", "- 60-second models: " + ", ".join(f"{m['name']} accuracy {m['oos_ic']:.3f} trust {m['max_trusted_beta']:.2f}"
                                                       for m in s["slow_models"]) + f". Champion: {s['champion']}.", ""]  # fmt: skip
    lines += ["## How to read this", "",
              "The engine trades for real only where a lesson's *lower bound* beats the fee. If every verdict above says",
              "'avoid', the trained engine will correctly place no orders at this fee level, however long it trains.",
              "The learned state is in `models/state/`; the engine loads it at start-up when it is more experienced than its own."]  # fmt: skip
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--coins", default="BTC,ETH,SOL")
    ap.add_argument("--push", action="store_true", help="commit and push the learned state and report at the end")
    ap.add_argument("--progress-min", type=float, default=30.0)
    args = ap.parse_args()
    coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    _stay_awake()
    end = time.time() + args.hours * 3600
    out = ROOT / "models" / "training"
    out.mkdir(parents=True, exist_ok=True)
    procs: dict[str, subprocess.Popen[bytes]] = {}
    restarts: dict[str, int] = {}
    ports = {c: BASE_PORT + i for i, c in enumerate(coins)}
    dirs = {c: ROOT / "data" / "train" / c for c in coins}

    def start(coin: str) -> None:
        dirs[coin].mkdir(parents=True, exist_ok=True)
        log = open(dirs[coin] / "train.log", "ab")  # noqa: SIM115 - handed to the child process
        remaining = max(end - time.time(), 5.0)
        procs[coin] = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", "app.main", "--duration", f"{remaining:.0f}"], cwd=ROOT, env=_env(coin, ports[coin], dirs[coin]),
            stdout=log, stderr=subprocess.STDOUT,
        )  # fmt: skip

    for c in coins:
        start(c)
    print(f"training {coins} for {args.hours:g} h; dashboards on ports {list(ports.values())}", flush=True)
    last_progress = time.time()
    latest: dict[str, dict[str, Any]] = {}
    while time.time() < end - 20:
        time.sleep(30)
        for c in coins:
            if procs[c].poll() is not None and time.time() < end - 120:
                restarts[c] = restarts.get(c, 0) + 1
                print(f"{c}: process exited with {procs[c].returncode}; restarting (#{restarts[c]})", flush=True)
                time.sleep(min(60, 5 * restarts[c]))
                start(c)
            st = _state(ports[c])
            if st is not None:
                latest[c] = _summary(c, st)
        if time.time() - last_progress >= args.progress_min * 60:
            last_progress = time.time()
            snap = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "hours_left": (end - time.time()) / 3600, "markets": list(latest.values())}
            with open(out / "progress.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(snap, default=float) + "\n")
            for s in latest.values():
                print(f"[{snap['t']}] {s['coin']}: {s['seconds_learned'] / 3600:.1f} h learned, forecast accuracy "
                      f"{s['forecast_accuracy']:.2f}, practice fills {s['practice_fills']} edge {s['practice_edge_bps']:.2f}, "
                      f"best take lower bound {s['best_take_bps']}", flush=True)  # fmt: skip

    for c in coins:  # let each process finish its own clean shutdown, which saves its state
        try:
            procs[c].wait(timeout=180)
        except subprocess.TimeoutExpired:
            procs[c].terminate()

    exported = []
    for c in coins:
        dest = ROOT / "models" / "state" / f"engine-{c}.pkl"
        rc = subprocess.run([sys.executable, "-m", "app.research.export_state", str(dirs[c]), c, str(dest)], cwd=ROOT,  # noqa: S603
                            env=_env(c, ports[c], dirs[c]), check=False).returncode  # fmt: skip
        if rc == 0:
            exported.append(dest)
    final = [latest.get(c, {"coin": c, "up": False}) for c in coins]
    env = _env(coins[0], 0, dirs[coins[0]])
    fees = (float(env.get("HL_MAKER_FEE_BPS", 1.5)), float(env.get("HL_TAKER_FEE_BPS", 4.5)))
    (out / "report.json").write_text(json.dumps({"hours": args.hours, "restarts": restarts, "markets": final}, indent=1, default=float),
                                     encoding="utf-8")  # fmt: skip
    (out / "REPORT.md").write_text(_report(final, args.hours, restarts, fees), encoding="utf-8")
    print(f"exported {len(exported)} learned states; report in {out / 'REPORT.md'}", flush=True)
    if args.push:
        _git("add", "models/state", "models/training")
        _git("commit", "-q", "-m", f"Learned state and report from a {args.hours:g}-hour training run on live data\n\n"
                                   "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")  # fmt: skip
        rc = _git("push")
        print("pushed to GitHub" if rc == 0 else f"git push failed with code {rc}; the commit is local", flush=True)


if __name__ == "__main__":
    main()
