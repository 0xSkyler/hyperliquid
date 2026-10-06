# Ubuntu 24.04 VPS deployment

## Pick a region by measurement

From each candidate VPS:

```bash
python scripts/latency.py
```

It reports REST round-trip and WebSocket message-age percentiles to api.hyperliquid.xyz. Choose
the lowest; do not assume.

## One command (recommended)

On the VPS, as a user with sudo:

```bash
curl -fsSL https://raw.githubusercontent.com/0xSkyler/hyperliquid/main/deploy/install.sh | sudo bash
```

It installs system packages, creates an unprivileged `hltrader` user, puts the code in
`/opt/hltrader/app`, writes a `.env` with recording on, installs the systemd service and starts
it, then checks that it came up. Nothing is traded until you connect an account and press Start
in the control panel. The installer asks for no keys. Re-run the
same command to update: it never touches an existing `.env`, recordings or learned state. This
installer is exercised on a clean Ubuntu 24.04 machine by CI on every push.

The service restarts on failure, is capped at 1 GB of memory, runs with a read-only filesystem
apart from `data/`, and logs to journald (set `SystemMaxUse=500M` in `/etc/systemd/journald.conf`
to bound log size). Recordings are one `data/raw-YYYYMMDD.jsonl` per day, about 50 MB; a daily job
gzips previous days and the research tools read `.jsonl.gz` directly.

```bash
journalctl -u hltrader -f                 # logs
systemctl status hltrader                 # status
curl -s http://127.0.0.1:8787/health      # 200 when healthy, 503 with the fault list otherwise
sudo systemctl stop hltrader              # stop the service (does NOT close an open position)
```

To run research on the VPS recordings:

```bash
cd /opt/hltrader/app && sudo -u hltrader .venv/bin/python -m app.research.discovery "data/raw-*.jsonl*"
```

## Control panel

Open `http://127.0.0.1:8787` in a browser **on the server** (for example through RustDesk), or
from your own computer through an SSH tunnel. The top bar shows the state: grey "NOT CONNECTED",
amber "LIVE - STOPPED", red "LIVE - TRADING".

1. Paste the control token once to unlock the controls (the installer prints it; to see it again:
   `sudo cat /opt/hltrader/app/data/control_token`).
2. Paste your Hyperliquid API wallet key and press **Connect and fetch balance**. The key is
   checked with Hyperliquid and your balance is shown.
3. Press **Start trading**.

Also there: Stop trading, Close position and stop, Disconnect, and Risk (a leverage cap and risk
aversion). See `docs/LIVE.md` for what each does and what the messages mean.

If the saved key stops working (revoked, exchange unreachable), the engine stays up, sends
nothing, and shows the reason in a red banner.

## Docker

```bash
cp .env.example .env
docker compose up -d --build
docker compose logs -f trader
```

Compose starts the trader and Postgres, applies `migrations/001_init.sql`, restarts on failure and
uses `/health` as the container health check. The image is built on every push by CI.

## Dashboard and health

The dashboard binds to 127.0.0.1:8787 and has no authentication. Reach it with an SSH tunnel:

```bash
ssh -L 8787:127.0.0.1:8787 user@vps
```

`GET /health` returns 503 with the list of instrumentation faults whenever the kernel is blocking
trading; point your uptime monitor at it. `/api/state` includes feed age, WebSocket reconnect
count, dropped persistence records and decision-loop lag.
