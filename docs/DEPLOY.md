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
`/opt/hltrader/app`, writes a `.env` for **paper mode with recording on**, installs the systemd
service and starts it, then checks that it came up. It asks for no keys and stores none. Re-run the
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
sudo systemctl stop hltrader              # stop (does not close positions; irrelevant in paper mode)
```

To run research on the VPS recordings:

```bash
cd /opt/hltrader/app && sudo -u hltrader .venv/bin/python -m app.research.discovery "data/raw-*.jsonl*"
```

## Control panel

Open `http://127.0.0.1:8787` in a browser **on the server** (for example through RustDesk), or
from your own computer through an SSH tunnel. The top bar always shows the mode in force: green
for paper and shadow, amber for testnet, red for LIVE.

Paste the control token once to unlock the controls (the installer prints it; to see it again:
`sudo cat /opt/hltrader/app/data/control_token`). Then you can:

- **Switch mode** - Paper, Shadow, Testnet, Live. The engine restarts itself in a few seconds and
  keeps what it has learned. Testnet and Live are disabled until credentials are saved; Live also
  asks you to type the confirmation phrase, every time you enter it.
- **Pause / resume** - paused, it keeps watching and learning but sends no orders.
- **Close position and pause** - cancels resting orders, closes the whole position at market.
- **Save or remove Hyperliquid credentials** - account address and API wallet key.
- **Risk preferences** - risk aversion, a leverage cap, and the paper balance.

If a mode cannot be started (wrong key, exchange unreachable), the engine runs in paper mode and
shows the reason in a red banner instead of failing.

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
