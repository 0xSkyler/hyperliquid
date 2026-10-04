# Ubuntu 24.04 VPS deployment

## Pick a region by measurement

From each candidate VPS:

```bash
python scripts/latency.py
```

It reports REST round-trip and WebSocket message-age percentiles to api.hyperliquid.xyz. Choose
the lowest; do not assume.

## Native (systemd)

```bash
sudo apt update && sudo apt install -y python3.12-venv git chrony
sudo systemctl enable --now chrony            # clock sync: staleness checks depend on it
sudo useradd -r -m -d /opt/hltrader -s /usr/sbin/nologin hltrader
sudo -u hltrader git clone <your repo> /opt/hltrader/app
cd /opt/hltrader/app
sudo -u hltrader python3 -m venv .venv
sudo -u hltrader .venv/bin/pip install -e ".[fast]"
sudo -u hltrader cp .env.example .env && sudo chmod 600 .env
sudo cp deploy/hltrader.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now hltrader
journalctl -u hltrader -f
```

The unit restarts on failure, caps memory, and logs to journald (rotation is journald's:
set `SystemMaxUse=500M` in `/etc/systemd/journald.conf`). Data files under `data/` are append-only
JSONL; rotate with `deploy/logrotate.conf` (`sudo cp deploy/logrotate.conf /etc/logrotate.d/hltrader`).

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
