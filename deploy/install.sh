#!/usr/bin/env bash
# Install or update the trader on Ubuntu 24.04 and run it in PAPER mode under systemd.
#
#   curl -fsSL https://raw.githubusercontent.com/0xSkyler/hyperliquid/main/deploy/install.sh | sudo bash
#
# Safe to re-run: it updates the code and restarts the service, and never touches an existing
# .env, recordings or learned state. It does not ask for, create or store any keys: paper mode
# needs none. Going beyond paper mode is a manual edit of /opt/hltrader/app/.env (docs/LIVE.md).
set -euo pipefail

REPO="${HL_REPO:-https://github.com/0xSkyler/hyperliquid.git}"
BRANCH="${HL_BRANCH:-main}"
SRC="${HL_SRC:-}"            # install from a local checkout instead of cloning (used by CI)
HOME_DIR=/opt/hltrader
APP="$HOME_DIR/app"
SVC_USER=hltrader

[ "$(id -u)" -eq 0 ] || { echo "run as root (sudo)"; exit 1; }
export DEBIAN_FRONTEND=noninteractive

echo "==> packages"
apt-get update -qq
apt-get install -y -qq python3-venv python3-dev git chrony libgomp1 curl >/dev/null
systemctl enable --now chrony >/dev/null 2>&1 || true   # staleness checks depend on a correct clock

echo "==> user and code"
id "$SVC_USER" >/dev/null 2>&1 || useradd -r -m -d "$HOME_DIR" -s /usr/sbin/nologin "$SVC_USER"
mkdir -p "$HOME_DIR"
if [ -n "$SRC" ]; then
    mkdir -p "$APP"
    tar -C "$SRC" --exclude=.git --exclude=.venv --exclude=data -cf - . | tar -C "$APP" -xf -
elif [ -d "$APP/.git" ]; then
    git -C "$APP" fetch -q origin "$BRANCH"
    git -C "$APP" reset -q --hard "origin/$BRANCH"   # tracked files only; .env and data/ are untracked
else
    git clone -q --branch "$BRANCH" "$REPO" "$APP"
fi
mkdir -p "$APP/data"

echo "==> python environment"
[ -x "$APP/.venv/bin/python" ] || python3 -m venv "$APP/.venv"
"$APP/.venv/bin/pip" install -q --upgrade pip
"$APP/.venv/bin/pip" install -q -e "$APP[fast]"

echo "==> configuration"
if [ ! -f "$APP/.env" ]; then
    cp "$APP/.env.example" "$APP/.env"
    sed -i 's/^HL_MODE=.*/HL_MODE=paper/; s/^HL_RECORD_RAW=.*/HL_RECORD_RAW=1/' "$APP/.env"
    echo "    created .env (paper mode, recording on)"
else
    echo "    kept existing .env"
fi
MODE="$(grep -E '^HL_MODE=' "$APP/.env" | tail -1 | cut -d= -f2)"
chmod 600 "$APP/.env"
chown -R "$SVC_USER:$SVC_USER" "$HOME_DIR"

echo "==> service"
install -m 644 "$APP/deploy/hltrader.service" /etc/systemd/system/hltrader.service
install -m 755 "$APP/deploy/compress-recordings.sh" /etc/cron.daily/hltrader-compress
systemctl daemon-reload
systemctl enable -q hltrader
systemctl restart hltrader

echo "==> waiting for it to come up"
for _ in $(seq 1 30); do
    if curl -fsS -m 3 http://127.0.0.1:8787/api/state >/dev/null 2>&1; then break; fi
    sleep 2
done
if ! systemctl is-active -q hltrader; then
    echo "service failed to start:"; journalctl -u hltrader -n 40 --no-pager; exit 1
fi
curl -fsS -m 5 http://127.0.0.1:8787/api/state >/dev/null || { echo "dashboard not responding"; journalctl -u hltrader -n 40 --no-pager; exit 1; }

cat <<EOF

Installed and running. Mode: ${MODE}
  logs:       journalctl -u hltrader -f
  status:     systemctl status hltrader
  health:     curl -s http://127.0.0.1:8787/health
  dashboard:  from your own computer run   ssh -L 8787:127.0.0.1:8787 <user>@<this-server>
              then open http://127.0.0.1:8787
  stop:       sudo systemctl stop hltrader
  update:     re-run this installer
EOF
