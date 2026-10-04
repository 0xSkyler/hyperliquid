# Security

- Secrets live only in `.env` (git-ignored, chmod 600) or the process environment. Nothing in the
  code logs them. CI runs a secret scan on every push.
- Use a Hyperliquid **API agent wallet** for `HL_API_SECRET_KEY`. It can place orders but cannot
  withdraw. Never deploy your main wallet key.
- Run as an unprivileged user (see `deploy/hltrader.service`, which also sandboxes the filesystem).
- The dashboard is read-only, unauthenticated, and bound to localhost. Do not expose it; tunnel.
- News and all other fetched text is untrusted data: sanitised, length-capped, stored, displayed
  via `textContent`. It is never executed or interpreted as instructions and does not reach the
  decision engine.
- LIVE mode refuses to start without `HL_LIVE_CONFIRM`.
