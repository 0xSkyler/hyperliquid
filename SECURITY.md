# Security

- Secrets live only in `.env` (git-ignored, chmod 600) or the process environment. Nothing in the
  code logs them. CI runs a secret scan on every push.
- Use a Hyperliquid **API agent wallet** for `HL_API_SECRET_KEY`. It can place orders but cannot
  withdraw. Never deploy your main wallet key.
- Run as an unprivileged user (see `deploy/hltrader.service`, which also sandboxes the filesystem).
- The dashboard is bound to localhost. Its status pages need no login; every control action needs
  the token in `data/control_token` (random, readable only by the service user), must be addressed
  to the machine by a local name, and is refused if it originates from any other web page. Do not
  expose the port; use the server's own desktop or an SSH tunnel. Anyone who can read that token
  can switch the engine to live trading, so treat it like a password.
- An API wallet key saved through the control panel is stored in `data/control.json`
  (permissions 600) and is never sent back to the browser. Use an API wallet, which cannot withdraw.
- News and all other fetched text is untrusted data: sanitised, length-capped, stored, displayed
  via `textContent`. It is never executed or interpreted as instructions and does not reach the
  decision engine.
- Learned state is stored with Python `pickle` (`data/state/*.pkl`). Unpickling runs code, so treat
  that directory like the program itself: writable only by the service user, and never load a
  state file obtained from someone else.
- Live trading needs two operator actions in the control panel: connecting an account, and pressing
  Start. Connecting never starts trading. A main-wallet private key (which could withdraw funds) is
  detected and refused; only API wallet keys are stored.
