# Going from paper to real orders

The order path (`HyperliquidLive`) was written against the official `hyperliquid-python-sdk` but
has **not** been run against a funded account. Treat the first steps below as its test plan.

## Before you consider it

Run paper mode with `HL_RECORD_RAW=1` for at least several days and look at the dashboard's
calibration block. If `trusted_beta` is 0 in every regime, the engine has found no edge it can
defend statistically and it will not trade live either. Paper results are an upper bound on live
results: real queues are longer and real latency is worse than the simulation.

## The easy way: the control panel

Open the control panel (docs/DEPLOY.md), save your account address and API wallet key under
"Hyperliquid account", then press **Testnet**. To go further, press **LIVE** and type the
confirmation phrase. The steps below are the equivalent by hand, and the checks in step 4 apply
either way.

## 1. Testnet

1. Create an **API (agent) wallet** at app.hyperliquid-testnet.xyz -> More -> API. An agent wallet
   can trade but cannot withdraw. Never put your main wallet's private key on a server.
2. `pip install -e ".[live]"`
3. In `.env` (chmod 600, never committed):

   ```
   HL_MODE=testnet
   HL_ACCOUNT_ADDRESS=0x...      # the main account address the agent trades for
   HL_API_SECRET_KEY=0x...       # the agent wallet's private key
   ```
4. Start it and check, in order: the startup log shows your real testnet equity; a position opened
   by hand in the UI appears in the dashboard within a few seconds and the engine does not treat
   the account as flat; killing and restarting the process with an open position resumes with that
   position; pulling the network cable produces `HALTED_INSTRUMENTATION` and no orders.

On startup in testnet/live the engine sets BTC to cross margin at the venue's maximum leverage so
the venue never rejects for margin a size the utility engine chose. Actual exposure is whatever
the engine sizes, not that setting.

## 2. Mainnet

Same as above with a mainnet agent wallet and:

```
HL_MODE=live
HL_LIVE_CONFIRM=I_UNDERSTAND_REAL_MONEY
```

Without the confirmation variable the process refuses to start. Fund the account with an amount
you can lose entirely: with default preferences a 1.5% gap against a full-size position costs
about half the account.

## Stopping

`systemctl stop hltrader` (or Ctrl-C) stops decisions; it does **not** close positions or cancel
resting orders. Do that in the Hyperliquid UI.
