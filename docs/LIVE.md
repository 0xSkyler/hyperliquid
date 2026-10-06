# Live trading

Live is the only trading mode. Two things gate it, both in the control panel
(`http://127.0.0.1:8787` on the server): an account must be **connected**, and **Start** must be
pressed. Until then the engine only watches the market and learns.

## Connect

1. In Hyperliquid open **More > API**, create an API wallet, and press **Authorize**. Copy the
   private key it shows. An API wallet can trade but cannot withdraw.
2. In the control panel paste that key into "API wallet private key" and press **Connect and
   fetch balance**. The wallet address field is optional: the account the API wallet belongs to
   is looked up from the key.

The key is checked with Hyperliquid before anything is saved:

| What you see | What it means |
|---|---|
| Connected, balance shown | The key is valid; the account and its balance were found. |
| "does not recognise this API wallet" | The API wallet was not authorised, or the key was copied wrongly. |
| "main wallet ... not accepted" | You pasted your main wallet's private key. It can withdraw funds, so it is refused. |
| "authorised for account X, not Y" | The address you typed is not the one this API wallet trades for. Clear the address field. |

### About the balance

New Hyperliquid accounts use the **unified** account mode, where your USDC sits in the Spot
balance and is used directly as collateral for perpetuals. The panel shows the account mode and
reads the balance from the right place for it. In the older classic mode, USDC in Spot must be
moved to Perps before it can be traded (Portfolio > Transfer in Hyperliquid); the panel tells you
if that is the case.

Hyperliquid's minimum order is $10. A balance near that leaves the engine very few position sizes.

## Start and stop

- **Start trading** - real orders are sent whenever the engine finds a trade it trusts. The top
  bar turns red and reads "LIVE - TRADING". The choice survives restarts and reboots.
- **Stop trading** - nothing more is sent. An open position stays open.
- **Close position and stop** - cancels resting orders and closes the whole position at market.
- **Disconnect** - stops, and removes the key from the server. An open position stays open.

Connecting never starts trading on its own.

## What to expect

The engine trades only when one of its models has shown, on live data, forecasts accurate enough
to beat trading costs. The panel says in plain words why it is not trading. On everything measured
so far (README, "What the models have been trained on") no such edge has been found, so it may run
for a long time without placing an order. That is the engine protecting the balance, not a fault.

When it does trade it sizes positions itself, up to the **Max leverage** you set under Risk. With
the default of 40x a 1.5% move against a full-size position costs about half the account; set a
lower cap if that is not what you want.

On connecting, the engine sets BTC to cross margin at the venue's maximum leverage so the exchange
does not reject a size the engine chose. Actual exposure is whatever the engine sizes, within your cap.

## How this has been tested

The whole journey (connect, balance, Start, close position, disconnect) runs in the test suite
against a mock exchange through the real Hyperliquid SDK and its signing code, and the installer and
control panel run on a clean Ubuntu machine on every push. Checking a key, reading roles and reading
balances have been run against the real Hyperliquid API. **A real order has never been sent to the
real exchange by this code.** The first one will be yours; start with an amount you can lose.
