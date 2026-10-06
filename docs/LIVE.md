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

- **Start trading** - after a five-minute warm-up the scalper rests a post-only buy and sell around
  the price whenever a quote can pay for its fees, and keeps adjusting them. The top bar turns red
  and reads "LIVE - TRADING". The choice survives restarts and reboots.
- **Stop trading** - resting quotes are cancelled and nothing more is sent. An open position stays open.
- **Close position and stop** - cancels resting orders and closes the whole position at market.
- **Disconnect** - stops, and removes the key from the server. An open position stays open.

Connecting never starts trading on its own.

## Market

The **Market** box shows the market being traded. **Scan all markets** lists every Hyperliquid
perpetual with its spread, the margin left after maker fees on both legs, volume, and the size
resting at the touch. **Use** switches market (only when flat; trading stops until you press Start
again). A positive margin is necessary for a passive scalper to get paid, not sufficient: wide
spreads usually belong to thin markets that jump.

## What to expect

After Start the scalper does five things, and only the last two can cost money:

1. It **forecasts** the next 5 seconds from the order book. The Scalper box shows its accuracy.
2. It **practises** passive quoting on a simulator with pretend money, and shows the result
   ("Practice edge per fill").
3. It **keeps a record of lessons**: what every quote and take it could have made was worth, by
   situation. The panel's "Lessons learned" lists them with a verdict each.
4. It **rests real quotes** only where the lessons say a hit pays and practice confirms a profit
   after the maker fee, with confidence.
5. It **takes liquidity** only where takes at that forecast strength have been worth more than the
   taker fee, at the lower bound of the record.

On the Hyperliquid data recorded so far, practice quotes lose on every market tested (resting
quotes are picked off), and the forecast, although real, is worth about a tenth of the taker fee.
So expect "Real quoting: off (practising)" and no orders. That is the engine declining to lose your
balance, and the panel says so in plain words. If conditions change, it switches itself on.

Inventory is limited to 2x the balance by default (and never above the **Max leverage** you set
under Risk). With a $10 balance each quote is the exchange minimum of about $10, so one fill is
already 1x.

Hyperliquid gives each account a budget of order actions (10,000 to start, plus one per dollar
traded; after that, one action every 10 seconds). The scalper stays inside an action budget and
drops to the always-allowed rate when the account's budget runs low.

On connecting, the engine sets BTC to cross margin at the venue's maximum leverage so the exchange
does not reject a size the engine chose. Actual exposure is whatever the engine sizes, within your cap.

## How this has been tested

The whole journey (connect, balance, Start, close position, disconnect) runs in the test suite
against a mock exchange through the real Hyperliquid SDK and its signing code, and the installer and
control panel run on a clean Ubuntu machine on every push. Checking a key, reading roles and reading
balances have been run against the real Hyperliquid API. **A real order has never been sent to the
real exchange by this code.** The first one will be yours; start with an amount you can lose.
