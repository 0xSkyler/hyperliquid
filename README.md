# hyperliquid-autonomous-trader

An expected-utility trading engine for Hyperliquid perpetuals (BTC first) with a control panel:
connect your account with an API wallet key, see the balance, press Start.

**Read this first.** Nothing in this repository has a demonstrated edge on real markets. The engine
is built so that it *knows* that: it will not open a position until its own forecasts have shown a
statistically significant out-of-sample relationship with realized returns, net of fees. On BTC at
base-tier fees (9 bps taker round trip) the most likely behaviour after you press Start is that it
watches and places no orders. That is the system protecting the balance, not failing. The control
panel tells you in plain words why it is not trading.

## How it decides

Every second:

1. **Perceive** - order book (20 levels), trades and funding from the WebSocket, held in memory.
2. **Features** - touch/5-level imbalance, microprice, order-flow imbalance (1/5/30 s), trade-flow
   imbalance, vol-normalised returns (5/30/300 s), RSI, z-score, premium, spread.
3. **Forecast** - several models predict the forward return over `HL_HORIZON_S` every tick: an
   online linear regression, a small online neural network, gradient-boosted trees (LightGBM, refit
   periodically, discarded whenever they cannot beat "predict zero" on a holdout), and optionally a
   regression over automatically discovered features. Only the **champion's** forecast is traded.
4. **Calibrate** - each forecast is scored when its horizon elapses, from the price that was
   *executable* (decision time + latency). Per soft regime (trend/range/chaos) the engine keeps the
   lower confidence bound of the realized-vs-forecast slope. That number in [0, 1] multiplies the
   forecast. No evidence, or no relationship: it is 0 and the engine sees no edge.
5. **Decide** - score every target exposure from -maxLeverage to +maxLeverage (including "flat" and
   "hold") by expected CRRA utility of wealth over a fat-tailed return distribution with explicit
   gap/liquidation scenarios, net of fees, spread and book impact. Direction, size, leverage,
   pyramiding, reducing, reversing and doing nothing are all outcomes of that one comparison.
6. **Execute** - maker (post-only) vs taker (IOC marketable limit) chosen by comparing expected
   utility given estimated signal half-life, fill probability and measured maker adverse selection.
7. **Learn** - model, calibration, fill markouts and drawdown update continuously.

There are no indicator rules, fixed stops, fixed risk-per-trade or leverage caps (beyond the
venue's). The knobs that remain are *preferences*, not rules: `HL_RISK_AVERSION` (CRRA gamma;
1 = full Kelly, default 4) and the gap scenario (`HL_JUMP_PROB`, `HL_JUMP_SIZE`). With the defaults
and a strong calibrated edge the engine will go to roughly 35x on BTC. Raise gamma if that is not
what you want.

The **safety kernel** (`app/risk/kernel.py`) is the only absolute limit: stale or crossed book,
unknown or stale account state, unacknowledged orders, or a position the engine cannot explain
all stop new orders. On startup the exchange's position is adopted as truth; a restart never
assumes flat.

## The scalper

By default (`HL_STRATEGY=maker`) the engine trades as a scalper (`app/scalp/`). It has two hands.

**Passive quoting** (`quoter.py`) - rest a post-only buy below the price and a sell above it, and
earn the gap when both fill. Several times a second each quote is decided from:

- **Fair value** - the size-weighted microprice, moved by the fast forecast below.
- **Inventory** - quotes shift against the position so the reducing side fills first; the adding
  side stops at the inventory limit (`HL_SCALP_INVENTORY_X`, default 2x the balance), and inventory
  far past the limit is cut at market.
- **Required edge** - a quote must sit at least *maker fee + adverse selection + flow penalty* from
  fair value. Adverse selection is learned from fills: how far the price moves against a quote in
  the 5 seconds after it is hit.
- **Toxic flow** - one-sided aggressive trading against a quote widens that side.
- **Queue priority** - join the touch; step one tick inside only when the spread is wide.

**Fast forecast** (`alpha.py`) - an online model predicts the next 5 seconds of the mid from the top
of the book (queue imbalance, microprice, the last second of order flow). It pulls the vulnerable
quote before it is hit, and when the forecast alone is larger than the taker fee plus the spread,
the scalper takes liquidity at the touch.

**Practice before real money.** The same quoting logic runs all the time against a simulator on
the live feed, with pretend money. Real passive quotes are switched on only while that practice
shows resting quotes are worth more than their fee, with statistical confidence (lower confidence
bound over at least 20 practice fills). Real fills are then judged the same way, and passive
quoting is rested if they lose.

Order handling requotes at once when a resting quote has become too aggressive and lazily when it
is merely less competitive, inside an action budget, because Hyperliquid gives each account a
limited number of order actions. Bad data, Stop, or an unknown account state cancels every quote.

### What it measures on real Hyperliquid data

`python -m app.research.scalp_lab "data/rec/*.jsonl"` replays recordings through the scalper twice:
with fees set to zero (raw skill) and with real fees (what reaches the account). Results:
`models/scalp_lab.json`. On the recordings so far (seven markets, about an hour each - a small
sample of one market mood):

- **The fast forecast is real.** Out-of-sample correlation with the next 5 seconds is about 0.36 on
  BTC, ETH and SOL, trusted within minutes. On thin small-caps it is weak (0.06 - 0.18).
- **Taking on that forecast has positive skill before fees**: about +0.3 to +0.5 bps per trade after
  paying the spread on BTC, ETH and SOL. The taker fee is 4.5 bps, so with real fees it never takes.
- **Resting quotes are picked off on every market tested**: their fills are worth -0.6 to -4 bps five
  seconds later, before fees. With the engine's latency (about 150 ms plus a book feed that updates
  a few times a second) it is the slow party at the touch. Practice therefore keeps real quoting off.
- **Net of real fees, the scalper places no real orders on any of the seven markets and loses nothing.**

So the skill is there and it is roughly a tenth of the fee. Closing that gap takes a lower fee tier
or a faster connection to the exchange, not a better model. The panel's **Scan all markets** ranks
every perpetual by spread minus fees; `HL_STRATEGY=taker` restores the older directional behaviour.

## Champion / challenger

All models are scored on the same resolved forecasts. A challenger replaces the champion when its
*calibrated* forecast has significantly lower squared error than the champion's on the same
out-of-sample observations (paired test, corrected for overlapping horizons, threshold
`promote_z`). A model that has earned no trust is indistinguishable from "predict zero", so nothing
is ever promoted on noise. Promotions are logged to `data/promotions.jsonl` and shown on the
dashboard. Choose the line-up with `HL_MODELS` (default `ridge,mlp,tree,flow,ridge_disc`; first is
the starting champion).

## What the models have been trained on

Two kinds of learning, on two kinds of data.

**1. Chart models: 10 years of history, four timeframes.** `models/chart_{5m,1h,4h,1d}.txt` are
LightGBM models trained on Bitstamp BTC/USD candles from October 2016. Inputs: scale-free chart
features (multi-horizon returns, RSI, MACD, Bollinger and Donchian position, ADX, ATR, efficiency
ratio, stochastic, relative volume, candle shape, time of day) plus the current position of 36
classic strategy variants across 13 families (MA cross, time-series momentum, MACD, Donchian and
Keltner breakouts, breakout fade, Bollinger / RSI / stochastic / VWAP reversion, range trade, trend
pullback, ADX trend). Live, each model scores its latest closed Hyperliquid candle and the result is
one input feature per timeframe (`chart_5m` ... `chart_1d`).

Their walk-forward record (each year scored by a model trained only on earlier years; full detail
per year and per market regime in `models/chart_<tf>.json`):

| Timeframe | Out-of-sample correlation | Strongest 10% of forecasts, after a 9 bps round trip | Years net-positive |
|---|---|---|---|
| 5m | 0.065 early, ~0 since 2024 | -5.8 bps | 0 / 10 |
| 1h | 0.034, positive every year | -2.1 bps | 3 / 10 |
| 4h | 0.013, not significant | -1.1 bps | 4 / 9 |
| 1d | 0.019, not significant | +17.5 bps on 2,787 bars: too few to trust | 3 / 6 |

Read that as: charts carry a small, real signal at the 1-hour scale and essentially none at 5
minutes today, and nowhere is it reliably larger than trading costs.

**2. Strategy lab.** `models/strategy_lab.json` holds every variant x timeframe x market regime
(bull / bear / range x high / low volatility), net of taker fees. In sample, trend following at 4h
and 1d reaches a Sharpe near 1.0. Picked honestly (each year, choose the best variant on earlier
years, then trade it): Sharpe -0.83 at 5m, 0.07 at 1h, 0.27 at 4h, 0.18 at 1d, against 0.7 for
simply holding BTC. No classic strategy beat buy-and-hold out of sample.

**3. Swing lab: the 1-hour signal as a slower, maker-order lane.** `models/swing_lab.json`.
Trading the 1-hour signal directly (long or short while it is strong, flat otherwise) loses money
walk-forward even with maker orders: roughly -55% a year per unit of notional, profitable in 2 of 9
years. Maker orders fill about 88% of the time, but they fill when price is moving against them,
and the entries that go unfilled are the ones that would have won. A variant that holds each
position until the signal flips shows +64% a year, but it is long BTC 77% of the time and tracks
buy-and-hold in every year since 2022; that is market exposure, not the signal. For that reason
no swing lane is wired into live trading.

**4. Trade-flow model: 90 days of tick trades.** `models/flow_60s.txt` is trained on Binance
BTCUSDT futures trades, replayed second by second through the same market-state code the live
engine uses, on the seven features that mean the same thing on any venue (trade-flow imbalance,
volatility-normalised returns, RSI, z-score). It joins the live arena as the `flow_pretrained`
challenger with zero trust and must earn promotion on Hyperliquid. Its week-by-week record is in
`models/flow_60s.json`. Tick-trade data has no order-book sizes, so the book-imbalance features
are still learned live only.

**5. Order-flow models: live data only.** The ridge / neural / tree models learn online from the
full feature set, including the order-book features, from live data or from your own recordings
via warm start.

```bash
python -m app.research.history --years 10          # download or top up candles (resumable)
python -m app.research.strategy_lab                # every strategy x timeframe x regime
python -m app.research.chart_train                 # walk-forward report, then train and save
python -m app.research.swing_lab                   # 1-hour signal as a maker-order lane
python -m app.research.ticks --days 90             # download and featurise tick trades (~2 GB)
python -m app.research.flow_train                  # train the trade-flow model, walk-forward
python -m app.research.stress                      # engine behaviour in hostile environments
python scripts/record.py --coins BTC,ETH,ENA --minutes 60   # record several markets at once
python -m app.research.scalp_lab "data/rec/*.jsonl"  # the scalper's skill per fill, gross and net
python -m app.research.warmstart "data/raw-*.jsonl"  # teach the order-flow models from recordings
```

**Learned state persists.** Everything the order-flow models have learned (weights, calibration,
tree training buffer, champion, promotion history) is saved to `data/state/engine-<coin>.pkl` every
five minutes and on shutdown, and restored at startup. A state file saved with different features,
models or horizon is refused, not half-loaded. Do not warm-start twice from the same recording:
that counts the same evidence twice.

## Stress lab

`python -m app.research.stress` runs the real engine through eleven synthetic environments, each
with a genuine planted edge so the engine is leveraged when the shock arrives
(`models/stress_report.json`, default risk settings):

| Environment | Worst drawdown | Liquidated? |
|---|---|---|
| calm, trending, choppy, volatility spike, thin liquidity | 5 - 7% | no |
| edge disappears / edge reverses | 7 - 8% | no |
| corrupt (crossed) book for 30 s | 6% (kernel blocked trading) | no |
| flash crash, -8% in 20 s | 25% | no |
| feed outage, 2 min blind while price moves 1% | 25% | no |
| gap down, -3% between two ticks | 41% | no |

These are simulations with simplified liquidity. The gap and outage losses are the direct price of
running ~30x; `HL_RISK_AVERSION` and `HL_JUMP_SIZE` are the controls.

## Research tools (offline, on recorded data)

Record first: `HL_RECORD_RAW=1` writes one `data/raw-YYYYMMDD.jsonl` per UTC day. These tools need
days to weeks of it; on less they say so and produce nothing.

```bash
python -m app.research.discovery "data/raw-*.jsonl"   # feature discovery
python -m app.research.rl "data/raw-*.jsonl"          # reinforcement-learning study
```

**Feature discovery** generates about 200 candidate features (interactions, signed squares,
smoothed and de-trended versions) and keeps only those that explain what the linear model
*cannot*, walk-forward, with a Bonferroni-corrected threshold and a same-sign-in-every-fold
requirement. Survivors go to `data/discovered_features.json`. They are not deployed: on the next
start the engine adds a `ridge_disc` challenger that uses them and has to win promotion live.
This discovers features, not whole strategies; the "strategy" is always the utility engine.

**RL study** trains a tabular Q-learning agent (short / flat / long) on the first 70% of the
recording and reports its greedy performance on the last 30%. It is not connected to trading,
and its cost model (flat fee, no queue, latency or impact) is too simple to trust beyond
"worth a closer look".

## LLM news analysis (optional, paid)

Off by default. With `HL_LLM_NEWS=1` and Anthropic credentials (`ANTHROPIC_API_KEY`), each new
headline cluster is sent once to Claude (`HL_LLM_MODEL`, default `claude-opus-5-5`, at most
`HL_LLM_MAX_PER_POLL` calls per minute) and scored for BTC relevance, direction, magnitude,
confidence and rumour status. The answer is schema-constrained and range-checked, then collapsed
into one decayed number, the `news_llm` feature. It has no direct authority: the models learn
whether it predicts anything, and calibration decides whether that is believed. Headlines are
passed as quoted data and cannot instruct the system. If the primary model declines a headline,
the API's server-side fallback retries it on another model. News arrives minutes late and the
models forecast one minute ahead, so do not expect this feature to matter at the default horizon.

## Control panel

`http://127.0.0.1:8787` on the machine it runs on. Unlock it with the token in
`data/control_token`, paste a Hyperliquid API wallet key, press **Connect and fetch balance**, then
**Start trading**. Live is the only trading mode; nothing is sent until both steps are done. Also
there: Stop, Close position and stop, Disconnect, and a leverage cap. See `docs/LIVE.md`.

## Run

On a server, use the installer in `docs/DEPLOY.md`. For development:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q && ruff check . && mypy app backtest
python -m app.main                      # control panel on http://127.0.0.1:8787, connected to nothing
HL_MODE=paper python -m app.main        # developer switch: trade the built-in simulator instead
HL_RECORD_RAW=1 python -m app.main      # also record raw market data to data/raw-YYYYMMDD.jsonl
python -m backtest.run "data/raw-*.jsonl"  # replay through the same engine
```

## What is and is not built

Built and tested: Hyperliquid market-data adapter, paper venue (latency, book-walking taker fills,
queue-aware maker fills, fees, funding, margin rejects, liquidation), feature set, indicator
library, linear / neural / tree models, chart models for four timeframes trained on 10 years of
candles, strategy lab, stress lab, learned-state persistence and warm start, champion-challenger promotion, regime-aware calibration,
utility decision engine, the scalper (two-sided quoting, inventory and flow control, market scanner), safety kernel
with reconciliation, control panel, journal
(drawdown, fees, markouts), JSONL persistence off the hot path, dashboard, RSS news ingestion with
de-duplication, feature discovery, RL study, replay backtester, CI (including the Docker build).

Tested end to end against a mock exchange through the real Hyperliquid SDK (`tests/test_e2e_live.py`):
connect, balance in a unified account, Start, an order to close a position, fill reconciliation,
disconnect; and the scalper resting two post-only quotes, handling a fill, and cancelling on Stop
(`tests/test_e2e_scalper.py`). Key checking and balance reading have also been run against the real Hyperliquid API.

**Not verified** against the real service: an actual order on a funded account (none has ever been
sent), `PostgresSink`, and the LLM news call (tested with a stand-in client only).

Not built (from the original brief): economic calendar and surprise scoring, cross-exchange
lead/lag feeds, multi-asset ranking, market-making quoting, whole-strategy generation,
counterfactual and attribution reports, liquidation-cascade detection, RL in the live loop.
