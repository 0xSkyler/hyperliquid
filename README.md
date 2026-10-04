# hyperliquid-autonomous-trader

An expected-utility trading engine for Hyperliquid perpetuals (BTC first). It installs and runs in
**PAPER** mode: real market data, simulated fills, no keys, no money.

**Read this first.** Nothing in this repository has a demonstrated edge on real markets. The engine
is built so that it *knows* that: it will not open a position until its own forecasts have shown a
statistically significant out-of-sample relationship with realized returns, net of fees. On BTC at
base-tier fees (9 bps taker round trip) the most likely behaviour is that it watches and does
nothing. That is the system working, not failing. Do not fund it on the strength of the synthetic
backtest: that test plants an artificial edge to prove the machinery, nothing more.

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

## Champion / challenger

All models are scored on the same resolved forecasts. A challenger replaces the champion when its
*calibrated* forecast has significantly lower squared error than the champion's on the same
out-of-sample observations (paired test, corrected for overlapping horizons, threshold
`promote_z`). A model that has earned no trust is indistinguishable from "predict zero", so nothing
is ever promoted on noise. Promotions are logged to `data/promotions.jsonl` and shown on the
dashboard. Choose the line-up with `HL_MODELS` (default `ridge,mlp,tree,ridge_disc`; first is the
starting champion).

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

**3. Order-flow models: live data only.** The ridge / neural / tree models that actually drive
trading use order-book and trade-flow features, which do not exist in candle history. They learn
from live data, or from your own recordings via warm start.

```bash
python -m app.research.history --years 10          # download or top up candles (resumable)
python -m app.research.strategy_lab                # every strategy x timeframe x regime
python -m app.research.chart_train                 # walk-forward report, then train and save
python -m app.research.stress                      # engine behaviour in hostile environments
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

## Run

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q && ruff check . && mypy app backtest
python -m app.main                      # paper mode, dashboard on http://127.0.0.1:8787
HL_RECORD_RAW=1 python -m app.main      # also record raw market data to data/raw-YYYYMMDD.jsonl
python -m backtest.run "data/raw-*.jsonl"  # replay through the same engine
```

Modes (`HL_MODE`): `paper` (default), `shadow` (decides, never sends), `testnet`, `live`,
plus `python -m backtest.run` for backtest. See `.env.example`, `docs/DEPLOY.md`, `docs/LIVE.md`.

## What is and is not built

Built and tested: Hyperliquid market-data adapter, paper venue (latency, book-walking taker fills,
queue-aware maker fills, fees, funding, margin rejects, liquidation), feature set, indicator
library, linear / neural / tree models, chart models for four timeframes trained on 10 years of
candles, strategy lab, stress lab, learned-state persistence and warm start, champion-challenger promotion, regime-aware calibration,
utility decision engine, maker/taker selection, safety kernel with reconciliation, journal
(drawdown, fees, markouts), JSONL persistence off the hot path, dashboard, RSS news ingestion with
de-duplication, feature discovery, RL study, replay backtester, CI (including the Docker build).

Written but **not verified** against the real service: `HyperliquidLive` order placement (needs
keys; test on testnet first), `PostgresSink`, and the LLM news call (tested with a stand-in
client only; no API credentials were available).

Not built (from the original brief): economic calendar and surprise scoring, cross-exchange
lead/lag feeds, multi-asset ranking, market-making quoting, whole-strategy generation,
counterfactual and attribution reports, liquidation-cascade detection, RL in the live loop.
