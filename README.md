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

There are two separate kinds of learning, on two kinds of data:

- **Chart model (10 years of history).** `models/chart_model.txt` is a LightGBM model trained on
  5-minute BTC/USD candles from Bitstamp, October 2016 onward, using scale-free chart features
  (multi-horizon returns, RSI, MACD, Bollinger and Donchian position, ADX, ATR, efficiency ratio,
  stochastic, relative volume, candle shape, time of day). Live, it scores each closed 5-minute
  Hyperliquid candle and the result is one input feature, `chart_ctx`. Its year-by-year
  out-of-sample record is in `models/chart_model.json`; read it before trusting it.
- **Order-flow models (live data only).** The ridge / neural / tree models that actually drive
  trading use order-book and trade-flow features, which do not exist in candle history. They learn
  from live data, or from your own recordings via warm start.

```bash
python -m app.research.history --years 10          # download or top up candles (resumable)
python -m app.research.chart_train                 # walk-forward report, then train and save
python -m app.research.warmstart "data/raw-*.jsonl"  # teach the order-flow models from recordings
```

**Learned state persists.** Everything the order-flow models have learned (weights, calibration,
tree training buffer, champion, promotion history) is saved to `data/state/engine-<coin>.pkl` every
five minutes and on shutdown, and restored at startup. A state file saved with different features,
models or horizon is refused, not half-loaded. Do not warm-start twice from the same recording:
that counts the same evidence twice.

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
library, linear / neural / tree models, chart model trained on 10 years of candles, learned-state
persistence and warm start, champion-challenger promotion, regime-aware calibration,
utility decision engine, maker/taker selection, safety kernel with reconciliation, journal
(drawdown, fees, markouts), JSONL persistence off the hot path, dashboard, RSS news ingestion with
de-duplication, feature discovery, RL study, replay backtester, CI (including the Docker build).

Written but **not verified** against the real service: `HyperliquidLive` order placement (needs
keys; test on testnet first), `PostgresSink`, and the LLM news call (tested with a stand-in
client only; no API credentials were available).

Not built (from the original brief): economic calendar and surprise scoring, cross-exchange
lead/lag feeds, multi-asset ranking, market-making quoting, whole-strategy generation,
counterfactual and attribution reports, liquidation-cascade detection, RL in the live loop.
