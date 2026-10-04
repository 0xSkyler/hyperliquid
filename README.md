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
3. **Forecast** - an online regression (recursive least squares with forgetting) predicts the
   forward return over `HL_HORIZON_S`, with parameter uncertainty.
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

## Run

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q && ruff check . && mypy app backtest
python -m app.main                      # paper mode, dashboard on http://127.0.0.1:8787
HL_RECORD_RAW=1 python -m app.main      # also record raw market data to data/raw.jsonl
python -m backtest.run data/raw.jsonl   # replay through the same engine
```

Modes (`HL_MODE`): `paper` (default), `shadow` (decides, never sends), `testnet`, `live`,
plus `python -m backtest.run` for backtest. See `.env.example`, `docs/DEPLOY.md`, `docs/LIVE.md`.

## What is and is not built

Built and tested: Hyperliquid market-data adapter, paper venue (latency, book-walking taker fills,
queue-aware maker fills, fees, funding, margin rejects, liquidation), feature set, indicator
library, online model, regime-aware calibration, utility decision engine, maker/taker selection,
safety kernel with reconciliation, journal (drawdown, fees, markouts), JSONL persistence off the
hot path, dashboard, RSS news ingestion with de-duplication, replay backtester, CI config.

Written but **not verified** here: `HyperliquidLive` order placement (needs keys; test on testnet
first), `PostgresSink`, Docker image (no Docker on the build machine).

Not built (from the original brief): news/macro as model inputs, LLM news analysis, economic
calendar and surprise scoring, cross-exchange lead/lag feeds, multi-asset ranking, market-making
quoting, automated feature/strategy discovery, champion/challenger promotion, tree/neural/RL
models, counterfactual and attribution reports, liquidation-cascade detection. The engine is
structured so these plug in as additional features or venues; each should be added only with
recorded-data evidence that it improves out-of-sample calibration.
