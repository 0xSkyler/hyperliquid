# Training run: 24 hours on live Hyperliquid data

Finished 2026-10-08 06:23 UTC. No account was connected; nothing was traded.
Fees assumed: maker 1.5 bps, taker 4.5 bps per trade.

## BTC
- Learned from 19.9 hours of market data; process restarts: 1.
- Fast forecast (next 5 s): accuracy 0.07 (0 = none, 1 = perfect), trust 0.00, 72,165 forecasts scored.
- Practice quoting: 0 fills, worth 0.00 bps each after 5 s (needs more than the maker fee of 1.5). Real quoting earned: no.
- Quote lessons: 30 situations judged, 0 of them pay.
- Take lessons (what crossing the spread was worth, by forecast strength):

| Forecast (bps) | Times seen | Worth (bps) | Lower bound | Verdict |
|---|---|---|---|---|
| 0-0.1 | 1,762 | -0.07 | -0.11 | avoid: less than the fee |
| 0.1-0.2 | 1,794 | 0.01 | -0.03 | avoid: less than the fee |
| 0.2-0.4 | 4,059 | 0.15 | 0.12 | avoid: less than the fee |
| 0.4-0.8 | 4,580 | 0.44 | 0.40 | avoid: less than the fee |
| 0.8-1.6 | 951 | 0.56 | 0.45 | avoid: less than the fee |
| 1.6-3.2 | 262 | 0.00 | -0.07 | avoid: less than the fee |
| 3.2+ | 6,050 | -0.09 | -0.11 | avoid: less than the fee |

- 60-second models: ridge accuracy 0.215 trust 0.00, mlp accuracy -0.127 trust 0.00, tree accuracy 0.102 trust 0.00, flow_pretrained accuracy 0.124 trust 0.00. Champion: ridge.

## ETH
- Learned from 19.9 hours of market data; process restarts: 1.
- Fast forecast (next 5 s): accuracy 0.02 (0 = none, 1 = perfect), trust 0.00, 72,152 forecasts scored.
- Practice quoting: 87 fills, worth -1.56 bps each after 5 s (needs more than the maker fee of 1.5). Real quoting earned: no.
- Quote lessons: 32 situations judged, 0 of them pay.
- Take lessons (what crossing the spread was worth, by forecast strength):

| Forecast (bps) | Times seen | Worth (bps) | Lower bound | Verdict |
|---|---|---|---|---|
| 0-0.1 | 1,433 | -0.19 | -0.26 | avoid: less than the fee |
| 0.1-0.2 | 1,555 | -0.06 | -0.12 | avoid: less than the fee |
| 0.2-0.4 | 3,154 | 0.01 | -0.03 | avoid: less than the fee |
| 0.4-0.8 | 4,712 | 0.32 | 0.27 | avoid: less than the fee |
| 0.8-1.6 | 1,494 | 0.76 | 0.63 | avoid: less than the fee |
| 1.6-3.2 | 175 | 0.01 | -0.46 | avoid: less than the fee |
| 3.2+ | 6,935 | -0.18 | -0.19 | avoid: less than the fee |

- 60-second models: ridge accuracy 0.176 trust 0.00, mlp accuracy -0.024 trust 0.00, tree accuracy 0.068 trust 0.00, flow_pretrained accuracy 0.083 trust 0.00. Champion: ridge.

## SOL
- Learned from 19.9 hours of market data; process restarts: 1.
- Fast forecast (next 5 s): accuracy -0.01 (0 = none, 1 = perfect), trust 0.00, 72,150 forecasts scored.
- Practice quoting: 0 fills, worth 0.00 bps each after 5 s (needs more than the maker fee of 1.5). Real quoting earned: no.
- Quote lessons: 31 situations judged, 0 of them pay.
- Take lessons (what crossing the spread was worth, by forecast strength):

| Forecast (bps) | Times seen | Worth (bps) | Lower bound | Verdict |
|---|---|---|---|---|
| 0-0.1 | 1,281 | -0.42 | -0.49 | avoid: less than the fee |
| 0.1-0.2 | 1,305 | -0.38 | -0.45 | avoid: less than the fee |
| 0.2-0.4 | 3,079 | -0.21 | -0.26 | avoid: less than the fee |
| 0.4-0.8 | 4,713 | 0.07 | 0.03 | avoid: less than the fee |
| 0.8-1.6 | 1,883 | 0.56 | 0.47 | avoid: less than the fee |
| 1.6-3.2 | 92 | 0.71 | 0.15 | avoid: less than the fee |
| 3.2+ | 7,104 | -0.46 | -0.49 | avoid: less than the fee |

- 60-second models: ridge accuracy 0.156 trust 0.00, mlp accuracy 0.025 trust 0.00, tree accuracy 0.084 trust 0.00, flow_pretrained accuracy 0.092 trust 0.00. Champion: ridge.

## How to read this

The engine trades for real only where a lesson's *lower bound* beats the fee. If every verdict above says
'avoid', the trained engine will correctly place no orders at this fee level, however long it trains.
The learned state is in `models/state/`; the engine loads it at start-up when it is more experienced than its own.
