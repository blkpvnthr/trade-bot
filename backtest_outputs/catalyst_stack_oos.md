# Earnings-Catalyst Composite — OOS quintiles (horizon 10d, 20bps slip)

Universe 40 liquid names, last 10 reports each. PIT entry = T+1 after reportedDate. Composite = equal-weight z(transcript mgmt-sentiment, surprise%, news sentiment), z fit on TRAIN (<2025-01-01) only. Metrics below are OOS (entry ≥ train-end).

Events: 250 total / train 98 / **test 152**.

Train-learned feature orientation (sign of corr w/ fwd ret): **sent+, surp+, news+, pcr+, voi−**. Q5 = signal says BUY (long/calls); Q1 = signal says SELL/avoid (short/puts).

| Quintile (low→high score) | n | Mean fwd | Win% | Sharpe | PF | Mean excess vs SPY |
|---|---|---|---|---|---|---|
| Q1 (bottom) | 30 | +1.17% | 50% | +0.140 | 1.48 | +0.27% |
| Q2 | 30 | +0.56% | 53% | +0.075 | 1.20 | -0.81% |
| Q3 | 30 | -1.51% | 43% | -0.190 | 0.58 | -2.20% |
| Q4 | 30 | +0.39% | 57% | +0.048 | 1.13 | -0.54% |
| Q5 (top) | 32 | +1.01% | 47% | +0.078 | 1.26 | +0.31% |

_**Q5−Q1 spread: -0.16%** over 10d. Top-quintile mean +1.01% (Sharpe +0.078, PF 1.26)._

**VERDICT: NO clear signal: composite does not rank winners on this OOS sample. Don't deploy.**
_Caveat: liquid-name pilot, single horizon. PEAD is strongest in small caps; widen universe + test multiple horizons/slippage before trusting. Live confirmation (put/call, vol/OI) not in this test._