# Candidate A — ETF Short-Term Mean-Reversion (RSI2 dip, liquid ETFs) — GAUNTLET

8 ETFs, RSI(2)<10.0 + close>SMA200 → next-open; exit RSI>70.0 or 6d. 5bps round-trip. No parameter fitting.

| Set | n | Mean | Median | Win% | Sharpe | PF | MaxDD |
|---|---|---|---|---|---|---|---|
| TRAIN (<2024) | 157 | +0.14% | +0.81% | 64% | +0.046 | 1.14 | -4% |
| **OOS TEST (2024+)** | 193 | +0.61% | +0.80% | 75% | +0.356 | 2.57 | -2% |

## Daily-PnL Monte Carlo (OOS, 5000× bootstrap of 73-trade blocks)
- P(block profitable) = **100%**, median block return = **+55.03%**

## Diversification
- Correlation of daily PnL vs vol-gated INSIDER edge (65 common days): **-0.06**

## VERDICT

- PF≥2.0 OOS: YES (2.57)
- Positive MC median + P(profit)≥60%: YES
- MaxDD acceptable (>-25%): YES (-2%)
- Low insider-correlation (|r|<0.3): YES (-0.06)
- OOS n≥30: YES

**PASS → propose run.py wiring (ETF mean-reversion, shares).**