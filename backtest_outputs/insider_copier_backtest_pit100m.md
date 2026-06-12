# Insider Trade Copier — Backtest

_Generated 2026-06-07T19:06:33.708766Z_

## Parameters
- Source: **pit** (point-in-time SEC Form 345 data sets — survivorship-clean, delisted names included)
- Min open-market buy value: **$100,000,000** (trailing 2-day window)
- Filing lag before entry: **2 day(s)**, entry at next-day OPEN, exit at horizon CLOSE
- Horizons (trading days): [5, 10, 21, 63]
- Copier economics: $1,500/name, max 5 names/cycle

## Coverage
- Eligible tickers in universe: **111** — priced coverage **84%** (residual survivorship = unpriceable 16%)
- Symbols with both insider + price data: **1719**
- Symbols that produced signals: **84**
- Total unique copy signals: **124**

## Forward-return performance by holding horizon

| Horizon (d) | Trades | Win% | Mean | Median | Stdev | t-stat | Mean vs SPY | $/trade |
|---|---|---|---|---|---|---|---|---|
| 5 | 121 | 54.5% | +2.73% | +0.77% | 13.07% | +2.30 | +2.90% | $+41.01 |
| 10 | 113 | 51.3% | +0.67% | +1.00% | 15.65% | +0.45 | +0.61% | $+10.00 |
| 21 | 107 | 49.5% | +1.19% | +0.00% | 20.86% | +0.59 | +0.88% | $+17.87 |
| 63 | 101 | 41.6% | -5.65% | -5.03% | 28.12% | -2.02 | -6.91% | $-84.80 |

## Read
- **Mean vs SPY** is the edge that matters — raw return that just tracks the market is beta, not insider alpha.
- **t-stat** |·|>2 ≈ statistically distinguishable from zero at this sample size; small samples warrant caution.
- `$/trade` applies the live copier's fixed notional; positive-but-tiny edge × small size ≈ limited contribution
  to the $400–$500/day paper target. Treat as a *conviction modifier* candidate, not a standalone engine.

_No-lookahead: entry is delayed by the Form 4 filing lag and executed at the next-day open; a held name cannot re-signal._