# Insider Trade Copier — Backtest

_Generated 2026-06-07T17:56:36.279008Z_

## Parameters
- Source: **sec** (SEC Form 4 cache — small/mid-cap aware)
- Min open-market buy value: **$1,000,000** (trailing 2-day window)
- Filing lag before entry: **2 day(s)**, entry at next-day OPEN, exit at horizon CLOSE
- Horizons (trading days): [5, 10, 21]
- Copier economics: $1,500/name, max 5 names/cycle

## Coverage
- Symbols with both insider + price data: **35**
- Symbols that produced signals: **2** (COE, DOMO)
- Total unique copy signals: **2**

## Forward-return performance by holding horizon

| Horizon (d) | Trades | Win% | Mean | Median | Stdev | t-stat | Mean vs SPY | $/trade |
|---|---|---|---|---|---|---|---|---|
| 5 | 2 | 50.0% | +17.82% | +17.82% | 24.68% | +1.02 | +40.24% | $+267.38 |
| 10 | 1 | 100.0% | +25.36% | +25.36% | 0.00% | +0.00 | +23.04% | $+380.36 |
| 21 | 1 | 100.0% | +22.14% | +22.14% | 0.00% | +0.00 | +14.58% | $+332.14 |

## Read
- **Mean vs SPY** is the edge that matters — raw return that just tracks the market is beta, not insider alpha.
- **t-stat** |·|>2 ≈ statistically distinguishable from zero at this sample size; small samples warrant caution.
- `$/trade` applies the live copier's fixed notional; positive-but-tiny edge × small size ≈ limited contribution
  to the $400–$500/day paper target. Treat as a *conviction modifier* candidate, not a standalone engine.

_No-lookahead: entry is delayed by the Form 4 filing lag and executed at the next-day open; a held name cannot re-signal._