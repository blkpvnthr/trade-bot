# Insider Trade Copier — Backtest

_Generated 2026-06-07T17:51:50.544581Z_

## Parameters
- Min open-market buy value: **$1,000,000** (trailing 2-day window)
- Filing lag before entry: **0 day(s)**, entry at next-day OPEN, exit at horizon CLOSE
- Horizons (trading days): [21, 63]
- Copier economics: $1,500/name, max 5 names/cycle

## Coverage
- Symbols with both insider + price data: **20**
- Symbols that produced signals: **20** (AAPL, AMD, AMZN, AVGO, COST, CSCO, GOOGL, HD, JNJ, JPM, KO, MA, META, MSFT, NVDA, PEP, PG, UNH, V, WMT)
- Total unique copy signals: **217**

## Forward-return performance by holding horizon

| Horizon (d) | Trades | Win% | Mean | Median | Stdev | t-stat | Mean vs SPY | $/trade |
|---|---|---|---|---|---|---|---|---|
| 21 | 205 | 55.1% | +1.40% | +0.61% | 9.06% | +2.21 | -0.11% | $+20.97 |
| 63 | 141 | 66.7% | +2.42% | +2.94% | 18.20% | +1.58 | -1.68% | $+36.26 |

## Read
- **Mean vs SPY** is the edge that matters — raw return that just tracks the market is beta, not insider alpha.
- **t-stat** |·|>2 ≈ statistically distinguishable from zero at this sample size; small samples warrant caution.
- `$/trade` applies the live copier's fixed notional; positive-but-tiny edge × small size ≈ limited contribution
  to the $400–$500/day paper target. Treat as a *conviction modifier* candidate, not a standalone engine.

_No-lookahead: entry is delayed by the Form 4 filing lag and executed at the next-day open; a held name cannot re-signal._