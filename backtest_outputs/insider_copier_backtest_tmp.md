# Insider Trade Copier — Backtest

_Generated 2026-06-07T19:07:14.502055Z_

## Parameters
- Source: **pit** (point-in-time SEC Form 345 data sets — survivorship-clean, delisted names included)
- Min open-market buy value: **$25,000,000** (trailing 2-day window)
- Filing lag before entry: **2 day(s)**, entry at next-day OPEN, exit at horizon CLOSE
- Horizons (trading days): [5, 21, 63]
- Copier economics: $1,500/name, max 5 names/cycle

## Coverage
- Eligible tickers in universe: **374** — priced coverage **93%** (residual survivorship = unpriceable 7%)
- Symbols with both insider + price data: **1719**
- Symbols that produced signals: **331**
- Total unique copy signals: **486**

## Forward-return performance by holding horizon

| Horizon (d) | Trades | Win% | Mean | Median | Stdev | t-stat | Mean vs SPY | $/trade |
|---|---|---|---|---|---|---|---|---|
| 5 | 481 | 54.7% | +1.96% | +0.78% | 14.57% | +2.95 | +1.65% | $+29.38 |
| 21 | 434 | 46.5% | -0.36% | -1.17% | 21.53% | -0.35 | -1.27% | $-5.45 |
| 63 | 409 | 41.3% | +0.72% | -5.11% | 92.18% | +0.16 | -1.93% | $+10.80 |

## Read
- **Mean vs SPY** is the edge that matters — raw return that just tracks the market is beta, not insider alpha.
- **t-stat** |·|>2 ≈ statistically distinguishable from zero at this sample size; small samples warrant caution.
- `$/trade` applies the live copier's fixed notional; positive-but-tiny edge × small size ≈ limited contribution
  to the $400–$500/day paper target. Treat as a *conviction modifier* candidate, not a standalone engine.

_No-lookahead: entry is delayed by the Form 4 filing lag and executed at the next-day open; a held name cannot re-signal._