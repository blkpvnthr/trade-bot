# Insider Trade Copier — Backtest

_Generated 2026-11-19T18:42:52.242374Z_

## Parameters
- Source: **pit** (point-in-time SEC Form 345 data sets — survivorship-clean, delisted names included)
- Min open-market buy value: **$1,000,000** (trailing 2-day window)
- Filing lag before entry: **2 day(s)**, entry at next-day OPEN, exit at horizon CLOSE
- Horizons (trading days): [5, 10, 21, 63]
- Copier economics: $1,500/name, max 5 names/cycle

## Coverage
- Eligible tickers in universe: **216** — priced coverage **93%** (residual survivorship = unpriceable 7%)
- Symbols with both insider + price data: **409**
- Symbols that produced signals: **199**
- Total unique copy signals: **264**

## Forward-return performance by holding horizon

| Horizon (d) | Trades | Win% | Mean | Median | Stdev | t-stat | Mean vs SPY | $/trade |
|---|---|---|---|---|---|---|---|---|
| 5 | 253 | 58.5% | +3.09% | +1.21% | 12.57% | +3.92 | +2.30% | $+46.42 |
| 10 | 236 | 60.2% | +3.44% | +2.64% | 14.64% | +3.61 | +1.83% | $+51.57 |
| 21 | 213 | 61.0% | +5.03% | +3.07% | 19.16% | +3.83 | +3.10% | $+75.51 |
| 63 | 196 | 62.2% | +5.25% | +2.51% | 28.07% | +2.62 | +0.18% | $+78.69 |

## Read
- **Mean vs SPY** is the edge that matters — raw return that just tracks the market is beta, not insider alpha.
- **t-stat** |·|>2 ≈ statistically distinguishable from zero at this sample size; small samples warrant caution.
- `$/trade` applies the live copier's fixed notional; positive-but-tiny edge × small size ≈ limited contribution
  to the $400–$500/day paper target. Treat as a *conviction modifier* candidate, not a standalone engine.

_No-lookahead: entry is delayed by the Form 4 filing lag and executed at the next-day open; a held name cannot re-signal._