# Insider Trade Copier — Backtest

_Generated 2026-06-07T18:10:39.642392Z_

## Parameters
- Source: **pit** (point-in-time SEC Form 345 data sets — survivorship-clean, delisted names included)
- Min open-market buy value: **$1,000,000** (trailing 2-day window)
- Filing lag before entry: **2 day(s)**, entry at next-day OPEN, exit at horizon CLOSE
- Horizons (trading days): [5, 10, 21, 63]
- Copier economics: $1,500/name, max 5 names/cycle

## Coverage
- Eligible tickers in universe: **1806** — priced coverage **92%** (residual survivorship = unpriceable 8%)
- Symbols with both insider + price data: **1719**
- Symbols that produced signals: **1634**
- Total unique copy signals: **3729**

## Forward-return performance by holding horizon

| Horizon (d) | Trades | Win% | Mean | Median | Stdev | t-stat | Mean vs SPY | $/trade |
|---|---|---|---|---|---|---|---|---|
| 5 | 3498 | 57.8% | +3.15% | +1.11% | 32.01% | +5.82 | +2.80% | $+47.27 |
| 10 | 3231 | 57.4% | +4.06% | +1.42% | 45.89% | +5.03 | +3.48% | $+60.95 |
| 21 | 2969 | 53.1% | +4.47% | +0.85% | 61.67% | +3.95 | +3.44% | $+67.02 |
| 63 | 2587 | 47.7% | +9.99% | -1.04% | 203.28% | +2.50 | +7.29% | $+149.92 |

## Read
- **Mean vs SPY** is the edge that matters — raw return that just tracks the market is beta, not insider alpha.
- **t-stat** |·|>2 ≈ statistically distinguishable from zero at this sample size; small samples warrant caution.
- `$/trade` applies the live copier's fixed notional; positive-but-tiny edge × small size ≈ limited contribution
  to the $400–$500/day paper target. Treat as a *conviction modifier* candidate, not a standalone engine.

_No-lookahead: entry is delayed by the Form 4 filing lag and executed at the next-day open; a held name cannot re-signal._