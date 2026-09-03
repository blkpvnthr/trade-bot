# Insider Trade Copier — Backtest

_Generated 2026-06-07T17:59:57.054267Z_

## Parameters
- Source: **av** (AV INSIDER_TRANSACTIONS — mega-cap watchlist)
- Min open-market buy value: **$1,000,000** (trailing 2-day window)
- Filing lag before entry: **2 day(s)**, entry at next-day OPEN, exit at horizon CLOSE
- Horizons (trading days): [5, 10, 21, 63]
- Copier economics: $1,500/name, max 5 names/cycle

## Coverage
- Symbols with both insider + price data: **18**
- Symbols that produced signals: **16** (AUPH, BOC, DNUT, DOMO, ENR, ERIE, HOOD, IFF, L, LODE, NCLH, NSP, OTLK, PODD, RDN, SMMT)
- Total unique copy signals: **104**

## Forward-return performance by holding horizon

| Horizon (d) | Trades | Win% | Mean | Median | Stdev | t-stat | Mean vs SPY | $/trade |
|---|---|---|---|---|---|---|---|---|
| 5 | 100 | 54.0% | +3.72% | +0.48% | 12.72% | +2.92 | +3.11% | $+55.74 |
| 10 | 96 | 52.1% | +3.86% | +0.80% | 13.61% | +2.78 | +3.01% | $+57.94 |
| 21 | 88 | 58.0% | +4.70% | +1.79% | 17.01% | +2.59 | +2.84% | $+70.52 |
| 63 | 68 | 55.9% | +7.54% | +2.58% | 32.92% | +1.89 | +2.49% | $+113.07 |

## Read
- **Mean vs SPY** is the edge that matters — raw return that just tracks the market is beta, not insider alpha.
- **t-stat** |·|>2 ≈ statistically distinguishable from zero at this sample size; small samples warrant caution.
- `$/trade` applies the live copier's fixed notional; positive-but-tiny edge × small size ≈ limited contribution
  to the $400–$500/day paper target. Treat as a *conviction modifier* candidate, not a standalone engine.

_No-lookahead: entry is delayed by the Form 4 filing lag and executed at the next-day open; a held name cannot re-signal._