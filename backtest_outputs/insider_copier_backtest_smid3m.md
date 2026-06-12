# Insider Trade Copier — Backtest

_Generated 2026-06-07T18:00:31.295973Z_

## Parameters
- Source: **av** (AV INSIDER_TRANSACTIONS — mega-cap watchlist)
- Min open-market buy value: **$3,000,000** (trailing 2-day window)
- Filing lag before entry: **2 day(s)**, entry at next-day OPEN, exit at horizon CLOSE
- Horizons (trading days): [5, 21]
- Copier economics: $1,500/name, max 5 names/cycle

## Coverage
- Symbols with both insider + price data: **18**
- Symbols that produced signals: **15** (AUPH, BOC, DNUT, DOMO, ERIE, HOOD, IFF, L, LODE, NCLH, NSP, OTLK, PODD, RDN, SMMT)
- Total unique copy signals: **57**

## Forward-return performance by holding horizon

| Horizon (d) | Trades | Win% | Mean | Median | Stdev | t-stat | Mean vs SPY | $/trade |
|---|---|---|---|---|---|---|---|---|
| 5 | 57 | 50.9% | +4.46% | +0.17% | 14.01% | +2.40 | +4.23% | $+66.95 |
| 21 | 52 | 69.2% | +4.72% | +2.85% | 14.47% | +2.35 | +2.69% | $+70.84 |

## Read
- **Mean vs SPY** is the edge that matters — raw return that just tracks the market is beta, not insider alpha.
- **t-stat** |·|>2 ≈ statistically distinguishable from zero at this sample size; small samples warrant caution.
- `$/trade` applies the live copier's fixed notional; positive-but-tiny edge × small size ≈ limited contribution
  to the $400–$500/day paper target. Treat as a *conviction modifier* candidate, not a standalone engine.

_No-lookahead: entry is delayed by the Form 4 filing lag and executed at the next-day open; a held name cannot re-signal._