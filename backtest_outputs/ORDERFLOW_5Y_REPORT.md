# Order-Flow Buy/Exit Engines — 5-Year Backtest (genuine IEX ticks, sampled)

_Generated 2026-11-19 · branch 13K-development · falsification-first per CLAUDE.md_

## Question
Do the order-flow BUY-opportunity + profit-EXIT engines add entry/exit edge over the last
~5 years on the 5-minute timeframe?

## Why this is a SAMPLED real-tick study (not a 5-min OHLCV backtest)
The engines are **order-flow** engines: CVD, aggressive buy/sell classification, quote
imbalance and exhaustion require **tick-level trades/quotes**, not 5-min OHLCV. Deriving order
flow from bars is forbidden (it would be a mirage) and, with order flow absent, the buy score
(65% order-flow-weighted) never reaches an actionable level. Alpaca IEX *does* serve genuine
tick history back ~5 years, but fetching every tick for 5 years is infeasible (billions of
events). So we **sample** trading days across the window and replay the engines on 100%
**genuine IEX trades** for each day. Each intraday day is an independent sample, so sampling
across regimes is a valid OOS design for an intraday engine.

## Dataset
- **Source:** genuine Alpaca **IEX** trades (single-exchange proxy, NOT consolidated SIP;
  L1 only). Trades-only: real trades -> tick-rule classification -> real CVD/aggression/
  exhaustion. L1 quotes omitted for 5-year tractability (165k quotes/symbol-day); spread
  assumed 3 bps (realistic for these liquid mega-caps) for the entry-quality/spread gate.
- **Universe:** AAPL, MSFT, NVDA, AMD, SPY. **Sample:** 3 sessions/quarter, 2020-11 .. 2025-11
  = **275 symbol-days**. **Costs:** costs.py (spread+slippage+commission+impact, ~16 bps RT).
- **Walk-forward:** train < 2023-05-01, test (OOS) >= 2023-05-01.
- **Arms:** A baseline (buy first structured bar, hold to close); B entry_timing (enter only on
  BUY engine confirm; exit at close); C combined (B entry + intelligent profit-exit).

## Result 1 — production threshold (BUY_READY = 75, as shipped)
**The buy engine fires 0 entries across all 275 symbol-days.** On genuine IEX trades-only data
the buy score maxes around ~60; it never reaches 75. The production engine would simply **never
have traded** over 5 years of this data. (No trades -> no edge demonstrated, and no harm.)

## Result 2 — relaxed threshold (BUY_READY = 60, forced to fire) — net of costs

| Metric | Baseline | Entry-timing (B) | Combined (C) |
|---|---|---|---|
| **OOS trades** | 140 | 59 | 59 |
| **OOS net expectancy ($/trade)** | **+0.18** | **-3.64** | **-2.81** |
| OOS profit factor | 1.03 | 0.54 | 0.60 |
| OOS win rate | 49.3% | 37.3% | 37.3% |
| OOS trade Sharpe | +0.01 | -0.18 | -0.15 |
| OOS max drawdown ($) | -408 | -282 | -232 |
| Overall net expectancy | -1.34 | -3.44 | -3.09 |
| Overall profit factor | 0.84 | 0.55 | 0.57 |

- The buy-engine entry (B/C) is **worse than a naive buy-and-hold-the-day baseline** at every
  level — in-sample and out-of-sample — net of costs. Negative expectancy, PF ~0.55.
- The intelligent exit (C vs B) **marginally** reduces the loss (OOS PF 0.60 vs 0.54, smaller
  drawdown) but does not rescue a negative entry.
- The baseline itself is ~breakeven (OOS PF 1.03) — a low bar that B/C still fail.

## Robustness / honesty
- IEX-only proxy (not consolidated). Trades-only (L1 quote-imbalance signal absent, which
  weakens buy-pressure/entry-quality); full-quote over 5 years is infeasible.
- Sampled sessions (275 symbol-days), not every day. OOS split applied.
- Relaxed-60 is NOT the production config; it exists only to obtain a measurable sample.

## Conclusion
**VERDICT: REJECT (entry edge).** Across both readings the order-flow entry-timing engine shows
**no edge** on genuine IEX ticks sampled over 5 years: at the shipped threshold it never trades;
forced to trade, it underperforms a naive baseline after costs, out-of-sample. The intelligent
exit provides only marginal downside relief. This is consistent with the repo's track record
(every intraday price/flow pattern here has failed OOS).

**Implication:** keep both overlays in **SHADOW** (already the default) — do NOT enable entry
or exit enforcement on this evidence. The engines remain legitimately useful as shadow-mode
observability and the exit's downside protection; the entry **alpha is falsified** on this data.
The only validation that could change this is **forward paper (shadow) on the live, quote-aligned
IEX feed**, where the full order-flow signal (incl. quote imbalance) is present.

## Reproduce
```bash
# production threshold (0 entries) :
python backtest_orderflow_5y.py --symbols AAPL,MSFT,NVDA,AMD,SPY --days-per-quarter 3 --score-ready 75
# relaxed threshold (measurable, REJECT) :
python backtest_orderflow_5y.py --symbols AAPL,MSFT,NVDA,AMD,SPY --days-per-quarter 3 --score-ready 60
```
