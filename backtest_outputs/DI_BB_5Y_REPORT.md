# DI / Bollinger Entry-Exit Strategy — 5-Variant Backtest (2020-2025)

_Generated 2026-11-19 · branch 13K-development · falsification-first per CLAUDE.md_

## Rule (completed candle; no ADX/order-flow as mandatory conditions)
- **Entry long:** `DI+ > DI-  AND  close > BB_middle(20)`
- **Exit long:** `DI- > DI+  AND  close < BB_middle(20)`

## Method
- **Data:** actual Alpaca **IEX** historical bars (single-exchange proxy) for DI/Bollinger;
  genuine IEX **ticks** for variant E (order flow, never fabricated from bars).
- **Universe:** AAPL, MSFT, NVDA, AMD, TSLA, SPY, QQQ, AMZN, META, GOOGL.
- **Window:** 2020-11 .. 2025-11. **Walk-forward:** train < 2023-05-01, test (OOS) >=.
- **Timeframes evaluated independently:** 1-min, 5-min, 15-min (no OOS cherry-picking).
- **No look-ahead:** a completed-bar signal fills at the **next bar's open**.
- **Costs:** costs.py buckets (majors/equities/lev-ETF) + 2 bps/side slippage.
- **Variants:** A baseline (buy-and-hold); B DI/BB entry + protective exit (stop -2% /
  +4%-lock giveback 1.5%); C trend entry (close>SMA50 rising) + DI/BB exit; D DI/BB
  entry+exit; E = D + order-flow ranking + profit protection (genuine ticks, sampled).

## Out-of-sample results, net of costs

| TF | Variant | Trades | Net exp. ($/trade) | PF | Win% | Sharpe | Avg hold (min) | Verdict |
|---|---|---|---|---|---|---|---|---|
| — | **A baseline (buy-hold)** | 10 | **+2181** | 40.3 | 90% | 26.4 | — | BASELINE |
| 5m | B | 1959 | +6.03 | 1.19 | 41% | 1.07 | 1109 | INCONCLUSIVE |
| 5m | C | 18826 | −1.54 | 0.69 | 21% | −1.13 | 56 | **REJECT** |
| 5m | D | 15592 | −1.37 | 0.78 | 25% | −0.84 | 79 | **REJECT** |
| 15m | B | 1526 | +8.43 | 1.27 | 44% | 1.43 | 1280 | INCONCLUSIVE |
| 15m | C | 5927 | −0.33 | 0.96 | 28% | −0.17 | 186 | **REJECT** |
| 15m | D | 4796 | +0.20 | 1.02 | 30% | +0.07 | 266 | INCONCLUSIVE |
| 1m | B | 2350 | +5.39 | 1.18 | 40% | 1.00 | 947 | INCONCLUSIVE |
| 1m | C | 87570 | −1.79 | 0.40 | 15% | −2.75 | 11 | **REJECT** |
| 1m | D | 81656 | −1.77 | 0.47 | 19% | −2.49 | 14 | **REJECT** |

**Variant E (DI/BB + order-flow, genuine IEX ticks, 275 symbol-days, OOS):**
D = 334 trades / −0.69 exp / PF 0.79 ; **E = 162 trades / −1.49 exp / PF 0.50** -> order-flow
ranking/protection makes DI/BB **worse**. **REJECT.**

## Regime (OOS) — no consistent edge
- **C, D** are **negative in bull, bear, AND sideways** at 1m and 5m. At 15m, D is ~flat in
  bull (+0.01), negative in bear (−1.32), modestly positive only in sideways (+2.42) — not a
  robust, deployable regime edge.
- **B** is positive mainly in **bull** (+7 to +8) and sideways, weak/negative in bear — i.e.
  it is essentially DI/BB-timed **long-beta** that works when the market rises.

## Interpretation
1. **The DI/BB exit is the problem.** Every variant that uses it (C, D) **churns and bleeds to
   costs** — many trades, short holds, win rates 15-30%, net-negative. The DI− > DI− / below-mid
   exit fires on routine pullbacks and whipsaws out of trends.
2. **No DI/BB timing edge.** The only positive variant (B) rarely exits (protective stop seldom
   trips in a bull market) so it mostly **holds** — it is diluted long-beta that **massively
   underperforms** simply buying and holding (exp +6-8 vs +2181). DI/BB entry timing adds no
   value over holding.
3. **Finer timeframe = worse.** 1-min churns the most and loses the most; 15-min is least-bad.
4. **Order flow does not help** (E worse than D), consistent with the separate 5-year order-flow
   backtest (also REJECT).

## Verdict per variant
- **A** — baseline (buy-and-hold dominates all strategies).
- **B** — **INCONCLUSIVE**: positive but weak, does not beat baseline; it is long-beta, not a
  timing edge.
- **C** — **REJECT** (all timeframes; the DI/BB exit whipsaws).
- **D** — **REJECT** at 1m/5m; **INCONCLUSIVE** (marginal, sideways-only) at 15m — no deployable
  edge vs baseline.
- **E** — **REJECT** (order-flow intelligence does not improve DI/BB).

**Bottom line:** the DI/Bollinger rule is not a deployable edge on this data. Keep the overlay in
**SHADOW** (the shipped default) and do NOT enable DI/BB enforcement. These remain experimental
rules requiring out-of-sample and forward-paper validation; this backtest does not support them.

## Data limitations
IEX-only (single-exchange proxy, not consolidated SIP). Variant E uses a sampled real-tick
subset (ticks required; never fabricated). Baseline A is buy-and-hold per period. No parameter
search was run (DI=14, BB=20 fixed) — avoids overfitting but also was not optimized.

## Reproduce
```bash
python backtest_di_bb.py --symbols AAPL,MSFT,NVDA,AMD,TSLA,SPY,QQQ,AMZN,META,GOOGL --timeframes 5,15,1
python backtest_di_bb_E.py --days-per-quarter 3   # variant E on genuine ticks
```
