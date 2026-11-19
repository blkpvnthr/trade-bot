# Insider Copier — True Out-of-Sample Validation (2026 Q2)

_Generated 2026-11-19 · branch 13K-development · falsification-first per CLAUDE.md_

## Question

The insider trade copier (`run_insider_trade_copier`) mirrors large, recent open-market
insider BUYS. Its premise is a **timing** edge: copy the insider's purchase (via the Form 4
filing) and capture the forward move. The headline study (multi-year pooled, t ~5.8,
+2.8-7.3% vs SPY) was built from SEC structured Form 345 data sets through **2026 Q1**. This
report asks the only question that matters per the mission: **does the edge survive on data
the study never saw, under falsification?**

## Dataset

- **Source:** EDGAR daily index (`master.<YYYYMMDD>.idx`) crawled for all Form 4 filings dated
  **2026-04-01 .. 2026-06-30** — strictly AFTER the 2026 Q1 structured cutoff the study used.
  Ownership XML parsed with the production parser (`sec_form4_feed.parse_form4_xml`). This is a
  genuine forward window: these filings did not exist in the study's training data.
- **Scale:** 50,550 Form 4 filings scanned → 5,183 open-market buys → 1,095 tickers →
  **216 copier-eligible** (>$1M open-market buy within a trailing 2-day window). Comparable to
  the in-sample Q1 quarter (214 eligible).
- **Prices:** Alpaca daily bars (IEX) for every eligible ticker + SPY, refreshed **through
  2026-11-18** so 21d/63d exits resolve. Priced coverage **93-96%**; residual survivorship
  (unpriceable, mostly delisted) 4-7% — reported, not hidden.
- **No-lookahead:** entry delayed by the Form 4 filing lag (2d) at next-day open; a held name
  cannot re-signal. Additionally, because the real filing date is known for every event, trades
  whose entry preceded the actual filing (late filings) are flagged and DROPPED — the
  in-sample study could not do this.

## Metrics (out-of-sample 2026 Q2)

Headline (all trades) and the clean no-lookahead subset:

| Horizon | Trades | Win% | Mean vs SPY (all) | t (all) | Clean n | Clean mean vs SPY | Clean t | Clean bootstrap-95% |
|---|---|---|---|---|---|---|---|---|
| 5d  | 253 | 58.5% | +2.30% | +2.89 | 197 | **+2.66%** | +3.16 | [+1.02%, +4.37%] |
| 10d | 236 | 60.2% | +1.83% | +1.90 | 185 | **+2.71%** | +2.50 | [+0.70%, +5.03%] |
| 21d | 213 | 61.0% | +3.10% | +2.31 | 167 | **+4.82%** | +3.07 | [+1.86%, +7.98%] |
| 63d | 196 | 62.2% | +0.18% | — | 156 | +2.02% | +0.91 | [-2.25%, +6.37%] |

## Robustness / falsification

Three tests beyond the t-stat. The decisive one is the **timing placebo**.

1. **Bootstrap CI (clean subset):** excludes 0 at 5/10/21d → the positive excess is robust to
   resampling, not driven by a few names. (63d CI includes 0 → no edge at 63d.)

2. **Timing placebo — the key test.** For each real signal, draw many random entry dates on the
   SAME name and measure the same-horizon SPY-excess. This isolates **timing** from **selection**:

   | Horizon | Real mean vs SPY | Random-entry null mean | placebo p |
   |---|---|---|---|
   | 5d  | +2.30% | +1.59% | 0.151 |
   | 10d | +1.83% | +2.64% | 0.265 |
   | 21d | +3.10% | +3.95% | 0.323 |
   | 63d | +0.18% | +10.87% | 0.719 |

   **The real signal is statistically indistinguishable from random entry on the same names
   at every horizon.** Entering these stocks at a random time does as well (or better). The
   forward excess is **stock selection** (insiders buy names that outperform SPY), NOT
   insider-buy **timing**. (The null's fat tails give this test low power, so it does not
   *disprove* a small timing edge — but it cannot demonstrate one.)

3. **Dose-response (Spearman IC, cluster $ vs forward excess):** In-sample Q1 showed a
   significant 5d IC (+0.162, p=0.009). Out-of-sample Q2 it is **-0.05 (p=0.45)** — the
   dose-response did **not** replicate. The one timing-like signal in-sample vanished OOS.

4. **Lookahead audit:** 22% of trades (56/253 at 5d) entered before the real filing under the
   transaction-date+2 convention and were dropped. The clean subset's edge is slightly
   *stronger* than the full set → the edge is not a lookahead artifact, but the in-sample
   headline t-stats were modestly inflated by these tainted trades.

## Conclusion

- **Selection signal: holds out-of-sample.** A basket of names with recent >$1M insider-buy
  clusters beats SPY by ~+2.7% to +4.8% over 1-4 weeks, bootstrap-robust, lookahead-clean,
  survivorship-adjusted. This is real and reproducible.
- **Timing edge (the copier's actual premise): NOT demonstrated** in-sample or out-of-sample.
  The timing placebo fails at every horizon; the dose-response did not replicate OOS.

**VERDICT: QUARANTINE (5-21d) / REJECT (63d).**

Per the deployment gate ("survive placebo tests"), the copier is **not cleared for live
trading as a timing alpha**. What survives is a *selection* basket, which is a different
strategy than the timing copier and carries high small-cap variance. Legitimate next step:
run the insider-selection basket as a **paper study only** (observational), with eyes open
that it is selection, not timing, and that a single OOS quarter is not a regime sweep. Do NOT
wire live.

## Reproduce

```bash
python build_oos_insider_universe.py --start 2026-04-01 --end 2026-06-30 --rps 7
python fetch_oos_prices.py --events data/cache/oos_insider_events.json
python backtest_insider_copier.py --source pit \
    --events-file data/cache/oos_insider_events.json --horizons 5,10,21,63 --tag _oos2026q2
python insider_oos_falsify.py \
    --trades backtest_outputs/insider_copier_backtest_oos2026q2.json \
    --events data/cache/oos_insider_events.json \
    --filingdates data/cache/oos_insider_filingdates.json --horizon 5
```
