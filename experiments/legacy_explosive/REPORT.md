# Legacy Explosive-Alpha Re-Emulation — Backtest Report & Paper-Readiness Recommendation

**Date:** 2026-06-16 · **Mode:** analysis / paper only · **Live:** permanently impossible

This reproduces the old trading stack as a **separate, paper-only** `legacy_explosive_alpha` system
(not merged into RL, not merged into the event-study gate, no TJL changes) and evaluates it under the
current validation discipline. The first deliverable was **faithful reproduction**, not optimization.

## Pipeline (reproduced)
```
daily_features.csv
  → legacy_explosive_alpha_model.py      (Transformer seq classifier + features + explosive target)
  → legacy_explosive_alpha_backtest.py   (walk-forward 252/42/42 + EV/Kelly/convex sizing + risk sim)
  → legacy_explosive_alpha_orders.py     (rebalance_orders.csv / exit_orders.csv, executor schema)
  → legacy_explosive_alpha_executor.py   (PAPER-ONLY, live impossible, sanity/kill/sell-first/audit)
```
Model + backtest are a **byte-faithful split** of `walkforward_explosive_probabilistic_risk_sim.py`
(the original algorithm is preserved verbatim through `score_test_window` and the entire risk sim;
only the I/O entrypoint is new). The Transformer (20-bar per-symbol windows, train-only norm,
chronological val, positive-class weighting), explosive target, EV/Kelly/convex sizing, regime
exposure throttling, capital-preservation states, scale-ins, selective scale-outs, profit-lock,
early-failure / model-failure / edge-decay / time-decay / max-hold exits, cooldowns, QQQ-correlation
cap, portfolio-vol cap, and 5bps slippage are all intact.

## Data reality (critical caveat)
The legacy input `daily_features.csv` spans **only 2024-12-10 → 2026-05-12, 36 symbols** (12 survive
the 252-day training requirement). With train=252 / test=42 / step=42 that yields **2 OOS windows /
84 OOS trading days**. This is far too little to conclude anything statistically, and the Transformer
training is **non-deterministic run-to-run** (MPS), so all deltas below are *directional, not
significant*. Tuning thresholds on 2 windows would be curve-fitting.

## Faithful backtest result (no changes to logic)
| Strategy (OOS Dec'25–Apr'26, 12 symbols) | Sharpe | Total ret | MaxDD | Win | Notes |
|---|---|---|---|---|---|
| **Legacy explosive-alpha (faithful)** | **−1.0 to −1.6** | −3% to −5% | −7.6% to −8.8% | 41% | AUC 0.64–0.70 (real signal) |
| SPY | +0.76 | +3.3% | −8.9% | — | |
| QQQ | +0.77 | +4.1% | −11.7% | — | |
| **Equal-weight of its OWN 12 picks** | **+1.47** | +3.2% | −2.7% | — | the decisive comparison |

**The model classifies (AUC > 0.5) but the trading system loses money OOS**, and — decisively — it
**underperforms naive equal-weight of the very names it traded** (Sharpe 1.47 vs negative). The
selection + timing + exits destroyed value on names that rose. The ADR date-block bootstrap CI
includes zero. The "decent results" remembered from before were almost certainly in-sample / training
-period, or from a different/longer dataset — not present on these genuine OOS windows.

### Per-regime attribution (corroborates the retirement reason)
The user recalled retiring this because it **needed better sideways/chop/downside execution.** The
attribution reproduces exactly that: the bleed is concentrated in `TREND_RISK_ON` pullbacks
(−$7k to −$10k, win ~12–19%) while `VOLATILE_RISK_ON` was clean (5/5). The left tail was uncapped
(the original deliberately removed ATR/trailing stops).

## Improvement experiment (chop / sideways / downside execution)
Built as a **separate opt-in module** (`legacy_explosive_alpha_improved.py`) that preserves the
faithful baseline and ADDS protection (never loosens risk): (1) a conservative **hard downside stop**
(−8.5%, the left-tail cap the original lacked), (2) a **chop give-back lock** (bank sooner / tighter
in non-trending regimes), (3) **faster chop time-decay**, and — per the follow-up request —
(4) **regime-aware take-profit**.

**Deterministic A/B** (CPU + seeded + deterministic algorithms via `model.enable_determinism()`;
baseline reproduces bit-identically run-to-run — verified). Removing training noise lets us read the
exit-logic deltas cleanly:

| Variant (deterministic) | Sharpe | Total ret | MaxDD | TREND win |
|---|---|---|---|---|
| A faithful baseline | −3.43 | −12.2% | −14.0% | 18% |
| B + hard stop + chop give-back lock + chop time-decay | **−2.62** | −9.3% | −12.1% | 24% |
| C B + take-profit (18% trend / 9% chop) | **−2.62 (identical to B)** | −9.3% | −12.1% | 24% |

- The **chop/downside guards help** even deterministically (−3.43→−2.62, lower DD, TREND win
  18%→24%). Right *kind* of fix for the weakness — but still net-negative and still loses to
  equal-weight (1.47).
- **Take-profit is INERT here** — variant C is bit-identical to B. With a fixed model, the 18%/9% TP
  thresholds were never the binding exit (the give-back lock / other exits fired first). An earlier
  non-deterministic (MPS) run had suggested take-profit "hurt"; that was a **training-noise artifact
  from a different model draw**, now corrected. Net: on this data take-profit neither helps nor hurts.
- **Model-draw variance is huge**: the deterministic CPU baseline (−3.43) vs lucky MPS draws (≈ −1 to
  −1.6) span Sharpe −1 to −3.4 on the same data — direct evidence that 2 windows / 84 days is far too
  noisy to tune on. None of these deltas are statistically significant; they are mechanistic, not fitted.

## Long-history research backfill (resolves the 2-window confounder)
The production `features.py` downloads only **520 calendar days** and caps **MAX_SYMBOLS=30** — it is
a *daily production* builder, not a research backfill. That is the real reason the faithful replay had
only 2 OOS windows: **the feature file lacked history, not necessarily the model lacked signal.**

`legacy_explosive_alpha_feature_backfill.py` (research-only, never overwrites production) rebuilds the
**identical production feature schema** (via a frozen copy of `features.py`'s `build_features`) over
**adjusted bars, 2016–2026**, configurable universe, no 30-symbol cap, through the data-quality hard
gate. First research replay (`curated_legacy`, 82 req → 76 clean, 6 dropped by the gate): **158,876
rows · 76 symbols · 2,628 dates → 56 walk-forward windows** (vs 2).

`legacy_explosive_alpha_decompose.py` then separates **signal** (AUC, calibration/Brier, cross-sectional
rank IC, NDCG@5, top-decile forward returns, equal-weight-top-N "signal portfolio") from **execution**
(faithful vs improved vs +take-profit vs fixed-horizon-top-prob vs trend-filtered, each re-simulated
from the *same frozen predictions* → deterministic), benched vs SPY/QQQ/equal-weight with bootstrap CI.

> **Long-history results: _pending the 56-window run_** (training on MPS; execution A/B is deterministic
> because it re-simulates frozen predictions). The verdict below is from the 520-day file only and will
> be revised once the long-history decomposition lands. The honest interim status: the 2-window negative
> is **not** sufficient to conclude the Transformer has no value — only that the production feature file
> is insufficient for validation.

## Paper-readiness recommendation: **DO NOT paper-deploy yet.**
The full pipeline is reproduced and is **paper-safe** (live structurally impossible, sanity filters,
kill switch, sell-before-buy, cash buffer, audit logs — all tested). But the **edge is not present**
on the available OOS data: it loses money and underperforms equal-weight of its own universe. Paper
trading is for forward validation of a *candidate edge*; there isn't one to validate here yet.

To justify forward paper validation, in order:
1. **Regenerate `daily_features` from adjusted bars over a long, multi-regime history** (the current
   2-window / 84-day sample cannot validate anything). The daily feature generator is the missing
   pipeline piece to restore for this.
2. Re-run the faithful backtest on that history; the system must **beat equal-weight of its own picks
   AND SPY/QQQ, with a bootstrap CI on window returns excluding zero, and hold out-of-domain** (the
   sleeve validation gate). Only then mint a passing record and enable paper via the sleeve.
3. Develop/validate the chop-downside improvement **on that longer data** — not on these 2 windows.
   Keep the give-back lock; drop the fixed take-profit unless it proves out (it currently does not).

Until then both legacy sleeves remain **scanner-only / disabled** in the sleeve framework.
