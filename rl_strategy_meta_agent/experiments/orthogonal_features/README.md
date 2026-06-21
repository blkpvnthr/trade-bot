# Orthogonal feature sources (research branch)

## Why this branch exists

Three independent symbolic-search backbones now agree:

| generator | result |
|---|---|
| AlphaEvolve (evolutionary) | accepted alphas are volatility/giveback-dominated; no ensemble synergy |
| gplearn (symbolic regression) | VOLATILITY_REDISCOVERY: 0 orthogonal alphas, no breadth/combined-IC lift |
| EO 1.3.1 (genetic programming) | finds diverse stable alphas, but synergy ties IC-ranking -- one factor family |

The constraint is no longer **"can we search formulas better?"** It is **"do we have
enough independent input information to discover orthogonal alphas?"** "More optimizer =
more alpha" is now falsified twice. The next edge has to come from new information.

**So: no new evolutionary engines until the feature set is expanded.** AlphaEvolve,
gplearn, and EO stay as the candidate generators; this branch finds the *inputs* worth
giving them.

## The gate (`feature_gate.py`)

Every new feature source must clear one falsifiable test before any generator mines it:

1. **Standalone PIT IC** vs forward return (`fut_ret_1`), with block wild-bootstrap
   p-value + Benjamini-Hochberg FDR (reuses `multiple_testing`).
2. **Orthogonality** -- correlation to the volatility cluster (`atr_pct`, `rel_volume`,
   `volume_acceleration`) and rank-contribution/residual vs the existing accepted-alpha
   space (reuses `alpha_registry`).
3. **Incremental ensemble value** -- effective-breadth lift and **combined OOS IC lift**
   when added to the accepted set, net of turnover (reuses `alpha_synergy`).

Verdict per feature: `ORTHOGONAL_INFORMATIVE` (independent of the vol cluster AND lifts
combined OOS IC above a noise floor) / `REDUNDANT_VOLATILITY` / `WEAK` / `UNINFORMATIVE`.
Breadth alone is **not** enough -- the user's bar is orthogonal *and* a real OOS IC lift.

```bash
python experiments/orthogonal_features/run_feature_study.py   # pilot scan on the cached panel
```

## Pilot scan result (price-derived features, first scan)

| feature | OOS IC | vol corr | rank contrib | combined-OOS-IC lift | verdict |
|---|---|---|---|---|---|
| `idio_reversal` (idiosyncratic short-term reversal) | +0.031 | 0.03 | 0.82 | **+0.0134** | **ORTHOGONAL_INFORMATIVE** |
| `rel_strength_5` | -0.002 | 0.04 | 0.84 | -0.002 | WEAK |
| `downside_ratio` | -0.011 | 0.02 | 0.81 | +0.001 | WEAK |
| `ret_skew_20` | -0.018 | 0.02 | 0.90 | -0.000 | UNINFORMATIVE |
| `parkinson_vol` (volatility control) | -0.006 | 0.91 | 0.40 | -0.017 | REDUNDANT_VOLATILITY |

The volatility control is correctly flagged redundant; noise is rejected. **One**
price-derived feature clears: idiosyncratic short-term reversal -- it is orthogonal to
the volatility cluster and roughly doubles the accepted set's combined OOS IC
(0.012 -> 0.025) at moderate turnover (0.66). It is the classic short-term reversal
anomaly; before integration it still needs a standalone `event_study` (bootstrap +
placebo) and a **net-of-cost breakeven-bps** check (short-horizon reversal is the
textbook gross-positive / net-fragile signal). Everything else price-derived is
redundant or null -- confirming the thesis that durable *new* directional breadth must
come from data outside price/volume.

## Candidate sources (each must clear `feature_gate` before any generator touches it)

| source | data status | priority | note |
|---|---|---|---|
| relative volume / Stocks-in-Play | already in panel (`rel_volume`, `volume_acceleration`) | low | part of the volatility cluster |
| ORB opening range | already in panel (`or_high_dist`, `or_low_dist`) | low | already searched |
| market / sector regime | partly in panel (`spy_trend`, `qqq_trend`, `mkt_regime`) | med | **constant per cross-section** -- needs regime-conditional eval, not the cross-sectional gate |
| intraday microstructure | partly in panel (wicks, body, `vwap_dist`); book = Databento | low | book-skew already net-negative after spread |
| **news / earnings / catalyst** | TESTED -- see below | done | event-timing clears; sentiment fails |
| options flow / put-call / IV rank | AV chains, `book_skew_backtest.py` | low-med | options-positioning already falsified standalone |
| spread / slippage / fill-quality | executor journals have slippage | med | the project's known real edge, but a **sizing/timing** axis, not directional alpha |

## News / catalyst / FinBERT scan (the first EXTERNAL source)

`run_news_feature_study.py` aligns leak-free news features to the panel
(`news_features.py`: Alpaca news + FinBERT via `data_sources/`, `av_catalyst.earnings`
for PIT report dates; strictly publish_time < bar t, rolling window + recency decay) and
runs them through the same gate. Real data: **18,043 articles across all 24 symbols**,
2025-10 to 2026-06.

```bash
python experiments/orthogonal_features/run_news_feature_study.py [--backfill]
```

| feature | OOS IC | FDR | vol corr | rank contrib | combined-OOS-IC lift | turnover | verdict |
|---|---|---|---|---|---|---|---|
| `earnings_within_7d` | +0.009 | yes | 0.04 | 0.99 | **+0.0042** | 0.08 | **ORTHOGONAL_INFORMATIVE** |
| `earnings_within_14d` | +0.005 | yes | 0.02 | 1.00 | +0.0018 | 0.08 | WEAK |
| `event_recency_decay` | +0.004 | yes | 0.18 | 0.81 | +0.0015 | 0.16 | WEAK |
| `catalyst_score` / `news_volume_count` / `abnormal_news_volume` | ~0 | yes | <0.3 | >0.76 | ~0 | low | WEAK |
| `news_sentiment_mean/max/min`, `finbert_pos/neg` | ≤0 | mostly no | <0.25 | high | **negative** | low | UNINFORMATIVE |

**Verdict: external information DOES add orthogonal edge -- but it is event *timing*, not
sentiment.** `earnings_within_7d` is FDR-significant, uncorrelated with the volatility
cluster (0.04), fully independent of the accepted set (rank-contrib 0.99), lifts the
ensemble's combined OOS IC (0.012 -> 0.016) at near-zero turnover (0.08). Every FinBERT
**sentiment** feature is uninformative / negative OOS -- the same failure as the
standalone catalyst falsification ([[project_catalyst_stack]]). So unlike price-only
symbolic search (which only rediscovered volatility), an external *event-structure*
feature clears the bar; news *sentiment* does not.

Caveats: the effect is small (+0.0042 lift); `analyst_revision_flag` was excluded as
not PIT-backtestable (AV EARNINGS_ESTIMATES is snapshot-only); earnings dates are taken
from actual `reportedDate` as a proxy for the scheduled date (PIT-safe for "within Nd"
since the date is announced weeks ahead). `earnings_within_7d` is now the validated lead
-- next it needs a standalone `event_study` (bootstrap + placebo) + net-of-cost check
before any generator is allowed to mine the earnings-conditioned panel.

## Workflow for a new source

```
new source -> align to panel index (leak-free) -> feature_gate.study()
   -> ORTHOGONAL_INFORMATIVE ? -> event_study (bootstrap+placebo) + net-of-cost
        -> only then let AlphaEvolve / gplearn / EO mine over the enriched panel
```

Research only: no broker / order / execution / live code (enforced by
`tests/test_feature_gate.py`).
