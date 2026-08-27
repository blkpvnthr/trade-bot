# Legacy Transformer Signal — Diagnosis (control battery)

**Question:** Does the legacy explosive Transformer's probability carry predictive information
*incremental to simple baselines* (momentum, trend, equal-weight, random) once survivorship/universe
effects are removed?

**Answer: NO — no incremental *tradable* information.** A small within-day rank IC exists, but it is
dominated by a **static survivorship effect** and the genuine dynamic component is marginal and
produces zero spread over trivial baselines.

Diagnosis only — no deployment, no paper, no run.py, no Kronos, no TJL. Pooled over 5 training seeds;
2,352 OOS days, 74 symbols; all metrics block-bootstrapped (CI + one-sided p that mean ≤ 0).

## Information coefficients (within-day, cross-sectional → universe/level-neutral)
| Metric | mean | 95% CI | p(≤0) |
|---|---|---|---|
| Spearman IC (prob) | +0.041 | [+0.023, +0.060] | 0.000 |
| Pearson IC (prob) | +0.025 | [+0.002, +0.047] | 0.014 |
| momentum IC (ret_20d) | +0.016 | [−0.007, +0.037] | 0.085 (ns) |
| trend IC (close_vs_sma50) | +0.023 | [+0.000, +0.046] | 0.025 |
| **partial IC (prob \| momentum+trend)** | +0.029 | [+0.013, +0.044] | 0.001 |
| sector-neutral IC | +0.013 | [+0.001, +0.025] | 0.016 |
| mcap-neutral IC | +0.036 | [+0.016, +0.056] | 0.000 |

## Placebos (must collapse to ~0)
| Placebo | mean IC | result |
|---|---|---|
| prediction-shuffle (permute prob within day) | −0.005 | **collapses ✓** |
| **date-shuffle (prob of day t′ vs returns of day t)** | **+0.029** | **does NOT collapse ✗** |

The date-shuffle *not* collapsing is the tell: the IC survives breaking time-alignment → it is a
**persistent per-symbol structure**, not dynamic timing.

## Static vs dynamic decomposition (decisive)
| Component | value |
|---|---|
| Raw within-day rank IC | +0.041 (CI [+0.023,+0.060]) |
| **Dynamic** (symbol-time-demeaned) IC | **+0.015, CI [−0.003, +0.033], p=0.049** (not robust) |
| **Static** cross-symbol IC (mean-prob vs mean-return, n=74) | **ρ=+0.27, p=0.022** |
| Static share of raw IC | **63%** |

→ ~⅔ of the "signal" is the model persistently assigning higher probability to names that happened
to win over 2017–2026 in this curated universe (**survivorship**). The remaining dynamic/timing
component (~0.015) is borderline and not robust.

## Tradable spreads (top-5 by prob minus baseline, forward 5d)
| Spread | mean | 95% CI | significant? |
|---|---|---|---|
| top5 − equal-weight | +0.0010 | [−0.0014, +0.0033] | no |
| top5 − momentum-top5 | −0.0021 | [−0.0085, +0.0042] | no (negative) |
| top5 − random-5 | +0.0008 | [−0.0019, +0.0034] | no |

Earlier portfolio check: **83% of single-draw random-5 daily portfolios beat the Transformer top-5**
(random Sharpe ~1.9 vs real 1.66). The headline "Sharpe 1.66 beats SPY/QQQ/EQW" was universe drift.

## Robustness (drop-top)
Partial IC stays ~+0.02–0.03 and significant after dropping top 1/3/5 symbols and top 1/2 years —
but `top5 − momentum` stays **negative** throughout. The static effect is robust; the tradable edge
is robustly **absent**.

## Verdict
- raw rank IC positive: **yes** (but 63% static survivorship)
- dynamic IC robustly positive: **no** (CI includes 0)
- static survivorship dominates: **yes**
- tradable over simple baselines: **no**
- date-shuffle placebo collapses: **no** (confirms static artifact)

**Decision: REJECT the legacy Transformer as a deployable signal — another false positive under
rigorous controls.** Keep the MLP champion; no paper/live; no run.py changes.

## Implication for Kronos (parked)
The prerequisite for the Kronos-on-top experiment — "a true residual signal remains after removing
survivorship/universe" — is **not met**. The residual dynamic signal is marginal and non-tradable, so
there is no clean base to test Kronos increments against. **Kronos stays paused** (Stage 0 audit done,
feasible) until a genuine residual signal is established from some source. Do not deploy/fine-tune Kronos.
