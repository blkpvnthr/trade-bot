# Nightly Champion/Challenger Report — Insider — gate & sizing
_experiment 106c4ff12aeb5fe7_

## Champion

| Portfolio | trades | days | Sharpe | mean-net | CAGR | MaxDD |
|---|---|---|---|---|---|---|
| insider_volgate_flat | 200 | 504 | -0.542 | +1.71% | -26.0% | -81% |

## Challengers (ranked: PROMOTE → CONTINUE → REJECT)

| Challenger | verdict | ΔSharpe full | p | ΔSharpe OOS | trades | days | Sharpe | mean-net | CAGR | MaxDD | failed checks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| insider_fullbook_flat | **PROMOTE** | +3.131 | 0.003 | +2.525 | 1051 | 986 | +2.589 | +1.51% | +58.7% | -40% | — |
| insider_voltilt_light | **REJECT** | +2.912 | 0.003 | +2.249 | 1051 | 986 | +2.370 | +1.51% | +55.4% | -42% | drawdown |
| insider_voltilt_med | **REJECT** | +2.652 | 0.003 | +1.889 | 1051 | 986 | +2.109 | +1.51% | +51.4% | -43% | drawdown |
| insider_gate_p50 | **REJECT** | +0.753 | 0.203 | +1.871 | 499 | 810 | +0.211 | +1.68% | +1.9% | -72% | drawdown,significance |
| insider_volgate_buckets | **REJECT** | +0.089 | 0.333 | -0.449 | 200 | 504 | -0.453 | +1.71% | -24.0% | -79% | drawdown,per_year_nonneg,oos_sharpe_improvement,significance |

## Verdict legend
- **PROMOTE**: beats champion on every preregistered criterion (incl. bootstrap significance + per-year robustness).
- **CONTINUE**: improves but not yet significant / insufficient burn-in → keep observing.
- **REJECT**: no Sharpe improvement, or breaks drawdown / per-year robustness.

- `insider_fullbook_flat` — Full diversified book, flat sizing (the prior PROMOTE lead).
- `insider_voltilt_light` — Full book + LIGHT vol tilt (vol as a lean, not a hard gate).
- `insider_voltilt_med` — Full book + medium vol tilt.
- `insider_gate_p50` — Relaxed vol-gate (P50): more concurrent names, keeps mild vol selection.
- `insider_volgate_buckets` — Conviction A+8/A6/B4/C2 buckets on vol+breadth (expected: no gain).