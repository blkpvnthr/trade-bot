# Nightly Champion/Challenger Report — Mean-reversion — universe
_experiment edb64b371c38416d_

## Champion

| Portfolio | trades | days | Sharpe | mean-net | CAGR | MaxDD |
|---|---|---|---|---|---|---|
| meanrev_orig10_flat | 398 | 522 | +1.824 | +0.28% | +23.7% | -21% |

## Challengers (ranked: PROMOTE → CONTINUE → REJECT)

| Challenger | verdict | ΔSharpe full | p | ΔSharpe OOS | trades | days | Sharpe | mean-net | CAGR | MaxDD | failed checks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| meanrev_broad_etf_flat | **REJECT** | -0.116 | — | -1.217 | 2501 | 972 | +1.708 | +0.14% | +22.3% | -29% | sharpe_improvement,per_year_nonneg,oos_sharpe_improvement,significance |

## Verdict legend
- **PROMOTE**: beats champion on every preregistered criterion (incl. bootstrap significance + per-year robustness).
- **CONTINUE**: improves but not yet significant / insufficient burn-in → keep observing.
- **REJECT**: no Sharpe improvement, or breaks drawdown / per-year robustness.

- `meanrev_broad_etf_flat` — Broad ETF universe (all classes), flat (expected: REJECT — breadth dilutes).