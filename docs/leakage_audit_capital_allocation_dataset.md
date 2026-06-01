# Leakage audit — `capital_allocation_dataset`

- **Verdict: LEAKAGE DETECTED**
- target: `alpha_bucket_return`  | rows: 169  | features: 8  | win-rate: 0.1953

## Baselines (temporal expanding-window CV, out-of-sample)
- LogReg OOS AUC: None
- RandomForest OOS AUC: None
- RF-Regressor OOS R²: None
- macro-leak (model nails a future target OOS): **False**

## Flagged features (5)  — AUC>0.75 / R²>0.5 / near-deterministic

| feature | AUC | R² | Spearman² | reason |
|---|--:|--:|--:|---|
| `alpha_expectancy` | 1.0 | 1.0 | 1.0 | AUC 1.0>0.75; near-deterministic linear R² 1.0; Spearman² 1.0>0.5 |
| `volatility` | 0.9847 | 0.4173 | 0.9351 | AUC 0.9847>0.75; Spearman² 0.9351>0.5 |
| `alpha_perf` | 1.0 | 0.079 | 0.9853 | AUC 1.0>0.75; Spearman² 0.9853>0.5 |
| `buffett_quality` | 0.8953 | 0.0854 | 0.105 | AUC 0.8953>0.75 |
| `optimizer_perf` | 0.8953 | 0.0854 | 0.105 | AUC 0.8953>0.75 |

## Top univariate signal (all features)

| feature | AUC | R² | Spearman² |
|---|--:|--:|--:|
| `alpha_expectancy` | 1.0 | 1.0 | 1.0 |
| `volatility` | 0.9847 | 0.4173 | 0.9351 |
| `alpha_perf` | 1.0 | 0.079 | 0.9853 |
| `buffett_quality` | 0.8953 | 0.0854 | 0.105 |
| `optimizer_perf` | 0.8953 | 0.0854 | 0.105 |
| `drawdown` | 0.5037 | 0.0024 | 0.0726 |
| `market_state_confidence` | None | 0.0 | 0.0 |
| `waterfall_ratio` | None | 0.0 | 0.0 |

## Permutation importance (RandomForest, top 8)

| feature | importance |
|---|--:|
| `alpha_expectancy` | 0.0 |
| `alpha_perf` | 0.0 |
| `buffett_quality` | 0.0 |
| `market_state_confidence` | 0.0 |
| `drawdown` | 0.0 |
| `volatility` | 0.0 |
| `waterfall_ratio` | 0.0 |
| `optimizer_perf` | 0.0 |

## Interpretation
**Leakage present.** One or more features (near-)determine the target, and/or a baseline predicts a future target out-of-sample implausibly well. The target is likely derived from features (forbidden). Re-target from FUTURE bars before any model use.
