# Leakage audit — `buffett_dca_dataset`

- **Verdict: LEAKAGE DETECTED**
- target: `fwd_return_60d`  | rows: 4328  | features: 18  | win-rate: 0.9956

## Baselines (temporal expanding-window CV, out-of-sample)
- LogReg OOS AUC: None
- RandomForest OOS AUC: None
- RF-Regressor OOS R²: {'mean': 0.926, 'std': 0.0966, 'folds': 5}
- macro-leak (model nails a future target OOS): **True**

## Flagged features (5)  — AUC>0.75 / R²>0.5 / near-deterministic

| feature | AUC | R² | Spearman² | reason |
|---|--:|--:|--:|---|
| `conviction_score` | 0.9969 | 0.8379 | 0.4617 | AUC 0.9969>0.75; R² 0.8379>0.5 |
| `bal_score` | 0.9969 | 0.6923 | 0.4616 | AUC 0.9969>0.75; R² 0.6923>0.5 |
| `intrinsic_value_discount` | 0.6715 | 0.9273 | 0.4076 | R² 0.9273>0.5 |
| `composite_dca_score` | 0.7056 | 0.5971 | 0.3911 | R² 0.5971>0.5 |
| `optimized_target_weight` | 0.9913 | 0.0092 | 0.0014 | AUC 0.9913>0.75 |

## Top univariate signal (all features)

| feature | AUC | R² | Spearman² |
|---|--:|--:|--:|
| `conviction_score` | 0.9969 | 0.8379 | 0.4617 |
| `bal_score` | 0.9969 | 0.6923 | 0.4616 |
| `intrinsic_value_discount` | 0.6715 | 0.9273 | 0.4076 |
| `composite_dca_score` | 0.7056 | 0.5971 | 0.3911 |
| `optimized_target_weight` | 0.9913 | 0.0092 | 0.0014 |
| `mtf_alignment` | 0.5328 | 0.2398 | 0.1619 |
| `mtf_trend_slow` | 0.607 | 0.1001 | 0.278 |
| `rsi` | 0.5175 | 0.1602 | 0.158 |
| `mtf_trend_mid` | 0.5078 | 0.0574 | 0.1165 |
| `mtf_trend_fast` | 0.5205 | 0.0128 | 0.0304 |
| `atr_expansion` | 0.5223 | 0.0003 | 0.0102 |
| `current_weight` | None | 0.0 | 0.0 |

## Permutation importance (RandomForest, top 8)

| feature | importance |
|---|--:|
| `conviction_score` | 0.21749 |
| `intrinsic_value_discount` | 0.21567 |
| `bal_score` | 0.12874 |
| `mtf_trend_slow` | 0.00069 |
| `mtf_trend_fast` | 0.00019 |
| `bl_posterior_return` | 0.0 |
| `optimized_target_weight` | 0.0 |
| `premarket_gap_pct` | 0.0 |

## Interpretation
**Leakage present.** One or more features (near-)determine the target, and/or a baseline predicts a future target out-of-sample implausibly well. The target is likely derived from features (forbidden). Re-target from FUTURE bars before any model use.
