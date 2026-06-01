# Leakage audit — `alpha_exit_dataset`

- **Verdict: LEAKAGE DETECTED**
- target: `realized_R`  | rows: 54675  | features: 19  | win-rate: 0.3328

## Baselines (temporal expanding-window CV, out-of-sample)
- LogReg OOS AUC: {'mean': 1.0, 'std': 0.0, 'folds': 5}
- RandomForest OOS AUC: {'mean': 1.0, 'std': 0.0, 'folds': 5}
- RF-Regressor OOS R²: {'mean': 1.0, 'std': 0.0, 'folds': 5}
- macro-leak (model nails a future target OOS): **True**

## Flagged features (4)  — AUC>0.75 / R²>0.5 / near-deterministic

| feature | AUC | R² | Spearman² | reason |
|---|--:|--:|--:|---|
| `unrealized_R` | 1.0 | 1.0 | 1.0 | AUC 1.0>0.75; near-deterministic linear R² 1.0; Spearman² 1.0>0.5 |
| `premium_ratio` | 1.0 | 1.0 | 1.0 | AUC 1.0>0.75; near-deterministic linear R² 1.0; Spearman² 1.0>0.5 |
| `stop_distance` | 1.0 | 0.9718 | 0.6989 | AUC 1.0>0.75; R² 0.9718>0.5; Spearman² 0.6989>0.5 |
| `trail_high_ratio` | 1.0 | 0.9598 | 0.8162 | AUC 1.0>0.75; R² 0.9598>0.5; Spearman² 0.8162>0.5 |

## Top univariate signal (all features)

| feature | AUC | R² | Spearman² |
|---|--:|--:|--:|
| `unrealized_R` | 1.0 | 1.0 | 1.0 |
| `premium_ratio` | 1.0 | 1.0 | 1.0 |
| `stop_distance` | 1.0 | 0.9718 | 0.6989 |
| `trail_high_ratio` | 1.0 | 0.9598 | 0.8162 |
| `spy_regime` | 0.52 | 0.0067 | 0.0385 |
| `mtf_trend_slow` | 0.5169 | 0.0016 | 0.0153 |
| `atr_pct` | 0.5157 | 0.0015 | 0.0224 |
| `mtf_trend_mid` | 0.5163 | 0.0008 | 0.0008 |
| `time_to_close` | 0.5121 | 0.001 | 0.0002 |
| `mtf_alignment` | 0.5079 | 0.0006 | 0.0 |
| `probability_band` | 0.5021 | 0.0009 | 0.0076 |
| `sector_regime` | 0.5023 | 0.0005 | 0.0116 |

## Permutation importance (RandomForest, top 8)

| feature | importance |
|---|--:|
| `stop_distance` | 0.15934 |
| `unrealized_R` | 0.13312 |
| `trail_high_ratio` | 0.09815 |
| `premium_ratio` | 0.09404 |
| `time_to_close` | 0.0 |
| `vwap_distance` | 0.0 |
| `atr_pct` | 0.0 |
| `mtf_trend_slow` | 0.0 |

## Interpretation
**Leakage present.** One or more features (near-)determine the target, and/or a baseline predicts a future target out-of-sample implausibly well. The target is likely derived from features (forbidden). Re-target from FUTURE bars before any model use.
