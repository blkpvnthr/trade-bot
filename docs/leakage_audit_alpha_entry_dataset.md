# Leakage audit — `alpha_entry_dataset`

- **Verdict: CLEAN**
- target: `realized_R`  | rows: 12164  | features: 34  | win-rate: 0.5691

## Baselines (temporal expanding-window CV, out-of-sample)
- LogReg OOS AUC: {'mean': 0.4851, 'std': 0.0268, 'folds': 5}
- RandomForest OOS AUC: {'mean': 0.5225, 'std': 0.0404, 'folds': 5}
- RF-Regressor OOS R²: {'mean': -0.2266, 'std': 0.1083, 'folds': 5}
- macro-leak (model nails a future target OOS): **False**

## Flagged features (0)  — AUC>0.75 / R²>0.5 / near-deterministic

_None — no single feature is near-deterministic of the target._

## Top univariate signal (all features)

| feature | AUC | R² | Spearman² |
|---|--:|--:|--:|
| `vwap_distance` | 0.5328 | 0.0003 | 0.01 |
| `mtf_trend_mid` | 0.5323 | 0.0 | 0.0063 |
| `mtf_trend_slow` | 0.525 | 0.0 | 0.0045 |
| `mtf_alignment` | 0.5158 | 0.0012 | 0.0021 |
| `mtf_trend_fast` | 0.5147 | 0.0 | 0.0014 |
| `sector_regime` | 0.506 | 0.0016 | 0.0138 |
| `atr_pct` | 0.5042 | 0.0 | 0.0 |
| `rel_volume` | 0.5027 | 0.0 | 0.0001 |
| `setup_score` | 0.5024 | 0.0 | 0.0001 |
| `time_of_day` | 0.5014 | 0.0 | 0.0 |
| `catalyst_strength` | 0.5011 | 0.0 | 0.0006 |
| `spy_regime` | 0.501 | 0.0 | 0.0101 |

## Permutation importance (RandomForest, top 8)

| feature | importance |
|---|--:|
| `mtf_trend_fast` | 0.00668 |
| `time_of_day` | 0.00231 |
| `rel_volume` | 0.00029 |
| `optionlab_score` | 0.0 |
| `probability_of_profit` | 0.0 |
| `ga_regime_switch` | 0.0 |
| `oops_prob_total_loss` | -0.0 |
| `ga_bollinger_pctb` | -0.0 |

## Interpretation
**No leakage detected.** No feature near-determines the target and baselines do not predict it out-of-sample beyond chance. (Absence of leakage ≠ presence of edge.)
