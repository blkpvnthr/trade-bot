# SR setup edge audit

- universe: `QQQ,SPY,IWM,AAPL,NVDA,AMD,MSFT,TSLA`  | lookback: 365d  | RR: 3.0  | cost: 0.0005
- signals: 593014 (OOS holdout last 30% = 177910)
- **Leakage verdict: CLEAN**
- **Edge verdict: NO EDGE (clean but no OOS edge over control)**

## Leakage — baselines (temporal expanding-window CV, OOS)
- LogReg OOS AUC: {'mean': 0.5106, 'std': 0.0291, 'folds': 5}
- RandomForest OOS AUC: {'mean': 0.5292, 'std': 0.0326, 'folds': 5}
- RF-Regressor OOS R²: {'mean': -0.0106, 'std': 0.006, 'folds': 5}  (macro-leak=False)

### Flagged features (0)
_None — no feature near-determines realized_R._

## Top univariate signal (all features)
| feature | AUC | R² | Spearman² |
|---|--:|--:|--:|
| `minutes_since_open` | 0.5132 | 0.0001 | 0.0002 |
| `resistance_strength` | 0.5073 | 0.0006 | 0.0004 |
| `breakout_retest` | 0.5064 | 0.0 | 0.0003 |
| `support_distance_pct` | 0.5054 | 0.0007 | 0.0005 |
| `resistance_distance_pct` | 0.5048 | 0.0001 | 0.0012 |
| `missed_reversal_level_reclaim` | 0.5047 | 0.0001 | 0.0 |
| `support_strength` | 0.5037 | 0.0 | 0.0004 |
| `red_candle_streak` | 0.503 | 0.0 | 0.0 |
| `green_candle_streak` | 0.5028 | 0.0 | 0.0 |
| `rvol` | 0.5023 | 0.0 | 0.0 |

## OOS edge (last holdout) vs random-entry control

| group | n | win% | expectancy_R | PF | median_R |
|---|--:|--:|--:|--:|--:|
| **setups (all)** | 177910 | 0.4515 | -0.0465 | 0.797 | -0.044 |
| random control | 52688 | 0.4525 | -0.0395 | 0.85 | -0.05 |
| breakout_retest_continuation | 58750 | 0.453 | -0.0322 | 0.847 | -0.0416 |
| liquidity_sweep_reclaim | 5839 | 0.4564 | -0.0401 | 0.837 | -0.0365 |
| missed_reversal_reclaim | 111357 | 0.4501 | -0.0556 | 0.764 | -0.0453 |
| support_bounce | 1 | 1.0 | 0.0817 | inf | 0.0817 |
| vwap_reclaim | 1963 | 0.4692 | 0.0199 | 1.047 | -0.0881 |

## Interpretation
**Not activation-worthy.** Setups do not beat random entries OOS after cost — the apparent profit is stop/target geometry, not signal. Keep sr_rl_agent frozen; this matches the repo's prior RL-edge findings.
