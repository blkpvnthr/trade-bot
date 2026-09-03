# Insider book - market/regime-vol EXPOSURE scaling (Track-1-step-1)

_181 vol-gated trades, 100bps, 3d hold. Pure sizing; mean exposure matched across variants._

| Metric | 1_baseline | 2_trend | 3_vol | 4_combined |
|---|---|---|---|---|
| trades_taken | 169 | 169 | 169 | 169 |
| avg_deployed_exposure | 0.1 | 0.1048 | 0.1029 | 0.1057 |
| win_rate | 0.5799 | 0.5799 | 0.5799 | 0.5799 |
| avg_R_net | 0.02262 | 0.02262 | 0.02262 | 0.02262 |
| CAGR | 0.0764 | 0.0587 | 0.0601 | 0.0458 |
| sharpe_active | 2.569 | 2.271 | 2.075 | 1.663 |
| max_drawdown | -0.074 | -0.1189 | -0.1168 | -0.1538 |
| calmar | 1.033 | 0.494 | 0.514 | 0.298 |
| pf_size_weighted | 1.684 | 1.519 | 1.536 | 1.42 |
| top1_contribution | 0.1431 | 0.122 | 0.1698 | 0.2082 |
| top5_contribution | 0.4595 | 0.5749 | 0.551 | 0.7571 |
| total_return | 0.336 | 0.2517 | 0.258 | 0.1928 |
| total_drop_top1 | 0.2752 | 0.2094 | 0.2011 | 0.1352 |
| total_drop_top3 | 0.2024 | 0.1307 | 0.1228 | 0.0568 |
| total_drop_top5 | 0.1428 | 0.0632 | 0.0725 | -0.0062 |

## Regime breakdown (trade-level, where the edge concentrates)
| Regime | n | mean net | win rate |
|---|---|---|---|
| bull | 90 | +1.44% | 58% |
| bear | 73 | +4.30% | 63% |
| n/a | 18 | +1.28% | 61% |

## Read
- `win_rate`/`avg_R_net` are trade-level and IDENTICAL across variants by design (pure sizing). The discriminators are `sharpe_active`, `max_drawdown`, `calmar`, and the `drop_topN` robustness.
- `avg_deployed_exposure` ~ alloc confirms the mean-exposure normalization (timing test, not leverage).
- A variant only wins if it lifts Sharpe/Calmar or cuts drawdown WITHOUT the gain vanishing under drop-top-N (i.e. not just concentration in a few lucky names).