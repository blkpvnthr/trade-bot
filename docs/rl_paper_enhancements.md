# RL Enhancements from "Reinforcement Learning Framework for Quantitative Trading"

Source: Yasin & Gill (2024), arXiv:2411.07585. Applied to the offline two-engine RL
stack on 2026-05-31. All changes are offline/advisory — no execution-authority change,
no RL promotion, deployment gate intact. Suite: 233 tests passing.

## What the paper recommends → what we changed

| Paper finding | Section | Our change |
|---|---|---|
| Inconsistent feature scaling degrades the RL model; normalize inputs (Min-Max / Z-Score / Sigmoid / L2). | §8 | `rl_common.normalize_obs()` — deterministic, stateless **tanh** squash (configurable `RL_OBS_NORM=tanh\|clip\|none`) applied identically in every env `_obs()` **and** each `recommend()` (train/inference consistent, no fitted params to persist). Bounds unbounded features (expected_R, atr_expansion, unrealized_R). |
| A model with a 100% win rate that **traded once** is overfit, not deployable. | §11.5 | Trade-activity overfit guard in `deployment_recommendation`: a *positive* rec (paper/eligible) on `active_trades < 30` or `activity_ratio < 0.05` is downgraded to `shadow_mode_only` + `overfit_warning`. `compute_metrics` now reports `active_trades` / `activity_ratio`. No-op when the metric is absent (back-compatible). |
| γ=0.99, tuned learning rate / batch; LR controls exploration vs stability. | §11.3 (Table 1) | `train_rl_agents` hyperparameters made configurable & paper-aligned: `RL_GAMMA=0.99`, `RL_LEARNING_RATE`, `RL_BATCH_SIZE`, `RL_N_STEPS`, `RL_N_EPOCHS`; recorded in the retraining report. |
| Discrete vs continuous action spaces; a true "hold/do-nothing" state matters. | §3.3, §6.1 | Already modeled: entry has `wait`/`skip`, exit has `hold`, allocation is a continuous Box. (Confirmed, no change.) |
| Sharpe/Sortino/Calmar/Win-rate/Return/Volatility evaluation; walk-forward; strategy degradation → continuous refinement. | §11–§12 | Already implemented (`compute_metrics` + walk-forward evaluator + the deployment gate). Paper validates the design. |
| Indicator correlation: RSI strongly correlated with Momentum/MACD/CCI/ROC/Stoch (redundancy). | §9 | Noted; our per-agent feature sets are small/curated, so decorrelation is not yet required. Candidate future work. |

## Measured effect

Re-training with normalized observations + paper-aligned hyperparameters, on the same
out-of-sample data, improved the alpha-exit agent's expectancy **0.287 → 0.438 (+53%)**
(12,726 OOS samples), holding its `paper_mode_only` recommendation. `buffett_dca` stayed
`paper_mode_only`; `alpha_entry` remained `do_not_deploy` (skips too much to beat baseline);
`capital_allocation` remained `shadow_mode_only` (too few samples).

## Safety unchanged

`live_wired=False`, shadow OFF, **no promotion**. The two `paper_mode_only` results are
recommendations awaiting explicit approval. No thresholds weakened; `MIN_DEPLOY_SAMPLES`
unchanged; no order paths touched. Tests: `tests/test_rl_paper_enhancements.py` (10).
