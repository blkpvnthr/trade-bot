# "Optimizing Automated Trading Systems with DRL" → RL agent optimization

Source: Tran, Pham-Hi & Bui, *"Optimizing Automated Trading Systems with Deep Reinforcement
Learning"* (Algorithms 2023, 16, 23; MDPI). Reframes strategy **parameter optimization** as an
MDP and benchmarks DRL vs. Bayesian Optimization.

## The paper's transferable findings

1. **Sharpe ratio is the winning objective.** Their DDQN+Sharpe agent earned positive,
   low-volatility returns in *every* period; the cumulative-return objective had higher means
   but unstable, high-variance results. → optimize for *risk-adjusted* return.
2. **DDQN > Dueling-DDQN > Bayesian Optimization** for short-term, frequently re-tuned trading
   (DDQN also 5.83× faster than BO).
3. **Parameter tuning as objective-driven search** (MDP: state = history of (config, reward);
   action = next config; reward = strategy objective), with **walk-forward 80/20 anti-overfit**.

## What was already in the repo (pre-existing)

`rl_agent_core.py` already cited this paper and implemented: `CNNFeatureExtractor` (§3.2),
`SharpeRewardCalculator` (§3.4), `EnsembleAgentSelector` (Sharpe-scored), Leaky-ReLU + ADAM.
The Optuna tuners (`tune_rl_intraday.py`, `tune_sr_rl_agent.py`) already include Sharpe.

## What this cycle added (the remaining gaps)

| Gap vs. paper | Change |
|---|---|
| Sharpe was **not** the selection objective in the deploy gate (ranked on expectancy+DD) | `rl_common.deployment_recommendation` is now **Sharpe-primary**: a material Sharpe regression (`> RL_SHARPE_REGRESSION_TOL`, default 10%) returns `do_not_deploy` even if raw expectancy improved; the best case requires expectancy **and** drawdown **and** Sharpe to improve. |
| No objective-driven parameter search | **New `rl_parameter_optimizer.py`** — coordinate-ascent over the selection levers (`quality_pct`, `min_confluence`, `sent_gate_min`) maximizing **IS full-sample trade-Sharpe**, 80/20 walk-forward, terminates on repeat config (paper's stuck-rule), with a **generalization guard** that refuses to recommend a change whose IS gain doesn't hold OOS. Reuses the overlay's `select`/`simulate`. Emits ranked report + optional `.env` patch. |
| Sharpe reward fixed at weight 0.1 ("shaped rewards dominate") — opposite of paper | Trainer weight is now `RL_SHARPE_REWARD_WEIGHT` (default 0.10 unchanged); raise to 0.5–1.0 to test the Sharpe-dominant finding, compared via the now-Sharpe-primary deploy gate. |
| No Double-DQN agent (trainer is PPO/SAC) | `rl_agent_core.TRAN_DDQN_HYPERPARAMS` (Table 2: γ=0.98, lr=0.001, batch=40, replay=10k, target-update=10, ε 1→0.12 over 30%) + guarded `make_ddqn_agent` (sb3-contrib QR-DQN, Double-DQN-style, CNN extractor). Additive; PPO default unchanged. Verified to build. |

## Validation (this cycle, offline)

- Deploy gate: unit-tested — *expectancy-up but Sharpe-down → `do_not_deploy`* (the paper's
  core lesson); back-compatible when baseline Sharpe = 0.
- `rl_parameter_optimizer.py`: ran on 12,206 enriched trades (IS 9,764 / OOS 2,442). It lifted
  **IS Sharpe 0.215 → 0.255** by choosing `min_confluence=3`, but the walk-forward guard found
  the gain **did not hold OOS** (OOS Sharpe 0.290 → 0.210) and correctly recommended
  **KEEP_CURRENT_CONFIG** — demonstrating the anti-overfit discipline working as intended.
- `make_ddqn_agent`: builds a QR-DQN with the CNN extractor and all Table-2 params on a real
  Discrete env; raises a clear `ImportError` if sb3-contrib is absent.

## How to use

```bash
python3 rl_parameter_optimizer.py [--write-env-patch]   # Sharpe-objective walk-forward tune
RL_SHARPE_REWARD_WEIGHT=0.7 python3 rl_intraday_structure_trainer.py   # test Sharpe-dominant reward
```
The deploy gate (`deployment_recommendation`) now enforces Sharpe on every promotion decision.
Tunables: `RL_SHARPE_REGRESSION_TOL`, `RL_PO_MIN_IS_TRADES`, `RL_PO_MIN_OOS_TRADES`,
`RL_PO_OVERFIT_GAP`, `RL_SHARPE_REWARD_WEIGHT`.
