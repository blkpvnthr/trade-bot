#!/usr/bin/env python3
"""
train_rl_agents.py — Offline training for the four two-engine RL agents.
========================================================================

Trains (PPO) Alpha Entry / Alpha Exit / Buffett DCA, and (PPO over a continuous Box)
Capital Allocation, on the offline datasets. SAFETY: training is offline/simulated only —
no Alpaca, no orders, and trained models are NEVER auto-wired into live behavior. If
stable-baselines3 / gymnasium are unavailable (or training errors), it falls back to a
saved heuristic policy card so the pipeline still completes and reports a model artifact.

Writes: rl_models/rl_v2/<agent>.zip (or <agent>_heuristic.json) + a retraining report at
daily_system_outputs/YYYYMMDD/rl_retraining_report.json.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import rl_common as rc

logger = logging.getLogger(__name__)

# agent -> (env module, env class attr, dataset name, algo, reward description)
AGENTS = {
    "alpha_entry": ("rl_env_alpha_entry", "AlphaEntryEnv", "alpha_entry_dataset", "PPO",
                    "OOPS utility-adjusted realized_R - bid/ask - skew/kurtosis/leverage risk + avoid-loser/take-winner bonuses"),
    "alpha_exit": ("rl_env_alpha_exit", "AlphaExitEnv", "alpha_exit_dataset", "PPO",
                   "realized_R - giveback_from_peak - holding_risk + winner/overnight bonuses"),
    "buffett_dca": ("rl_env_buffett_dca", "BuffettDCAEnv", "buffett_dca_dataset", "PPO",
                    "forward_return + valuation_improvement - drawdown - overconcentration"),
    "capital_allocation": ("rl_env_capital_allocation", "CapitalAllocationEnv", "capital_allocation_dataset", "PPO",
                           "portfolio_log_return + Sharpe + Sortino - drawdown - turnover"),
}

# Per-agent training budgets. The enrichment enlarged the observation space unevenly
# (alpha_entry 34 features after GA+OR+OOPS, alpha_exit 19, buffett 18, allocation 8), and the
# agents differ in sample count and difficulty, so a single global budget under/over-trains some.
# Each
# agent reads RL_TRAIN_TIMESTEPS_<AGENT>; the global RL_TRAIN_TIMESTEPS is only a fallback
# when the per-agent value is unset.
RL_TRAIN_TIMESTEPS = int(os.getenv("RL_TRAIN_TIMESTEPS", "6000"))
PER_AGENT_TIMESTEPS = {
    "alpha_entry": 20000,
    "alpha_exit": 6000,
    "buffett_dca": 4000,
    "capital_allocation": 6000,
}


def agent_timesteps(agent: str) -> int:
    """Per-agent timestep budget: RL_TRAIN_TIMESTEPS_<AGENT> > per-agent default > global."""
    env_val = os.getenv(f"RL_TRAIN_TIMESTEPS_{agent.upper()}")
    if env_val:
        return int(env_val)
    return PER_AGENT_TIMESTEPS.get(agent, RL_TRAIN_TIMESTEPS)
# Paper-aligned, configurable hyperparameters (Yasin & Gill 2024, §11.3 Table 1: γ=0.99,
# batch 128; tuned learning rate). PPO uses its own stable LR default; all overridable.
RL_LEARNING_RATE = float(os.getenv("RL_LEARNING_RATE", "3e-4"))
RL_GAMMA = float(os.getenv("RL_GAMMA", "0.99"))
RL_BATCH_SIZE = int(os.getenv("RL_BATCH_SIZE", "64"))
RL_N_STEPS = int(os.getenv("RL_N_STEPS", "64"))
RL_N_EPOCHS = int(os.getenv("RL_N_EPOCHS", "4"))

# Optional DQN with experience replay + a periodically-updated target network (Roy 2024,
# the "DQN" in TDH-DQN). SB3's DQN provides exactly the two stabilisation methods the thesis
# describes (a replay buffer that decorrelates observations, and a frozen target network).
# Hyperparameters follow the thesis Table 6.1 (γ=0.99, lr=1e-3, batch=86, min-ε=0.01).
# Discrete-action agents only — capital_allocation is a continuous Box, so it stays PPO.
# DEFAULT OFF: the AGENTS table keeps PPO, so current results/tests are unchanged unless a
# run opts in via RL_ALGO_<AGENT>=DQN (or RL_DISCRETE_ALGO=DQN).
RL_DQN_LEARNING_RATE = float(os.getenv("RL_DQN_LEARNING_RATE", "1e-3"))
RL_DQN_BATCH_SIZE = int(os.getenv("RL_DQN_BATCH_SIZE", "86"))
RL_DQN_BUFFER = int(os.getenv("RL_DQN_BUFFER", "10000"))
RL_DQN_EXPLORATION_FRACTION = float(os.getenv("RL_DQN_EXPLORATION_FRACTION", "0.3"))
RL_DQN_FINAL_EPS = float(os.getenv("RL_DQN_FINAL_EPS", "0.01"))


def resolve_algo(agent: str, algo: Optional[str]) -> str:
    """Per-agent algorithm: explicit arg > RL_ALGO_<AGENT> env > RL_DISCRETE_ALGO > table default."""
    if algo:
        return algo
    return (os.getenv(f"RL_ALGO_{agent.upper()}") or os.getenv("RL_DISCRETE_ALGO")
            or AGENTS[agent][3])


def _is_discrete(env) -> bool:
    return hasattr(env.action_space, "n")


def _make_env(agent: str):
    mod_name, cls_name, _, _, _ = AGENTS[agent]
    mod = __import__(mod_name)
    return getattr(mod, cls_name)()


def train_one(agent: str, timesteps: Optional[int] = None, algo: Optional[str] = None,
              out_dir: Path = None) -> Dict[str, Any]:
    """Train (or heuristic-save) one agent. Always returns a result dict; never raises."""
    if agent not in AGENTS:
        return {"agent": agent, "saved": False, "error": "unknown_agent"}
    out_dir = Path(out_dir) if out_dir else rc.RL_MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    mod_name, cls_name, dataset, default_algo, reward_desc = AGENTS[agent]
    env = _make_env(agent)
    samples = len(getattr(env, "rows", []))
    timesteps = agent_timesteps(agent) if timesteps is None else int(timesteps)
    algo = resolve_algo(agent, algo)
    result = {"agent": agent, "algo": algo, "dataset": dataset, "training_samples": samples,
              "reward_function": reward_desc, "timesteps": timesteps, "saved": False,
              "model_path": None, "status": None}
    # OOPS feature coverage is a data-quality gate: the alpha-entry model can train without
    # these features, but the report should reveal whether the paper-aligned utility layer
    # was actually present in the offline dataset.
    if agent == "alpha_entry":
        try:
            from oops_option_utility import OOPS_FIELDS
            rows = getattr(env, "rows", []) or []
            denom = max(1, len(rows) * len(OOPS_FIELDS))
            nonzero = sum(1 for row in rows for f in OOPS_FIELDS if abs(float(row.get(f, 0.0) or 0.0)) > 1e-12)
            result["oops_feature_coverage"] = round(nonzero / denom, 4)
            result["oops_fields"] = OOPS_FIELDS
        except Exception as exc:
            result["oops_feature_coverage_error"] = str(exc)[:120]

    use_sb3 = algo != "heuristic" and rc.GYM_AVAILABLE and rc.sb3_available()
    if use_sb3:
        try:
            want_dqn = algo.upper() == "DQN"
            if want_dqn and not _is_discrete(env):
                # DQN cannot act on a continuous Box (capital_allocation) — keep PPO.
                want_dqn = False
                algo = "PPO"
                result["algo"] = "PPO"
                result["algo_note"] = "dqn_requires_discrete_actions->ppo"
            if want_dqn:
                from stable_baselines3 import DQN
                model = DQN("MlpPolicy", env, learning_rate=RL_DQN_LEARNING_RATE, gamma=RL_GAMMA,
                            batch_size=RL_DQN_BATCH_SIZE, buffer_size=RL_DQN_BUFFER,
                            learning_starts=min(64, max(8, timesteps // 10)),
                            exploration_fraction=RL_DQN_EXPLORATION_FRACTION,
                            exploration_final_eps=RL_DQN_FINAL_EPS, verbose=0, device="cpu")
                model.learn(total_timesteps=max(64, timesteps))
                result["hyperparameters"] = {"algo": "DQN", "learning_rate": RL_DQN_LEARNING_RATE,
                                             "gamma": RL_GAMMA, "batch_size": RL_DQN_BATCH_SIZE,
                                             "buffer_size": RL_DQN_BUFFER,
                                             "exploration_fraction": RL_DQN_EXPLORATION_FRACTION,
                                             "exploration_final_eps": RL_DQN_FINAL_EPS}
                status = "trained_dqn"
            else:
                from stable_baselines3 import PPO
                model = PPO("MlpPolicy", env, learning_rate=RL_LEARNING_RATE, gamma=RL_GAMMA,
                            n_steps=RL_N_STEPS, batch_size=RL_BATCH_SIZE, n_epochs=RL_N_EPOCHS,
                            verbose=0, device="cpu")
                model.learn(total_timesteps=max(64, timesteps))
                result["hyperparameters"] = {"algo": "PPO", "learning_rate": RL_LEARNING_RATE,
                                             "gamma": RL_GAMMA, "batch_size": RL_BATCH_SIZE,
                                             "n_steps": RL_N_STEPS, "n_epochs": RL_N_EPOCHS}
                status = "trained_ppo"
            p = out_dir / f"{agent}.zip"
            model.save(str(p))
            result.update({"saved": True, "model_path": str(p), "status": status})
            rc.write_json(out_dir / f"{agent}_card.json",
                          {"agent": agent, "algo": algo, "model_path": str(p),
                           "training_samples": samples, "reward_function": reward_desc,
                           "trained_at_utc": rc.now_iso()})
            return result
        except Exception as exc:
            logger.warning("SB3 training failed for %s (%s) — heuristic fallback", agent, exc)
            result["status"] = f"sb3_failed:{str(exc)[:80]}"

    # Heuristic fallback — save a policy card so a model artifact always exists.
    p = out_dir / f"{agent}_heuristic.json"
    rc.write_json(p, {"agent": agent, "algo": "heuristic", "training_samples": samples,
                      "reward_function": reward_desc, "trained_at_utc": rc.now_iso(),
                      "note": "stable-baselines3/gymnasium unavailable or training skipped"})
    result.update({"saved": True, "model_path": str(p), "algo": "heuristic",
                   "status": result["status"] or "heuristic_fallback"})
    return result


def train_all(timesteps: Optional[int] = None) -> Dict[str, Any]:
    report = {"generated_at_utc": rc.now_iso(), "date_tag": rc.data_tag(),
              "sb3_available": rc.sb3_available(), "gym_available": rc.GYM_AVAILABLE,
              "offline_only": True, "live_wired": False, "agents": {}}
    for agent in AGENTS:
        report["agents"][agent] = train_one(agent, timesteps=timesteps)
    out = rc.DAILY_OUTPUTS / rc.data_tag()
    rc.write_json(out / "rl_retraining_report.json", report)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rep = train_all()
    print("RL training complete (offline, not wired to live):")
    for a, r in rep["agents"].items():
        print(f"  {a}: {r['status']} samples={r['training_samples']} -> {r['model_path']}")
