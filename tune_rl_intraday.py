"""
tune_rl_intraday.py — QuantumQuant
Optuna hyperparameter tuner for the Intraday RL Agent.

Optimises:
  MAXIMISE  profit_factor, dip_reclaim_win_rate, support_reclaim_win_rate,
            avg_capture_ratio, sharpe, total_pnl
  MINIMISE  avg_mae_pct, premature exits, missed setups, excessive drawdown

Penalises:  inactive models (n_trades < threshold), overtrading, low count.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TUNE_N_TRIALS       = int(os.environ.get("TUNE_N_TRIALS",  30))
TUNE_TIMESTEPS      = int(os.environ.get("TUNE_TIMESTEPS", 150_000))   # short run per trial
TUNE_STUDY_NAME     = os.environ.get("TUNE_STUDY_NAME",   "rl_intraday_v1")
TUNE_STORAGE        = os.environ.get("TUNE_STORAGE",       None)        # e.g. sqlite:///tune.db
TUNE_BEST_PATH      = os.environ.get("TUNE_BEST_PATH",     "models/rl_intraday_best_params.json")
TUNE_METRICS_PATH   = os.environ.get("TUNE_METRICS_PATH",  "metrics/rl_intraday_tune_metrics.json")
RL_DATA_PATH        = os.environ.get("RL_DATA_PATH",       "data/intraday_bars.parquet")

MIN_TRADES_PER_EPISODE = int(os.environ.get("MIN_TRADES_PER_EPISODE", 3))
MAX_TRADES_PER_EPISODE = int(os.environ.get("MAX_TRADES_PER_EPISODE", 25))

OBJECTIVE_WEIGHTS = dict(
    profit_factor            = 2.0,
    dip_reclaim_win_rate     = 2.0,
    support_reclaim_win_rate = 1.5,
    avg_capture_ratio        = 1.5,
    sharpe                   = 1.0,
    certainty_equivalent     = 1.25,
    positive_skew            = 0.35,
    total_pnl                = 0.50,
    # penalties (negative contribution)
    avg_mae_pct              = -1.5,
    excess_kurtosis          = -0.40,
    transaction_cost_penalty = -1.00,
)


def _oops_metric(m: Dict, name: str, default: float = 0.0) -> float:
    try:
        return float(m.get(name, default) or default)
    except Exception:
        return default

def _ensure_oops_risk_metrics(m: Dict) -> Dict:
    """Normalize optional OOPS risk metrics so old env metrics still score safely."""
    out = dict(m or {})
    out.setdefault("certainty_equivalent", out.get("oops_certainty_equivalent", 0.0))
    out.setdefault("positive_skew", max(0.0, _oops_metric(out, "oops_skew", out.get("skew", 0.0))))
    out.setdefault("excess_kurtosis", max(0.0, _oops_metric(out, "oops_kurtosis", out.get("kurtosis", 3.0)) - 3.0))
    out.setdefault("transaction_cost_penalty", _oops_metric(out, "oops_bid_ask_penalty", 0.0))
    return out


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def objective(trial) -> float:
    """Optuna objective — returns scalar composite score."""
    try:
        import optuna
        import pandas as pd
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}")

    from rl_agent_core import get_policy_kwargs
    from rl_intraday_structure_trainer import (
        IntradayRLEnv, load_bars, RL_REPLAY_PATH,
    )

    # ---- Suggest hyperparameters ----------------------------------------
    lr        = trial.suggest_float("learning_rate",  1e-5, 5e-4, log=True)
    n_steps   = trial.suggest_categorical("n_steps",  [512, 1024, 2048])
    batch     = trial.suggest_categorical("batch_size",[64, 128, 256])
    gamma     = trial.suggest_float("gamma",           0.95, 0.999)
    ent_coef  = trial.suggest_float("ent_coef",        1e-4, 0.05, log=True)
    clip_r    = trial.suggest_float("clip_range",      0.1, 0.4)
    gae_lam   = trial.suggest_float("gae_lambda",      0.90, 0.99)

    # Reward-shape weights
    r_sup_touch  = trial.suggest_float("R_SUPPORT_TOUCH",  0.05, 0.50)
    r_rec_vwap   = trial.suggest_float("R_RECLAIM_VWAP",   0.05, 0.40)
    r_vol_spike  = trial.suggest_float("R_VOLUME_SPIKE",   0.05, 0.40)
    r_abs        = trial.suggest_float("R_ABSORPTION",     0.05, 0.40)
    r_lwr        = trial.suggest_float("R_LOWER_WICK",     0.02, 0.30)
    r_mfe3r      = trial.suggest_float("R_MFE_3R",         0.30, 1.50)
    p_missed     = trial.suggest_float("P_MISSED_SETUP",   -0.40, -0.05)
    p_prem       = trial.suggest_float("P_PREMATURE_ENTRY",-0.60, -0.05)
    p_no_vol     = trial.suggest_float("P_NO_VOL_CONFIRM", -0.40, -0.05)

    # Inject reward weights via env vars
    os.environ["R_SUPPORT_TOUCH"]    = str(r_sup_touch)
    os.environ["R_RECLAIM_VWAP"]     = str(r_rec_vwap)
    os.environ["R_VOLUME_SPIKE"]     = str(r_vol_spike)
    os.environ["R_ABSORPTION"]       = str(r_abs)
    os.environ["R_LOWER_WICK"]       = str(r_lwr)
    os.environ["R_MFE_3R"]           = str(r_mfe3r)
    os.environ["P_MISSED_SETUP"]     = str(p_missed)
    os.environ["P_PREMATURE_ENTRY"]  = str(p_prem)
    os.environ["P_NO_VOL_CONFIRM"]   = str(p_no_vol)

    # ---- Build env -------------------------------------------------------
    try:
        df     = load_bars(RL_DATA_PATH)
    except Exception as e:
        logger.error("Cannot load data: %s", e)
        return -999.0

    symbol   = os.path.basename(RL_DATA_PATH).split("_")[0].upper()
    n        = len(df)
    split    = int(n * 0.85)
    train_df = df.iloc[:split].reset_index(drop=True)
    eval_df  = df.iloc[split:].reset_index(drop=True)

    def make_env(bars):
        def _init():
            env = IntradayRLEnv(bars, symbol=symbol,
                                replay_path=f"/tmp/tune_replay_{trial.number}.csv")
            return Monitor(env)
        return _init

    train_env = DummyVecEnv([make_env(train_df)])
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # ---- Train -----------------------------------------------------------
    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs = get_policy_kwargs(),
        learning_rate = lr,
        n_steps       = n_steps,
        batch_size    = min(batch, n_steps),
        gamma         = gamma,
        gae_lambda    = gae_lam,
        clip_range    = clip_r,
        ent_coef      = ent_coef,
        n_epochs      = 5,
        verbose       = 0,
    )
    model.learn(total_timesteps=TUNE_TIMESTEPS)

    # ---- Evaluate --------------------------------------------------------
    eval_env = DummyVecEnv([make_env(eval_df)])
    metrics  = _evaluate(model, eval_env, symbol, eval_df)

    # ---- Score -----------------------------------------------------------
    score = _composite_score(metrics, trial.number)

    # Save params for this trial
    params = {k: v for k, v in trial.params.items()}
    params["score"]   = score
    params["metrics"] = metrics
    _save_if_best(params, score)

    return score


def _evaluate(model, eval_env, symbol: str, eval_df) -> Dict:
    from rl_intraday_structure_trainer import IntradayRLEnv, RL_REPLAY_PATH
    import pandas as pd

    env = IntradayRLEnv(eval_df, symbol=symbol, replay_path=f"/tmp/tune_eval_replay.csv")
    obs, _ = env.reset()
    done   = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
    return env.episode_metrics or {}


def _composite_score(m: Dict, trial_num: int) -> float:
    if not m:
        return -999.0

    m = _ensure_oops_risk_metrics(m)
    n_trades = m.get("n_trades", 0)

    # Inactivity penalty
    if n_trades < MIN_TRADES_PER_EPISODE:
        logger.info("Trial %d: inactive (trades=%d) → penalised", trial_num, n_trades)
        return -50.0 - (MIN_TRADES_PER_EPISODE - n_trades) * 5.0

    # Overtrading penalty
    overtrade_pen = 0.0
    if n_trades > MAX_TRADES_PER_EPISODE:
        overtrade_pen = (n_trades - MAX_TRADES_PER_EPISODE) * 1.0

    score = 0.0
    for metric, weight in OBJECTIVE_WEIGHTS.items():
        val = m.get(metric, 0.0)
        if val is None:
            val = 0.0
        score += weight * float(val)

    score -= overtrade_pen

    # Drawdown penalty
    mae = m.get("avg_mae_pct", 0.0)
    if mae > 0.03:
        score -= (mae - 0.03) * 20.0

    logger.info("Trial %d: score=%.4f  trades=%d  PF=%.2f  DR_WR=%.1f%%",
                trial_num, score, n_trades,
                m.get("profit_factor", 0),
                m.get("dip_reclaim_win_rate", 0) * 100)
    return score


# ---------------------------------------------------------------------------
# Best params persistence
# ---------------------------------------------------------------------------

_best_score = -float("inf")

def _save_if_best(params: Dict, score: float):
    global _best_score
    if score > _best_score:
        _best_score = score
        os.makedirs(os.path.dirname(TUNE_BEST_PATH) if os.path.dirname(TUNE_BEST_PATH) else ".", exist_ok=True)
        with open(TUNE_BEST_PATH, "w") as f:
            json.dump(params, f, indent=2, default=str)
        logger.info("New best params saved → %s  (score=%.4f)", TUNE_BEST_PATH, score)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_tuning():
    try:
        import optuna
    except ImportError:
        raise ImportError("optuna required: pip install optuna")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name  = TUNE_STUDY_NAME,
        storage     = TUNE_STORAGE,
        direction   = "maximize",
        load_if_exists = True,
        sampler     = optuna.samplers.TPESampler(seed=42),
        pruner      = optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    logger.info("Starting Optuna tuning: %d trials  ×  %d steps", TUNE_N_TRIALS, TUNE_TIMESTEPS)
    study.optimize(objective, n_trials=TUNE_N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    logger.info("=" * 60)
    logger.info("BEST TRIAL  #%d  score=%.4f", best.number, best.value)
    for k, v in best.params.items():
        logger.info("  %-35s : %s", k, v)

    # Save final metrics
    summary = dict(best_score=best.value, best_params=best.params, n_trials=len(study.trials))
    os.makedirs(os.path.dirname(TUNE_METRICS_PATH) if os.path.dirname(TUNE_METRICS_PATH) else ".", exist_ok=True)
    with open(TUNE_METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Tune metrics → %s", TUNE_METRICS_PATH)

    return study


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_tuning()