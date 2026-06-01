"""
tune_sr_rl_agent.py — QuantumQuant
Optuna hyperparameter tuner for the SR RL Agent.
Mirrors tune_rl_intraday.py objective structure but targets SR-specific metrics.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SR_TUNE_N_TRIALS   = int(os.environ.get("SR_TUNE_N_TRIALS",  25))
SR_TUNE_TIMESTEPS  = int(os.environ.get("SR_TUNE_TIMESTEPS", 100_000))
SR_TUNE_STUDY_NAME = os.environ.get("SR_TUNE_STUDY_NAME",   "sr_rl_v1")
SR_TUNE_STORAGE    = os.environ.get("SR_TUNE_STORAGE",       None)
SR_TUNE_BEST_PATH  = os.environ.get("SR_TUNE_BEST_PATH",    "models/sr_rl_best_params.json")
SR_TUNE_METRICS    = os.environ.get("SR_TUNE_METRICS",      "metrics/sr_rl_tune_metrics.json")
SR_DATA_PATH       = os.environ.get("SR_DATA_PATH",         "data/intraday_bars.parquet")

MIN_SR_TRADES      = int(os.environ.get("SR_MIN_TRADES",  2))
MAX_SR_TRADES      = int(os.environ.get("SR_MAX_TRADES", 30))

OBJECTIVE_WEIGHTS = dict(
    profit_factor            = 2.5,
    dip_reclaim_win_rate     = 2.0,
    support_reclaim_win_rate = 1.5,
    avg_capture_ratio        = 1.5,
    win_rate                 = 1.0,
    certainty_equivalent     = 1.25,
    positive_skew            = 0.35,
    # penalties
    avg_mae_pct              = -2.0,
    excess_kurtosis          = -0.40,
    transaction_cost_penalty = -1.00,
    missed_dip_reclaim_count = -0.5,   # per missed setup
)

_best_score = -float("inf")


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
    try:
        import optuna
        import pandas as pd
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as e:
        raise ImportError(f"Missing: {e}")

    from rl_agent_core import get_policy_kwargs
    from sr_rl_agent import SRRLEnv

    # ---- Suggest -------------------------------------------------------
    lr       = trial.suggest_float("learning_rate",  1e-5, 5e-4, log=True)
    n_steps  = trial.suggest_categorical("n_steps",  [512, 1024, 2048])
    batch    = trial.suggest_categorical("batch_size",[64, 128, 256])
    gamma    = trial.suggest_float("gamma",           0.95, 0.999)
    ent_coef = trial.suggest_float("ent_coef",        1e-4, 0.05, log=True)
    clip_r   = trial.suggest_float("clip_range",      0.1, 0.4)

    # Reward weights
    r_sweep  = trial.suggest_float("SR_R_SWEEP_RECLAIM",  0.10, 0.60)
    r_bounce = trial.suggest_float("SR_R_SUPPORT_BOUNCE", 0.10, 0.50)
    r_vwap   = trial.suggest_float("SR_R_VWAP_RECLAIM",   0.05, 0.40)
    r_mfe3r  = trial.suggest_float("SR_R_MFE_3R",         0.30, 1.50)
    p_missed = trial.suggest_float("SR_P_MISSED",        -0.40, -0.05)
    p_no_set = trial.suggest_float("SR_P_NO_SETUP",      -0.50, -0.05)

    os.environ["SR_R_SWEEP_RECLAIM"]   = str(r_sweep)
    os.environ["SR_R_SUPPORT_BOUNCE"]  = str(r_bounce)
    os.environ["SR_R_VWAP_RECLAIM"]    = str(r_vwap)
    os.environ["SR_R_MFE_3R"]          = str(r_mfe3r)
    os.environ["SR_P_MISSED"]          = str(p_missed)
    os.environ["SR_P_NO_SETUP"]        = str(p_no_set)

    # ---- Load data -------------------------------------------------------
    try:
        if SR_DATA_PATH.endswith(".parquet"):
            df = pd.read_parquet(SR_DATA_PATH)
        else:
            df = pd.read_csv(SR_DATA_PATH, parse_dates=["timestamp"])
    except Exception as e:
        logger.error("Data load error: %s", e)
        return -999.0

    symbol   = os.path.basename(SR_DATA_PATH).split("_")[0].upper()
    split    = int(len(df) * 0.85)
    train_df = df.iloc[:split].reset_index(drop=True)
    eval_df  = df.iloc[split:].reset_index(drop=True)

    def make_env(bars):
        def _init():
            e = SRRLEnv(bars, symbol=symbol,
                        replay_path=f"/tmp/sr_tune_{trial.number}.csv")
            return Monitor(e)
        return _init

    train_env = DummyVecEnv([make_env(train_df)])
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs = get_policy_kwargs(),
        learning_rate = lr,
        n_steps       = n_steps,
        batch_size    = min(batch, n_steps),
        gamma         = gamma,
        gae_lambda    = 0.95,
        clip_range    = clip_r,
        ent_coef      = ent_coef,
        n_epochs      = 5,
        verbose       = 0,
    )
    model.learn(total_timesteps=SR_TUNE_TIMESTEPS)

    # ---- Evaluate -------------------------------------------------------
    eval_env = DummyVecEnv([make_env(eval_df)])
    inner    = eval_env.envs[0].env if hasattr(eval_env.envs[0], "env") else eval_env.envs[0]
    obs, _   = inner.reset()
    done     = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = inner.step(int(action))
        done = term or trunc
    m = inner.ep_metrics or {}

    score = _composite_score(m, trial.number)
    _save_if_best({**trial.params, "score": score, "metrics": m}, score)
    return score


def _composite_score(m: Dict, trial_num: int) -> float:
    if not m:
        return -999.0

    n = m.get("n_trades", 0)
    if n < MIN_SR_TRADES:
        return -50.0 - (MIN_SR_TRADES - n) * 5.0

    overtrade_pen = max(0.0, n - MAX_SR_TRADES) * 1.5

    score = 0.0
    for metric, weight in OBJECTIVE_WEIGHTS.items():
        val = float(m.get(metric, 0.0) or 0.0)
        score += weight * val

    score -= overtrade_pen

    # Drawdown penalty
    mae = m.get("avg_mae_pct", 0.0)
    if mae and mae > 0.025:
        score -= (mae - 0.025) * 20.0

    logger.info("SR Trial %d: score=%.4f  trades=%d  PF=%.2f  DR_WR=%.1f%%",
                trial_num, score, n,
                m.get("profit_factor", 0),
                m.get("dip_reclaim_win_rate", 0) * 100)
    return score


def _save_if_best(params: Dict, score: float):
    global _best_score
    if score > _best_score:
        _best_score = score
        os.makedirs(
            os.path.dirname(SR_TUNE_BEST_PATH) if os.path.dirname(SR_TUNE_BEST_PATH) else ".",
            exist_ok=True,
        )
        with open(SR_TUNE_BEST_PATH, "w") as f:
            json.dump(params, f, indent=2, default=str)
        logger.info("SR new best → %s  (score=%.4f)", SR_TUNE_BEST_PATH, score)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_tuning():
    try:
        import optuna
    except ImportError:
        raise ImportError("pip install optuna")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name     = SR_TUNE_STUDY_NAME,
        storage        = SR_TUNE_STORAGE,
        direction      = "maximize",
        load_if_exists = True,
        sampler        = optuna.samplers.TPESampler(seed=42),
        pruner         = optuna.pruners.MedianPruner(n_startup_trials=4),
    )

    logger.info("SR tuning: %d trials × %d steps", SR_TUNE_N_TRIALS, SR_TUNE_TIMESTEPS)
    study.optimize(objective, n_trials=SR_TUNE_N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    logger.info("=" * 60)
    logger.info("SR BEST TRIAL #%d  score=%.4f", best.number, best.value)
    for k, v in best.params.items():
        logger.info("  %-35s : %s", k, v)

    summary = dict(best_score=best.value, best_params=best.params, n_trials=len(study.trials))
    os.makedirs(
        os.path.dirname(SR_TUNE_METRICS) if os.path.dirname(SR_TUNE_METRICS) else ".",
        exist_ok=True,
    )
    with open(SR_TUNE_METRICS, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("SR tune metrics → %s", SR_TUNE_METRICS)
    return study


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_tuning()