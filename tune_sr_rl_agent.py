#!/usr/bin/env python3
"""
tune_sr_rl_agent.py

Optuna tuner for sr_rl_agent.py.

This tuner repeatedly trains the support/resistance RL agent with different
hyperparameters and reward-shaping values, then scores the resulting metrics.

Expected trainer:
    sr_rl_agent.py

Expected training command:
    python sr_rl_agent.py --train

Optional live command after training:
    python sr_rl_agent.py --execute

Run:
    python tune_sr_rl_agent.py

Useful overrides:
    SR_RL_TUNER_TRIALS=50 python tune_sr_rl_agent.py
    SR_RL_DATA_DATE=20260514 python tune_sr_rl_agent.py
    SR_RL_TUNER_TIMESTEPS=25000 python tune_sr_rl_agent.py
    SR_RL_TRAINER_FILE=sr_rl_agent.py python tune_sr_rl_agent.py

This tuner is long-only by default. It passes env vars that should make the
agent learn:
    - WAIT / HOLD
    - ENTER_LONG near support, VWAP reclaim, or breakout through resistance
    - EXIT_LONG near resistance, failed support, giveback, or weak structure

If sr_rl_agent.py does not yet consume all env vars below, it will still run;
those variables are harmless until you wire them into the agent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import optuna


# =============================================================================
# Date / paths
# =============================================================================

ET = ZoneInfo("America/New_York")
TODAY_TAG = datetime.now(ET).strftime("%Y%m%d")
YESTERDAY_TAG = (datetime.now(ET) - timedelta(days=1)).strftime("%Y%m%d")

DATA_TAG = os.getenv(
    "SR_RL_DATA_DATE",
    os.getenv("RL_DATA_DATE", os.getenv("RUN_DATA_DATE", TODAY_TAG)),
)

PROJECT_ROOT = Path(__file__).resolve().parent
TRAINER_PATH = PROJECT_ROOT / os.getenv("SR_RL_TRAINER_FILE", "sr_rl_agent.py")

METRICS_PATHS = [
    PROJECT_ROOT / "system-data" / DATA_TAG / f"{DATA_TAG}_sr_rl_training_metrics.json",
    PROJECT_ROOT / "system-data" / DATA_TAG / f"{DATA_TAG}_sr_rl_agent_metrics.json",
    PROJECT_ROOT / "system-data" / DATA_TAG / "sr_rl_training_metrics.json",
    PROJECT_ROOT / "system-data" / DATA_TAG / "sr_rl_agent_metrics.json",
    PROJECT_ROOT / "system-data" / DATA_TAG / "latest_sr_rl_metrics.json",
    PROJECT_ROOT / "system-data" / TODAY_TAG / f"{TODAY_TAG}_sr_rl_training_metrics.json",
    PROJECT_ROOT / "system-data" / YESTERDAY_TAG / f"{YESTERDAY_TAG}_sr_rl_training_metrics.json",
]

BEST_PARAMS_PATH = PROJECT_ROOT / "system-data" / DATA_TAG / f"{DATA_TAG}_sr_rl_best_params.json"
BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_BAD_SCORE = -9999.0


# =============================================================================
# Metrics helpers
# =============================================================================

def find_metrics_file() -> Path | None:
    for path in METRICS_PATHS:
        if path.exists():
            return path

    root = PROJECT_ROOT / "system-data"
    if not root.exists():
        return None

    candidates = sorted(
        list(root.glob("**/*sr*rl*metrics*.json"))
        + list(root.glob("**/*sr*training*metrics*.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_metrics() -> tuple[dict[str, Any], Path | None]:
    path = find_metrics_file()
    if path is None:
        return {}, None

    try:
        return json.loads(path.read_text(encoding="utf-8")), path
    except Exception as exc:
        print(f"[metrics] failed to read {path}: {exc}", flush=True)
        return {}, path


def nested_get(payload: dict[str, Any], keys: list[str], default: Any = 0.0) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def metric_float(metrics: dict[str, Any], *paths_or_names: str, default: float = 0.0) -> float:
    for item in paths_or_names:
        if "." in item:
            value = nested_get(metrics, item.split("."), None)
        else:
            value = metrics.get(item, None)

        if value is None:
            continue

        try:
            out = float(value)
            if out == out and out not in (float("inf"), float("-inf")):
                return out
        except Exception:
            continue

    return default


def score_metrics(metrics: dict[str, Any]) -> float:
    """
    Score the SR RL agent.

    Main objective:
      - enter early near support/reclaim/breakout setups
      - exit efficiently near resistance or when structure fails
      - capture MFE while limiting giveback/MAE
      - avoid overtrading/chop
      - produce enough trades to learn
    """
    ok = bool(metrics.get("ok", True))
    rows = metric_float(metrics, "rows", "training_rows", "dataset_rows")
    timesteps = metric_float(metrics, "timesteps", "total_timesteps")

    trade_count = metric_float(metrics, "trade_count", "diagnostics.trade_count", "num_trades")
    entry_count = metric_float(metrics, "entry_count", "diagnostics.entry_count", "enter_long_count", default=trade_count)
    exit_count = metric_float(metrics, "exit_count", "diagnostics.exit_count", "exit_long_count", default=trade_count)

    win_rate = metric_float(metrics, "win_rate", "diagnostics.win_rate")
    profit_factor = metric_float(metrics, "profit_factor", "diagnostics.profit_factor")
    avg_pnl = metric_float(metrics, "avg_pnl_pct", "avg_return", "diagnostics.avg_pnl_pct")
    total_return = metric_float(metrics, "total_return", "total_pnl_pct", "diagnostics.total_return")
    sharpe = metric_float(metrics, "sharpe", "sharpe_ratio", "diagnostics.sharpe")
    max_drawdown = abs(metric_float(metrics, "max_drawdown", "max_drawdown_pct", "diagnostics.max_drawdown"))

    avg_mfe = metric_float(metrics, "avg_mfe_pct", "diagnostics.avg_mfe_pct")
    avg_mae = abs(metric_float(metrics, "avg_mae_pct", "diagnostics.avg_mae_pct"))
    avg_giveback = abs(metric_float(metrics, "avg_giveback_pct", "diagnostics.avg_giveback_pct"))
    avg_capture = metric_float(metrics, "avg_capture_ratio", "diagnostics.avg_capture_ratio")
    avg_hold = metric_float(metrics, "avg_hold_bars", "avg_bars_held", "diagnostics.avg_hold_bars")

    support_entry_quality = metric_float(
        metrics,
        "support_entry_quality",
        "diagnostics.support_entry_quality",
        "avg_support_entry_score",
        default=0.0,
    )
    resistance_exit_quality = metric_float(
        metrics,
        "resistance_exit_quality",
        "diagnostics.resistance_exit_quality",
        "avg_resistance_exit_score",
        default=0.0,
    )
    chop_rate = metric_float(metrics, "chop_rate", "trade_churn_rate", "diagnostics.chop_rate", default=0.0)

    if avg_mfe <= 0:
        avg_mfe = max(avg_pnl, 0.0)
    if avg_capture <= 0 and avg_mfe > 0 and avg_pnl > 0:
        avg_capture = avg_pnl / max(avg_mfe, 1e-9)
    if profit_factor <= 0 and win_rate > 0:
        profit_factor = max(0.1, win_rate / max(1.0 - win_rate, 0.05))

    if not ok:
        return DEFAULT_BAD_SCORE

    score = 0.0

    # Core profitability / quality.
    score += min(profit_factor, 6.0) * 9.0
    score += win_rate * 24.0
    score += min(max(avg_pnl * 1000.0, -25.0), 40.0) * 1.8
    score += min(max(total_return * 100.0, -30.0), 60.0)
    score += min(max(sharpe, -2.0), 5.0) * 8.0

    # SR-specific behavior.
    score += min(max(support_entry_quality, 0.0), 1.0) * 25.0
    score += min(max(resistance_exit_quality, 0.0), 1.0) * 25.0

    # Exit efficiency.
    score += min(max(avg_capture, 0.0), 1.5) * 45.0
    score += min(max(avg_mfe * 1000.0, 0.0), 35.0)

    # Multibar hold quality.
    if avg_hold < 3.0:
        score -= (3.0 - avg_hold) * 18.0
    elif avg_hold <= 32.0:
        score += avg_hold * 3.0
    else:
        score += 96.0 - min(avg_hold - 32.0, 25.0) * 2.0

    # Learning/activity.
    score += min(trade_count / 20.0, 10.0)
    score += min(rows / 10000.0, 8.0)

    # Risk/chop penalties.
    score -= min(max_drawdown * 150.0, 45.0)
    score -= min(avg_mae * 900.0, 35.0)
    score -= min(avg_giveback * 900.0, 35.0)
    score -= min(chop_rate * 35.0, 35.0)

    # Degenerate policy penalties.
    if rows < 500:
        score -= 120.0
    if trade_count < 10:
        score -= 80.0
    if entry_count < 10:
        score -= 60.0
    if exit_count < max(1.0, entry_count * 0.60):
        score -= 45.0
    if win_rate < 0.35:
        score -= (0.35 - win_rate) * 90.0
    if avg_capture < 0.40:
        score -= (0.40 - avg_capture) * 65.0
    if avg_pnl < 0:
        score -= abs(avg_pnl) * 3500.0

    if timesteps > 0 and trade_count / max(timesteps, 1.0) < 0.00005:
        score -= 25.0

    return float(score)


# =============================================================================
# Trial setup
# =============================================================================

def objective(trial: optuna.Trial) -> float:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Lock date.
    env["SR_RL_DATA_DATE"] = DATA_TAG
    env["RUN_DATA_DATE"] = DATA_TAG
    env["RL_DATA_DATE"] = DATA_TAG

    # Long-only production safety.
    env["SR_RL_LONG_ONLY"] = "1"
    env["SR_RL_ALLOW_SHORTS"] = "0"
    env["SR_RL_EXECUTE"] = "0"
    env["SR_RL_TRAIN"] = "1"

    # Dataset / run size.
    env["SR_RL_TRAIN_LOOKBACK_DAYS"] = str(
        trial.suggest_int("train_lookback_days", 20, 90)
    )
    env["SR_RL_TIMESTEPS"] = os.getenv(
        "SR_RL_TUNER_TIMESTEPS",
        str(trial.suggest_categorical("timesteps", [25000, 50000, 75000, 100000])),
    )
    env["SR_RL_MAX_EPISODE_BARS"] = str(
        trial.suggest_categorical("max_episode_bars", [180, 260, 390])
    )

    # PPO hyperparameters.
    env["SR_RL_LEARNING_RATE"] = str(
        trial.suggest_float("learning_rate", 2e-5, 5e-4, log=True)
    )
    env["SR_RL_GAMMA"] = str(
        trial.suggest_float("gamma", 0.985, 0.9995)
    )
    env["SR_RL_N_STEPS"] = str(
        trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
    )
    env["SR_RL_BATCH_SIZE"] = str(
        trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    )
    env["SR_RL_ENT_COEF"] = str(
        trial.suggest_float("ent_coef", 0.003, 0.12, log=True)
    )

    # Support/resistance detection parameters.
    env["SR_RL_PIVOT_WINDOW"] = str(
        trial.suggest_int("pivot_window", 3, 9)
    )
    env["SR_RL_LEVEL_MERGE_PCT"] = str(
        trial.suggest_float("level_merge_pct", 0.0015, 0.012)
    )
    env["SR_RL_SUPPORT_TOUCH_BUFFER_PCT"] = str(
        trial.suggest_float("support_touch_buffer_pct", 0.0015, 0.012)
    )
    env["SR_RL_RESISTANCE_TOUCH_BUFFER_PCT"] = str(
        trial.suggest_float("resistance_touch_buffer_pct", 0.0015, 0.012)
    )
    env["SR_RL_BREAKOUT_BUFFER_PCT"] = str(
        trial.suggest_float("breakout_buffer_pct", 0.001, 0.008)
    )

    # Reward shaping: entries.
    env["SR_RL_REWARD_SCALE"] = str(
        trial.suggest_float("reward_scale", 60.0, 180.0)
    )
    env["SR_RL_SUPPORT_ENTRY_BONUS"] = str(
        trial.suggest_float("support_entry_bonus", 0.75, 6.0)
    )
    env["SR_RL_RECLAIM_ENTRY_BONUS"] = str(
        trial.suggest_float("reclaim_entry_bonus", 0.50, 5.0)
    )
    env["SR_RL_BREAKOUT_ENTRY_BONUS"] = str(
        trial.suggest_float("breakout_entry_bonus", 0.50, 5.0)
    )
    env["SR_RL_BAD_ENTRY_PENALTY"] = str(
        trial.suggest_float("bad_entry_penalty", 0.5, 5.0)
    )

    # Reward shaping: holding and exits.
    env["SR_RL_HOLD_WINNER_MULT"] = str(
        trial.suggest_float("hold_winner_mult", 10.0, 100.0)
    )
    env["SR_RL_RESISTANCE_EXIT_BONUS"] = str(
        trial.suggest_float("resistance_exit_bonus", 0.75, 7.0)
    )
    env["SR_RL_MFE_CAPTURE_MULT"] = str(
        trial.suggest_float("mfe_capture_mult", 5.0, 40.0)
    )
    env["SR_RL_GIVEBACK_PENALTY_MULT"] = str(
        trial.suggest_float("giveback_penalty_mult", 4.0, 35.0)
    )
    env["SR_RL_MAE_PENALTY_MULT"] = str(
        trial.suggest_float("mae_penalty_mult", 3.0, 24.0)
    )
    env["SR_RL_MIN_QUALITY_HOLD_BARS"] = str(
        trial.suggest_int("min_quality_hold_bars", 3, 16)
    )
    env["SR_RL_FLAT_HOLD_PENALTY"] = str(
        trial.suggest_float("flat_hold_penalty", 0.0005, 0.014)
    )

    # Live gating thresholds.
    env["SR_RL_MIN_ENTRY_CONFIDENCE"] = str(
        trial.suggest_float("min_entry_confidence", 0.45, 0.75)
    )
    env["SR_RL_MIN_EXIT_CONFIDENCE"] = str(
        trial.suggest_float("min_exit_confidence", 0.40, 0.72)
    )

    print("\n" + "=" * 100, flush=True)
    print(f"STARTING SR RL OPTUNA TRIAL {trial.number}", flush=True)
    print("=" * 100, flush=True)
    print(f"[data tag] {DATA_TAG}", flush=True)
    print(f"[trainer] {TRAINER_PATH}", flush=True)
    print("[trial params]", flush=True)
    print(json.dumps(trial.params, indent=2), flush=True)
    print("=" * 100 + "\n", flush=True)

    if not TRAINER_PATH.exists():
        print(f"[trial {trial.number}] trainer not found: {TRAINER_PATH}", flush=True)
        return DEFAULT_BAD_SCORE

    # Delete stale metrics before trial to avoid scoring an old run.
    for path in METRICS_PATHS:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass

    process = subprocess.Popen(
        [sys.executable, "-u", str(TRAINER_PATH), "--train"],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=None,
        stderr=None,
        text=True,
    )
    process.wait()

    print("\n" + "-" * 100, flush=True)
    print(f"[trial {trial.number}] trainer returncode={process.returncode}", flush=True)
    print("-" * 100, flush=True)

    if process.returncode != 0:
        print(f"[trial {trial.number}] trainer exited non-zero", flush=True)
        return DEFAULT_BAD_SCORE

    metrics, metrics_path = load_metrics()
    if not metrics:
        print(f"[trial {trial.number}] metrics missing", flush=True)
        for path in METRICS_PATHS:
            print(f"  - {path}", flush=True)
        return DEFAULT_BAD_SCORE

    score = score_metrics(metrics)

    result = {
        "trial_number": trial.number,
        "score": score,
        "params": trial.params,
        "metrics_path": str(metrics_path),
        "metrics": metrics,
        "data_tag": DATA_TAG,
        "trainer": str(TRAINER_PATH),
        "created_at_utc": datetime.now(tz=ZoneInfo("UTC")).isoformat(),
    }

    print("\n" + "-" * 100, flush=True)
    print(f"SR RL TRIAL {trial.number} RESULTS", flush=True)
    print("-" * 100, flush=True)
    print(f"[metrics path] {metrics_path}", flush=True)
    print(json.dumps(metrics, indent=2, default=str), flush=True)
    print(f"[score] {score:.6f}", flush=True)
    print("-" * 100 + "\n", flush=True)

    existing = {}
    if BEST_PARAMS_PATH.exists():
        try:
            existing = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    if not existing or score > float(existing.get("score", DEFAULT_BAD_SCORE)):
        BEST_PARAMS_PATH.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"[best] updated {BEST_PARAMS_PATH}", flush=True)

    return float(score)


def main() -> None:
    storage = os.getenv("SR_RL_TUNER_STORAGE", "")
    study_kwargs: dict[str, Any] = {
        "direction": "maximize",
        "study_name": os.getenv("SR_RL_TUNER_STUDY_NAME", "sr_rl_agent_optimizer"),
    }
    if storage:
        study_kwargs["storage"] = storage
        study_kwargs["load_if_exists"] = True

    study = optuna.create_study(**study_kwargs)

    study.optimize(
        objective,
        n_trials=int(os.getenv("SR_RL_TUNER_TRIALS", "50")),
        show_progress_bar=True,
        catch=(Exception,),
    )

    print("\nBEST PARAMS:")
    print(json.dumps(study.best_params, indent=2))

    print("\nBEST SCORE:")
    print(study.best_value)

    final = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study_name": study.study_name,
        "data_tag": DATA_TAG,
        "created_at_utc": datetime.now(tz=ZoneInfo("UTC")).isoformat(),
    }
    BEST_PARAMS_PATH.write_text(json.dumps(final, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n[saved] {BEST_PARAMS_PATH}")


if __name__ == "__main__":
    main()
