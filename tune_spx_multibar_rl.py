#!/usr/bin/env python3
"""
tune_spx_multibar_rl.py

Optuna tuner for spx_price_action_layer.py's intraday multibar-gain RL agent.

This tuner trains the SPX/SPY agent repeatedly with different hyperparameters.
The agent learns:

    0 = HOLD
    1 = ENTER_LONG
    2 = EXIT_LONG

The trainer is expected to be:
    spx_price_action_layer.py

And that file should support:
    python spx_price_action_layer.py --train-rl --once

The tuned agent uses:
    - 1m features
    - 5m features
    - 15m features
    - 1h features
    - 4h features
    - RSI
    - SMA5 cross up SMA20
    - SMA20 below SMA5 / SMA5 above SMA20
    - ATR / ATR%
    - MACD / signal / histogram / histogram slope

Run:
    python tune_spx_multibar_rl.py

Useful overrides:
    SPX_RL_TUNER_TRIALS=50 python tune_spx_multibar_rl.py
    SPX_LAYER_DATA_DATE=20260514 python tune_spx_multibar_rl.py
    SPX_RL_TUNER_TIMESTEPS=25000 python tune_spx_multibar_rl.py
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

# Prefer explicit date, otherwise default to today. If you run after midnight
# to tune the prior session, export SPX_LAYER_DATA_DATE or RL_DATA_DATE.
DATA_TAG = os.getenv(
    "SPX_LAYER_DATA_DATE",
    os.getenv("RL_DATA_DATE", os.getenv("RUN_DATA_DATE", TODAY_TAG)),
)

PROJECT_ROOT = Path(__file__).resolve().parent
TRAINER_PATH = PROJECT_ROOT / os.getenv("SPX_RL_TRAINER_FILE", "spx_price_action_layer.py")

METRICS_PATHS = [
    PROJECT_ROOT / "system-data" / DATA_TAG / f"{DATA_TAG}_spx_multibar_gain_rl_metrics.json",
    PROJECT_ROOT / "system-data" / DATA_TAG / "spx_multibar_gain_rl_metrics.json",
    PROJECT_ROOT / "system-data" / DATA_TAG / "latest_spx_multibar_gain_rl_metrics.json",
    PROJECT_ROOT / "system-data" / TODAY_TAG / f"{TODAY_TAG}_spx_multibar_gain_rl_metrics.json",
    PROJECT_ROOT / "system-data" / YESTERDAY_TAG / f"{YESTERDAY_TAG}_spx_multibar_gain_rl_metrics.json",
]

BEST_PARAMS_PATH = PROJECT_ROOT / "system-data" / DATA_TAG / f"{DATA_TAG}_spx_multibar_gain_best_params.json"
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
        root.glob("**/*spx*multibar*rl*metrics*.json"),
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
    Score the SPX multibar RL agent.

    Main objective:
      - profitable long-only entries
      - good MFE capture
      - enough trade activity to learn
      - enough holding period to catch multibar gains
      - avoid heavy giveback/MAE/overtrading
    """
    ok = bool(metrics.get("ok", False))
    rows = metric_float(metrics, "rows")
    timesteps = metric_float(metrics, "timesteps")

    trade_count = metric_float(metrics, "trade_count", "diagnostics.trade_count")
    win_rate = metric_float(metrics, "win_rate", "diagnostics.win_rate")
    avg_pnl = metric_float(metrics, "avg_pnl_pct", "diagnostics.avg_pnl_pct")
    avg_capture = metric_float(metrics, "avg_capture_ratio", "diagnostics.avg_capture_ratio")
    avg_hold = metric_float(metrics, "avg_hold_bars", "diagnostics.avg_hold_bars")

    # Optional richer metrics if you add them later in spx_price_action_layer.py.
    profit_factor = metric_float(metrics, "profit_factor", "diagnostics.profit_factor")
    avg_mfe = metric_float(metrics, "avg_mfe_pct", "diagnostics.avg_mfe_pct")
    avg_mae = abs(metric_float(metrics, "avg_mae_pct", "diagnostics.avg_mae_pct"))
    avg_giveback = abs(metric_float(metrics, "avg_giveback_pct", "diagnostics.avg_giveback_pct"))
    entry_count = metric_float(metrics, "entry_count", "diagnostics.enter_long_count", default=trade_count)
    exit_count = metric_float(metrics, "exit_count", "diagnostics.exit_long_count", default=trade_count)

    # Fallback for current metrics where avg_mfe/giveback are not emitted yet.
    if avg_mfe <= 0:
        avg_mfe = max(avg_pnl, 0.0)
    if profit_factor <= 0:
        profit_factor = max(0.1, (win_rate / max(1.0 - win_rate, 0.05))) if win_rate > 0 else 0.0

    if not ok:
        return DEFAULT_BAD_SCORE

    score = 0.0

    # Quality components.
    score += min(profit_factor, 6.0) * 7.0
    score += win_rate * 22.0
    score += min(max(avg_pnl * 1000.0, -20.0), 35.0) * 1.6
    score += min(max(avg_capture, 0.0), 1.5) * 55.0
    score += min(max(avg_mfe * 1000.0, 0.0), 35.0)

    # Multibar objective: reward holding long enough, but not forever.
    if avg_hold < 4.0:
        score -= (4.0 - avg_hold) * 18.0
    elif avg_hold <= 28.0:
        score += avg_hold * 4.0
    else:
        score += 112.0 - min(avg_hold - 28.0, 20.0) * 2.0

    # Learning / activity.
    score += min(trade_count / 20.0, 10.0)
    score += min(rows / 10000.0, 8.0)

    # Risk penalties.
    score -= min(avg_mae * 900.0, 30.0)
    score -= min(avg_giveback * 900.0, 30.0)

    # Hard-ish penalties for no learning / degenerate policies.
    if rows < 500:
        score -= 120.0
    if trade_count < 10:
        score -= 80.0
    if entry_count < 10:
        score -= 60.0
    if exit_count < max(1.0, entry_count * 0.60):
        score -= 40.0
    if win_rate < 0.35:
        score -= (0.35 - win_rate) * 80.0
    if avg_capture < 0.45:
        score -= (0.45 - avg_capture) * 60.0
    if avg_pnl < 0:
        score -= abs(avg_pnl) * 3000.0

    # Avoid configs that simply train forever with low activity.
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
    env["SPX_LAYER_DATA_DATE"] = DATA_TAG
    env["RUN_DATA_DATE"] = DATA_TAG
    env["RL_DATA_DATE"] = DATA_TAG

    # Make training happen and avoid live trading while tuning.
    env["SPX_LAYER_RL_ENABLED"] = "1"
    env["SPX_LAYER_RL_TRAIN_ON_START"] = "1"
    env["SPX_LAYER_EXECUTE"] = "0"

    # Dataset / run size.
    env["SPX_LAYER_RL_TRAIN_LOOKBACK_DAYS"] = str(
        trial.suggest_int("train_lookback_days", 20, 75)
    )
    env["SPX_LAYER_RL_TIMESTEPS"] = os.getenv(
        "SPX_RL_TUNER_TIMESTEPS",
        str(trial.suggest_categorical("timesteps", [25000, 50000, 75000, 100000])),
    )
    env["SPX_LAYER_RL_MAX_EPISODE_BARS"] = str(
        trial.suggest_categorical("max_episode_bars", [180, 260, 390])
    )

    # PPO hyperparameters.
    env["SPX_LAYER_RL_LEARNING_RATE"] = str(
        trial.suggest_float("learning_rate", 2e-5, 4e-4, log=True)
    )
    env["SPX_LAYER_RL_GAMMA"] = str(
        trial.suggest_float("gamma", 0.985, 0.9995)
    )
    env["SPX_LAYER_RL_N_STEPS"] = str(
        trial.suggest_categorical("n_steps", [256, 512, 1024])
    )
    env["SPX_LAYER_RL_BATCH_SIZE"] = str(
        trial.suggest_categorical("batch_size", [64, 128, 256])
    )
    env["SPX_LAYER_RL_ENT_COEF"] = str(
        trial.suggest_float("ent_coef", 0.005, 0.12, log=True)
    )

    # Reward shaping: entry/exits.
    env["SPX_LAYER_RL_REWARD_SCALE"] = str(
        trial.suggest_float("reward_scale", 60.0, 180.0)
    )
    env["SPX_LAYER_RL_ENTRY_BONUS"] = str(
        trial.suggest_float("entry_bonus", 0.75, 5.0)
    )
    env["SPX_LAYER_RL_BAD_ENTRY_PENALTY"] = str(
        trial.suggest_float("bad_entry_penalty", 0.5, 4.0)
    )
    env["SPX_LAYER_RL_EARLY_MULTIBAR_BONUS_MULT"] = str(
        trial.suggest_float("early_multibar_bonus_mult", 150.0, 650.0)
    )

    # Reward shaping: holding and exit efficiency.
    env["SPX_LAYER_RL_HOLD_WINNER_MULT"] = str(
        trial.suggest_float("hold_winner_mult", 15.0, 100.0)
    )
    env["SPX_LAYER_RL_MFE_CAPTURE_MULT"] = str(
        trial.suggest_float("mfe_capture_mult", 6.0, 35.0)
    )
    env["SPX_LAYER_RL_GIVEBACK_PENALTY_MULT"] = str(
        trial.suggest_float("giveback_penalty_mult", 4.0, 30.0)
    )
    env["SPX_LAYER_RL_MAE_PENALTY_MULT"] = str(
        trial.suggest_float("mae_penalty_mult", 3.0, 22.0)
    )
    env["SPX_LAYER_RL_MIN_QUALITY_HOLD_BARS"] = str(
        trial.suggest_int("min_quality_hold_bars", 3, 14)
    )
    env["SPX_LAYER_RL_FLAT_HOLD_PENALTY"] = str(
        trial.suggest_float("flat_hold_penalty", 0.0005, 0.012)
    )

    # Live gating thresholds to be saved with trial context.
    env["SPX_LAYER_RL_MIN_ENTRY_CONFIDENCE"] = str(
        trial.suggest_float("min_entry_confidence", 0.45, 0.72)
    )
    env["SPX_LAYER_RL_MIN_EXIT_CONFIDENCE"] = str(
        trial.suggest_float("min_exit_confidence", 0.40, 0.70)
    )

    print("\n" + "=" * 100, flush=True)
    print(f"STARTING SPX MULTIBAR RL OPTUNA TRIAL {trial.number}", flush=True)
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
        [sys.executable, "-u", str(TRAINER_PATH), "--train-rl", "--once"],
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
    print(f"SPX MULTIBAR RL TRIAL {trial.number} RESULTS", flush=True)
    print("-" * 100, flush=True)
    print(f"[metrics path] {metrics_path}", flush=True)
    print(json.dumps(metrics, indent=2, default=str), flush=True)
    print(f"[score] {score:.6f}", flush=True)
    print("-" * 100 + "\n", flush=True)

    # Store best-so-far manually for easy loading by shell/run.py.
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
    storage = os.getenv("SPX_RL_TUNER_STORAGE", "")
    study_kwargs: dict[str, Any] = {
        "direction": "maximize",
        "study_name": os.getenv("SPX_RL_TUNER_STUDY_NAME", "spx_multibar_gain_rl_optimizer"),
    }
    if storage:
        study_kwargs["storage"] = storage
        study_kwargs["load_if_exists"] = True

    study = optuna.create_study(**study_kwargs)

    study.optimize(
        objective,
        n_trials=int(os.getenv("SPX_RL_TUNER_TRIALS", "50")),
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
