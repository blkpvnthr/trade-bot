#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import optuna


# =============================================================================
# Date / paths
# =============================================================================

ET = ZoneInfo("America/New_York")

try:
    import market_context_builder
    TODAY_TAG = market_context_builder.TODAY_TAG
except Exception:
    TODAY_TAG = datetime.now(ET).strftime("%Y%m%d")

YESTERDAY_TAG = (datetime.now(ET) - timedelta(days=1)).strftime("%Y%m%d")

# Prefer explicit override. Default to yesterday because this often runs after
# midnight while training the previous regular session.
DATA_TAG = os.getenv("RL_DATA_DATE", YESTERDAY_TAG)

PROJECT_ROOT = Path(__file__).resolve().parent
TRAINER_PATH = PROJECT_ROOT / "rl_intraday_structure_trainer.py"

METRICS_PATHS = [
    PROJECT_ROOT / "system-data" / DATA_TAG / f"{DATA_TAG}_rl_intraday_training_metrics.json",
    PROJECT_ROOT / "system-data" / DATA_TAG / f"{DATA_TAG}_latest_rl_metrics.json",
    PROJECT_ROOT / "system-data" / DATA_TAG / f"{DATA_TAG}_rl_intraday_metrics.json",
    PROJECT_ROOT / "system-data" / DATA_TAG / "latest_rl_metrics.json",
    PROJECT_ROOT / "system-data" / TODAY_TAG / f"{TODAY_TAG}_rl_intraday_training_metrics.json",
    PROJECT_ROOT / "system-data" / YESTERDAY_TAG / f"{YESTERDAY_TAG}_rl_intraday_training_metrics.json",
    PROJECT_ROOT / "system-data" / "latest_rl_metrics.json",
    PROJECT_ROOT / "latest_rl_metrics.json",
]

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
        root.glob("**/*rl*metrics*.json"),
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
            f = float(value)
            if f == f and f not in (float("inf"), float("-inf")):
                return f
        except Exception:
            continue

    return default


def score_metrics(metrics: dict[str, Any]) -> float:
    diagnostics = metrics.get("diagnostics", {})
    tq = diagnostics.get("trade_quality", {}) if isinstance(diagnostics, dict) else {}

    profit_factor = metric_float(tq, "profit_factor")
    win_rate = metric_float(tq, "win_rate")
    avg_hold_bars = metric_float(tq, "avg_hold_bars")
    avg_capture_ratio = metric_float(tq, "avg_capture_ratio")
    avg_mfe_pct = metric_float(tq, "avg_mfe_pct")
    avg_mae_pct = metric_float(tq, "avg_mae_pct")
    avg_giveback = metric_float(tq, "avg_giveback_from_mfe_pct")
    round_trips = metric_float(tq, "round_trips")
    total_pnl = metric_float(metrics, "diagnostics.total_realized_pnl")

    long_entries = metric_float(
        tq,
        "long_entries",
        default=metric_float(metrics, "diagnostics.enter_long_count"),
    )
    long_exits = metric_float(
        tq,
        "long_exits",
        default=metric_float(metrics, "diagnostics.exit_long_count"),
    )

    # ADD_LONG / pyramiding metrics emitted by the v2 trainer.
    add_longs = metric_float(
        tq,
        "add_long_count",
        "add_longs",
        default=metric_float(metrics, "diagnostics.add_long_count"),
    )
    add_long_ratio = metric_float(
        tq,
        "add_long_ratio",
        default=add_longs / max(long_entries, 1.0),
    )
    avg_add_long_reward = metric_float(
        tq,
        "avg_add_long_reward",
        "avg_add_reward",
        default=metric_float(metrics, "diagnostics.avg_add_long_reward"),
    )
    avg_add_long_pnl = metric_float(
        tq,
        "avg_add_long_pnl",
        "avg_pyramid_pnl",
        "avg_add_pnl",
        default=metric_float(metrics, "diagnostics.avg_add_long_pnl"),
    )

    premature_exit_count = metric_float(
        tq,
        "premature_exit_count",
        default=metric_float(metrics, "diagnostics.premature_exit_count"),
    )
    scalp_exit_ratio = metric_float(
        tq,
        "scalp_exit_ratio",
        default=premature_exit_count / max(round_trips, 1.0),
    )

    avg_continuation_reward = metric_float(tq, "avg_continuation_reward")

    # Defensive compatibility checks: long-only trainer should not emit shorts.
    short_entries = metric_float(
        tq,
        "short_entries",
        default=metric_float(metrics, "diagnostics.enter_short_count"),
    )
    short_exits = metric_float(
        tq,
        "short_exits",
        default=metric_float(metrics, "diagnostics.exit_short_count"),
    )

    # Cap unstable pieces so tiny denominators cannot dominate.
    pf_component = min(profit_factor, 8.0) * 5.0
    pnl_component = min(max(total_pnl / 1000.0, -10.0), 15.0) * 2.2
    activity_component = min(round_trips / 300.0, 6.0)

    # Target band for pyramiding: it should appear frequently enough to learn,
    # but not fire on every trade.
    if add_long_ratio <= 0:
        add_component = -25.0
    elif add_long_ratio < 0.05:
        add_component = -10.0 + add_long_ratio * 100.0
    elif add_long_ratio <= 0.55:
        add_component = add_long_ratio * 45.0
    else:
        add_component = 24.75 - (add_long_ratio - 0.55) * 35.0

    # Continuation objective:
    # - profitable long-only trading
    # - longer holds
    # - better MFE capture
    # - actual useful pyramiding
    # - lower MAE/giveback
    # - fewer scalp exits
    score = (
        pf_component
        + win_rate * 16.0
        + avg_capture_ratio * 52.0
        + min(avg_mfe_pct * 800.0, 26.0)
        + min(avg_hold_bars, 18.0) * 4.5
        + pnl_component
        + activity_component
        + add_component
        + min(max(avg_add_long_reward, 0.0), 12.0) * 2.0
        + min(max(avg_add_long_pnl, 0.0), 12.0) * 2.0
        + min(max(avg_continuation_reward, 0.0), 10.0) * 1.5
        - min(avg_mae_pct * 700.0, 26.0)
        - min(avg_giveback * 700.0, 26.0)
        - scalp_exit_ratio * 40.0
    )

    if round_trips < 40:
        score -= 60.0

    if long_entries < 40:
        score -= 40.0

    if long_exits < max(1.0, long_entries * 0.75):
        score -= 25.0

    if avg_hold_bars < 5.0:
        score -= (5.0 - avg_hold_bars) * 20.0

    if avg_capture_ratio < 0.55:
        score -= (0.55 - avg_capture_ratio) * 35.0

    # Hard penalty if the supposedly long-only trainer emits any short actions.
    if short_entries > 0 or short_exits > 0:
        score -= 250.0 + (short_entries + short_exits) * 0.25

    return float(score)


# =============================================================================
# Optuna objective
# =============================================================================

def objective(trial: optuna.Trial) -> float:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Keep subprocesses on the same dataset through midnight.
    env["RL_DATA_DATE"] = DATA_TAG

    # Curriculum mode. Re-enable advanced feature stacks later after ADD_LONG
    # behavior is stable.
    env["RL_INTRADAY_TIMESTEPS"] = os.getenv("RL_TUNER_TIMESTEPS", "25000")
    env["RL_POLICY_MODE"] = os.getenv("RL_TUNER_POLICY_MODE", "ppo")
    env["RL_USE_TRANSFORMER_FEATURES"] = os.getenv("RL_TUNER_USE_TRANSFORMER_FEATURES", "0")
    env["RL_USE_MULTI_TIMEFRAME_FUSION"] = os.getenv("RL_TUNER_USE_MULTI_TIMEFRAME_FUSION", "0")
    env["RL_USE_HIERARCHICAL_GATING"] = os.getenv("RL_TUNER_USE_HIERARCHICAL_GATING", "0")

    # Long-only safety flags.
    env["RL_LONG_ONLY"] = "1"
    env["RL_DISABLE_SHORTS"] = "1"
    env["RL_ALLOW_SHORTS"] = "0"
    env["RL_DISALLOW_SHORTS_WHEN_NOT_ALLOWED"] = "1"
    env["RL_VALID_SHORT_REWARD"] = "0.0"
    env["RL_SHORT_CHURN_PENALTY"] = "0.0"
    env["RL_LONG_SHORT_BALANCE_PENALTY"] = "0.0"

    # V2 trainer uses 4 actions:
    #   0 HOLD, 1 ENTER_LONG, 2 EXIT_LONG, 3 ADD_LONG.
    # Include multiple env names for compatibility with prior trainer variants.
    env["RL_ENABLE_ADD_LONG"] = "1"
    env["RL_ENABLE_PYRAMIDING"] = "1"
    env["RL_ADD_LONG_ENABLED"] = "1"

    # Exploration curriculum:
    # - when flat: force long entries sometimes
    # - when already long: bias exploration toward ADD_LONG
    env["RL_FORCE_EXPLORATION_ACTIONS"] = "1"
    env["RL_FORCE_EXPLORATION_LONG_ONLY"] = "1"
    env["RL_FORCE_ADD_LONG_EXPLORATION"] = "1"
    env["RL_FORCE_EXPLORATION_EPISODE_FRAC"] = str(
        trial.suggest_float("force_exploration_frac", 0.12, 0.36)
    )
    env["RL_FORCE_EXPLORATION_PROB"] = str(
        trial.suggest_float("force_exploration_prob", 0.05, 0.16)
    )
    env["RL_FORCE_ADD_LONG_EXPLORATION_PROB"] = str(
        trial.suggest_float("force_add_long_exploration_prob", 0.08, 0.30)
    )

    # PPO hyperparameters.
    env["RL_INTRADAY_LEARNING_RATE"] = str(
        trial.suggest_float("learning_rate", 2e-5, 2.5e-4, log=True)
    )
    env["RL_INTRADAY_GAMMA"] = str(
        trial.suggest_float("gamma", 0.988, 0.999)
    )
    env["RL_INTRADAY_ENT_COEF"] = str(
        trial.suggest_float("ent_coef", 0.035, 0.13)
    )
    env["RL_INTRADAY_BATCH_SIZE"] = str(
        trial.suggest_categorical("batch_size", [64, 128])
    )
    env["RL_INTRADAY_N_STEPS"] = str(
        trial.suggest_categorical("n_steps", [256, 512])
    )

    # MFE/MAE and exit capture objective.
    env["RL_PEAK_EXIT_LOOKAHEAD_BARS"] = str(
        trial.suggest_int("peak_exit_lookahead", 20, 60)
    )
    env["RL_EXIT_NEAR_HIGH_BONUS"] = str(
        trial.suggest_float("exit_near_high_bonus", 2.5, 7.5)
    )
    env["RL_MISSED_UPSIDE_PENALTY"] = str(
        trial.suggest_float("missed_upside_penalty", 1.5, 7.0)
    )
    env["RL_MAX_GIVEBACK_FROM_MFE"] = str(
        trial.suggest_float("max_giveback", 0.006, 0.026)
    )
    env["RL_MFE_CAPTURE_REWARD_MULT"] = str(
        trial.suggest_float("mfe_capture_reward_mult", 10.0, 28.0)
    )
    env["RL_MAE_PENALTY_MULT"] = str(
        trial.suggest_float("mae_penalty_mult", 3.0, 12.0)
    )
    env["RL_GIVEBACK_CAPTURE_PENALTY_MULT"] = str(
        trial.suggest_float("giveback_capture_penalty_mult", 4.0, 18.0)
    )

    # Hold quality / anti-scalp controls.
    env["RL_LONG_HOLD_TARGET_BARS"] = str(
        trial.suggest_float("hold_target_bars", 6.0, 20.0)
    )
    env["RL_AVG_HOLD_REWARD_MULT"] = str(
        trial.suggest_float("avg_hold_reward_mult", 0.45, 1.45)
    )
    env["RL_MIN_QUALITY_HOLD_BARS"] = str(
        trial.suggest_int("min_quality_hold_bars", 4, 10)
    )
    env["RL_EARLY_EXIT_PROFIT_PENALTY"] = str(
        trial.suggest_float("early_exit_profit_penalty", 1.0, 4.5)
    )
    env["RL_QUALITY_HOLD_PROFIT_BONUS"] = str(
        trial.suggest_float("quality_hold_profit_bonus", 2.0, 7.0)
    )

    # Continuation hold rewards.
    env["RL_HOLD_WINNER_PNL_MULT"] = str(
        trial.suggest_float("hold_winner_pnl_mult", 25.0, 85.0)
    )
    env["RL_HOLD_TREND_STRENGTH_MULT"] = str(
        trial.suggest_float("hold_trend_strength_mult", 0.05, 0.45)
    )
    env["RL_HOLD_FUTURE_CONTINUATION_MULT"] = str(
        trial.suggest_float("hold_future_continuation_mult", 100.0, 320.0)
    )
    env["RL_HOLD_CONTINUATION_CAP"] = str(
        trial.suggest_float("hold_continuation_cap", 3.0, 10.0)
    )

    # ADD_LONG / pyramiding controls.
    # V2 trainer names:
    #   RL_ADD_LONG_REWARD
    #   RL_ADD_LONG_MIN_UNREALIZED_PCT
    #   RL_ADD_LONG_MIN_HOLD_BARS
    #   RL_ADD_LONG_MAX_COUNT
    #   RL_ADD_LONG_FRACTION
    #   RL_ADD_LONG_FUTURE_REWARD_MULT
    #   RL_ADD_LONG_FUTURE_REWARD_CAP
    #   RL_BAD_ADD_LONG_PENALTY
    env["RL_ADD_LONG_REWARD"] = str(
        trial.suggest_float("add_long_reward", 5.0, 14.0)
    )
    env["RL_ADD_LONG_MIN_UNREALIZED_PCT"] = str(
        trial.suggest_float("add_long_min_unrealized_pct", 0.0008, 0.006)
    )
    env["RL_ADD_LONG_MIN_HOLD_BARS"] = str(
        trial.suggest_int("add_long_min_hold_bars", 1, 7)
    )
    env["RL_ADD_LONG_MAX_COUNT"] = str(
        trial.suggest_int("add_long_max_count", 1, 4)
    )
    env["RL_ADD_LONG_FRACTION"] = str(
        trial.suggest_float("add_long_fraction", 0.15, 0.70)
    )
    env["RL_ADD_LONG_FUTURE_REWARD_MULT"] = str(
        trial.suggest_float("add_long_future_reward_mult", 200.0, 650.0)
    )
    env["RL_ADD_LONG_FUTURE_REWARD_CAP"] = str(
        trial.suggest_float("add_long_future_reward_cap", 2.0, 8.0)
    )
    env["RL_BAD_ADD_LONG_PENALTY"] = str(
        trial.suggest_float("bad_add_long_penalty", 0.50, 4.0)
    )

    # Backward-compatible aliases used by prior tuner/trainer drafts.
    env["RL_ADD_LONG_CONTINUATION_BONUS_MULT"] = env["RL_ADD_LONG_FUTURE_REWARD_MULT"]

    # Safer continuation mask knobs. V2 trainer will ignore these if not present.
    env["RL_ADD_LONG_REQUIRE_ABOVE_VWAP"] = os.getenv("RL_ADD_LONG_REQUIRE_ABOVE_VWAP", "0")
    env["RL_ADD_LONG_REQUIRE_POSITIVE_MOMENTUM"] = os.getenv("RL_ADD_LONG_REQUIRE_POSITIVE_MOMENTUM", "1")
    env["RL_ADD_LONG_MIN_FUTURE_RETURN"] = str(
        trial.suggest_float("add_long_min_future_return", -0.001, 0.004)
    )

    # Dynamic trailing exits. Wider than earlier versions so the policy has room
    # to ride continuation and pyramid into winners.
    env["RL_USE_FIXED_TAKE_PROFIT"] = "0"
    env["RL_USE_DYNAMIC_GIVEBACK_EXIT"] = "1"
    env["RL_DYNAMIC_GIVEBACK_MIN_MFE"] = str(
        trial.suggest_float("dynamic_giveback_min_mfe", 0.006, 0.022)
    )
    env["RL_DYNAMIC_GIVEBACK_FRACTION"] = str(
        trial.suggest_float("dynamic_giveback_fraction", 0.50, 0.85)
    )
    env["RL_DYNAMIC_GIVEBACK_FLOOR"] = str(
        trial.suggest_float("dynamic_giveback_floor", 0.003, 0.010)
    )
    env["RL_DYNAMIC_GIVEBACK_MIN_HOLD_BARS"] = str(
        trial.suggest_int("dynamic_giveback_min_hold_bars", 5, 14)
    )

    # Keep exploration incentives high enough to avoid HOLD collapse.
    env["RL_MIN_TRADES_PER_EPISODE"] = "2"
    env["RL_NO_TRADE_EPISODE_PENALTY"] = "25.0"
    env["RL_FLAT_HOLD_STEP_PENALTY"] = "0.006"
    env["RL_VALID_ENTRY_REWARD"] = "4.0"
    env["RL_FIRST_TRADE_BONUS"] = "8.0"

    print("\n" + "=" * 100, flush=True)
    print(f"STARTING LONG-ONLY ADD_LONG V2 OPTUNA TRIAL {trial.number}", flush=True)
    print("=" * 100, flush=True)
    print("[data tag]", DATA_TAG, flush=True)
    print("[trial params]", flush=True)
    print(json.dumps(trial.params, indent=2), flush=True)
    print(f"[trainer] {TRAINER_PATH}", flush=True)
    print(f"[timesteps] {env['RL_INTRADAY_TIMESTEPS']}", flush=True)
    print("=" * 100 + "\n", flush=True)

    if not TRAINER_PATH.exists():
        print(f"[trial {trial.number}] FAILED: trainer not found at {TRAINER_PATH}", flush=True)
        return DEFAULT_BAD_SCORE

    process = subprocess.Popen(
        [sys.executable, "-u", str(TRAINER_PATH)],
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
        print(f"[trial {trial.number}] FAILED: trainer exited non-zero", flush=True)
        return DEFAULT_BAD_SCORE

    metrics, metrics_path = load_metrics()
    if not metrics:
        print(f"[trial {trial.number}] FAILED: metrics file not found or empty", flush=True)
        for path in METRICS_PATHS:
            print(f"  - {path}", flush=True)
        return DEFAULT_BAD_SCORE

    score = score_metrics(metrics)

    print("\n" + "-" * 100, flush=True)
    print(f"LONG-ONLY ADD_LONG V2 TRIAL {trial.number} RESULTS", flush=True)
    print("-" * 100, flush=True)
    print(f"[metrics path] {metrics_path}", flush=True)
    print(json.dumps(metrics, indent=2, default=str), flush=True)
    print(f"[score] {score:.6f}", flush=True)
    print("-" * 100 + "\n", flush=True)

    return float(score)


def main() -> None:
    study = optuna.create_study(
        direction="maximize",
        study_name=os.getenv("RL_TUNER_STUDY_NAME", "intraday_long_only_addlong_v2_optimizer"),
    )

    study.optimize(
        objective,
        n_trials=int(os.getenv("RL_TUNER_TRIALS", "100")),
        show_progress_bar=True,
        catch=(Exception,),
    )

    print("\nBEST PARAMS:")
    print(json.dumps(study.best_params, indent=2))

    print("\nBEST SCORE:")
    print(study.best_value)


if __name__ == "__main__":
    main()
