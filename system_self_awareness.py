#!/usr/bin/env python3
"""
system_self_awareness.py

Monitors system health, data freshness, model reliability, decision quality,
and execution risk. Produces diagnostics that run.py can use to reduce size,
enter SAFE mode, disable options, disable pyramiding, or raise buy thresholds.

This module NEVER submits orders. It only reads data and writes diagnostics.
Exit/sell behavior in run.py is never affected by outputs from this module.

Outputs (all dated under system-data/YYYYMMDD/ or logs/execution/YYYYMMDD/):
  YYYYMMDD_system_health.json
  YYYYMMDD_model_scorecard.json
  YYYYMMDD_self_assessment.json
  YYYYMMDD_decision_audit.jsonl   (append-only)

Usage:
  python system_self_awareness.py            # full run, writes all outputs
  python system_self_awareness.py --dry-run  # prints assessment, writes nothing

Integration with run.py:
  from system_self_awareness import load_self_assessment, self_assessment_size_multiplier
  assessment = load_self_assessment()
  mult = self_assessment_size_multiplier(assessment)  # apply to buy sizes only
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.env_loader import load_instance_env

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
load_instance_env()

try:
    from market_context_builder import TODAY_TAG
except Exception:
    TODAY_TAG = datetime.now().strftime("%Y%m%d")

DATA_TAG: str = os.getenv("RUN_DATA_DATE", os.getenv("RL_DATA_DATE", TODAY_TAG))

OUT_DIR = PROJECT_ROOT / "system-data" / DATA_TAG
LOG_DIR = PROJECT_ROOT / "logs" / "execution" / DATA_TAG
DAILY_OUT_DIR = PROJECT_ROOT / "daily_system_outputs" / DATA_TAG

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Tunable thresholds via env vars
# ---------------------------------------------------------------------------

SA_MAX_FILE_AGE_MINUTES: float = float(os.getenv("SA_MAX_FILE_AGE_MINUTES", "20"))
SA_SESSION_FILE_AGE_MINUTES: float = float(os.getenv("SA_SESSION_FILE_AGE_MINUTES", "600"))
SA_CRITICAL_FILE_AGE_MINUTES: float = float(os.getenv("SA_CRITICAL_FILE_AGE_MINUTES", "60"))
SA_HALT_EXEC_RISK: float = float(os.getenv("SA_HALT_EXEC_RISK", "0.85"))
SA_SAFE_EXEC_RISK: float = float(os.getenv("SA_SAFE_EXEC_RISK", "0.65"))
SA_SAFE_DATA_QUALITY: float = float(os.getenv("SA_SAFE_DATA_QUALITY", "0.50"))
SA_SAFE_REGIME_RISK: float = float(os.getenv("SA_SAFE_REGIME_RISK", "0.70"))
SA_CAUTIOUS_REGIME_RISK: float = float(os.getenv("SA_CAUTIOUS_REGIME_RISK", "0.45"))
SA_MIN_MODEL_ALIGNMENT: float = float(os.getenv("SA_MIN_MODEL_ALIGNMENT", "0.30"))
SA_MAX_CONSECUTIVE_LOSSES: int = int(os.getenv("SA_MAX_CONSECUTIVE_LOSSES", "3"))
SA_DAILY_LOSS_HALT_PCT: float = float(os.getenv("SA_DAILY_LOSS_HALT_PCT", "0.025"))
SA_DAILY_LOSS_SAFE_PCT: float = float(os.getenv("SA_DAILY_LOSS_SAFE_PCT", "0.015"))
SA_ANOMALY_EQUITY_DROP_PCT: float = float(os.getenv("SA_ANOMALY_EQUITY_DROP_PCT", "0.05"))
SA_MIN_WIN_RATE_FOR_NORMAL: float = float(os.getenv("SA_MIN_WIN_RATE_FOR_NORMAL", "0.30"))
SA_ENABLED: bool = os.getenv("SA_ENABLED", "1").lower() in {"1", "true", "yes"}

# ---------------------------------------------------------------------------
# Input paths
# ---------------------------------------------------------------------------

_MARKET_CONTEXT_PATH = DAILY_OUT_DIR / f"{DATA_TAG}_market_context.json"
if not _MARKET_CONTEXT_PATH.exists():
    _MARKET_CONTEXT_PATH = OUT_DIR / f"{DATA_TAG}_market_context.json"

MARKET_CONTEXT_PATH = _MARKET_CONTEXT_PATH
PREMARKET_PLAN_PATH = OUT_DIR / f"{DATA_TAG}_premarket_trade_plan.json"
STRUCTURE_MAP_PATH = OUT_DIR / f"{DATA_TAG}_intraday_structure_map.json"
FEATURE_STREAM_DIR = OUT_DIR / f"{DATA_TAG}_Feature_stream"
FEATURE_STREAM_CSV_PATH = FEATURE_STREAM_DIR / f"{DATA_TAG}_intraday_feature_stream.csv"
FEATURE_STREAM_STATE_PATH = FEATURE_STREAM_DIR / f"{DATA_TAG}_symbol_state.json"
FORECAST_ENSEMBLE_PATH = OUT_DIR / f"{DATA_TAG}_forecast_ensemble_signals.json"
JOURNAL_PATH = LOG_DIR / f"{DATA_TAG}_executor_journal.csv"
EQUITY_CURVE_PATH = LOG_DIR / f"{DATA_TAG}_executor_equity_curve.csv"
EXPERIENCE_REPLAY_PATH = LOG_DIR / f"{DATA_TAG}_live_experience_replay.jsonl"
RL_MODEL_PATH = OUT_DIR / f"{DATA_TAG}_rl_intraday_structure_model.zip"
RL_METRICS_PATH = OUT_DIR / f"{DATA_TAG}_rl_intraday_training_metrics.json"
ES_METRICS_PATH = OUT_DIR / f"{DATA_TAG}_evolution_strategy_metrics.json"
ES_SIGNALS_PATH = OUT_DIR / f"{DATA_TAG}_evolution_strategy_signals.json"
OPTION_BOOSTER_SUMMARY_PATH = OUT_DIR / f"{DATA_TAG}_latest_option_booster_summary.json"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

SYSTEM_HEALTH_PATH = OUT_DIR / f"{DATA_TAG}_system_health.json"
MODEL_SCORECARD_PATH = OUT_DIR / f"{DATA_TAG}_model_scorecard.json"
SELF_ASSESSMENT_PATH = OUT_DIR / f"{DATA_TAG}_self_assessment.json"
DECISION_AUDIT_PATH = LOG_DIR / f"{DATA_TAG}_decision_audit.jsonl"

# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def _file_age_minutes(path: Path) -> float | None:
    """Return file age in minutes, or None if path does not exist."""
    try:
        return (time.time() - path.stat().st_mtime) / 60.0
    except Exception:
        return None

def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return default if default is not None else {}

def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8", errors="ignore") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []

def _read_jsonl(path: Path, max_lines: int = 500) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        with path.open(encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return rows

def _to_float(val: Any, default: float = 0.0) -> float:
    try:
        v = float(val)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default

def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))

def _session_started() -> bool:
    """Return True if the execution journal exists for today — i.e. run.py has started a session."""
    return JOURNAL_PATH.exists()

def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")

def _append_audit(entry: dict[str, Any]) -> None:
    entry.setdefault("timestamp_utc", _now_utc())
    with DECISION_AUDIT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# =============================================================================
# 1. System Health Monitor
# =============================================================================

def check_system_health() -> dict[str, Any]:
    """Check process-level health: disk space, key file existence, recent errors."""
    issues: list[str] = []
    warnings: list[str] = []

    # Disk space check (warn if < 500 MB free)
    disk_free_mb: float | None = None
    try:
        import shutil
        stat = shutil.disk_usage(PROJECT_ROOT)
        disk_free_mb = stat.free / (1024 * 1024)
        if disk_free_mb < 200:
            issues.append(f"disk_critical: {disk_free_mb:.0f}MB free")
        elif disk_free_mb < 500:
            warnings.append(f"disk_low: {disk_free_mb:.0f}MB free")
    except Exception as e:
        warnings.append(f"disk_check_failed: {e}")

    # Key file existence
    critical_files = {
        "market_context": MARKET_CONTEXT_PATH,
        "premarket_plan": PREMARKET_PLAN_PATH,
        "structure_map": STRUCTURE_MAP_PATH,
    }
    file_status: dict[str, dict[str, Any]] = {}
    for name, path in critical_files.items():
        age = _file_age_minutes(path)
        exists = path.exists()
        file_status[name] = {
            "exists": exists,
            "path": str(path),
            "age_minutes": round(age, 1) if age is not None else None,
            "stale": age is not None and age > SA_MAX_FILE_AGE_MINUTES,
        }
        if not exists:
            issues.append(f"missing_critical_file: {name}")
        elif age is not None and age > SA_CRITICAL_FILE_AGE_MINUTES:
            issues.append(f"very_stale: {name} age={age:.0f}m")
        elif age is not None and age > SA_MAX_FILE_AGE_MINUTES:
            warnings.append(f"stale: {name} age={age:.0f}m")

    # Check execution journal for error rows
    journal_rows = _read_csv_rows(JOURNAL_PATH)
    error_events = [r for r in journal_rows if "error" in str(r.get("event", "")).lower() or "fail" in str(r.get("message", "")).lower()]
    if len(error_events) > 3:
        warnings.append(f"journal_errors: {len(error_events)} error events today")

    # SPX price action log — check if subprocess is producing output
    spx_log = LOG_DIR / f"{DATA_TAG}_spx_price_action_layer_from_run.log"
    spx_age = _file_age_minutes(spx_log)
    if spx_age is not None and spx_age > 30:
        warnings.append(f"spx_layer_stale: age={spx_age:.0f}m")
    elif spx_age is None:
        warnings.append("spx_layer_log_missing")

    health_score = 1.0
    health_score -= 0.15 * len(issues)
    health_score -= 0.05 * len(warnings)
    health_score = _clip01(health_score)

    result = {
        "created_at_utc": _now_utc(),
        "health_score": round(health_score, 3),
        "issues": issues,
        "warnings": warnings,
        "file_status": file_status,
        "disk_free_mb": round(disk_free_mb, 1) if disk_free_mb is not None else None,
        "journal_rows": len(journal_rows),
        "journal_error_events": len(error_events),
    }
    print(f"[self-awareness] system_health score={health_score:.2f} issues={len(issues)} warnings={len(warnings)}")
    return result


# =============================================================================
# 2. Data Freshness Monitor
# =============================================================================

def check_data_freshness() -> dict[str, Any]:
    """Score freshness of every major input file. Missing = 0. Fresh = 1."""
    items: dict[str, dict[str, Any]] = {}
    score_parts: list[float] = []

    # Session-planning files are generated by premarket_data_collector / premarket_trade_planner.
    # Before the session starts (no journal yet), they are expected to be absent.
    # Scoring them as 0 would cause a spurious HALT_NEW_BUYS before the session even opens.
    _pre_session = not _session_started()
    _SESSION_PLANNING_NAMES = {"premarket_plan", "structure_map"}

    def _check(name: str, path: Path, weight: float, max_age: float = SA_MAX_FILE_AGE_MINUTES) -> float:
        age = _file_age_minutes(path)
        exists = path.exists()
        if not exists:
            # Pre-session: planning files are expected to be absent — treat as neutral.
            if _pre_session and name in _SESSION_PLANNING_NAMES:
                items[name] = {"exists": False, "age_minutes": None, "fresh": False,
                               "weight": weight, "score": 0.5, "pre_session_neutral": True}
                return 0.5 * weight
            items[name] = {"exists": False, "age_minutes": None, "fresh": False, "weight": weight}
            return 0.0
        if age is None:
            items[name] = {"exists": False, "age_minutes": None, "fresh": False, "weight": weight}
            return 0.0
        fresh = age <= max_age
        # Partial credit: decay linearly between max_age and 2x max_age
        if fresh:
            contrib = 1.0
        else:
            decay = min(age / max(max_age * 2, 1.0), 1.0)
            contrib = max(1.0 - decay, 0.0)
        items[name] = {"exists": True, "age_minutes": round(age, 1), "fresh": fresh, "weight": weight, "score": round(contrib, 3)}
        return contrib * weight

    total_weight = 0.0

    def _add(name: str, path: Path, weight: float, max_age: float = SA_MAX_FILE_AGE_MINUTES) -> None:
        nonlocal total_weight
        score_parts.append(_check(name, path, weight, max_age))
        total_weight += weight

    # Session planning artifacts are generated before the open and remain
    # valid for that trading session; live streams retain the short TTL.
    _add("market_context", MARKET_CONTEXT_PATH, 0.20, SA_SESSION_FILE_AGE_MINUTES)
    _add("premarket_plan", PREMARKET_PLAN_PATH, 0.25, SA_SESSION_FILE_AGE_MINUTES)
    _add("structure_map", STRUCTURE_MAP_PATH, 0.20, SA_SESSION_FILE_AGE_MINUTES)
    _add("feature_stream", FEATURE_STREAM_CSV_PATH, 0.15, SA_MAX_FILE_AGE_MINUTES)
    _add("forecast_ensemble", FORECAST_ENSEMBLE_PATH, 0.10, SA_MAX_FILE_AGE_MINUTES * 3)
    _add("journal", JOURNAL_PATH, 0.05, SA_MAX_FILE_AGE_MINUTES * 6)
    _add("equity_curve", EQUITY_CURVE_PATH, 0.05, SA_MAX_FILE_AGE_MINUTES * 6)

    raw = sum(score_parts) / max(total_weight, 1e-9)
    data_quality_score = _clip01(raw)

    # Semantic checks on top of freshness
    semantic_issues: list[str] = []

    plan = _read_json(PREMARKET_PLAN_PATH)
    if plan:
        status = str(plan.get("status") or plan.get("planner_status") or "").lower()
        if status and status not in {"ok", "normal"}:
            semantic_issues.append(f"plan_status={status}")
        long_allowed = plan.get("long_allowed_symbols") or []
        if not long_allowed:
            semantic_issues.append("plan_no_long_allowed_symbols")
            data_quality_score = _clip01(data_quality_score - 0.10)

    struct = _read_json(STRUCTURE_MAP_PATH)
    if struct:
        structures = struct.get("structures") or []
        if not structures:
            semantic_issues.append("structure_map_empty")
            data_quality_score = _clip01(data_quality_score - 0.10)

    if semantic_issues:
        data_quality_score = _clip01(data_quality_score - 0.05 * len(semantic_issues))

    result = {
        "created_at_utc": _now_utc(),
        "data_quality_score": round(data_quality_score, 3),
        "items": items,
        "semantic_issues": semantic_issues,
    }
    print(f"[self-awareness] data_freshness quality={data_quality_score:.2f} semantic_issues={semantic_issues}")
    return result


# =============================================================================
# 3. Model Artifact Freshness Monitor
# =============================================================================

def check_model_artifacts() -> dict[str, Any]:
    """Check model file ages, RL metrics, forecast signals, and ES health."""
    artifacts: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    score_parts: list[float] = []

    def _artifact(name: str, path: Path, max_age_hours: float = 24.0) -> float:
        age_min = _file_age_minutes(path)
        if age_min is None:
            artifacts[name] = {"exists": False, "age_hours": None}
            issues.append(f"missing_model_artifact: {name}")
            return 0.0
        age_hr = age_min / 60.0
        fresh = age_hr <= max_age_hours
        artifacts[name] = {
            "exists": True,
            "age_hours": round(age_hr, 2),
            "fresh": fresh,
            "path": str(path),
        }
        if not fresh:
            issues.append(f"stale_model_artifact: {name} age={age_hr:.1f}h")
        return 1.0 if fresh else max(1.0 - (age_hr / max(max_age_hours * 2, 1.0)), 0.0)

    # RL model
    rl_score = _artifact("rl_model", RL_MODEL_PATH, 24.0)
    score_parts.append(rl_score * 0.30)

    # RL metrics — check production_ready flag
    rl_metrics = _read_json(RL_METRICS_PATH)
    if rl_metrics:
        held_out = rl_metrics.get("held_out_evaluation", {})
        prod_ready = bool(held_out.get("production_ready", True))  # default true if old format
        test_win = _to_float(held_out.get("test_win_rate"), -1.0)
        artifacts["rl_metrics"] = {
            "exists": True,
            "production_ready": prod_ready,
            "test_win_rate": test_win,
            "test_episode_count": held_out.get("test_episode_count"),
        }
        rl_metrics_score = 1.0 if prod_ready else 0.40
        if test_win >= 0 and test_win < 0.35:
            issues.append(f"rl_low_test_win_rate: {test_win:.1%}")
            rl_metrics_score *= 0.60
    else:
        artifacts["rl_metrics"] = {"exists": False}
        rl_metrics_score = 0.50  # neutral if no metrics file (may not have trained yet)

    score_parts.append(rl_metrics_score * 0.20)

    # Evolution strategy metrics
    es_metrics = _read_json(ES_METRICS_PATH)
    if es_metrics:
        es_ok = bool(es_metrics.get("ok", True))
        symbols_trained = len(es_metrics.get("symbols_trained") or [])
        artifacts["evolution_strategy"] = {
            "exists": True,
            "ok": es_ok,
            "symbols_trained": symbols_trained,
        }
        es_score = 1.0 if es_ok and symbols_trained > 0 else 0.40
        if not es_ok:
            issues.append("evolution_strategy_not_ok")
    else:
        artifacts["evolution_strategy"] = {"exists": False}
        es_score = 0.50
        issues.append("evolution_strategy_metrics_missing")

    score_parts.append(es_score * 0.15)

    # Forecast ensemble signals
    forecast = _read_json(FORECAST_ENSEMBLE_PATH)
    if forecast:
        signals = forecast.get("signals") or []
        non_default = [s for s in signals if abs(_to_float(s.get("ensemble_score"), 0.5) - 0.5) > 0.05]
        fc_source = signals[0].get("forecast_source", "") if signals else ""
        is_real_model = "forecast_engine" in str(fc_source)
        artifacts["forecast_ensemble"] = {
            "exists": True,
            "signal_count": len(signals),
            "non_default_signals": len(non_default),
            "source": fc_source,
            "real_model": is_real_model,
        }
        fc_score = 1.0 if is_real_model and non_default else 0.50
    else:
        artifacts["forecast_ensemble"] = {"exists": False}
        fc_score = 0.50  # neutral — may not have run yet
        issues.append("forecast_ensemble_missing")

    score_parts.append(fc_score * 0.20)

    # Feature stream state
    ss = _read_json(FEATURE_STREAM_STATE_PATH)
    if ss:
        sym_count = ss.get("symbol_count", 0)
        artifacts["feature_stream_state"] = {"exists": True, "symbol_count": sym_count}
        fs_score = 1.0 if sym_count > 0 else 0.30
    else:
        artifacts["feature_stream_state"] = {"exists": False}
        fs_score = 0.30
        issues.append("feature_stream_state_missing")

    score_parts.append(fs_score * 0.15)

    alignment_score = _clip01(sum(score_parts))

    result = {
        "created_at_utc": _now_utc(),
        "model_alignment_score": round(alignment_score, 3),
        "artifacts": artifacts,
        "issues": issues,
    }
    print(f"[self-awareness] model_artifacts alignment={alignment_score:.2f} issues={issues}")
    return result


# =============================================================================
# 4. Decision Audit Logger
# =============================================================================

def log_decision_audit(
    event_type: str,
    details: dict[str, Any],
    assessment_snapshot: dict[str, Any] | None = None,
) -> None:
    """Append a single decision event to the audit JSONL. Never blocks."""
    entry: dict[str, Any] = {
        "timestamp_utc": _now_utc(),
        "data_tag": DATA_TAG,
        "event_type": event_type,
        "details": details,
    }
    if assessment_snapshot:
        entry["assessment_mode"] = assessment_snapshot.get("execution_mode_recommendation")
        entry["assessment_size_mult"] = assessment_snapshot.get("recommended_size_multiplier")
    _append_audit(entry)


def audit_gate_decisions(journal_rows: list[dict[str, str]]) -> dict[str, Any]:
    """Summarize recent gate outcomes from the journal for audit trail."""
    buys = [r for r in journal_rows if str(r.get("trade_side", "")).lower() == "buy"]
    sells = [r for r in journal_rows if str(r.get("trade_side", "")).lower() == "sell"]
    errors = [r for r in journal_rows if "error" in str(r.get("event", "")).lower()]
    skipped = [r for r in journal_rows if "skip" in str(r.get("event", "")).lower()]

    summary = {
        "total_journal_rows": len(journal_rows),
        "buy_events": len(buys),
        "sell_events": len(sells),
        "error_events": len(errors),
        "skipped_events": len(skipped),
        "symbols_bought": sorted({str(r.get("trade_symbol", "")) for r in buys if r.get("trade_symbol")}),
        "symbols_sold": sorted({str(r.get("trade_symbol", "")) for r in sells if r.get("trade_symbol")}),
    }
    return summary


# =============================================================================
# 5. Model Scorecard Tracker
# =============================================================================

def build_model_scorecard(artifact_check: dict[str, Any]) -> dict[str, Any]:
    """Produce a per-model scorecard from available metrics files."""
    scorecard: dict[str, Any] = {
        "created_at_utc": _now_utc(),
        "data_tag": DATA_TAG,
        "models": {},
    }

    # RL intraday model
    rl_metrics = _read_json(RL_METRICS_PATH)
    if rl_metrics:
        held = rl_metrics.get("held_out_evaluation", {})
        in_sample_quality = rl_metrics.get("diagnostics", {}).get("trade_quality", {})
        scorecard["models"]["rl_intraday"] = {
            "exists": True,
            "production_ready": bool(held.get("production_ready", True)),
            "train_episodes": held.get("train_episode_count"),
            "test_episodes": held.get("test_episode_count"),
            "test_win_rate": held.get("test_win_rate"),
            "test_avg_return": held.get("test_avg_trade_return"),
            "test_max_drawdown": held.get("test_max_drawdown"),
            "in_sample_win_rate": in_sample_quality.get("win_rate"),
            "in_sample_profit_factor": in_sample_quality.get("profit_factor"),
            "in_sample_avg_hold_bars": in_sample_quality.get("avg_hold_bars"),
        }
    else:
        scorecard["models"]["rl_intraday"] = {"exists": False}

    # Evolution strategy
    es_metrics = _read_json(ES_METRICS_PATH)
    if es_metrics:
        scorecard["models"]["evolution_strategy"] = {
            "exists": True,
            "ok": bool(es_metrics.get("ok", False)),
            "symbols_trained": len(es_metrics.get("symbols_trained") or []),
            "symbols_skipped": len(es_metrics.get("symbols_skipped") or []),
        }
    else:
        scorecard["models"]["evolution_strategy"] = {"exists": False}

    # Forecast ensemble
    forecast = _read_json(FORECAST_ENSEMBLE_PATH)
    if forecast:
        signals = forecast.get("signals") or []
        scores = [_to_float(s.get("ensemble_score"), 0.5) for s in signals]
        import statistics
        scorecard["models"]["forecast_ensemble"] = {
            "exists": True,
            "signal_count": len(signals),
            "mean_ensemble_score": round(statistics.mean(scores), 3) if scores else None,
            "stddev_ensemble_score": round(statistics.stdev(scores), 3) if len(scores) > 1 else None,
            "pct_buy_approved": round(sum(1 for s in signals if s.get("forecast_buy_approved")) / max(len(signals), 1), 3),
            "source": signals[0].get("forecast_source") if signals else None,
        }
    else:
        scorecard["models"]["forecast_ensemble"] = {"exists": False}

    # Options booster recent performance
    ob_summary = _read_json(OPTION_BOOSTER_SUMMARY_PATH)
    if ob_summary:
        scorecard["models"]["options_booster"] = {
            "exists": True,
            "enabled": bool(ob_summary.get("enabled", False)),
            "flow_candidates": ob_summary.get("flow_candidates", 0),
            "orders_generated": ob_summary.get("orders", 0),
        }
    else:
        scorecard["models"]["options_booster"] = {"exists": False}

    print(f"[self-awareness] model_scorecard built for {list(scorecard['models'].keys())}")
    return scorecard


# =============================================================================
# 6. Trade Outcome Analyzer
# =============================================================================

def analyze_trade_outcomes() -> dict[str, Any]:
    """Read journal + equity curve to assess recent execution quality."""
    journal_rows = _read_csv_rows(JOURNAL_PATH)
    equity_rows = _read_csv_rows(EQUITY_CURVE_PATH)

    result: dict[str, Any] = {
        "created_at_utc": _now_utc(),
        "journal_rows": len(journal_rows),
        "equity_rows": len(equity_rows),
        "issues": [],
        "warnings": [],
    }

    # --- Equity curve analysis ---
    equities: list[float] = []
    for r in equity_rows:
        eq = _to_float(r.get("equity") or r.get("equity_after") or r.get("portfolio_value"), -1.0)
        if eq > 0:
            equities.append(eq)

    if equities:
        start_eq = equities[0]
        current_eq = equities[-1]
        peak_eq = max(equities)
        daily_pnl_pct = (current_eq - start_eq) / max(start_eq, 1.0)
        drawdown_from_peak = (current_eq - peak_eq) / max(peak_eq, 1.0)
        result["start_equity"] = round(start_eq, 2)
        result["current_equity"] = round(current_eq, 2)
        result["peak_equity"] = round(peak_eq, 2)
        result["daily_pnl_pct"] = round(daily_pnl_pct, 4)
        result["drawdown_from_peak_pct"] = round(drawdown_from_peak, 4)

        if daily_pnl_pct <= -SA_DAILY_LOSS_HALT_PCT:
            result["issues"].append(f"daily_loss_halt_threshold: pnl={daily_pnl_pct:.2%}")
        elif daily_pnl_pct <= -SA_DAILY_LOSS_SAFE_PCT:
            result["warnings"].append(f"daily_loss_safe_threshold: pnl={daily_pnl_pct:.2%}")
        if drawdown_from_peak <= -SA_ANOMALY_EQUITY_DROP_PCT:
            result["warnings"].append(f"intraday_drawdown_from_peak: dd={drawdown_from_peak:.2%}")
    else:
        result["daily_pnl_pct"] = 0.0
        result["drawdown_from_peak_pct"] = 0.0

    # --- Journal trade analysis ---
    fills = [r for r in journal_rows if str(r.get("event", "")).lower() in {"fill", "order_fill", "filled", "buy_fill", "sell_fill"}
             or (r.get("filled_qty") and _to_float(r.get("filled_qty"), 0) > 0)]
    buys = [r for r in fills if str(r.get("trade_side", "")).lower() == "buy"]
    sells = [r for r in fills if str(r.get("trade_side", "")).lower() == "sell"]

    # Slippage analysis
    slippages: list[float] = []
    for r in fills:
        s = _to_float(r.get("slippage_pct"), None)
        if s is not None and s != 0.0:
            slippages.append(s)

    result["fills_today"] = len(fills)
    result["buys_today"] = len(buys)
    result["sells_today"] = len(sells)
    result["avg_slippage_pct"] = round(sum(slippages) / len(slippages), 5) if slippages else 0.0

    if len(slippages) > 0 and sum(slippages) / len(slippages) > 0.005:
        result["warnings"].append(f"high_avg_slippage: {result['avg_slippage_pct']:.3%}")

    # Consecutive loss detection from experience replay
    replay_rows = _read_jsonl(EXPERIENCE_REPLAY_PATH, max_lines=200)
    if replay_rows:
        pnls = [_to_float(r.get("realized_pnl") or r.get("pnl"), None) for r in replay_rows]
        pnls = [p for p in pnls if p is not None]
        if pnls:
            win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
            result["replay_trade_count"] = len(pnls)
            result["replay_win_rate"] = round(win_rate, 3)
            # Consecutive losses from the tail
            consec = 0
            for p in reversed(pnls):
                if p < 0:
                    consec += 1
                else:
                    break
            result["consecutive_losses"] = consec
            if consec >= SA_MAX_CONSECUTIVE_LOSSES:
                result["warnings"].append(f"consecutive_losses={consec}>={SA_MAX_CONSECUTIVE_LOSSES}")
            if win_rate < SA_MIN_WIN_RATE_FOR_NORMAL and len(pnls) >= 5:
                result["warnings"].append(f"low_win_rate: {win_rate:.1%}<{SA_MIN_WIN_RATE_FOR_NORMAL:.1%}")
        else:
            result["replay_trade_count"] = 0
            result["consecutive_losses"] = 0
    else:
        result["consecutive_losses"] = 0

    print(f"[self-awareness] trade_outcomes fills={result['fills_today']} daily_pnl={result.get('daily_pnl_pct',0):.2%} issues={result['issues']}")
    return result


# =============================================================================
# 7. Anomaly Detector
# =============================================================================

def detect_anomalies(
    health: dict[str, Any],
    freshness: dict[str, Any],
    artifacts: dict[str, Any],
    outcomes: dict[str, Any],
) -> dict[str, Any]:
    """Cross-component anomaly detection. Combines signals from all monitors."""
    anomalies: list[dict[str, Any]] = []
    severity_score = 0.0

    def _flag(severity: str, code: str, detail: str) -> None:
        nonlocal severity_score
        weights = {"critical": 0.30, "high": 0.20, "medium": 0.10, "low": 0.05}
        anomalies.append({"severity": severity, "code": code, "detail": detail})
        severity_score += weights.get(severity, 0.05)
        print(f"[self-awareness] anomaly [{severity}] {code}: {detail}")

    # Data gap
    dq = _to_float(freshness.get("data_quality_score"), 1.0)
    if dq < 0.25:
        _flag("critical", "data_quality_critical", f"dq={dq:.2f}")
    elif dq < 0.50:
        _flag("high", "data_quality_low", f"dq={dq:.2f}")

    # Missing critical files
    # Pre-session: planning files (premarket_plan, structure_map) are expected to be absent
    # before run.py has started. Flag them at low severity so they don't trigger HALT.
    _pre_session_detect = not _session_started()
    _SESSION_PLAN_NAMES = {"premarket_plan", "structure_map"}
    for name, info in freshness.get("items", {}).items():
        if not info.get("exists") and name in {"market_context", "premarket_plan", "structure_map"}:
            if _pre_session_detect and name in _SESSION_PLAN_NAMES:
                _flag("low", f"premarket_file_not_yet_generated:{name}",
                      "session not started — file will be created by premarket collector")
            else:
                _flag("critical", f"critical_file_missing:{name}", f"path={info.get('path','?')}")

    # Stale plan
    plan_info = freshness.get("items", {}).get("premarket_plan", {})
    if plan_info.get("exists") and plan_info.get("stale"):
        _flag("high", "premarket_plan_stale", f"age={plan_info.get('age_minutes')}m")

    # Daily loss exceeded
    for issue in outcomes.get("issues", []):
        if "daily_loss_halt" in issue:
            _flag("critical", "daily_loss_halt", issue)
        elif "daily_loss_safe" in issue:
            _flag("high", "daily_loss_safe", issue)

    # Consecutive losses
    consec = outcomes.get("consecutive_losses", 0)
    if consec >= SA_MAX_CONSECUTIVE_LOSSES + 2:
        _flag("high", "excessive_consecutive_losses", f"consec={consec}")
    elif consec >= SA_MAX_CONSECUTIVE_LOSSES:
        _flag("medium", "consecutive_losses", f"consec={consec}")

    # RL model not production-ready
    rl_info = artifacts.get("artifacts", {}).get("rl_metrics", {})
    if rl_info.get("exists") and not rl_info.get("production_ready", True):
        _flag("medium", "rl_not_production_ready", "held_out evaluation failed or insufficient test set")

    # No sells despite open positions (possible exit suppression — flag for investigation)
    sells_today = outcomes.get("sells_today", None)
    buys_today = outcomes.get("buys_today", 0)
    if sells_today is not None and buys_today > 2 and sells_today == 0:
        _flag("low", "no_sells_despite_buys", f"buys={buys_today} sells=0 (investigate exits)")

    # High slippage
    if _to_float(outcomes.get("avg_slippage_pct"), 0.0) > 0.008:
        _flag("medium", "high_slippage", f"avg={outcomes['avg_slippage_pct']:.3%}")

    # Disk space
    if "disk_critical" in str(health.get("issues", [])):
        _flag("critical", "disk_space_critical", "less than 200MB free")

    anomaly_score = _clip01(severity_score)
    result = {
        "created_at_utc": _now_utc(),
        "anomaly_score": round(anomaly_score, 3),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
    }
    print(f"[self-awareness] anomalies detected={len(anomalies)} score={anomaly_score:.2f}")
    return result


# =============================================================================
# 8. Self-Assessment Summary
# =============================================================================

def _regime_risk_from_context(market_context: dict[str, Any]) -> tuple[float, list[str]]:
    """Derive a 0-1 regime risk score from market context. Higher = more risk."""
    reasons: list[str] = []
    bias = str(market_context.get("overall_market_bias", "")).lower()
    permission = str(market_context.get("trade_permission", "")).lower()
    risk_level = str(market_context.get("risk_level", "")).lower()
    etf_ctx = market_context.get("key_etf_context_summary") or {}
    risk_signal = str(etf_ctx.get("risk_signal_bias", "")).lower()
    sector_ctx = market_context.get("sector_context_summary") or {}
    sector_rotation = str(sector_ctx.get("sector_rotation_bias", "")).lower()

    score = 0.25  # baseline

    if bias == "bearish":
        score += 0.35
        reasons.append("market_bias=bearish")
    elif bias == "mixed":
        score += 0.15
        reasons.append("market_bias=mixed")
    elif bias == "bullish":
        score -= 0.10
        reasons.append("market_bias=bullish")

    if permission in {"disabled", "halt", "halt_new_buys"}:
        score += 0.30
        reasons.append(f"trade_permission={permission}")
    elif permission == "cautious":
        score += 0.15
        reasons.append("trade_permission=cautious")

    if risk_signal == "risk_off":
        score += 0.20
        reasons.append("risk_signal=risk_off")
    elif risk_signal == "risk_on":
        score -= 0.05

    if risk_level == "high":
        score += 0.10
        reasons.append("risk_level=high")
    elif risk_level == "low":
        score -= 0.05

    if sector_rotation == "defensive":
        score += 0.05
        reasons.append("sector_rotation=defensive")

    return _clip01(score), reasons


def build_self_assessment(
    health: dict[str, Any],
    freshness: dict[str, Any],
    artifacts: dict[str, Any],
    outcomes: dict[str, Any],
    anomalies: dict[str, Any],
    scorecard: dict[str, Any],
) -> dict[str, Any]:
    """
    Aggregate all monitor outputs into a single self-assessment.

    The execution_mode_recommendation is advisory only. run.py uses it to
    reduce risk but MUST NOT use it to block exits.
    """
    reasons: list[str] = []

    # Core scores
    data_quality = _to_float(freshness.get("data_quality_score"), 1.0)
    model_alignment = _to_float(artifacts.get("model_alignment_score"), 1.0)
    anomaly_score = _to_float(anomalies.get("anomaly_score"), 0.0)

    # Regime risk from market context
    market_context = _read_json(MARKET_CONTEXT_PATH)
    if market_context:
        regime_risk, regime_reasons = _regime_risk_from_context(market_context)
        reasons.extend(regime_reasons)
    else:
        regime_risk = 0.70  # unknown context → elevated risk
        reasons.append("market_context_missing→elevated_risk")

    # Execution risk
    exec_risk_parts: list[float] = []
    if outcomes.get("issues"):
        exec_risk_parts.append(0.70)
        reasons.extend(outcomes["issues"])
    if outcomes.get("warnings"):
        exec_risk_parts.append(0.35)
        reasons.extend(outcomes.get("warnings", []))
    # Graduated anomaly → execution risk: a maxed anomaly_score from pre-market missing files
    # should not immediately push execution_risk to 1.0 and trigger HALT_NEW_BUYS.
    if anomaly_score > 0.75:
        exec_risk_parts.append(anomaly_score)
    elif anomaly_score > 0.5:
        exec_risk_parts.append(anomaly_score * 0.65)
    daily_loss = _to_float(outcomes.get("daily_pnl_pct"), 0.0)
    if daily_loss < -SA_DAILY_LOSS_HALT_PCT:
        exec_risk_parts.append(0.95)
        reasons.append(f"daily_loss={daily_loss:.2%}>halt_threshold")
    elif daily_loss < -SA_DAILY_LOSS_SAFE_PCT:
        exec_risk_parts.append(0.70)
        reasons.append(f"daily_loss={daily_loss:.2%}>safe_threshold")
    execution_risk = _clip01(max(exec_risk_parts) if exec_risk_parts else 0.10)

    # Recommended size multiplier (buy-side only — exits always at 1.0)
    size_mult = 1.0
    if data_quality < 0.30:
        size_mult = min(size_mult, 0.25)
        reasons.append(f"data_quality_very_low={data_quality:.2f}→size_25pct")
    elif data_quality < 0.50:
        size_mult = min(size_mult, 0.50)
        reasons.append(f"data_quality_low={data_quality:.2f}→size_50pct")
    elif data_quality < 0.70:
        size_mult = min(size_mult, 0.75)

    if regime_risk > 0.70:
        size_mult = min(size_mult, 0.50)
        reasons.append(f"regime_risk_high={regime_risk:.2f}→size_50pct")
    elif regime_risk > 0.50:
        size_mult = min(size_mult, 0.75)

    if execution_risk > SA_SAFE_EXEC_RISK:
        size_mult = min(size_mult, 0.40)
        reasons.append(f"execution_risk_high={execution_risk:.2f}→size_40pct")
    elif execution_risk > 0.40:
        size_mult = min(size_mult, 0.70)

    if model_alignment < SA_MIN_MODEL_ALIGNMENT:
        size_mult = min(size_mult, 0.60)
        reasons.append(f"model_alignment_low={model_alignment:.2f}→size_60pct")

    # Options and pyramiding gates
    disable_options = (
        data_quality < SA_SAFE_DATA_QUALITY
        or regime_risk > SA_SAFE_REGIME_RISK
        or execution_risk > SA_SAFE_EXEC_RISK
        or model_alignment < SA_MIN_MODEL_ALIGNMENT
    )
    disable_pyramiding = (
        execution_risk > 0.40
        or regime_risk > SA_CAUTIOUS_REGIME_RISK
        or data_quality < 0.60
        or (outcomes.get("consecutive_losses", 0) >= SA_MAX_CONSECUTIVE_LOSSES)
    )

    # Confidence threshold adjustment
    if regime_risk > SA_SAFE_REGIME_RISK or data_quality < SA_SAFE_DATA_QUALITY:
        raise_confidence_by = 0.05
    elif regime_risk > SA_CAUTIOUS_REGIME_RISK or data_quality < 0.70:
        raise_confidence_by = 0.02
    else:
        raise_confidence_by = 0.0

    # Execution mode recommendation
    if (
        execution_risk >= SA_HALT_EXEC_RISK
        or data_quality < 0.20
        or "trade_permission=halt" in " ".join(reasons).lower()
        or "trade_permission=disabled" in " ".join(reasons).lower()
        or "daily_loss_halt" in " ".join(anomalies.get("anomalies", [{}])[0].get("code", "") if anomalies.get("anomalies") else "")
    ):
        mode = "HALT_NEW_BUYS"
        reasons.append("→HALT_NEW_BUYS")
    elif (
        data_quality < SA_SAFE_DATA_QUALITY
        or regime_risk > SA_SAFE_REGIME_RISK
        or model_alignment < SA_MIN_MODEL_ALIGNMENT
        or execution_risk >= SA_SAFE_EXEC_RISK
    ):
        mode = "SAFE"
        reasons.append("→SAFE")
    elif (
        regime_risk > SA_CAUTIOUS_REGIME_RISK
        or data_quality < 0.70
        or execution_risk > 0.40
    ):
        mode = "CAUTIOUS"
        reasons.append("→CAUTIOUS")
    else:
        mode = "NORMAL"

    # Dedup reasons
    reasons = list(dict.fromkeys(r for r in reasons if r))

    assessment = {
        "created_at_utc": _now_utc(),
        "data_tag": DATA_TAG,
        "execution_mode_recommendation": mode,
        "data_quality_score": round(data_quality, 3),
        "model_alignment_score": round(model_alignment, 3),
        "regime_risk_score": round(regime_risk, 3),
        "execution_risk_score": round(execution_risk, 3),
        "anomaly_score": round(anomaly_score, 3),
        "recommended_size_multiplier": round(size_mult, 3),
        "disable_options": disable_options,
        "disable_pyramiding": disable_pyramiding,
        "raise_confidence_threshold_by": round(raise_confidence_by, 3),
        "reasons": reasons,
        "health_score": round(_to_float(health.get("health_score"), 1.0), 3),
        "market_bias": str(market_context.get("overall_market_bias", "unknown")) if market_context else "unknown",
        "trade_permission": str(market_context.get("trade_permission", "unknown")) if market_context else "unknown",
    }

    print(
        f"[self-awareness] assessment: mode={mode} "
        f"dq={data_quality:.2f} ma={model_alignment:.2f} "
        f"regime_risk={regime_risk:.2f} exec_risk={execution_risk:.2f} "
        f"size_mult={size_mult:.2f} "
        f"disable_options={disable_options} disable_pyramiding={disable_pyramiding}"
    )
    return assessment


# =============================================================================
# Public API for run.py integration
# =============================================================================

def load_self_assessment() -> dict[str, Any]:
    """
    Load today's self-assessment from disk. Returns a safe default (NORMAL mode,
    size_mult=1.0) if the file is missing, stale, or unreadable.

    run.py should call this during startup and use the result to adjust risk
    parameters for BUY orders only. SELL/exit orders must never depend on this.
    """
    default: dict[str, Any] = {
        "execution_mode_recommendation": "NORMAL",
        "data_quality_score": 1.0,
        "model_alignment_score": 1.0,
        "regime_risk_score": 0.0,
        "execution_risk_score": 0.0,
        "recommended_size_multiplier": 1.0,
        "disable_options": False,
        "disable_pyramiding": False,
        "raise_confidence_threshold_by": 0.0,
        "reasons": ["assessment_not_available"],
        "source": "default",
    }
    if not SELF_ASSESSMENT_PATH.exists():
        return default
    age = _file_age_minutes(SELF_ASSESSMENT_PATH)
    if age is not None and age > SA_CRITICAL_FILE_AGE_MINUTES:
        default["reasons"] = [f"assessment_stale_age={age:.0f}m"]
        return default
    try:
        data = json.loads(SELF_ASSESSMENT_PATH.read_text(encoding="utf-8", errors="ignore"))
        data["source"] = "file"
        return data
    except Exception:
        return default


def self_assessment_size_multiplier(assessment: dict[str, Any] | None = None) -> float:
    """Return the recommended buy-size multiplier (0-1). Safe default is 1.0."""
    if assessment is None:
        assessment = load_self_assessment()
    return _clip01(_to_float(assessment.get("recommended_size_multiplier"), 1.0))


def self_assessment_mode(assessment: dict[str, Any] | None = None) -> str:
    """Return the recommended execution mode string."""
    if assessment is None:
        assessment = load_self_assessment()
    return str(assessment.get("execution_mode_recommendation", "NORMAL"))


def self_assessment_raises_confidence(assessment: dict[str, Any] | None = None) -> float:
    """Return how much to raise the buy confidence threshold (additive, >= 0)."""
    if assessment is None:
        assessment = load_self_assessment()
    return max(0.0, _to_float(assessment.get("raise_confidence_threshold_by"), 0.0))


# =============================================================================
# Main
# =============================================================================

def run_self_assessment(dry_run: bool = False) -> dict[str, Any]:
    """
    Execute all monitors and write outputs. Returns the self_assessment dict.
    Never raises — all exceptions are caught and result in conservative defaults.
    """
    if not SA_ENABLED:
        print("[self-awareness] disabled via SA_ENABLED=0")
        return load_self_assessment()

    print(f"[self-awareness] starting assessment data_tag={DATA_TAG}")

    try:
        health = check_system_health()
    except Exception as e:
        print(f"[self-awareness] system_health failed: {e}")
        health = {"health_score": 0.50, "issues": [f"monitor_error:{e}"], "warnings": []}

    try:
        freshness = check_data_freshness()
    except Exception as e:
        print(f"[self-awareness] data_freshness failed: {e}")
        freshness = {"data_quality_score": 0.30, "items": {}, "semantic_issues": [f"monitor_error:{e}"]}

    try:
        artifacts = check_model_artifacts()
    except Exception as e:
        print(f"[self-awareness] model_artifacts failed: {e}")
        artifacts = {"model_alignment_score": 0.50, "artifacts": {}, "issues": [f"monitor_error:{e}"]}

    try:
        outcomes = analyze_trade_outcomes()
    except Exception as e:
        print(f"[self-awareness] trade_outcomes failed: {e}")
        outcomes = {"issues": [], "warnings": [], "daily_pnl_pct": 0.0, "consecutive_losses": 0}

    try:
        anomalies = detect_anomalies(health, freshness, artifacts, outcomes)
    except Exception as e:
        print(f"[self-awareness] anomaly_detection failed: {e}")
        anomalies = {"anomaly_score": 0.0, "anomaly_count": 0, "anomalies": []}

    try:
        scorecard = build_model_scorecard(artifacts)
    except Exception as e:
        print(f"[self-awareness] model_scorecard failed: {e}")
        scorecard = {"models": {}, "created_at_utc": _now_utc()}

    try:
        assessment = build_self_assessment(health, freshness, artifacts, outcomes, anomalies, scorecard)
    except Exception as e:
        print(f"[self-awareness] self_assessment failed: {e}")
        assessment = {
            "created_at_utc": _now_utc(),
            "data_tag": DATA_TAG,
            "execution_mode_recommendation": "SAFE",
            "data_quality_score": 0.0,
            "model_alignment_score": 0.0,
            "regime_risk_score": 1.0,
            "execution_risk_score": 1.0,
            "recommended_size_multiplier": 0.25,
            "disable_options": True,
            "disable_pyramiding": True,
            "raise_confidence_threshold_by": 0.10,
            "reasons": [f"assessment_critical_failure:{e}"],
        }

    if dry_run:
        print("[self-awareness] dry-run mode — not writing any files")
        print(json.dumps(assessment, indent=2, default=str))
        return assessment

    # Write outputs
    try:
        _write_json(SYSTEM_HEALTH_PATH, {**health, "anomalies": anomalies.get("anomalies", [])})
        print(f"[self-awareness] wrote {SYSTEM_HEALTH_PATH}")
    except Exception as e:
        print(f"[self-awareness] write system_health failed: {e}")

    try:
        _write_json(MODEL_SCORECARD_PATH, scorecard)
        print(f"[self-awareness] wrote {MODEL_SCORECARD_PATH}")
    except Exception as e:
        print(f"[self-awareness] write model_scorecard failed: {e}")

    try:
        _write_json(SELF_ASSESSMENT_PATH, assessment)
        print(f"[self-awareness] wrote {SELF_ASSESSMENT_PATH}")
    except Exception as e:
        print(f"[self-awareness] write self_assessment failed: {e}")

    # Audit trail
    try:
        journal_rows = _read_csv_rows(JOURNAL_PATH)
        gate_summary = audit_gate_decisions(journal_rows)
        log_decision_audit("self_assessment_complete", {
            "gate_summary": gate_summary,
            "freshness": {"data_quality_score": freshness.get("data_quality_score")},
            "artifacts": {"model_alignment_score": artifacts.get("model_alignment_score")},
            "outcomes": {"daily_pnl_pct": outcomes.get("daily_pnl_pct"), "consecutive_losses": outcomes.get("consecutive_losses")},
            "anomaly_count": anomalies.get("anomaly_count", 0),
        }, assessment_snapshot=assessment)
        print(f"[self-awareness] appended to {DECISION_AUDIT_PATH}")
    except Exception as e:
        print(f"[self-awareness] audit log write failed: {e}")

    print(
        f"[self-awareness] complete → mode={assessment['execution_mode_recommendation']} "
        f"size_mult={assessment['recommended_size_multiplier']:.2f} "
        f"disable_options={assessment['disable_options']} "
        f"disable_pyramiding={assessment['disable_pyramiding']}"
    )
    return assessment


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    result = run_self_assessment(dry_run=dry_run)
    sys.exit(0)
