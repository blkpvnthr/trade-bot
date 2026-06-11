#!/usr/bin/env python3
"""
run_optimizer_agent.py

Analyzes execution journals, model metrics, and live experience replay data
to diagnose why trades are being blocked, assess gate effectiveness, and
produce ranked parameter-tuning recommendations for run.py.

Standalone script (does NOT import run.py).
Outputs: system-data/YYYYMMDD/YYYYMMDD_optimizer_report.json
         system-data/YYYYMMDD/YYYYMMDD_optimizer_env_patch.txt  (optional env overrides)

Usage:
    python run_optimizer_agent.py
    python run_optimizer_agent.py --date 20260526
    python run_optimizer_agent.py --days 7
    python run_optimizer_agent.py --write-env-patch
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Environment + paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

ET = ZoneInfo("America/New_York")
TODAY_TAG = datetime.now(ET).strftime("%Y%m%d")

SYSTEM_DATA = PROJECT_ROOT / "system-data"
DAILY_OUTPUTS = PROJECT_ROOT / "daily_system_outputs"
EXEC_LOGS = PROJECT_ROOT / "logs" / "execution"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _json_load(path: Path) -> dict | list | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _csv_load(path: Path, **kwargs) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except Exception:
        return None


def load_experience_replay(date_tag: str) -> pd.DataFrame | None:
    # Try CSV first; fall back to JSONL (the live executor writes .jsonl, not .csv)
    csv_path = EXEC_LOGS / date_tag / f"{date_tag}_live_experience_replay.csv"
    df = _csv_load(csv_path)
    if df is not None and not df.empty:
        return df

    jsonl_path = EXEC_LOGS / date_tag / f"{date_tag}_live_experience_replay.jsonl"
    if not jsonl_path.exists():
        return None
    try:
        rows = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
        if not rows:
            return None
        return pd.DataFrame(rows)
    except Exception:
        return None


def load_executor_journal(date_tag: str) -> pd.DataFrame | None:
    path = EXEC_LOGS / date_tag / f"{date_tag}_executor_journal.csv"
    df = _csv_load(path)
    if df is None or df.empty:
        return None
    return df


def load_forecast_metrics(date_tag: str) -> dict:
    path = SYSTEM_DATA / date_tag / f"{date_tag}_forecast_engine_metrics.json"
    return _json_load(path) or {}


def load_model_scorecard(date_tag: str) -> dict:
    path = SYSTEM_DATA / date_tag / f"{date_tag}_model_scorecard.json"
    return _json_load(path) or {}


def load_self_assessment(date_tag: str) -> dict:
    path = SYSTEM_DATA / date_tag / f"{date_tag}_self_assessment.json"
    return _json_load(path) or {}


def load_system_health(date_tag: str) -> dict:
    path = SYSTEM_DATA / date_tag / f"{date_tag}_system_health.json"
    data = _json_load(path)
    if data is None:
        # File doesn't exist yet — pre-market / session not started. Use sentinel so
        # callers can distinguish "truly zero" from "file not created yet".
        return {"health_score": None, "file_missing": True, "issues": [], "journal_error_events": 0}
    return data


def load_evolution_metrics(date_tag: str) -> dict:
    path = SYSTEM_DATA / date_tag / f"{date_tag}_evolution_strategy_metrics.json"
    return _json_load(path) or {}


def load_premarket_plan(date_tag: str) -> dict:
    path = SYSTEM_DATA / date_tag / f"{date_tag}_premarket_trade_plan.json"
    return _json_load(path) or {}


def load_market_context(date_tag: str) -> dict:
    p1 = DAILY_OUTPUTS / date_tag / f"{date_tag}_market_context.json"
    p2 = SYSTEM_DATA / date_tag / f"{date_tag}_market_context.json"
    return _json_load(p1) or _json_load(p2) or {}


def available_date_tags(n_days: int) -> list[str]:
    """Return date tags for the last n trading days that have execution logs."""
    tags = []
    cursor = datetime.now(ET).date()
    checked = 0
    while len(tags) < n_days and checked < 60:
        tag = cursor.strftime("%Y%m%d")
        if (EXEC_LOGS / tag).exists():
            tags.append(tag)
        cursor -= timedelta(days=1)
        checked += 1
    return tags


# ---------------------------------------------------------------------------
# Skip reason parser
# ---------------------------------------------------------------------------

_SKIP_FIELDS_RE = re.compile(r"([a-z_]+):([0-9.]+)")

def parse_skip_reason(reason: str) -> dict[str, Any]:
    """Break a skip reason string into structured fields."""
    if not isinstance(reason, str):
        return {"raw": ""}
    parts = [p.strip() for p in reason.split(",") if p.strip()]
    flags = [p for p in parts if ":" not in p]
    kv = {}
    for p in parts:
        m = _SKIP_FIELDS_RE.match(p)
        if m:
            kv[m.group(1)] = float(m.group(2))
    return {"raw": reason, "flags": flags, "kv": kv}


def extract_skip_gate(reason: str) -> str:
    """Return the primary gate name from a skip reason."""
    if not isinstance(reason, str) or not reason:
        return "unknown"
    if reason.startswith("price_direction_not_confirmed"):
        return "direction_gate"
    if reason.startswith("forecast_"):
        return "forecast_gate"
    if reason.startswith("ml_"):
        return "ml_gate"
    if "execution_alpha" in reason:
        return "alpha_gate"
    if "stale" in reason:
        return "staleness_gate"
    if "safe_mode" in reason:
        return "safe_mode"
    if "structure" in reason.lower():
        return "structure_gate"
    if "confidence" in reason.lower():
        return "confidence_gate"
    return reason.split(":")[0][:40]


# ---------------------------------------------------------------------------
# Gate effectiveness analysis
# ---------------------------------------------------------------------------

def analyze_skip_patterns(df: pd.DataFrame) -> dict[str, Any]:
    """
    For a replay/journal dataframe, compute:
    - Top skip gates by frequency
    - ML probability distribution of blocked trades
    - Direction score distributions for blocked vs approved
    - Gate co-occurrence analysis
    """
    skipped = df[df["status"] == "skipped"].copy() if "status" in df.columns else pd.DataFrame()
    executed = df[(df["executed"] == True) | (df["status"] == "executed")].copy() if "status" in df.columns else pd.DataFrame()

    if skipped.empty and "event" in df.columns:
        skipped = df[df["event"] == "SKIP"].copy()
        executed = df[df["event"].isin(["ORDER", "FILLED", "BUY", "SELL"])].copy()

    results: dict[str, Any] = {}

    # --- Gate frequency ---
    if "message" in skipped.columns:
        gate_counts: Counter = Counter()
        sub_reasons: Counter = Counter()
        ml_probs_blocked: list[float] = []
        dir_scores_1m: list[float] = []
        dir_scores_5m: list[float] = []
        dir_scores_15m: list[float] = []

        for _, row in skipped.iterrows():
            msg = str(row.get("message", ""))
            # 20260528+ journals have a separate skip_reason column; prefer it when message is blank
            skip_reason_col = str(row.get("skip_reason", ""))
            if not msg.strip() or msg == "nan":
                msg = skip_reason_col
            reason = str(row.get("reason", msg)) or skip_reason_col
            gate = extract_skip_gate(msg or reason)
            gate_counts[gate] += 1

            parsed = parse_skip_reason(msg or reason)
            for flag in parsed["flags"]:
                sub_reasons[flag] += 1
            if "ml_next_bar_prob" in parsed["kv"]:
                ml_probs_blocked.append(parsed["kv"]["ml_next_bar_prob"])

            if "dir_1m_direction_score" in row:
                v = pd.to_numeric(row["dir_1m_direction_score"], errors="coerce")
                if pd.notna(v):
                    dir_scores_1m.append(float(v))
            if "dir_5m_direction_score" in row:
                v = pd.to_numeric(row["dir_5m_direction_score"], errors="coerce")
                if pd.notna(v):
                    dir_scores_5m.append(float(v))
            if "dir_15m_direction_score" in row:
                v = pd.to_numeric(row["dir_15m_direction_score"], errors="coerce")
                if pd.notna(v):
                    dir_scores_15m.append(float(v))

        results["gate_counts"] = dict(gate_counts.most_common(20))
        results["sub_reason_counts"] = dict(sub_reasons.most_common(20))
        results["n_skipped"] = len(skipped)
        results["n_executed"] = len(executed)
        results["skip_rate"] = len(skipped) / max(len(skipped) + len(executed), 1)

        if ml_probs_blocked:
            arr = np.array(ml_probs_blocked)
            results["ml_prob_blocked"] = {
                "count": len(arr),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "pct_above_0_45": float((arr >= 0.45).mean()),
                "pct_above_0_48": float((arr >= 0.48).mean()),
                "pct_above_0_50": float((arr >= 0.50).mean()),
            }

        for label, scores in [("1m", dir_scores_1m), ("5m", dir_scores_5m), ("15m", dir_scores_15m)]:
            if scores:
                arr = np.array(scores)
                results[f"dir_score_{label}_blocked"] = {
                    "count": len(arr),
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "pct_above_0_60": float((arr >= 0.60).mean()),
                    "pct_above_0_70": float((arr >= 0.70).mean()),
                }

    elif "message" in df.columns or "skip_reason" in df.columns:
        # Journal format (both old single-message and new skip_reason+message columns)
        event_col = df["event"] if "event" in df.columns else pd.Series(dtype=str)
        skip_rows = df[event_col == "SKIP"] if not event_col.empty else pd.DataFrame()
        gate_counts: Counter = Counter()
        for _, row in skip_rows.iterrows():
            msg = str(row.get("message", ""))
            skip_reason_col = str(row.get("skip_reason", ""))
            text = msg if (msg.strip() and msg != "nan") else skip_reason_col
            gate_counts[extract_skip_gate(text)] += 1
        results["gate_counts"] = dict(gate_counts.most_common(20))
        results["n_skipped"] = len(skip_rows)
        results["n_executed"] = len(df[event_col.isin(["ORDER", "BUY", "SELL"])]) if not event_col.empty else 0
        results["skip_rate"] = results["n_skipped"] / max(results["n_skipped"] + results["n_executed"], 1)

    return results


# ---------------------------------------------------------------------------
# Model health check
# ---------------------------------------------------------------------------

def diagnose_model_layers(date_tag: str) -> dict[str, Any]:
    """Check each model layer for health, effectiveness, and calibration."""
    scorecard = load_model_scorecard(date_tag)
    forecast_metrics = load_forecast_metrics(date_tag)
    es_metrics = load_evolution_metrics(date_tag)
    self_assessment = load_self_assessment(date_tag)
    health = load_system_health(date_tag)

    # When today's scorecard doesn't exist yet (pre-market / session not started),
    # fall back to yesterday's scorecard so we don't flood the report with stale-file
    # false positives before the session has had a chance to build today's artifacts.
    yesterday_tag = (datetime.strptime(date_tag, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
    scorecard_missing = not scorecard or not scorecard.get("models")
    if scorecard_missing:
        scorecard = load_model_scorecard(yesterday_tag) or {}
    if not es_metrics or not es_metrics.get("symbols_trained"):
        es_metrics = load_evolution_metrics(yesterday_tag) or {}

    issues: list[str] = []
    warnings_: list[str] = []
    layers: dict[str, Any] = {}

    # --- RL Intraday ---
    # Check the actual file on disk first — the scorecard may be from yesterday when the
    # model didn't exist yet (pre-market fallback), so a stale "exists=False" is a false positive.
    _rl_model_file = SYSTEM_DATA / date_tag / f"{date_tag}_rl_intraday_structure_model.zip"
    rl = scorecard.get("models", {}).get("rl_intraday", {})
    _rl_exists = _rl_model_file.exists() or rl.get("exists", False)
    if not _rl_exists:
        issues.append("rl_intraday: model file missing — timing gate is disabled or defaulting to neutral")
        layers["rl_intraday"] = {"status": "missing", "impact": "high"}
    else:
        layers["rl_intraday"] = {"status": "ok"}

    # --- Evolution Strategy ---
    es = scorecard.get("models", {}).get("evolution_strategy", {})
    n_trained = es.get("symbols_trained", 0)
    n_skipped = es.get("symbols_skipped", 0)
    # Also check raw metrics in case scorecard is stale
    if n_trained == 0:
        raw_trained = es_metrics.get("symbols_trained")
        if isinstance(raw_trained, list):
            n_trained = len(raw_trained)
        raw_skipped = es_metrics.get("symbols_skipped")
        if isinstance(raw_skipped, list):
            n_skipped = len(raw_skipped)
    if n_trained == 0:
        root_cause = "no intraday bar history available for any symbol"
        issues.append(f"evolution_strategy: 0/{n_trained + n_skipped} symbols trained — {root_cause}")
        layers["evolution_strategy"] = {
            "status": "broken",
            "trained": 0,
            "skipped": n_skipped,
            "impact": "medium",
            "root_cause": root_cause,
        }
    else:
        layers["evolution_strategy"] = {"status": "ok", "trained": n_trained}

    # --- Forecast Ensemble ---
    fc = scorecard.get("models", {}).get("forecast_ensemble", {})
    pct_approved = fc.get("pct_buy_approved", -1)
    if pct_approved is None:
        pct_approved = -1
    fc_source = fc.get("source") or "unknown"
    mean_score = fc.get("mean_ensemble_score") or 0.0  # guard None → 0.0
    if pct_approved == 0.0 or fc_source == "no_data":
        issues.append(
            f"forecast_ensemble: 0% buy-approved, source='{fc_source}' — ensemble not gating; all signals failing. "
            f"mean_score={mean_score:.3f} (near coin-flip at {mean_score:.3f})"
        )
        layers["forecast_ensemble"] = {
            "status": "ineffective",
            "pct_buy_approved": pct_approved,
            "mean_score": mean_score,
            "source": fc_source,
            "impact": "high",
            "note": "Consider lowering forecast_min_ensemble_score or switching to soft_gate mode",
        }
    elif pct_approved < 0.20:
        warnings_.append(f"forecast_ensemble: only {pct_approved:.0%} buy-approved (low)")
        layers["forecast_ensemble"] = {"status": "low", "pct_buy_approved": pct_approved}
    else:
        layers["forecast_ensemble"] = {"status": "ok", "pct_buy_approved": pct_approved}

    # --- Options Booster ---
    opt = scorecard.get("models", {}).get("options_booster", {})
    if opt.get("enabled") and opt.get("flow_candidates", 0) == 0:
        warnings_.append("options_booster: enabled but zero flow candidates — chain data may be empty")
        layers["options_booster"] = {"status": "no_candidates", "impact": "low"}
    else:
        layers["options_booster"] = {"status": "ok" if opt.get("enabled") else "disabled"}

    # --- Self-assessment alignment ---
    model_align = self_assessment.get("model_alignment_score", 1.0)
    if model_align < 0.50:
        warnings_.append(
            f"model_alignment_score={model_align:.2f} — models are disagreeing significantly; "
            "system may be in a conflicted state where no layer has enough conviction"
        )

    # --- Staleness ---
    health_issues = health.get("issues", [])
    for h in health_issues:
        if "stale" in h:
            issues.append(f"data_staleness: {h}")

    return {
        "layers": layers,
        "issues": issues,
        "warnings": warnings_,
        "model_alignment_score": model_align,
        # None when file doesn't exist yet (pre-market); 0 only when it truly scored 0.
        "health_score": health.get("health_score"),
        "journal_error_events": health.get("journal_error_events", 0),
    }


# ---------------------------------------------------------------------------
# Threshold sensitivity analysis
# ---------------------------------------------------------------------------

def analyze_threshold_sensitivity(df: pd.DataFrame) -> dict[str, Any]:
    """
    For direction-gated skips that have direction scores logged,
    compute how many would pass at various threshold levels.
    Returns suggested threshold adjustments with projected unlock rates.
    """
    if df is None or df.empty or "status" not in df.columns:
        return {}

    direction_blocked = df[
        (df["status"] == "skipped") &
        df["message"].fillna("").str.startswith("price_direction_not_confirmed")
    ].copy() if "message" in df.columns else pd.DataFrame()

    if direction_blocked.empty:
        return {}

    results: dict[str, Any] = {}
    total_blocked = len(direction_blocked)

    # ML probability gate sweep
    if "dir_1m_direction_score" in direction_blocked.columns:
        ml_col = None
        for col in ["ml_next_bar_probability", "ml_prob"]:
            if col in direction_blocked.columns:
                ml_col = col
                break

        if ml_col:
            probs = pd.to_numeric(direction_blocked[ml_col], errors="coerce").dropna()
            if not probs.empty:
                thresholds_ml = [0.40, 0.43, 0.45, 0.47, 0.48, 0.50]
                ml_sweep = {}
                for t in thresholds_ml:
                    would_pass = int((probs >= t).sum())
                    ml_sweep[str(t)] = {
                        "would_unlock": total_blocked - would_pass,
                        "pct_unlock": round((total_blocked - would_pass) / total_blocked, 3),
                    }
                results["ml_threshold_sweep"] = ml_sweep

        # Direction score sweep
        for tf, col in [("1m", "dir_1m_direction_score"), ("5m", "dir_5m_direction_score"), ("15m", "dir_15m_direction_score")]:
            if col in direction_blocked.columns:
                scores = pd.to_numeric(direction_blocked[col], errors="coerce").dropna()
                if not scores.empty:
                    sweep = {}
                    for t in [0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65]:
                        fails_at_t = int((scores < t).sum())
                        sweep[str(t)] = {
                            "fails_count": fails_at_t,
                            "pct_fail": round(fails_at_t / len(scores), 3),
                        }
                    results[f"direction_{tf}_sweep"] = sweep

        # Lower-low analysis
        ll_col_1m = "dir_1m_lower_lows"
        if ll_col_1m in direction_blocked.columns:
            ll_count = direction_blocked[ll_col_1m].astype(str).isin(["True", "1", "true"]).sum()
            results["lower_low_structure_blocks"] = {
                "count": int(ll_count),
                "pct_of_direction_blocks": round(ll_count / total_blocked, 3),
                "note": (
                    "require_no_lower_low_for_buy is blocking these. "
                    "Consider disabling for high-conviction direction scores (>0.80 combined)."
                ),
            }

    results["total_direction_blocked"] = total_blocked
    return results


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------

def generate_recommendations(
    skip_analysis: dict[str, Any],
    model_diagnosis: dict[str, Any],
    threshold_sensitivity: dict[str, Any],
    date_tag: str,
) -> list[dict[str, Any]]:
    """Produce ranked, actionable recommendations."""
    recs: list[dict[str, Any]] = []
    prio = 0

    def add(title, body, env_patch=None, impact="medium", confidence="medium"):
        nonlocal prio
        prio += 1
        recs.append({
            "rank": prio,
            "title": title,
            "description": body,
            "impact": impact,
            "confidence": confidence,
            "env_patch": env_patch or [],
        })

    # --- Model layer issues (deduped) ---
    _seen_issues: set[str] = set()
    stale_issues = [i for i in model_diagnosis.get("issues", []) if "stale" in i]

    for issue in model_diagnosis.get("issues", []):
        key = issue[:40]
        if key in _seen_issues:
            continue
        _seen_issues.add(key)

        if "rl_intraday" in issue:
            add(
                "Retrain RL intraday timing model",
                "The rl_intraday model is missing. The timing gate defaults to neutral (pass-all). "
                "Run: python rl_intraday_structure_trainer.py to generate a model checkpoint. "
                "Keep in shadow mode (RL_SHADOW_MODE=1) until validated.",
                impact="high",
                confidence="high",
            )
        elif "evolution_strategy" in issue and "0/" in issue:
            add(
                "Fix Evolution Strategy: no intraday bar history",
                "The ES model trained 0 symbols because no intraday bar history was available. "
                "Check that build_intraday_training_dataset.py ran successfully before run.py starts. "
                "Verify system-data/{date}/*_intraday_bars_1m.csv exists and is non-empty. "
                "Until fixed, ES signals are absent — the system is missing a signal layer.",
                env_patch=["# Ensure intraday dataset builder runs before main loop"],
                impact="medium",
                confidence="high",
            )
        elif "forecast_ensemble" in issue and "0%" in issue:
            add(
                "Fix Forecast Ensemble: 0% buy-approved (no_data source)",
                "The forecast ensemble is producing 0% buy approvals with source='no_data'. "
                "This means the underlying price history isn't loading, or all models are scoring near 0.50 "
                "and failing the forecast_min_ensemble_score threshold (currently 0.62). "
                "Options: (1) Lower threshold to 0.55, (2) verify _forecast_load_price_history returns data, "
                "(3) ensure forecast_ensemble_soft_gate=True so a failed forecast doesn't block otherwise-valid trades.",
                env_patch=["FORECAST_ENSEMBLE_SOFT_GATE=true"],
                impact="high",
                confidence="high",
            )
        elif "stale" in issue and len(stale_issues) > 0 and issue == stale_issues[0]:
            # Only emit one staleness recommendation covering all stale files
            stale_labels = ", ".join(
                i.split(": ", 1)[-1].split(" age=")[0] for i in stale_issues
            )
            add(
                "Stale planning data detected — re-run premarket collector",
                f"Stale files: {stale_labels}. Plans older than MAX_PLAN_AGE_MINUTES trigger SAFE mode, "
                "suppressing new buys. Ensure the premarket collector runs continuously during session hours. "
                "For post-session analysis, raise MAX_PLAN_AGE_MINUTES to avoid SAFE mode.",
                env_patch=["MAX_PLAN_AGE_MINUTES=30"],
                impact="high",
                confidence="high",
            )

    # --- Skip rate ---
    skip_rate = skip_analysis.get("skip_rate", 0)
    n_skipped = skip_analysis.get("n_skipped", 0)
    n_executed = skip_analysis.get("n_executed", 0)
    if skip_rate > 0.95 and n_skipped > 100:
        add(
            f"Critical: {skip_rate:.0%} skip rate ({n_skipped} skips vs {n_executed} orders)",
            "The system is effectively not trading. Nearly every candidate is being blocked before execution. "
            "This suggests gates are miscalibrated for current market conditions. "
            "Investigate: (1) direction gate thresholds, (2) ML probability gate, (3) structure requirements.",
            impact="critical",
            confidence="high",
        )

    # --- Direction gate ---
    gate_counts = skip_analysis.get("gate_counts", {})
    dir_blocks = gate_counts.get("direction_gate", 0)
    if dir_blocks > 50:
        ll_analysis = threshold_sensitivity.get("lower_low_structure_blocks", {})
        ll_pct = ll_analysis.get("pct_of_direction_blocks", 0)

        body = (
            f"direction_gate is blocking {dir_blocks} trades. "
            f"Lower-low structure criterion accounts for {ll_pct:.0%} of direction blocks. "
        )
        patches = []

        # ML prob recommendation
        ml_sweep = threshold_sensitivity.get("ml_threshold_sweep", {})
        if ml_sweep:
            # Find threshold that unlocks ~20% of blocked with ml_next_bar_prob_low
            best_t = 0.50
            for t_str, info in sorted(ml_sweep.items(), key=lambda x: float(x[0])):
                if info["pct_unlock"] >= 0.15:
                    best_t = float(t_str)
                    break
            if best_t < 0.50:
                body += (
                    f"Lowering ML_NEXT_BAR_MIN_BUY_PROBABILITY from 0.50 to {best_t} "
                    f"would unlock ~{ml_sweep.get(str(best_t), {}).get('pct_unlock', 0):.0%} of blocked trades."
                )
                patches.append(f"ML_NEXT_BAR_MIN_BUY_PROBABILITY={best_t}")

        if ll_pct > 0.40:
            body += (
                " require_no_lower_low_for_buy is a major blocker. "
                "Consider: (1) disable it entirely (REQUIRE_NO_LOWER_LOW_FOR_BUY=0), "
                "(2) override it when combined 1m+5m+15m direction scores are all >0.80."
            )
            patches.append("REQUIRE_NO_LOWER_LOW_FOR_BUY=0")

        add("Relax direction gate — dominant skip reason", body, env_patch=patches, impact="high", confidence="medium")

    # --- ML probability calibration ---
    ml_stats = skip_analysis.get("ml_prob_blocked", {})
    if ml_stats:
        mean_prob = ml_stats.get("mean", 0.5)
        pct_above_48 = ml_stats.get("pct_above_0_48", 0)
        if mean_prob >= 0.44 and pct_above_48 > 0.30:
            add(
                f"ML gate miscalibrated: mean blocked probability = {mean_prob:.3f}",
                f"{pct_above_48:.0%} of blocked trades had ML prob >= 0.48, just below the 0.50 threshold. "
                "The model is near coin-flip for these, suggesting the threshold is too tight relative to model uncertainty. "
                "Lowering ml_next_bar_min_buy_probability to 0.45 would reduce ML as a standalone gate "
                "while keeping it as a confidence modifier.",
                env_patch=["ML_NEXT_BAR_MIN_BUY_PROBABILITY=0.45"],
                impact="medium",
                confidence="medium",
            )

    # --- Direction score quality ---
    dir_1m = skip_analysis.get("dir_score_1m_blocked", {})
    if dir_1m:
        mean_1m = dir_1m.get("mean", 0)
        pct_above_70 = dir_1m.get("pct_above_0_70", 0)
        if mean_1m > 0.65 and pct_above_70 > 0.50:
            add(
                f"High-quality signals being blocked: mean 1m direction score = {mean_1m:.2f} on blocked trades",
                f"{pct_above_70:.0%} of blocked trades had 1m direction score > 0.70 — these are strong signals. "
                "The pre-trade direction gate is likely rejecting good entries due to structure flags "
                "(lower_low or combined_direction_weak) that conflict with higher-timeframe confirmation. "
                "Consider: relax min_pretrade_direction_score from 0.62 to 0.58 "
                "or allow entries when 1m+5m+15m are all >= 0.70 regardless of structure flags.",
                env_patch=[
                    "MIN_PRETRADE_DIRECTION_SCORE=0.58",
                ],
                impact="high",
                confidence="medium",
            )

    # --- Model alignment ---
    align = model_diagnosis.get("model_alignment_score", 1.0)
    if align < 0.45:
        add(
            f"Low model alignment score ({align:.2f}) — models are in conflict",
            "When model_alignment_score is low, the self-assessment may raise confidence thresholds "
            "or lower size multipliers, making it even harder to enter. "
            "With the ES and forecast layers broken, this is expected. "
            "Fix the broken layers first; the alignment score should improve naturally.",
            impact="medium",
            confidence="high",
        )

    # --- Multi-day gate counts: surface persistent patterns the single-day view misses ---
    multi_day_gates = skip_analysis.get("multi_day_gate_counts", {})
    single_day_gates = skip_analysis.get("gate_counts", {})
    for gate, count in multi_day_gates.items():
        # Only fire if multi-day count is significant AND not already covered by single-day analysis
        if count < 200:
            continue
        if gate in single_day_gates and single_day_gates[gate] > 50:
            continue  # already handled above
        if gate == "direction_gate":
            add(
                f"Persistent direction gate pressure: {count} blocks over multi-day window",
                f"direction_gate blocked {count} trades across recent sessions — far exceeding single-day counts. "
                "This is a structural issue, not a one-off. The lower_low_structure and ml_next_bar_prob_low "
                "sub-reasons are co-occurring on high-direction-score candidates (mean 1m score ~0.79). "
                "Recommended: (1) Set REQUIRE_NO_LOWER_LOW_FOR_BUY=0 to stop lower_low from being a hard block, "
                "(2) lower ML_NEXT_BAR_MIN_BUY_PROBABILITY from 0.50 to 0.45 since the model is near coin-flip "
                "on these trades.",
                env_patch=["REQUIRE_NO_LOWER_LOW_FOR_BUY=0", "ML_NEXT_BAR_MIN_BUY_PROBABILITY=0.45"],
                impact="high",
                confidence="high",
            )
        elif "existing open order" in gate and count > 100:
            add(
                f"Persistent 'existing open order' suppression: {count} blocks over multi-day window",
                f"'{gate}' blocked {count} candidates across recent sessions. "
                "This suggests the system is re-evaluating the same symbol every cycle while an order is pending. "
                "Consider: (1) increase order timeout before re-evaluating, "
                "(2) ensure filled orders are removed from the candidate queue promptly.",
                impact="medium",
                confidence="medium",
            )

    # --- System health (only if not already flagged by staleness, and file actually exists) ---
    _hs = model_diagnosis.get("health_score")
    if _hs is not None and _hs < 0.6 and not stale_issues:
        add(
            "System health low — diagnose SAFE mode triggers",
            f"health_score={model_diagnosis['health_score']:.2f}. "
            "SAFE mode is suppressing new buys. Check system health report for root cause.",
            impact="high",
            confidence="medium",
        )

    # --- Error events ---
    n_errors = model_diagnosis.get("journal_error_events", 0)
    if n_errors > 10:
        add(
            f"{n_errors} journal error events today",
            "High error count in execution journal suggests crashes or unhandled exceptions in "
            "order processing loops. Check logs for stack traces. Errors may be causing cycles to "
            "skip processing entirely.",
            impact="medium",
            confidence="high",
        )

    # --- Missing Python packages (yfinance / openai) ---
    # Detect from run.log if build_market_context failed due to missing packages
    _run_log = Path("logs/run.log")
    _missing_pkg_errors: set[str] = set()
    if _run_log.exists():
        try:
            import re as _re
            _today = date_tag  # YYYYMMDD string
            # Only scan last 500 lines to stay fast
            _lines = _run_log.read_text(errors="replace").splitlines()[-500:]
            for _ln in _lines:
                if "build_market_context failed" in _ln and "No module named" in _ln:
                    _m = _re.search(r"No module named '([^']+)'", _ln)
                    if _m:
                        _missing_pkg_errors.add(_m.group(1))
        except Exception:
            pass
    if _missing_pkg_errors:
        _pkgs = ", ".join(sorted(_missing_pkg_errors))
        add(
            f"Missing Python packages prevent live market-context: {_pkgs}",
            f"build_market_context failed with 'No module named ...' for: {_pkgs}. "
            "When the market context builder fails, the system falls back to a neutral/cautious context "
            "instead of assessing the real regime. This means bearish market days may still allow trades "
            "during the error window. Fix: run  pip install yfinance openai  in the same Python environment "
            "used to launch run.py. Then verify: python -c \"import yfinance, openai; print('ok')\".",
            impact="high",
            confidence="high",
        )

    return recs


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(date_tag: str, n_days: int = 1, write_env_patch: bool = False) -> dict[str, Any]:
    print(f"[optimizer] building report for date_tag={date_tag}, n_days={n_days}")

    # Load primary date data
    replay = load_experience_replay(date_tag)
    journal = load_executor_journal(date_tag)
    market_ctx = load_market_context(date_tag)

    # Fallback to journal if replay unavailable
    primary_df = replay if replay is not None else journal

    skip_analysis = analyze_skip_patterns(primary_df) if primary_df is not None else {}
    model_diagnosis = diagnose_model_layers(date_tag)
    threshold_sensitivity = (
        analyze_threshold_sensitivity(replay) if replay is not None else {}
    )

    # Multi-day aggregation if requested
    if n_days > 1:
        all_tags = available_date_tags(n_days)
        print(f"[optimizer] multi-day analysis over {len(all_tags)} days: {all_tags}")
        multi_skips: Counter = Counter()
        for tag in all_tags:
            df = load_experience_replay(tag)
            if df is None:
                df = load_executor_journal(tag)
            if df is not None:
                sa = analyze_skip_patterns(df)
                for k, v in sa.get("gate_counts", {}).items():
                    multi_skips[k] += v
        skip_analysis["multi_day_gate_counts"] = dict(multi_skips.most_common(20))

    recommendations = generate_recommendations(
        skip_analysis, model_diagnosis, threshold_sensitivity, date_tag
    )

    report = {
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "date_tag": date_tag,
        "market_bias": market_ctx.get("overall_market_bias", "unknown"),
        "trade_permission": market_ctx.get("trade_permission", "unknown"),
        "skip_analysis": skip_analysis,
        "model_diagnosis": model_diagnosis,
        "threshold_sensitivity": threshold_sensitivity,
        "recommendations": recommendations,
    }

    # Write report
    out_dir = SYSTEM_DATA / date_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{date_tag}_optimizer_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[optimizer] report written → {report_path}")

    # Write env patch (deduplicated by env key)
    if write_env_patch:
        patches: list[str] = []
        seen_keys: set[str] = set()
        for rec in recommendations:
            if rec.get("env_patch"):
                rec_patches = []
                for line in rec["env_patch"]:
                    key = line.split("=")[0].strip() if "=" in line else line
                    if key and key not in seen_keys:
                        seen_keys.add(key)
                        rec_patches.append(line)
                if rec_patches:
                    patches.append(f"# [{rec['rank']}] {rec['title']}")
                    patches.extend(rec_patches)
                    patches.append("")
        if patches:
            patch_path = out_dir / f"{date_tag}_optimizer_env_patch.txt"
            with open(patch_path, "w") as f:
                f.write("\n".join(patches))
            print(f"[optimizer] env patch written → {patch_path}")

    return report


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

def print_summary(report: dict[str, Any]) -> None:
    date_tag = report["date_tag"]
    sa = report.get("skip_analysis", {})
    md = report.get("model_diagnosis", {})
    recs = report.get("recommendations", [])

    print()
    print("=" * 72)
    print(f"  OPTIMIZER REPORT  [{date_tag}]")
    print("=" * 72)
    print(f"  Market bias   : {report.get('market_bias', '?')}")
    print(f"  Trade perm    : {report.get('trade_permission', '?')}")
    _hs_display = md.get("health_score")
    print(f"  Health score  : {f'{_hs_display:.2f}' if _hs_display is not None else 'n/a (pre-market)'}")
    print(f"  Model align   : {md.get('model_alignment_score', '?'):.2f}")
    print()
    print(f"  Skip rate     : {sa.get('skip_rate', 0):.1%}  "
          f"({sa.get('n_skipped', '?')} skips / {sa.get('n_executed', '?')} orders)")

    gc = sa.get("gate_counts", {})
    if gc:
        print(f"\n  Top skip gates:")
        for gate, count in sorted(gc.items(), key=lambda x: -x[1])[:8]:
            print(f"    {gate:<35} {count}")

    ml_stats = sa.get("ml_prob_blocked", {})
    if ml_stats:
        print(f"\n  ML prob on blocked trades: mean={ml_stats['mean']:.3f}  "
              f"pct>=0.48={ml_stats['pct_above_0_48']:.0%}  "
              f"pct>=0.50={ml_stats['pct_above_0_50']:.0%}")

    print(f"\n  Model layer issues ({len(md.get('issues', []))}):")
    for issue in md.get("issues", []):
        print(f"    ⚠  {issue[:80]}")

    print(f"\n  Recommendations ({len(recs)}):")
    for rec in recs:
        impact_sym = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(rec["impact"], "•")
        print(f"\n  [{rec['rank']}] {impact_sym} {rec['title']}")
        print(f"       {rec['description'][:160]}")
        if rec.get("env_patch"):
            print(f"       ENV: {' | '.join(rec['env_patch'][:3])}")

    print()
    print("=" * 72)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="run.py optimization agent")
    parser.add_argument("--date", default=TODAY_TAG, help="Date tag YYYYMMDD (default: today)")
    parser.add_argument("--days", type=int, default=1, help="Number of historical days to include")
    parser.add_argument("--write-env-patch", action="store_true", help="Write .env override suggestions")
    parser.add_argument("--json-only", action="store_true", help="Skip human-readable summary")
    args = parser.parse_args()

    report = build_report(
        date_tag=args.date,
        n_days=args.days,
        write_env_patch=args.write_env_patch,
    )

    if not args.json_only:
        print_summary(report)


if __name__ == "__main__":
    main()
