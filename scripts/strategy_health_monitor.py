#!/usr/bin/env python3
"""
scripts/strategy_health_monitor.py

8 health checks for each strategy variant:
  1. strategy_silent      — no heartbeat file or mtime > 15 min
  2. no_signals           — 0 signals generated in last N bars of replay
  3. no_universe          — 0 symbols in evaluated set
  4. stale_market_data    — most recent bar timestamp > 2 hours old
  5. stale_structure_maps — premarket pivot structure > 24 hours old
  6. stale_premarket_plan — premarket plan file > 24 hours old
  7. missing_evaluator    — research_rankings.json missing or > 48 hours old
  8. repeated_skip_reason — one skip reason accounts for > 80% of skips

Writes daily_system_outputs/health_report.json.

Usage:
  python scripts/strategy_health_monitor.py [--strategy STRATEGY_ID]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.registry import STRATEGY_REGISTRY

LOG_BASE    = PROJECT_ROOT / "logs" / "execution"
OUTPUT_DIR  = PROJECT_ROOT / "daily_system_outputs"

# Thresholds
HEARTBEAT_STALE_MINUTES   = 15
MARKET_DATA_STALE_HOURS   = 2
STRUCTURE_STALE_HOURS     = 24
PLAN_STALE_HOURS          = 24
RANKINGS_STALE_HOURS      = 48
SKIP_DOMINANCE_THRESHOLD  = 0.80   # single reason > 80% = repeated_skip_reason


# ─────────────────────────────────────────────────────────────────────────────
# Individual health checks
# ─────────────────────────────────────────────────────────────────────────────

def _check_strategy_silent(strategy_id: str) -> dict[str, Any]:
    """Check 1: heartbeat file freshness."""
    heartbeat_path = Path("/tmp/trader_heartbeat")

    # Per-strategy heartbeat if it exists (in case we add per-strategy files later)
    per_strategy = Path(f"/tmp/trader_heartbeat_{strategy_id}")
    path = per_strategy if per_strategy.exists() else heartbeat_path

    if not path.exists():
        return {"status": "warn", "message": "heartbeat file absent — pod may not be running"}

    age_minutes = (time.time() - path.stat().st_mtime) / 60
    if age_minutes > HEARTBEAT_STALE_MINUTES:
        return {"status": "warn",
                "message": f"heartbeat stale ({age_minutes:.0f} min old, threshold={HEARTBEAT_STALE_MINUTES} min)"}
    return {"status": "ok", "message": f"heartbeat {age_minutes:.1f} min old"}


def _check_no_signals(strategy_id: str) -> dict[str, Any]:
    """Check 2: 0 signals in replay output."""
    replay_path = OUTPUT_DIR / "replay_results.json"
    if not replay_path.exists():
        return {"status": "warn", "message": "replay_results.json missing — run replay_market_day.py"}

    try:
        data = json.loads(replay_path.read_text())
        for strat in data.get("strategies", []):
            if strat.get("strategy_id") == strategy_id:
                total = (strat.get("total_buy", 0) + strat.get("total_sell", 0) +
                         strat.get("total_short", 0) + strat.get("total_cover", 0))
                if total == 0:
                    return {"status": "warn", "message": "0 signals in last replay run"}
                return {"status": "ok", "message": f"{total} signals in last replay"}
        return {"status": "warn", "message": f"{strategy_id} not found in replay results"}
    except Exception as exc:
        return {"status": "error", "message": f"could not parse replay_results.json: {exc}"}


def _check_no_universe(strategy_id: str) -> dict[str, Any]:
    """Check 3: empty symbol universe in activity report."""
    activity_path = OUTPUT_DIR / "strategy_activity_report.json"
    if not activity_path.exists():
        return {"status": "warn", "message": "strategy_activity_report.json missing"}

    try:
        data = json.loads(activity_path.read_text())
        for strat in data.get("strategies", []):
            if strat.get("strategy_id") == strategy_id:
                symbols_evaluated = strat.get("symbols", {}).get("evaluated", [])
                if not symbols_evaluated:
                    return {"status": "warn", "message": "0 symbols evaluated — universe may be empty"}
                return {"status": "ok", "message": f"{len(symbols_evaluated)} symbols evaluated"}
        return {"status": "warn", "message": f"{strategy_id} not in activity report"}
    except Exception as exc:
        return {"status": "error", "message": f"could not parse activity report: {exc}"}


def _check_stale_market_data(strategy_id: str) -> dict[str, Any]:
    """Check 4: most recent bar timestamp in execution events."""
    log_dir = LOG_BASE / strategy_id
    events_files = sorted(log_dir.glob("*execution_events*.jsonl")) if log_dir.exists() else []
    if not events_files:
        return {"status": "warn", "message": "no execution events log found"}

    latest_ts: datetime | None = None
    try:
        with events_files[-1].open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    ts_str = rec.get("bar_time") or rec.get("timestamp") or rec.get("ts")
                    if ts_str:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if latest_ts is None or ts > latest_ts:
                            latest_ts = ts
                except Exception:
                    pass
    except Exception:
        pass

    if latest_ts is None:
        return {"status": "warn", "message": "no bar timestamps found in events log"}

    age_hours = (datetime.now(timezone.utc) - latest_ts).total_seconds() / 3600
    if age_hours > MARKET_DATA_STALE_HOURS:
        return {"status": "warn",
                "message": f"latest bar is {age_hours:.1f}h old (threshold={MARKET_DATA_STALE_HOURS}h)"}
    return {"status": "ok", "message": f"latest bar {age_hours:.2f}h old"}


def _check_stale_structure_maps(strategy_id: str) -> dict[str, Any]:
    """Check 5: premarket structure files freshness."""
    # Look for pivot structure / premarket plan files
    candidates = list(PROJECT_ROOT.glob("daily_system_outputs/premarket_plan*.json")) + \
                 list(PROJECT_ROOT.glob("logs/execution/*/premarket_structure*.json"))

    if not candidates:
        return {"status": "warn", "message": "no premarket structure map files found"}

    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    age_hours = (time.time() - newest.stat().st_mtime) / 3600
    if age_hours > STRUCTURE_STALE_HOURS:
        return {"status": "warn",
                "message": f"structure map is {age_hours:.1f}h old (threshold={STRUCTURE_STALE_HOURS}h)"}
    return {"status": "ok", "message": f"structure map {age_hours:.1f}h old"}


def _check_stale_premarket_plan(strategy_id: str) -> dict[str, Any]:
    """Check 6: premarket_trade_planner output freshness."""
    candidates = list(PROJECT_ROOT.glob("**/premarket_plan*.json")) + \
                 list(PROJECT_ROOT.glob("**/trade_plan*.json"))

    if not candidates:
        return {"status": "warn", "message": "no premarket plan files found"}

    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    age_hours = (time.time() - newest.stat().st_mtime) / 3600
    if age_hours > PLAN_STALE_HOURS:
        return {"status": "warn",
                "message": f"premarket plan is {age_hours:.1f}h old (threshold={PLAN_STALE_HOURS}h)"}
    return {"status": "ok", "message": f"premarket plan {age_hours:.1f}h old"}


def _check_missing_evaluator(strategy_id: str) -> dict[str, Any]:
    """Check 7: research_rankings.json presence and freshness (same for all strategies)."""
    rankings_path = OUTPUT_DIR / "research_rankings.json"
    if not rankings_path.exists():
        return {"status": "warn", "message": "research_rankings.json missing — run evaluate_variants.py"}

    age_hours = (time.time() - rankings_path.stat().st_mtime) / 3600
    if age_hours > RANKINGS_STALE_HOURS:
        return {"status": "warn",
                "message": f"rankings are {age_hours:.1f}h old (threshold={RANKINGS_STALE_HOURS}h)"}
    return {"status": "ok", "message": f"rankings {age_hours:.1f}h old"}


def _check_repeated_skip_reason(strategy_id: str) -> dict[str, Any]:
    """Check 8: single skip reason dominates."""
    jsonl_path = LOG_BASE / strategy_id / "skipped_signals.jsonl"
    if not jsonl_path.exists():
        return {"status": "ok", "message": "no skip log (no paper trades yet)"}

    from collections import Counter
    by_reason: Counter = Counter()
    total = 0
    try:
        with jsonl_path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    by_reason[rec.get("reason", "unknown")] += 1
                    total += 1
                except Exception:
                    pass
    except Exception:
        return {"status": "error", "message": "could not read skipped_signals.jsonl"}

    if total == 0:
        return {"status": "ok", "message": "skip log empty"}

    top_reason, top_count = by_reason.most_common(1)[0]
    dominance = top_count / total
    if dominance > SKIP_DOMINANCE_THRESHOLD:
        return {"status": "warn",
                "message": f"'{top_reason}' accounts for {dominance*100:.0f}% of skips "
                           f"({top_count}/{total}) — investigate"}
    return {"status": "ok",
            "message": f"top skip reason '{top_reason}' at {dominance*100:.0f}% ({top_count}/{total})"}


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

CHECKS = [
    ("strategy_silent",     _check_strategy_silent),
    ("no_signals",          _check_no_signals),
    ("no_universe",         _check_no_universe),
    ("stale_market_data",   _check_stale_market_data),
    ("stale_structure_maps",_check_stale_structure_maps),
    ("stale_premarket_plan",_check_stale_premarket_plan),
    ("missing_evaluator",   _check_missing_evaluator),
    ("repeated_skip_reason",_check_repeated_skip_reason),
]


def _health_for_strategy(strategy_id: str) -> dict[str, Any]:
    check_results: dict[str, Any] = {}
    overall = "ok"

    for check_name, check_fn in CHECKS:
        try:
            result = check_fn(strategy_id)
        except Exception as exc:
            result = {"status": "error", "message": f"check raised: {exc}"}

        check_results[check_name] = result
        if result["status"] in ("warn", "error") and overall == "ok":
            overall = result["status"]

    warnings = [name for name, r in check_results.items() if r["status"] == "warn"]
    errors   = [name for name, r in check_results.items() if r["status"] == "error"]

    return {
        "strategy_id": strategy_id,
        "overall": overall,
        "warnings": warnings,
        "errors": errors,
        "checks": check_results,
    }


def run_health_monitor(
    strategy_filter: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    strategies = list(STRATEGY_REGISTRY.keys())
    if strategy_filter:
        strategies = [s for s in strategies if s == strategy_filter]

    reports: list[dict[str, Any]] = []
    for sid in strategies:
        rep = _health_for_strategy(sid)
        reports.append(rep)

        if verbose:
            icon = {"ok": "OK  ", "warn": "WARN", "error": "ERR "}[rep["overall"]]
            print(f"  [{icon}] {sid}")
            for name in rep["warnings"] + rep["errors"]:
                msg = rep["checks"][name]["message"]
                prefix = "WARN" if rep["checks"][name]["status"] == "warn" else "ERR "
                print(f"         [{prefix}] {name}: {msg}")

    total_ok   = sum(1 for r in reports if r["overall"] == "ok")
    total_warn = sum(1 for r in reports if r["overall"] == "warn")
    total_err  = sum(1 for r in reports if r["overall"] == "error")

    summary = {
        "strategies": reports,
        "summary": {
            "total": len(reports),
            "ok": total_ok,
            "warn": total_warn,
            "error": total_err,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy health monitor — 8 checks per variant")
    parser.add_argument("--strategy", default=None, help="Limit to one strategy_id")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.quiet:
        print("=== Strategy Health Monitor ===\n")

    summary = run_health_monitor(strategy_filter=args.strategy, verbose=not args.quiet)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "health_report.json"
    out_path.write_text(json.dumps(summary, indent=2))

    if not args.quiet:
        s = summary["summary"]
        print(f"\nResult: {s['ok']} ok / {s['warn']} warn / {s['error']} error")
        print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
