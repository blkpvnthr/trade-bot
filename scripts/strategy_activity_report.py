#!/usr/bin/env python3
"""
scripts/strategy_activity_report.py

Parse execution logs to produce a per-strategy activity summary:
  - Signals generated / skipped / total
  - Orders placed / rejected (from executor journal CSV)
  - Top skip reasons (from skipped_signals.jsonl)
  - Top gating failures
  - Symbols evaluated
  - Strategy uptime estimate (heartbeat files)

Writes daily_system_outputs/strategy_activity_report.json.

Usage:
  python scripts/strategy_activity_report.py [--date YYYY-MM-DD] [--strategy STRATEGY_ID]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.registry import STRATEGY_REGISTRY

LOG_BASE    = PROJECT_ROOT / "logs" / "execution"
OUTPUT_DIR  = PROJECT_ROOT / "daily_system_outputs"


# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_skipped_signals(jsonl_path: Path) -> dict[str, Any]:
    """Parse skipped_signals.jsonl → counts by reason, gating_condition, symbol."""
    if not jsonl_path.exists():
        return {"total": 0, "by_reason": {}, "by_gating": {}, "by_symbol": {}}

    by_reason: Counter = Counter()
    by_gating: Counter = Counter()
    by_symbol: Counter = Counter()
    total = 0

    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                total += 1
                by_reason[rec.get("reason", "unknown")] += 1
                gc = rec.get("gating_condition", "")
                if gc:
                    by_gating[gc] += 1
                sym = rec.get("symbol", "")
                if sym:
                    by_symbol[sym] += 1
            except (json.JSONDecodeError, AttributeError):
                pass

    return {
        "total": total,
        "by_reason": dict(by_reason.most_common(10)),
        "by_gating": dict(by_gating.most_common(10)),
        "by_symbol": dict(by_symbol.most_common(10)),
    }


def _parse_executor_journal(csv_path: Path) -> dict[str, Any]:
    """Parse {DATE}_executor_journal.csv → order counts, symbols traded."""
    if not csv_path.exists():
        return {"orders_placed": 0, "orders_rejected": 0, "symbols_traded": []}

    try:
        import pandas as pd
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return {"orders_placed": 0, "orders_rejected": 0, "symbols_traded": []}

    # Detect status column (varies by run.py version)
    status_col = next((c for c in df.columns if c.lower() in ("status", "order_status", "result")), None)
    sym_col    = next((c for c in df.columns if c.lower() in ("symbol", "sym")), None)

    placed   = len(df)
    rejected = 0
    if status_col:
        rejected = int((df[status_col].str.upper().isin(["REJECTED", "FAILED", "ERROR"])).sum())

    symbols: list[str] = []
    if sym_col:
        symbols = df[sym_col].dropna().unique().tolist()

    return {
        "orders_placed": placed,
        "orders_rejected": rejected,
        "symbols_traded": symbols,
    }


def _parse_metrics_json(metrics_path: Path) -> dict[str, Any]:
    """Parse metrics.json → equity, exposure, heartbeat info."""
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text())
    except Exception:
        return {}


def _estimate_uptime(log_dir: Path, trade_date: date) -> str:
    """
    Estimate uptime from execution_events.jsonl timestamps.
    Returns human-readable string like "6h 32m" or "unknown".
    """
    events_path = log_dir / f"{trade_date}_{trade_date}_execution_events.jsonl"
    # Try date-prefixed name
    candidates = sorted(log_dir.glob(f"*{trade_date}*execution_events*.jsonl"))
    if not candidates:
        candidates = sorted(log_dir.glob("*execution_events*.jsonl"))
    if not candidates:
        return "unknown"

    timestamps: list[datetime] = []
    with candidates[-1].open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                ts_str = rec.get("timestamp") or rec.get("ts") or rec.get("time")
                if ts_str:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    timestamps.append(ts)
            except Exception:
                pass

    if len(timestamps) < 2:
        return "unknown"

    span = timestamps[-1] - timestamps[0]
    hours, remainder = divmod(int(span.total_seconds()), 3600)
    minutes = remainder // 60
    return f"{hours}h {minutes}m"


def _collect_signals_from_execution_events(log_dir: Path, trade_date: date) -> dict[str, Any]:
    """Parse execution_events.jsonl for signal counts."""
    candidates = sorted(log_dir.glob(f"*{trade_date}*execution_events*.jsonl"))
    if not candidates:
        candidates = sorted(log_dir.glob("*execution_events*.jsonl"))
    if not candidates:
        return {"generated": 0, "symbols_evaluated": []}

    generated = 0
    symbols: set[str] = set()

    with candidates[-1].open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                etype = rec.get("event_type", rec.get("type", ""))
                if etype in ("signal_generated", "signal", "entry_signal", "buy_signal", "short_signal"):
                    generated += 1
                sym = rec.get("symbol", "")
                if sym:
                    symbols.add(sym)
            except Exception:
                pass

    return {"generated": generated, "symbols_evaluated": sorted(symbols)}


# ─────────────────────────────────────────────────────────────────────────────
# Per-strategy aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _report_for_strategy(strategy_id: str, trade_date: date) -> dict[str, Any]:
    log_dir = LOG_BASE / strategy_id
    date_str = str(trade_date)

    skipped = _parse_skipped_signals(log_dir / "skipped_signals.jsonl")

    # Find journal CSV for this date
    journal_candidates = sorted(log_dir.glob(f"*{date_str}*executor_journal*.csv"))
    if not journal_candidates:
        journal_candidates = sorted(log_dir.glob("*executor_journal*.csv"))
    journal = _parse_executor_journal(journal_candidates[-1] if journal_candidates else Path("__none__"))

    metrics = _parse_metrics_json(log_dir / "metrics.json")
    signal_events = _collect_signals_from_execution_events(log_dir, trade_date)
    uptime = _estimate_uptime(log_dir, trade_date)

    # Combine symbol lists
    symbols_all: set[str] = set(signal_events["symbols_evaluated"]) | set(journal["symbols_traded"])

    # Compute signal totals
    signals_generated = signal_events["generated"]
    signals_skipped   = skipped["total"]
    signals_total     = signals_generated + signals_skipped

    has_any_data = (
        signals_total > 0
        or journal["orders_placed"] > 0
        or bool(metrics)
        or log_dir.exists()
    )

    return {
        "strategy_id": strategy_id,
        "date": date_str,
        "log_dir_exists": log_dir.exists(),
        "has_activity": has_any_data,
        "uptime": uptime,
        "signals": {
            "generated": signals_generated,
            "skipped": signals_skipped,
            "total": signals_total,
        },
        "orders": {
            "placed": journal["orders_placed"],
            "rejected": journal["orders_rejected"],
        },
        "skip_analysis": {
            "top_reasons": skipped["by_reason"],
            "top_gating": skipped["by_gating"],
        },
        "symbols": {
            "evaluated": sorted(symbols_all),
            "traded": sorted(set(journal["symbols_traded"])),
        },
        "metrics_snapshot": metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_report(
    trade_date: date,
    strategy_filter: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    strategies = list(STRATEGY_REGISTRY.keys())
    if strategy_filter:
        strategies = [s for s in strategies if s == strategy_filter]

    reports: list[dict[str, Any]] = []
    for sid in strategies:
        rep = _report_for_strategy(sid, trade_date)
        reports.append(rep)
        if verbose:
            sigs = rep["signals"]
            ords = rep["orders"]
            activity = "active" if rep["has_activity"] else "no data"
            print(f"  {sid}: {activity}  uptime={rep['uptime']}  "
                  f"signals={sigs['generated']}gen/{sigs['skipped']}skipped  "
                  f"orders={ords['placed']}placed/{ords['rejected']}rejected")
            if rep["skip_analysis"]["top_reasons"]:
                reasons_str = ", ".join(
                    f"{r}:{c}" for r, c in list(rep["skip_analysis"]["top_reasons"].items())[:3]
                )
                print(f"    top skip reasons: {reasons_str}")

    summary = {
        "date": str(trade_date),
        "strategies": reports,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate strategy activity report")
    parser.add_argument("--date", default=None, help="Trade date YYYY-MM-DD")
    parser.add_argument("--strategy", default=None, help="Limit to one strategy_id")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.date:
        trade_date = date.fromisoformat(args.date)
    else:
        today = date.today()
        offset = max(1, today.weekday() - 4) if today.weekday() >= 5 else 1
        trade_date = today - timedelta(days=offset)

    if not args.quiet:
        print(f"=== Strategy Activity Report: {trade_date} ===\n")

    summary = run_report(trade_date, strategy_filter=args.strategy, verbose=not args.quiet)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "strategy_activity_report.json"
    out_path.write_text(json.dumps(summary, indent=2))

    if not args.quiet:
        active = sum(1 for r in summary["strategies"] if r["has_activity"])
        print(f"\n{active}/{len(summary['strategies'])} strategies have log data")
        print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
