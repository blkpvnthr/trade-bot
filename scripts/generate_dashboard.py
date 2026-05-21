#!/usr/bin/env python3
"""
scripts/generate_dashboard.py

Aggregate rankings, activity, health, and allocation into a single
daily_system_outputs/dashboard.json for downstream frontends or monitoring.

Usage:
  python scripts/generate_dashboard.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.registry import STRATEGY_REGISTRY

OUTPUT_DIR = PROJECT_ROOT / "daily_system_outputs"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _summarize_rankings(rankings: list | None) -> dict[str, Any]:
    if not rankings:
        return {"available": False, "top_strategy": None, "count": 0}
    top = rankings[0] if rankings else {}
    return {
        "available": True,
        "count": len(rankings),
        "top_strategy": top.get("strategy_id"),
        "top_sharpe": top.get("metrics", {}).get("sharpe_ratio"),
    }


def _summarize_activity(activity: dict | None) -> dict[str, Any]:
    if not activity:
        return {"available": False}
    strategies = activity.get("strategies", [])
    active = [s for s in strategies if s.get("has_activity")]
    total_signals = sum(
        s.get("signals", {}).get("total", 0) for s in strategies
    )
    total_orders = sum(
        s.get("orders", {}).get("placed", 0) for s in strategies
    )
    return {
        "available": True,
        "date": activity.get("date"),
        "active_strategies": len(active),
        "total_strategies": len(strategies),
        "total_signals": total_signals,
        "total_orders": total_orders,
    }


def _summarize_health(health: dict | None) -> dict[str, Any]:
    if not health:
        return {"available": False}
    s = health.get("summary", {})
    # Collect any errors across all strategies
    errors: list[str] = []
    for strat in health.get("strategies", []):
        for check_name in strat.get("errors", []):
            errors.append(f"{strat['strategy_id']}.{check_name}")
    return {
        "available": True,
        "ok": s.get("ok", 0),
        "warn": s.get("warn", 0),
        "error": s.get("error", 0),
        "critical_errors": errors,
    }


def _summarize_allocation(alloc: dict | None) -> dict[str, Any]:
    if not alloc:
        return {"available": False}
    return {
        "available": True,
        "total_capital": alloc.get("total_capital"),
        "data_source": alloc.get("data_source"),
        "paper_only": alloc.get("paper_only", True),
        "allocations": [
            {
                "strategy_id": a["strategy_id"],
                "allocation_pct": a["allocation_pct"],
                "allocation_dollars": a["allocation_dollars"],
            }
            for a in alloc.get("allocations", [])
        ],
    }


def _summarize_replay(replay: dict | None) -> dict[str, Any]:
    if not replay:
        return {"available": False}
    strategies = replay.get("strategies", [])
    total_signals = sum(
        s.get("total_buy", 0) + s.get("total_sell", 0) +
        s.get("total_short", 0) + s.get("total_cover", 0)
        for s in strategies
    )
    exceptions = sum(len(s.get("exceptions", [])) for s in strategies)
    return {
        "available": True,
        "date": replay.get("date"),
        "data_source": replay.get("data_source"),
        "symbols": len(replay.get("symbols", [])),
        "total_signals": total_signals,
        "strategy_exceptions": exceptions,
    }


def _summarize_overlap(overlap: dict | None) -> dict[str, Any]:
    if not overlap:
        return {"available": False}
    risks = overlap.get("duplicate_exposure_risks", [])
    conflicts = sum(p.get("conflicting_bars", 0) for p in overlap.get("pairwise_correlation", []))
    return {
        "available": True,
        "duplicate_exposure_risks": len(risks),
        "conflicting_bars_total": conflicts,
        "risk_pairs": [r["pair"] for r in risks],
    }


def generate_dashboard(verbose: bool = True) -> dict[str, Any]:
    rankings  = _load_json(OUTPUT_DIR / "research_rankings.json")
    activity  = _load_json(OUTPUT_DIR / "strategy_activity_report.json")
    health    = _load_json(OUTPUT_DIR / "health_report.json")
    alloc     = _load_json(OUTPUT_DIR / "capital_allocation.json")
    replay    = _load_json(OUTPUT_DIR / "replay_results.json")
    overlap   = _load_json(OUTPUT_DIR / "strategy_overlap.json")

    dashboard = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategies_registered": list(STRATEGY_REGISTRY.keys()),
        "rankings":   _summarize_rankings(rankings),
        "activity":   _summarize_activity(activity),
        "health":     _summarize_health(health),
        "allocation": _summarize_allocation(alloc),
        "replay":     _summarize_replay(replay),
        "overlap":    _summarize_overlap(overlap),
        "source_files": {
            "rankings":   str(OUTPUT_DIR / "research_rankings.json"),
            "activity":   str(OUTPUT_DIR / "strategy_activity_report.json"),
            "health":     str(OUTPUT_DIR / "health_report.json"),
            "allocation": str(OUTPUT_DIR / "capital_allocation.json"),
            "replay":     str(OUTPUT_DIR / "replay_results.json"),
            "overlap":    str(OUTPUT_DIR / "strategy_overlap.json"),
        },
    }

    if verbose:
        h = dashboard["health"]
        a = dashboard["activity"]
        r = dashboard["rankings"]
        print(f"  Health:     {h.get('ok',0)} ok / {h.get('warn',0)} warn / {h.get('error',0)} error")
        print(f"  Activity:   {a.get('active_strategies',0)}/{a.get('total_strategies',0)} active  "
              f"signals={a.get('total_signals',0)}  orders={a.get('total_orders',0)}")
        print(f"  Rankings:   {'available' if r['available'] else 'not yet'}  "
              f"top={r.get('top_strategy','n/a')}")
        alloc_sum = dashboard["allocation"]
        if alloc_sum.get("available"):
            print(f"  Allocation: ${alloc_sum['total_capital']:,.0f} paper  "
                  f"source={alloc_sum['data_source']}")

    return dashboard


def main() -> None:
    print("=== Dashboard Generator ===\n")
    dashboard = generate_dashboard(verbose=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "dashboard.json"
    out_path.write_text(json.dumps(dashboard, indent=2))
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
