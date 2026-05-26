#!/usr/bin/env python3
"""
scripts/claude_self_optimization_agent.py

Recommendation-only monitoring agent for the default `run.py` paper executor.

What it does:
  - builds or refreshes the intraday runtime guard report
  - analyzes missed opportunities, universe coverage, and runtime defects
  - writes a structured JSON diagnosis plus a readable markdown brief
  - can optionally run in watch mode throughout the trading day

What it does not do:
  - submit orders
  - edit env vars or strategy thresholds
  - auto-apply any optimization changes
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from zoneinfo import ZoneInfo

from runtime_guard_report import RuntimeGuardPaths, build_report


ET = ZoneInfo("America/New_York")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATE_TAG = datetime.now(ET).strftime("%Y%m%d")


def _round_or_none(value: Any, digits: int = 4) -> float | None:
    try:
        if value is None:
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _top_counter_keys(items: dict[str, Any], limit: int = 5) -> list[str]:
    return list(items.keys())[:limit] if isinstance(items, dict) else []


@dataclass
class AgentPaths:
    date_tag: str
    out_dir: Path
    log_dir: Path

    @property
    def json_path(self) -> Path:
        return self.out_dir / f"{self.date_tag}_self_optimization_agent.json"

    @property
    def md_path(self) -> Path:
        return self.out_dir / f"{self.date_tag}_self_optimization_agent.md"

    @property
    def events_path(self) -> Path:
        return self.log_dir / f"{self.date_tag}_self_optimization_agent_events.jsonl"


def _write_event(path: Path, event: str, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def _build_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    runtime = report.get("runtime_status", {})
    universe = report.get("universe_coverage", {})
    planner = report.get("planner_health", {})
    skip_summary = report.get("skip_summary", {})
    missed = report.get("likely_missed_opportunities", [])

    top_errors = runtime.get("top_error_messages", {})
    if top_errors:
        error, count = next(iter(top_errors.items()))
        findings.append(
            {
                "category": "runtime_defect",
                "severity": "high",
                "title": "Executor error is polluting live feedback",
                "evidence": {
                    "error": error,
                    "count": count,
                },
                "why_it_matters": "Repeated execution-path errors make it harder to trust missed-opportunity and order-outcome diagnostics.",
            }
        )

    if missed:
        top = missed[0]
        findings.append(
            {
                "category": "missed_opportunity",
                "severity": "high",
                "title": f"Repeated strong setup blocked for {top.get('symbol')}",
                "evidence": top,
                "why_it_matters": "The system saw a high-quality setup multiple times but never converted it into a paper position.",
            }
        )

    selected_count = int(universe.get("selected_symbol_count") or 0)
    evaluated_count = int(universe.get("evaluated_symbol_count") or 0)
    valid_plan_count = int(planner.get("valid_plan_count") or 0)
    if selected_count >= 15 and valid_plan_count <= 1:
        findings.append(
            {
                "category": "universe_quality",
                "severity": "medium",
                "title": "Universe breadth exceeds premarket plan quality",
                "evidence": {
                    "selected_symbol_count": selected_count,
                    "valid_plan_count": valid_plan_count,
                    "long_allowed_count": planner.get("long_allowed_count"),
                    "evaluated_symbol_count": evaluated_count,
                },
                "why_it_matters": "A wide universe with very few valid plans usually dilutes attention and increases low-conviction churn.",
            }
        )

    long_allowed_not_evaluated = universe.get("long_allowed_not_evaluated", []) or []
    if len(long_allowed_not_evaluated) >= 8:
        findings.append(
            {
                "category": "coverage_gap",
                "severity": "medium",
                "title": "Many long-allowed names never reached active evaluation",
                "evidence": {
                    "count": len(long_allowed_not_evaluated),
                    "sample": long_allowed_not_evaluated[:10],
                },
                "why_it_matters": "Potential opportunity loss can happen before entry gating if promising names never make it into active order planning.",
            }
        )

    top_gating = skip_summary.get("top_gating_conditions", {})
    if top_gating:
        gating, count = next(iter(top_gating.items()))
        findings.append(
            {
                "category": "gate_pressure",
                "severity": "medium",
                "title": "One buy-side gate dominates skip behavior",
                "evidence": {
                    "gating_condition": gating,
                    "count": count,
                    "top_symbols": skip_summary.get("top_symbols", {}),
                },
                "why_it_matters": "When one gating pattern dominates, the system may be underreacting to a recurring market condition or overfiltering valid setups.",
            }
        )

    return findings


def _build_improvements(report: dict[str, Any]) -> list[dict[str, Any]]:
    improvements: list[dict[str, Any]] = []
    runtime = report.get("runtime_status", {})
    universe = report.get("universe_coverage", {})
    planner = report.get("planner_health", {})
    missed = report.get("likely_missed_opportunities", [])
    order_actions = report.get("order_actions_by_symbol", {})

    if runtime.get("top_error_messages"):
        improvements.append(
            {
                "priority": 1,
                "theme": "stability",
                "proposal": "Restart the default paper executor after verifying only one run.py process is active, so the in-memory process picks up the fixed live-experience logging code.",
                "expected_benefit": "Cleaner journal/error data and better confidence in downstream monitoring.",
                "guardrail": "Do not start a second executor; replace only the stale process.",
            }
        )

    if missed:
        top = missed[0]
        improvements.append(
            {
                "priority": 2,
                "theme": "entry_gate_calibration",
                "proposal": f"Replay the blocked {top.get('symbol')} setups and compare the ML next-bar buy gate against realized short-horizon path after stream confirmation.",
                "expected_benefit": "Find out whether the ML gate is lagging strong continuation entries rather than filtering genuine losers.",
                "guardrail": "Calibrate from replay evidence first; do not loosen the gate globally from one symbol.",
            }
        )

    selected_count = int(universe.get("selected_symbol_count") or 0)
    valid_plan_count = int(planner.get("valid_plan_count") or 0)
    if selected_count and valid_plan_count <= 1:
        improvements.append(
            {
                "priority": 3,
                "theme": "universe_selection",
                "proposal": "Tighten premarket ranking so the selected universe is closer to the names that actually become valid trade structures.",
                "expected_benefit": "Higher signal density and less attention spent on names that never progress to actionable plans.",
                "guardrail": "Improve precision before increasing symbol count.",
            }
        )

    if universe.get("long_allowed_not_evaluated"):
        improvements.append(
            {
                "priority": 4,
                "theme": "coverage",
                "proposal": "Audit why long-allowed symbols are not entering active evaluation or order-plan generation during the session.",
                "expected_benefit": "Reduce pre-entry opportunity loss caused by pipeline coverage gaps.",
                "guardrail": "Preserve stale-plan and market-context safety checks.",
            }
        )

    duplicate_pressure = 0
    for counts in order_actions.values():
        duplicate_pressure += int(counts.get("buy:SKIP", 0)) if isinstance(counts, dict) else 0
    if duplicate_pressure:
        improvements.append(
            {
                "priority": 5,
                "theme": "order_lifecycle",
                "proposal": "Review open-order refresh cadence and duplicate-order suppression timing around partially filled or recently exited names.",
                "expected_benefit": "Reduce dead time where valid follow-on setups are blocked by stale order state.",
                "guardrail": "Keep duplicate order prevention intact; improve state freshness instead.",
            }
        )

    return improvements


def _build_scorecard(report: dict[str, Any], findings: list[dict[str, Any]]) -> dict[str, Any]:
    runtime = report.get("runtime_status", {})
    planner = report.get("planner_health", {})
    universe = report.get("universe_coverage", {})
    missed = report.get("likely_missed_opportunities", [])

    defect_penalty = min(sum(v for v in runtime.get("top_error_messages", {}).values()), 50)
    missed_penalty = min(len(missed) * 10, 30)
    breadth_penalty = 0
    selected_count = int(universe.get("selected_symbol_count") or 0)
    valid_plan_count = int(planner.get("valid_plan_count") or 0)
    if selected_count >= 15 and valid_plan_count <= 1:
        breadth_penalty = 15

    quality_score = max(0, 100 - defect_penalty - missed_penalty - breadth_penalty)

    return {
        "runtime_quality_score": quality_score,
        "open_high_severity_findings": sum(1 for item in findings if item.get("severity") == "high"),
        "open_medium_severity_findings": sum(1 for item in findings if item.get("severity") == "medium"),
        "selected_symbol_count": selected_count,
        "valid_plan_count": valid_plan_count,
        "likely_missed_count": len(missed),
        "top_skip_conditions": _top_counter_keys(report.get("skip_summary", {}).get("top_gating_conditions", {})),
    }


def _render_markdown(diagnosis: dict[str, Any]) -> str:
    report = diagnosis["runtime_report"]
    scorecard = diagnosis["scorecard"]
    lines: list[str] = []

    lines.append(f"# Self-Optimization Agent Report - {diagnosis['date_tag']}")
    lines.append("")
    lines.append("## Snapshot")
    lines.append(f"- Generated UTC: {diagnosis['generated_at_utc']}")
    lines.append(f"- Paper-only confirmed: {report['paper_runtime'].get('paper_only_confirmed')}")
    lines.append(f"- Runtime quality score: {scorecard.get('runtime_quality_score')}/100")
    lines.append(f"- Current positions: {len(report['runtime_status'].get('current_positions', []))}")
    lines.append(f"- Likely missed opportunities: {scorecard.get('likely_missed_count')}")
    lines.append("")

    lines.append("## Findings")
    for item in diagnosis["findings"]:
        lines.append(f"- [{item['severity']}] {item['title']}")
        lines.append(f"  Reason: {item['why_it_matters']}")
        lines.append(f"  Evidence: `{json.dumps(item['evidence'], default=str)}`")
    if not diagnosis["findings"]:
        lines.append("- No major findings.")
    lines.append("")

    lines.append("## Improvements")
    for item in diagnosis["improvements"]:
        lines.append(f"- [P{item['priority']}] {item['proposal']}")
        lines.append(f"  Benefit: {item['expected_benefit']}")
        lines.append(f"  Guardrail: {item['guardrail']}")
    if not diagnosis["improvements"]:
        lines.append("- No improvement proposals generated.")
    lines.append("")

    lines.append("## Universe And Opportunity Notes")
    universe = report.get("universe_coverage", {})
    lines.append(
        f"- Selected={universe.get('selected_symbol_count')} long_allowed={universe.get('long_allowed_count')} evaluated={universe.get('evaluated_symbol_count')} traded={universe.get('traded_symbol_count')}"
    )
    if universe.get("long_allowed_not_evaluated"):
        lines.append(f"- Long-allowed but unevaluated sample: {', '.join(universe['long_allowed_not_evaluated'][:8])}")
    missed = report.get("likely_missed_opportunities", [])
    if missed:
        top = missed[0]
        lines.append(
            f"- Top missed candidate: {top.get('symbol')} direction={top.get('direction_score')} early={top.get('early_continuation_score')} rel_volume={top.get('rel_volume')}"
        )
    lines.append("")

    return "\n".join(lines) + "\n"


def run_agent(date_tag: str) -> dict[str, Any]:
    report = build_report(date_tag)
    findings = _build_findings(report)
    improvements = _build_improvements(report)
    scorecard = _build_scorecard(report, findings)

    diagnosis = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_tag": date_tag,
        "agent_name": "claude_self_optimization_agent",
        "scorecard": scorecard,
        "findings": findings,
        "improvements": improvements,
        "runtime_report": report,
    }
    return diagnosis


def _persist(diagnosis: dict[str, Any], paths: AgentPaths) -> None:
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)

    with paths.json_path.open("w", encoding="utf-8") as fh:
        json.dump(diagnosis, fh, indent=2, default=str)
        fh.write("\n")
    paths.md_path.write_text(_render_markdown(diagnosis), encoding="utf-8")

    _write_event(
        paths.events_path,
        "self_optimization_snapshot",
        {
            "json_path": str(paths.json_path),
            "md_path": str(paths.md_path),
            "runtime_quality_score": diagnosis["scorecard"].get("runtime_quality_score"),
            "high_findings": diagnosis["scorecard"].get("open_high_severity_findings"),
            "likely_missed_count": diagnosis["scorecard"].get("likely_missed_count"),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Recommendation-only monitoring agent for run.py paper runtime")
    parser.add_argument("--date", default=DEFAULT_DATE_TAG, help="Trading date tag in YYYYMMDD format")
    parser.add_argument("--watch", action="store_true", help="Refresh diagnostics continuously")
    parser.add_argument("--interval", type=int, default=120, help="Watch interval in seconds")
    parser.add_argument("--stdout", action="store_true", help="Print JSON diagnosis to stdout")
    args = parser.parse_args()

    runtime_paths = RuntimeGuardPaths(
        date_tag=args.date,
        log_dir=PROJECT_ROOT / "logs" / "execution" / args.date,
        out_dir=PROJECT_ROOT / "system-data" / args.date,
        daily_out_dir=PROJECT_ROOT / "daily_system_outputs" / args.date,
    )
    paths = AgentPaths(
        date_tag=args.date,
        out_dir=runtime_paths.daily_out_dir,
        log_dir=runtime_paths.log_dir,
    )

    while True:
        diagnosis = run_agent(args.date)
        _persist(diagnosis, paths)
        print(f"[self-opt-agent] wrote {paths.json_path}")
        print(f"[self-opt-agent] wrote {paths.md_path}")
        if args.stdout:
            print(json.dumps(diagnosis, indent=2, default=str))
        if not args.watch:
            break
        time.sleep(max(15, args.interval))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
