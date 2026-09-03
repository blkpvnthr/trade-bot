#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(os.getenv("TRADE_BOT_ROOT", Path.cwd()))
load_dotenv(ROOT / ".env")

MODEL = os.getenv("DAILY_LLM_ANALYST_MODEL", "gpt-4.1-mini")
MAX_LOG_LINES = int(os.getenv("DAILY_LLM_MAX_LOG_LINES", "1200"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def today_tag() -> str:
    return datetime.now().strftime("%Y%m%d")


def read_text_tail(path: Path, max_lines: int = MAX_LOG_LINES) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(errors="ignore").splitlines()
        return "\n".join(lines[-max_lines:])
    except Exception as e:
        return f"[read_error] {path}: {e}"


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(errors="ignore"))
    except Exception as e:
        return {"_error": str(e), "_path": str(path)}


def read_jsonl_tail(path: Path, max_lines: int = 500) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="ignore").splitlines()[-max_lines:]:
        try:
            out.append(json.loads(line))
        except Exception:
            out.append({"raw": line})
    return out


def summarize_csv(path: Path, max_rows: int = 25) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}

    try:
        df = pd.read_csv(path)
        summary = {
            "exists": True,
            "path": str(path),
            "rows": len(df),
            "columns": list(df.columns),
            "head": df.head(max_rows).to_dict(orient="records"),
        }

        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            summary["numeric_summary"] = numeric.describe().round(4).to_dict()

        if "symbol" in df.columns:
            summary["symbols"] = sorted(df["symbol"].dropna().astype(str).unique().tolist())[:100]

        if "side" in df.columns:
            summary["side_counts"] = df["side"].astype(str).value_counts().to_dict()

        if "status" in df.columns:
            summary["status_counts"] = df["status"].astype(str).value_counts().to_dict()

        return summary
    except Exception as e:
        return {"exists": True, "path": str(path), "error": str(e)}


def collect_day(tag: str) -> dict[str, Any]:
    system_dir = ROOT / "system-data" / tag
    log_dir = ROOT / "logs" / "execution" / tag
    router_dir = ROOT / "logs" / "execution" / "webhook_order_router"

    csv_files = [
        system_dir / f"{tag}_daily_walkforward_emulated_rebalance_orders.csv",
        system_dir / f"{tag}_daily_walkforward_emulated_exit_orders.csv",
        system_dir / f"{tag}_stream_confirmed_rebalance_orders.csv",
        system_dir / f"{tag}_stream_confirmed_trade_plan.csv",
        log_dir / f"{tag}_executor_journal.csv",
        log_dir / f"{tag}_executor_equity_curve.csv",
    ]

    json_files = [
        system_dir / f"{tag}_market_context.json",
        system_dir / f"{tag}_premarket_trade_plan.json",
        system_dir / f"{tag}_intraday_structure_map.json",
        system_dir / f"{tag}_sr_rl_live_signals.json",
        system_dir / f"{tag}_spx_price_action_signal.json",
        system_dir / f"{tag}_forecast_ensemble_signals.json",
        system_dir / f"{tag}_data_validation_state.json",
    ]

    logs = {
        "run_log_tail": read_text_tail(ROOT / "logs" / "run.log"),
        "webhook_router_log_tail": read_text_tail(ROOT / "logs" / "webhook_router.log"),
        "execution_events": read_jsonl_tail(log_dir / f"{tag}_execution_events.jsonl"),
        "root_execution_events": read_jsonl_tail(ROOT / "logs" / "execution" / "execution_events.jsonl"),
        "webhook_router_events": read_jsonl_tail(router_dir / f"{tag}_execution_events.jsonl"),
    }

    return {
        "date_tag": tag,
        "root": str(ROOT),
        "system_dir_exists": system_dir.exists(),
        "log_dir_exists": log_dir.exists(),
        "csv_summaries": {p.name: summarize_csv(p) for p in csv_files},
        "json_payloads": {p.name: read_json(p) for p in json_files},
        "logs": logs,
    }


SCHEMA = {
    "name": "daily_trading_report",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "date_tag": {"type": "string"},
            "executive_summary": {"type": "string"},
            "system_health": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "status": {"type": "string"},
                    "critical_issues": {"type": "array", "items": {"type": "string"}},
                    "warnings": {"type": "array", "items": {"type": "string"}},
                    "working_components": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["status", "critical_issues", "warnings", "working_components"],
            },
            "trading_activity": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "orders_detected": {"type": "integer"},
                    "submitted_orders": {"type": "integer"},
                    "filled_orders": {"type": "integer"},
                    "symbols_traded_or_signaled": {"type": "array", "items": {"type": "string"}},
                    "notable_orders": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["orders_detected", "submitted_orders", "filled_orders", "symbols_traded_or_signaled", "notable_orders"],
            },
            "strategy_insights": {"type": "array", "items": {"type": "string"}},
            "risk_insights": {"type": "array", "items": {"type": "string"}},
            "data_quality_issues": {"type": "array", "items": {"type": "string"}},
            "recommended_actions": {"type": "array", "items": {"type": "string"}},
            "markdown_report": {"type": "string"},
        },
        "required": [
            "date_tag",
            "executive_summary",
            "system_health",
            "trading_activity",
            "strategy_insights",
            "risk_insights",
            "data_quality_issues",
            "recommended_actions",
            "markdown_report",
        ],
    },
    "strict": True,
}


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = f"""
You are QuantumQuant's daily trading-system analyst.

Analyze this day's logs, order files, model outputs, data quality state, router events,
market context, premarket plan, RL signals, forecast signals, and execution journal.

Focus on:
- what actually happened
- whether TradingView/webhook/order-router/executor/Alpaca pipeline worked
- submitted/fill status when visible
- data failures, stale files, missing prices, auth issues, rate limits
- whether strategies generated useful signals
- risk-control behavior
- concrete fixes for tomorrow
- header-only CSV files are valid empty artifacts, not corrupt files
- stale LLM/sentiment inputs should be reported as confidence degradation, not as hard trade blockers

Do not invent fills or PnL. If unavailable, say unavailable.

DATA:
{json.dumps(payload, default=str)[:180000]}
"""

    resp = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": "You produce precise operational trading-system reports. Be blunt, factual, and action-oriented.",
            },
            {"role": "user", "content": prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": SCHEMA["name"],
                "schema": SCHEMA["schema"],
                "strict": True,
            }
        },
    )

    return json.loads(resp.output_text)


def main() -> None:
    tag = os.getenv("RUN_DATA_DATE") or (os.sys.argv[1] if len(os.sys.argv) > 1 else today_tag())

    payload = collect_day(tag)
    report = analyze(payload)

    out_dir = ROOT / "daily_system_outputs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{tag}_llm_daily_report.json"
    md_path = out_dir / f"{tag}_llm_daily_report.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(report["markdown_report"], encoding="utf-8")

    print(f"[saved] {json_path}")
    print(f"[saved] {md_path}")
    print(report["executive_summary"])


if __name__ == "__main__":
    main()
