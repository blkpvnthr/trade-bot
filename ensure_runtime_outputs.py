#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ET = ZoneInfo("America/New_York")
ROOT = Path(os.getenv("TRADE_BOT_ROOT", Path(__file__).resolve().parent))

ORDER_COLUMNS = [
    "symbol", "side", "qty", "order_qty", "delta_shares", "delta_dollars",
    "price", "entry_price", "stop_price", "target_price", "stop_loss",
    "take_profit", "risk_per_share", "reward_per_share", "risk_reward_ratio",
    "target_weight", "current_weight", "delta_weight", "abs_delta_dollars",
    "capped_dollars", "confidence_used", "confidence_multiplier", "reason",
    "source", "client_order_id", "planned_at",
]

EXIT_ORDER_COLUMNS = [
    "symbol", "side", "qty", "order_qty", "delta_shares", "delta_dollars",
    "price", "reason", "target_weight", "current_weight", "delta_weight",
    "confidence_used", "exit_reason", "planned_at",
]

STREAM_TRADE_PLAN_COLUMNS = [
    "symbol", "side", "setup_type", "entry_price", "stop_price",
    "target_price", "stop_loss", "take_profit", "risk_per_share",
    "risk_pct", "take_profit_r_multiple", "qty", "notional",
    "trailing_stop_enabled", "trail_percent", "trail_price",
    "time_in_force", "detected_at_utc", "source", "reason",
    "direction_score",
]

JOURNAL_COLUMNS = [
    "timestamp_utc", "cycle_id", "event", "market_open",
    "equity_before", "equity_after", "cash_before", "cash_after",
    "buying_power_before", "buying_power_after",
    "gross_exposure_before", "gross_exposure_after",
    "trade_symbol", "trade_side", "trade_qty", "trade_price",
    "estimated_notional", "target_weight", "current_weight", "delta_weight",
    "confidence", "cash_buffer", "max_gross_exposure", "max_position_weight",
    "max_order_notional", "stop_price", "target_price", "bracket_attached",
    "client_order_id", "order_id", "order_status", "filled_qty",
    "filled_avg_price", "slippage_pct", "skip_reason", "broker_message",
    "message",
]

EQUITY_CURVE_COLUMNS = [
    "timestamp_utc", "label", "cycle_id", "equity", "cash",
    "buying_power", "portfolio_value",
]


def today_tag() -> str:
    return datetime.now(ET).strftime("%Y%m%d")


def _csv_header(path: Path) -> list[str]:
    if not path.exists() or path.stat().st_size <= 1:
        return []
    try:
        with path.open("r", newline="", encoding="utf-8", errors="ignore") as fh:
            header = next(csv.reader(fh), [])
        return [str(col).strip() for col in header if str(col).strip()]
    except Exception:
        return []


def _csv_has_columns(path: Path) -> bool:
    return bool(_csv_header(path))


def ensure_csv(path: Path, columns: list[str]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _csv_header(path)
    if existing and all(col in existing for col in columns):
        return False
    if existing:
        try:
            import pandas as pd

            df = pd.read_csv(path)
            for col in columns:
                if col not in df.columns:
                    df[col] = ""
            ordered = columns + [col for col in df.columns if col not in columns]
            df[ordered].to_csv(path, index=False)
            return True
        except Exception:
            pass
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=columns).writeheader()
    return True


def touch_jsonl(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return False
    path.write_text("", encoding="utf-8")
    return True


def ensure_runtime_outputs(date_tag: str | None = None, root: Path | str | None = None) -> dict[str, list[str]]:
    tag = str(date_tag or today_tag())
    base = Path(root) if root is not None else ROOT
    system_dir = base / "system-data" / tag
    dated_log_dir = base / "logs" / "execution" / tag
    root_log_dir = base / "logs" / "execution"
    system_dir.mkdir(parents=True, exist_ok=True)
    dated_log_dir.mkdir(parents=True, exist_ok=True)
    root_log_dir.mkdir(parents=True, exist_ok=True)

    csv_specs = {
        system_dir / f"{tag}_daily_walkforward_emulated_rebalance_orders.csv": ORDER_COLUMNS,
        system_dir / f"{tag}_daily_walkforward_emulated_exit_orders.csv": EXIT_ORDER_COLUMNS,
        system_dir / f"{tag}_stream_confirmed_rebalance_orders.csv": ORDER_COLUMNS,
        system_dir / f"{tag}_stream_confirmed_trade_plan.csv": STREAM_TRADE_PLAN_COLUMNS,
        system_dir / f"{tag}_pivot_reversal_rebalance_orders.csv": ORDER_COLUMNS,
        system_dir / f"{tag}_pivot_reversal_exit_orders.csv": EXIT_ORDER_COLUMNS,
        dated_log_dir / f"{tag}_executor_journal.csv": JOURNAL_COLUMNS,
        root_log_dir / "executor_journal.csv": JOURNAL_COLUMNS,
        dated_log_dir / f"{tag}_executor_equity_curve.csv": EQUITY_CURVE_COLUMNS,
        root_log_dir / "executor_equity_curve.csv": EQUITY_CURVE_COLUMNS,
    }

    jsonl_paths = [
        dated_log_dir / f"{tag}_execution_events.jsonl",
        root_log_dir / "execution_events.jsonl",
        dated_log_dir / f"{tag}_live_experience_replay.jsonl",
    ]

    result = {"csv_initialized": [], "jsonl_touched": []}
    for path, columns in csv_specs.items():
        if ensure_csv(path, columns):
            result["csv_initialized"].append(str(path))
    for path in jsonl_paths:
        if touch_jsonl(path):
            result["jsonl_touched"].append(str(path))
    return result


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else os.getenv("RUN_DATA_DATE") or today_tag()
    result = ensure_runtime_outputs(tag)
    print(json.dumps({"date_tag": tag, **result}, indent=2))


if __name__ == "__main__":
    main()
