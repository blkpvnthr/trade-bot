#!/usr/bin/env python3
"""
scripts/runtime_guard_report.py

Intraday paper-runtime inspection for the default run.py executor.

Purpose:
  - confirm paper-only runtime posture
  - summarize current executor activity and latest outcomes
  - surface repeated buy-side skips that look like likely missed opportunities
  - compare universe selection inputs against actual evaluated/traded symbols
  - write a compact JSON report for automation review throughout the day

This script is diagnostic only. It does not place orders or edit configs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import dotenv_values
from zoneinfo import ZoneInfo


ET = ZoneInfo("America/New_York")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_DATE_TAG = datetime.now(ET).strftime("%Y%m%d")


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _parse_ts(value: Any) -> datetime | None:
    text = _iso_or_none(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open() as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _latest(paths: list[Path]) -> Path | None:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def _extract_ml_prob(gating_condition: str) -> float | None:
    match = re.search(r"ml_next_bar_prob_low:([0-9.]+)", gating_condition or "")
    if not match:
        return None
    return _safe_float(match.group(1))


@dataclass
class RuntimeGuardPaths:
    date_tag: str
    log_dir: Path
    out_dir: Path
    daily_out_dir: Path

    @property
    def journal_path(self) -> Path:
        return self.log_dir / f"{self.date_tag}_executor_journal.csv"

    @property
    def skipped_path(self) -> Path:
        return self.log_dir / "skipped_signals.jsonl"

    @property
    def latest_positions_path(self) -> Path | None:
        return _latest(sorted(self.log_dir.glob(f"positions_after_{self.date_tag}_*.csv")))

    @property
    def latest_order_plan_path(self) -> Path | None:
        return _latest(sorted(self.log_dir.glob(f"order_plan_{self.date_tag}_*.csv")))

    @property
    def latest_universe_path(self) -> Path:
        return self.out_dir / f"{self.date_tag}_latest_llm_sentiment_universe.json"

    @property
    def premarket_plan_path(self) -> Path:
        return self.out_dir / f"{self.date_tag}_premarket_trade_plan.json"

    @property
    def structure_map_path(self) -> Path:
        return self.out_dir / f"{self.date_tag}_intraday_structure_map.json"

    @property
    def market_context_path(self) -> Path:
        preferred = self.daily_out_dir / f"{self.date_tag}_market_context.json"
        if preferred.exists():
            return preferred
        return self.out_dir / f"{self.date_tag}_market_context.json"

    @property
    def report_path(self) -> Path:
        return self.daily_out_dir / f"{self.date_tag}_runtime_guard_report.json"


def _load_executor_journal(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        return [], {}

    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))

    parsed: list[dict[str, Any]] = []
    action_counter: Counter[str] = Counter()
    error_messages: Counter[str] = Counter()

    for row in rows:
        if len(row) < 6:
            continue
        if row[0] == "timestamp_utc" or row[2] == "action":
            continue
        action = row[2]
        rec = {
            "timestamp_utc": row[0],
            "cycle_id": row[1],
            "action": action,
            "paper": row[3],
            "equity_before": _safe_float(row[4]),
            "equity_after": _safe_float(row[5]),
            "symbol": row[12] if len(row) > 12 else "",
            "side": row[13] if len(row) > 13 else "",
            "qty": _safe_float(row[14]) if len(row) > 14 else None,
            "price": _safe_float(row[15]) if len(row) > 15 else None,
            "message": row[-1] if row else "",
        }
        parsed.append(rec)
        action_counter[action] += 1
        if action == "ERROR" and rec["message"]:
            error_messages[rec["message"]] += 1

    summary = {
        "total_rows": len(parsed),
        "actions": dict(action_counter),
        "top_error_messages": dict(error_messages.most_common(8)),
        "last_event": parsed[-1] if parsed else None,
    }
    return parsed, summary


def _load_order_plan_rows(log_dir: Path, date_tag: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best_by_symbol: dict[str, dict[str, Any]] = {}

    for path in sorted(log_dir.glob(f"order_plan_{date_tag}_*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        planned_at = None
        if "planned_at" in df.columns:
            planned_at = df["planned_at"].dropna().astype(str).iloc[-1] if not df["planned_at"].dropna().empty else None
        for _, rec in df.iterrows():
            symbol = str(rec.get("symbol", "")).strip().upper()
            side = str(rec.get("side", "")).strip().lower()
            if not symbol or side != "buy":
                continue
            row = {
                "path": str(path),
                "cycle_id": path.stem.replace("order_plan_", ""),
                "symbol": symbol,
                "planned_at": _iso_or_none(rec.get("planned_at") or planned_at),
                "reason": str(rec.get("reason", "") or ""),
                "source": str(rec.get("source", "") or ""),
                "stream_confirmed": bool(rec.get("stream_confirmed")) if pd.notna(rec.get("stream_confirmed")) else False,
                "setup_type": str(rec.get("setup_type", "") or ""),
                "early_continuation_score": _safe_float(rec.get("early_continuation_score")),
                "direction_score": _safe_float(rec.get("direction_score")),
                "direction_score_1m": _safe_float(rec.get("direction_score_1m")),
                "direction_score_5m": _safe_float(rec.get("direction_score_5m")),
                "direction_score_15m": _safe_float(rec.get("direction_score_15m")),
                "rel_volume": _safe_float(rec.get("rel_volume")),
                "price": _safe_float(rec.get("price")),
                "confidence_used": _safe_float(rec.get("confidence_used")),
                "capped_dollars": _safe_float(rec.get("capped_dollars")),
                "order_qty": _safe_int(rec.get("order_qty")),
            }
            rows.append(row)

            quality = max(
                row["direction_score"] or 0.0,
                row["early_continuation_score"] or 0.0,
            )
            prior = best_by_symbol.get(symbol)
            prior_quality = max(
                (prior or {}).get("direction_score") or 0.0,
                (prior or {}).get("early_continuation_score") or 0.0,
            )
            if prior is None or quality > prior_quality:
                best_by_symbol[symbol] = row

    return rows, best_by_symbol


def _load_skips(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = _read_jsonl(path)
    reason_counter: Counter[str] = Counter()
    gating_counter: Counter[str] = Counter()
    by_symbol: Counter[str] = Counter()

    parsed: list[dict[str, Any]] = []
    for rec in records:
        symbol = str(rec.get("symbol", "")).strip().upper()
        side = str(rec.get("side", "")).strip().lower()
        if side != "buy":
            continue
        gating = str(rec.get("gating_condition", "") or "")
        parsed_rec = {
            "timestamp_utc": rec.get("timestamp_utc"),
            "cycle_id": rec.get("cycle_id"),
            "symbol": symbol,
            "gating_condition": gating,
            "reason": str(rec.get("reason", "") or ""),
            "entry_reason": str(rec.get("entry_reason", "") or ""),
            "ml_prob": _extract_ml_prob(gating),
        }
        parsed.append(parsed_rec)
        if parsed_rec["reason"]:
            reason_counter[parsed_rec["reason"]] += 1
        if gating:
            gating_counter[gating] += 1
        if symbol:
            by_symbol[symbol] += 1

    return parsed, {
        "buy_skip_count": len(parsed),
        "top_reasons": dict(reason_counter.most_common(8)),
        "top_gating_conditions": dict(gating_counter.most_common(8)),
        "top_symbols": dict(by_symbol.most_common(8)),
    }


def _current_positions(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if df.empty:
        return []
    out: list[dict[str, Any]] = []
    for _, rec in df.iterrows():
        out.append(
            {
                "symbol": str(rec.get("symbol", "")).strip().upper(),
                "qty": _safe_float(rec.get("qty")),
                "current_price": _safe_float(rec.get("current_price")),
                "market_value": _safe_float(rec.get("market_value")),
                "unrealized_pl": _safe_float(rec.get("unrealized_pl")),
                "unrealized_plpc": _safe_float(rec.get("unrealized_plpc")),
            }
        )
    return out


def _load_universe_symbols(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    if not data:
        return {"symbols": [], "selected_symbols": []}

    symbols = [str(s).strip().upper() for s in data.get("symbols", []) if str(s).strip()]
    selections = data.get("selections", [])
    selected_symbols: list[str] = []
    if isinstance(selections, list):
        for item in selections:
            if isinstance(item, dict):
                symbol = str(item.get("symbol", "")).strip().upper()
                if symbol:
                    selected_symbols.append(symbol)
            elif isinstance(item, str):
                selected_symbols.append(item.strip().upper())

    return {
        "symbols": symbols,
        "selected_symbols": selected_symbols,
        "market_tone": data.get("market_tone"),
        "source": data.get("source"),
        "target_count": data.get("target_count"),
    }


def _order_actions_by_symbol(journal_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in journal_rows:
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        side = str(row.get("side", "")).strip().lower()
        action = str(row.get("action", "")).strip().upper()
        key = f"{side}:{action}"
        out[symbol][key] += 1
    return {sym: dict(counts) for sym, counts in out.items()}


def _likely_missed_opportunities(
    skips: list[dict[str, Any]],
    best_plan_by_symbol: dict[str, dict[str, Any]],
    journal_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    buy_orders_after_symbol: Counter[str] = Counter()
    for row in journal_rows:
        if str(row.get("action", "")).upper() != "ORDER":
            continue
        if str(row.get("side", "")).lower() != "buy":
            continue
        symbol = str(row.get("symbol", "")).strip().upper()
        if symbol:
            buy_orders_after_symbol[symbol] += 1

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for skip in skips:
        symbol = skip["symbol"]
        if symbol:
            grouped[symbol].append(skip)

    candidates: list[dict[str, Any]] = []
    for symbol, items in grouped.items():
        plan = best_plan_by_symbol.get(symbol, {})
        direction_score = plan.get("direction_score") or 0.0
        early_score = plan.get("early_continuation_score") or 0.0
        rel_volume = plan.get("rel_volume")
        stream_confirmed = bool(plan.get("stream_confirmed"))
        later_buy_orders = buy_orders_after_symbol.get(symbol, 0)
        skip_count = len(items)
        top_gating = Counter(item["gating_condition"] for item in items if item["gating_condition"]).most_common(2)
        avg_ml_prob = [item["ml_prob"] for item in items if item["ml_prob"] is not None]
        avg_ml = sum(avg_ml_prob) / len(avg_ml_prob) if avg_ml_prob else None

        likely_missed = (
            later_buy_orders == 0
            and skip_count >= 2
            and (
                stream_confirmed
                or direction_score >= 0.78
                or early_score >= 0.78
            )
        )
        if not likely_missed:
            continue

        candidates.append(
            {
                "symbol": symbol,
                "skip_count": skip_count,
                "later_buy_orders": later_buy_orders,
                "stream_confirmed": stream_confirmed,
                "direction_score": round(direction_score, 4) if direction_score else None,
                "early_continuation_score": round(early_score, 4) if early_score else None,
                "rel_volume": round(rel_volume, 4) if rel_volume is not None else None,
                "avg_blocked_ml_prob": round(avg_ml, 4) if avg_ml is not None else None,
                "top_gating_conditions": [{"condition": cond, "count": cnt} for cond, cnt in top_gating],
                "latest_plan_reason": plan.get("reason"),
                "latest_plan_source": plan.get("source"),
            }
        )

    candidates.sort(
        key=lambda item: (
            item["skip_count"],
            item["stream_confirmed"],
            item["direction_score"] or 0.0,
            item["early_continuation_score"] or 0.0,
        ),
        reverse=True,
    )
    return candidates[:10]


def _recommendations(
    journal_summary: dict[str, Any],
    skip_summary: dict[str, Any],
    missed: list[dict[str, Any]],
    premarket_plan: dict[str, Any],
    market_context: dict[str, Any],
) -> list[str]:
    recs: list[str] = []

    top_errors = journal_summary.get("top_error_messages", {})
    if top_errors:
        top_error = next(iter(top_errors))
        recs.append(f"Investigate repeated executor error: {top_error}")

    if missed:
        symbols = ", ".join(item["symbol"] for item in missed[:4])
        recs.append(
            f"Review repeated blocked buys for {symbols}; focus on whether ML next-bar gating is too slow after stream confirmation."
        )

    top_gating = skip_summary.get("top_gating_conditions", {})
    if top_gating:
        gating_reason = next(iter(top_gating))
        if "ml_next_bar_prob_low" in gating_reason:
            recs.append(
                "Compare blocked buy timestamps against later favorable price paths before changing ML thresholds; prioritize calibration over loosening gates."
            )
        elif "existing open order for symbol" in gating_reason:
            recs.append(
                "Audit open-order lifecycle and fill-state refresh cadence to reduce duplicate-suppression dead time."
            )

    valid_plan_count = _safe_int(premarket_plan.get("valid_plan_count")) or 0
    selected_symbol_count = _safe_int(premarket_plan.get("selected_symbol_count")) or 0
    long_allowed = premarket_plan.get("long_allowed_symbols") or []
    if selected_symbol_count >= 15 and valid_plan_count <= 1:
        recs.append(
            f"Universe breadth is high ({selected_symbol_count} selected) but valid premarket plans are scarce ({valid_plan_count}); tighten premarket ranking quality before expanding symbols."
        )
    if selected_symbol_count and len(long_allowed) / max(selected_symbol_count, 1) < 0.5:
        recs.append(
            "Long-allowed coverage is thin relative to selected symbols; inspect structure-map exclusions to improve universe precision."
        )

    trade_permission = str(market_context.get("trade_permission", "")).lower()
    if trade_permission in {"cautious", "disabled"}:
        recs.append(
            f"Market context is {trade_permission}; keep optimization focused on selectivity and execution quality, not higher trade count."
        )

    return recs[:6]


def build_report(date_tag: str) -> dict[str, Any]:
    paths = RuntimeGuardPaths(
        date_tag=date_tag,
        log_dir=PROJECT_ROOT / "logs" / "execution" / date_tag,
        out_dir=PROJECT_ROOT / "system-data" / date_tag,
        daily_out_dir=PROJECT_ROOT / "daily_system_outputs" / date_tag,
    )
    paths.daily_out_dir.mkdir(parents=True, exist_ok=True)

    env_values = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}
    paper_only = str(env_values.get("APCA_PAPER", "")).lower() == "true"
    journal_rows, journal_summary = _load_executor_journal(paths.journal_path)
    skip_rows, skip_summary = _load_skips(paths.skipped_path)
    order_plan_rows, best_plan_by_symbol = _load_order_plan_rows(paths.log_dir, date_tag)
    positions = _current_positions(paths.latest_positions_path)

    premarket_plan = _load_json(paths.premarket_plan_path)
    structure_map = _load_json(paths.structure_map_path)
    market_context = _load_json(paths.market_context_path)
    universe = _load_universe_symbols(paths.latest_universe_path)
    order_actions = _order_actions_by_symbol(journal_rows)
    missed = _likely_missed_opportunities(skip_rows, best_plan_by_symbol, journal_rows)

    traded_symbols = sorted(
        {
            str(row.get("symbol", "")).strip().upper()
            for row in journal_rows
            if str(row.get("action", "")).upper() == "ORDER" and str(row.get("symbol", "")).strip()
        }
    )
    evaluated_symbols = sorted(best_plan_by_symbol.keys())
    selected_symbols = [s for s in universe.get("selected_symbols", []) if s]
    long_allowed = [str(s).strip().upper() for s in premarket_plan.get("long_allowed_symbols", []) if str(s).strip()]
    symbols_with_skips = sorted({row["symbol"] for row in skip_rows if row.get("symbol")})

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_tag": date_tag,
        "paper_runtime": {
            "apca_paper": env_values.get("APCA_PAPER"),
            "paper_only_confirmed": paper_only,
        },
        "runtime_status": {
            "journal_path": str(paths.journal_path),
            "journal_exists": paths.journal_path.exists(),
            "journal_last_modified_utc": datetime.fromtimestamp(paths.journal_path.stat().st_mtime, tz=timezone.utc).isoformat()
            if paths.journal_path.exists()
            else None,
            "last_event": journal_summary.get("last_event"),
            "action_counts": journal_summary.get("actions", {}),
            "top_error_messages": journal_summary.get("top_error_messages", {}),
            "current_positions": positions,
        },
        "planner_health": {
            "market_context_bias": market_context.get("overall_market_bias"),
            "trade_permission": market_context.get("trade_permission"),
            "risk_level": market_context.get("risk_level"),
            "premarket_status": premarket_plan.get("status"),
            "premarket_execution_mode": premarket_plan.get("execution_mode"),
            "selected_symbol_count": premarket_plan.get("selected_symbol_count"),
            "valid_plan_count": premarket_plan.get("valid_plan_count"),
            "long_allowed_count": len(long_allowed),
            "structure_symbol_count": structure_map.get("symbol_count"),
        },
        "universe_coverage": {
            "llm_source": universe.get("source"),
            "llm_market_tone": universe.get("market_tone"),
            "llm_symbol_count": len(universe.get("symbols", [])),
            "selected_symbol_count": len(selected_symbols),
            "long_allowed_count": len(long_allowed),
            "evaluated_symbol_count": len(evaluated_symbols),
            "traded_symbol_count": len(traded_symbols),
            "symbols_with_skips_count": len(symbols_with_skips),
            "selected_not_long_allowed": sorted(set(selected_symbols) - set(long_allowed))[:20],
            "long_allowed_not_evaluated": sorted(set(long_allowed) - set(evaluated_symbols))[:20],
            "evaluated_not_selected": sorted(set(evaluated_symbols) - set(selected_symbols))[:20],
            "traded_symbols": traded_symbols,
        },
        "skip_summary": skip_summary,
        "order_actions_by_symbol": order_actions,
        "likely_missed_opportunities": missed,
        "recommendations": _recommendations(
            journal_summary=journal_summary,
            skip_summary=skip_summary,
            missed=missed,
            premarket_plan=premarket_plan,
            market_context=market_context,
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build intraday runtime guard report for default run.py paper executor")
    parser.add_argument("--date", default=DEFAULT_DATE_TAG, help="Trading date tag in YYYYMMDD format")
    parser.add_argument("--stdout", action="store_true", help="Print JSON report to stdout")
    args = parser.parse_args()

    report = build_report(args.date)
    paths = RuntimeGuardPaths(
        date_tag=args.date,
        log_dir=PROJECT_ROOT / "logs" / "execution" / args.date,
        out_dir=PROJECT_ROOT / "system-data" / args.date,
        daily_out_dir=PROJECT_ROOT / "daily_system_outputs" / args.date,
    )
    paths.daily_out_dir.mkdir(parents=True, exist_ok=True)
    with paths.report_path.open("w") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    print(f"[runtime-guard] wrote {paths.report_path}")
    if args.stdout:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
