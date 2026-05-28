#!/usr/bin/env python3
"""
scripts/diagnose_strategy.py

Analyse execution logs for a strategy variant and surface failure patterns.

Identifies:
  1. late_entries        — entry after ORB/sweep window; high slippage vs plan
  2. bad_regime          — entries taken when market_context flagged bearish/mixed
  3. weak_volume         — volume at entry below confirmation threshold
  4. excessive_reentry   — same symbol entered >N times in a session
  5. winners_giving_back — MFE >> close pnl (profit given back before exit)
  6. stops_too_tight     — high stop-hit rate on small adverse moves
  7. stops_too_wide      — large losses per stopped trade vs ATR
  8. poor_symbol_selection — consistent losers in same symbol across days
  9. overtrading         — trade_count >> expected for strategy type
 10. undertrading        — trade_count << expected (too many filters blocking)

Outputs:
  research_requests/diagnose_{strategy_id}_{date}.json
  Prints human-readable summary to stdout.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")
TODAY = datetime.now(ET).strftime("%Y%m%d")


def load_execution_events(strategy_id: str, lookback_days: int = 30) -> list[dict]:
    """Load all JSONL execution events for a strategy variant."""
    from datetime import timedelta
    log_root = PROJECT_ROOT / "logs" / "execution"
    cutoff = datetime.now(ET) - timedelta(days=lookback_days)
    events = []
    for date_dir in sorted(log_root.glob("*")):
        try:
            dir_date = datetime.strptime(date_dir.name, "%Y%m%d").replace(tzinfo=ET)
        except ValueError:
            continue
        if dir_date < cutoff:
            continue
        for pattern in [
            f"{date_dir.name}_execution_events.jsonl",
            f"{date_dir.name}_executed_orders.jsonl",
            f"{date_dir.name}_sr_rl_agent_events.jsonl",
        ]:
            candidate = date_dir / strategy_id / pattern
            if not candidate.exists():
                candidate = date_dir / pattern
            if not candidate.exists():
                continue
            try:
                with open(candidate) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ev = json.loads(line)
                            sid = ev.get("strategy_id", ev.get("strategy", ""))
                            if strategy_id in sid or not sid:
                                events.append(ev)
                        except json.JSONDecodeError:
                            continue
            except OSError:
                continue
    return events


def diagnose(strategy_id: str, lookback_days: int = 30) -> dict[str, Any]:
    events = load_execution_events(strategy_id, lookback_days)

    if not events:
        return {
            "strategy_id": strategy_id,
            "trade_count": 0,
            "findings": [],
            "summary": "No execution data found for this variant.",
        }

    df = pd.DataFrame(events)

    findings: list[dict[str, Any]] = []

    # ── 1. Late entries ────────────────────────────────────────────────────
    if "entry_delay_bars" in df.columns:
        late = df[df["entry_delay_bars"] > 3]
        if len(late) > 0:
            avg_delay = float(late["entry_delay_bars"].mean())
            findings.append({
                "pattern": "late_entries",
                "severity": "high" if avg_delay > 8 else "medium",
                "count": len(late),
                "detail": f"{len(late)} entries delayed >{3} bars (avg {avg_delay:.1f} bars).",
                "recommendation": "Reduce entry filters or check data latency. Consider limit orders.",
            })

    # ── 2. Bad regime alignment ────────────────────────────────────────────
    if "regime" in df.columns:
        bad_regime = df[df["regime"].isin(["bearish", "mixed"])]
        if len(bad_regime) > len(df) * 0.3:
            findings.append({
                "pattern": "bad_regime",
                "severity": "high",
                "count": len(bad_regime),
                "detail": f"{len(bad_regime)}/{len(df)} entries taken in unfavourable regime.",
                "recommendation": "Add regime gate in strategy config or tighten market_context filter.",
            })

    # ── 3. Weak volume at entry ────────────────────────────────────────────
    if "volume_ratio" in df.columns:
        weak_vol = df[df["volume_ratio"] < 1.0]
        if len(weak_vol) > len(df) * 0.25:
            findings.append({
                "pattern": "weak_volume",
                "severity": "medium",
                "count": len(weak_vol),
                "detail": f"{len(weak_vol)} entries with volume_ratio < 1.0.",
                "recommendation": "Increase volume_confirm_mult param or tighten volume_lookback.",
            })

    # ── 4. Excessive reentry ──────────────────────────────────────────────
    if "symbol" in df.columns and "date" in df.columns:
        reentry_counts = df.groupby(["date", "symbol"]).size()
        excessive = reentry_counts[reentry_counts > 3]
        if len(excessive) > 0:
            findings.append({
                "pattern": "excessive_reentry",
                "severity": "medium",
                "count": int(excessive.sum()),
                "detail": f"{len(excessive)} symbol-day combinations with >3 entries.",
                "recommendation": "Add max_entries_per_day or no-reentry cooldown per symbol.",
            })

    # ── 5. Winners giving back ────────────────────────────────────────────
    if "mfe" in df.columns and "pnl" in df.columns:
        df_pnl = df.dropna(subset=["mfe", "pnl"])
        df_pnl = df_pnl[pd.to_numeric(df_pnl["mfe"], errors="coerce").notna()]
        df_pnl["mfe"] = pd.to_numeric(df_pnl["mfe"])
        df_pnl["pnl"] = pd.to_numeric(df_pnl["pnl"])
        winners = df_pnl[df_pnl["mfe"] > 0]
        if len(winners) > 3:
            giveback_ratio = (winners["mfe"] - winners["pnl"]) / winners["mfe"].replace(0, np.nan)
            high_giveback = giveback_ratio[giveback_ratio > 0.5]
            if len(high_giveback) > len(winners) * 0.4:
                findings.append({
                    "pattern": "winners_giving_back",
                    "severity": "high",
                    "count": len(high_giveback),
                    "detail": f"{len(high_giveback)} winners gave back >50% of MFE before exit.",
                    "recommendation": "Add trailing stop or reduce atr_target_mult. Consider partial exits.",
                })

    # ── 6. Stops too tight ────────────────────────────────────────────────
    if "exit_reason" in df.columns:
        stop_hits = df[df["exit_reason"].str.contains("stop", case=False, na=False)]
        if len(stop_hits) > len(df) * 0.5:
            findings.append({
                "pattern": "stops_too_tight",
                "severity": "high",
                "count": len(stop_hits),
                "detail": f"{len(stop_hits)}/{len(df)} exits were stop hits (>{50:.0f}%).",
                "recommendation": "Increase atr_stop_mult. Check if ATR period needs adjustment.",
            })

    # ── 7. Stops too wide ─────────────────────────────────────────────────
    if "pnl" in df.columns and "atr_at_entry" in df.columns:
        losers = df[pd.to_numeric(df["pnl"], errors="coerce") < 0].copy()
        losers["pnl"] = pd.to_numeric(losers["pnl"], errors="coerce")
        losers["atr_at_entry"] = pd.to_numeric(losers["atr_at_entry"], errors="coerce")
        losers = losers.dropna(subset=["pnl", "atr_at_entry"])
        if len(losers) > 3:
            loss_atr_ratio = (-losers["pnl"]) / losers["atr_at_entry"].replace(0, np.nan)
            wide_stops = loss_atr_ratio[loss_atr_ratio > 3.0]
            if len(wide_stops) > len(losers) * 0.3:
                findings.append({
                    "pattern": "stops_too_wide",
                    "severity": "medium",
                    "count": len(wide_stops),
                    "detail": f"{len(wide_stops)} losses exceeded 3× ATR at entry.",
                    "recommendation": "Reduce atr_stop_mult or add max-loss-per-trade cap.",
                })

    # ── 8. Poor symbol selection ──────────────────────────────────────────
    if "symbol" in df.columns and "pnl" in df.columns:
        sym_pnl = df.groupby("symbol")["pnl"].apply(
            lambda x: pd.to_numeric(x, errors="coerce").sum()
        )
        chronic_losers = sym_pnl[sym_pnl < -50]
        if len(chronic_losers) > 0:
            findings.append({
                "pattern": "poor_symbol_selection",
                "severity": "medium",
                "count": len(chronic_losers),
                "detail": f"Symbols with cumulative loss >$50: {list(chronic_losers.index[:5])}.",
                "recommendation": "Add symbol-level exclusion list. Review universe selection quality.",
            })

    # ── 9. Overtrading ────────────────────────────────────────────────────
    if "date" in df.columns:
        daily_counts = df.groupby("date").size()
        high_count_days = daily_counts[daily_counts > 20]
        if len(high_count_days) > len(daily_counts) * 0.3:
            findings.append({
                "pattern": "overtrading",
                "severity": "medium",
                "count": int(high_count_days.sum()),
                "detail": f"{len(high_count_days)} days with >20 trades.",
                "recommendation": "Add max_entries_per_day limit. Tighten confirmation filters.",
            })

    # ── 10. Undertrading ──────────────────────────────────────────────────
    if len(events) > 0 and "date" in df.columns:
        days_active = df["date"].nunique() if "date" in df.columns else lookback_days
        trades_per_day = len(df) / max(days_active, 1)
        if trades_per_day < 0.5 and days_active >= 5:
            findings.append({
                "pattern": "undertrading",
                "severity": "low",
                "count": len(df),
                "detail": f"Only {trades_per_day:.1f} trades/day over {days_active} active days.",
                "recommendation": "Loosen volume/confirmation thresholds or expand symbol universe.",
            })

    return {
        "strategy_id": strategy_id,
        "trade_count": len(df),
        "lookback_days": lookback_days,
        "generated_at": TODAY,
        "findings": findings,
        "summary": (
            f"Found {len(findings)} pattern(s) across {len(df)} trades."
            if findings else "No significant failure patterns detected."
        ),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose a strategy variant")
    parser.add_argument("strategy_id", help="Strategy variant ID (e.g. tux_mtf_original)")
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--out-dir", default="research_requests")
    args = parser.parse_args()

    result = diagnose(args.strategy_id, args.lookback_days)

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"diagnose_{args.strategy_id}_{TODAY}.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Also write to daily_system_outputs/strategy_diagnostics.json (last-run wins)
    daily_out = PROJECT_ROOT / "daily_system_outputs"
    daily_out.mkdir(parents=True, exist_ok=True)
    with open(daily_out / "strategy_diagnostics.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"[diagnose] {result['summary']}")
    for finding in result["findings"]:
        print(f"  [{finding['severity'].upper()}] {finding['pattern']}: {finding['detail']}")
        print(f"    → {finding['recommendation']}")
    print(f"\n[diagnose] full report → {out_path}")


if __name__ == "__main__":
    main()
