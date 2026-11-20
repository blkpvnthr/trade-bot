#!/usr/bin/env python3
"""Pre-market operational rehearsal: one full paper/dry-run cycle.

Exercises the exact production path in run.py (universe -> features ->
scanner candidates -> target/rebalance/exit plan -> risk/kill-switch gating
-> dry-run order gate) but with every shared state/log/output file
redirected into an isolated timestamped directory first, so this rehearsal
cannot race with or overwrite anything a live run.py process sharing this
checkout is reading or writing (logs/execution/*.json, the scanner CSVs,
the kill-switch state file).

Never calls the broker's order-submission methods (dry_run=True all the
way down to AlpacaBroker.submit_market_order / submit_limit_order).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import run  # noqa: E402

TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
REHEARSAL_DIR = PROJECT_ROOT / "daily_system_outputs" / f"premarket_rehearsal_{TS}"
REHEARSAL_LOG_DIR = REHEARSAL_DIR / "logs"
REHEARSAL_DIR.mkdir(parents=True, exist_ok=True)
REHEARSAL_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Redirect every shared state/output path module-level constant run.py would
# otherwise read/write, so this process touches nothing a live run.py
# process in this checkout is also touching.
run.OUT_DIR = REHEARSAL_DIR
run.LOG_DIR = REHEARSAL_LOG_DIR
run.DEFAULT_REBALANCE_ORDERS = REHEARSAL_DIR / "rebalance_orders.csv"
run.DEFAULT_EXIT_ORDERS = REHEARSAL_DIR / "exit_orders.csv"
run.JOURNAL_PATH = REHEARSAL_LOG_DIR / "executor_journal.csv"
run.EQUITY_CURVE_PATH = REHEARSAL_LOG_DIR / "executor_equity_curve.csv"
run.EVENTS_JSONL = REHEARSAL_LOG_DIR / "execution_events.jsonl"
run.KILL_SWITCH_PATH = REHEARSAL_LOG_DIR / "kill_switch_state.json"
run.SCANNER_STATE_PATH = REHEARSAL_LOG_DIR / "scanner_state.json"
run.SCANNER_STATE_PATH_V2 = run.SCANNER_STATE_PATH
run.SCANNER_SCORES_PATH = REHEARSAL_DIR / "latest_scanner_scores.csv"
run.OPTION_BOOSTER_CANDIDATES_PATH = REHEARSAL_DIR / "latest_option_booster_candidates.csv"
run.OPTION_CANDIDATES_PATH = REHEARSAL_DIR / "latest_realtime_options_candidates.csv"

print(f"[rehearsal] isolated output dir = {REHEARSAL_DIR}")
print(f"[rehearsal] isolated log dir    = {REHEARSAL_LOG_DIR}")
print("[rehearsal] NOTE: live run.py processes in this checkout are untouched -- "
      "all paths above are redirected copies, not the shared logs/ or daily_system_outputs/ files.")

config = run.ExecutorConfig(
    paper=True,
    live=False,
    dry_run=True,
    poll_once=True,
    rebalance_orders_path=run.DEFAULT_REBALANCE_ORDERS,
    exit_orders_path=run.DEFAULT_EXIT_ORDERS,
    require_market_open=False,
    allow_closed_market=True,
    enable_kill_switch=True,
    scanner=run.ScannerConfig(),
    options_booster=run.OptionsTradingConfig(enabled=False),
)

run.print_startup_banner_and_guard(config)

summary: dict = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "rehearsal_dir": str(REHEARSAL_DIR),
    "status": "unknown",
}

try:
    trader = run.ProductionOrderExecutor(config)

    candidates_df = None
    if trader.scanner is not None:
        rebalance, exits, scores = trader.scanner.build_orders()
        candidates_df = scores
    else:
        rebalance, exits = trader._load_orders()

    plan, results, acct_after, positions_before = trader.run_once_from_frames(rebalance, exits)

    results_df = results if hasattr(results, "to_dict") else __import__("pandas").DataFrame(results)
    planned = results_df[results_df.get("status", "") == "dry_run_planned"] if not results_df.empty else results_df

    summary.update(
        {
            "status": "ok",
            "candidate_count": int(len(candidates_df)) if candidates_df is not None else 0,
            "scanner_signal_rows": int(len(rebalance)) + int(len(exits)),
            "plan_rows": int(len(plan)) if plan is not None else 0,
            "planned_order_count": int(len(planned)) if not results_df.empty else 0,
            "result_rows": results_df.to_dict(orient="records") if not results_df.empty else [],
            "broker_dry_run_order_attempts": trader.broker.dry_run_order_attempts,
            "broker_dry_run_flag": trader.broker.dry_run,
            "account_after": asdict(acct_after),
            "open_positions_before": int(len(positions_before)) if positions_before is not None else 0,
            "kill_switch_state_path": str(run.KILL_SWITCH_PATH),
            "kill_switch_state": trader._read_kill_state(),
        }
    )

    print("\n===== REHEARSAL SUMMARY =====")
    print(f"candidate_count (scanner scores rows) = {summary['candidate_count']}")
    print(f"scanner_signal_rows (rebalance+exits before sizing) = {summary['scanner_signal_rows']}")
    print(f"plan_rows (after risk/cash/exposure sizing) = {summary['plan_rows']}")
    print(f"planned_order_count (dry_run_planned, would-submit) = {summary['planned_order_count']}")
    print(f"broker.submit_order reached = {summary['broker_dry_run_order_attempts'] > 0} "
          f"(dry_run_order_attempts={summary['broker_dry_run_order_attempts']}, all blocked pre-network)")
    for r in summary["result_rows"]:
        print(f"  - {r.get('symbol')} {r.get('side')} qty={r.get('qty')} status={r.get('status')} reason={r.get('reason', r.get('message', ''))}")

except Exception as exc:
    summary["status"] = "error"
    summary["error"] = repr(exc)
    print(f"[rehearsal] ERROR: {exc!r}")
    raise
finally:
    summary_path = REHEARSAL_DIR / "rehearsal_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[rehearsal] wrote summary -> {summary_path}")
