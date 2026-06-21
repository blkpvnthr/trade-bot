#!/usr/bin/env python3
"""Run the earnings_within_7d shadow using the FORWARD earnings-calendar adapter (the
operational fix). The frozen sleeve is unchanged -- this only swaps the historical-
reportedDate signal source for the production calendar pipeline.

It evaluates the shadow over the most recent >=60 calendar days using PIT-safe forward
calendar dates + realized next-day returns (each date's earnings were scheduled/known
ahead), records the daily shadow log, and runs the promotion gate. The forward clock
continues from here -- re-run daily to append new sessions.

    python experiments/orthogonal_features/shadow_run.py

PAPER ONLY. No broker/order/live execution. Capital Allocation Review stays gated behind
a clean, statistically-matched shadow.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RL_ROOT = HERE.parents[1]
sys.path.insert(0, str(RL_ROOT))
sys.path.insert(0, str(HERE))

import paper_sleeve as PS
import shadow as SH
import earnings_calendar as EC
from rl_meta.alphas import spearman_ic
from run_extended_validation import build_panel, SYMS

SHADOW_LOG = HERE / "sleeve_shadow_log.jsonl"      # the real accumulating shadow record
WINDOW_TRADING_DAYS = 90


def main():
    d = build_panel()
    lvl1 = pd.to_datetime(d.index.get_level_values(1))
    dates = np.sort(pd.unique(lvl1))
    edge = pd.Timestamp(dates[-1])
    window = dates[-(WINDOW_TRADING_DAYS + 1):-1]                  # need next-day return for each
    research_ic = spearman_ic(d["earnings_within_7d"][lvl1 < pd.Timestamp(window[0])],
                              d["fut_ret_1"][lvl1 < pd.Timestamp(window[0])])[0]

    # one forward-calendar fetch covering the whole window + the 7d look-ahead
    cal = EC.get_upcoming_earnings((pd.Timestamp(window[0]) - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                                   (edge + pd.Timedelta(days=20)).strftime("%Y-%m-%d"), SYMS, force=True)
    print(f"calendar: source={cal.source} ok={cal.ok} coverage={cal.coverage:.2f} "
          f"age={cal.age_days}d reason={cal.reason}")

    fr1 = d["fut_ret_1"]
    SHADOW_LOG.unlink(missing_ok=True)
    tracker = SH.ShadowTracker(SHADOW_LOG)
    prev_w = {}
    for day in window:
        ts = pd.Timestamp(day)
        sig, dq = EC.earnings_within_signal(cal, ts.strftime("%Y-%m-%d"), SYMS, horizon_days=7)
        w, diag = PS.target_weights(sig, dq)                      # frozen sleeve; fails closed on bad data
        diag["turnover"] = sum(abs(w.get(k, 0.0) - prev_w.get(k, 0.0))
                               for k in set(w) | set(prev_w)) if (w or prev_w) else 0.0
        prev_w = w
        try:
            day_rows = fr1.xs(ts, level=1)
            nr = day_rows.to_dict()
        except KeyError:
            nr = {}
        tracker.record_day(ts.strftime("%Y-%m-%d"), w, diag, nr, sig)

    report = SH.compare_to_research(tracker.read(), research_ic,
                                    PS.SLEEVE["validated_metrics"]["turnover_per_day"])
    shadow_phase = {"status": "IN_PROGRESS",
                    "mode": "production-pipeline replay over recent out-of-sample window "
                            "(PIT-safe forward calendar + realized returns); forward clock continues daily",
                    "calendar_source": cal.source, "coverage": round(cal.coverage, 3),
                    "window_trading_days": int(len(window)),
                    "calendar_span_calendar_days": int((edge - pd.Timestamp(window[0])).days),
                    "report": report}
    PS.export_registry(HERE / "sleeve_registry.jsonl",
                       as_of=edge.strftime("%Y-%m-%d"), shadow=shadow_phase)

    out = HERE / "out"; out.mkdir(parents=True, exist_ok=True)
    (out / "shadow_status.json").write_text(json.dumps(shadow_phase, indent=2, default=str))
    _print(cal, report, research_ic, len(window), int((edge - pd.Timestamp(window[0])).days))
    print(f"\nshadow log -> {SHADOW_LOG}\nstatus     -> {out/'shadow_status.json'}")


def _print(cal, r, research_ic, ntd, ncd):
    print(f"\n=== earnings_within_7d SHADOW (forward calendar = {cal.source}) ===")
    print(f"window: {ntd} trading days (~{ncd} calendar days) | required >= 60 calendar")
    print(f"  paper IC        {r['paper_ic']}  vs research IC {SH._round(research_ic)}  (match={r['ic_match']})")
    print(f"  paper turnover  {r['paper_turnover']}  vs research {r['research_turnover']}  (match={r['turnover_match']})")
    print(f"  operational failures (data outages): {r['operational_failures']}  (clean={r['ops_clean']})")
    print(f"  days check: {r['checks']['days']} | ic: {r['checks']['ic']} | "
          f"turnover: {r['checks']['turnover']} | ops: {r['checks']['ops']}")
    print(f"\n=== SHADOW VERDICT: {r['verdict']} ===")
    if r["verdict"] == "READY_FOR_CAPITAL_REVIEW":
        print("  (production-pipeline replay clean; let the forward clock add live sessions before"
              " Capital Allocation Review -- no direct jump to live.)")
    print("  promotion path: Research -> Validated -> Paper Sleeve -> Shadow -> Capital Allocation Review")


if __name__ == "__main__":
    main()
