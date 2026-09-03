#!/usr/bin/env python3
"""Repaint audit for sniper_strategy (source + exported trades).

Runs the static Pine-source scan and the trade-statistics implausibility check
(rl_meta.repaint_audit) on sniper_strategy and reports whether it is safe,
suspect, or invalid. The bar-by-bar replay comparison is approximated by the
trade-statistics check (no faithful Python re-implementation of the Pine exists;
the adverse-excursion distribution is the practical lookahead test).

Research-only: reads files; no broker/order/risk path.

    python scripts/repaint_audit.py --strategy sniper_strategy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta.repaint_audit import scan_pine_source, signal_implausibility, combine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="sniper_strategy")
    ap.add_argument("--pine", default=None, help="path to the Pine .txt (default pinescripts/<strategy>.txt)")
    ap.add_argument("--dataset", default="data/trade_outcomes/trade_outcome_dataset.parquet")
    ap.add_argument("--out", default="data/trade_outcomes")
    args = ap.parse_args()

    pine_path = Path(args.pine) if args.pine else Path("pinescripts") / f"{args.strategy}.txt"
    static = {"static_verdict": "unknown", "note": f"source not found: {pine_path}"}
    if pine_path.exists():
        static = scan_pine_source(pine_path.read_text())

    df = pd.read_parquet(args.dataset)
    trades = df[df["strategy_id"] == args.strategy]
    stats = signal_implausibility(trades) if len(trades) else {"stats_verdict": "unknown",
                                                               "note": "no trades"}
    verdict = combine(static, stats) if static.get("static_verdict") != "unknown" \
        and stats.get("stats_verdict") != "unknown" else "unknown"

    rep = {"strategy": args.strategy, "pine_source": str(pine_path),
           "static": static, "stats": stats, "verdict": verdict}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"repaint_audit_{args.strategy}.json").write_text(json.dumps(rep, indent=2, default=str))
    _print(rep)
    print(f"\nreport -> {out / f'repaint_audit_{args.strategy}.json'}")


def _print(rep):
    s = rep["static"]; st = rep["stats"]
    print(f"=== repaint audit: {rep['strategy']} ===\n")
    print(f"STATIC scan ({rep['pine_source']}): verdict {s.get('static_verdict')} "
          f"score {s.get('score')}")
    print(f"  signal built on repainting security(): {s.get('signal_uses_repainting_security')}")
    for h in s.get("hazards", []):
        print(f"  [sev {h['severity']} x{h['count']}] {h['description']}")
    print(f"\nSTATS implausibility (n={st.get('n')}): verdict {st.get('stats_verdict')}")
    print(f"  win_rate {st.get('win_rate')} | profit_factor {st.get('profit_factor')} | "
          f"mae~0 rate {st.get('mae_zero_rate')} | median_mae_pct {st.get('median_mae_pct')}")
    for f in st.get("flags", []):
        print(f"  FLAG: {f}")
    print(f"\n=== VERDICT: {rep['verdict'].upper()} ===")
    if rep["verdict"] == "invalid":
        print("-> Do NOT trust sniper_strategy return metrics. The strategy's entries depend on a "
              "repainting higher-timeframe security() call (pre-v3 Pine), and its realized trades "
              "are statistically implausible. Treat all sniper P&L as artifact until rebuilt non-"
              "repainting and re-backtested bar-by-bar.")
    elif rep["verdict"] == "suspect":
        print("-> sniper_strategy shows repaint/lookahead warning signs; verify before trusting returns.")
    else:
        print("-> No repaint hazards detected; sniper_strategy returns may be trusted (still paper-only).")


if __name__ == "__main__":
    main()
