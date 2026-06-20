#!/usr/bin/env python3
"""Audit human WINS/LOSSES/FLAT folder labels against actual PnL outcomes.

The folder a CSV was filed under is a human PRIOR ("I think this strategy works
on this ticker"). The actual outcome comes from Net PnL. This report compares
the two per strategy-symbol -- it never lets the label affect the metrics.

    python scripts/label_audit.py --timeframe 15Min
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta.performance import PerformanceDB
from rl_meta.labels import label_mismatch, bucket_accuracy, normalize_bucket


def _rows(db, timeframe):
    labels = db.labels(timeframe)
    out = []
    for c in db.cells(split="full"):
        if c["symbol"] == "AGGREGATE" or c["timeframe"] != timeframe:
            continue
        m = c["metrics"]
        if m["trade_count"] == 0:
            continue
        net = m["expectancy"] * m["trade_count"]
        out.append({
            "strategy_id": c["strategy_id"], "symbol": c["symbol"],
            "timeframe": c["timeframe"],
            "human_bucket": normalize_bucket(
                labels.get((c["strategy_id"], c["symbol"], c["timeframe"]))),
            "trade_count": m["trade_count"], "win_rate": m["win_rate"],
            "profit_factor": m["profit_factor"], "expectancy": m["expectancy"],
            "max_drawdown": m["max_drawdown"], "blown_up": m["blown_up"],
            "net_pnl": net,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", default="15Min")
    args = ap.parse_args()
    db = PerformanceDB()
    rows = _rows(db, args.timeframe)
    if not rows:
        print("No per-symbol TradingView data. Run import + backtest_strategies first.")
        return

    by_strat = defaultdict(list)
    for r in rows:
        by_strat[r["strategy_id"]].append(r)

    # 1) per strategy-symbol summary
    print("=== Strategy-symbol summary (outcome from PnL; bucket = human prior) ===")
    print(f"{'strategy':<18}{'sym':<6}{'bucket':<8}{'trades':>7}{'win%':>6}{'PF':>7}"
          f"{'expR$':>8}{'netPnL$':>11}{'maxDD':>7}{'blown':>6}")
    for sid in sorted(by_strat):
        for r in sorted(by_strat[sid], key=lambda x: -x["net_pnl"]):
            pf = "inf" if r["profit_factor"] == float("inf") else f"{r['profit_factor']:.2f}"
            print(f"{sid[:18]:<18}{r['symbol']:<6}{r['human_bucket']:<8}"
                  f"{r['trade_count']:>7}{r['win_rate']*100:>5.1f}%{pf:>7}"
                  f"{r['expectancy']:>8.2f}{r['net_pnl']:>11,.0f}"
                  f"{r['max_drawdown']*100:>6.1f}%{str(r['blown_up']):>6}")

    # 2) bucket accuracy
    acc = bucket_accuracy(rows)
    print("\n=== Bucket accuracy (human label vs actual outcome) ===")
    for b, (rate, n_ok, n) in acc.items():
        print(f"  {b:<7} {n_ok}/{n} = {rate*100:.0f}%")

    # 3) mismatch report
    print("\n=== Mismatches (human label disagrees with actual PnL) ===")
    miss = [(r, label_mismatch(r["human_bucket"], r["net_pnl"], r["profit_factor"]))
            for r in rows]
    miss = [(r, why) for r, why in miss if why]
    if not miss:
        print("  none")
    for r, why in sorted(miss, key=lambda x: x[0]["strategy_id"]):
        print(f"  {r['strategy_id'][:18]:<18} {r['symbol']:<6} {why}")

    # 4) works / fails by ticker
    print("\n=== Works / fails by ticker (by net PnL) ===")
    for sid in sorted(by_strat):
        ranked = sorted(by_strat[sid], key=lambda x: -x["net_pnl"])
        works = ", ".join(f"{r['symbol']}({r['net_pnl']:+,.0f})" for r in ranked[:4])
        fails = ", ".join(f"{r['symbol']}({r['net_pnl']:+,.0f})" for r in ranked[-3:]
                          if r["net_pnl"] < 0)
        print(f"  {sid[:18]:<18} works: {works}")
        if fails:
            print(f"  {'':<18} fails: {fails}")


if __name__ == "__main__":
    main()
