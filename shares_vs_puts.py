#!/usr/bin/env python3
"""shares_vs_puts.py — the one comparison: shares-only (A) vs shares+puts (B).

Everything identical between legs (QQQ/SPY/IWM, 200DMA+2%, rising-200 gate, same sizing,
same weekly rebalance) — the ONLY difference is whether the bear sleeve buys ~45Δ puts (B)
or sits in cash (A, --no-puts). Same data source (DAILY_SOURCE=yf) for both so shares P&L is
identical in any window where B held no puts.

Question being answered: does the put hedge improve portfolio-level return-per-unit-risk
enough to justify the options machinery? Per-leg panel + portfolio aggregate + worst-window.
"""
from __future__ import annotations
import os, sys, json, subprocess
from pathlib import Path

ROOT = "/Users/blkpvnthr/Desktop/trader"
WINDOWS = [
    ("2015-16",  "2015-01-01", "2016-12-31"),
    ("2018",     "2018-01-01", "2019-01-01"),
    ("2022",     "2022-01-03", "2023-01-03"),
    ("2023",     "2023-01-01", "2024-01-01"),
    ("2024",     "2024-01-01", "2025-01-01"),
    ("H1-25",    "2025-01-01", "2025-07-01"),
    ("2025-26",  "2025-06-02", "2026-04-27"),
]


def run_leg(tag, start, end, no_puts):
    rt = f"svp_{'A' if no_puts else 'B'}_{tag}"
    cmd = [sys.executable, "trend_overlay_engine.py", "--symbols", "QQQ,SPY,IWM",
           "--start", start, "--end", end, "--require-rising-200", "--run-tag", rt]
    if no_puts:
        cmd.append("--no-puts")
    env = dict(os.environ, DAILY_SOURCE="yf")
    subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    return json.load(open(Path(ROOT) / "backtest_outputs" / rt / "summary.json"))


def row(s):
    m = s.get("metrics", {})
    bk = s.get("by_kind", {})
    return dict(ret=s.get("return_pct", 0), dd=s.get("max_drawdown_pct", 0),
                cagr=m.get("cagr_pct", 0), sharpe=m.get("sharpe", 0), sortino=m.get("sortino", 0),
                ulcer=m.get("ulcer", 0), wmo=m.get("worst_month_pct", 0),
                sh=bk.get("shares", {}).get("pnl", 0), pu=bk.get("long_put", {}).get("pnl", 0))


def main():
    A, B = {}, {}
    for tag, start, end in WINDOWS:
        print(f"  running {tag} ...", flush=True)
        B[tag] = row(run_leg(tag, start, end, no_puts=False))
        A[tag] = row(run_leg(tag, start, end, no_puts=True))

    hdr = f"{'Window':<9}{'ret%':>8}{'maxDD':>8}{'CAGR':>7}{'Sharpe':>8}{'Sortino':>9}{'Ulcer':>7}{'wMo%':>7}"
    for name, D in (("A — SHARES ONLY", A), ("B — SHARES + PUTS", B)):
        print(f"\n=== {name} ===")
        print(hdr)
        for tag, _, _ in WINDOWS:
            r = D[tag]
            print(f"{tag:<9}{r['ret']:>8.2f}{r['dd']:>8.2f}{r['cagr']:>7.1f}{r['sharpe']:>8.2f}"
                  f"{r['sortino']:>9.2f}{r['ulcer']:>7.2f}{r['wmo']:>7.1f}")

    # portfolio-level aggregates across windows (equal-weight per window)
    def agg(D):
        n = len(WINDOWS)
        avg = lambda k: sum(D[t[0]][k] for t in WINDOWS) / n
        worst_ret = min(D[t[0]]["ret"] for t in WINDOWS)
        worst_dd = max(D[t[0]]["dd"] for t in WINDOWS)
        return dict(avg_ret=avg("ret"), avg_dd=avg("dd"), avg_sharpe=avg("sharpe"),
                    avg_sortino=avg("sortino"), avg_ulcer=avg("ulcer"),
                    worst_ret=worst_ret, worst_dd=worst_dd,
                    win_windows=sum(1 for t in WINDOWS if D[t[0]]["ret"] > 0))
    aA, aB = agg(A), agg(B)
    print("\n=== PORTFOLIO AGGREGATE (equal-weight across 7 windows) ===")
    print(f"{'metric':<16}{'A shares':>12}{'B sh+puts':>12}{'delta(B-A)':>12}")
    for label, k in [("avg ret%", "avg_ret"), ("avg maxDD%", "avg_dd"), ("avg Sharpe", "avg_sharpe"),
                     ("avg Sortino", "avg_sortino"), ("avg Ulcer", "avg_ulcer"),
                     ("worst window%", "worst_ret"), ("worst DD%", "worst_dd"),
                     ("positive windows", "win_windows")]:
        print(f"{label:<16}{aA[k]:>12.2f}{aB[k]:>12.2f}{aB[k]-aA[k]:>12.2f}")

    json.dump({"A": A, "B": B, "agg_A": aA, "agg_B": aB},
              open(Path(ROOT) / "backtest_outputs" / "shares_vs_puts.json", "w"), indent=2)
    print("\nSVP_DONE")


if __name__ == "__main__":
    main()
