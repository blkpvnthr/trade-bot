#!/usr/bin/env python3
"""Out-of-sample validation: run ONE fixed config across several historical windows
to test whether the in-sample $/day holds up. Reads each run's summary.json cleanly."""
import csv
import json
import os
import subprocess
import sys

ROOT = "/Users/blkpvnthr/Desktop/trader"
SYMS = "ASX,MU,NBIX,NOK,VICR,LRCX,DELL"
# Fixed config under test: 4% risk per trade, overnight-hold ON, 15% exposure.
CONFIG = ["--risk-pct", "0.04", "--exposure-pct", "0.15", "--force-overnight"]
WINDOWS = [
    ("2026-05-12", "2026-06-04", "in-sample"),
    ("2026-04-13", "2026-05-08", "oos-1"),
    ("2026-03-16", "2026-04-09", "oos-2"),
    ("2026-02-17", "2026-03-12", "oos-3"),
    ("2026-01-20", "2026-02-12", "oos-4"),
]

print(f"{'window':>22} {'label':>10} | {'days':>4} {'$/day':>7} {'total$':>8} "
      f"{'win%':>5} {'maxDD%':>7} {'halts':>5} {'trades':>6}")
print("-" * 86)
agg = []
for st, en, label in WINDOWS:
    tag = f"oos_{st}_{en}".replace("-", "")
    r = subprocess.run(
        [sys.executable, "backtest_alpha_replay.py", "--symbols", SYMS,
         "--start", st, "--end", en, *CONFIG, "--quiet", "--run-tag", tag],
        cwd=ROOT, capture_output=True, text=True)
    p = os.path.join(ROOT, "backtest_outputs", tag, "summary.json")
    if not os.path.exists(p):
        print(f"{st+'..'+en:>22} {label:>10} | run failed: {r.stderr.strip()[-80:]}")
        continue
    s = json.load(open(p))
    d = list(csv.DictReader(open(os.path.join(ROOT, "backtest_outputs", tag, "daily.csv"))))
    halts = sum(1 for x in d if x["halted"] == "True")
    agg.append(s["avg_realized_pnl_per_day"])
    print(f"{st+'..'+en:>22} {label:>10} | {s['trading_days']:>4} "
          f"{s['avg_realized_pnl_per_day']:>7.0f} {s['total_realized_pnl']:>8.0f} "
          f"{s['win_rate_pct']:>5.1f} {s['max_drawdown_pct']:>7.2f} {halts:>5} "
          f"{s['entries_filled']:>6}")

if agg:
    import statistics
    print("-" * 86)
    mean = statistics.mean(agg)
    sd = statistics.pstdev(agg) if len(agg) > 1 else 0.0
    oos = agg[1:] if len(agg) > 1 else []
    print(f"  all windows  : mean ${mean:.0f}/day  std ${sd:.0f}  "
          f"min ${min(agg):.0f}  max ${max(agg):.0f}")
    if oos:
        print(f"  OOS only     : mean ${statistics.mean(oos):.0f}/day  "
              f"min ${min(oos):.0f}  max ${max(oos):.0f}  "
              f"(in-sample was ${agg[0]:.0f})")
print("OOS_DONE")
