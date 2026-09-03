#!/usr/bin/env python3
"""Phases (b) larger optionable universe + (c) slippage/commission re-rank.

(c): re-run the tickers.txt grid WITH transaction costs and diff vs the saved
     no-cost grid → shows how much edge survives realistic fills.
(b): sweep a 16-name liquid optionable universe (costs ON = most realistic) to
     test whether a bigger candidate pool fixes the small-sample fragility.
"""
import csv
import os
import shutil
import subprocess
import sys

ROOT = "/Users/blkpvnthr/Desktop/trader"
PY = sys.executable
SWEEP_CSV = os.path.join(ROOT, "backtest_outputs", "sweep", "sweep_results.csv")

TICKERS = "ASX,MU,NBIX,NOK,VICR,LRCX,DELL"
LARGE = "SPY,QQQ,AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA,AMD,AVGO,NFLX,MU,DELL,LRCX,NBIX"
START, END = "2026-05-12", "2026-06-04"


def run_sweep(symbols, costs, entry_interval, save_as):
    args = [PY, "backtest_alpha_replay.py", "--sweep", "--symbols", symbols,
            "--start", START, "--end", END, "--entry-interval", str(entry_interval)]
    if costs:
        args.append("--costs")
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stdout.write("STDERR:\n" + r.stderr[-1500:] + "\n")
    dst = os.path.join(ROOT, "backtest_outputs", "sweep", save_as)
    if os.path.exists(SWEEP_CSV):
        shutil.copy(SWEEP_CSV, dst)
    return dst


def load(path):
    if not os.path.exists(path):
        return {}
    out = {}
    for row in csv.DictReader(open(path)):
        key = (row["risk_pct"], row["exposure_pct"], row["overnight"])
        out[key] = row
    return out


# ── Phase (c): cost re-rank on tickers.txt ──────────────────────────────────
print("\n########## PHASE (c): TRANSACTION-COST RE-RANK (tickers.txt) ##########\n", flush=True)
nocost_backup = os.path.join(ROOT, "backtest_outputs", "sweep", "tickers_nocost.csv")
# Preserve the existing no-cost grid (sweep_results.csv currently holds it).
if os.path.exists(SWEEP_CSV):
    shutil.copy(SWEEP_CSV, nocost_backup)
costs_csv = run_sweep(TICKERS, costs=True, entry_interval=5, save_as="tickers_costs.csv")

nc = load(nocost_backup)
ct = load(costs_csv)
if nc and ct:
    print("\n  COST IMPACT ($/day, no-cost → with-cost):")
    print(f"  {'risk%':>6} {'expo%':>6} {'overnight':>9} | {'nocost':>7} {'costs':>7} {'delta':>7}")
    print("  " + "-" * 52)
    for key in sorted(ct, key=lambda k: -float(ct[k]["avg_per_day"])):
        if key not in nc:
            continue
        a = float(nc[key]["avg_per_day"]); b = float(ct[key]["avg_per_day"])
        r, e, o = key
        print(f"  {float(r)*100:>5.1f}% {float(e)*100:>5.0f}% {('YES' if o=='True' else 'no'):>9} | "
              f"{a:>7.0f} {b:>7.0f} {b-a:>+7.0f}")

# ── Phase (b): larger optionable universe (costs ON) ────────────────────────
print("\n\n########## PHASE (b): LARGER OPTIONABLE UNIVERSE (16 names, costs ON) ##########", flush=True)
print(f"  universe: {LARGE}")
print("  (entry-interval=10 for speed on the bigger universe)\n", flush=True)
run_sweep(LARGE, costs=True, entry_interval=10, save_as="largeuniv_costs.csv")

print("\nBC_DONE", flush=True)
