#!/usr/bin/env python3
"""Clean cost re-rank: run the sizing/overnight sweep grid TWICE (no-cost vs with-cost)
at the LIVE Alpaca account equity, on the same universe/window, and diff per config.
Both legs use identical equity so the cost delta is honest."""
import csv
import os
import shutil
import subprocess
import sys

ROOT = "/Users/blkpvnthr/Desktop/trader"
PY = sys.executable
SWEEP_CSV = os.path.join(ROOT, "backtest_outputs", "sweep", "sweep_results.csv")

# --- live account equity from Alpaca ---
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))
k = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_KEY")
s = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET")
paper = str(os.getenv("ALPACA_PAPER_ONLY", "true")).lower() in {"1", "true", "yes", "paper"}
from alpaca.trading.client import TradingClient
acct = TradingClient(k, s, paper=paper).get_account()
EQUITY = round(float(acct.equity), 2)

# Universe = the original cost-rerank tickers set (cached, fast, directly comparable).
TICKERS = "ASX,MU,NBIX,NOK,VICR,LRCX,DELL"
START, END = "2026-05-12", "2026-06-04"

print(f"CLEAN COST RE-RANK  @ LIVE account equity ${EQUITY:,.2f} ({'PAPER' if paper else 'LIVE'})")
print(f"universe: {TICKERS}   window: {START}..{END}")
print("both legs identical except --costs → honest cost delta\n", flush=True)


AV_CHAINS = "--av-chains" in sys.argv or os.getenv("RERANK_AV", "") == "1"


def run(costs, save_as):
    args = [PY, "backtest_alpha_replay.py", "--sweep", "--symbols", TICKERS,
            "--start", START, "--end", END, "--equity", str(EQUITY)]
    if costs:
        args.append("--costs")
    if AV_CHAINS:
        args.append("--av-chains")
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-800:])
    dst = os.path.join(ROOT, "backtest_outputs", "sweep", save_as)
    if os.path.exists(SWEEP_CSV):
        shutil.copy(SWEEP_CSV, dst)
    return dst


def load(path):
    out = {}
    if os.path.exists(path):
        for row in csv.DictReader(open(path)):
            out[(row["risk_pct"], row["exposure_pct"], row["overnight"])] = row
    return out


_sfx = "_av" if AV_CHAINS else ""
print(f"chain source: {'REAL AV historical' if AV_CHAINS else 'synthetic Black-Scholes'}\n", flush=True)
print(">>> leg 1/2: NO costs ...", flush=True)
nc = load(run(False, f"rerank_acct_nocost{_sfx}.csv"))
print(">>> leg 2/2: WITH costs ...", flush=True)
ct = load(run(True, f"rerank_acct_costs{_sfx}.csv"))

print("\n" + "=" * 78)
print(f"COST IMPACT @ ${EQUITY:,.0f}  chains={'REAL_AV' if AV_CHAINS else 'synthetic'}  ($/day, no-cost → with-cost)")
print("=" * 78)
print(f"  {'risk%':>6} {'expo%':>6} {'overnight':>9} | {'nocost':>8} {'costs':>8} {'delta':>8} {'trades':>7}")
print("  " + "-" * 60)
for key in sorted(ct, key=lambda kk: -float(ct[kk]["avg_per_day"])):
    if key not in nc:
        continue
    a = float(nc[key]["avg_per_day"]); b = float(ct[key]["avg_per_day"])
    r, e, o = key
    print(f"  {float(r)*100:>5.1f}% {float(e)*100:>5.0f}% {('YES' if o=='True' else 'no'):>9} | "
          f"{a:>8.1f} {b:>8.1f} {b-a:>+8.1f} {ct[key]['entries']:>7}")
print("=" * 78)
print(f"  saved: rerank_acct_nocost.csv / rerank_acct_costs.csv  (backtest_outputs/sweep/)")
print("RERANK_DONE")
