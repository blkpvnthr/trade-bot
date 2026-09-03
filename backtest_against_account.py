#!/usr/bin/env python3
"""Backtest the alpha strategy AGAINST the live Alpaca account: use the account's
real equity as starting capital, across one or more universe files.

Usage:
    python backtest_against_account.py                       # default: tickers2.txt tickers4.txt
    python backtest_against_account.py tickers2.txt foo.txt  # explicit universe files
"""
import json
import os
import subprocess
import sys

import pandas as pd

ROOT = "/Users/blkpvnthr/Desktop/trader"
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))
k = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_KEY")
s = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET")
paper = str(os.getenv("ALPACA_PAPER_ONLY", "true")).lower() in {"1", "true", "yes", "paper"}

from alpaca.trading.client import TradingClient
c = TradingClient(k, s, paper=paper)
acct = c.get_account()
equity = round(float(acct.equity), 2)
pos = c.get_all_positions()

START, END = "2026-05-12", "2026-06-04"
CACHE_S, CACHE_E = START.replace("-", ""), END.replace("-", "")
_argv = sys.argv[1:]
AV_CHAINS = "--av-chains" in _argv
UNIVERSE_FILES = [a for a in _argv if not a.startswith("--")] or ["tickers2.txt", "tickers4.txt"]

CONFIGS = [
    ("noovn", "$500/trade, no overnight", ["--risk-dollars", "500"]),
    ("ovn",   "$500/trade + overnight",   ["--risk-dollars", "500", "--force-overnight"]),
]

print(f"ACCOUNT: equity=${equity:,.2f}  endpoint={'PAPER' if paper else 'LIVE'}  "
      f"(live holdings: {len(pos)} positions, NOT traded here)")
print(f"WINDOW: {START}..{END}   universes: {', '.join(UNIVERSE_FILES)}\n")


def parse_universe(path):
    raw = open(path).read()
    return [t.strip().strip('"').upper() for t in raw.replace("\n", ",").split(",")
            if t.strip().strip('"')]


def has_data(sym):
    p = os.path.join(ROOT, "backtest_cache", f"{sym}_{CACHE_S}_{CACHE_E}_1m.parquet")
    if not os.path.exists(p):
        return None
    try:
        return len(pd.read_parquet(p)) > 0
    except Exception:
        return False


for uf in UNIVERSE_FILES:
    path = uf if os.path.isabs(uf) else os.path.join(ROOT, uf)
    stem = os.path.splitext(os.path.basename(path))[0]
    names = parse_universe(path)
    syms = ",".join(names)
    print("=" * 92)
    print(f"UNIVERSE {os.path.basename(path)} ({len(names)} names): {syms}")
    print("=" * 92)
    print(f"  {'config':<28} | {'$/day':>7} {'total$':>8} {'win%':>5} {'maxDD%':>7} {'trades':>6} {'finalEq':>9}")
    print("  " + "-" * 78)
    traded = {}
    for key, label, extra in CONFIGS:
        tag = f"acct_{stem}_{key}" + ("_av" if AV_CHAINS else "")
        args = [sys.executable, "backtest_alpha_replay.py", "--symbols", syms,
                "--start", START, "--end", END, "--equity", str(equity),
                "--exposure-pct", "0.15", "--costs", "--quiet", "--run-tag", tag, *extra]
        if AV_CHAINS:
            args.append("--av-chains")
        r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
        sp = os.path.join(ROOT, "backtest_outputs", tag, "summary.json")
        if not os.path.exists(sp):
            print(f"  {label:<28} | FAILED: {r.stderr.strip()[-60:]}")
            continue
        js = json.load(open(sp))
        print(f"  {label:<28} | {js['avg_realized_pnl_per_day']:>7.1f} {js['total_realized_pnl']:>8.0f} "
              f"{js['win_rate_pct']:>5.1f} {js['max_drawdown_pct']:>7.2f} {js['entries_filled']:>6} "
              f"{js['final_equity']:>9.0f}")
        tf = os.path.join(ROOT, "backtest_outputs", tag, "trades.csv")
        if os.path.exists(tf) and os.path.getsize(tf) > 0:
            try:
                t = pd.read_csv(tf)
            except pd.errors.EmptyDataError:
                t = None
            if t is not None and len(t) and "event" in t.columns:
                for u, n in t[t.event == "entry"].groupby("underlying").size().items():
                    traded[u] = traded.get(u, 0) + int(n)

    # coverage
    n_tr = n_nd = n_ns = 0
    print(f"\n  COVERAGE ({os.path.basename(path)}):")
    print(f"  {'ticker':<8}{'data?':<7}{'entries':>8}   status")
    print("  " + "-" * 50)
    for s_ in names:
        hd = has_data(s_)
        e = traded.get(s_, 0)
        data = "n/a" if hd is None else ("yes" if hd else "NO")
        if e > 0:
            status = "TRADED"; n_tr += 1
        elif hd is False:
            status = "no Alpaca data (skipped)"; n_nd += 1
        elif hd:
            status = "data, no qualifying setup"; n_ns += 1
        else:
            status = "not fetched"
        print(f"  {s_:<8}{data:<7}{e:>8}   {status}")
    print("  " + "-" * 50)
    print(f"  traded {n_tr} | no-data {n_nd} | data-no-signal {n_ns}  (of {len(names)})\n")

print("ACCT_BT_DONE")
