#!/usr/bin/env python3
"""Coverage report for a universe file: which names had data, which actually traded."""
import os
import sys
import glob
import pandas as pd

ROOT = "/Users/blkpvnthr/Desktop/trader"
UNI_FILE = os.path.join(ROOT, "tickers2.txt")
START, END = "20260512", "20260604"
TAGS = ["acct_def", "acct_ovn"]

raw = open(UNI_FILE).read()
names = [t.strip().strip('"').upper() for t in raw.replace("\n", ",").split(",")
         if t.strip().strip('"')]

# data availability from cached 1m parquet
def has_data(sym):
    p = os.path.join(ROOT, "backtest_cache", f"{sym}_{START}_{END}_1m.parquet")
    if not os.path.exists(p):
        return None  # unknown (not fetched)
    try:
        return len(pd.read_parquet(p)) > 0
    except Exception:
        return False

# traded set (union across configs)
traded = {}
for tag in TAGS:
    f = os.path.join(ROOT, "backtest_outputs", tag, "trades.csv")
    if os.path.exists(f):
        t = pd.read_csv(f)
        if len(t):
            for u, n in t[t.event == "entry"].groupby("underlying").size().items():
                traded[u] = traded.get(u, 0) + int(n)

print(f"UNIVERSE: {UNI_FILE}  ({len(names)} names)\n")
print(f"  {'ticker':<8}{'data?':<8}{'traded?':<9}{'entries':>8}   status")
print("  " + "-" * 56)
n_traded = n_nodata = n_nosig = 0
for s in names:
    hd = has_data(s)
    entries = traded.get(s, 0)
    if hd is None:
        data = "n/a"
    elif hd:
        data = "yes"
    else:
        data = "NO"
    if entries > 0:
        status = "TRADED"; n_traded += 1
    elif hd is False:
        status = "no Alpaca data (skipped)"; n_nodata += 1
    elif hd:
        status = "data, but no qualifying setup"; n_nosig += 1
    else:
        status = "not fetched"
    print(f"  {s:<8}{data:<8}{('yes' if entries else 'no'):<9}{entries:>8}   {status}")
print("  " + "-" * 56)
print(f"  traded: {n_traded}   no-data: {n_nodata}   data-but-no-signal: {n_nosig}   of {len(names)}")
