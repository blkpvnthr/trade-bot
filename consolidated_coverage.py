#!/usr/bin/env python3
"""Consolidated coverage across all universe files run via backtest_against_account.py.
For each ticker: had Alpaca 1m data? did it trade (entries across either config)?"""
import os
import glob
import pandas as pd

ROOT = "/Users/blkpvnthr/Desktop/trader"
CACHE_S, CACHE_E = "20260512", "20260604"
UNIVERSE_FILES = ["tickers.txt", "tickers2.txt", "tickers3.txt", "tickers4.txt"]


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


def traded_for(stem):
    out = {}
    for tag in (f"acct_{stem}_noovn", f"acct_{stem}_ovn"):
        f = os.path.join(ROOT, "backtest_outputs", tag, "trades.csv")
        if os.path.exists(f):
            t = pd.read_csv(f)
            if len(t):
                for u, n in t[t.event == "entry"].groupby("underlying").size().items():
                    out[u] = out.get(u, 0) + int(n)
    return out


grand = {"names": 0, "traded": 0, "nodata": 0, "nosig": 0}
print("=" * 70)
print("CONSOLIDATED COVERAGE — all universes (account backtest, $500/trade)")
print("=" * 70)
for uf in UNIVERSE_FILES:
    path = os.path.join(ROOT, uf)
    if not os.path.exists(path):
        print(f"\n[{uf}] missing — skipped")
        continue
    stem = os.path.splitext(uf)[0]
    names = parse_universe(path)
    traded = traded_for(stem)
    tr = nd = ns = 0
    rows = []
    for s in names:
        hd = has_data(s)
        e = traded.get(s, 0)
        if e > 0:
            st = "TRADED"; tr += 1
        elif hd is False:
            st = "no data"; nd += 1
        elif hd:
            st = "data, no setup"; ns += 1
        else:
            st = "not fetched"
        rows.append((s, "yes" if hd else ("NO" if hd is False else "n/a"), e, st))
    print(f"\n[{uf}]  {len(names)} names  →  traded {tr} | no-data {nd} | data-no-signal {ns}")
    print(f"  {'ticker':<8}{'data':<6}{'entries':>8}  status")
    for s, d, e, st in rows:
        print(f"  {s:<8}{d:<6}{e:>8}  {st}")
    grand["names"] += len(names); grand["traded"] += tr
    grand["nodata"] += nd; grand["nosig"] += ns

print("\n" + "=" * 70)
print(f"GRAND TOTAL: {grand['names']} ticker-slots  |  traded {grand['traded']}  "
      f"|  no-data {grand['nodata']}  |  data-no-signal {grand['nosig']}")
print("=" * 70)
