#!/usr/bin/env python3
"""edge_scan_universe.py — search for the optimal universe AND structure.

Runs the held-to-expiry edge scan (edge_scan_spreads.scan) across a basket of liquid
optionable underlyings on REAL AV chains, and ranks each name by expectancy + t-stat
for each structure (long call / put credit spread / iron condor).

Goal: find where a *structural* edge (esp. the VRP credit spread) is strongest AND
most statistically robust — not just where directional beta happened to pay.
"""
import sys, json
sys.path.insert(0, "/Users/blkpvnthr/Desktop/trader")
import edge_scan_spreads as E

UNIVERSE = sys.argv[1:] or [
    "SPY","QQQ","IWM","DIA",                 # index ETFs (tight, liquid)
    "AAPL","MSFT","NVDA","AMD","META","AMZN","GOOGL","TSLA",  # megacap singles
]
STRUCTS = [("long_call","LONG CALL"), ("put_credit_spread","PUT CR SPREAD"), ("iron_condor","IRON CONDOR")]

rows = {}
for sym in UNIVERSE:
    try:
        print(f">>> scanning {sym} ...", flush=True)
        rows[sym] = E.scan(sym)
    except Exception as ex:
        print(f"    {sym} failed: {str(ex)[:90]}", flush=True)

for key, label in STRUCTS:
    ranked = []
    for sym, res in rows.items():
        s = E.summarize(res.get(key, []))
        if s: ranked.append((sym, s))
    ranked.sort(key=lambda x: -x[1]["t"])   # rank by statistical robustness (t-stat)
    print(f"\n================ {label} — ranked by t-stat (robustness) ================")
    print(f"  {'sym':<6}{'n':>4}{'exp/trade':>11}{'total$':>9}{'win%':>7}{'worst$':>9}{'t~':>6}")
    for sym, s in ranked:
        print(f"  {sym:<6}{s['n']:>4}{s['exp']:>11.1f}{s['total']:>9.0f}{s['win']:>7.1f}{s['worst']:>9.0f}{s['t']:>6.1f}")

print("\nUNIVERSE_SCAN_DONE")
