#!/usr/bin/env python3
"""multiasset_sweep.py — param-robustness + borrow-cost sweep for the multi-asset vol-target trend.

Holds the strategy fixed, perturbs each parameter, and adds realistic borrow cost (5.5%/yr on
the levered portion, which the first run omitted). Robust IFF Sharpe stays ~1.0 across the grid.
"""
from __future__ import annotations
import math, statistics, bisect
from datetime import date
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import yfinance as yf

EQUITY = ["SPY", "QQQ", "IWM"]; DIVERSIFIERS = ["TLT", "GLD", "DBC", "UUP"]; ALL = EQUITY + DIVERSIFIERS
START, END = date(2008, 1, 1), date(2026, 6, 1)
COST_BPS = 5; BORROW = 0.055

raw = yf.download(ALL, start="2006-01-01", end="2026-06-06", progress=False, auto_adjust=True)["Close"].dropna(how="all")
px = {s: {d.date(): float(v) for d, v in raw[s].dropna().items()} for s in ALL}
DAYS = {s: sorted(px[s]) for s in ALL}

# precompute (param-independent): SMA200, rising, daily log returns
SMA = {}; RISE = {}; LR = {}
for s in ALL:
    ds = DAYS[s]; a = [px[s][d] for d in ds]; sm = {}; ri = {}
    for i, d in enumerate(ds):
        sm[d] = sum(a[i-199:i+1]) / 200 if i >= 199 else None
    for i, d in enumerate(ds):
        ri[d] = bool(i >= 220 and sm[d] and sm[ds[i-21]] and sm[d] > sm[ds[i-21]])
    SMA[s] = sm; RISE[s] = ri
    LR[s] = [math.log(a[i] / a[i-1]) for i in range(1, len(ds))]  # aligned to ds[1:]

def last_idx(s, d):
    i = bisect.bisect_right(DAYS[s], d) - 1
    return i if i >= 0 else None
def signal(s, d, band):
    i = last_idx(s, d)
    if i is None: return False
    dd = DAYS[s][i]; sm = SMA[s][dd]
    return bool(sm and px[s][dd] > sm * band and RISE[s][dd])
def vol(s, d, w):
    i = last_idx(s, d)
    if i is None or i < w: return None
    seg = LR[s][i-w:i]  # last w returns ending at dd
    return float(np.std(seg) * math.sqrt(252)) if len(seg) == w else None
def lret_seg(s, d, w):
    i = last_idx(s, d)
    return LR[s][i-w:i] if (i is not None and i >= w) else None
def mret(s, d0, d1):
    i0, i1 = last_idx(s, d0), last_idx(s, d1)
    if i0 is None or i1 is None: return 0.0
    return px[s][DAYS[s][i1]] / px[s][DAYS[s][i0]] - 1

months = []; seen = set()
for d in sorted(set().union(*[set(px[s]) for s in ALL])):
    if START <= d <= END and (d.year, d.month) not in seen:
        months.append(d); seen.add((d.year, d.month))

def run(target_vol=0.12, max_lev=1.5, w=60, band=1.02, assets=ALL, borrow=BORROW):
    eq = 1.0; rets = []; vals = [1.0]; prev = {}
    for i in range(1, len(months)):
        d0, d1 = months[i-1], months[i]
        act = [s for s in assets if signal(s, d0, band) and vol(s, d0, w)]
        if act:
            iv = {s: 1.0 / vol(s, d0, w) for s in act}; tot = sum(iv.values())
            wraw = {s: iv[s] / tot for s in act}
            if len(act) > 1:
                ml = min(len(lret_seg(s, d0, w)) for s in act)
                R = np.array([lret_seg(s, d0, w)[-ml:] for s in act]); cov = np.cov(R) * 252
                wv = np.array([wraw[s] for s in act]); pv = math.sqrt(max(wv @ cov @ wv, 1e-9))
            else:
                pv = vol(act[0], d0, w)
            lev = min(target_vol / pv, max_lev) if pv > 0 else 1.0
            wt = {s: wraw[s] * lev for s in act}
        else:
            wt = {}
        port = sum(wt.get(s, 0) * mret(s, d0, d1) for s in wt)
        turn = sum(abs(wt.get(s, 0) - prev.get(s, 0)) for s in set(wt) | set(prev))
        borrow_cost = max(0.0, sum(wt.values()) - 1.0) * borrow / 12
        net = port - turn * COST_BPS / 10000 - borrow_cost
        eq *= (1 + net); rets.append(net); vals.append(eq); prev = wt
    m = statistics.mean(rets); sd = statistics.pstdev(rets) or 1e-9
    sh = m / sd * math.sqrt(12)
    peak = vals[0]; mdd = 0
    for v in vals: peak = max(peak, v); mdd = max(mdd, (peak - v) / peak)
    cagr = vals[-1] ** (12 / len(rets)) - 1
    return cagr * 100, sh, mdd * 100, (cagr / (mdd or 1e-9)) if mdd else 0

def show(label, **kw):
    cg, sh, dd, cal = run(**kw)
    print(f"  {label:<30}{cg:>7.1f}{sh:>8.2f}{dd:>8.1f}{cal:>8.2f}")

print(f"ROBUSTNESS SWEEP (multi-asset vol-target trend, 2008-2026, WITH {BORROW*100:.1f}% borrow + {COST_BPS}bps cost)\n")
print(f"  {'config':<30}{'CAGR%':>7}{'Sharpe':>8}{'maxDD%':>8}{'Calmar':>8}")
print("  " + "-" * 60)
show("BASE (tv12 lev1.5 w60 b1.02)")
print("  -- target-vol x leverage-cap grid --")
for tv in (0.10, 0.12, 0.15):
    for ml in (1.25, 1.5, 2.0):
        show(f"tv{int(tv*100)} lev{ml}", target_vol=tv, max_lev=ml)
print("  -- vol-window --")
for w in (40, 60, 90): show(f"window {w}d", w=w)
print("  -- trend band --")
for b in (1.00, 1.02, 1.05): show(f"band {b}", band=b)
print("  -- drop-one asset (robustness to single asset) --")
for s in DIVERSIFIERS: show(f"drop {s}", assets=[a for a in ALL if a != s])
show("equity-only (no diversifiers)", assets=EQUITY)
print("\nSWEEP_DONE")
