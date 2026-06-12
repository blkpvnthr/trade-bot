#!/usr/bin/env python3
"""options_overlay_backtest.py — which options structure (if any) IMPROVES the trend long?

Bull-regime trend long on SPY (price>200DMA*1.02 & rising), monthly ~30 DTE, real AV chains.
Compares three expressions of the SAME long signal on equal share-notional:
  SHARES        : own SPY                                  (baseline)
  DEEP-ITM CALL : buy ~0.80Δ call (stock replacement)     -> leverage + defined downside, low theta
  COVERED CALL  : own SPY + sell ~0.30Δ call (buy-write)   -> VRP income, capped upside
Cash when not bull (all three). Reports CAGR / Sharpe / maxDD / worst-month.
"""
from __future__ import annotations
import sys, math, statistics
from datetime import date
sys.path.insert(0, "/Users/blkpvnthr/Desktop/trader")
import edge_scan_spreads as E
import credit_spread_engine as C
import warnings; warnings.filterwarnings("ignore")
import yfinance as yf

sp = yf.download("SPY", start="2013-06-01", end="2026-06-06", progress=False, auto_adjust=False)
if hasattr(sp.columns, "nlevels") and sp.columns.nlevels > 1: sp.columns = sp.columns.droplevel(1)
S = {d.date(): float(c) for d, c in zip(sp.index, sp["Close"])}
SD = sorted(S)

def f(x, d=0.0):
    try: return float(x)
    except: return d
def soa(d):
    for t in SD:
        if t >= d: return S[t]
    return None
def sma(d, n=200):
    days = [x for x in SD if x <= d]
    if len(days) < n: return None
    return sum(S[x] for x in days[-n:]) / n
def rising(d):
    days = [x for x in SD if x <= d]
    if len(days) < 221: return False
    a = sma(days[-1]); b = sma(days[-21])
    return a is not None and b is not None and a > b
def bull(d):
    c = soa(d); s = sma(d)
    return bool(s and c > s * 1.02 and rising(d))
def pick_delta(rows, target, exp):
    best = None; bd = 9
    for r in rows:
        if str(r.get("type")).lower() != "call" or r.get("expiration", "")[:10] != exp: continue
        if f(r.get("ask")) <= 0: continue
        dl = abs(f(r.get("delta")))
        if dl > 0 and abs(dl - target) < bd: best = r; bd = abs(dl - target)
    return best

START, END = date(2015, 1, 1), date(2026, 4, 1)
entries = []; seen = set()
for t in SD:
    if START <= t <= END and (t.year, t.month) not in seen:
        entries.append(t); seen.add((t.year, t.month))

R = {"shares": [], "deep_itm_call": [], "covered_call": []}
for ent in entries:
    if not bull(ent):
        for k in R: R[k].append(0.0)            # cash when not bull
        continue
    rows = E.raw_chain("SPY", ent)
    Sen = S[ent]
    # pick ~30 DTE expiry
    exps = {}
    for r in rows:
        e = r.get("expiration", "")[:10]
        if not e: continue
        try: dt = (date.fromisoformat(e) - ent).days
        except: continue
        if 20 <= dt <= 45: exps[e] = abs(dt - 30)
    if not rows or not exps:
        for k in R: R[k].append(0.0)
        continue
    exp = min(exps, key=exps.get); exp_d = date.fromisoformat(exp); Sx = soa(exp_d)
    sh = Sx / Sen - 1
    R["shares"].append(sh)
    # deep-ITM call (~0.80Δ): pay ask; payoff = intrinsic at expiry; return on share notional
    di = pick_delta(rows, 0.80, exp)
    if di:
        K = f(di["strike"]); prem = f(di["ask"])
        R["deep_itm_call"].append((max(0.0, Sx - K) - prem) / Sen)
    else:
        R["deep_itm_call"].append(sh)
    # covered call (~0.30Δ): own shares + short call (sell bid); capped at K + premium
    cc = pick_delta(rows, 0.30, exp)
    if cc:
        K = f(cc["strike"]); prem = f(cc["bid"])
        R["covered_call"].append((min(Sx, K) - Sen + prem) / Sen)
    else:
        R["covered_call"].append(sh)

def stats(rets):
    m = statistics.mean(rets); sd = statistics.pstdev(rets) or 1e-9
    sharpe = m / sd * math.sqrt(12)
    eq = [1.0]
    for r in rets: eq.append(eq[-1] * (1 + r))
    peak = eq[0]; mdd = 0
    for x in eq: peak = max(peak, x); mdd = max(mdd, (peak - x) / peak)
    cagr = eq[-1] ** (12 / len(rets)) - 1
    return cagr * 100, sharpe, mdd * 100, min(rets) * 100, (eq[-1] - 1) * 100

print(f"OPTIONS OVERLAY ON TREND LONG — SPY, real chains, {START}..{END} ({len(entries)} months)\n")
print(f"  {'structure':<16}{'CAGR%':>8}{'Sharpe':>8}{'maxDD%':>8}{'worstMo%':>9}{'totalRet%':>10}")
for k in ["shares", "deep_itm_call", "covered_call"]:
    cg, sh, dd, wm, tot = stats(R[k])
    print(f"  {k:<16}{cg:>8.1f}{sh:>8.2f}{dd:>8.1f}{wm:>9.1f}{tot:>10.0f}")
print("\nOVERLAY_DONE")
