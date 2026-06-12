#!/usr/bin/env python3
"""strangle_kelly_backtest.py — short SPX/SPY strangles on REAL chains, 2008-2026.

Tests the tastytrade/Kelly short-premium framework against the crashes the book's
bull-only 5-yr study skipped (2008 GFC, 2020 COVID, 2022). Monthly ~45-DTE strangles,
real AV chains (sell at bid = conservative). Two deltas (10Δ, 5Δ) × two managements:
  • UNMANAGED: hold to expiry, settle at intrinsic (book's "worst case")
  • MANAGED  : take-profit 50% of credit / stop 2× credit loss, BS-marked daily
Reports per-variant: win% (W), avg win/loss, realized R=avgW/avgL, Kelly break-even R
for that W, worst trade, total P&L, and verdict (positive expectancy iff R > break-even-R).
"""
from __future__ import annotations
import sys, math, statistics
from datetime import date, datetime, timedelta
sys.path.insert(0, "/Users/blkpvnthr/Desktop/trader")
import edge_scan_spreads as E
import credit_spread_engine as C
_bs_call = C._bs_call; bs_put = C.bs_put

import warnings; warnings.filterwarnings("ignore")
import yfinance as yf
_spx = yf.download("SPY", start="2007-12-01", end="2026-06-06", progress=False, auto_adjust=False)
if hasattr(_spx.columns, "nlevels") and _spx.columns.nlevels > 1: _spx.columns = _spx.columns.droplevel(1)
S = {d.date(): float(c) for d, c in zip(_spx.index, _spx["Close"])}
SDAYS = sorted(S)
def spx_on_or_after(d):
    for t in SDAYS:
        if t >= d: return S[t], t
    return None, None

def f(x, dv=0.0):
    try: return float(x)
    except: return dv

# Kelly break-even R for a win prob W:  R_be = (1-W)/W
def kelly_be(W): return (1-W)/W if W>0 else 9.99

def pick_delta(rows, typ, target, exp):
    best=None; bd=9
    for r in rows:
        if str(r.get("type")).lower()!=typ or r.get("expiration","")[:10]!=exp: continue
        if f(r.get("bid"))<=0: continue
        dl=abs(f(r.get("delta")))
        if dl<=0: continue
        if abs(dl-target)<bd: best=r; bd=abs(dl-target)
    return best

START, END = date(2008,1,1), date(2026,4,1)
# monthly entry = first trading day of each month
entries=[]
d=START
seen=set()
for t in SDAYS:
    if START<=t<=END and (t.year,t.month) not in seen:
        entries.append(t); seen.add((t.year,t.month))

# variants accumulate per-trade pnl (per 1 contract = x100)
variants={(dlt,mode):[] for dlt in (0.10,0.05) for mode in ("unmanaged","managed")}
worst={(dlt,mode):0 for dlt in (0.10,0.05) for mode in ("unmanaged","managed")}
used=0

def manage(Kc,ivc,Kp,ivp,credit,ent,exp):
    """BS daily mark; TP 50% credit / SL 2x credit / expiry. Returns pnl per share."""
    days=[t for t in SDAYS if ent<=t<=exp]
    for t in days:
        Sx=S[t]; dte=max((exp-t).days,0); T=dte/365.0
        val=_bs_call(Sx,Kc,max(T,1e-6),0.043,max(ivc,1e-4))["price"]+bs_put(Sx,Kp,T,0.043,max(ivp,1e-4))
        if val<=0.5*credit: return credit-val           # take profit
        if val-credit>=2*credit: return credit-val       # stop (loss ~2x)
    Sx,_=spx_on_or_after(exp)
    return credit-(max(0,Sx-Kc)+max(0,Kp-Sx))            # expiry settle

for ent in entries:
    rows=E.raw_chain("SPY", ent)
    if not rows: continue
    # expiry closest to 45 DTE
    exps={}
    for r in rows:
        e=r.get("expiration","")[:10]
        if not e: continue
        try: dt=(date.fromisoformat(e)-ent).days
        except: continue
        if 30<=dt<=60: exps[e]=abs(dt-45)
    if not exps: continue
    exp=min(exps,key=exps.get); exp_d=date.fromisoformat(exp)
    Sexp,_=spx_on_or_after(exp_d)
    if Sexp is None: continue
    used+=1
    for dlt in (0.10,0.05):
        c=pick_delta(rows,"call",dlt,exp); p=pick_delta(rows,"put",dlt,exp)
        if not (c and p): continue
        Kc=f(c.get("strike")); Kp=f(p.get("strike"))
        credit=f(c.get("bid"))+f(p.get("bid"))
        if credit<=0: continue
        ivc=f(c.get("implied_volatility")) or .2; ivp=f(p.get("implied_volatility")) or .2
        # unmanaged
        pnl_u=(credit-(max(0,Sexp-Kc)+max(0,Kp-Sexp)))*100
        variants[(dlt,"unmanaged")].append(pnl_u); worst[(dlt,"unmanaged")]=min(worst[(dlt,"unmanaged")],pnl_u)
        # managed
        pnl_m=manage(Kc,ivc,Kp,ivp,credit,ent,exp_d)*100
        variants[(dlt,"managed")].append(pnl_m); worst[(dlt,"managed")]=min(worst[(dlt,"managed")],pnl_m)

def report(name,pnls,wrst):
    if not pnls: print(f"  {name}: no trades"); return
    n=len(pnls); tot=sum(pnls)
    wins=[x for x in pnls if x>0]; losses=[x for x in pnls if x<=0]
    W=len(wins)/n
    aw=sum(wins)/len(wins) if wins else 0; al=abs(sum(losses)/len(losses)) if losses else 1e-9
    R=aw/al if al>0 else 9.99; be=kelly_be(W)
    verdict="POSITIVE-exp (R>BE)" if R>be else "NEGATIVE-exp (R<BE)"
    print(f"  {name:<22} n={n} W={W*100:.0f}% avgW=${aw:.0f} avgL=$-{al:.0f} R={R:.2f} (BE={be:.2f}) "
          f"worst=${wrst:.0f} total=${tot:.0f}  -> {verdict}")

print(f"\nSHORT SPY STRANGLES on REAL chains  {START}..{END}  (monthly ~45 DTE, sell@bid)")
print(f"entries used: {used}\n")
for dlt in (0.10,0.05):
    print(f"--- {int(dlt*100)}-delta strangle ---")
    report(f"{int(dlt*100)}Δ unmanaged", variants[(dlt,'unmanaged')], worst[(dlt,'unmanaged')])
    report(f"{int(dlt*100)}Δ managed(50/2x)", variants[(dlt,'managed')], worst[(dlt,'managed')])
# crash-month spotlight (10Δ unmanaged)
print("\nWorst 6 individual strangles (10Δ unmanaged) — the tail the bull-only study hid:")
pe=sorted(zip(entries[:len(variants[(0.10,'unmanaged')])], variants[(0.10,'unmanaged')]), key=lambda x:x[1])[:6]
for d,pnl in pe: print(f"   {d}  ${pnl:.0f}")
print("STRANGLE_DONE")
