#!/usr/bin/env python3
"""strangle_roc_gap.py — ROC/margin sizing + gap-through stress for short SPY strangles.

PART 1: Reg-T naked-strangle margin per trade -> return-on-capital (is it capital-efficient?).
PART 2: gap stress — assume the 2× stop actually fills at 2/3/4/5× credit in gaps; P&L decay.
Reuses cached AV chains (real, 2008-2026, 10Δ monthly managed 50%TP/2×SL).
"""
from __future__ import annotations
import sys, math, statistics
from datetime import date
sys.path.insert(0, "/Users/blkpvnthr/Desktop/trader")
import edge_scan_spreads as E
import credit_spread_engine as C
_bs_call = C._bs_call; bs_put = C.bs_put
import warnings; warnings.filterwarnings("ignore")
import yfinance as yf
_spx = yf.download("SPY", start="2007-12-01", end="2026-06-06", progress=False, auto_adjust=False)
if hasattr(_spx.columns,"nlevels") and _spx.columns.nlevels>1: _spx.columns=_spx.columns.droplevel(1)
S={d.date():float(c) for d,c in zip(_spx.index,_spx["Close"])}; SDAYS=sorted(S)
def f(x,dv=0.0):
    try: return float(x)
    except: return dv
def soa(d):
    for t in SDAYS:
        if t>=d: return S[t]
    return None
def pick(rows,typ,target,exp):
    best=None;bd=9
    for r in rows:
        if str(r.get("type")).lower()!=typ or r.get("expiration","")[:10]!=exp: continue
        if f(r.get("bid"))<=0: continue
        dl=abs(f(r.get("delta")));
        if dl>0 and abs(dl-target)<bd: best=r;bd=abs(dl-target)
    return best
def regt_margin(U,Kc,Kp,cmid,pmid):
    creq=100*max(0.20*U-max(Kc-U,0)+cmid, 0.10*U+cmid)
    preq=100*max(0.20*U-max(U-Kp,0)+pmid, 0.10*U+pmid)
    return max(creq,preq)+min(cmid,pmid)*100   # greater side + other-side premium

START,END=date(2008,1,1),date(2026,4,1)
entries=[]; seen=set()
for t in SDAYS:
    if START<=t<=END and (t.year,t.month) not in seen: entries.append(t); seen.add((t.year,t.month))

trades=[]  # per trade: (date, margin, pnl_managed, stopped, credit, stop_buyback)
for ent in entries:
    rows=E.raw_chain("SPY",ent)
    if not rows: continue
    exps={}
    for r in rows:
        e=r.get("expiration","")[:10]
        if not e: continue
        try: dt=(date.fromisoformat(e)-ent).days
        except: continue
        if 30<=dt<=60: exps[e]=abs(dt-45)
    if not exps: continue
    exp=min(exps,key=exps.get); exp_d=date.fromisoformat(exp); Sexp=soa(exp_d)
    if Sexp is None: continue
    c=pick(rows,"call",0.10,exp); p=pick(rows,"put",0.10,exp)
    if not(c and p): continue
    Kc=f(c.get("strike"));Kp=f(p.get("strike"));U=S[ent]
    cb=f(c.get("bid"));pb=f(p.get("bid"));credit=cb+pb
    if credit<=0: continue
    ivc=f(c.get("implied_volatility")) or .2; ivp=f(p.get("implied_volatility")) or .2
    margin=regt_margin(U,Kc,Kp,cb,pb)
    # managed sim (daily BS): TP 50% / SL 2x ; record if stopped + the buyback value
    days=[t for t in SDAYS if ent<=t<=exp_d]; pnl=None; stopped=False; stopval=None
    for t in days:
        Sx=S[t];dte=max((exp_d-t).days,0);T=dte/365.0
        val=_bs_call(Sx,Kc,max(T,1e-6),.043,max(ivc,1e-4))["price"]+bs_put(Sx,Kp,T,.043,max(ivp,1e-4))
        if val<=0.5*credit: pnl=(credit-val)*100; break
        if val-credit>=2*credit: pnl=(credit-val)*100; stopped=True; stopval=val; break
    if pnl is None: pnl=(credit-(max(0,Sexp-Kc)+max(0,Kp-Sexp)))*100
    trades.append(dict(d=ent,margin=margin,pnl=pnl,stopped=stopped,credit=credit,stopval=stopval))

# ---- PART 1: ROC ----
mar=[t["margin"] for t in trades]; pnls=[t["pnl"] for t in trades]
avg_mar=sum(mar)/len(mar); tot=sum(pnls); n=len(trades)
yrs=(entries[-1]-entries[0]).days/365.25
roc_per=[t["pnl"]/t["margin"] for t in trades]  # per-trade ROC
mean_r=statistics.mean(roc_per); sd_r=statistics.pstdev(roc_per)
sharpe=mean_r/sd_r*math.sqrt(12) if sd_r>0 else 0   # ~12 trades/yr
ann_roc=mean_r*12*100
print("="*70); print("PART 1 — ROC / MARGIN (10Δ managed strangle, Reg-T naked margin)"); print("="*70)
print(f"  trades={n}  avg margin/trade=${avg_mar:,.0f}  (SPY-scaled: ~${mar[0]:,.0f} in '08 -> ~${mar[-1]:,.0f} in '26)")
print(f"  total P&L=${tot:,.0f}  ann ROC (on margin)={ann_roc:.1f}%/yr  Sharpe(on-margin)={sharpe:.2f}")
print(f"  -> capital efficiency: {'GOOD' if ann_roc>15 else 'modest' if ann_roc>8 else 'poor'} "
      f"(vs trend-overlay ~11%/yr at 12% DD, Sharpe 0.61)")

# ---- PART 2: gap stress ----
print("\n"+"="*70); print("PART 2 — GAP-THROUGH STRESS (stop fills at K× credit instead of 2×)"); print("="*70)
nstop=sum(1 for t in trades if t["stopped"])
print(f"  stopped trades: {nstop}/{n}")
print(f"  {'gap mult':>9}{'total P&L':>12}{'worst trade':>13}{'vs base %':>10}")
base=None
for gm in (2,3,4,5):
    adj=[]
    for t in trades:
        if t["stopped"]:
            adj.append((t["credit"]-gm*t["credit"])*100)   # stop fills at gm× credit
        else:
            adj.append(t["pnl"])
    tt=sum(adj); wr=min(adj)
    if base is None: base=tt
    print(f"  {gm:>8}×{tt:>12,.0f}{wr:>13,.0f}{100*(tt-base)/abs(base) if base else 0:>9.0f}%")
print("\nSTRANGLE_RG_DONE")
