#!/usr/bin/env python3
"""portfolio_combo.py — trend overlay + small short-strangle sleeve (Part 3).

Monthly return streams, 2015-2026:
  TREND  : shares-only trend overlay proxy = equal-wt (QQQ,SPY,IWM) monthly return when
           bull (QQQ>200DMA+2% AND 200DMA rising), else cash (0).
  STRANGLE: 10Δ managed SPY strangle monthly ROC (pnl/Reg-T margin), real chains.
Blend at strangle weights 0/5/10/15/20%; report correlation + combined CAGR/Sharpe/maxDD/worst.
"""
from __future__ import annotations
import os, sys, math, statistics
from datetime import date
os.environ["DAILY_SOURCE"]="yf"
sys.path.insert(0,"/Users/blkpvnthr/Desktop/trader")
import edge_scan_spreads as E
import credit_spread_engine as C
_bs_call=C._bs_call; bs_put=C.bs_put
import warnings; warnings.filterwarnings("ignore")

def f(x,dv=0.0):
    try: return float(x)
    except: return dv

START,END=date(2015,1,1),date(2026,4,1)
# --- monthly close series for index trio + regime (QQQ 200DMA rising) ---
closes={s:E.get_closes(s) for s in ["QQQ","SPY","IWM"]}
qd=sorted(closes["QQQ"]); qc=closes["QQQ"]
from collections import deque
win=deque(maxlen=200); sma={}
for d in qd:
    win.append(qc[d]); sma[d]=sum(win)/len(win) if len(win)==200 else None
def rising200(d):
    days=[x for x in qd if x<=d]
    if len(days)<221: return False
    return sma[days[-1]] is not None and sma[days[-21]] is not None and sma[days[-1]]>sma[days[-21]]
def bull(d):
    days=[x for x in qd if x<=d]
    c=qc[days[-1]]; s=sma[days[-1]]
    return bool(s and c>s*1.02 and rising200(d))
# month-end trading days
def month_ends(days):
    out={}
    for d in days: out[(d.year,d.month)]=d
    return [out[k] for k in sorted(out)]
me=[d for d in month_ends(qd) if START<=d<=END]

# --- strangle monthly ROC (reuse cached chains) ---
import yfinance as yf
sp=yf.download("SPY",start="2007-12-01",end="2026-06-06",progress=False,auto_adjust=False)
if hasattr(sp.columns,"nlevels") and sp.columns.nlevels>1: sp.columns=sp.columns.droplevel(1)
SPY={d.date():float(c) for d,c in zip(sp.index,sp["Close"])}; SD=sorted(SPY)
def soa(d):
    for t in SD:
        if t>=d: return SPY[t]
    return None
def pick(rows,typ,tg,exp):
    b=None;bd=9
    for r in rows:
        if str(r.get("type")).lower()!=typ or r.get("expiration","")[:10]!=exp: continue
        if f(r.get("bid"))<=0: continue
        dl=abs(f(r.get("delta")))
        if dl>0 and abs(dl-tg)<bd: b=r;bd=abs(dl-tg)
    return b
def regt(U,Kc,Kp,cm,pm):
    return max(100*max(0.20*U-max(Kc-U,0)+cm,0.10*U+cm),100*max(0.20*U-max(U-Kp,0)+pm,0.10*U+pm))+min(cm,pm)*100
def strangle_month_roc(ent):
    rows=E.raw_chain("SPY",ent)
    if not rows: return None
    exps={}
    for r in rows:
        e=r.get("expiration","")[:10]
        if not e: continue
        try: dt=(date.fromisoformat(e)-ent).days
        except: continue
        if 30<=dt<=60: exps[e]=abs(dt-45)
    if not exps: return None
    exp=min(exps,key=exps.get); exp_d=date.fromisoformat(exp); Sx=soa(exp_d)
    if Sx is None: return None
    c=pick(rows,"call",0.10,exp); p=pick(rows,"put",0.10,exp)
    if not(c and p): return None
    Kc=f(c.get("strike"));Kp=f(p.get("strike"));U=SPY.get(ent) or soa(ent)
    cb=f(c.get("bid"));pb=f(p.get("bid"));credit=cb+pb
    if credit<=0: return None
    ivc=f(c.get("implied_volatility")) or .2; ivp=f(p.get("implied_volatility")) or .2
    m=regt(U,Kc,Kp,cb,pb); pnl=None
    for t in [x for x in SD if ent<=x<=exp_d]:
        S0=SPY[t];dte=max((exp_d-t).days,0);T=dte/365.0
        v=_bs_call(S0,Kc,max(T,1e-6),.043,max(ivc,1e-4))["price"]+bs_put(S0,Kp,T,.043,max(ivp,1e-4))
        if v<=0.5*credit: pnl=(credit-v)*100;break
        if v-credit>=2*credit: pnl=(credit-v)*100;break
    if pnl is None: pnl=(credit-(max(0,Sx-Kc)+max(0,Kp-Sx)))*100
    return pnl/m

# --- build aligned monthly return streams ---
tr=[]; st=[]; months=[]
for i in range(1,len(me)):
    d0=me[i-1]; d1=me[i]
    # trend monthly return: long eq-wt trio if bull at d0, else 0
    if bull(d0):
        r=statistics.mean([closes[s][d1]/closes[s][d0]-1 for s in ["QQQ","SPY","IWM"]])
    else: r=0.0
    sr=strangle_month_roc(d0)
    if sr is None: continue
    tr.append(r); st.append(sr); months.append(d1)

def stats(rets):
    mean=statistics.mean(rets); sd=statistics.pstdev(rets) or 1e-9
    sharpe=mean/sd*math.sqrt(12)
    eq=[1.0]
    for r in rets: eq.append(eq[-1]*(1+r))
    peak=eq[0];mdd=0
    for x in eq: peak=max(peak,x);mdd=max(mdd,(peak-x)/peak)
    cagr=eq[-1]**(12/len(rets))-1
    return cagr*100,sharpe,mdd*100,min(rets)*100
corr=statistics.correlation(tr,st) if len(tr)>2 else 0
print("="*74); print(f"PORTFOLIO COMBO  2015-2026  ({len(tr)} months)  corr(trend,strangle)={corr:+.2f}"); print("="*74)
print(f"  {'allocation':<22}{'CAGR%':>8}{'Sharpe':>8}{'maxDD%':>8}{'worstMo%':>9}")
for ws in (0.0,0.05,0.10,0.15,0.20,1.0):
    blend=[(1-ws)*tr[i]+ws*st[i] for i in range(len(tr))]
    cg,sh,dd,wm=stats(blend)
    lbl=f"{int((1-ws)*100)}% trend/{int(ws*100)}% strangle" if ws<1 else "100% strangle"
    print(f"  {lbl:<22}{cg:>8.1f}{sh:>8.2f}{dd:>8.1f}{wm:>9.1f}")
print("\nPORTFOLIO_DONE")
