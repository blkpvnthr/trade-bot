#!/usr/bin/env python3
"""vol_arb_backtest.py — IV-vs-RV timed short premium (delta-neutral iron condors).

Tests the vol-arb thesis: sell premium ONLY when implied vol is rich vs realized vol.
SPY, monthly ~45-DTE IRON CONDORS (sell 10Δ strangle, buy 5Δ wings = defined risk,
delta-neutral), real AV chains 2008-2026, hold to expiry.

Per trade records P&L and the IV/RV ratio at entry (ATM implied vol / trailing-20d realized).
Then: (1) bin P&L by IV/RV to see if rich vol predicts profit; (2) compare ALWAYS-sell vs
only-sell-when-IV/RV>threshold (the timing signal) on expectancy / win% / Sharpe / total.
"""
from __future__ import annotations
import sys, math, statistics
from datetime import date
sys.path.insert(0, "/Users/blkpvnthr/Desktop/trader")
import edge_scan_spreads as E
import warnings; warnings.filterwarnings("ignore")
import yfinance as yf

sp = yf.download("SPY", start="2007-09-01", end="2026-06-06", progress=False, auto_adjust=False)
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
def realized_vol(ent, n=20):
    days = [x for x in SD if x <= ent]
    if len(days) < n + 1: return None
    rs = [math.log(S[days[i]] / S[days[i-1]]) for i in range(len(days)-n, len(days))]
    return statistics.pstdev(rs) * math.sqrt(252)
def pick(rows, typ, tgt, exp):
    b = None; bd = 9
    for r in rows:
        if str(r.get("type")).lower() != typ or r.get("expiration", "")[:10] != exp: continue
        if f(r.get("bid")) <= 0: continue
        dl = abs(f(r.get("delta")))
        if dl > 0 and abs(dl - tgt) < bd: b = r; bd = abs(dl - tgt)
    return b
def atm_iv(rows, spot, exp):
    best = None; bd = 1e9
    for r in rows:
        if r.get("expiration", "")[:10] != exp: continue
        iv = f(r.get("implied_volatility"))
        if iv <= 0: continue
        d = abs(f(r.get("strike")) - spot)
        if d < bd: best = iv; bd = d
    return best

START, END = date(2008, 1, 1), date(2026, 4, 1)
entries = []; seen = set()
for t in SD:
    if START <= t <= END and (t.year, t.month) not in seen:
        entries.append(t); seen.add((t.year, t.month))

trades = []   # (ratio, pnl, credit, maxloss)
for ent in entries:
    rows = E.raw_chain("SPY", ent)
    if not rows: continue
    Sen = S[ent]
    exps = {}
    for r in rows:
        e = r.get("expiration", "")[:10]
        if not e: continue
        try: dt = (date.fromisoformat(e) - ent).days
        except: continue
        if 30 <= dt <= 60: exps[e] = abs(dt - 45)
    if not exps: continue
    exp = min(exps, key=exps.get); exp_d = date.fromisoformat(exp); Sx = soa(exp_d)
    if Sx is None: continue
    c10 = pick(rows, "call", 0.10, exp); c5 = pick(rows, "call", 0.05, exp)
    p10 = pick(rows, "put", 0.10, exp); p5 = pick(rows, "put", 0.05, exp)
    if not (c10 and c5 and p10 and p5): continue
    Kc10 = f(c10["strike"]); Kc5 = f(c5["strike"]); Kp10 = f(p10["strike"]); Kp5 = f(p5["strike"])
    if not (Kc5 > Kc10 and Kp5 < Kp10): continue
    credit = (f(c10["bid"]) - f(c5["ask"])) + (f(p10["bid"]) - f(p5["ask"]))   # sell 10Δ, buy 5Δ
    if credit <= 0: continue
    maxloss = max(Kc5 - Kc10, Kp10 - Kp5) - credit
    iv = atm_iv(rows, Sen, exp); rv = realized_vol(ent)
    if not iv or not rv: continue
    ratio = iv / rv
    call_loss = min(max(0.0, Sx - Kc10), Kc5 - Kc10)
    put_loss = min(max(0.0, Kp10 - Sx), Kp10 - Kp5)
    pnl = (credit - call_loss - put_loss) * 100
    trades.append((ent, ratio, pnl, credit, maxloss * 100))

def block(name, ts, yrs):
    if not ts: print(f"  {name:<22} no trades"); return
    pnls = [t[2] for t in ts]; n = len(pnls); w = sum(1 for x in pnls if x > 0) / n
    m = statistics.mean(pnls); sd = statistics.pstdev(pnls) or 1e-9
    sharpe = m / sd * math.sqrt(12)
    avg_risk = statistics.mean([t[4] for t in ts]) or 1e-9
    roc_ann = (sum(pnls) / yrs) / avg_risk * 100   # ann P&L / avg capital-at-risk
    print(f"  {name:<22} n={n:<3} win%={w*100:4.0f}  avgP&L=${m:6.0f}  Sharpe={sharpe:5.2f}  "
          f"worst=${min(pnls):6.0f}  annROC={roc_ann:5.0f}%  total=${sum(pnls):7.0f}")

def analyze(label, ts, yrs):
    print(f"\n{'='*86}\n{label}   (n={len(ts)} months, {yrs:.0f}y)\n{'='*86}")
    print("  -- P&L by IV/RV bin --")
    for lo, hi, lbl in [(0,0.9,"IV<RV <0.9"),(0.9,1.1,"IV≈RV 0.9-1.1"),(1.1,1.4,"rich 1.1-1.4"),(1.4,99,"very rich >1.4")]:
        block(lbl, [t for t in ts if lo <= t[1] < hi], yrs)
    print("  -- threshold sweep (sell only when IV/RV > x) --")
    block("ALWAYS", ts, yrs)
    for thr in (1.1, 1.2, 1.3, 1.4, 1.5):
        block(f">{thr}", [t for t in ts if t[1] > thr], yrs)

ydays = lambda ts: max((ts[-1][0]-ts[0][0]).days/365.25, 1) if ts else 1
print(f"\nVOL-ARB IRON CONDORS — SPY real chains, hold-to-expiry  (total {len(trades)} months)")
analyze("FULL 2008-2026", trades, ydays(trades))
recent = [t for t in trades if t[0].year >= 2015]
analyze("RECENT 2015-2026  (excludes fat-VRP 2009-14 era)", recent, ydays(recent))
print("\nVOLARB_DONE")
