#!/usr/bin/env python3
"""portfolio_combo_volarb.py — trend overlay + vol-arb iron-condor OVERLAY (2015-2026).

Realistic model: stay 100% in the trend shares; overlay defined-risk condors that fire only
when IV/RV>1.4 (they need only max-loss as margin, so they don't displace trend capital).
  combined_monthly = trend_ret + risk_budget * volarb_unit
    trend_ret    : eq-wt(QQQ,SPY,IWM) monthly return when bull (price>200DMA*1.02 & rising), else 0
    volarb_unit  : condor P&L / max-loss in fire months (IV/RV>1.4), else 0   (return on defined risk)
    risk_budget  : fraction of capital risked per condor fire (0 / 2% / 5% / 10%)
Reports correlation + combined CAGR/Sharpe/maxDD/worst-month vs trend-only.
"""
from __future__ import annotations
import os, sys, math, statistics
from datetime import date
os.environ["DAILY_SOURCE"] = "yf"
sys.path.insert(0, "/Users/blkpvnthr/Desktop/trader")
import edge_scan_spreads as E
import warnings; warnings.filterwarnings("ignore")
import yfinance as yf

def f(x, d=0.0):
    try: return float(x)
    except: return d

# --- trend side: QQQ/SPY/IWM closes + QQQ 200DMA regime ---
closes = {s: E.get_closes(s) for s in ["QQQ", "SPY", "IWM"]}
qd = sorted(closes["QQQ"]); qc = closes["QQQ"]
def sma(d, n=200):
    days = [x for x in qd if x <= d]
    return sum(qc[x] for x in days[-n:]) / n if len(days) >= n else None
def rising(d):
    days = [x for x in qd if x <= d]
    if len(days) < 221: return False
    a, b = sma(days[-1]), sma(days[-21]); return a and b and a > b
def bull(d):
    days = [x for x in qd if x <= d]; c = qc[days[-1]]; s = sma(days[-1])
    return bool(s and c > s * 1.02 and rising(d))
def month_ends(days):
    out = {}
    for d in days: out[(d.year, d.month)] = d
    return [out[k] for k in sorted(out)]

# --- vol-arb side: SPY daily + chains ---
sp = yf.download("SPY", start="2013-06-01", end="2026-06-06", progress=False, auto_adjust=False)
if hasattr(sp.columns, "nlevels") and sp.columns.nlevels > 1: sp.columns = sp.columns.droplevel(1)
SPY = {d.date(): float(c) for d, c in zip(sp.index, sp["Close"])}; SD = sorted(SPY)
def soa(d):
    for t in SD:
        if t >= d: return SPY[t]
    return None
def rvol(ent, n=20):
    days = [x for x in SD if x <= ent]
    if len(days) < n + 1: return None
    rs = [math.log(SPY[days[i]] / SPY[days[i-1]]) for i in range(len(days)-n, len(days))]
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
        dd = abs(f(r.get("strike")) - spot)
        if dd < bd: best = iv; bd = dd
    return best
def volarb_unit(ent):
    """Return condor P&L/maxloss if IV/RV>1.4 fires this month, else 0."""
    rows = E.raw_chain("SPY", ent)
    if not rows: return 0.0
    Sen = SPY.get(ent) or soa(ent)
    exps = {}
    for r in rows:
        e = r.get("expiration", "")[:10]
        if not e: continue
        try: dt = (date.fromisoformat(e) - ent).days
        except: continue
        if 30 <= dt <= 60: exps[e] = abs(dt - 45)
    if not exps: return 0.0
    exp = min(exps, key=exps.get); exp_d = date.fromisoformat(exp); Sx = soa(exp_d)
    if Sx is None: return 0.0
    iv = atm_iv(rows, Sen, exp); rv = rvol(ent)
    if not iv or not rv or iv / rv <= 1.4: return 0.0          # only fire when very rich
    c10, c5 = pick(rows, "call", 0.10, exp), pick(rows, "call", 0.05, exp)
    p10, p5 = pick(rows, "put", 0.10, exp), pick(rows, "put", 0.05, exp)
    if not (c10 and c5 and p10 and p5): return 0.0
    Kc10, Kc5, Kp10, Kp5 = f(c10["strike"]), f(c5["strike"]), f(p10["strike"]), f(p5["strike"])
    if not (Kc5 > Kc10 and Kp5 < Kp10): return 0.0
    credit = (f(c10["bid"]) - f(c5["ask"])) + (f(p10["bid"]) - f(p5["ask"]))
    if credit <= 0: return 0.0
    maxloss = max(Kc5 - Kc10, Kp10 - Kp5) - credit
    if maxloss <= 0: return 0.0
    pnl = credit - min(max(0.0, Sx - Kc10), Kc5 - Kc10) - min(max(0.0, Kp10 - Sx), Kp10 - Kp5)
    return pnl / maxloss

START, END = date(2015, 1, 1), date(2026, 4, 1)
me = [d for d in month_ends(qd) if START <= d <= END]
tr, va = [], []
for i in range(1, len(me)):
    d0, d1 = me[i-1], me[i]
    tr.append(statistics.mean([closes[s][d1] / closes[s][d0] - 1 for s in ["QQQ", "SPY", "IWM"]]) if bull(d0) else 0.0)
    va.append(volarb_unit(d0))

def stats(rets):
    m = statistics.mean(rets); sd = statistics.pstdev(rets) or 1e-9
    sh = m / sd * math.sqrt(12); eq = [1.0]
    for r in rets: eq.append(eq[-1] * (1 + r))
    peak = eq[0]; mdd = 0
    for x in eq: peak = max(peak, x); mdd = max(mdd, (peak - x) / peak)
    return (eq[-1] ** (12 / len(rets)) - 1) * 100, sh, mdd * 100, min(rets) * 100

fires = sum(1 for x in va if x != 0)
corr_all = statistics.correlation(tr, va) if len(tr) > 2 else 0
fire_pairs = [(t, v) for t, v in zip(tr, va) if v != 0]
corr_fire = statistics.correlation([t for t, _ in fire_pairs], [v for _, v in fire_pairs]) if len(fire_pairs) > 2 else 0
print(f"\nTREND + VOL-ARB CONDOR OVERLAY  2015-2026  ({len(tr)} months, {fires} condor fires)")
print(f"corr(trend, volarb) all months = {corr_all:+.2f}   |   on fire-months only = {corr_fire:+.2f}\n")
print(f"  {'overlay risk-budget':<22}{'CAGR%':>8}{'Sharpe':>8}{'maxDD%':>8}{'worstMo%':>9}")
for rb in (0.0, 0.02, 0.05, 0.10):
    combo = [tr[i] + rb * va[i] for i in range(len(tr))]
    cg, sh, dd, wm = stats(combo)
    lbl = "trend only" if rb == 0 else f"trend + {int(rb*100)}% condor"
    print(f"  {lbl:<22}{cg:>8.1f}{sh:>8.2f}{dd:>8.1f}{wm:>9.1f}")
print("\nCOMBO_DONE")
