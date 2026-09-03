#!/usr/bin/env python3
"""volarb_backtest_full.py — vol-arb iron-condor backtest with TRADE LOG + EQUITY CURVE.

Strategy (validated): SPY monthly ~45-DTE iron condor (sell 10Δ strangle, buy 5Δ wings),
fire ONLY when ATM IV / trailing-20d RV > 1.4, hold to expiry, real AV chains.
Risk-parity sizing: each fire risks `risk_pct` of CURRENT equity (contracts = risk$/max-loss).

Writes:
  logs/backtest/volarb_trades.csv   (one row per trade)
  logs/backtest/volarb_equity.csv   (month-end equity curve)
Prints full performance summary (both on total capital and on risked capital).
"""
from __future__ import annotations
import sys, math, csv, statistics
from datetime import date
from pathlib import Path
sys.path.insert(0, "/Users/blkpvnthr/Desktop/trader")
import edge_scan_spreads as E
import warnings; warnings.filterwarnings("ignore")
import yfinance as yf

CAPITAL0 = 100_000.0
RISK_PCT = 0.05          # risk 5% of equity per condor fire (defined risk)
THRESH = 1.4
OUT = Path("/Users/blkpvnthr/Desktop/trader/logs/backtest"); OUT.mkdir(parents=True, exist_ok=True)

sp = yf.download("SPY", start="2007-09-01", end="2026-06-06", progress=False, auto_adjust=False)
if hasattr(sp.columns, "nlevels") and sp.columns.nlevels > 1: sp.columns = sp.columns.droplevel(1)
S = {d.date(): float(c) for d, c in zip(sp.index, sp["Close"])}; SD = sorted(S)

def f(x, d=0.0):
    try: return float(x)
    except: return d
def soa(d):
    for t in SD:
        if t >= d: return S[t]
    return None
def rvol(ent, n=20):
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
        dd = abs(f(r.get("strike")) - spot)
        if dd < bd: best = iv; bd = dd
    return best

START, END = date(2008, 1, 1), date(2026, 4, 1)
months = []; seen = set()
for t in SD:
    if START <= t <= END and (t.year, t.month) not in seen:
        months.append(t); seen.add((t.year, t.month))

equity = CAPITAL0
trades = []; curve = []
for ent in months:
    fired = False
    rows = E.raw_chain("SPY", ent)
    if rows:
        Sen = S[ent]
        exps = {}
        for r in rows:
            e = r.get("expiration", "")[:10]
            if not e: continue
            try: dt = (date.fromisoformat(e) - ent).days
            except: continue
            if 30 <= dt <= 60: exps[e] = abs(dt - 45)
        if exps:
            exp = min(exps, key=exps.get); exp_d = date.fromisoformat(exp); Sx = soa(exp_d)
            iv = atm_iv(rows, Sen, exp); rv = rvol(ent)
            if Sx and iv and rv and iv / rv > THRESH:
                sc, lc = pick(rows, "call", 0.10, exp), pick(rows, "call", 0.05, exp)
                spu, lpu = pick(rows, "put", 0.10, exp), pick(rows, "put", 0.05, exp)
                if sc and lc and spu and lpu:
                    Kc_s, Kc_l = f(sc["strike"]), f(lc["strike"]); Kp_s, Kp_l = f(spu["strike"]), f(lpu["strike"])
                    if Kc_l > Kc_s and Kp_l < Kp_s:
                        credit = (f(sc["bid"]) - f(lc["ask"])) + (f(spu["bid"]) - f(lpu["ask"]))
                        maxloss = max(Kc_l - Kc_s, Kp_s - Kp_l) - credit
                        if credit > 0 and maxloss > 0:
                            per_contract_risk = maxloss * 100
                            contracts = max(1, int((RISK_PCT * equity) / per_contract_risk))
                            pnl_per = (credit - min(max(0.0, Sx - Kc_s), Kc_l - Kc_s)
                                       - min(max(0.0, Kp_s - Sx), Kp_s - Kp_l)) * 100
                            pnl = pnl_per * contracts
                            equity += pnl
                            status = "win" if pnl > 0 else "loss"
                            trades.append(dict(entry=ent.isoformat(), expiry=exp, dte=(exp_d-ent).days,
                                iv=round(iv,4), rv=round(rv,4), iv_rv=round(iv/rv,3),
                                Kp_long=Kp_l, Kp_short=Kp_s, Kc_short=Kc_s, Kc_long=Kc_l,
                                credit=round(credit,2), maxloss_per=round(maxloss,2), contracts=contracts,
                                settle_px=round(Sx,2), status=status, pnl=round(pnl,2), equity=round(equity,2)))
                            fired = True
    curve.append(dict(month=ent.isoformat(), equity=round(equity, 2), fired=int(fired)))

# write logs
with open(OUT/"volarb_trades.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(trades[0].keys())); w.writeheader(); w.writerows(trades)
with open(OUT/"volarb_equity.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=["month","equity","fired"]); w.writeheader(); w.writerows(curve)

# summary
pnls=[t["pnl"] for t in trades]; n=len(pnls); wins=[p for p in pnls if p>0]
W=len(wins)/n if n else 0
yrs=(months[-1]-months[0]).days/365.25
cagr=(equity/CAPITAL0)**(1/yrs)-1
mret=[curve[i]["equity"]/curve[i-1]["equity"]-1 for i in range(1,len(curve))]
sharpe=statistics.mean(mret)/(statistics.pstdev(mret) or 1e-9)*math.sqrt(12)
peak=CAPITAL0; mdd=0
for c in curve: peak=max(peak,c["equity"]); mdd=max(mdd,(peak-c["equity"])/peak)
roc_per=[t["pnl"]/(t["maxloss_per"]*t["contracts"]) for t in trades]
print(f"\nVOL-ARB CONDOR BACKTEST  SPY 2008-2026  (start ${CAPITAL0:,.0f}, risk {RISK_PCT*100:.0f}%/fire, IV/RV>{THRESH})")
print(f"  trades(fires) = {n} over {len(months)} months ({n/yrs:.1f}/yr)   win% = {W*100:.0f}")
print(f"  final equity  = ${equity:,.0f}   total P&L = ${equity-CAPITAL0:,.0f}")
print(f"  CAGR(on full capital) = {cagr*100:.1f}%/yr   Sharpe = {sharpe:.2f}   maxDD = {mdd*100:.1f}%")
print(f"  avg return on RISKED capital/trade = {statistics.mean(roc_per)*100:.1f}%   (best ${max(pnls):,.0f} / worst ${min(pnls):,.0f})")
print(f"  logs -> logs/backtest/volarb_trades.csv ({n} rows) + volarb_equity.csv ({len(curve)} rows)")
print("\n  last 5 trades:")
for t in trades[-5:]:
    print(f"    {t['entry']}  IV/RV {t['iv_rv']}  {t['Kp_long']:.0f}/{t['Kp_short']:.0f}p {t['Kc_short']:.0f}/{t['Kc_long']:.0f}c  x{t['contracts']}  {t['status']:<4} ${t['pnl']:>8,.0f}  eq ${t['equity']:,.0f}")
print("VOLARB_FULL_DONE")
