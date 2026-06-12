#!/usr/bin/env python3
"""multi_asset_trend.py — multi-asset 200DMA trend overlay w/ vol-targeted sizing.

3-way attribution (each monthly-rebalanced, 2008-2026, total-return/adj-close):
  A) equity-only  : SPY/QQQ/IWM, per-asset 200DMA gate, equal-weight active   (~deployed)
  B) +multi-asset : add TLT/GLD/DBC/UUP (bonds/gold/commodities/$), equal-weight active
  C) +vol-target  : multi-asset, inverse-vol weights, leverage to 12% target vol (cap 1.5x)
Per-asset signal: price>200DMA*1.02 AND 200DMA 20d-slope>0, else that asset -> cash.

Honest eval: equity curve, CAGR/Sharpe/maxDD/Calmar/worst-month, + rebalance-cost sensitivity.
Logs: logs/backtest/multiasset_equity.csv  +  multiasset_trades.csv (variant C monthly weights).
"""
from __future__ import annotations
import math, csv, statistics
from datetime import date
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import yfinance as yf

EQUITY = ["SPY", "QQQ", "IWM"]
DIVERSIFIERS = ["TLT", "GLD", "DBC", "UUP"]
ALL = EQUITY + DIVERSIFIERS
START, END = date(2008, 1, 1), date(2026, 6, 1)
TARGET_VOL = 0.12
MAX_LEV = 1.5
COST_BPS = 5  # per-side rebalance cost in bps of turnover
OUT = Path("/Users/blkpvnthr/Desktop/trader/logs/backtest"); OUT.mkdir(parents=True, exist_ok=True)

# --- load adjusted (total-return) daily closes ---
raw = yf.download(ALL, start="2006-01-01", end="2026-06-06", progress=False, auto_adjust=True)["Close"]
raw = raw.dropna(how="all")
px = {s: {d.date(): float(v) for d, v in raw[s].dropna().items()} for s in ALL}
alldays = sorted(set().union(*[set(px[s]) for s in ALL]))

def sma(s, d, n=200):
    days = [x for x in px[s] if x <= d]
    return sum(px[s][x] for x in days[-n:]) / n if len(days) >= n else None
def rising(s, d):
    days = [x for x in px[s] if x <= d]
    if len(days) < 221: return False
    a, b = sma(s, days[-1]), sma(s, days[-21]); return bool(a and b and a > b)
def signal(s, d):
    days = [x for x in px[s] if x <= d]
    if not days: return False
    c = px[s][days[-1]]; m = sma(s, d)
    return bool(m and c > m * 1.02 and rising(s, d))
def daily_rets(s, d, n=60):
    days = [x for x in px[s] if x <= d][-(n+1):]
    if len(days) < n + 1: return None
    return np.array([math.log(px[s][days[i]] / px[s][days[i-1]]) for i in range(1, len(days))])
def vol(s, d, n=60):
    r = daily_rets(s, d, n); return float(np.std(r) * math.sqrt(252)) if r is not None else None
def mret(s, d0, d1):
    a = [x for x in px[s] if x <= d0]; b = [x for x in px[s] if x <= d1]
    if not a or not b: return 0.0
    return px[s][b[-1]] / px[s][a[-1]] - 1

# month starts
months = []; seen = set()
for d in alldays:
    if START <= d <= END and (d.year, d.month) not in seen:
        months.append(d); seen.add((d.year, d.month))

def weights_A(d):
    act = [s for s in EQUITY if signal(s, d)]
    return {s: 1.0 / len(act) for s in act} if act else {}
def weights_B(d):
    act = [s for s in ALL if signal(s, d)]
    return {s: 1.0 / len(act) for s in act} if act else {}
def weights_C(d):
    act = [s for s in ALL if signal(s, d) and vol(s, d)]
    if not act: return {}, 0.0, 0.0
    iv = {s: 1.0 / vol(s, d) for s in act}
    tot = sum(iv.values()); wraw = {s: iv[s] / tot for s in act}
    # covariance (trailing 60d) for portfolio vol
    R = []
    minlen = min(len(daily_rets(s, d)) for s in act)
    for s in act: R.append(daily_rets(s, d)[-minlen:])
    cov = np.cov(np.array(R)) * 252 if len(act) > 1 else np.array([[vol(act[0], d) ** 2]])
    wv = np.array([wraw[s] for s in act])
    pv = float(math.sqrt(max(wv @ np.atleast_2d(cov) @ wv, 1e-9)))
    lev = min(TARGET_VOL / pv, MAX_LEV) if pv > 0 else 1.0
    return {s: wraw[s] * lev for s in act}, lev, pv

def run(weight_fn, label, log_trades=False):
    eq = 1.0; rets = []; curve = []; prev_w = {}; tradelog = []
    for i in range(1, len(months)):
        d0, d1 = months[i-1], months[i]
        if weight_fn == "C":
            w, lev, pv = weights_C(d0)
        else:
            w = (weights_A if weight_fn == "A" else weights_B)(d0); lev = sum(w.values()); pv = 0.0
        gross = mret_port = sum(w.get(s, 0) * mret(s, d0, d1) for s in w)
        turnover = sum(abs(w.get(s, 0) - prev_w.get(s, 0)) for s in set(w) | set(prev_w))
        cost = turnover * COST_BPS / 10000.0
        net = mret_port - cost
        eq *= (1 + net); rets.append(net)
        curve.append((d1.isoformat(), round(eq, 4)))
        if log_trades:
            tradelog.append(dict(month=d0.isoformat(), n_active=len(w), leverage=round(lev, 2),
                                 port_vol_est=round(pv, 3), gross_ret=round(gross, 4),
                                 turnover=round(turnover, 2), net_ret=round(net, 4), equity=round(eq, 4),
                                 weights=";".join(f"{s}:{w[s]:.2f}" for s in sorted(w))))
        prev_w = w
    return rets, curve, tradelog

def stats(rets, curve):
    m = statistics.mean(rets); sd = statistics.pstdev(rets) or 1e-9
    sh = m / sd * math.sqrt(12)
    vals = [1.0] + [c[1] for c in curve]
    peak = vals[0]; mdd = 0
    for v in vals: peak = max(peak, v); mdd = max(mdd, (peak - v) / peak)
    yrs = len(rets) / 12
    cagr = vals[-1] ** (1 / yrs) - 1
    calmar = cagr / mdd if mdd > 0 else 0
    return cagr * 100, sh, mdd * 100, calmar, min(rets) * 100

print(f"MULTI-ASSET TREND OVERLAY  2008-2026  (monthly, vol-target {TARGET_VOL*100:.0f}%, cap {MAX_LEV}x, cost {COST_BPS}bps)\n")
print(f"  {'variant':<28}{'CAGR%':>8}{'Sharpe':>8}{'maxDD%':>8}{'Calmar':>8}{'worstMo%':>9}")
results = {}
for code, lbl in [("A", "A equity-only (~deployed)"), ("B", "B +multi-asset eqw"), ("C", "C +vol-target")]:
    rets, curve, tl = run(code, lbl, log_trades=(code == "C"))
    results[code] = (rets, curve, tl)
    cg, sh, dd, cal, wm = stats(rets, curve)
    print(f"  {lbl:<28}{cg:>8.1f}{sh:>8.2f}{dd:>8.1f}{cal:>8.2f}{wm:>9.1f}")

# write logs (all 3 equity curves + variant C trades)
Acurve = {m: e for m, e in results["A"][1]}; Bcurve = {m: e for m, e in results["B"][1]}; Ccurve = {m: e for m, e in results["C"][1]}
with open(OUT/"multiasset_equity.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["month", "equity_A_eqonly", "equity_B_multi", "equity_C_voltgt"])
    for m in [c[0] for c in results["C"][1]]:
        w.writerow([m, Acurve.get(m, ""), Bcurve.get(m, ""), Ccurve.get(m, "")])
with open(OUT/"multiasset_trades.csv", "w", newline="") as fh:
    tl = results["C"][2]; w = csv.DictWriter(fh, fieldnames=list(tl[0].keys())); w.writeheader(); w.writerows(tl)
print(f"\n  logs -> logs/backtest/multiasset_equity.csv + multiasset_trades.csv ({len(results['C'][2])} months)")
print("  recent variant-C allocations:")
for t in results["C"][2][-4:]:
    print(f"    {t['month']}  lev {t['leverage']}  volEst {t['port_vol_est']}  [{t['weights']}]  net {t['net_ret']*100:+.1f}%")
print("MULTIASSET_DONE")
