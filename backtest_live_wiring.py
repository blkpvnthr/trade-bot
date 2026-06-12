#!/usr/bin/env python3
"""backtest_live_wiring.py — backtest the ACTUAL deployed engine logic.

Imports live_trend_engine.compute_allocation (the single source of truth the live engine uses)
and replays it monthly on historical data with the DEPLOYED defaults (target-vol 10%, lev cap
1.25x, window 60, band 2%) + realistic borrow (5.5%) and turnover (5bps) cost.
Proves the deployed code reproduces the validated multi-asset vol-target result.
Logs: logs/backtest/livewiring_trades.csv + livewiring_equity.csv
"""
from __future__ import annotations
import sys, math, csv, bisect, statistics
from datetime import date
from pathlib import Path
sys.path.insert(0, "/Users/blkpvnthr/Desktop/trader")
import warnings; warnings.filterwarnings("ignore")
import yfinance as yf
import live_trend_engine as LE   # <-- the deployed engine; we call its real allocation fn

TV, MAXLEV, WIN, BAND = 0.10, 1.25, 60, 0.02
BORROW, COST_BPS = 0.055, 5
OUT = Path("/Users/blkpvnthr/Desktop/trader/logs/backtest"); OUT.mkdir(parents=True, exist_ok=True)
ASSETS = LE.UNIVERSE   # exactly the deployed universe

raw = yf.download(ASSETS, start="2006-01-01", end="2026-06-06", progress=False, auto_adjust=True)["Close"].dropna(how="all")
PX = {s: {d.date(): float(v) for d, v in raw[s].dropna().items()} for s in ASSETS}
DT = {s: sorted(PX[s]) for s in ASSETS}

def trunc(s, d):
    i = bisect.bisect_right(DT[s], d)
    return {DT[s][j]: PX[s][DT[s][j]] for j in range(i)}
def mret(s, d0, d1):
    a = bisect.bisect_right(DT[s], d0) - 1; b = bisect.bisect_right(DT[s], d1) - 1
    if a < 0 or b < 0: return 0.0
    return PX[s][DT[s][b]] / PX[s][DT[s][a]] - 1

START, END = date(2008, 1, 1), date(2026, 6, 1)
alld = sorted(set().union(*[set(PX[s]) for s in ASSETS]))
months = []; seen = set()
for d in alld:
    if START <= d <= END and (d.year, d.month) not in seen:
        months.append(d); seen.add((d.year, d.month))

equity = 1.0; rets = []; curve = []; trades = []; prev = {}
for i in range(1, len(months)):
    d0, d1 = months[i-1], months[i]
    closes = {s: trunc(s, d0) for s in ASSETS}
    alloc = LE.compute_allocation(closes, equity=1.0, target_vol=TV, max_leverage=MAXLEV, window=WIN, band=BAND)
    w = alloc["weights"]; lev = alloc["lev"]
    port = sum(w.get(s, 0) * mret(s, d0, d1) for s in w)
    turn = sum(abs(w.get(s, 0) - prev.get(s, 0)) for s in set(w) | set(prev))
    borrow_cost = max(0.0, sum(w.values()) - 1.0) * BORROW / 12
    net = port - turn * COST_BPS / 10000 - borrow_cost
    equity *= (1 + net); rets.append(net); curve.append((d1.isoformat(), round(equity, 5)))
    trades.append(dict(month=d0.isoformat(), active="|".join(alloc["active"]), leverage=round(lev, 3),
                       port_vol_raw=round(alloc["pv"], 4), gross_ret=round(port, 5),
                       turnover=round(turn, 3), net_ret=round(net, 5), equity=round(equity, 5),
                       weights=";".join(f"{s}:{w[s]:.3f}" for s in sorted(w))))
    prev = w

with open(OUT/"livewiring_trades.csv", "w", newline="") as fh:
    cw = csv.DictWriter(fh, fieldnames=list(trades[0].keys())); cw.writeheader(); cw.writerows(trades)
with open(OUT/"livewiring_equity.csv", "w", newline="") as fh:
    cw = csv.writer(fh); cw.writerow(["month", "equity"]); cw.writerows(curve)

m = statistics.mean(rets); sd = statistics.pstdev(rets) or 1e-9
sh = m / sd * math.sqrt(12)
vals = [1.0] + [c[1] for c in curve]; peak = vals[0]; mdd = 0
for v in vals: peak = max(peak, v); mdd = max(mdd, (peak - v) / peak)
yrs = len(rets) / 12; cagr = vals[-1] ** (1 / yrs) - 1
n_fire = sum(1 for t in trades if t["active"])
avg_lev = statistics.mean([t["leverage"] for t in trades if t["active"]]) if n_fire else 0
print(f"BACKTEST OF DEPLOYED WIRING (live_trend_engine.compute_allocation), 2008-2026")
print(f"  params: target_vol {TV:.0%}  max_lev {MAXLEV}x  window {WIN}d  band {BAND:.0%}  +{BORROW:.1%} borrow +{COST_BPS}bps")
print(f"  months={len(rets)}  invested={n_fire}  avg_leverage={avg_lev:.2f}x")
print(f"  CAGR={cagr*100:.1f}%/yr   Sharpe={sh:.2f}   maxDD={mdd*100:.1f}%   Calmar={cagr/mdd if mdd else 0:.2f}   final={vals[-1]:.2f}x")
print(f"  (expected ~ sweep tv10/lev1.25 line: CAGR~8.1 Sharpe~0.95 maxDD~13.5 Calmar~0.60)")
print(f"  logs -> logs/backtest/livewiring_trades.csv + livewiring_equity.csv")
print("  recent allocations:")
for t in trades[-4:]:
    print(f"    {t['month']}  lev {t['leverage']}  [{t['weights']}]  net {t['net_ret']*100:+.1f}%")
print("LIVEWIRING_BT_DONE")
