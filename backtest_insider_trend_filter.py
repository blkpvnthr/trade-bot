#!/usr/bin/env python3
"""Does a trend filter (price > SMA50 / SMA200) help or hurt the insider edge?
Buckets the validated PIT insider signals by entry-day trend and compares forward alpha vs SPY.
Uses cached bars only (no live Alpaca)."""
from datetime import timedelta
import numpy as np, pandas as pd
import backtest_insider_copier as bic

events = bic.load_pit_buy_events()
spy = bic.load_bars("SPY")
spc = spy["close"].astype(float); spc.index = [d.date() if hasattr(d, "date") else d for d in spy.index]

MIN_VALUE, LAG, HOLDS = 1_000_000.0, 2, [5, 10]
rows = []
for sym, buys in events.items():
    bars = bic.load_bars(sym)
    if bars is None or bars.empty:
        continue
    c = bars["close"].astype(float); c.index = [d.date() if hasattr(d, "date") else d for d in bars.index]
    td = sorted(c.index)
    idx = {d: i for i, d in enumerate(td)}
    for d, val in buys:
        if float(val) < MIN_VALUE:
            continue
        entry = bic._next_trading_on_or_after(d + timedelta(days=LAG), td)
        i = idx.get(entry)
        if i is None or i < 200:
            continue
        px = float(c.loc[entry]); hist = c.iloc[:i + 1]
        rec = {"px": px, "above200": px > hist.tail(200).mean(), "above50": px > hist.tail(50).mean()}
        for H in HOLDS:
            ex = bic._shift_trading_days(entry, H, td)
            if ex in c.index and entry in spc.index and ex in spc.index:
                rec[f"a{H}"] = (c.loc[ex] / px - 1) - (spc.loc[ex] / spc.loc[entry] - 1)
        rows.append(rec)

df = pd.DataFrame(rows)
print(f"Total qualifying insider signals (>$1M, PIT): {len(df)}\n")


def stat(sub, H):
    a = sub.get(f"a{H}")
    a = a.dropna() if a is not None else pd.Series(dtype=float)
    if len(a) < 5:
        return None
    t = a.mean() / a.std() * np.sqrt(len(a)) if a.std() else 0.0
    return len(a), a.mean(), (a > 0).mean(), t


for H in HOLDS:
    print(f"=== {H}-day forward ALPHA vs SPY, by entry-day trend ===")
    for name, sub in [("ALL signals (no trend filter)", df),
                      ("price > SMA200 (old gate passes)", df[df.above200]),
                      ("price < SMA200 (old gate BLOCKS = dip buys)", df[~df.above200]),
                      ("price > SMA50 (new gate passes)", df[df.above50]),
                      ("price < SMA50 (new gate BLOCKS)", df[~df.above50])]:
        s = stat(sub, H)
        if s:
            print(f"  {name:44} n={s[0]:5d}  alpha={s[1]:+.2%}  win={s[2]:.0%}  t={s[3]:.2f}")
    print()
