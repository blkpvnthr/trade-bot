#!/usr/bin/env python3
"""
study_premarket_relvol.py — PRE-SCREEN for the premarket-high-relvol → ATM-10DTE-call strategy.

Before fetching intraday + real option chains, answer the cheap question: does a high-relative-volume
day even have positive DIRECTIONAL follow-through in SHARES? If buying the mover and riding the day
isn't profitable in shares (no spread, no theta), the 10DTE ATM-CALL version is dead on arrival.

Daily proxy for the scan (vol>100K, price>$1, relvol high): relvol = day volume / 20d avg volume.
Measures per high-relvol day:
  - open->close  : realistic "ride to close" capture
  - open->high   : BEST case = exit exactly at today's high (the target; not achievable prospectively)
  - open->low    : worst case (proxy for an opening-range-low stop being run)
Per-year + bootstrap MC. NOTE: this does NOT model the ORB-breakout entry / ORB-low stop (needs
intraday) — it's the upper-bound directional screen. If open->close is negative/flat, stop here.
"""
from __future__ import annotations

import glob
import os
from datetime import date as _date

import numpy as np
import pandas as pd

import backtest_insider_copier as base

RELVOL_MIN = 2.0
PRICE_MIN = 1.0
VOL_MIN = 100_000
TRAIN_END = _date(2024, 1, 1)


def _pf(v):
    w, l = v[v > 0], v[v < 0]
    return (w.sum() / abs(l.sum())) if len(l) else float("inf")


def collect():
    rows = []
    for fp in glob.glob(str(base.BARS_DIR / "*_day.parquet")):
        sym = os.path.basename(fp).rsplit("_day", 1)[0]
        try:
            b = pd.read_parquet(fp)
        except Exception:
            continue
        if len(b) < 60:
            continue
        o = b["open"].astype(float).values; h = b["high"].astype(float).values
        l = b["low"].astype(float).values; c = b["close"].astype(float).values
        v = b["volume"].astype(float).values
        idx = [d.date() if hasattr(d, "date") else d for d in b.index]
        avg20 = pd.Series(v).rolling(20).mean().values
        for i in range(21, len(b)):
            if avg20[i - 1] <= 0 or o[i] < PRICE_MIN or v[i] < VOL_MIN:
                continue
            relvol = v[i] / avg20[i - 1]
            if relvol < RELVOL_MIN or o[i] <= 0:
                continue
            rows.append({"sym": sym, "date": idx[i], "relvol": relvol,
                         "oc": c[i] / o[i] - 1, "oh": h[i] / o[i] - 1, "ol": l[i] / o[i] - 1,
                         "gap": (o[i] / c[i - 1] - 1) if c[i - 1] > 0 else 0.0})
    return rows


def _yr_table(rows, key):
    out = []
    for yr in range(2021, 2027):
        yv = np.array([r[key] for r in rows if r["date"].year == yr])
        if len(yv) >= 20:
            out.append((yr, len(yv), yv.mean(), (yv > 0).mean(), _pf(yv)))
    return out


def main():
    rows = collect()
    print(f"high-relvol day-events (relvol>{RELVOL_MIN}, px>${PRICE_MIN}, vol>{VOL_MIN:,}): {len(rows):,}")
    oc = np.array([r["oc"] for r in rows]); oh = np.array([r["oh"] for r in rows]); ol = np.array([r["ol"] for r in rows])
    rng = np.random.RandomState(7)
    test_oc = np.array([r["oc"] for r in rows if r["date"] >= TRAIN_END])
    meds = [np.prod(1 + rng.choice(test_oc, size=min(100, len(test_oc)), replace=True)) - 1 for _ in range(4000)] if len(test_oc) > 20 else [0]
    mc_p = float(np.mean([m > 0 for m in meds])); mc_med = float(np.median(meds))
    L = ["# Premarket high-relvol → SHARES pre-screen (upper-bound directional test)", "",
         f"{len(rows):,} high-relvol day-events. Entry at OPEN, no costs (most generous). "
         "open→close = ride-to-close; open→high = best-case 'exit at today's high'; open→low = stop-run proxy.", "",
         "| Capture | mean | median | win% | PF |", "|---|---|---|---|---|",
         f"| open→close (realistic) | {oc.mean()*100:+.2f}% | {np.median(oc)*100:+.2f}% | {(oc>0).mean()*100:.0f}% | {_pf(oc):.2f} |",
         f"| open→high (best case) | {oh.mean()*100:+.2f}% | {np.median(oh)*100:+.2f}% | — | — |",
         f"| open→low (worst case) | {ol.mean()*100:+.2f}% | {np.median(ol)*100:+.2f}% | — | — |", "",
         f"Mean gap at open: {np.mean([r['gap'] for r in rows])*100:+.2f}% (are these gap-and-fade or continuation?)", "",
         "## open→close by year (does the momentum continue, regime-stable?)", "",
         "| Year | n | mean | win% | PF |", "|---|---|---|---|---|"]
    yr_fail = False; nyr = 0
    for yr, n, m, w, pf in _yr_table(rows, "oc"):
        nyr += 1
        if pf < 1.2:
            yr_fail = True
        L.append(f"| {yr}{' ⚠' if pf<1.2 else ''} | {n:,} | {m*100:+.2f}% | {w*100:.0f}% | {pf:.2f} |")
    L += ["", f"MC (OOS open→close): P(profit)={mc_p*100:.0f}%, median={mc_med*100:+.2f}%", ""]
    # verdict — note: ATM 10DTE call needs the underlying move to clear ~7% spread + theta over the hold
    survives = (_pf(oc) >= 1.3) and (oc.mean() > 0) and (not yr_fail)
    L += ["## VERDICT (shares upper bound)",
          f"- open→close PF {_pf(oc):.2f}, mean {oc.mean()*100:+.2f}%, per-year {'STABLE' if not yr_fail else 'UNSTABLE'}",
          f"- **{'PROCEED to intraday ORB + option-chain test' if survives else 'STOP — no directional edge in shares; the 10DTE ATM-call version cannot work (option friction ≫ this).'}**",
          "",
          "Reminder: even if shares show a small edge, an ATM 10DTE call needs the underlying to move "
          "~7%+ (spread) plus cover ~10d of theta just to BREAK EVEN — so the bar for the option version "
          "is far higher than for shares."]
    out = "\n".join(L)
    (base.PROJECT_ROOT / "backtest_outputs" / "premarket_relvol_prescreen.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
