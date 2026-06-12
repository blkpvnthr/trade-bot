#!/usr/bin/env python3
"""gate_backtest.py — backtest the structure-GATE logic on the underlying (daily, 2018-2026).

WHAT THIS DOES (and does NOT) test:
  ✓ the gate's trade logic: long breakout when price>SMA200 & momentum>0, TP=next structure
    resistance, stop=structure support; measures win%, avg R, expectancy, profit factor.
  ✗ NOT the TradingView screener selection (point-in-time, no historical premarket/ratings).
  ✗ NOT intraday ORB/VWAP/POC (daily approximation: prior-day-high breakout, swing levels).
  ✗ NOT options P&L (tests the UNDERLYING move; a long call would amplify w/ convexity - theta).

Daily-bar approximation of the gate:
  entry trigger : price>SMA200 AND mom10>0 AND SMA50>SMA200 AND today breaks prior-day-high
  entry price   : prior-day-high (the breakout level)
  stop          : recent swing low (structure support)
  target (TP)   : nearest swing high above entry (structure resistance); skip if R:R<1.5
  exit          : first of TP / stop / 20-day time stop (close)
"""
from __future__ import annotations
import sys, statistics
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import yfinance as yf
import numpy as np

HERE = Path("/Users/blkpvnthr/Desktop/trader")
MIN_RR = 1.5
TIME_STOP = 20
SWING_W = 5
LOOKBACK_RES = 60


def read_watchlist():
    import re
    p = HERE / "options_watchlist.txt"
    toks = re.split(r"[\s,]+", p.read_text())
    syms = [t.strip().strip('"').strip("'").upper() for t in toks]
    return [s for s in syms if s and "/" not in s and all(c.isalnum() or c in ".-" for c in s)]


def swing_highs_lows(h, l, w=SWING_W):
    hi = [i for i in range(w, len(h) - w) if h[i] == max(h[i - w:i + w + 1])]
    lo = [i for i in range(w, len(l) - w) if l[i] == min(l[i - w:i + w + 1])]
    return set(hi), set(lo)


def backtest_symbol(sym):
    df = yf.download(sym, start="2017-01-01", end="2026-06-06", progress=False, auto_adjust=False)
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1: df.columns = df.columns.droplevel(1)
    if len(df) < 260: return []
    o = df["Open"].values; h = df["High"].values; l = df["Low"].values; c = df["Close"].values
    n = len(c)
    sma200 = np.convolve(c, np.ones(200) / 200, mode="full")[:n]; sma200[:199] = np.nan
    sma50 = np.convolve(c, np.ones(50) / 50, mode="full")[:n]; sma50[:49] = np.nan
    for i in range(200, n): sma200[i] = c[i - 199:i + 1].mean(); sma50[i] = c[i - 49:i + 1].mean()
    swH, swL = swing_highs_lows(h, l)
    trades = []; i = 210
    while i < n - 1:
        # signal evaluated on bar i-1, act on bar i
        p = i - 1
        if not (c[p] > sma200[p] and sma50[p] > sma200[p]): i += 1; continue
        mom10 = c[p] / c[p - 10] - 1 if p >= 10 else 0
        if mom10 <= 0: i += 1; continue
        entry = h[p]                                   # breakout over prior-day-high
        if h[i] < entry: i += 1; continue              # no breakout today
        # stop = nearest swing low below entry (structure support)
        lows = [l[j] for j in swL if j < i and l[j] < entry]
        stop = max(lows) if lows else l[p]
        # TP = nearest swing high above entry within lookback (structure resistance)
        highs = sorted(h[j] for j in swH if i - LOOKBACK_RES <= j < i and h[j] > entry)
        if not highs: i += 1; continue
        tp = highs[0]
        risk = entry - stop; reward = tp - entry
        if risk <= 0 or reward / risk < MIN_RR: i += 1; continue
        # simulate forward
        outcome = None
        for j in range(i, min(i + TIME_STOP, n)):
            if l[j] <= stop: outcome = (stop - entry) / risk; break      # -1R
            if h[j] >= tp: outcome = (tp - entry) / risk; break          # +reward/risk R
        if outcome is None: outcome = (c[min(i + TIME_STOP, n) - 1] - entry) / risk
        trades.append(outcome)
        # move past this trade's horizon (no overlapping positions per name)
        i = j + 1 if outcome is not None else i + 1
    return trades


def main():
    syms = read_watchlist()
    print(f"GATE BACKTEST (underlying, daily, 2018-2026) — {len(syms)} names, min R:R {MIN_RR}\n")
    allR = []; per = []
    for s in syms:
        try:
            t = backtest_symbol(s)
        except Exception:
            t = []
        if t:
            allR.extend(t); per.append((s, len(t), sum(1 for x in t if x > 0) / len(t), statistics.mean(t)))
    if not allR:
        print("no trades"); return
    n = len(allR); wins = [x for x in allR if x > 0]; losses = [x for x in allR if x <= 0]
    W = len(wins) / n
    aw = statistics.mean(wins) if wins else 0; al = abs(statistics.mean(losses)) if losses else 1e-9
    exp = statistics.mean(allR)
    pf = sum(wins) / abs(sum(losses)) if losses else float("inf")
    print(f"  trades={n}  win%={W*100:.1f}  avgWin={aw:+.2f}R  avgLoss={-al:+.2f}R  expectancy={exp:+.3f}R/trade  profit_factor={pf:.2f}")
    print(f"  total R={sum(allR):+.0f}   (Kelly break-even R-ratio for this win% = {(1-W)/W:.2f}; realized W/L ratio = {aw/al:.2f})")
    verdict = "POSITIVE expectancy" if exp > 0 else "NEGATIVE expectancy"
    print(f"  -> {verdict}\n")
    print("  per-name (top/bottom by expectancy):")
    per.sort(key=lambda x: -x[3])
    for s, k, w, e in per[:6] + per[-6:]:
        print(f"     {s:<6} n={k:<3} win%={w*100:4.0f}  exp={e:+.2f}R")
    print("GATE_BT_DONE")


if __name__ == "__main__":
    main()
