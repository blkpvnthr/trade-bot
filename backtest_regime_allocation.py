#!/usr/bin/env python3
"""Research: low-turnover regime allocation to maintain a positive equity curve.

The honest version of "use the universe to stay positive": hold a bull ETF in a
confirmed up-regime, rotate to an inverse ETF (or cash) in a down-regime, rebalanced
DAILY on completed daily bars. Low turnover -> low cost drag; a slow trend filter is
predictive, not reactive. Tested over multiple years (multi-regime) vs buy-and-hold.

Falsification-first: standard (untuned) SMA lengths 50 and 200; walk-forward by year;
realistic switch cost. No look-ahead (signal uses yesterday's completed bar).
RESEARCH ONLY - not wired into any live engine.
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

SWITCH_COST = 0.0010  # 10 bps per switch (round-trip-ish on a liquid ETF swap)


def daily_bars(symbols, years):
    cli = StockHistoricalDataClient(os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"))
    end = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=int(years * 365))
    out = {}
    for s in symbols:
        req = StockBarsRequest(symbol_or_symbols=[s], timeframe=TimeFrame.Day, start=start, end=end, feed=DataFeed.IEX)
        df = cli.get_stock_bars(req).df
        if df is None or df.empty:
            print(f"[data] {s}: none"); continue
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        out[s] = df.set_index("timestamp")["close"].astype(float).sort_index()
        print(f"[data] {s}: {len(out[s])} daily bars {out[s].index.min().date()}..{out[s].index.max().date()}")
    return out


def daily_ret(close):
    return close.pct_change().fillna(0.0)


def regime_strategy(spy, bull, bear, sma_len, use_cash_only=False):
    """Allocate next day based on yesterday's completed regime (no look-ahead)."""
    sma = spy.rolling(sma_len).mean()
    up = (spy > sma) & (sma > sma.shift(5))  # above a RISING sma = up-regime
    pos = up.shift(1).fillna(False)          # act on yesterday's signal
    bull_r, bear_r = daily_ret(bull), daily_ret(bear)
    # align
    idx = pos.index
    out = pd.Series(0.0, index=idx)
    prev = None
    for t in idx:
        if use_cash_only:
            r = bull_r.get(t, 0.0) if pos[t] else 0.0
            state = "bull" if pos[t] else "cash"
        else:
            if pos[t]:
                r, state = bull_r.get(t, 0.0), "bull"
            else:
                r, state = bear_r.get(t, 0.0), "bear"
        if prev is not None and state != prev:
            r -= SWITCH_COST
        out[t] = r
        prev = state
    return out.loc[sma.dropna().index[0]:]  # drop warmup


def metrics(ret, label):
    eq = (1 + ret).cumprod()
    yrs = max(len(ret) / 252.0, 1e-9)
    sh = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if len(eq) else 0.0
    dd = (eq / eq.cummax() - 1).min() if len(eq) else 0.0
    return {"label": label, "Sharpe": round(sh, 2), "CAGR%": round(100 * cagr, 1),
            "maxDD%": round(100 * dd, 1), "final_x": round(float(eq.iloc[-1]), 2)}


def by_year(ret):
    return {int(y): round(100 * ((1 + g).prod() - 1), 1) for y, g in ret.groupby(ret.index.year)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=9.0)
    ap.add_argument("--bull", default="QQQ")
    ap.add_argument("--bear", default="SH")   # 1x inverse SPY (tighter than 3x SQQQ)
    args = ap.parse_args()
    data = daily_bars(["SPY", args.bull, args.bear], args.years)
    if not all(s in data for s in ["SPY", args.bull, args.bear]):
        print("missing data"); return
    spy, bull, bear = data["SPY"], data[args.bull], data[args.bear]

    print(f"\nbull={args.bull} bear={args.bear} switch_cost={SWITCH_COST*1e4:.0f}bps")
    bh = daily_ret(spy)
    print("\nBUY-HOLD SPY     :", metrics(bh, "bh"))
    for n in (50, 200):
        inv = regime_strategy(spy, bull, bear, n)
        cash = regime_strategy(spy, bull, bear, n, use_cash_only=True)
        print(f"\nREGIME SMA{n} (bull/inverse):", metrics(inv, f"inv{n}"))
        print(f"  per-year %:", by_year(inv))
        print(f"REGIME SMA{n} (bull/CASH)   :", metrics(cash, f"cash{n}"))


if __name__ == "__main__":
    main()
