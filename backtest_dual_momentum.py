#!/usr/bin/env python3
"""Research: dual momentum across broad-index ETFs (RESEARCH ONLY).

Classic Antonacci dual momentum on {SPY, QQQ, DIA, IWM} (IWM = Russell 2000 proxy for
RUT). Monthly rebalance: relative momentum picks the strongest trailing-return index;
absolute momentum requires that index's own trailing return > 0, else go to cash.
Low turnover, standard (untuned) 12- and 6-month lookbacks, multi-year, no look-ahead
(decide at month t from data <= t, hold month t+1). Compared vs buy-hold SPY and an
equal-weight monthly-rebalanced basket. Not wired into any live engine.
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

SWITCH_COST = 0.0010  # 10 bps per rebalance change


def monthly_closes(symbols, years):
    cli = StockHistoricalDataClient(os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"))
    end = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=int(years * 365))
    cols = {}
    for s in symbols:
        df = cli.get_stock_bars(StockBarsRequest(symbol_or_symbols=[s], timeframe=TimeFrame.Day,
                                                 start=start, end=end, feed=DataFeed.IEX)).df
        if df is None or df.empty:
            print(f"[data] {s}: none"); continue
        c = df.reset_index()
        c.columns = [x.lower() for x in c.columns]
        c["timestamp"] = pd.to_datetime(c["timestamp"], utc=True)
        s_close = c.set_index("timestamp")["close"].astype(float).sort_index()
        cols[s] = s_close.resample("ME").last()
        print(f"[data] {s}: {len(s_close)} daily -> {len(cols[s])} monthly, {s_close.index.min().date()}..{s_close.index.max().date()}")
    return pd.DataFrame(cols).dropna()


def metrics(monthly_ret, label):
    r = monthly_ret.dropna()
    eq = (1 + r).cumprod()
    yrs = max(len(r) / 12.0, 1e-9)
    sh = (r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else 0.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if len(eq) else 0.0
    dd = (eq / eq.cummax() - 1).min() if len(eq) else 0.0
    return {"label": label, "Sharpe": round(sh, 2), "CAGR%": round(100 * cagr, 1),
            "maxDD%": round(100 * dd, 1), "final_x": round(float(eq.iloc[-1]), 2), "months": len(r)}


def dual_momentum(m, lookback):
    """m: DataFrame of monthly closes. Returns monthly strategy return series."""
    mom = m / m.shift(lookback) - 1.0         # trailing return known at each month-end
    mret = m.pct_change()                      # realized monthly returns
    idx = m.index
    out = pd.Series(0.0, index=idx)
    held = []
    prev = None
    for i in range(lookback, len(idx) - 1):
        row = mom.iloc[i]
        winner = row.idxmax()
        pick = winner if row[winner] > 0 else "CASH"   # absolute-momentum filter
        r = 0.0 if pick == "CASH" else float(mret[winner].iloc[i + 1])
        if prev is not None and pick != prev:
            r -= SWITCH_COST
        out.iloc[i + 1] = r
        held.append(pick)
        prev = pick
    return out.iloc[lookback + 1:], pd.Series(held).value_counts().to_dict()


def by_year(r):
    return {int(y): round(100 * ((1 + g).prod() - 1), 1) for y, g in r.groupby(r.index.year)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10.0)
    args = ap.parse_args()
    syms = ["SPY", "QQQ", "DIA", "IWM"]
    m = monthly_closes(syms, args.years)
    if m.empty or len(m) < 15:
        print("insufficient data"); return
    print(f"\nperiod: {m.index.min().date()}..{m.index.max().date()}  ({len(m)} months)  switch_cost={SWITCH_COST*1e4:.0f}bps")

    spy_bh = m["SPY"].pct_change().iloc[1:]
    ew = m[syms].pct_change().mean(axis=1).iloc[1:]
    print("\nCONTROLS:")
    print("  buy-hold SPY       :", metrics(spy_bh, "spy_bh"))
    print("  equal-weight 4 (mo):", metrics(ew, "ew4"))

    for lb in (12, 6):
        r, held = dual_momentum(m, lb)
        print(f"\nDUAL MOMENTUM lookback={lb}m:", metrics(r, f"dm{lb}"))
        print(f"  holdings count: {held}")
        print(f"  per-year %: {by_year(r)}")


if __name__ == "__main__":
    main()
