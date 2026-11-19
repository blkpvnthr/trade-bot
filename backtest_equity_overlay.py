#!/usr/bin/env python3
"""Research: equity-curve-reactive inverse-rotation overlay (RESEARCH ONLY).

Hypothesis: when the daily equity curve flattens/reverses, rotate into inverse/bear
ETFs to reclaim a positive equity curve. Falsification-first: compare the overlay's
equity curve to the baseline (trade the full universe every day). A reactive overlay
is only worth wiring if it raises net Sharpe AND cuts drawdown out-of-sample.

Path-dependent: the regime each day is decided from the OVERLAY's own equity slope
(no look-ahead - slope uses days strictly before today).
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

import trade
from backtest_daily_green import fetch_5min, rth_days, simulate_day

INVERSE = {"SQQQ", "SOXS", "SH", "SDS", "PSQ", "DOG", "SPXU", "SDOW", "SPXS", "QID",
           "DRIP", "TECS", "NVD"}  # bear / inverse ETFs in the universe


def all_trades(frames, slippage_bps, rr):
    rows = []
    for sym, df in frames.items():
        for date, day_df in rth_days(df):
            tr = simulate_day(day_df, "A", slippage_bps, rr)
            if tr:
                rows.append({"symbol": sym, "date": date, "ret": tr["ret"]})
    return pd.DataFrame(rows)


def day_return(day_trades, symbols, max_positions):
    sub = day_trades[day_trades["symbol"].isin(symbols)].head(max_positions)
    return float(sub["ret"].mean()) if len(sub) else 0.0


def equity_metrics(day_ret: pd.Series, label):
    eq = (1 + day_ret).cumprod()
    years = max(len(day_ret) / 252.0, 1e-9)
    sharpe = (day_ret.mean() / day_ret.std() * np.sqrt(252)) if day_ret.std() > 0 else 0.0
    cagr = eq.iloc[-1] ** (1 / years) - 1 if len(eq) else 0.0
    dd = (eq / eq.cummax() - 1).min() if len(eq) else 0.0
    return {"label": label, "net_Sharpe": round(sharpe, 2), "CAGR%": round(100 * cagr, 1),
            "maxDD%": round(100 * dd, 1), "final_eq": round(float(eq.iloc[-1]), 3),
            "down_days%": round(100 * float((day_ret < 0).mean()), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--slope-lb", type=int, default=5, help="equity-slope lookback (days)")
    args = ap.parse_args()

    syms = trade.load_trading_universe()
    frames = fetch_5min(syms, args.days)
    tdf = all_trades(frames, args.slippage_bps, args.rr)
    dates = sorted(tdf["date"].unique())
    by_day = {d: g for d, g in tdf.groupby("date")}
    full_syms = set(tdf["symbol"].unique())

    # Baseline: full universe every day
    base = pd.Series({d: day_return(by_day[d], full_syms, args.max_positions) for d in dates}).sort_index()

    # Overlay: regime from the overlay's OWN trailing equity slope (no look-ahead)
    ov, eq = {}, [1.0]
    inverse_days = 0
    for i, d in enumerate(dates):
        if i >= args.slope_lb:
            slope = eq[-1] - eq[-1 - args.slope_lb]
            regime_inverse = slope <= 0  # flatten or reverse
        else:
            regime_inverse = False
        syms_today = INVERSE if regime_inverse else full_syms
        r = day_return(by_day[d], syms_today, args.max_positions)
        ov[d] = r
        eq.append(eq[-1] * (1 + r))
        inverse_days += int(regime_inverse)
    ov = pd.Series(ov).sort_index()

    print(f"[overlay] days={len(dates)} inverse_regime_days={inverse_days} ({100*inverse_days/len(dates):.0f}%)")
    # Full-period
    print("\nFULL PERIOD:")
    print("  baseline:", equity_metrics(base, "baseline"))
    print("  overlay :", equity_metrics(ov, "overlay"))
    # OOS (last 35%) - prior holdout burned, so this is a rough forward slice
    cut = dates[int(len(dates) * 0.65)]
    print(f"\nOOS (> {cut}):")
    print("  baseline:", equity_metrics(base[base.index > cut], "baseline_oos"))
    print("  overlay :", equity_metrics(ov[ov.index > cut], "overlay_oos"))
    # Does the overlay help on the baseline's WORST stretch?
    base_eq = (1 + base).cumprod()
    worst_end = base_eq.idxmin() if len(base_eq) else None
    print(f"\nbaseline final_eq={float(base_eq.iloc[-1]):.3f} ; overlay final_eq={float((1+ov).cumprod().iloc[-1]):.3f}")


if __name__ == "__main__":
    main()
