#!/usr/bin/env python3
"""
liquidity_sweep_signal.py — Liquidity Sweep Trend + Delta signal, computed on the UNDERLYING
stock/ETF bars (NOT on the option contract). Used to decide WHEN to scan/buy/exit options.

Pine logic (ported exactly):
    basis     = SMA(EMA(close, len), len)
    deviation = EMA(|close - basis|, len*3) * mult
    upper     = basis + deviation
    lower     = basis - deviation
    trend persists: close>upper → +1 ; close<lower → -1 ; else hold
    bullishTrend = crossover(trend, 0)      bearishTrend = crossunder(trend, 0)
    delta        = upVolume - downVolume     (per-bar approx: +vol if close>=open else -vol)
    buySignal    = bullishTrend and delta>0
    sellSignal   = bearishTrend and delta<0

This is a DIRECTION/timing signal on the underlying — it does not select or price options.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def liquidity_sweep_signals(df: pd.DataFrame, length: int = 12, mult: float = 2.0) -> pd.DataFrame:
    """Append upper/lower/trend/delta/buySignal/sellSignal columns to underlying OHLCV bars."""
    ema1 = df["close"].ewm(span=length, adjust=False).mean()
    basis = ema1.rolling(length).mean()
    deviation = (df["close"] - basis).abs().ewm(span=length * 3, adjust=False).mean() * mult
    upper = basis + deviation
    lower = basis - deviation

    trend = []
    current = 0
    for close, up, low in zip(df["close"], upper, lower):
        if pd.notna(up) and close > up:
            current = 1
        elif pd.notna(low) and close < low:
            current = -1
        trend.append(current)

    out = df.copy()
    out["upper"] = upper
    out["lower"] = lower
    out["trend"] = trend
    # per-bar volume-delta approximation (TradingView LTF delta, current-bar form)
    out["delta"] = np.where(out["close"] >= out["open"], out["volume"], -out["volume"])
    out["deltaBullish"] = out["delta"] > 0
    out["deltaBearish"] = out["delta"] < 0

    prev_trend = out["trend"].shift(1)
    out["bullishTrend"] = (out["trend"] > 0) & (prev_trend <= 0)   # crossover(trend, 0)
    out["bearishTrend"] = (out["trend"] < 0) & (prev_trend >= 0)   # crossunder(trend, 0)
    out["buySignal"] = out["bullishTrend"] & out["deltaBullish"]
    out["sellSignal"] = out["bearishTrend"] & out["deltaBearish"]
    return out


def latest_signal(df: pd.DataFrame, length: int = 12, mult: float = 2.0) -> Optional[dict]:
    """Latest-bar signal for the live gate: {buySignal, sellSignal, trend, delta}. None if too short."""
    if df is None or len(df) < max(length * 2, 30) or not {"open", "close", "volume"} <= set(df.columns):
        return None
    s = liquidity_sweep_signals(df, length, mult).iloc[-1]
    return {"buySignal": bool(s["buySignal"]), "sellSignal": bool(s["sellSignal"]),
            "trend": int(s["trend"]), "delta": float(s["delta"])}


if __name__ == "__main__":
    import backtest_alpha_momentum_options as mom
    for sym in ("AAPL", "NVDA", "SPY"):
        b = mom._bars(sym)
        if b is None or b.empty:
            continue
        sig = liquidity_sweep_signals(b, 12, 2.0)
        nb = int(sig["buySignal"].sum()); ns = int(sig["sellSignal"].sum())
        print(f"{sym}: {len(b)} bars, buySignals={nb}, sellSignals={ns}, last trend={sig['trend'].iloc[-1]}")
