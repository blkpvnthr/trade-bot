"""Historical fixed(-0.5%) vs dynamic(ATR-adaptive) protective-exit comparison.

Leak-free replay on GENUINE cached 1-minute bars (alpaca_minute_cache/), resampled
to 5-minute RTH bars. Both arms share IDENTICAL entries, take-profit, max-hold, and
costs; they differ ONLY in the protective LOSS exit:

  * FIXED  : exit when unrealized return <= -loss_warning (-0.5%).
  * DYNAMIC: market_scan.dynamic_loss.DynamicLossMonitor -- ATR-adaptive soft stop
             + confirmed deterioration (the exact logic wired into the engine).

No look-ahead: ATR(14), VWAP, and deterioration at bar i use only bars <= i; the
exit decision at bar i fills at bar i's close. Round-trip costs (BASELINE_COSTS)
are applied equally to both arms. Reports net return, Sharpe, max drawdown,
protective-exit count, turnover, mean MAE, and total costs per arm.

Usage: python backtest_dynamic_loss.py [--start 2024-04-01] [--end 2024-06-01]
"""
from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from costs import BASELINE_COSTS
from market_scan.baseline import BASELINE_UNIVERSE
from market_scan.dynamic_loss import DynamicLossMonitor

CACHE = os.path.join(os.path.dirname(__file__), "alpaca_minute_cache")
RT_COST = BASELINE_COSTS.round_trip_cost_bps() / 10_000.0   # fraction, applied once/trade

TAKE_PROFIT = 0.012      # +1.2% shared take-profit
MAX_HOLD_BARS = 24       # 24 x 5min = 2h shared max hold
WARNING = 0.005          # -0.5% fixed loss threshold (and dynamic warning floor)


def _load_5min(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    files = sorted(glob.glob(os.path.join(CACHE, f"{symbol}_*.csv")))
    if not files:
        return None
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=["timestamp"])
        except Exception:
            continue
        frames.append(df)
    if not frames:
        return None
    m = pd.concat(frames, ignore_index=True)
    m = m[(m["timestamp"] >= start) & (m["timestamp"] < end)].copy()
    if m.empty:
        return None
    m = m.set_index("timestamp").sort_index()
    # Regular US session in UTC (09:30-16:00 ET == 13:30-20:00 UTC).
    m = m.between_time("13:30", "20:00")
    if m.empty:
        return None
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum"}
    bars = m.resample("5min").agg(agg).dropna(subset=["close"])
    if bars.empty:
        return None
    # Rolling session VWAP proxy (cumulative within each day) and Wilder ATR(14).
    bars["day"] = bars.index.date
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    pv = tp * bars["volume"]
    bars["vwap"] = (pv.groupby(bars["day"]).cumsum()
                    / bars["volume"].groupby(bars["day"]).cumsum().replace(0, np.nan))
    prev_close = bars["close"].shift(1)
    tr = pd.concat([(bars["high"] - bars["low"]).abs(),
                    (bars["high"] - prev_close).abs(),
                    (bars["low"] - prev_close).abs()], axis=1).max(axis=1)
    bars["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    bars["atr_pct"] = bars["atr14"] / bars["close"]
    return bars


@dataclass
class Trade:
    ret: float          # net return fraction after round-trip cost
    mae: float          # max adverse excursion (most-negative unrealized return)
    loss_exit: bool     # exited via the protective loss rule


def _deteriorating(bars: pd.DataFrame, i: int) -> bool:
    row = bars.iloc[i]
    if i >= 1 and row["close"] < bars.iloc[i - 1]["low"]:
        return True                                  # lower low
    if not math.isnan(row["vwap"]) and row["close"] < row["vwap"]:
        return True                                  # lost VWAP
    if i >= 3:
        r3 = (row["close"] - bars.iloc[i - 3]["close"]) / bars.iloc[i - 3]["close"]
        if r3 < 0:
            return True                              # negative 3-bar momentum
    return False


def _simulate_symbol(symbol: str, bars: pd.DataFrame, mon: DynamicLossMonitor):
    fixed: list[Trade] = []
    dyn: list[Trade] = []
    for _day, day_bars in bars.groupby("day"):
        db = day_bars.reset_index()
        n = len(db)
        if n < 6:
            continue
        entry = float(db.iloc[0]["close"])           # enter at first RTH 5m close
        if entry <= 0:
            continue
        mon.reset(symbol)
        f_done = d_done = False
        f_ret = d_ret = None
        mae = 0.0
        for i in range(1, n):
            px = float(db.iloc[i]["close"])
            plpc = (px - entry) / entry
            mae = min(mae, plpc)
            tp_hit = plpc >= TAKE_PROFIT
            last = (i == n - 1) or (i >= MAX_HOLD_BARS)
            # FIXED arm
            if not f_done:
                if plpc <= -WARNING:
                    f_ret, f_done = plpc, True
                    f_loss = True
                elif tp_hit or last:
                    f_ret, f_done = plpc, True
                    f_loss = False
            # DYNAMIC arm
            if not d_done:
                atr = db.iloc[i]["atr_pct"]
                atr = float(atr) if atr == atr else None
                dec = mon.observe(symbol, plpc=plpc, atr_pct=atr,
                                  deteriorating=_deteriorating(db, i),
                                  now=float(db.iloc[i]["timestamp"].timestamp()))
                if dec.should_exit:
                    d_ret, d_done = plpc, True
                    d_loss = True
                elif tp_hit or last:
                    d_ret, d_done = plpc, True
                    d_loss = False
            if f_done and d_done:
                break
        if f_ret is not None:
            fixed.append(Trade(f_ret - RT_COST, mae, f_loss))
        if d_ret is not None:
            dyn.append(Trade(d_ret - RT_COST, mae, d_loss))
    return fixed, dyn


def _metrics(trades: list[Trade]) -> dict:
    if not trades:
        return {"trades": 0}
    r = np.array([t.ret for t in trades])
    cum = np.cumsum(r)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak)
    sharpe = (r.mean() / r.std() * math.sqrt(len(r))) if r.std() > 1e-12 else 0.0
    loss_exits = sum(1 for t in trades if t.loss_exit)
    return {
        "trades": len(trades),
        "net_return_total": float(cum[-1]),
        "net_return_mean_bps": float(r.mean() * 10_000),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd.min()),
        "protective_exits": loss_exits,
        "turnover_exits": len(trades),
        "mean_mae_bps": float(np.mean([t.mae for t in trades]) * 10_000),
        "total_cost_bps": float(len(trades) * RT_COST * 10_000),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-04-01")
    ap.add_argument("--end", default="2024-06-01")
    args = ap.parse_args()

    # One shared monitor instance (reset per trade) with the locked Stage B defaults.
    mon = DynamicLossMonitor(warning_pct=WARNING, adaptive_min_pct=0.005,
                             adaptive_max_pct=0.020, atr_multiplier=1.5,
                             confirmation_seconds=30.0, min_observations=3,
                             hard_stop_pct=0.025)
    all_fixed: list[Trade] = []
    all_dyn: list[Trade] = []
    covered = []
    for sym in BASELINE_UNIVERSE:
        bars = _load_5min(sym, args.start, args.end)
        if bars is None:
            continue
        f, d = _simulate_symbol(sym, bars, mon)
        all_fixed += f
        all_dyn += d
        covered.append(sym)

    print(f"[backtest] window={args.start}..{args.end} symbols_with_data={len(covered)}/{len(BASELINE_UNIVERSE)}")
    print(f"[backtest] covered: {covered}")
    fm, dm = _metrics(all_fixed), _metrics(all_dyn)
    print("\n=== FIXED (-0.5%) ===")
    for k, v in fm.items():
        print(f"  {k}: {v}")
    print("\n=== DYNAMIC (ATR-adaptive + confirmation) ===")
    for k, v in dm.items():
        print(f"  {k}: {v}")
    if fm.get("trades") and dm.get("trades"):
        print("\n=== DELTA (dynamic - fixed) ===")
        for k in ("net_return_total", "sharpe", "max_drawdown", "protective_exits",
                  "mean_mae_bps"):
            print(f"  {k}: {dm[k] - fm[k]:+.4f}")


if __name__ == "__main__":
    main()
