"""Historical fixed(-0.5%) vs dynamic(ATR-adaptive) protective-exit comparison.

Leak-free replay on GENUINE cached 1-minute bars (alpaca_minute_cache/), resampled
to 5-minute RTH bars, across the full 30-symbol BASELINE universe. Both arms share
IDENTICAL entries, take-profit, max-hold, and costs; they differ ONLY in the
protective LOSS exit:

  * FIXED  : exit when unrealized return <= -loss_warning (-0.5%).
  * DYNAMIC: market_scan.dynamic_loss.DynamicLossMonitor -- ATR-adaptive soft stop
             + confirmed deterioration (the exact logic wired into the engine).

Entry (shared, long-only momentum proxy for the engine's long bias): enter at the
first RTH 5m close of a day only if the opening 3 bars show positive momentum and
price is above the opening-range high -- so the comparison runs on plausible
long setups, not blind daily opens. The EXIT comparison is entry-agnostic (both
arms share the entry); the proxy only shapes which days produce trades.

No look-ahead: ATR(14), VWAP, deterioration, and the entry filter at bar i use
only bars <= i; the exit decision at bar i fills at bar i's close. Round-trip
costs (BASELINE_COSTS) applied equally. Reports per-symbol + aggregate: net return
after costs, Sharpe, max drawdown, win rate, net expectancy, protective/premature
exits, capital utilization, trade frequency, mean MAE.

Usage: python backtest_dynamic_loss.py [--start 2024-04-01] [--end 2024-06-01]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from costs import BASELINE_COSTS
from market_scan.baseline import BASELINE_UNIVERSE
from market_scan.dynamic_loss import DynamicLossMonitor

CACHE = os.path.join(os.path.dirname(__file__), "alpaca_minute_cache")
RT_COST = BASELINE_COSTS.round_trip_cost_bps() / 10_000.0
TAKE_PROFIT = 0.012
MAX_HOLD_BARS = 24
WARNING = 0.005


def _load_5min(symbol, start, end):
    files = sorted(glob.glob(os.path.join(CACHE, f"{symbol}_*.csv")))
    if not files:
        return None
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, parse_dates=["timestamp"]))
        except Exception:
            pass
    if not frames:
        return None
    m = pd.concat(frames, ignore_index=True)
    m = m[(m["timestamp"] >= start) & (m["timestamp"] < end)].copy()
    if m.empty:
        return None
    m = m.set_index("timestamp").sort_index().between_time("13:30", "20:00")
    if m.empty:
        return None
    bars = m.resample("5min").agg({"open": "first", "high": "max", "low": "min",
                                   "close": "last", "volume": "sum"}).dropna(subset=["close"])
    if bars.empty:
        return None
    bars["day"] = bars.index.date
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    pv = tp * bars["volume"]
    bars["vwap"] = (pv.groupby(bars["day"]).cumsum()
                    / bars["volume"].groupby(bars["day"]).cumsum().replace(0, np.nan))
    prev_close = bars["close"].shift(1)
    tr = pd.concat([(bars["high"] - bars["low"]).abs(),
                    (bars["high"] - prev_close).abs(),
                    (bars["low"] - prev_close).abs()], axis=1).max(axis=1)
    bars["atr_pct"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / bars["close"]
    return bars


@dataclass
class Trade:
    ret: float
    mae: float
    loss_exit: bool
    hold_bars: int
    premature: bool       # a loss exit the price later recovered from (within horizon)


def _deteriorating(db, i):
    row = db.iloc[i]
    if i >= 1 and row["close"] < db.iloc[i - 1]["low"]:
        return True
    if not math.isnan(row["vwap"]) and row["close"] < row["vwap"]:
        return True
    if i >= 3:
        r3 = (row["close"] - db.iloc[i - 3]["close"]) / db.iloc[i - 3]["close"]
        if r3 < 0:
            return True
    return False


def _entry_ok(db):
    """Long momentum proxy: opening 3 bars positive + above opening-range high."""
    if len(db) < 6:
        return False
    o = float(db.iloc[0]["close"])
    mom = (float(db.iloc[2]["close"]) - o) / o if o else 0.0
    orh = float(db.iloc[:3]["high"].max())
    return mom > 0 and float(db.iloc[3]["close"]) > orh


def _simulate_symbol(symbol, bars, mon):
    fixed, dyn = [], []
    for _day, day_bars in bars.groupby("day"):
        db = day_bars.reset_index()
        n = len(db)
        if n < 6 or not _entry_ok(db):
            continue
        entry = float(db.iloc[3]["close"])        # enter after the opening-range confirmation
        if entry <= 0:
            continue
        mon.reset(symbol)
        plpc_path = [(float(db.iloc[i]["close"]) - entry) / entry for i in range(n)]
        mae = 0.0

        def _recovers_after(i):
            return any(plpc_path[j] >= 0 for j in range(i + 1, min(n, 4 + MAX_HOLD_BARS)))

        f_done = d_done = False
        for i in range(4, n):
            plpc = plpc_path[i]
            mae = min(mae, plpc)
            tp_hit = plpc >= TAKE_PROFIT
            last = (i == n - 1) or (i - 3 >= MAX_HOLD_BARS)
            if not f_done:
                if plpc <= -WARNING:
                    fixed.append(Trade(plpc - RT_COST, mae, True, i - 3, _recovers_after(i))); f_done = True
                elif tp_hit or last:
                    fixed.append(Trade(plpc - RT_COST, mae, False, i - 3, False)); f_done = True
            if not d_done:
                atr = db.iloc[i]["atr_pct"]
                atr = float(atr) if atr == atr else None
                dec = mon.observe(symbol, plpc=plpc, atr_pct=atr,
                                  deteriorating=_deteriorating(db, i),
                                  now=float(db.iloc[i]["timestamp"].timestamp()))
                if dec.should_exit:
                    dyn.append(Trade(plpc - RT_COST, mae, True, i - 3, _recovers_after(i))); d_done = True
                elif tp_hit or last:
                    dyn.append(Trade(plpc - RT_COST, mae, False, i - 3, False)); d_done = True
            if f_done and d_done:
                break
    return fixed, dyn


def _metrics(trades):
    if not trades:
        return {"trades": 0}
    r = np.array([t.ret for t in trades])
    cum = np.cumsum(r)
    dd = cum - np.maximum.accumulate(cum)
    wins = r[r > 0]
    losses = r[r <= 0]
    sharpe = (r.mean() / r.std() * math.sqrt(len(r))) if r.std() > 1e-12 else 0.0
    return {
        "trades": len(trades),
        "net_return_total": round(float(cum[-1]), 4),
        "net_return_mean_bps": round(float(r.mean() * 10_000), 2),
        "win_rate": round(float((r > 0).mean()), 3),
        "net_expectancy_bps": round(float(r.mean() * 10_000), 2),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(float(dd.min()), 4),
        "protective_exits": int(sum(1 for t in trades if t.loss_exit)),
        "premature_exits": int(sum(1 for t in trades if t.premature)),
        "avg_hold_bars": round(float(np.mean([t.hold_bars for t in trades])), 2),
        "capital_utilization": round(float(np.mean([t.hold_bars for t in trades]) / MAX_HOLD_BARS), 3),
        "mean_mae_bps": round(float(np.mean([t.mae for t in trades]) * 10_000), 2),
        "avg_win_bps": round(float(wins.mean() * 10_000) if len(wins) else 0.0, 2),
        "avg_loss_bps": round(float(losses.mean() * 10_000) if len(losses) else 0.0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-04-01")
    ap.add_argument("--end", default="2024-06-01")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    mon = DynamicLossMonitor(warning_pct=WARNING, adaptive_min_pct=0.005,
                             adaptive_max_pct=0.020, atr_multiplier=1.5,
                             confirmation_seconds=30.0, min_observations=3, hard_stop_pct=0.025)
    all_fixed, all_dyn, per_symbol, covered = [], [], {}, []
    for sym in BASELINE_UNIVERSE:
        bars = _load_5min(sym, args.start, args.end)
        if bars is None:
            continue
        f, d = _simulate_symbol(sym, bars, mon)
        all_fixed += f
        all_dyn += d
        covered.append(sym)
        per_symbol[sym] = {"fixed": _metrics(f), "dynamic": _metrics(d)}

    fm, dm = _metrics(all_fixed), _metrics(all_dyn)
    # Economic + risk acceptance gates (dynamic must not be worse on the core axes).
    gates = {}
    if fm.get("trades") and dm.get("trades"):
        gates = {
            "net_return_not_worse": dm["net_return_total"] >= fm["net_return_total"],
            "sharpe_not_worse": dm["sharpe"] >= fm["sharpe"],
            "drawdown_not_worse": dm["max_drawdown"] >= fm["max_drawdown"],  # less negative = better
            "fewer_or_equal_premature_exits": dm["premature_exits"] <= fm["premature_exits"],
            "turnover_not_higher": dm["protective_exits"] <= fm["protective_exits"],
        }
    passed = bool(gates) and all(gates.values())

    print(f"[backtest] window={args.start}..{args.end} symbols_with_data={len(covered)}/{len(BASELINE_UNIVERSE)}")
    print(f"[backtest] covered={covered}")
    print("\n=== AGGREGATE FIXED  ==="); [print(f"  {k}: {v}") for k, v in fm.items()]
    print("\n=== AGGREGATE DYNAMIC ==="); [print(f"  {k}: {v}") for k, v in dm.items()]
    print("\n=== PER-SYMBOL (net_total fixed -> dynamic | protective fixed->dyn) ===")
    for s in covered:
        pf, pd_ = per_symbol[s]["fixed"], per_symbol[s]["dynamic"]
        print(f"  {s:5} trades={pf.get('trades',0):3} net {pf.get('net_return_total',0):+.4f} -> "
              f"{pd_.get('net_return_total',0):+.4f} | protective {pf.get('protective_exits',0)}->"
              f"{pd_.get('protective_exits',0)} | premature {pf.get('premature_exits',0)}->{pd_.get('premature_exits',0)}")
    print("\n=== ACCEPTANCE GATES ===")
    for k, v in gates.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"\n[backtest] ACCEPTANCE: {'PASS -> dynamic recommended' if passed else 'FAIL -> retain fixed'}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"window": [args.start, args.end], "covered": covered,
                       "aggregate": {"fixed": fm, "dynamic": dm},
                       "per_symbol": per_symbol, "gates": gates, "passed": passed}, fh, indent=2)
        print(f"[backtest] wrote {args.json}")


if __name__ == "__main__":
    main()
