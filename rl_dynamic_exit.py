#!/usr/bin/env python3
"""
rl_dynamic_exit.py — Can an asymmetric exit beat the fixed 3:1 bracket? (real-bar test)
=======================================================================================

The fixed bracket (+3R target / −1R stop) pins profit_factor to the identity 3·w/(1−w), so
PF can only rise by raising the win rate. This module tests the other lever the literature
points to: a DYNAMIC, asymmetric exit (scale-out + trailing stop) that lets winners run past
3R and cuts losers, breaking that ceiling. It re-derives BOTH exits from the SAME entries on
the SAME REAL 1-minute bars (system-data/*/*_intraday_bars_1m.csv resampled to 5m) so the
comparison is apples-to-apples — no peak_R approximation.

Conservative path assumption: within a bar the LOW is checked before the HIGH (worst case for
a long), so neither policy is flattered. OFFLINE; local bars only; no network, no orders.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import rl_common as rc

WARMUP = 20
STRIDE = int(os.getenv("DYNEXIT_STRIDE", "6"))
HORIZON = int(os.getenv("DYNEXIT_HORIZON", "36"))      # forward 5m bars (~3h) before time-stop
RISK_PCT = float(os.getenv("DYNEXIT_RISK_PCT", "0.003"))
MAX_PER_SYMBOL = int(os.getenv("DYNEXIT_MAX_PER_SYMBOL", "60"))
MAX_SYMBOLS = int(os.getenv("DYNEXIT_MAX_SYMBOLS", "80"))


def _resample_5m(df):
    try:
        return df.resample("5min").agg({"open": "first", "high": "max", "low": "min",
                                        "close": "last", "volume": "sum"}).dropna()
    except Exception:
        return df


def fixed_bracket(entry: float, risk: float, fwd) -> float:
    """+3R target / −1R stop. Returns realized R (mark-to-last if neither hit)."""
    stop, target = entry - risk, entry + 3.0 * risk
    for _, b in fwd.iterrows():
        if float(b["low"]) <= stop:            # check stop first (conservative)
            return -1.0
        if float(b["high"]) >= target:
            return 3.0
    last = float(fwd["close"].iloc[-1]) if len(fwd) else entry
    return round((last - entry) / risk, 4)


def dynamic_exit(entry: float, risk: float, fwd) -> float:
    """Scale-out + trailing exit. Lets the runner exceed 3R; trails to lock gains.

    1/3 off at +1R (stop→breakeven), 1/3 off at +2R (stop→+1R), final 1/3 trails 1R below the
    peak. Returns the blended realized R (in R units). Conservative: stop checked before highs.
    """
    shares = 1.0
    realized = 0.0
    stop = entry - risk
    peak = entry
    took_1r = took_2r = False
    for _, b in fwd.iterrows():
        hi, lo = float(b["high"]), float(b["low"])
        # stop (incl. trailing) first
        if lo <= stop:
            realized += shares * ((stop - entry) / risk)
            shares = 0.0
            break
        peak = max(peak, hi)
        if not took_1r and hi >= entry + 1.0 * risk:
            realized += (1.0 / 3.0) * 1.0          # lock 1R on first third
            shares -= 1.0 / 3.0
            stop = max(stop, entry)                # → breakeven
            took_1r = True
        if not took_2r and hi >= entry + 2.0 * risk:
            realized += (1.0 / 3.0) * 2.0          # lock 2R on second third
            shares -= 1.0 / 3.0
            stop = max(stop, entry + 1.0 * risk)   # → +1R
            took_2r = True
        if took_2r:
            stop = max(stop, peak - risk)          # trail final third 1R below peak
    if shares > 0:                                  # time-stop: mark remaining to last close
        last = float(fwd["close"].iloc[-1]) if len(fwd) else entry
        realized += shares * ((last - entry) / risk)
    return round(realized, 4)


def _entries_from_file(path: str) -> List[Tuple[float, float, Any]]:
    """Yield (entry, risk, forward_5m_frame) tuples from one 1m bar file (real symbols)."""
    out: List[Tuple[float, float, Any]] = []
    try:
        import pandas as pd
        df = pd.read_csv(path)
        if not {"symbol", "timestamp", "open", "high", "low", "close", "volume"}.issubset(df.columns):
            return out
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])
        syms = [s for s in df["symbol"].astype(str).unique() if s and not s.isdigit()][:MAX_SYMBOLS]
        for sym in syms:
            g = df[df["symbol"].astype(str) == sym].sort_values("timestamp").set_index("timestamp")
            b5 = _resample_5m(g[["open", "high", "low", "close", "volume"]])
            if len(b5) < WARMUP + 2:
                continue
            kept = 0
            for i in range(WARMUP, len(b5) - 1, STRIDE):
                if kept >= MAX_PER_SYMBOL:
                    break
                entry = float(b5["close"].iloc[i])
                if entry <= 0:
                    continue
                out.append((entry, max(entry * RISK_PCT, 1e-6), b5.iloc[i + 1:i + 1 + HORIZON]))
                kept += 1
    except Exception:
        return out
    return out


def evaluate(train_frac: float = 0.7) -> Dict[str, Any]:
    files = sorted(glob.glob(str(rc.PROJECT_ROOT / "system-data" / "*" / "*_intraday_bars_1m.csv")))
    entries: List[Tuple[float, float, Any]] = []
    for f in files:
        entries.extend(_entries_from_file(f))
    if not entries:
        return {"error": "no 1m bar files / entries"}
    fixed = [fixed_bracket(e, r, fwd) for e, r, fwd in entries]
    dyn = [dynamic_exit(e, r, fwd) for e, r, fwd in entries]
    k = max(1, int(len(entries) * train_frac))

    def report(rs: List[float]) -> Dict[str, Any]:
        m = rc.compute_metrics(rs)
        return {"trades": m["trades"], "win_rate": m["win_rate"], "profit_factor": m["profit_factor"],
                "expectancy": m["expectancy"], "apr": m["apr"], "max_drawdown_R": m["max_drawdown"]}

    return {"entries": len(entries),
            "fixed_3R": {"in_sample": report(fixed[:k]), "out_of_sample": report(fixed[k:])},
            "dynamic": {"in_sample": report(dyn[:k]), "out_of_sample": report(dyn[k:])}}


if __name__ == "__main__":
    res = evaluate()
    if "error" in res:
        print(res["error"]); raise SystemExit(0)
    print(f"Dynamic vs fixed-3:1 exit on REAL 1m bars — {res['entries']} entries")
    cols = ["trades", "win_rate", "profit_factor", "expectancy", "apr", "max_drawdown_R"]
    print(f"\n{'policy/split':22s}" + "".join(f"{c:>16s}" for c in cols))
    for pol in ("fixed_3R", "dynamic"):
        for split in ("in_sample", "out_of_sample"):
            m = res[pol][split]
            print(f"{pol+'/'+split:22s}" + "".join(f"{m[c]:>16}" for c in cols))
    rc.write_json(rc.DAILY_OUTPUTS / rc.data_tag() / "rl_dynamic_exit.json", res)
