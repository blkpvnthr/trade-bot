#!/usr/bin/env python3
"""
study_orb_iv_ablation.py — the 2×2 ablation: ATM / ATM+ORB / ATM+IV / ATM+ORB+IV.

Real liquid chains only (AV HISTORICAL_OPTIONS), ATM δ0.45-0.65, spread ≤10%, enter ASK /
exit BID + commission. ORB requires INTRADAY underlying bars (Alpaca 1-min) to build the
09:30-09:45 ET opening range and confirm the breakout on the ENTRY day — daily bars can't.
IV gate = contract IV ∈ [0.30, 0.50]. The option is priced at the daily chain; ORB/IV are
ENTRY FILTERS (which trades to take).

Reports per arm: mean, median, Sharpe, win rate, max drawdown, trade count.
Key question: does the IV gate still help AFTER ORB is applied?

Usage: python study_orb_iv_ablation.py --max-entries 200
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import backtest_insider_copier as base
import backtest_alpha_momentum_options as mom
import backtest_ablation_studies as ab
import av_historical_options as avo
import study_atm_filters as atm

HOLD = 10
COMMISSION = 0.65
IV_MIN, IV_MAX = 0.30, 0.50
DTE_LO, DTE_HI = 21, 49
INTRADAY_DIR = base.PROJECT_ROOT / "data" / "cache" / "intraday_1m"
_ET = None


def _et():
    global _ET
    if _ET is None:
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("America/New_York")
    return _ET


def _alpaca_client():
    for line in Path(".env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY_ID")
    sec = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET_KEY")
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(key, sec)


_CLIENT = None


def orb_breakout(sym: str, date: str) -> Optional[bool]:
    """True if the underlying broke above its 09:30-09:45 ET opening-range high later in the
    session on `date`. None if intraday data unavailable. Cached per (sym, date)."""
    global _CLIENT
    INTRADAY_DIR.mkdir(parents=True, exist_ok=True)
    cache = INTRADAY_DIR / f"{sym}_{date}.parquet"
    df = None
    if cache.exists():
        try:
            df = pd.read_parquet(cache)
        except Exception:
            df = None
    if df is None:
        try:
            if _CLIENT is None:
                _CLIENT = _alpaca_client()
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            from alpaca.data.enums import DataFeed
            start = datetime.fromisoformat(date)
            end = datetime.fromisoformat(date).replace(hour=23, minute=59)
            req = StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                                   start=start, end=end, feed=DataFeed.IEX)
            raw = _CLIENT.get_stock_bars(req).df
            if raw is None or raw.empty:
                cache.write_bytes(b"")  # mark as empty to avoid refetch
                return None
            if isinstance(raw.index, pd.MultiIndex):
                raw = raw.xs(sym, level=0)
            df = raw[["open", "high", "low", "close"]].copy()   # OHLC — engulfing needs open
            df.to_parquet(cache)
        except Exception:
            return None
    if df is None or df.empty or "open" not in df.columns:
        return None
    import alpha_call_engine as ace
    sig, _entry, status = ace.detect_orb_sweep_reclaim(df, orb_minutes=15, candle_minutes=5,
                                                       require_reclaim=True)
    if status in ("no_bars", "no_session_open_bars", "error"):
        return None                      # no usable intraday data
    return bool(sig)                     # sweep→pullback→bullish-engulf occurred


def run(max_entries: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sym in atm.LIQUID:
        if len(rows) >= max_entries:
            break
        bars = mom._bars(sym)
        if bars is None or bars.empty or len(bars) < 80:
            continue
        td = list(bars.index)
        close = {d: float(bars.loc[d, "close"]) for d in td}
        high = {d: float(bars.loc[d, "high"]) for d in td}
        vol = {d: float(bars.loc[d, "volume"]) for d in td}
        for i in ab._momentum_entry_indices(td, close, high, vol):
            if len(rows) >= max_entries:
                break
            entry_dt = td[i + 1]; exit_dt = td[i + 1 + HOLD]
            ed, xd = entry_dt.isoformat(), exit_dt.isoformat()
            spot = float(bars.loc[entry_dt, "open"]); S1 = float(bars.loc[exit_dt, "close"])
            if spot <= 0 or S1 <= 0:
                continue
            chain = avo.get_chain(sym, ed, calls_only=True)
            c = atm._atm_in_bucket(chain, DTE_LO, DTE_HI)
            if not c:
                continue
            ret = atm._realized(c, sym, xd, S1)
            if ret is None:
                continue
            orb = orb_breakout(sym, ed)            # True/False/None (None=no intraday data)
            iv = float(c.get("implied_volatility", 0))
            rows.append({"sym": sym, "date": ed, "ret": ret, "iv": iv,
                         "orb": orb, "iv_pass": IV_MIN <= iv <= IV_MAX})
    return rows


def _max_dd(returns_in_order, frac=0.20):
    eq = peak = 1.0; mdd = 0.0
    for r in returns_in_order:
        eq *= (1.0 + frac * r)
        if eq <= 0:
            return -1.0
        peak = max(peak, eq); mdd = min(mdd, eq / peak - 1.0)
    return mdd


def _stat(rows):
    v = [r["ret"] for r in rows]
    if not v:
        return None
    n = len(v); m = statistics.mean(v); sd = statistics.pstdev(v) if n > 1 else 0.0
    ordered = [r["ret"] for r in sorted(rows, key=lambda x: x["date"])]
    return {"n": n, "mean": m, "median": statistics.median(v),
            "sharpe": (m / sd) if sd > 0 else 0.0, "win": sum(1 for x in v if x > 0) / n,
            "mdd": _max_dd(ordered)}


def report(rows):
    orb_known = [r for r in rows if r["orb"] is not None]
    arms = {
        "ATM baseline": rows,
        "ATM + ORB": [r for r in orb_known if r["orb"]],
        "ATM + IV": [r for r in rows if r["iv_pass"]],
        "ATM + ORB + IV": [r for r in orb_known if r["orb"] and r["iv_pass"]],
    }
    L = ["# 2×2 Ablation — ATM / +ORB / +IV / +ORB+IV (real liquid chains)", "",
         f"{len(rows)} ATM trades, {len(orb_known)} with intraday ORB data. Real chains, "
         f"ATM δ0.45-0.65, spread≤10%, ask→bid+${COMMISSION}/side. ORB=underlying broke 09:30-09:45 "
         f"high after 09:45 on entry day (Alpaca 1-min). IV gate=[{IV_MIN},{IV_MAX}].", "",
         "| Arm | n | Mean | Median | Sharpe | Win% | MaxDD |",
         "|---|---|---|---|---|---|---|"]
    for name, sub in arms.items():
        s = _stat(sub)
        if not s:
            L.append(f"| {name} | 0 |—|—|—|—|—|"); continue
        L.append(f"| **{name}** | {s['n']} | {s['mean']*100:+.1f}% | **{s['median']*100:+.1f}%** | "
                 f"{s['sharpe']:+.3f} | {s['win']*100:.0f}% | {s['mdd']*100:.0f}% |")
    L += ["", "## Read",
          "- Judge by MEDIAN + win + MaxDD (means are tail-inflated on long calls).",
          "- KEY: does ATM+ORB+IV beat ATM+ORB? If yes, IV adds value AFTER the structural ORB gate.",
          "  If the ORB column alone already captures most of the median lift, IV is redundant (prefer ORB — not fitted).",
          "_Real fills; ORB is a date-filter (option still priced at the daily chain). Modest sample — robustness scan._"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-entries", type=int, default=200)
    args = ap.parse_args()
    print(f"2×2 ORB/IV ablation on real liquid chains (max {args.max_entries})...")
    rows = run(args.max_entries)
    md = report(rows)
    ab.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ab.OUT_DIR / "orb_iv_ablation.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
