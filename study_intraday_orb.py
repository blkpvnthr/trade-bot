#!/usr/bin/env python3
"""
study_intraday_orb.py — THE correct test of the sweep-reclaim ORB: real INTRADAY option entry.

Fixes the measurement mismatch (underlying=intraday pattern, option=daily-chain proxy). Here the
option is entered at the ACTUAL engulfing-candle timestamp using REAL Alpaca 5-min option bars
(2024+), and exited at the exit-day close. Compares two ENTRY models on identical liquid names:

  1. ATM baseline      — enter the ATM call at 09:45 ET (right after the opening range), every day
  2. Sweep-Reclaim ORB — enter the SAME ATM call at the engulf-candle close, only when the
                         sweep→pullback→bullish-engulf pattern fires on the underlying

Real: underlying 1-min (Alpaca) for the pattern + engulf timestamp; ATM contract OCC from the AV
daily chain (δ0.45-0.65, spread≤10%); option entry/exit from Alpaca 5-min OPTION bars. Entry at
ASK (bar close + ½ real daily-chain spread), exit at BID (− ½ spread), commission included.

Reports per arm: mean, median, Sharpe, win, max drawdown, MAE, MFE, trade count.
Usage: python study_intraday_orb.py --max-entries 150
"""
from __future__ import annotations

import argparse
import os
import statistics
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import backtest_insider_copier as base
import backtest_alpha_momentum_options as mom
import backtest_ablation_studies as ab
import av_historical_options as avo
import study_atm_filters as atm
import study_orb_iv_ablation as orbm
import alpha_call_engine as ace

HOLD_FETCH = 10                 # option-bar fetch window (reuses the 10-day cache)
HORIZONS = [0, 1, 2, 3]         # matched-horizon exits in TRADING days (0 = same-day close)
COMMISSION = 0.65
DTE_LO, DTE_HI = 21, 49
ALPACA_OPT_START = "2024-02-01"        # Alpaca option data availability
OPTBARS_DIR = base.PROJECT_ROOT / "data" / "cache" / "opt_bars_5m"
_OCLIENT = None


def _opt_client():
    global _OCLIENT
    if _OCLIENT is None:
        for line in Path(".env").read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
        key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY_ID")
        sec = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET_KEY")
        from alpaca.data.historical.option import OptionHistoricalDataClient
        _OCLIENT = OptionHistoricalDataClient(key, sec)
    return _OCLIENT


def opt_bars(occ: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Real Alpaca 5-min option bars for `occ` over [start_date, end_date], ET-indexed. Cached."""
    OPTBARS_DIR.mkdir(parents=True, exist_ok=True)
    cache = OPTBARS_DIR / f"{occ}_{start_date}_{end_date}.parquet"
    if cache.exists():
        try:
            df = pd.read_parquet(cache)
            return df if not df.empty else None
        except Exception:
            pass
    try:
        from alpaca.data.requests import OptionBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        req = OptionBarsRequest(symbol_or_symbols=occ, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                                start=datetime.fromisoformat(start_date),
                                end=datetime.fromisoformat(end_date) + timedelta(days=1))
        raw = _opt_client().get_option_bars(req).df
        if raw is None or raw.empty:
            cache.write_bytes(b""); return None
        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.xs(occ, level=0)
        idx = pd.to_datetime(raw.index)
        raw.index = idx.tz_convert(orbm._et()) if idx.tz is not None else idx.tz_localize("UTC").tz_convert(orbm._et())
        df = raw[["open", "high", "low", "close"]].copy()
        df.to_parquet(cache)
        return df
    except Exception:
        try:
            cache.write_bytes(b"")
        except Exception:
            pass
        return None


def _bar_at(df: pd.DataFrame, ts) -> Optional[pd.Series]:
    """The 5-min bar at slot `ts` (exact), else the first bar at/after `ts` that day."""
    same_day = df[df.index.date == ts.date()]
    if same_day.empty:
        return None
    exact = same_day[same_day.index == ts]
    if not exact.empty:
        return exact.iloc[0]
    after = same_day[same_day.index >= ts]
    return after.iloc[0] if not after.empty else None


def _trade(df, entry_ts, exit_date, spread_pct):
    """Build one trade from option bars: enter ASK at entry_ts, exit BID at exit-day close.
    Returns (ret, mae, mfe) or None."""
    eb = _bar_at(df, entry_ts)
    if eb is None or float(eb["close"]) <= 0:
        return None
    entry_ask = float(eb["close"]) * (1 + spread_pct / 2)
    exit_day = df[df.index.date == datetime.fromisoformat(exit_date).date()]
    if exit_day.empty:
        return None
    exit_bid = float(exit_day.iloc[-1]["close"]) * (1 - spread_pct / 2)
    comm = COMMISSION / 100.0
    ret = max(-1.0, (exit_bid - comm) / (entry_ask + comm) - 1.0)
    hold = df[(df.index > entry_ts) & (df.index.date <= datetime.fromisoformat(exit_date).date())]
    if hold.empty:
        mae = mfe = 0.0
    else:
        mae = float(hold["low"].min()) / entry_ask - 1.0
        mfe = float(hold["high"].max()) / entry_ask - 1.0
    return ret, mae, mfe


def run(max_entries: int) -> Dict[int, Dict[str, List[dict]]]:
    # results[horizon] = {"baseline": [...], "orb": [...]}
    results = {h: {"baseline": [], "orb": []} for h in HORIZONS}
    n_seen = 0
    for sym in atm.LIQUID:
        if n_seen >= max_entries:
            break
        bars = mom._bars(sym)
        if bars is None or bars.empty or len(bars) < 80:
            continue
        td = list(bars.index)
        close = {d: float(bars.loc[d, "close"]) for d in td}
        high = {d: float(bars.loc[d, "high"]) for d in td}
        vol = {d: float(bars.loc[d, "volume"]) for d in td}
        for i in ab._momentum_entry_indices(td, close, high, vol):
            if n_seen >= max_entries:
                break
            if i + 1 + HOLD_FETCH >= len(td):
                continue
            entry_dt = td[i + 1]
            ed = entry_dt.isoformat()
            if ed < ALPACA_OPT_START:                 # Alpaca option data starts 2024-02
                continue
            chain = avo.get_chain(sym, ed, calls_only=True)
            c = atm._atm_in_bucket(chain, DTE_LO, DTE_HI)
            if not c:
                continue
            occ = c["symbol"]; spread_pct = (c["ask_price"] - c["bid_price"]) / max(c["mid"], 1e-6)
            odf = opt_bars(occ, ed, td[i + 1 + HOLD_FETCH].isoformat())   # 10-day window (cache reuse)
            if odf is None or odf.empty:
                continue
            n_seen += 1
            et = orbm._et()
            base_ts = pd.Timestamp(entry_dt.year, entry_dt.month, entry_dt.day, 9, 45, tz=et)
            u = _underlying_intraday(sym, ed)
            sig, ets = False, None
            if u is not None:
                sig, _px, _status, ets = ace._orb_sweep_reclaim_full(u, 15, 5, True)
            for h in HORIZONS:
                xd = td[i + 1 + h].isoformat()        # same-day (h=0) .. 3-day exit close
                tb = _trade(odf, base_ts, xd, spread_pct)
                if tb:
                    results[h]["baseline"].append({"sym": sym, "date": ed, "ret": tb[0],
                                                   "mae": tb[1], "mfe": tb[2]})
                if sig and ets is not None:
                    to = _trade(odf, ets, xd, spread_pct)
                    if to:
                        results[h]["orb"].append({"sym": sym, "date": ed, "ret": to[0],
                                                  "mae": to[1], "mfe": to[2]})
    return results


def _underlying_intraday(sym: str, date: str) -> Optional[pd.DataFrame]:
    """Reuse the cached OHLC 1-min underlying bars (fetched by study_orb_iv_ablation)."""
    cache = orbm.INTRADAY_DIR / f"{sym}_{date}.parquet"
    if not cache.exists():
        orbm.orb_breakout(sym, date)        # triggers a fetch+cache
    try:
        df = pd.read_parquet(cache)
        return df if (not df.empty and "open" in df.columns) else None
    except Exception:
        return None


def _max_dd(rows):
    eq = peak = 1.0; mdd = 0.0
    for r in sorted(rows, key=lambda x: x["date"]):
        eq *= (1.0 + 0.20 * r["ret"])
        if eq <= 0:
            return -1.0
        peak = max(peak, eq); mdd = min(mdd, eq / peak - 1.0)
    return mdd


def _stat(rows):
    if not rows:
        return None
    v = [r["ret"] for r in rows]; n = len(v); m = statistics.mean(v)
    sd = statistics.pstdev(v) if n > 1 else 0.0
    return {"n": n, "mean": m, "median": statistics.median(v),
            "sharpe": (m / sd) if sd > 0 else 0.0, "win": sum(1 for x in v if x > 0) / n,
            "mdd": _max_dd(rows), "mae": statistics.mean([r["mae"] for r in rows]),
            "mfe": statistics.mean([r["mfe"] for r in rows])}


def report(results):
    L = ["# Matched-Horizon ORB Test — ATM baseline vs Sweep-Reclaim ORB (real intraday option bars)", "",
         "THE last fair ORB test: an intraday-structure entry deserves a SHORT exit, not a 10-day swing. "
         "REAL Alpaca 5-min OPTION bars (2024+). Baseline enters the ATM call at 09:45 ET; Sweep-Reclaim "
         "at the engulf-candle close on pattern days. Same liquid names, ATM δ0.45-0.65, spread≤10%, "
         f"ASK in / BID out (½ real spread) + ${COMMISSION}/side. Exit = close at the horizon.", ""]
    label = {0: "same-day", 1: "1-day", 2: "2-day", 3: "3-day"}
    for h in HORIZONS:
        L += [f"## Exit horizon: {label[h]}", "",
              "| Arm | n | Mean | Median | Sharpe | Win% | MaxDD | avg MAE | avg MFE |",
              "|---|---|---|---|---|---|---|---|---|"]
        for name, key in [("ATM baseline (09:45)", "baseline"), ("Sweep-Reclaim ORB", "orb")]:
            s = _stat(results[h][key])
            if not s:
                L.append(f"| {name} | 0 |—|—|—|—|—|—|—|"); continue
            L.append(f"| **{name}** | {s['n']} | {s['mean']*100:+.1f}% | **{s['median']*100:+.1f}%** | "
                     f"{s['sharpe']:+.3f} | {s['win']*100:.0f}% | {s['mdd']*100:.0f}% | "
                     f"{s['mae']*100:+.0f}% | {s['mfe']*100:+.0f}% |")
        L.append("")
    L += ["_If Sweep-Reclaim beats the baseline at SHORT horizons (median+win+Sharpe), the pattern is alive "
          "(intraday entry needs intraday exit). If it loses at every horizon → archive ORB research._"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-entries", type=int, default=250)
    args = ap.parse_args()
    print(f"Matched-horizon ORB test (real Alpaca option bars, 2024+, max {args.max_entries})...")
    results = run(args.max_entries)
    md = report(results)
    ab.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ab.OUT_DIR / "matched_horizon_orb.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
