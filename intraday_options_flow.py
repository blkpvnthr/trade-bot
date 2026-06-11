#!/usr/bin/env python3
"""
intraday_options_flow.py — does UNUSUAL intraday directional options flow (vs a name's own
intraday baseline) predict the underlying's next move? Uses Databento OPRA TCBBO (trade + NBBO
-before → aggressor side) for the options tape and Databento equities ohlcv-1m for the underlying.

Per session, per BAR (default 5min): directional pressure = (call buy−sell) − (put buy−sell),
size-weighted, aggressor-classified by trade price vs the NBBO mid right before the trade. Each
bar's pressure is z-scored vs the session's TRAILING intraday baseline (no lookahead). Target =
underlying close→close return over the next FWD minutes. Pools across sessions; reports the
forward-return quintile spread by flow-z and the per-session sign (stability).

NOT a production signal — research only, same multi-regime/sign-stability bar as everything else.
Costs: OPRA covered by plan (verify get_cost before large pulls). Reads DATABENTO_API from .env.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import databento as db

ROOT = Path(__file__).resolve().parent
EQUITY_DATASET = "DBEQ.BASIC"   # consolidated US equities for the underlying 1m bars


def _key():
    for line in open(ROOT / ".env"):
        if line.startswith("DATABENTO_API="):
            return line.split("=", 1)[1].strip()
        if line.startswith("DATABENTO_API_KEY="):
            return line.split("=", 1)[1].strip()
    return ""


def _client():
    return db.Historical(_key())


def _parse_cp(sym: str) -> str:
    return sym.strip()[-15:][6]   # 'C' or 'P'


def flow_bars(c, sym: str, date: str, bar: str = "5min") -> pd.DataFrame:
    """Aggressor-classified directional options pressure per bar for one RTH session.
    Returns DataFrame indexed by bar-end with columns: pressure, total_vol."""
    df = c.timeseries.get_range(dataset="OPRA.PILLAR", schema="tcbbo", symbols=f"{sym}.OPT",
                                stype_in="parent", start=f"{date}T13:30", end=f"{date}T20:00").to_df()
    if df.empty:
        return pd.DataFrame()
    df = df[df["price"] > 0].copy()
    mid = (df["bid_px_00"] + df["ask_px_00"]) / 2.0
    sign = np.where(df["price"] > mid, 1.0, np.where(df["price"] < mid, -1.0, 0.0))  # buy=+1 sell=-1
    cp = df["symbol"].map(_parse_cp)
    # directional pressure: + when calls are bought / puts are sold; − when puts bought / calls sold
    pol = np.where(cp.values == "C", 1.0, -1.0)
    df["press"] = sign * df["size"].values * pol
    df["vol"] = df["size"].values
    g = df.resample(bar).agg(pressure=("press", "sum"), total_vol=("vol", "sum"))
    return g


def underlying_1m(c, sym: str, date: str) -> pd.Series:
    """Underlying 1-min close (UTC) for the session. None on failure."""
    try:
        u = c.timeseries.get_range(dataset=EQUITY_DATASET, schema="ohlcv-1m", symbols=sym,
                                   stype_in="raw_symbol", start=f"{date}T13:30", end=f"{date}T20:00").to_df()
        if u.empty:
            return None
        return u["close"].astype(float)
    except Exception as e:
        print(f"  underlying {sym} {date}: {repr(e)[:90]}")
        return None


def session_events(c, sym: str, date: str, bar_min: int, fwd_min: int, base_win: int):
    """Per-bar events: intraday-baseline flow z-score + underlying forward return."""
    fb = flow_bars(c, sym, date, bar=f"{bar_min}min")
    if fb.empty or len(fb) < base_win + 3:
        return []
    px = underlying_1m(c, sym, date)
    if px is None or px.empty:
        return []
    # underlying price at each bar-end and fwd_min later (1-min grid, ffill)
    px = px.sort_index()
    rows = []
    pe = fb["pressure"].values
    for i in range(base_win, len(fb) - max(1, fwd_min // bar_min)):
        hist = pe[i - base_win:i]
        sd = hist.std()
        if sd <= 0:
            continue
        z = (pe[i] - hist.mean()) / sd        # unusual directional pressure vs intraday baseline
        t = fb.index[i]
        t_fwd = t + pd.Timedelta(minutes=fwd_min)
        try:
            p0 = float(px.asof(t)); p1 = float(px.asof(t_fwd))
        except Exception:
            continue
        if not (p0 > 0 and p1 > 0):
            continue
        rows.append({"date": date, "t": t, "z": float(z), "fwd_ret": p1 / p0 - 1.0})
    return rows


def _stat(arr):
    if not arr:
        return None
    v = np.array(arr, float)
    sd = v.std()
    return {"n": len(v), "mean": float(v.mean()), "win": float((v > 0).mean()),
            "sharpe": float(v.mean() / sd) if sd > 0 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--dates", default="2025-09-10,2026-06-08", help="comma list YYYY-MM-DD")
    ap.add_argument("--bar-min", type=int, default=5)
    ap.add_argument("--fwd-min", type=int, default=30)
    ap.add_argument("--base-win", type=int, default=6, help="trailing bars for intraday baseline")
    args = ap.parse_args()
    c = _client()
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    all_ev, per_session = [], []
    for d in dates:
        ev = session_events(c, args.symbol, d, args.bar_min, args.fwd_min, args.base_win)
        if not ev:
            print(f"  {d}: no events"); continue
        zs = np.array([e["z"] for e in ev]); rs = np.array([e["fwd_ret"] for e in ev])
        corr = float(np.corrcoef(zs, rs)[0, 1]) if zs.std() > 0 else 0.0
        per_session.append({"date": d, "n": len(ev), "corr": corr})
        all_ev.extend(ev)
        print(f"  {d}: {len(ev)} bars, corr(flow_z, fwd_ret)={corr:+.3f}")
    if len(all_ev) < 30:
        print("insufficient events"); return
    all_ev.sort(key=lambda e: e["z"])
    n = len(all_ev); qn = n // 5
    q1 = _stat([e["fwd_ret"] for e in all_ev[:qn]]); q5 = _stat([e["fwd_ret"] for e in all_ev[-qn:]])
    zs = np.array([e["z"] for e in all_ev]); rs = np.array([e["fwd_ret"] for e in all_ev])
    pooled_corr = float(np.corrcoef(zs, rs)[0, 1])
    spread = (q5["mean"] - q1["mean"]) * 1e4
    L = [f"# Intraday Options Flow → {args.symbol} next-{args.fwd_min}min move",
         f"{len(dates)} sessions, {args.bar_min}min bars, intraday baseline = trailing {args.base_win} bars. "
         f"Directional pressure = aggressor-classified (call buy−sell)−(put buy−sell). {len(all_ev)} bar-events.", "",
         f"Pooled corr(flow_z, fwd_ret) = **{pooled_corr:+.3f}**",
         f"High-flow quintile (Q5) next-{args.fwd_min}min: mean {q5['mean']*1e4:+.1f}bps win {q5['win']*100:.0f}% (n={q5['n']})",
         f"Low-flow quintile (Q1): mean {q1['mean']*1e4:+.1f}bps win {q1['win']*100:.0f}%",
         f"**Q5−Q1 spread: {spread:+.1f} bps**", "",
         "| Session | bars | corr(flow_z, fwd_ret) |", "|---|---|---|"]
    for s in per_session:
        L.append(f"| {s['date']} | {s['n']} | {s['corr']:+.3f} |")
    L += ["", "_Pilot: 1 symbol, few sessions — directional read only. Scale to many sessions/regimes + "
          "sign-stability before any belief. Intraday spread/fees not modeled; NOT for production._"]
    out = "\n".join(L)
    (ROOT / "backtest_outputs" / f"intraday_flow_{args.symbol}.md").write_text(out)
    print("\n" + out)


if __name__ == "__main__":
    main()
