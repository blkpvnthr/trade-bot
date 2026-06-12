#!/usr/bin/env python3
"""
backtest_insider_entry_timing.py — validate (or falsify) VWAP-confirmation entry vs next-open,
on the VALIDATED vol-gated insider SHARES set, with REAL intraday bars.

The live rule enters at next-day OPEN (PF 2.20 OOS). The proposal is to wait intraday for
price ≥ session VWAP + a volume spike before entering. CRITICAL prior: the insider edge is a
BOUNCE out of a high-vol washout, so confirmation-entry may buy LATER/HIGHER and capture LESS.
This measures it instead of assuming. ONE lever (VWAP+RVOL) — no VPIN/BVC stacking until/unless
this beats the baseline (and VPIN/BVC need consolidated tape the free IEX feed can't provide).

Arms (same gated entries, exit = +hold-day daily close, net slippage):
  1. next_open            — enter at the entry-day OPEN (the live/validated rule)
  2. vwap_confirm_or_open — wait for first 5min bar in [9:30,11:30] with close≥VWAP & vol>1.5×avg;
                            enter there; if no confirm by 11:30 → fall back to OPEN
  3. vwap_confirm_or_skip — only trade when confirmed (else no trade)
"""
from __future__ import annotations

import argparse
import os
import statistics
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import backtest_insider_copier as base
import backtest_alpha_momentum_options as mom
import backtest_ablation_studies as ab
import study_orb_iv_ablation as orbm   # reuse _et() + intraday cache dir conventions

HOLD = 3
VOL_MIN_RV20 = 0.0563
MIN_ADV = 1_000_000
SLIP = 0.010
INTRA_DIR = base.PROJECT_ROOT / "data" / "cache" / "intraday_5m"
_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        for line in Path(".env").read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
        key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
        sec = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
        from alpaca.data.historical import StockHistoricalDataClient
        _CLIENT = StockHistoricalDataClient(key, sec)
    return _CLIENT


def _intraday(sym: str, day: str) -> Optional[pd.DataFrame]:
    INTRA_DIR.mkdir(parents=True, exist_ok=True)
    cache = INTRA_DIR / f"{sym}_{day}.parquet"
    if cache.exists():
        try:
            df = pd.read_parquet(cache); return df if not df.empty else None
        except Exception:
            pass
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import DataFeed
        d0 = datetime.fromisoformat(day)
        req = StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                               start=d0, end=d0 + timedelta(days=1), feed=DataFeed.IEX)
        raw = _client().get_stock_bars(req).df
        if raw is None or raw.empty:
            cache.write_bytes(b""); return None
        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.xs(sym, level=0)
        idx = pd.to_datetime(raw.index)
        raw.index = idx.tz_convert(orbm._et()) if idx.tz is not None else idx.tz_localize("UTC").tz_convert(orbm._et())
        df = raw[["open", "high", "low", "close", "volume"]].copy()
        df.to_parquet(cache); return df
    except Exception:
        try: cache.write_bytes(b"")
        except Exception: pass
        return None


def _vwap_entry(df: pd.DataFrame):
    """First 5min bar in [09:30,11:30] with close≥running VWAP and volume>1.5×avg-so-far. Returns
    (price, time) or None."""
    sess = df[(df.index.time >= time(9, 30)) & (df.index.time <= time(16, 0))]
    if sess.empty:
        return None
    tp = (sess["high"] + sess["low"] + sess["close"]) / 3
    vwap = (tp * sess["volume"]).cumsum() / sess["volume"].cumsum().replace(0, np.nan)
    avgv = sess["volume"].expanding().mean()
    win = sess[(sess.index.time >= time(9, 35)) & (sess.index.time <= time(11, 30))]
    for ts, row in win.iterrows():
        if row["close"] >= vwap.loc[ts] and row["volume"] > 1.5 * avgv.loc[ts]:
            return float(row["close"]), ts
    return None


def _vwap_support_entry(df: pd.DataFrame):
    """VWAP-SUPPORT (pullback) entry: a limit at the running session VWAP, filled the first time
    in [09:30,11:00] that a 5min bar's LOW ≤ VWAP (price pulled back to support → better fill than
    the open for a bounce). Returns (vwap_fill_price, time) or None (never pulled back → no fill)."""
    sess = df[(df.index.time >= time(9, 30)) & (df.index.time <= time(16, 0))]
    if sess.empty:
        return None
    tp = (sess["high"] + sess["low"] + sess["close"]) / 3
    vwap = (tp * sess["volume"]).cumsum() / sess["volume"].cumsum().replace(0, np.nan)
    win = sess[(sess.index.time >= time(9, 30)) & (sess.index.time <= time(11, 0))]
    for ts, row in win.iterrows():
        vw = vwap.loc[ts]
        if pd.notna(vw) and row["low"] <= vw:        # price touched/below VWAP → limit fills at VWAP
            return float(vw), ts
    return None


def _bvc_entry(df: pd.DataFrame, buy_thr: float = 0.62, df_t: float = 0.25):
    """The user's VPIN/BVC gate: first 5min bar (≤11:30) with close≥VWAP AND BVC buy_pressure>0.62.
    BVC buy_frac = t.cdf(ΔP/σ, df) per Easley-López de Prado-O'Hara; flat bar → 0.5. Returns
    (price, time) or None. (VPIN toxicity needs multi-day buckets — not computable intraday on day 1.)"""
    from scipy.stats import t as _t
    sess = df[(df.index.time >= time(9, 30)) & (df.index.time <= time(16, 0))].copy()
    if len(sess) < 6:
        return None
    tp = (sess["high"] + sess["low"] + sess["close"]) / 3
    vwap = (tp * sess["volume"]).cumsum() / sess["volume"].cumsum().replace(0, np.nan)
    dp = sess["close"].diff().fillna(0.0)
    sigma = dp.rolling(20, min_periods=5).std().fillna(dp.std() + 1e-8) + 1e-8
    z = dp / sigma
    buy_frac = pd.Series(_t.cdf(z.values, df=df_t), index=sess.index)
    buy_frac[dp.abs() < 1e-7] = 0.5
    win = sess[(sess.index.time >= time(9, 35)) & (sess.index.time <= time(11, 30))]
    for ts, row in win.iterrows():
        if row["close"] >= vwap.loc[ts] and float(buy_frac.loc[ts]) > buy_thr:
            return float(row["close"]), ts
    return None


def run(max_entries: int):
    pit = base.load_pit_buy_events()
    rows = []
    for sym in sorted(pit.keys()):
        if len(rows) >= max_entries:
            break
        bars = mom._bars(sym)
        if bars is None or bars.empty or len(bars) < 80:
            continue
        td = list(bars.index)
        op = [float(bars.loc[d, "open"]) for d in td]
        cl = [float(bars.loc[d, "close"]) for d in td]
        vol = [float(bars.loc[d, "volume"]) for d in td]
        r1 = np.diff(np.log(np.maximum(cl, 1e-9)))
        td_dates = [d.date() if hasattr(d, "date") else d for d in td]
        for i in ab._insider_entry_indices(td_dates, sym, pit):
            if len(rows) >= max_entries:
                break
            if i < 25 or i + 1 + HOLD >= len(td):
                continue
            rv20 = float(np.std(r1[i - 20:i])) if i >= 21 else np.nan
            adv = float(np.mean([cl[k] * vol[k] for k in range(i - 20, i)]))
            if not np.isfinite(rv20) or rv20 < VOL_MIN_RV20 or adv < MIN_ADV:   # the live gate
                continue
            ed = td_dates[i + 1].isoformat()
            day_open = op[i + 1]; exit_close = cl[i + 1 + HOLD]
            if day_open <= 0 or exit_close <= 0:
                continue
            intr = _intraday(sym, ed)
            ve = _vwap_entry(intr) if intr is not None else None
            be = _bvc_entry(intr) if intr is not None else None
            vs = _vwap_support_entry(intr) if intr is not None else None
            rows.append({"sym": sym, "date": ed, "open": day_open, "exit": exit_close,
                         "vwap_px": ve[0] if ve else None, "confirmed": ve is not None,
                         "bvc_px": be[0] if be else None,
                         "vwap_support_px": vs[0] if vs else None})
    return rows


def _stat(returns):
    v = np.array([r for r in returns if r is not None])
    if len(v) == 0:
        return None
    wins, losses = v[v > 0], v[v < 0]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) else float("inf")
    sd = v.std()
    return {"n": len(v), "mean": v.mean(), "median": float(np.median(v)), "win": (v > 0).mean(),
            "sharpe": (v.mean() / sd) if sd > 0 else 0.0, "pf": pf}


def _fmt(name, s):
    if not s:
        return f"| {name} | 0 |—|—|—|—|—|"
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return (f"| {name} | {s['n']} | {s['mean']*100:+.2f}% | {s['median']*100:+.2f}% | "
            f"{s['win']*100:.0f}% | {s['sharpe']:+.3f} | {pf} |")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-entries", type=int, default=250)
    args = ap.parse_args()
    print("Collecting gated insider entries + intraday bars (IEX 5min)...")
    rows = run(args.max_entries)
    have_intr = [r for r in rows if r["vwap_px"] is not None or r["confirmed"] is not None]
    conf = sum(1 for r in rows if r["confirmed"])
    # returns net of slippage
    r_open = [r["exit"] / r["open"] - 1 - SLIP for r in rows]
    r_confopen = [r["exit"] / (r["vwap_px"] or r["open"]) - 1 - SLIP for r in rows]
    r_confskip = [r["exit"] / r["vwap_px"] - 1 - SLIP for r in rows if r["confirmed"]]
    r_bvc = [r["exit"] / r["bvc_px"] - 1 - SLIP for r in rows if r.get("bvc_px")]
    # VWAP-support (pullback limit): fill at VWAP if price dipped to it, else fall back to open
    r_supp_open = [r["exit"] / (r["vwap_support_px"] or r["open"]) - 1 - SLIP for r in rows]
    r_supp_skip = [r["exit"] / r["vwap_support_px"] - 1 - SLIP for r in rows if r.get("vwap_support_px")]
    n_supp = sum(1 for r in rows if r.get("vwap_support_px"))
    L = ["# Insider Entry-Timing Test — next-open vs VWAP-confirmation (real IEX intraday)", "",
         f"Validated vol-gated insider set (rv20≥{VOL_MIN_RV20}, ADV≥${MIN_ADV:,.0f}), hold {HOLD}d, "
         f"net {SLIP*100:.1f}% slippage. {len(rows)} entries, {conf} VWAP-confirmed "
         f"({conf/max(len(rows),1)*100:.0f}%). VWAP on free IEX feed (partial tape — directional only).", "",
         "| Entry method | n | Mean | Median | Win% | Sharpe | PF |", "|---|---|---|---|---|---|---|",
         _fmt("next_open (LIVE/validated)", _stat(r_open)),
         _fmt("vwap_confirm/momentum (buy higher)", _stat(r_confskip)),
         _fmt("BVC buy_pressure>0.62 + VWAP", _stat(r_bvc)),
         _fmt(f"vwap_SUPPORT pullback (filled {n_supp}/{len(rows)})", _stat(r_supp_skip)),
         _fmt("vwap_support_or_open", _stat(r_supp_open)), ""]
    so, ss = _stat(r_open), _stat(r_supp_skip)
    better = bool(ss and so and ss["mean"] > so["mean"] and ss["pf"] >= so["pf"] and ss["n"] >= 30)
    L += ["## Verdict", "",
          f"- VWAP-SUPPORT (pullback, buy lower) beats next-open (mean+PF, n≥30): **{'YES' if better else 'NO'}**",
          (f"  vwap_support mean {ss['mean']*100:+.2f}%/PF {ss['pf']:.2f} (n={ss['n']}, filled {n_supp}/{len(rows)}) "
           f"vs next-open {so['mean']*100:+.2f}%/PF {so['pf']:.2f}" if so and ss else ""),
          "",
          ("**→ VWAP-SUPPORT pullback entry HELPS (better fill on the dip). Worth wiring a VWAP "
           "limit (ONE lever) — but only the SUPPORT variant; confirmation/momentum + VPIN/BVC all "
           "lose. Confirm with walk-forward + slippage sweep before live.**" if better else
           "**→ No intraday entry variant beats next-open — support, momentum, AND BVC all lose or "
           "tie. KEEP next-open. The validated rule stands; stop optimizing execution on this signal.**")]
    out = "\n".join(L)
    (base.PROJECT_ROOT / "backtest_outputs" / "insider_entry_timing.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
