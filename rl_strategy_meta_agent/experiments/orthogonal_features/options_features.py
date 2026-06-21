"""Leak-free options / IV-structure features aligned to the (symbol, bar) panel.

Options markets describe how participants are positioned BEFORE information arrives --
a different channel from price (what happened) and earnings timing (when info comes).
Source: Alpha Vantage HISTORICAL_OPTIONS (as-of-date full chain with IV + greeks + OI,
via av_options_adapter._av_get). We fetch one chain per (symbol, trading date), compute
a compact daily feature row, and align it LEAK-FREE: a bar on date D uses only the chain
of the most recent date STRICTLY BEFORE D (prior session EOD, known before D's open).

ATM is found by delta (|delta|~0.5), so no spot is needed. Tenors are picked by DTE.
Research only: read-only market data, no broker/order/live path.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

TRADER_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TRADER_ROOT))

CACHE = TRADER_ROOT / "backtest_cache" / "avopt_features"
CACHE.mkdir(parents=True, exist_ok=True)
DAILY_COLS = ["iv7", "iv30", "term_slope", "skew_25d", "pc_oi_ratio", "total_oi"]


def _atm_iv(rows, target_dte):
    """ATM IV (avg call+put, |delta|~0.5) at the expiration closest to target_dte days."""
    by_exp = {}
    for r in rows:
        dte = (pd.Timestamp(r["expiration"]) - pd.Timestamp(r["date"])).days
        if dte >= 1:
            by_exp.setdefault(dte, []).append(r)
    if not by_exp:
        return np.nan, None
    dte = min(by_exp, key=lambda d: abs(d - target_dte))
    leg = by_exp[dte]
    ivs = []
    for typ, tgt in (("call", 0.5), ("put", -0.5)):
        cands = [r for r in leg if str(r.get("type")).lower() == typ and r.get("delta") not in (None, "")]
        if cands:
            atm = min(cands, key=lambda r: abs(float(r["delta"]) - tgt))
            iv = float(atm.get("implied_volatility") or 0)
            if 0 < iv < 5:
                ivs.append(iv)
    return (float(np.mean(ivs)) if ivs else np.nan), dte


def _skew_25d(rows):
    """25-delta put IV minus 25-delta call IV at the ~30D expiration (risk-reversal sign)."""
    by_exp = {}
    for r in rows:
        dte = (pd.Timestamp(r["expiration"]) - pd.Timestamp(r["date"])).days
        if dte >= 1:
            by_exp.setdefault(dte, []).append(r)
    if not by_exp:
        return np.nan
    leg = by_exp[min(by_exp, key=lambda d: abs(d - 30))]
    def wing(typ, tgt):
        cands = [r for r in leg if str(r.get("type")).lower() == typ and r.get("delta") not in (None, "")]
        if not cands:
            return np.nan
        c = min(cands, key=lambda r: abs(float(r["delta"]) - tgt))
        iv = float(c.get("implied_volatility") or 0)
        return iv if 0 < iv < 5 else np.nan
    return wing("put", -0.25) - wing("call", 0.25)


def _daily_features(rows):
    if not rows:
        return None
    iv7, _ = _atm_iv(rows, 7)
    iv30, _ = _atm_iv(rows, 30)
    call_oi = sum(float(r.get("open_interest") or 0) for r in rows if str(r.get("type")).lower() == "call")
    put_oi = sum(float(r.get("open_interest") or 0) for r in rows if str(r.get("type")).lower() == "put")
    return {"iv7": iv7, "iv30": iv30, "term_slope": (iv30 - iv7) if np.isfinite(iv7) and np.isfinite(iv30) else np.nan,
            "skew_25d": _skew_25d(rows), "pc_oi_ratio": put_oi / (call_oi + 1.0), "total_oi": call_oi + put_oi}


def _fetch_one(symbol, ds):
    from av_options_adapter import _av_get
    try:
        rows = _av_get({"function": "HISTORICAL_OPTIONS", "symbol": symbol, "date": ds}).get("data") or []
    except Exception:
        return ds, None
    return ds, _daily_features(rows)


def backfill(symbols, start, end, workers=8, force=False):
    """Fetch + cache compact daily option features per symbol over [start, end]."""
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]
    counts = {}
    for s in symbols:
        cf = CACHE / f"{s}.parquet"
        if cf.exists() and not force:
            counts[s] = len(pd.read_parquet(cf)); continue
        recs = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_fetch_one, s, d): d for d in dates}
            for fu in as_completed(futs):
                ds, feat = fu.result()
                if feat:
                    recs[ds] = feat
        df = pd.DataFrame.from_dict(recs, orient="index").sort_index()
        df.index = pd.to_datetime(df.index)
        if not df.empty:
            df.to_parquet(cf)
        counts[s] = len(df)
    return counts


def load_features(symbols) -> dict:
    """sym -> DataFrame(index=date, cols=DAILY + iv_rank/iv_percentile/oi_change_1d/5d)."""
    out = {}
    for s in symbols:
        cf = CACHE / f"{s}.parquet"
        if not cf.exists():
            out[s] = pd.DataFrame(); continue
        df = pd.read_parquet(cf).sort_index()
        iv = df["iv30"]
        roll = iv.rolling(63, min_periods=10)
        lo, hi = roll.min(), roll.max()
        df["iv_rank"] = ((iv - lo) / (hi - lo + 1e-9)).clip(0, 1)
        df["iv_percentile"] = iv.rolling(63, min_periods=10).apply(
            lambda w: (w < w[-1]).mean(), raw=True)
        df["oi_change_1d"] = df["total_oi"].pct_change(1)
        df["oi_change_5d"] = df["total_oi"].pct_change(5)
        out[s] = df
    return out


def align(panel_index, feats: dict, cols) -> pd.DataFrame:
    """Leak-free: bar on date D gets the feature from the latest feat date STRICTLY < D."""
    out = pd.DataFrame(0.0, index=panel_index, columns=cols)
    syms = panel_index.get_level_values(0)
    bdates = pd.to_datetime(panel_index.get_level_values(1)).normalize().to_numpy("datetime64[D]")
    for s in pd.unique(syms):
        df = feats.get(s)
        if df is None or df.empty:
            continue
        fdates = df.index.normalize().to_numpy("datetime64[D]")
        rows = np.where(syms == s)[0]
        for j, c in enumerate(cols):
            vals = df[c].to_numpy() if c in df else None
            if vals is None:
                continue
            for r in rows:
                k = np.searchsorted(fdates, bdates[r], side="left") - 1   # strict < bar date
                out.iat[r, j] = vals[k] if k >= 0 and np.isfinite(vals[k]) else 0.0
    return out


def _demo():
    # leak-free check: a chain dated D must not reach a bar on date D, only D+1 onward
    idx = pd.MultiIndex.from_product(
        [["X"], pd.date_range("2024-03-04 10:00", periods=4, freq="D")], names=["symbol", "timestamp"])
    feats = {"X": pd.DataFrame({"iv_rank": [0.0, 0.9, 0.0, 0.0]},
                               index=pd.to_datetime(["2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07"]))}
    out = align(idx, feats, ["iv_rank"])
    v = out["iv_rank"].to_numpy()
    # bar 2024-03-05 must NOT see the 03-05 chain (same day); bar 03-06 sees 03-05's 0.9
    assert v[1] == 0.0, f"LEAK: same-day chain reached bar: {v}"
    assert v[2] == 0.9, f"prior-day chain not propagated: {v}"
    print("options_features._demo OK  leak-free prior-session alignment verified", list(v))


if __name__ == "__main__":
    _demo()
