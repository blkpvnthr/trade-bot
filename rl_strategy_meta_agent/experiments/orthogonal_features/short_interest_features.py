"""Leak-free short-interest / short-volume features aligned to a daily (symbol, date)
panel. Source: Massive (api.massive.com), a FINRA-data aggregator.

Two genuinely different short-pressure channels:
  * SHORT INTEREST (/stocks/v1/short-interest) -- bi-weekly positioning: short_interest
    shares, days_to_cover, settlement_date. The classic squeeze/positioning signal.
    FINRA disseminates ~8 business days AFTER the settlement date, so each reading is
    aligned to an AVAILABILITY date = settlement_date + a conservative lag, then
    forward-filled. No lookahead.
  * SHORT VOLUME (/stocks/v1/short-volume) -- daily off-exchange short-sale flow:
    short_volume_ratio. Known after the session closes, so a bar on day D uses only the
    latest reading STRICTLY BEFORE D (prior session).

Daily horizon only (the spec: avoid 15m churn). Borrow fee / utilization are NOT in
this feed (proprietary securities-lending data) -- reported as unavailable, not faked.
Research only: read-only market data, no broker/order/live path.
"""
from __future__ import annotations

import os
import sys
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd

TRADER_ROOT = Path(__file__).resolve().parents[3]
CACHE = TRADER_ROOT / "backtest_cache" / "massive_short"
CACHE.mkdir(parents=True, exist_ok=True)
BASE = "https://api.massive.com"
SI_PUB_LAG_DAYS = 12          # FINRA short-interest dissemination lag (conservative)


def _key():
    env = TRADER_ROOT / ".env"
    if "MASSIVE_API_KEY" not in os.environ and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("MASSIVE_API_KEY="):
                os.environ["MASSIVE_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("MASSIVE_API_KEY", "")


def _get(path, params, retries=3):
    params = {**params, "apiKey": _key(), "limit": 50000}
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    for a in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if a == retries - 1:
                raise
            time.sleep(1.0 * (a + 1))


# ---------------------------------------------------------------- fetch + cache
def fetch_short_volume(symbols, start, end, force=False) -> dict:
    out = {}
    for s in symbols:
        cf = CACHE / f"sv_{s}.parquet"
        if cf.exists() and not force:
            out[s] = pd.read_parquet(cf); continue
        d = _get("/stocks/v1/short-volume",
                 {"ticker": s, "date.gte": start, "date.lte": end, "sort": "date.asc"})
        rows = d.get("results") or []
        df = pd.DataFrame([{"date": r["date"], "sv_ratio": float(r.get("short_volume_ratio") or 0) / 100.0}
                           for r in rows])
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]); df = df.sort_values("date").reset_index(drop=True)
            df.to_parquet(cf)
        out[s] = df
    return out


def fetch_short_interest(symbols, start, end, force=False) -> dict:
    out = {}
    for s in symbols:
        cf = CACHE / f"si_{s}.parquet"
        if cf.exists() and not force:
            out[s] = pd.read_parquet(cf); continue
        d = _get("/stocks/v1/short-interest",
                 {"ticker": s, "settlement_date.gte": start, "settlement_date.lte": end,
                  "sort": "settlement_date.asc"})
        rows = d.get("results") or []
        df = pd.DataFrame([{"settlement_date": r["settlement_date"],
                            "short_interest": float(r.get("short_interest") or 0),
                            "days_to_cover": float(r.get("days_to_cover") or 0)} for r in rows])
        if not df.empty:
            df["settlement_date"] = pd.to_datetime(df["settlement_date"])
            df = df.sort_values("settlement_date").reset_index(drop=True)
            df.to_parquet(cf)
        out[s] = df
    return out


# ---------------------------------------------------------------- features
SV_COLS = ["sv_ratio", "sv_ratio_chg_5d", "sv_ratio_z20", "abnormal_sv"]
SI_COLS = ["si_days_to_cover", "si_dtc_chg", "si_dtc_z", "si_shares_chg"]
ALL_COLS = SV_COLS + SI_COLS


def _sv_features(df):
    if df.empty:
        return df
    r = df.set_index("date")["sv_ratio"].astype(float)
    f = pd.DataFrame(index=r.index)
    f["sv_ratio"] = r
    f["sv_ratio_chg_5d"] = r.diff(5)
    m, sd = r.rolling(20, min_periods=5).mean(), r.rolling(20, min_periods=5).std()
    f["sv_ratio_z20"] = (r - m) / (sd + 1e-9)
    f["abnormal_sv"] = r / (m + 1e-9)
    return f


def _si_features(df):
    if df.empty:
        return df
    dtc = df.set_index("settlement_date")["days_to_cover"].astype(float)
    sh = df.set_index("settlement_date")["short_interest"].astype(float)
    f = pd.DataFrame(index=dtc.index)
    f["si_days_to_cover"] = dtc
    f["si_dtc_chg"] = dtc.diff()
    m, sd = dtc.rolling(6, min_periods=3).mean(), dtc.rolling(6, min_periods=3).std()
    f["si_dtc_z"] = (dtc - m) / (sd + 1e-9)
    f["si_shares_chg"] = sh.pct_change()
    f.index = f.index + pd.Timedelta(days=SI_PUB_LAG_DAYS)        # availability date
    return f


def align(panel_index: pd.MultiIndex, short_vol: dict, short_int: dict, cfg=None) -> pd.DataFrame:
    """Leak-free daily features: each bar on date D uses only readings whose availability
    date is STRICTLY BEFORE D (prior session for SV; settlement+pub-lag for SI)."""
    out = pd.DataFrame(0.0, index=panel_index, columns=ALL_COLS)
    syms = panel_index.get_level_values(0)
    bdates = pd.to_datetime(panel_index.get_level_values(1)).normalize().to_numpy("datetime64[D]")
    for s in pd.unique(syms):
        rows = np.where(syms == s)[0]
        for feats, cols in ((_sv_features(short_vol.get(s, pd.DataFrame())), SV_COLS),
                            (_si_features(short_int.get(s, pd.DataFrame())), SI_COLS)):
            if feats is None or feats.empty:
                continue
            fdates = feats.index.normalize().to_numpy("datetime64[D]")
            for c in cols:
                vals = feats[c].to_numpy()
                for r in rows:
                    k = np.searchsorted(fdates, bdates[r], side="left") - 1   # strict < D
                    v = vals[k] if k >= 0 else np.nan
                    out.iat[r, out.columns.get_loc(c)] = v if np.isfinite(v) else 0.0
    return out


# ---------------------------------------------------------------- self-check
def _demo():
    idx = pd.MultiIndex.from_product(
        [["X"], pd.date_range("2024-03-04", periods=6, freq="D")], names=["symbol", "timestamp"])
    sv = {"X": pd.DataFrame({"date": pd.to_datetime(["2024-03-04", "2024-03-05", "2024-03-06"]),
                             "sv_ratio": [0.30, 0.55, 0.40]})}
    si = {"X": pd.DataFrame(columns=["settlement_date", "short_interest", "days_to_cover"])}
    out = align(idx, sv, si)
    v = out["sv_ratio"].to_numpy()
    bars = idx.get_level_values(1)
    # the 03-05 reading (ratio .55) must NOT reach the 03-05 bar, only 03-06 onward
    assert v[bars == pd.Timestamp("2024-03-05")][0] == 0.30, f"LEAK: same-day SV reached bar {v}"
    assert v[bars == pd.Timestamp("2024-03-06")][0] == 0.55, f"prior-day SV not propagated {v}"
    print("short_interest_features._demo OK  leak-free prior-session alignment verified", list(v))


if __name__ == "__main__":
    _demo()
