import numpy as np
import pandas as pd

from rl_meta.bar_coverage import (fetch_range, merge_bars, missing_ranges,
                                  coverage_ok, fetch_with_coverage, cached_bars)
from rl_meta.data import COLUMNS
from rl_meta.trade_features import build_row, LOOKBACK


def _bars(start, end, symbol="NVDA"):
    idx = pd.date_range(f"{start} 09:30", f"{end} 16:00", freq="15min", tz="UTC")
    n = len(idx)
    return pd.DataFrame({"timestamp": idx, "open": 100.0, "high": 101.0,
                         "low": 99.0, "close": 100.0 + np.arange(n) * 0.01,
                         "volume": 1e5, "symbol": [symbol] * n})[COLUMNS]


def fake_fetch(sym, tf, s, e):
    return _bars(s, e, sym)


# -- date-range with 30-day lookback --------------------------------------
def test_fetch_range_30day_lookback():
    s, e = fetch_range("2025-03-04 09:45", "2026-06-18 15:45", lookback_days=30)
    assert s == "2025-02-02" and e == "2026-06-19"


# -- merge: no duplicate bar rows -----------------------------------------
def test_merge_dedups_overlapping():
    a = _bars("2025-01-01", "2025-01-10")
    b = _bars("2025-01-05", "2025-01-15")          # overlaps a
    m = merge_bars([a, b])
    assert m["timestamp"].duplicated().sum() == 0
    assert m["timestamp"].is_monotonic_increasing
    assert m["timestamp"].min() == a["timestamp"].min()
    assert m["timestamp"].max() == b["timestamp"].max()


# -- missing-range detection / merging ------------------------------------
def test_missing_ranges():
    df = _bars("2025-02-01", "2025-06-01")
    assert missing_ranges("2025-01-01", "2025-08-01", df) == \
        [("2025-01-01", "2025-02-01"), ("2025-06-01", "2025-08-01")]
    assert missing_ranges("2025-03-01", "2025-05-01", df) == []   # fully covered
    assert missing_ranges("2025-01-01", "2025-08-01", pd.DataFrame(columns=COLUMNS)) == \
        [("2025-01-01", "2025-08-01")]


def test_cache_coverage_detection_and_reuse(tmp_path):
    s, e = "2025-02-02", "2025-06-19"
    df1, st1 = fetch_with_coverage("NVDA", "15Min", s, e, tmp_path, fake_fetch)
    assert st1 == "fetched" and not df1.empty
    df2, st2 = fetch_with_coverage("NVDA", "15Min", s, e, tmp_path, fake_fetch)
    assert st2 == "cached"                          # second call hits cache
    assert cached_bars("NVDA", "15Min", tmp_path)["timestamp"].duplicated().sum() == 0


def test_fetch_range_merging_extends_cache(tmp_path):
    # cache Feb-Jun, then need Jan-Aug -> fetch only the two gaps, merge clean
    fetch_with_coverage("NVDA", "15Min", "2025-02-01", "2025-06-01", tmp_path, fake_fetch)
    df, st = fetch_with_coverage("NVDA", "15Min", "2025-01-01", "2025-08-01",
                                 tmp_path, fake_fetch)
    assert st == "fetched"
    assert df["timestamp"].min().date().isoformat() <= "2025-01-01"
    assert df["timestamp"].max().date().isoformat() >= "2025-07-31"
    assert df["timestamp"].duplicated().sum() == 0


# -- coverage_ok ----------------------------------------------------------
def test_coverage_ok_requires_pre_history():
    bars = _bars("2025-01-01", "2025-03-01")
    ee = pd.Timestamp("2025-02-15 10:00", tz="UTC")
    lx = pd.Timestamp("2025-02-28 15:00", tz="UTC")
    assert coverage_ok(bars, ee, lx)[0]
    # entry on the very first day -> not enough pre-history
    ee2 = bars["timestamp"].iloc[5]
    assert not coverage_ok(bars, ee2, lx)[0]
    assert not coverage_ok(pd.DataFrame(columns=COLUMNS), ee, lx)[0]


# -- rebuild populates pre-entry features when bars exist ------------------
def test_build_row_populates_with_real_like_bars():
    bars = _bars("2025-03-01", "2025-03-20")
    entry = bars["timestamp"].iloc[LOOKBACK + 5]
    exit_ = bars["timestamp"].iloc[LOOKBACK + 9]
    trade = {"entry_ts": entry, "exit_ts": exit_, "entry_price": 100.0,
             "exit_price": 101.0, "qty": 1000, "raw_pnl": 500.0, "return_pct": 0.05,
             "mfe_pct": 0.06, "mae_pct": -0.02, "mfe_usd": 1.0, "mae_usd": -1.0,
             "is_long": True}
    row = build_row(trade, bars, {"strategy_id": "s", "symbol": "NVDA",
                                  "timeframe": "15Min"}, {})
    assert row["has_pre_features"] and not np.isnan(row["rsi14"])
    assert not np.isnan(row["close_vs_sma20"])


def test_missing_bars_still_graceful():
    trade = {"entry_ts": pd.Timestamp("2025-03-03 10:00", tz="UTC"),
             "exit_ts": pd.Timestamp("2025-03-03 11:00", tz="UTC"),
             "entry_price": 100.0, "exit_price": 105.0, "qty": 1000, "raw_pnl": 500.0,
             "return_pct": 0.05, "mfe_pct": 0.06, "mae_pct": -0.02,
             "mfe_usd": 1.0, "mae_usd": -1.0, "is_long": True}
    row = build_row(trade, None, {"strategy_id": "s", "symbol": "X", "timeframe": "15Min"}, {})
    assert row["has_pre_features"] is False and row["normalized_pnl"] == 5.0


# -- no live broker path ---------------------------------------------------
def test_no_broker_path_in_new_files():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for f in ("rl_meta/bar_coverage.py", "scripts/fetch_trade_context_bars.py"):
        t = (root / f).read_text()
        for bad in ("submit_order", "TradingClient", "/v2/orders", "place_order"):
            assert bad not in t
