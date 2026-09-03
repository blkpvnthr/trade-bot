"""Coverage-aware bar fetching for the trade-outcome dataset.

For each (symbol, timeframe) in the labeled trades, work out how much 15Min
history is needed (>= 30 calendar days before the earliest entry through the
latest exit), fetch ONLY the missing ranges, merge with any cached bars (no
duplicate timestamps), and validate that there is enough pre-entry history to
compute the indicators. Data pull only -- no orders, no live trading.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import cache_path, fetch_bars, BARS_DIR, COLUMNS

LOOKBACK_DAYS = 30
MIN_PRE_BARS = 50          # bars needed before the earliest entry (SMA50/RSI/ATR)


def fetch_range(earliest_entry, latest_exit, lookback_days: int = LOOKBACK_DAYS):
    """(start_date, end_date) ISO strings: earliest_entry - lookback .. exit + 1d."""
    s = pd.Timestamp(earliest_entry) - pd.Timedelta(days=lookback_days)
    e = pd.Timestamp(latest_exit) + pd.Timedelta(days=1)
    return s.date().isoformat(), e.date().isoformat()


def merge_bars(dfs) -> pd.DataFrame:
    """Concat, dedup by timestamp, sort -> contiguous, unique bar rows."""
    dfs = [d for d in dfs if d is not None and not d.empty]
    if not dfs:
        return pd.DataFrame(columns=COLUMNS)
    m = pd.concat(dfs, ignore_index=True)
    m["timestamp"] = pd.to_datetime(m["timestamp"], utc=True)
    m = m.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return m[COLUMNS]


def cached_bars(symbol: str, timeframe: str, bars_dir=BARS_DIR) -> pd.DataFrame:
    """Union of every cached {symbol}_{tf}_*.csv (deduped, sorted)."""
    files = sorted(Path(bars_dir).glob(f"{symbol}_{timeframe}_*.csv"))
    out = []
    for f in files:
        d = pd.read_csv(f)
        if not d.empty:
            d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        out.append(d)
    return merge_bars(out)


def missing_ranges(req_start: str, req_end: str, df: pd.DataFrame) -> list:
    """Sub-ranges of [req_start, req_end] not covered by cached `df`. Coarse:
    treats the cache as contiguous between its min and max date."""
    if df is None or df.empty:
        return [(req_start, req_end)]
    cmin = df["timestamp"].min().date().isoformat()
    cmax = df["timestamp"].max().date().isoformat()
    gaps = []
    if req_start < cmin:
        gaps.append((req_start, cmin))
    if req_end > cmax:
        gaps.append((cmax, req_end))
    return gaps


def coverage_ok(df: pd.DataFrame, earliest_entry, latest_exit,
                min_pre_bars: int = MIN_PRE_BARS) -> tuple:
    if df is None or df.empty:
        return False, "no bars"
    ee = pd.Timestamp(earliest_entry); lx = pd.Timestamp(latest_exit)
    before = int((df["timestamp"] < ee).sum())
    if before < min_pre_bars:
        return False, f"only {before} bars before earliest_entry (<{min_pre_bars})"
    if df["timestamp"].max() < lx:
        return False, "bars end before latest_exit"
    return True, "ok"


def fetch_with_coverage(symbol: str, timeframe: str, req_start: str, req_end: str,
                        bars_dir=BARS_DIR, fetch_fn=None, refresh: bool = False) -> tuple:
    """Fetch only the missing ranges, merge with cache, write one canonical file.
    Returns (merged_df, status) where status in cached|fetched|empty."""
    fetch_fn = fetch_fn or fetch_bars
    existing = pd.DataFrame(columns=COLUMNS) if refresh else \
        cached_bars(symbol, timeframe, bars_dir)
    gaps = ([(req_start, req_end)] if (refresh or existing.empty)
            else missing_ranges(req_start, req_end, existing))
    if not gaps:
        return existing, "cached"
    fetched = [fetch_fn(symbol, timeframe, s, e) for s, e in gaps]
    merged = merge_bars([existing] + fetched)
    if merged.empty:
        return merged, "empty"
    cmin = merged["timestamp"].min().date().isoformat()
    cmax = merged["timestamp"].max().date().isoformat()
    p = cache_path(symbol, timeframe, cmin, cmax, bars_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(p, index=False)
    return merged, "fetched"


def _demo():
    def fake_fetch(sym, tf, s, e):
        idx = pd.date_range(f"{s} 09:30", f"{e} 16:00", freq="15min", tz="UTC")
        n = len(idx)
        return pd.DataFrame({"timestamp": idx, "open": 100.0, "high": 101.0,
                             "low": 99.0, "close": 100.0, "volume": 1e5,
                             "symbol": [sym] * n})[COLUMNS]
    s, e = fetch_range("2025-03-04 09:45", "2025-06-18 15:45")
    assert s == "2025-02-02" and e == "2025-06-19"
    import tempfile
    d = Path(tempfile.mkdtemp())
    df, st = fetch_with_coverage("NVDA", "15Min", s, e, d, fake_fetch)
    assert st == "fetched" and not df.empty
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].duplicated().sum() == 0
    df2, st2 = fetch_with_coverage("NVDA", "15Min", s, e, d, fake_fetch)
    assert st2 == "cached"        # second call reuses the cache
    print("bar_coverage._demo OK", s, e, len(df), "bars")


if __name__ == "__main__":
    _demo()
