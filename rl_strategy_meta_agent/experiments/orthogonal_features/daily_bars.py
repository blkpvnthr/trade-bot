"""Multi-year adjusted daily bars (Alpaca SIP, adjustment=ALL) for the panel symbols,
cached to parquet. Research-only market-data read; no broker/order/live path.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

TRADER_ROOT = Path(__file__).resolve().parents[3]
CACHE = TRADER_ROOT / "backtest_cache" / "daily_bars"
CACHE.mkdir(parents=True, exist_ok=True)


def _env():
    env = TRADER_ROOT / ".env"
    if "APCA_API_KEY_ID" not in os.environ and env.exists():
        for line in env.read_text().splitlines():
            for k in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
                if line.startswith(k + "="):
                    os.environ[k] = line.split("=", 1)[1].strip().strip('"').strip("'")


def fetch(symbols, start="2016-01-01", end="2026-06-18", force=False) -> pd.DataFrame:
    """Return a (symbol, date) daily OHLCV panel, adjusted for splits/dividends. Cached."""
    cf = CACHE / f"bars_{start}_{end}.parquet"
    if cf.exists() and not force:
        df = pd.read_parquet(cf)
        return df[df.index.get_level_values(0).isin(symbols)]
    _env()
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    c = StockHistoricalDataClient(os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"))
    req = StockBarsRequest(symbol_or_symbols=list(symbols), timeframe=TimeFrame.Day,
                           start=datetime.fromisoformat(start).replace(tzinfo=timezone.utc),
                           end=datetime.fromisoformat(end).replace(tzinfo=timezone.utc),
                           adjustment=Adjustment.ALL, feed=DataFeed.SIP)
    df = c.get_stock_bars(req).df  # MultiIndex (symbol, timestamp)
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = df.index.set_names(["symbol", "timestamp"])
    # normalize timestamp to tz-naive date
    df.index = pd.MultiIndex.from_arrays(
        [df.index.get_level_values(0),
         pd.to_datetime(df.index.get_level_values(1)).tz_convert("America/New_York").normalize().tz_localize(None)],
        names=["symbol", "timestamp"])
    df = df.sort_index()
    df.to_parquet(cf)
    return df


if __name__ == "__main__":
    syms = ["NVDA", "AAPL", "ARM"]
    d = fetch(syms, "2016-01-01", "2026-06-18")
    print("daily_bars OK", d.shape, "| per-symbol spans:")
    for s in syms:
        idx = d.loc[s].index
        print(f"  {s}: {len(idx)} bars {idx.min().date()} -> {idx.max().date()}")
