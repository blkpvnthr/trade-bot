"""Alpaca historical-bars data layer. RESEARCH-ONLY.

This module ONLY reads historical OHLCV bars from the Alpaca *Data* API
(https://data.alpaca.markets). It never touches the trading API, never
submits orders, and has no live-execution path. Credentials come from the
environment:

    APCA_API_KEY_ID
    APCA_API_SECRET_KEY
    APCA_API_BASE_URL     # trading base, used ONLY for a paper-safety note

Bars are cached as CSV under data/bars/{symbol}_{timeframe}_{start}_{end}.csv
and reused unless refresh=True.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

from .registry import DATA_DIR, ROOT

# Data API host. Deliberately NOT the trading host -- we never trade.
DATA_HOST = os.environ.get("APCA_API_DATA_URL", "https://data.alpaca.markets")
BARS_DIR = DATA_DIR / "bars"
UNIVERSE_PATH = ROOT / "config" / "universe.json"

TIMEFRAMES = {"1Day", "1Hour", "15Min", "5Min"}
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
DEFAULT_FEED = os.environ.get("ALPACA_DATA_FEED", "iex")  # free tier = iex


class DataError(RuntimeError):
    pass


def read_universe(path: Path = UNIVERSE_PATH) -> list:
    return json.loads(Path(path).read_text())


def get_credentials() -> tuple:
    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise DataError(
            "Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY. Set paper-account "
            "data credentials in the environment (research-only, no trading).")
    # Paper-safety note: the trading base url is read but never used to trade.
    base = os.environ.get("APCA_API_BASE_URL", "")
    if base and "paper" not in base and "data" not in base:
        # Not fatal -- we never hit the trading API -- but warn loudly.
        print(f"[data] note: APCA_API_BASE_URL={base!r} is not a paper url; "
              "this module never trades regardless.")
    return key, secret


def cache_path(symbol: str, timeframe: str, start: str, end: str,
               bars_dir: Path = BARS_DIR) -> Path:
    return Path(bars_dir) / f"{symbol}_{timeframe}_{start}_{end}.csv"


def normalize_bars(raw_bars: list, symbol: str) -> pd.DataFrame:
    """Map Alpaca bar dicts (t/o/h/l/c/v) to the normalized schema. Pure."""
    rows = [{
        "timestamp": b["t"],
        "open": float(b["o"]),
        "high": float(b["h"]),
        "low": float(b["l"]),
        "close": float(b["c"]),
        "volume": float(b["v"]),
        "symbol": symbol,
    } for b in raw_bars]
    df = pd.DataFrame(rows, columns=COLUMNS)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _http_get_json(url: str, headers: dict) -> dict:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:  # nosec - GET only, data host
        return json.loads(resp.read().decode("utf-8"))


def fetch_bars(symbol: str, timeframe: str, start: str, end: str | None = None,
               feed: str = DEFAULT_FEED, adjustment: str = "all") -> pd.DataFrame:
    """Fetch bars from the Alpaca Data API (paginated). Network call."""
    if timeframe not in TIMEFRAMES:
        raise DataError(f"timeframe must be one of {sorted(TIMEFRAMES)}")
    end = end or date.today().isoformat()
    key, secret = get_credentials()
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    all_bars, page_token = [], None
    while True:
        params = {"timeframe": timeframe, "start": start, "end": end,
                  "limit": 10000, "adjustment": adjustment, "feed": feed}
        if page_token:
            params["page_token"] = page_token
        url = f"{DATA_HOST}/v2/stocks/{urllib.parse.quote(symbol)}/bars?" \
              + urllib.parse.urlencode(params)
        payload = _http_get_json(url, headers)
        all_bars.extend(payload.get("bars") or [])
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    return normalize_bars(all_bars, symbol)


def load_bars(symbol: str, timeframe: str, start: str, end: str | None = None,
              refresh: bool = False, offline: bool = False,
              bars_dir: Path = BARS_DIR) -> pd.DataFrame:
    """Cache-aware loader. Reads CSV cache unless refresh=True or it's missing.

    offline=True never hits the network: a cache miss raises DataError so
    tests / the RL fallback stay fully offline (no creds required)."""
    end = end or date.today().isoformat()
    path = cache_path(symbol, timeframe, start, end, bars_dir)
    if path.exists() and not refresh:
        df = pd.read_csv(path)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df[COLUMNS]
    if offline:
        raise DataError(f"no cached bars for {symbol} {timeframe} (offline)")
    df = fetch_bars(symbol, timeframe, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_universe_bars(symbols: list, timeframe: str, start: str,
                       end: str | None = None, refresh: bool = False,
                       offline: bool = False) -> dict:
    """Return {symbol: DataFrame}. Symbols with no/empty data are skipped."""
    out = {}
    for sym in symbols:
        try:
            df = load_bars(sym, timeframe, start, end, refresh=refresh,
                           offline=offline)
        except DataError as e:
            print(f"[data] {sym}: {e}")
            continue
        if df is not None and not df.empty:
            out[sym] = df
        else:
            print(f"[data] {sym}: no bars returned")
    return out


def _demo():
    raw = [
        {"t": "2023-01-04T05:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100},
        {"t": "2023-01-03T05:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.2, "v": 200},
    ]
    df = normalize_bars(raw, "NVDA")
    assert list(df.columns) == COLUMNS
    assert df["timestamp"].is_monotonic_increasing  # sorted ascending
    assert df.iloc[0]["close"] == 1.2 and (df["symbol"] == "NVDA").all()
    assert cache_path("NVDA", "1Day", "2018-01-01", "2025-01-01").name == \
        "NVDA_1Day_2018-01-01_2025-01-01.csv"
    print("data._demo OK")


if __name__ == "__main__":
    _demo()
