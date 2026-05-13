#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from market_context_builder import TODAY_TAG

try:
    from alpaca.data.enums import DataFeed
except Exception:
    DataFeed = None


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

OUT_DIR = PROJECT_ROOT / f"{TODAY_TAG}_daily_system_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRUCTURE_MAP_PATH = OUT_DIR / f"{TODAY_TAG}_intraday_structure_map.json"
OUTPUT_PATH = OUT_DIR / f"{TODAY_TAG}_intraday_bars_1m.csv"

LOOKBACK_DAYS = int(os.getenv("RL_INTRADAY_LOOKBACK_DAYS", "30"))
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex").lower().strip()
ALPACA_CHUNK_SIZE = int(os.getenv("ALPACA_CHUNK_SIZE", "25"))

DEFAULT_SYMBOLS = [
    "SPY", "QQQ", "NVDA", "TSLA",
    "AAPL", "MSFT", "AMD", "META",
]


def normalize_symbol(value: Any) -> str:
    return str(value or "").upper().strip().replace(".", "-")


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def load_structure_symbols() -> list[str]:
    if not STRUCTURE_MAP_PATH.exists():
        return DEFAULT_SYMBOLS

    try:
        payload = json.loads(STRUCTURE_MAP_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[structure] failed to read {STRUCTURE_MAP_PATH}: {exc}")
        return DEFAULT_SYMBOLS

    symbols: list[str] = []

    if isinstance(payload, dict):
        if isinstance(payload.get("long_allowed_symbols"), list):
            symbols.extend(payload["long_allowed_symbols"])

        if isinstance(payload.get("structures"), list):
            for item in payload["structures"]:
                if isinstance(item, dict):
                    symbols.append(item.get("symbol") or item.get("ticker") or "")

        for key in ["symbols", "structure_maps", "intraday_structure_maps"]:
            value = payload.get(key)
            if isinstance(value, dict):
                symbols.extend(value.keys())
            elif isinstance(value, list):
                symbols.extend(value)

    symbols = sorted(set(normalize_symbol(s) for s in symbols if normalize_symbol(s)))

    return symbols or DEFAULT_SYMBOLS


def get_client() -> StockHistoricalDataClient:
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")

    if not key or not secret:
        raise RuntimeError("Missing Alpaca credentials in .env.")

    return StockHistoricalDataClient(key, secret)


def get_feed() -> Any:
    if DataFeed is None:
        return None

    return DataFeed.SIP if ALPACA_DATA_FEED == "sip" else DataFeed.IEX


def fetch_intraday_bars(symbols: list[str]) -> pd.DataFrame:
    client = get_client()
    feed = get_feed()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)

    frames: list[pd.DataFrame] = []

    print(f"[dataset] downloading {LOOKBACK_DAYS} days of 1m bars")
    print(f"[dataset] feed={ALPACA_DATA_FEED}")
    print(f"[dataset] symbols={len(symbols)} {symbols}")

    for batch in chunked(symbols, ALPACA_CHUNK_SIZE):
        request_kwargs = {
            "symbol_or_symbols": batch,
            "timeframe": TimeFrame.Minute,
            "start": start,
            "end": end,
        }

        if feed is not None:
            request_kwargs["feed"] = feed

        try:
            request = StockBarsRequest(**request_kwargs)
            bars = client.get_stock_bars(request)

            if bars is None or bars.df is None or bars.df.empty:
                print(f"[dataset] empty batch={batch}")
                continue

            df = bars.df.reset_index()
            frames.append(df)

            counts = df.groupby("symbol").size().to_dict()
            print(f"[dataset] batch_counts={counts}")

        except Exception as exc:
            print(f"[dataset] failed batch={batch}: {exc}")

    if not frames:
        raise RuntimeError("No bars returned from Alpaca for any requested symbol.")

    df = pd.concat(frames, ignore_index=True)

    required = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise RuntimeError(f"Alpaca response missing required columns: {missing}")

    df = df[required].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["symbol"] = df["symbol"].map(normalize_symbol)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "symbol", "open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0)

    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    return df


def main() -> None:
    symbols = load_structure_symbols()
    df = fetch_intraday_bars(symbols)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"[saved] {OUTPUT_PATH}")
    print(f"[rows] {len(df):,}")
    print(f"[symbols_returned] {df['symbol'].nunique()}")
    print(",".join(sorted(df["symbol"].unique())))


if __name__ == "__main__":
    main()