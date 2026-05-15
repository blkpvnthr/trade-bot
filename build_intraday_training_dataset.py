#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

try:
    from alpaca.data.enums import DataFeed
except Exception:
    DataFeed = None

try:
    from market_context_builder import TODAY_TAG
except Exception:
    TODAY_TAG = datetime.now().strftime("%Y%m%d")


# =============================================================================
# Paths / Config
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

OUT_DIR = PROJECT_ROOT / "system-data" / f"{TODAY_TAG}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_STREAM_DIR = OUT_DIR / f"{TODAY_TAG}_Feature_stream"
FEATURE_STREAM_DIR.mkdir(parents=True, exist_ok=True)

STRUCTURE_MAP_PATH = OUT_DIR / f"{TODAY_TAG}_intraday_structure_map.json"

RAW_OUTPUT_PATH = OUT_DIR / f"{TODAY_TAG}_intraday_bars_1m.csv"
FEATURE_OUTPUT_PATH = OUT_DIR / f"{TODAY_TAG}_intraday_training_dataset.csv"
EXIT_DATASET_PATH = OUT_DIR / f"{TODAY_TAG}_exit_training_dataset.csv"
MTF_CONTEXT_PATH = OUT_DIR / f"{TODAY_TAG}_multitimeframe_context.json"
SUMMARY_PATH = OUT_DIR / f"{TODAY_TAG}_intraday_dataset_summary.json"

SYMBOL_STATE_PATH = FEATURE_STREAM_DIR / f"{TODAY_TAG}_symbol_state.json"
FEATURE_STREAM_CSV_PATH = FEATURE_STREAM_DIR / f"{TODAY_TAG}_intraday_feature_stream.csv"
FEATURE_STREAM_JSONL_PATH = FEATURE_STREAM_DIR / f"{TODAY_TAG}_intraday_feature_stream.jsonl"

LOOKBACK_DAYS = int(os.getenv("RL_INTRADAY_LOOKBACK_DAYS", "30"))
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex").lower().strip()
ALPACA_CHUNK_SIZE = int(os.getenv("ALPACA_CHUNK_SIZE", "25"))

MIN_ROWS_PER_SYMBOL = int(os.getenv("INTRADAY_MIN_ROWS_PER_SYMBOL", "120"))

FUTURE_HORIZONS = [1, 3, 5, 10, 15, 30]
EXIT_HORIZONS = [5, 15, 30]

TAKE_PROFIT_THRESHOLD = float(os.getenv("EXIT_LABEL_TAKE_PROFIT_PCT", "0.008"))
GIVEBACK_THRESHOLD = float(os.getenv("EXIT_LABEL_GIVEBACK_PCT", "0.004"))
CUT_LOSER_THRESHOLD = float(os.getenv("EXIT_LABEL_CUT_LOSER_PCT", "-0.006"))
CONTINUATION_THRESHOLD = float(os.getenv("EXIT_LABEL_CONTINUATION_PCT", "0.002"))

DEFAULT_SYMBOLS = pd.read_csv("data/bullish_2026-05-13.csv")["Symbol"].to_list()

# ============================================================================
# Helpers
# =============================================================================

def normalize_symbol(value: Any) -> str:
    return str(value or "").upper().strip().replace(".", "-")


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i: i + size] for i in range(0, len(items), size)]


def safe_json(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.floating):
        v = float(value)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(safe_json(payload), indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {path}")


# =============================================================================
# Universe loading
# =============================================================================

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


# =============================================================================
# Alpaca
# =============================================================================

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


# =============================================================================
# 1m Feature Engineering
# =============================================================================

def add_symbol_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("timestamp").copy()

    g["ret_1m"] = g["close"].pct_change()
    g["ret_3m"] = g["close"].pct_change(3)
    g["ret_5m"] = g["close"].pct_change(5)
    g["ret_10m"] = g["close"].pct_change(10)
    g["ret_15m"] = g["close"].pct_change(15)
    g["ret_30m"] = g["close"].pct_change(30)

    g["log_ret_1m"] = np.log(g["close"] / g["close"].shift(1))

    for window in [5, 10, 15, 20, 30, 60]:
        g[f"sma_{window}"] = g["close"].rolling(window).mean()
        g[f"ema_{window}"] = g["close"].ewm(span=window, adjust=False).mean()
        g[f"dist_sma_{window}"] = g["close"] / g[f"sma_{window}"] - 1.0
        g[f"volatility_{window}"] = g["ret_1m"].rolling(window).std()
        g[f"volume_sma_{window}"] = g["volume"].rolling(window).mean()
        g[f"rvol_{window}"] = g["volume"] / g[f"volume_sma_{window}"]

    high_low = g["high"] - g["low"]
    high_close = (g["high"] - g["close"].shift()).abs()
    low_close = (g["low"] - g["close"].shift()).abs()

    g["true_range"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    g["atr_14"] = g["true_range"].rolling(14).mean()
    g["atr_pct"] = g["atr_14"] / g["close"]

    g["range_pct"] = (g["high"] - g["low"]) / g["close"]
    g["body_pct"] = (g["close"] - g["open"]) / g["open"]
    g["upper_wick_pct"] = (g["high"] - g[["open", "close"]].max(axis=1)) / g["close"]
    g["lower_wick_pct"] = (g[["open", "close"]].min(axis=1) - g["low"]) / g["close"]

    delta = g["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)

    g["rsi_14"] = 100 - (100 / (1 + rs))

    ema_12 = g["close"].ewm(span=12, adjust=False).mean()
    ema_26 = g["close"].ewm(span=26, adjust=False).mean()

    g["macd"] = ema_12 - ema_26
    g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
    g["macd_hist"] = g["macd"] - g["macd_signal"]

    typical = (g["high"] + g["low"] + g["close"]) / 3
    session_date = g["timestamp"].dt.tz_convert("America/New_York").dt.date

    g["_session_date"] = session_date
    g["cum_vol"] = g.groupby("_session_date")["volume"].cumsum()
    g["cum_pv"] = (typical * g["volume"]).groupby(g["_session_date"]).cumsum()
    g["vwap"] = g["cum_pv"] / g["cum_vol"].replace(0, np.nan)
    g["dist_vwap"] = g["close"] / g["vwap"] - 1.0

    g["rolling_high_15"] = g["high"].rolling(15).max()
    g["rolling_low_15"] = g["low"].rolling(15).min()
    g["rolling_high_30"] = g["high"].rolling(30).max()
    g["rolling_low_30"] = g["low"].rolling(30).min()

    g["breakout_15"] = (g["close"] > g["rolling_high_15"].shift(1)).astype(int)
    g["breakdown_15"] = (g["close"] < g["rolling_low_15"].shift(1)).astype(int)

    g["trend_5_20"] = (g["ema_5"] > g["ema_20"]).astype(int)
    g["trend_10_30"] = (g["ema_10"] > g["ema_30"]).astype(int)
    g["above_vwap"] = (g["close"] > g["vwap"]).astype(int)

    g = g.drop(columns=["_session_date"], errors="ignore")

    return g


# =============================================================================
# Multi-timeframe Context
# =============================================================================

def resample_ohlcv(g: pd.DataFrame, rule: str) -> pd.DataFrame:
    x = g.sort_values("timestamp").copy()
    x = x.set_index("timestamp")

    out = x.resample(rule).agg(
        {
            "symbol": "last",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.reset_index()

    return out


def detect_pivot_levels(
    bars: pd.DataFrame,
    window: int = 3,
    max_levels: int = 8,
    cluster_pct: float = 0.003,
) -> dict[str, Any]:
    if bars.empty or len(bars) < window * 2 + 5:
        return {
            "support_levels": [],
            "resistance_levels": [],
            "nearest_support": None,
            "nearest_resistance": None,
        }

    x = bars.sort_values("timestamp").copy()
    current_price = float(x["close"].iloc[-1])

    pivot_lows: list[float] = []
    pivot_highs: list[float] = []

    for i in range(window, len(x) - window):
        low = float(x["low"].iloc[i])
        high = float(x["high"].iloc[i])

        left_lows = x["low"].iloc[i - window:i]
        right_lows = x["low"].iloc[i + 1:i + window + 1]

        left_highs = x["high"].iloc[i - window:i]
        right_highs = x["high"].iloc[i + 1:i + window + 1]

        if low <= left_lows.min() and low <= right_lows.min():
            pivot_lows.append(low)

        if high >= left_highs.max() and high >= right_highs.max():
            pivot_highs.append(high)

    def cluster(levels: list[float]) -> list[dict[str, Any]]:
        clustered: list[dict[str, Any]] = []

        for level in sorted(levels):
            placed = False

            for c in clustered:
                if abs(level / c["price"] - 1.0) <= cluster_pct:
                    c["values"].append(level)
                    c["price"] = float(sum(c["values"]) / len(c["values"]))
                    c["touch_count"] = len(c["values"])
                    placed = True
                    break

            if not placed:
                clustered.append(
                    {
                        "price": float(level),
                        "touch_count": 1,
                        "values": [level],
                    }
                )

        for c in clustered:
            c["distance_pct"] = float(c["price"] / current_price - 1.0)
            c["weight"] = float(c["touch_count"] / max(abs(c["distance_pct"]), 0.001))

        clustered = sorted(clustered, key=lambda z: z["weight"], reverse=True)

        clean = []

        for c in clustered[:max_levels]:
            clean.append(
                {
                    "price": round(float(c["price"]), 4),
                    "touch_count": int(c["touch_count"]),
                    "distance_pct": round(float(c["distance_pct"]), 6),
                    "weight": round(float(c["weight"]), 4),
                }
            )

        return clean

    supports = [x for x in cluster(pivot_lows) if x["price"] <= current_price]
    resistances = [x for x in cluster(pivot_highs) if x["price"] >= current_price]

    nearest_support = (
        sorted(supports, key=lambda z: abs(z["distance_pct"]))[0]
        if supports
        else None
    )

    nearest_resistance = (
        sorted(resistances, key=lambda z: abs(z["distance_pct"]))[0]
        if resistances
        else None
    )

    return {
        "support_levels": supports,
        "resistance_levels": resistances,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
    }


def add_timeframe_features(bars: pd.DataFrame, prefix: str) -> pd.DataFrame:
    x = bars.sort_values("timestamp").copy()

    x[f"{prefix}_ret_1"] = x["close"].pct_change()
    x[f"{prefix}_ret_3"] = x["close"].pct_change(3)
    x[f"{prefix}_ret_6"] = x["close"].pct_change(6)

    for w in [5, 10, 20]:
        x[f"{prefix}_sma_{w}"] = x["close"].rolling(w).mean()
        x[f"{prefix}_ema_{w}"] = x["close"].ewm(span=w, adjust=False).mean()
        x[f"{prefix}_dist_sma_{w}"] = x["close"] / x[f"{prefix}_sma_{w}"] - 1.0

    x[f"{prefix}_trend_stack"] = (
        (x[f"{prefix}_ema_5"] > x[f"{prefix}_ema_10"])
        & (x[f"{prefix}_ema_10"] > x[f"{prefix}_ema_20"])
    ).astype(int)

    x[f"{prefix}_range_pct"] = (x["high"] - x["low"]) / x["close"]
    x[f"{prefix}_rolling_high_20"] = x["high"].rolling(20).max()
    x[f"{prefix}_rolling_low_20"] = x["low"].rolling(20).min()

    x[f"{prefix}_breakout_20"] = (
        x["close"] > x[f"{prefix}_rolling_high_20"].shift(1)
    ).astype(int)

    x[f"{prefix}_breakdown_20"] = (
        x["close"] < x[f"{prefix}_rolling_low_20"].shift(1)
    ).astype(int)

    typical = (x["high"] + x["low"] + x["close"]) / 3
    cumulative_volume = x["volume"].cumsum().replace(0, np.nan)
    x[f"{prefix}_vwap"] = (typical * x["volume"]).cumsum() / cumulative_volume
    x[f"{prefix}_dist_vwap"] = x["close"] / x[f"{prefix}_vwap"] - 1.0

    return x


def build_multitimeframe_context_for_symbol(
    g: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    symbol = str(g["symbol"].iloc[0])
    base = g.sort_values("timestamp").copy()

    timeframe_specs = {
        "5m": "5min",
        "15m": "15min",
        "1h": "1h",
        "4h": "4h",
    }

    context: dict[str, Any] = {
        "symbol": symbol,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "timeframes": {},
    }

    enriched = base.copy()

    for label, rule in timeframe_specs.items():
        tf_bars = resample_ohlcv(base, rule)
        tf_bars = add_timeframe_features(tf_bars, label)

        levels = detect_pivot_levels(
            tf_bars,
            window=2 if label in {"5m", "15m"} else 1,
            max_levels=8,
            cluster_pct=0.003 if label == "5m" else (
                0.004 if label == "15m" else 0.006
            ),
        )

        latest = tf_bars.iloc[-1].to_dict() if not tf_bars.empty else {}

        context["timeframes"][label] = {
            "bar_count": int(len(tf_bars)),
            "latest_close": to_float(latest.get("close")),
            "trend_stack": int(to_float(latest.get(f"{label}_trend_stack"))),
            "ret_1": to_float(latest.get(f"{label}_ret_1")),
            "ret_3": to_float(latest.get(f"{label}_ret_3")),
            "dist_vwap": to_float(latest.get(f"{label}_dist_vwap")),
            "breakout_20": int(to_float(latest.get(f"{label}_breakout_20"))),
            "breakdown_20": int(to_float(latest.get(f"{label}_breakdown_20"))),
            "support_resistance": levels,
        }

        keep_cols = [
            "timestamp",
            f"{label}_ret_1",
            f"{label}_ret_3",
            f"{label}_ret_6",
            f"{label}_dist_sma_5",
            f"{label}_dist_sma_10",
            f"{label}_dist_sma_20",
            f"{label}_trend_stack",
            f"{label}_range_pct",
            f"{label}_breakout_20",
            f"{label}_breakdown_20",
            f"{label}_dist_vwap",
        ]

        available = [c for c in keep_cols if c in tf_bars.columns]
        tf_keep = tf_bars[available].copy().sort_values("timestamp")

        enriched = pd.merge_asof(
            enriched.sort_values("timestamp"),
            tf_keep,
            on="timestamp",
            direction="backward",
        )

        ns = levels.get("nearest_support")
        nr = levels.get("nearest_resistance")

        enriched[f"{label}_nearest_support"] = ns.get("price") if ns else np.nan
        enriched[f"{label}_nearest_resistance"] = nr.get("price") if nr else np.nan

        enriched[f"{label}_dist_to_support"] = (
            enriched["close"] / enriched[f"{label}_nearest_support"] - 1.0
            if ns
            else np.nan
        )

        enriched[f"{label}_dist_to_resistance"] = (
            enriched[f"{label}_nearest_resistance"] / enriched["close"] - 1.0
            if nr
            else np.nan
        )

    return enriched, context


def write_multitimeframe_context(contexts: list[dict[str, Any]]) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol_count": len(contexts),
        "timeframes": ["1m", "5m", "15m", "1h", "4h"],
        "symbols": [x.get("symbol") for x in contexts],
        "contexts": contexts,
    }

    write_json(MTF_CONTEXT_PATH, payload)


# =============================================================================
# Labels
# =============================================================================

def add_future_labels(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("timestamp").copy()

    for h in FUTURE_HORIZONS:
        g[f"future_close_{h}m"] = g["close"].shift(-h)
        g[f"future_return_{h}m"] = g[f"future_close_{h}m"] / g["close"] - 1.0
        g[f"future_up_{h}m"] = (g[f"future_return_{h}m"] > 0).astype(int)

    for h in EXIT_HORIZONS:
        future_high = g["high"].shift(-1).rolling(h).max().shift(-(h - 1))
        future_low = g["low"].shift(-1).rolling(h).min().shift(-(h - 1))

        g[f"future_max_return_{h}m"] = future_high / g["close"] - 1.0
        g[f"future_min_return_{h}m"] = future_low / g["close"] - 1.0

    return g


def add_exit_learning_labels(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("timestamp").copy()

    g["rolling_peak_close_30"] = g["close"].rolling(30).max()
    g["distance_from_recent_peak"] = g["close"] / g["rolling_peak_close_30"] - 1.0
    g["profit_giveback_from_peak"] = g["rolling_peak_close_30"] / g["close"] - 1.0

    g["max_adverse_excursion_15m"] = g["future_min_return_15m"]
    g["max_favorable_excursion_15m"] = g["future_max_return_15m"]

    g["exit_label"] = "HOLD"

    giveback_now = (
        (g["distance_from_recent_peak"] <= -GIVEBACK_THRESHOLD)
        & (g["ret_5m"] < 0)
        & (g["macd_hist"] < g["macd_hist"].shift(1))
    )

    take_profit_now = (
        (g["future_max_return_15m"] < CONTINUATION_THRESHOLD)
        & (g["ret_15m"] >= TAKE_PROFIT_THRESHOLD)
    )

    cut_loser_now = (
        (g["future_min_return_15m"] <= CUT_LOSER_THRESHOLD)
        & (g["ret_5m"] < 0)
        & (g["close"] < g["vwap"])
    )

    scale_out_now = (
        (g["ret_15m"] > TAKE_PROFIT_THRESHOLD)
        & (g["future_max_return_15m"] > 0)
        & (g["future_min_return_15m"] < -GIVEBACK_THRESHOLD)
    )

    g.loc[scale_out_now, "exit_label"] = "SCALE_OUT"
    g.loc[take_profit_now, "exit_label"] = "TAKE_PROFIT_NOW"
    g.loc[giveback_now, "exit_label"] = "TRAIL_STOP_NOW"
    g.loc[cut_loser_now, "exit_label"] = "CUT_LOSER_NOW"

    label_map = {
        "HOLD": 0,
        "TAKE_PROFIT_NOW": 1,
        "TRAIL_STOP_NOW": 2,
        "CUT_LOSER_NOW": 3,
        "SCALE_OUT": 4,
    }

    g["exit_label_id"] = g["exit_label"].map(label_map).fillna(0).astype(int)

    g["prob_hold_training_target"] = (
        (g["future_return_15m"] > CONTINUATION_THRESHOLD)
        & (g["future_min_return_15m"] > -GIVEBACK_THRESHOLD)
    ).astype(int)

    g["giveback_risk_training_target"] = (
        g["future_min_return_15m"] <= -GIVEBACK_THRESHOLD
    ).astype(int)

    g["reversal_risk_training_target"] = (
        (g["future_return_15m"] < 0)
        & (g["macd_hist"] < 0)
        & (g["close"] < g["vwap"])
    ).astype(int)

    g["expected_value_next_5m"] = g["future_return_5m"]
    g["expected_value_next_15m"] = g["future_return_15m"]
    g["expected_value_next_30m"] = g["future_return_30m"]

    return g


# =============================================================================
# Dataset Build
# =============================================================================

def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    mtf_contexts: list[dict[str, Any]] = []

    for symbol, g in raw.groupby("symbol"):
        if len(g) < MIN_ROWS_PER_SYMBOL:
            print(f"[features] skip {symbol}: only {len(g)} rows")
            continue

        x = add_symbol_features(g)
        x, context = build_multitimeframe_context_for_symbol(x)
        x = add_future_labels(x)
        x = add_exit_learning_labels(x)

        mtf_contexts.append(context)
        frames.append(x)

    if not frames:
        raise RuntimeError("No symbols had enough rows for feature construction.")

    write_multitimeframe_context(mtf_contexts)

    df = pd.concat(frames, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# =============================================================================
# Feature Stream
# =============================================================================

def write_feature_stream(features: pd.DataFrame) -> None:
    latest_rows: list[dict[str, Any]] = []

    for symbol, g in features.groupby("symbol"):
        g = g.sort_values("timestamp")
        row = g.iloc[-1].to_dict()
        latest_rows.append(row)

    latest = pd.DataFrame(latest_rows)
    latest.to_csv(FEATURE_STREAM_CSV_PATH, index=False)

    with FEATURE_STREAM_JSONL_PATH.open("w", encoding="utf-8") as f:
        for row in latest_rows:
            f.write(json.dumps(safe_json(row)) + "\n")

    state = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol_count": len(latest_rows),
        "symbols": sorted(latest["symbol"].astype(str).unique()) if not latest.empty else [],
        "rows": safe_json(latest_rows),
    }

    write_json(SYMBOL_STATE_PATH, state)

    print(f"[saved] {FEATURE_STREAM_CSV_PATH}")
    print(f"[saved] {FEATURE_STREAM_JSONL_PATH}")


# =============================================================================
# Summary
# =============================================================================

def dataset_summary(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    exit_df: pd.DataFrame,
) -> dict[str, Any]:
    label_counts = (
        exit_df["exit_label"].value_counts().to_dict()
        if "exit_label" in exit_df.columns
        else {}
    )

    rows_by_symbol = raw.groupby("symbol").size().sort_values(ascending=False).to_dict()

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "today_tag": TODAY_TAG,
        "feed": ALPACA_DATA_FEED,
        "lookback_days": LOOKBACK_DAYS,
        "raw_rows": int(len(raw)),
        "feature_rows": int(len(features)),
        "exit_dataset_rows": int(len(exit_df)),
        "symbols_requested": int(raw["symbol"].nunique()),
        "symbols_with_features": int(features["symbol"].nunique()),
        "rows_by_symbol": {str(k): int(v) for k, v in rows_by_symbol.items()},
        "future_horizons": FUTURE_HORIZONS,
        "exit_horizons": EXIT_HORIZONS,
        "multitimeframe_context": {
            "enabled": True,
            "timeframes": ["1m", "5m", "15m", "1h", "4h"],
            "path": str(MTF_CONTEXT_PATH),
        },
        "exit_label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "outputs": {
            "raw_bars": str(RAW_OUTPUT_PATH),
            "features": str(FEATURE_OUTPUT_PATH),
            "exit_dataset": str(EXIT_DATASET_PATH),
            "multitimeframe_context": str(MTF_CONTEXT_PATH),
            "symbol_state": str(SYMBOL_STATE_PATH),
            "feature_stream_csv": str(FEATURE_STREAM_CSV_PATH),
            "feature_stream_jsonl": str(FEATURE_STREAM_JSONL_PATH),
        },
    }


def exit_columns() -> list[str]:
    base_cols = [
        "timestamp", "symbol", "open", "high", "low", "close", "volume",
        "ret_1m", "ret_3m", "ret_5m", "ret_10m", "ret_15m", "ret_30m",
        "dist_sma_5", "dist_sma_10", "dist_sma_15", "dist_sma_20",
        "dist_sma_30", "dist_sma_60",
        "volatility_5", "volatility_15", "volatility_30", "volatility_60",
        "rvol_5", "rvol_15", "rvol_30", "rvol_60",
        "atr_pct", "range_pct", "body_pct",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "vwap", "dist_vwap", "above_vwap",
        "breakout_15", "breakdown_15",
    ]

    mtf_cols: list[str] = []

    for tf in ["5m", "15m", "1h", "4h"]:
        mtf_cols.extend(
            [
                f"{tf}_ret_1",
                f"{tf}_ret_3",
                f"{tf}_ret_6",
                f"{tf}_dist_sma_5",
                f"{tf}_dist_sma_10",
                f"{tf}_dist_sma_20",
                f"{tf}_trend_stack",
                f"{tf}_range_pct",
                f"{tf}_breakout_20",
                f"{tf}_breakdown_20",
                f"{tf}_dist_vwap",
                f"{tf}_nearest_support",
                f"{tf}_nearest_resistance",
                f"{tf}_dist_to_support",
                f"{tf}_dist_to_resistance",
            ]
        )

    label_cols = [
        "future_return_5m", "future_return_15m", "future_return_30m",
        "future_max_return_5m", "future_max_return_15m", "future_max_return_30m",
        "future_min_return_5m", "future_min_return_15m", "future_min_return_30m",
        "max_adverse_excursion_15m",
        "max_favorable_excursion_15m",
        "profit_giveback_from_peak",
        "prob_hold_training_target",
        "giveback_risk_training_target",
        "reversal_risk_training_target",
        "expected_value_next_5m",
        "expected_value_next_15m",
        "expected_value_next_30m",
        "exit_label",
        "exit_label_id",
    ]

    return base_cols + mtf_cols + label_cols


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    symbols = load_structure_symbols()

    raw = fetch_intraday_bars(symbols)
    raw.to_csv(RAW_OUTPUT_PATH, index=False)
    print(f"[saved] {RAW_OUTPUT_PATH}")

    features = build_features(raw)
    features.to_csv(FEATURE_OUTPUT_PATH, index=False)
    print(f"[saved] {FEATURE_OUTPUT_PATH}")

    available_exit_cols = [c for c in exit_columns() if c in features.columns]

    exit_df = features[available_exit_cols].dropna(
        subset=["future_return_15m", "exit_label"]
    )

    exit_df.to_csv(EXIT_DATASET_PATH, index=False)
    print(f"[saved] {EXIT_DATASET_PATH}")

    write_feature_stream(features)

    summary = dataset_summary(raw, features, exit_df)
    write_json(SUMMARY_PATH, summary)

    print(f"[rows_raw] {len(raw):,}")
    print(f"[rows_features] {len(features):,}")
    print(f"[rows_exit_dataset] {len(exit_df):,}")
    print(f"[symbols_returned] {raw['symbol'].nunique()}")
    print(",".join(sorted(raw["symbol"].unique())))


if __name__ == "__main__":
    main()