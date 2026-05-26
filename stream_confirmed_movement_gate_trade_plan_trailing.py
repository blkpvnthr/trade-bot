#!/usr/bin/env python3
"""
stream_confirmed_movement_gate.py

Live streamed-bar setup discovery + confirmation layer.

Purpose
-------
This layer watches ALL of the day's symbols, not just skipped orders.

It builds a live watch universe from:
    - today's rebalance order file
    - today's pending direction watchlist
    - today's LLM sentiment universe
    - today's locked trading universe
    - today's premarket trade plan

It then watches streamed/polled 1-minute bars and promotes symbols only after
movement is confirmed.

This solves the case where run.py rejects buys for:

    price_direction_not_confirmed:
        combined_direction_weak
        lower_low_structure
        active_pullback
        ml_next_bar_prob_low

Instead of waiting for only those rejected rows, this layer keeps watching every
constituent of the day's universe and discovers symbols once movement confirms.

Default:
    python stream_confirmed_movement_gate.py --poll

Outputs:
    system-data/YYYYMMDD/YYYYMMDD_stream_confirmed_rebalance_orders.csv
    logs/execution/YYYYMMDD/stream_confirmed_movement_events.jsonl
    logs/execution/YYYYMMDD/stream_confirmed_movement_state.json
    logs/execution/YYYYMMDD/pending_direction_watchlist.csv

By default this does NOT submit orders directly. It writes confirmed order rows
for run.py to merge into execution.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from config.env_loader import load_instance_env

from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

try:
    from alpaca.trading.requests import TrailingStopOrderRequest
except Exception:
    TrailingStopOrderRequest = None


ET = ZoneInfo("America/New_York")
TODAY_TAG = datetime.now(ET).strftime("%Y%m%d")
DATA_TAG = os.getenv("RUN_DATA_DATE", os.getenv("STREAM_CONFIRM_DATA_DATE", TODAY_TAG))

PROJECT_ROOT = Path(__file__).resolve().parent
load_instance_env()

OUT_DIR = PROJECT_ROOT / "system-data" / DATA_TAG
LOG_DIR = PROJECT_ROOT / "logs" / "execution" / DATA_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

REBALANCE_ORDERS_PATH = OUT_DIR / f"{DATA_TAG}_daily_walkforward_emulated_rebalance_orders.csv"
PENDING_WATCHLIST_PATH = LOG_DIR / "pending_direction_watchlist.csv"
CONFIRMED_ORDERS_PATH = OUT_DIR / f"{DATA_TAG}_stream_confirmed_rebalance_orders.csv"
TRADE_PLAN_PATH = OUT_DIR / f"{DATA_TAG}_stream_confirmed_trade_plan.csv"
TRADE_PLAN_JSON_PATH = OUT_DIR / f"{DATA_TAG}_stream_confirmed_trade_plan.json"
EVENTS_PATH = LOG_DIR / "stream_confirmed_movement_events.jsonl"
STATE_PATH = LOG_DIR / "stream_confirmed_movement_state.json"

LLM_UNIVERSE_PATH = OUT_DIR / f"{DATA_TAG}_latest_llm_sentiment_universe.json"
LOCKED_UNIVERSE_PATH = PROJECT_ROOT / f"{DATA_TAG}_locked_trading_universe.txt"
PREMARKET_PLAN_PATH = OUT_DIR / f"{DATA_TAG}_premarket_trade_plan.json"

ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex").lower().strip()
PAPER = os.getenv("APCA_PAPER", "1").lower() in {"1", "true", "yes"}

STREAM_CONFIRM_EXECUTE = os.getenv("STREAM_CONFIRM_EXECUTE", "0").lower() in {"1", "true", "yes"}
SCAN_RELOAD_SECONDS = int(os.getenv("STREAM_CONFIRM_RELOAD_SECONDS", "10"))
MAX_WATCH_SYMBOLS = int(os.getenv("STREAM_CONFIRM_MAX_WATCH_SYMBOLS", "75"))

MIN_1M_DIRECTION_SCORE = float(os.getenv("STREAM_CONFIRM_MIN_1M_SCORE", "0.62"))
MIN_5M_DIRECTION_SCORE = float(os.getenv("STREAM_CONFIRM_MIN_5M_SCORE", "0.58"))
MIN_15M_DIRECTION_SCORE = float(os.getenv("STREAM_CONFIRM_MIN_15M_SCORE", "0.52"))
MIN_COMBINED_DIRECTION_SCORE = float(os.getenv("STREAM_CONFIRM_MIN_COMBINED_SCORE", "0.64"))
MIN_REL_VOLUME = float(os.getenv("STREAM_CONFIRM_MIN_REL_VOLUME", "0.85"))

# Early multibar-gain detection.
# This tries to catch the start of continuation, before the move is too extended.
EARLY_CONTINUATION_MIN_SCORE = float(os.getenv("STREAM_CONFIRM_EARLY_CONTINUATION_MIN_SCORE", "0.68"))
EARLY_CONTINUATION_MIN_3BAR_RETURN = float(os.getenv("STREAM_CONFIRM_MIN_3BAR_RETURN", "0.0012"))
EARLY_CONTINUATION_MIN_5BAR_RETURN = float(os.getenv("STREAM_CONFIRM_MIN_5BAR_RETURN", "0.0018"))
EARLY_CONTINUATION_MAX_10BAR_RETURN = float(os.getenv("STREAM_CONFIRM_MAX_10BAR_RETURN", "0.018"))
EARLY_CONTINUATION_MIN_VOLUME_ACCEL = float(os.getenv("STREAM_CONFIRM_MIN_VOLUME_ACCEL", "1.05"))
EARLY_CONTINUATION_REQUIRE_MACD_UPSLOPE = os.getenv("STREAM_CONFIRM_REQUIRE_MACD_UPSLOPE", "1").lower() in {"1", "true", "yes"}
EARLY_CONTINUATION_TOP_N = int(os.getenv("STREAM_CONFIRM_TOP_N", "8"))

MIN_BREAKOUT_BUFFER_PCT = float(os.getenv("STREAM_CONFIRM_BREAKOUT_BUFFER_PCT", "0.0015"))
MAX_PULLBACK_FROM_HIGH_PCT = float(os.getenv("STREAM_CONFIRM_MAX_PULLBACK_FROM_HIGH_PCT", "-0.004"))
MAX_WATCH_MINUTES = int(os.getenv("STREAM_CONFIRM_MAX_WATCH_MINUTES", "240"))

ALLOW_ABOVE_VWAP_REQUIRED = os.getenv("STREAM_CONFIRM_REQUIRE_ABOVE_VWAP", "1").lower() in {"1", "true", "yes"}
REQUIRE_HIGHER_HIGH = os.getenv("STREAM_CONFIRM_REQUIRE_HIGHER_HIGH", "1").lower() in {"1", "true", "yes"}
REQUIRE_NO_LOWER_LOW = os.getenv("STREAM_CONFIRM_REQUIRE_NO_LOWER_LOW", "1").lower() in {"1", "true", "yes"}

DEFAULT_ORDER_DOLLARS = float(os.getenv("STREAM_CONFIRM_DEFAULT_ORDER_DOLLARS", "1500"))
MAX_ORDER_DOLLARS = float(os.getenv("STREAM_CONFIRM_MAX_ORDER_DOLLARS", "6500"))
MIN_ORDER_DOLLARS = float(os.getenv("STREAM_CONFIRM_MIN_ORDER_DOLLARS", "100"))
TIME_IN_FORCE = TimeInForce.DAY

# Trade-plan/risk parameters for confirmed early multibar setups.
TRADE_RISK_PCT = float(os.getenv("STREAM_CONFIRM_TRADE_RISK_PCT", "0.006"))
TAKE_PROFIT_R_MULTIPLE = float(os.getenv("STREAM_CONFIRM_TAKE_PROFIT_R", "2.0"))
ATR_STOP_MULT = float(os.getenv("STREAM_CONFIRM_ATR_STOP_MULT", "1.2"))
MIN_STOP_PCT = float(os.getenv("STREAM_CONFIRM_MIN_STOP_PCT", "0.004"))
MAX_STOP_PCT = float(os.getenv("STREAM_CONFIRM_MAX_STOP_PCT", "0.012"))

# Alpaca trailing-stop order settings. Alpaca supports trailing_stop orders for
# exits. This file emits trailing-stop instructions for run.py and can submit
# them directly when STREAM_CONFIRM_EXECUTE=1.
TRAILING_STOP_ENABLED = os.getenv("STREAM_CONFIRM_TRAILING_STOP_ENABLED", "1").lower() in {"1", "true", "yes"}
TRAIL_PERCENT = float(os.getenv("STREAM_CONFIRM_TRAIL_PERCENT", "0.65"))
TRAIL_PRICE_MIN = float(os.getenv("STREAM_CONFIRM_TRAIL_PRICE_MIN", "0.05"))


@dataclass
class WatchSymbol:
    symbol: str
    source: str
    side: str = "buy"
    order_qty: int = 0
    price: float = 0.0
    capped_dollars: float = 0.0
    confidence_used: float = 0.0
    reason: str = ""
    added_at_utc: str = ""


@dataclass
class MovementState:
    symbol: str
    last_price: float = 0.0
    session_high: float = 0.0
    session_low: float = 0.0
    prev_swing_high: float = 0.0
    prev_swing_low: float = 0.0
    vwap: float = 0.0
    rel_volume: float = 0.0
    atr_1m: float = 0.0
    direction_1m: float = 0.0
    direction_5m: float = 0.0
    direction_15m: float = 0.0
    combined_direction: float = 0.0
    setup_type: str = "none"
    early_continuation_score: float = 0.0
    ret_3bar: float = 0.0
    ret_5bar: float = 0.0
    ret_10bar: float = 0.0
    volume_accel: float = 0.0
    macd_upslope: bool = False
    not_overextended: bool = True
    confirmed: bool = False
    confirm_reason: str = ""


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_utc_iso() -> str:
    return now_utc().isoformat()


def f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return default
        return y
    except Exception:
        return default


def i(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def normalize_symbol(x: Any) -> str:
    return str(x or "").upper().strip().replace(".", "-")


def safe_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    return value


def write_event(event: str, payload: dict[str, Any] | None = None) -> None:
    row = {
        "timestamp_utc": now_utc_iso(),
        "event": event,
        "payload": safe_json(payload or {}),
    }
    with EVENTS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=safe_json) + "\n")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe_json(payload), indent=2) + "\n", encoding="utf-8")


def get_credentials() -> tuple[str, str]:
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY")
    return key, secret


def get_feed() -> DataFeed:
    return DataFeed.SIP if ALPACA_DATA_FEED == "sip" else DataFeed.IEX


def data_client() -> StockHistoricalDataClient:
    key, secret = get_credentials()
    return StockHistoricalDataClient(key, secret)


def trading_client() -> TradingClient:
    key, secret = get_credentials()
    return TradingClient(key, secret, paper=PAPER)


def append_pending_direction_watch(
    order_row: dict[str, Any] | pd.Series,
    reject_reason: str,
    path: Path = PENDING_WATCHLIST_PATH,
) -> None:
    if isinstance(order_row, pd.Series):
        data = order_row.to_dict()
    else:
        data = dict(order_row)

    symbol = normalize_symbol(data.get("symbol"))
    side = str(data.get("side", "buy")).lower()

    if not symbol or side != "buy":
        return

    row = {
        "symbol": symbol,
        "side": "buy",
        "order_qty": i(data.get("order_qty", data.get("qty", 0))),
        "price": f(data.get("price")),
        "capped_dollars": f(data.get("capped_dollars", data.get("delta_dollars", 0.0))),
        "confidence_used": f(data.get("confidence_used", data.get("trade_confidence", 0.0))),
        "reason": str(reject_reason),
        "rejected_at_utc": now_utc_iso(),
        "source": "direction_reject",
    }

    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, index=False, mode="a", header=header)
    write_event("pending_direction_watch_added", row)


def _watch_from_row(row: pd.Series, source: str) -> WatchSymbol | None:
    symbol = normalize_symbol(row.get("symbol"))
    if not symbol:
        return None

    return WatchSymbol(
        symbol=symbol,
        source=source,
        side="buy",
        order_qty=i(row.get("order_qty", row.get("qty", 0))),
        price=f(row.get("price", row.get("last_price", 0.0))),
        capped_dollars=f(row.get("capped_dollars", row.get("target_dollars", row.get("abs_delta_dollars", DEFAULT_ORDER_DOLLARS)))),
        confidence_used=f(row.get("confidence_used", row.get("trade_confidence", row.get("confidence", 0.0)))),
        reason=str(row.get("reason", source)),
        added_at_utc=str(row.get("rejected_at_utc", now_utc_iso())),
    )


def load_rebalance_symbols() -> dict[str, WatchSymbol]:
    out: dict[str, WatchSymbol] = {}
    if not REBALANCE_ORDERS_PATH.exists():
        return out

    try:
        df = pd.read_csv(REBALANCE_ORDERS_PATH)
    except Exception:
        return out

    if df.empty or "symbol" not in df.columns:
        return out

    for _, row in df.iterrows():
        side = str(row.get("side", "buy")).lower()
        if side != "buy":
            continue
        item = _watch_from_row(row, "rebalance_orders")
        if item:
            out[item.symbol] = item

    return out


def load_pending_symbols() -> dict[str, WatchSymbol]:
    out: dict[str, WatchSymbol] = {}
    if not PENDING_WATCHLIST_PATH.exists():
        return out

    try:
        df = pd.read_csv(PENDING_WATCHLIST_PATH)
    except Exception:
        return out

    if df.empty or "symbol" not in df.columns:
        return out

    cutoff = now_utc() - timedelta(minutes=MAX_WATCH_MINUTES)

    for _, row in df.iterrows():
        rejected_at = pd.to_datetime(row.get("rejected_at_utc"), utc=True, errors="coerce")
        if not pd.isna(rejected_at) and rejected_at.to_pydatetime() < cutoff:
            continue
        item = _watch_from_row(row, "pending_direction_watchlist")
        if item:
            out[item.symbol] = item

    return out


def load_llm_symbols() -> dict[str, WatchSymbol]:
    out: dict[str, WatchSymbol] = {}
    payload = read_json(LLM_UNIVERSE_PATH, {})
    symbols: list[str] = []

    if isinstance(payload, dict):
        if isinstance(payload.get("symbols"), list):
            symbols.extend(payload.get("symbols", []))
        if isinstance(payload.get("selections"), list):
            for row in payload.get("selections", []):
                if isinstance(row, dict):
                    symbols.append(row.get("symbol"))

    for sym in symbols:
        symbol = normalize_symbol(sym)
        if symbol:
            out[symbol] = WatchSymbol(
                symbol=symbol,
                source="llm_sentiment_universe",
                capped_dollars=DEFAULT_ORDER_DOLLARS,
                confidence_used=0.50,
                reason="daily_universe_member",
                added_at_utc=now_utc_iso(),
            )

    return out


def load_locked_universe_symbols() -> dict[str, WatchSymbol]:
    out: dict[str, WatchSymbol] = {}
    if not LOCKED_UNIVERSE_PATH.exists():
        return out

    try:
        raw = LOCKED_UNIVERSE_PATH.read_text(encoding="utf-8")
    except Exception:
        return out

    for token in raw.replace("\n", ",").split(","):
        symbol = normalize_symbol(token)
        if symbol:
            out[symbol] = WatchSymbol(
                symbol=symbol,
                source="locked_trading_universe",
                capped_dollars=DEFAULT_ORDER_DOLLARS,
                confidence_used=0.50,
                reason="locked_universe_member",
                added_at_utc=now_utc_iso(),
            )

    return out


def load_premarket_plan_symbols() -> dict[str, WatchSymbol]:
    out: dict[str, WatchSymbol] = {}
    payload = read_json(PREMARKET_PLAN_PATH, {})

    if not isinstance(payload, dict):
        return out

    rows = []
    for key in ["daily_trade_plans", "trade_plans", "plans", "candidates"]:
        if isinstance(payload.get(key), list):
            rows.extend(payload.get(key))

    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = normalize_symbol(row.get("symbol"))
        if not symbol:
            continue
        valid = bool(row.get("valid_plan", row.get("is_valid", True)))
        if not valid:
            continue
        out[symbol] = WatchSymbol(
            symbol=symbol,
            source="premarket_trade_plan",
            price=f(row.get("entry_price", row.get("price", 0.0))),
            capped_dollars=DEFAULT_ORDER_DOLLARS,
            confidence_used=f(row.get("confidence", row.get("score", 0.50)), 0.50),
            reason="premarket_plan_member",
            added_at_utc=now_utc_iso(),
        )

    return out


def load_watch_universe() -> dict[str, WatchSymbol]:
    """
    Full day-symbol universe.

    Priority:
      pending rejected orders > rebalance orders > premarket plans > LLM universe > locked universe
    """
    merged: dict[str, WatchSymbol] = {}

    for source_loader in [
        load_locked_universe_symbols,
        load_llm_symbols,
        load_premarket_plan_symbols,
        load_rebalance_symbols,
        load_pending_symbols,
    ]:
        merged.update(source_loader())

    if len(merged) > MAX_WATCH_SYMBOLS:
        # Keep actual order rows first, then truncate.
        ordered = sorted(
            merged.values(),
            key=lambda x: (
                0 if x.source in {"pending_direction_watchlist", "rebalance_orders"} else 1,
                -x.confidence_used,
                x.symbol,
            ),
        )[:MAX_WATCH_SYMBOLS]
        merged = {x.symbol: x for x in ordered}

    return merged


def remove_pending_symbol(symbol: str) -> None:
    symbol = normalize_symbol(symbol)
    if not PENDING_WATCHLIST_PATH.exists() or not symbol:
        return

    try:
        df = pd.read_csv(PENDING_WATCHLIST_PATH)
    except Exception:
        return

    if df.empty or "symbol" not in df.columns:
        return

    df = df[df["symbol"].map(normalize_symbol) != symbol].copy()
    df.to_csv(PENDING_WATCHLIST_PATH, index=False)


def already_confirmed_today(symbol: str) -> bool:
    symbol = normalize_symbol(symbol)
    if not CONFIRMED_ORDERS_PATH.exists():
        return False
    try:
        df = pd.read_csv(CONFIRMED_ORDERS_PATH)
    except Exception:
        return False
    if df.empty or "symbol" not in df.columns:
        return False
    return symbol in set(df["symbol"].map(normalize_symbol))


def fetch_recent_1m(symbol: str, lookback_days: int = 2) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=get_feed(),
    )
    try:
        df = data_client().get_stock_bars(req).df
    except Exception as exc:
        _preview = str(exc)[:300]
        print(f"[stream-confirm] bar fetch failed symbol={symbol}: {exc}")
        write_event("external_http_error", {
            "module": "stream_confirmed_movement_gate",
            "symbol": symbol,
            "url_host": "api.alpaca.markets",
            "status_code": getattr(getattr(exc, "response", None), "status_code", None),
            "response_preview": _preview,
        })
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    if "symbol" in df.columns:
        df = df[df["symbol"].astype(str).str.upper() == symbol.upper()].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy().sort_values("timestamp").reset_index(drop=True)
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    vol = out["volume"].astype(float)

    out["ret_1"] = close.pct_change()
    out["ret_3"] = close.pct_change(3)
    out["ret_5"] = close.pct_change(5)

    out["sma5"] = close.rolling(5).mean()
    out["sma20"] = close.rolling(20).mean()
    out["dist_sma5"] = close / out["sma5"] - 1.0
    out["dist_sma20"] = close / out["sma20"] - 1.0

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi14"] = 100.0 - (100.0 / (1.0 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()
    out["atr_pct"] = out["atr14"] / close

    typical = (high + low + close) / 3.0
    pv = typical * vol
    out["vwap"] = pv.cumsum() / vol.replace(0, np.nan).cumsum()

    out["rel_volume"] = vol / vol.rolling(20).mean().replace(0, np.nan)
    out["rolling_high_10"] = high.rolling(10).max()
    out["rolling_low_10"] = low.rolling(10).min()
    out["rolling_high_20"] = high.rolling(20).max()
    out["rolling_low_20"] = low.rolling(20).min()

    return out.replace([np.inf, -np.inf], np.nan)


def resample_features(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df_1m.empty:
        return pd.DataFrame()

    x = df_1m.copy().set_index("timestamp").sort_index()
    out = pd.DataFrame({
        "open": x["open"].resample(rule).first(),
        "high": x["high"].resample(rule).max(),
        "low": x["low"].resample(rule).min(),
        "close": x["close"].resample(rule).last(),
        "volume": x["volume"].resample(rule).sum(),
    }).dropna().reset_index()
    return add_features(out)


def direction_score(row: pd.Series) -> float:
    parts = [
        f(row.get("ret_1")) > 0,
        f(row.get("ret_3")) > 0,
        f(row.get("macd_hist")) > 0,
        f(row.get("rsi14"), 50.0) >= 50.0,
        f(row.get("close")) >= f(row.get("vwap"), f(row.get("close"))),
        f(row.get("dist_sma5")) >= -0.001,
    ]
    return float(np.mean([1.0 if p else 0.0 for p in parts]))



def early_continuation_features(bars: pd.DataFrame) -> dict[str, Any]:
    """
    Detect the start of a multibar gain.

    We want the first clean continuation phase:
      - 3-bar and 5-bar returns are positive
      - 10-bar return is not already too extended
      - MACD histogram is improving
      - volume is accelerating
      - price is above VWAP/SMA5 without a lower-low break
    """
    if bars.empty or len(bars) < 25:
        return {
            "ret_3bar": 0.0,
            "ret_5bar": 0.0,
            "ret_10bar": 0.0,
            "volume_accel": 0.0,
            "macd_upslope": False,
            "not_overextended": True,
            "score": 0.0,
        }

    row = bars.iloc[-1]
    close = bars["close"].astype(float)
    volume = bars["volume"].astype(float)

    ret_3 = close.iloc[-1] / max(close.iloc[-4], 1e-9) - 1.0 if len(close) >= 4 else 0.0
    ret_5 = close.iloc[-1] / max(close.iloc[-6], 1e-9) - 1.0 if len(close) >= 6 else 0.0
    ret_10 = close.iloc[-1] / max(close.iloc[-11], 1e-9) - 1.0 if len(close) >= 11 else 0.0

    vol_recent = volume.tail(3).mean()
    vol_base = volume.tail(20).head(17).mean()
    volume_accel = float(vol_recent / max(vol_base, 1e-9)) if vol_base > 0 else 0.0

    macd_hist = bars["macd_hist"].astype(float).fillna(0.0)
    macd_upslope = bool(len(macd_hist) >= 4 and macd_hist.iloc[-1] > macd_hist.iloc[-2] > macd_hist.iloc[-3])

    price = f(row.get("close"))
    vwap = f(row.get("vwap"), price)
    sma5 = f(row.get("sma5"), price)
    sma20 = f(row.get("sma20"), price)
    above_vwap = price >= vwap
    above_sma5 = price >= sma5
    sma5_above_sma20 = sma5 >= sma20

    recent_low = f(bars.tail(5)["low"].min(), price)
    prior_low = f(bars.tail(12).head(7)["low"].min(), price)
    no_lower_low = recent_low >= prior_low * (1.0 + MAX_PULLBACK_FROM_HIGH_PCT)

    not_overextended = ret_10 <= EARLY_CONTINUATION_MAX_10BAR_RETURN

    score = float(np.clip(np.mean([
        ret_3 >= EARLY_CONTINUATION_MIN_3BAR_RETURN,
        ret_5 >= EARLY_CONTINUATION_MIN_5BAR_RETURN,
        not_overextended,
        volume_accel >= EARLY_CONTINUATION_MIN_VOLUME_ACCEL,
        macd_upslope or not EARLY_CONTINUATION_REQUIRE_MACD_UPSLOPE,
        above_vwap,
        above_sma5,
        sma5_above_sma20,
        no_lower_low,
    ]), 0.0, 1.0))

    # Add a small continuous component so stronger-but-not-overextended names rank first.
    score = float(np.clip(
        score * 0.78
        + min(max(ret_3 / 0.006, 0.0), 1.0) * 0.08
        + min(max(ret_5 / 0.010, 0.0), 1.0) * 0.08
        + min(max(volume_accel / 2.0, 0.0), 1.0) * 0.06,
        0.0,
        1.0,
    ))

    return {
        "ret_3bar": float(ret_3),
        "ret_5bar": float(ret_5),
        "ret_10bar": float(ret_10),
        "volume_accel": float(volume_accel),
        "macd_upslope": bool(macd_upslope),
        "not_overextended": bool(not_overextended),
        "score": float(score),
    }


def evaluate_symbol(symbol: str, watch: WatchSymbol) -> MovementState:
    bars = add_features(fetch_recent_1m(symbol))
    if bars.empty or len(bars) < 30:
        return MovementState(symbol=symbol, confirm_reason="not_enough_bars")

    bars_5m = resample_features(bars, "5min")
    bars_15m = resample_features(bars, "15min")
    early = early_continuation_features(bars)

    row = bars.iloc[-1]
    row5 = bars_5m.iloc[-1] if not bars_5m.empty else row
    row15 = bars_15m.iloc[-1] if not bars_15m.empty else row

    price = f(row.get("close"))
    vwap = f(row.get("vwap"), price)
    rel_volume = f(row.get("rel_volume"), 1.0)
    atr_1m = f(row.get("atr14"), price * 0.002)

    d1 = direction_score(row)
    d5 = direction_score(row5)
    d15 = direction_score(row15)
    combined = 0.54 * d1 + 0.31 * d5 + 0.15 * d15

    recent = bars.tail(20)
    prev = bars.tail(30).head(20)

    recent_high = f(recent["high"].max(), price)
    recent_low = f(recent["low"].min(), price)
    prev_high = f(prev["high"].max(), price)
    prev_low = f(prev["low"].min(), price)

    higher_high = recent_high > prev_high * (1.0 + MIN_BREAKOUT_BUFFER_PCT)
    no_lower_low = recent_low >= prev_low * (1.0 + MAX_PULLBACK_FROM_HIGH_PCT)
    above_vwap = price >= vwap
    has_volume = rel_volume >= MIN_REL_VOLUME

    early_continuation = early["score"] >= EARLY_CONTINUATION_MIN_SCORE
    reclaim = above_vwap and d1 >= MIN_1M_DIRECTION_SCORE and d5 >= 0.50
    breakout = higher_high and above_vwap and combined >= MIN_COMBINED_DIRECTION_SCORE
    reversal = no_lower_low and d1 >= 0.67 and f(row.get("ret_3")) > 0

    setup_type = (
        "early_multibar_continuation" if early_continuation
        else "breakout" if breakout
        else "vwap_reclaim" if reclaim
        else "reversal_reclaim" if reversal
        else "none"
    )

    confirmed = (
        d1 >= MIN_1M_DIRECTION_SCORE
        and d5 >= MIN_5M_DIRECTION_SCORE
        and d15 >= MIN_15M_DIRECTION_SCORE
        and combined >= MIN_COMBINED_DIRECTION_SCORE
        and has_volume
        and setup_type != "none"
        and early["not_overextended"]
    )

    reasons = []
    if d1 < MIN_1M_DIRECTION_SCORE:
        reasons.append(f"1m_weak:{d1:.2f}")
    if d5 < MIN_5M_DIRECTION_SCORE:
        reasons.append(f"5m_weak:{d5:.2f}")
    if d15 < MIN_15M_DIRECTION_SCORE:
        reasons.append(f"15m_weak:{d15:.2f}")
    if combined < MIN_COMBINED_DIRECTION_SCORE:
        reasons.append(f"combined_weak:{combined:.2f}")
    if ALLOW_ABOVE_VWAP_REQUIRED and not above_vwap:
        confirmed = False
        reasons.append("below_vwap")
    if REQUIRE_HIGHER_HIGH and setup_type == "breakout" and not higher_high:
        confirmed = False
        reasons.append("no_higher_high")
    if REQUIRE_NO_LOWER_LOW and not no_lower_low:
        confirmed = False
        reasons.append("lower_low_still_active")
    if not has_volume:
        reasons.append(f"rel_volume_low:{rel_volume:.2f}")
    if early["score"] < EARLY_CONTINUATION_MIN_SCORE:
        reasons.append(f"early_continuation_score_low:{early['score']:.2f}")
    if not early["not_overextended"]:
        confirmed = False
        reasons.append(f"already_extended_10bar:{early['ret_10bar']:.2%}")
    if EARLY_CONTINUATION_REQUIRE_MACD_UPSLOPE and not early["macd_upslope"]:
        reasons.append("macd_not_yet_upslope")
    if setup_type == "none":
        reasons.append("no_confirmed_setup_type")

    if confirmed:
        reasons.append(
            f"confirmed_{setup_type}:price={price:.2f}:d1={d1:.2f}:d5={d5:.2f}:"
            f"d15={d15:.2f}:combined={combined:.2f}:early={early['score']:.2f}:"
            f"ret3={early['ret_3bar']:.2%}:ret5={early['ret_5bar']:.2%}:"
            f"volaccel={early['volume_accel']:.2f}"
        )

    return MovementState(
        symbol=symbol,
        last_price=price,
        session_high=recent_high,
        session_low=recent_low,
        prev_swing_high=prev_high,
        prev_swing_low=prev_low,
        vwap=vwap,
        rel_volume=rel_volume,
        atr_1m=atr_1m,
        direction_1m=d1,
        direction_5m=d5,
        direction_15m=d15,
        combined_direction=combined,
        setup_type=setup_type,
        early_continuation_score=float(early["score"]),
        ret_3bar=float(early["ret_3bar"]),
        ret_5bar=float(early["ret_5bar"]),
        ret_10bar=float(early["ret_10bar"]),
        volume_accel=float(early["volume_accel"]),
        macd_upslope=bool(early["macd_upslope"]),
        not_overextended=bool(early["not_overextended"]),
        confirmed=confirmed,
        confirm_reason=",".join(reasons),
    )



def build_trade_plan(watch: WatchSymbol, state: MovementState) -> dict[str, Any]:
    """
    Build entry / stop / take-profit / trailing-stop plan for a confirmed symbol.

    Entry uses the latest confirmed streamed price as the reference. Stops use
    the larger of:
      - configured minimum stop percentage
      - ATR-based stop
    capped by MAX_STOP_PCT to avoid oversized risk.

    Take profit is expressed as R-multiple from entry to stop.
    A trailing stop is also emitted so run.py can place/update a trailing exit.
    """
    entry = max(float(state.last_price), 0.01)
    atr_stop_pct = (max(float(state.atr_1m), entry * 0.001) * ATR_STOP_MULT) / entry
    stop_pct = min(max(max(atr_stop_pct, MIN_STOP_PCT), TRADE_RISK_PCT), MAX_STOP_PCT)

    stop_loss = entry * (1.0 - stop_pct)
    risk_per_share = entry - stop_loss
    take_profit = entry + risk_per_share * TAKE_PROFIT_R_MULTIPLE

    trail_percent = max(0.01, float(TRAIL_PERCENT))
    trail_price = max(TRAIL_PRICE_MIN, entry * (trail_percent / 100.0))

    capped = min(max(watch.capped_dollars or DEFAULT_ORDER_DOLLARS, MIN_ORDER_DOLLARS), MAX_ORDER_DOLLARS)
    qty = watch.order_qty if watch.order_qty > 0 else int(capped // entry)
    qty = max(0, int(qty))

    plan = {
        "symbol": watch.symbol,
        "side": "buy",
        "setup_type": state.setup_type,
        "entry_price": round(entry, 4),
        "stop_loss": round(stop_loss, 4),
        "take_profit": round(take_profit, 4),
        "risk_per_share": round(risk_per_share, 4),
        "risk_pct": round(stop_pct, 6),
        "take_profit_r_multiple": TAKE_PROFIT_R_MULTIPLE,
        "qty": qty,
        "notional": round(qty * entry, 2),
        "trailing_stop_enabled": bool(TRAILING_STOP_ENABLED),
        "trail_percent": round(trail_percent, 4),
        "trail_price": round(trail_price, 4),
        "time_in_force": str(TIME_IN_FORCE.value if hasattr(TIME_IN_FORCE, "value") else TIME_IN_FORCE),
        "detected_at_utc": now_utc_iso(),
        "source": watch.source,
        "reason": f"early_multibar_trade_plan:{state.confirm_reason}",
        "direction_score": state.combined_direction,
        "direction_score_1m": state.direction_1m,
        "direction_score_5m": state.direction_5m,
        "direction_score_15m": state.direction_15m,
        "early_continuation_score": state.early_continuation_score,
        "ret_3bar": state.ret_3bar,
        "ret_5bar": state.ret_5bar,
        "ret_10bar": state.ret_10bar,
        "volume_accel": state.volume_accel,
        "vwap": state.vwap,
        "rel_volume": state.rel_volume,
        "atr_1m": state.atr_1m,
        "order_type": "market_entry_with_trailing_stop_exit",
    }
    return plan


def append_trade_plan(plan: dict[str, Any]) -> None:
    df = pd.DataFrame([safe_json(plan)])
    header = not TRADE_PLAN_PATH.exists()
    df.to_csv(TRADE_PLAN_PATH, index=False, mode="a", header=header)

    payload = read_json(TRADE_PLAN_JSON_PATH, [])
    if not isinstance(payload, list):
        payload = []
    # Avoid duplicate symbols in same generated plan file. Keep latest.
    payload = [x for x in payload if normalize_symbol(x.get("symbol")) != normalize_symbol(plan.get("symbol"))]
    payload.append(safe_json(plan))
    write_json(TRADE_PLAN_JSON_PATH, payload)
    write_event("stream_confirmed_trade_plan", plan)


def submit_trailing_stop_exit(symbol: str, qty: int, trail_percent: float, trail_price: float) -> None:
    """
    Submit a trailing stop exit order for a filled/held position.

    This is best-effort because a trailing stop submitted before the entry fill
    may be rejected by Alpaca. In normal run.py integration, run.py should submit
    the trailing stop after confirming the entry fill.
    """
    if not TRAILING_STOP_ENABLED or qty <= 0:
        return

    if TrailingStopOrderRequest is None:
        write_event("trailing_stop_unavailable", {"symbol": symbol, "reason": "TrailingStopOrderRequest missing"})
        return

    try:
        client = trading_client()
        req = TrailingStopOrderRequest(
            symbol=symbol,
            qty=int(qty),
            side=OrderSide.SELL,
            time_in_force=TIME_IN_FORCE,
            trail_percent=str(round(trail_percent, 4)),
        )
        order = client.submit_order(req)
        write_event(
            "stream_confirmed_trailing_stop_submitted",
            {
                "symbol": symbol,
                "qty": int(qty),
                "trail_percent": trail_percent,
                "trail_price_reference": trail_price,
                "order_id": getattr(order, "id", ""),
                "status": str(getattr(order, "status", "")),
            },
        )
    except Exception as exc:
        write_event(
            "stream_confirmed_trailing_stop_error",
            {
                "symbol": symbol,
                "qty": int(qty),
                "trail_percent": trail_percent,
                "error": str(exc),
            },
        )

def append_confirmed_order(watch: WatchSymbol, state: MovementState) -> None:
    if already_confirmed_today(watch.symbol):
        return

    plan = build_trade_plan(watch, state)
    qty = int(plan["qty"])

    if qty <= 0:
        write_event("stream_confirm_qty_zero", {"symbol": watch.symbol, "state": asdict(state), "plan": plan})
        return

    append_trade_plan(plan)

    row = {
        "symbol": watch.symbol,
        "side": "buy",
        "order_qty": qty,
        "price": float(plan["entry_price"]),
        "entry_price": float(plan["entry_price"]),
        "stop_loss": float(plan["stop_loss"]),
        "take_profit": float(plan["take_profit"]),
        "risk_per_share": float(plan["risk_per_share"]),
        "risk_pct": float(plan["risk_pct"]),
        "take_profit_r_multiple": float(plan["take_profit_r_multiple"]),
        "trailing_stop_enabled": bool(plan["trailing_stop_enabled"]),
        "trail_percent": float(plan["trail_percent"]),
        "trail_price": float(plan["trail_price"]),
        "abs_delta_dollars": float(plan["notional"]),
        "capped_dollars": float(min(plan["notional"], MAX_ORDER_DOLLARS)),
        "confidence_used": float(max(watch.confidence_used, state.combined_direction, state.early_continuation_score)),
        "confidence_multiplier": float(1.0 + max(watch.confidence_used, state.combined_direction, state.early_continuation_score)),
        "reason": f"stream_confirmed_{state.setup_type}:{state.confirm_reason}:source={watch.source}",
        "stream_confirmed": True,
        "stream_confirmed_at_utc": now_utc_iso(),
        "source": watch.source,
        "setup_type": state.setup_type,
        "early_continuation_score": state.early_continuation_score,
        "ret_3bar": state.ret_3bar,
        "ret_5bar": state.ret_5bar,
        "ret_10bar": state.ret_10bar,
        "volume_accel": state.volume_accel,
        "macd_upslope": state.macd_upslope,
        "direction_score": state.combined_direction,
        "direction_score_1m": state.direction_1m,
        "direction_score_5m": state.direction_5m,
        "direction_score_15m": state.direction_15m,
        "rel_volume": state.rel_volume,
        "vwap": state.vwap,
        "order_type": "market_entry_with_trailing_stop_exit",
    }

    df = pd.DataFrame([row])
    header = not CONFIRMED_ORDERS_PATH.exists()
    df.to_csv(CONFIRMED_ORDERS_PATH, index=False, mode="a", header=header)
    write_event("stream_confirmed_order", row)


def submit_order_now(watch: WatchSymbol, state: MovementState) -> None:
    plan = build_trade_plan(watch, state)
    qty = int(plan["qty"])

    if qty <= 0:
        return

    client = trading_client()
    req = MarketOrderRequest(
        symbol=watch.symbol,
        qty=int(qty),
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        time_in_force=TIME_IN_FORCE,
    )

    order = client.submit_order(req)
    write_event(
        "stream_confirmed_order_submitted",
        {
            "symbol": watch.symbol,
            "qty": int(qty),
            "price_ref": plan["entry_price"],
            "trade_plan": plan,
            "state": asdict(state),
            "order_id": getattr(order, "id", ""),
            "status": str(getattr(order, "status", "")),
        },
    )

    # Best-effort. If Alpaca rejects because the entry has not filled yet,
    # run.py should submit the trailing stop after fill confirmation using the
    # emitted trade plan.
    submit_trailing_stop_exit(
        symbol=watch.symbol,
        qty=qty,
        trail_percent=float(plan["trail_percent"]),
        trail_price=float(plan["trail_price"]),
    )


def promote_if_confirmed(symbol: str, watch: WatchSymbol) -> None:
    if already_confirmed_today(symbol):
        return

    state = evaluate_symbol(symbol, watch)

    all_state = read_json(STATE_PATH, {})
    all_state[symbol] = asdict(state)
    write_json(STATE_PATH, all_state)

    print(
        f"[stream-confirm] {symbol} source={watch.source} confirmed={state.confirmed} "
        f"setup={state.setup_type} price={state.last_price:.2f} "
        f"early={state.early_continuation_score:.2f} ret3={state.ret_3bar:.2%} "
        f"ret5={state.ret_5bar:.2%} volaccel={state.volume_accel:.2f} "
        f"d1={state.direction_1m:.2f} d5={state.direction_5m:.2f} "
        f"d15={state.direction_15m:.2f} combined={state.combined_direction:.2f} "
        f"reason={state.confirm_reason}"
    )

    write_event("stream_confirm_evaluated", {"watch": asdict(watch), "state": asdict(state)})

    if not state.confirmed:
        return

    append_confirmed_order(watch, state)

    if STREAM_CONFIRM_EXECUTE:
        submit_order_now(watch, state)

    if watch.source == "pending_direction_watchlist":
        remove_pending_symbol(symbol)


_STOP = False


def _handle_stop(signum: int, frame: Any) -> None:
    global _STOP
    _STOP = True
    print("[stream-confirm] shutdown requested")


async def bar_handler(bar: Any) -> None:
    symbol = normalize_symbol(getattr(bar, "symbol", ""))
    if not symbol:
        return

    universe = load_watch_universe()
    if symbol not in universe:
        return

    promote_if_confirmed(symbol, universe[symbol])


async def subscription_manager(stream: StockDataStream) -> None:
    subscribed: set[str] = set()

    while not _STOP:
        universe = load_watch_universe()
        symbols = set(universe.keys())

        new_symbols = sorted(symbols - subscribed)
        if new_symbols:
            print(f"[stream-confirm] subscribing symbols={new_symbols}")
            stream.subscribe_bars(bar_handler, *new_symbols)
            subscribed.update(new_symbols)

        await asyncio.sleep(max(1, SCAN_RELOAD_SECONDS))


async def run_stream() -> None:
    key, secret = get_credentials()

    stream = StockDataStream(
        api_key=key,
        secret_key=secret,
        feed=get_feed(),
    )

    manager_task = asyncio.create_task(subscription_manager(stream))

    try:
        await stream._run_forever()
    finally:
        manager_task.cancel()


def run_polling_fallback() -> None:
    print("[stream-confirm] polling fallback active")
    while not _STOP:
        universe = load_watch_universe()
        print(f"[stream-confirm] watching symbols={len(universe)}")

        ranked: list[tuple[float, str, WatchSymbol, MovementState | None]] = []
        for symbol, row in universe.items():
            try:
                state = evaluate_symbol(symbol, row)
                ranked.append((state.early_continuation_score, symbol, row, state))
            except Exception as exc:
                print(f"[stream-confirm] {symbol} rank error={exc}")
                write_event("stream_confirm_symbol_error", {"symbol": symbol, "error": str(exc)})

        ranked.sort(key=lambda x: x[0], reverse=True)

        for _, symbol, row, precomputed_state in ranked[:max(1, EARLY_CONTINUATION_TOP_N)]:
            try:
                # Reuse the precomputed state to avoid a second fetch.
                state = precomputed_state
                if state is None:
                    promote_if_confirmed(symbol, row)
                    continue

                all_state = read_json(STATE_PATH, {})
                all_state[symbol] = asdict(state)
                write_json(STATE_PATH, all_state)

                print(
                    f"[stream-confirm] {symbol} source={row.source} confirmed={state.confirmed} "
                    f"setup={state.setup_type} price={state.last_price:.2f} "
                    f"early={state.early_continuation_score:.2f} ret3={state.ret_3bar:.2%} "
                    f"ret5={state.ret_5bar:.2%} volaccel={state.volume_accel:.2f} "
                    f"d1={state.direction_1m:.2f} d5={state.direction_5m:.2f} "
                    f"d15={state.direction_15m:.2f} combined={state.combined_direction:.2f} "
                    f"reason={state.confirm_reason}"
                )

                write_event("stream_confirm_evaluated", {"watch": asdict(row), "state": asdict(state)})

                if state.confirmed:
                    append_confirmed_order(row, state)
                    if STREAM_CONFIRM_EXECUTE:
                        submit_order_now(row, state)
                    if row.source == "pending_direction_watchlist":
                        remove_pending_symbol(symbol)

            except Exception as exc:
                print(f"[stream-confirm] {symbol} error={exc}")
                write_event("stream_confirm_symbol_error", {"symbol": symbol, "error": str(exc)})

        time.sleep(max(1, SCAN_RELOAD_SECONDS))


def main() -> None:
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    print("===== STREAM CONFIRMED MOVEMENT GATE =====")
    print(f"[paths] rebalance_orders={REBALANCE_ORDERS_PATH}")
    print(f"[paths] pending={PENDING_WATCHLIST_PATH}")
    print(f"[paths] confirmed_orders={CONFIRMED_ORDERS_PATH}")
    print(f"[paths] trade_plan_csv={TRADE_PLAN_PATH}")
    print(f"[paths] trade_plan_json={TRADE_PLAN_JSON_PATH}")
    print(f"[config] execute_now={STREAM_CONFIRM_EXECUTE}")
    print(f"[config] feed={ALPACA_DATA_FEED}")
    print(f"[config] max_watch_symbols={MAX_WATCH_SYMBOLS}")
    print(f"[config] early_continuation_min_score={EARLY_CONTINUATION_MIN_SCORE}")
    print(f"[config] top_n_ranked={EARLY_CONTINUATION_TOP_N}")
    print(f"[config] trailing_stop_enabled={TRAILING_STOP_ENABLED}")
    print(f"[config] take_profit_r={TAKE_PROFIT_R_MULTIPLE}")

    if "--poll" in sys.argv or os.getenv("STREAM_CONFIRM_USE_POLLING", "1") in {"1", "true", "yes"}:
        run_polling_fallback()
        return

    try:
        asyncio.run(run_stream())
    except Exception as exc:
        print(f"[stream-confirm] stream failed, falling back to polling: {exc}")
        write_event("stream_confirm_stream_failed", {"error": str(exc)})
        run_polling_fallback()


if __name__ == "__main__":
    main()
