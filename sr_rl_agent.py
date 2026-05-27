#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from config.env_loader import load_instance_env
from gymnasium import spaces
from stable_baselines3 import PPO
from zoneinfo import ZoneInfo

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


ET = ZoneInfo("America/New_York")
YESTERDAY_TAG = (datetime.now(ET) - timedelta(days=1)).strftime("%Y%m%d")
TODAY_TAG = datetime.now(ET).strftime("%Y%m%d")
DATA_TAG = os.getenv("RUN_DATA_DATE", os.getenv("STREAM_CONFIRM_DATA_DATE", os.getenv("RL_DATA_DATE", TODAY_TAG)))

PROJECT_ROOT = Path(__file__).resolve().parent
load_instance_env()

OUT_DIR = PROJECT_ROOT / "system-data" / f"{DATA_TAG}"
LOG_DIR = PROJECT_ROOT / "logs" / "execution"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

PREMARKET_TRADE_PLAN_PATH = OUT_DIR / f"{DATA_TAG}_premarket_trade_plan.json"
MARKET_CONTEXT_PATH = OUT_DIR / f"{DATA_TAG}_market_context.json"
MODEL_PATH = OUT_DIR / f"{DATA_TAG}_sr_rl_agent_ppo.zip"
SIGNALS_PATH = OUT_DIR / f"{DATA_TAG}_sr_rl_live_signals.json"
SHARED_BARS_1M_PATH = OUT_DIR / f"{DATA_TAG}_stream_confirmed_bars_1m.csv"

ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex").lower()

SCAN_SECONDS = int(os.getenv("SR_RL_SCAN_SECONDS", "60"))
LOOKBACK_DAYS = int(os.getenv("SR_RL_LOOKBACK_DAYS", "90"))
LEVEL_WINDOW = int(os.getenv("SR_RL_PIVOT_WINDOW", os.getenv("SR_RL_LEVEL_WINDOW", "5")))
LEVEL_CLUSTER_PCT = float(os.getenv("SR_RL_LEVEL_MERGE_PCT", os.getenv("SR_RL_LEVEL_CLUSTER_PCT", "0.003")))
ENTRY_DISTANCE_PCT = float(os.getenv("SR_RL_ENTRY_DISTANCE_PCT", "0.004"))
STOP_BEYOND_LEVEL_PCT = float(os.getenv("SR_RL_STOP_BEYOND_LEVEL_PCT", "0.01"))
TAKE_PROFIT_R_MULTIPLE = float(os.getenv("SR_RL_TAKE_PROFIT_R_MULTIPLE", "3.0"))
RISK_PER_TRADE_PCT = float(os.getenv("SR_RL_RISK_PER_TRADE_PCT", "0.005"))
MAX_POSITION_PCT = float(os.getenv("SR_RL_MAX_POSITION_PCT", "0.08"))

TRAIN_TIMESTEPS = int(os.getenv("SR_RL_TIMESTEPS", os.getenv("SR_RL_TRAIN_TIMESTEPS", "50000")))
TRAIN_LOOKBACK_DAYS = int(os.getenv("SR_RL_TRAIN_LOOKBACK_DAYS", "5"))
TRAIN_MIN_ROWS = int(os.getenv("SR_RL_TRAIN_MIN_ROWS", "500"))
MAX_EPISODE_BARS = int(os.getenv("SR_RL_MAX_EPISODE_BARS", "390"))

# PPO hyperparameters (tunable via Optuna)
PPO_LEARNING_RATE = float(os.getenv("SR_RL_LEARNING_RATE", "3e-4"))
PPO_GAMMA = float(os.getenv("SR_RL_GAMMA", "0.98"))
PPO_N_STEPS = int(os.getenv("SR_RL_N_STEPS", "512"))
PPO_BATCH_SIZE = int(os.getenv("SR_RL_BATCH_SIZE", "64"))
PPO_ENT_COEF = float(os.getenv("SR_RL_ENT_COEF", "0.01"))

# S/R touch buffers (tuner uses separate per-side values)
SUPPORT_TOUCH_BUFFER_PCT = float(os.getenv("SR_RL_SUPPORT_TOUCH_BUFFER_PCT", os.getenv("SR_RL_ENTRY_DISTANCE_PCT", "0.004")))
RESISTANCE_TOUCH_BUFFER_PCT = float(os.getenv("SR_RL_RESISTANCE_TOUCH_BUFFER_PCT", os.getenv("SR_RL_ENTRY_DISTANCE_PCT", "0.004")))
BREAKOUT_BUFFER_PCT = float(os.getenv("SR_RL_BREAKOUT_BUFFER_PCT", "0.003"))

# Live action confidence thresholds
MIN_ENTRY_CONFIDENCE = float(os.getenv("SR_RL_MIN_ENTRY_CONFIDENCE", "0.55"))
MIN_EXIT_CONFIDENCE = float(os.getenv("SR_RL_MIN_EXIT_CONFIDENCE", "0.50"))

# Set to True during train_agent() so bar-fetch functions can fall back to
# Alpaca REST history when the shared stream cache hasn't been populated yet
# (e.g. premarket training before the router starts writing bars).
_ALLOW_HISTORICAL_FALLBACK: bool = False

ACTION_WAIT = 0
ACTION_BUY = 1
ACTION_SELL = 2

RSI_PERIOD = int(os.getenv("SR_RL_RSI_PERIOD", "14"))
MACD_FAST = int(os.getenv("SR_RL_MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("SR_RL_MACD_SLOW", "26"))
MACD_SIGNAL = int(os.getenv("SR_RL_MACD_SIGNAL", "9"))
VOL_RATIO_PERIOD = int(os.getenv("SR_RL_VOL_RATIO_PERIOD", "20"))
VOL_RATIO_CAP = float(os.getenv("SR_RL_VOL_RATIO_CAP", "5.0"))

# ── Reward shaping constants ────────────────────────────────────────────────
# Entry quality
ENTRY_QUALITY_BONUS   = float(os.getenv("SR_RL_ENTRY_QUALITY_BONUS",   "0.08"))
ENTRY_QUALITY_PENALTY = float(os.getenv("SR_RL_ENTRY_QUALITY_PENALTY", "-0.06"))

# Per-bar green / red drift while in position
UNREALIZED_R_SCALE = float(os.getenv("SR_RL_UNREALIZED_R_SCALE", "0.003"))

# Violent single-bar move (in either direction)
VIOLENT_MOVE_THRESHOLD = float(os.getenv("SR_RL_VIOLENT_MOVE_THRESHOLD", "0.005"))  # 0.5%
VIOLENT_MOVE_BONUS     = float(os.getenv("SR_RL_VIOLENT_MOVE_BONUS",     "0.25"))
VIOLENT_MOVE_PENALTY   = float(os.getenv("SR_RL_VIOLENT_MOVE_PENALTY",   "-0.12"))

# Fizzle: price stuck near entry after N bars
FIZZLE_MIN_BARS   = int(os.getenv("SR_RL_FIZZLE_MIN_BARS",   "30"))
FIZZLE_FLAT_PCT   = float(os.getenv("SR_RL_FIZZLE_FLAT_PCT", "0.002"))   # ±0.2%
FIZZLE_PENALTY    = float(os.getenv("SR_RL_FIZZLE_PENALTY",  "-0.07"))

# Reversal: was up ≥1.5 %, gave it all back (applied once per trade)
REVERSAL_UP_PCT   = float(os.getenv("SR_RL_REVERSAL_UP_PCT",   "0.015"))
REVERSAL_BACK_PCT = float(os.getenv("SR_RL_REVERSAL_BACK_PCT", "0.003"))
REVERSAL_PENALTY  = float(os.getenv("SR_RL_REVERSAL_PENALTY",  "-0.50"))

# Multi-day hold bonus at TP (per full trading day, ~390 1m bars)
MULTIDAY_BARS     = int(os.getenv("SR_RL_MULTIDAY_BARS",  "390"))
MULTIDAY_BONUS    = float(os.getenv("SR_RL_MULTIDAY_BONUS", "0.20"))

# Early loss-cut bonus: fraction of avoided stop-loss distance rewarded
EARLY_CUT_BONUS_SCALE   = float(os.getenv("SR_RL_EARLY_CUT_BONUS_SCALE",   "0.40"))

# Premature exit penalty: applied when a winning trade is closed too early
MIN_HOLD_BARS           = int(os.getenv("SR_RL_MIN_HOLD_BARS",           "15"))
PREMATURE_THRESHOLD_R   = float(os.getenv("SR_RL_PREMATURE_THRESHOLD_R", "0.50"))
PREMATURE_EXIT_PENALTY  = float(os.getenv("SR_RL_PREMATURE_EXIT_PENALTY", "-0.30"))

# Tuner-compatible reward params (map onto existing or extend reward architecture)
REWARD_SCALE            = float(os.getenv("SR_RL_REWARD_SCALE",             "1.0"))
SUPPORT_ENTRY_BONUS     = float(os.getenv("SR_RL_SUPPORT_ENTRY_BONUS",      "0.08"))
RECLAIM_ENTRY_BONUS     = float(os.getenv("SR_RL_RECLAIM_ENTRY_BONUS",      "0.06"))
BREAKOUT_ENTRY_BONUS    = float(os.getenv("SR_RL_BREAKOUT_ENTRY_BONUS",     "0.05"))
BAD_ENTRY_PENALTY       = abs(float(os.getenv("SR_RL_BAD_ENTRY_PENALTY",    "0.06")))
HOLD_WINNER_MULT        = float(os.getenv("SR_RL_HOLD_WINNER_MULT",         "30.0"))
RESISTANCE_EXIT_BONUS   = float(os.getenv("SR_RL_RESISTANCE_EXIT_BONUS",    "1.0"))
MFE_CAPTURE_MULT        = float(os.getenv("SR_RL_MFE_CAPTURE_MULT",         "15.0"))
GIVEBACK_PENALTY_MULT   = float(os.getenv("SR_RL_GIVEBACK_PENALTY_MULT",    "10.0"))
MAE_PENALTY_MULT        = float(os.getenv("SR_RL_MAE_PENALTY_MULT",         "5.0"))
MIN_QUALITY_HOLD_BARS   = int(os.getenv("SR_RL_MIN_QUALITY_HOLD_BARS",      str(int(os.getenv("SR_RL_MIN_HOLD_BARS", "15")))))
FLAT_HOLD_PENALTY_PCT   = float(os.getenv("SR_RL_FLAT_HOLD_PENALTY",        "0.002"))

N_FEATURES = 12  # 8 price/SR features + rsi + macd_hist + vwap_dist + vol_ratio


# ──────────────────────────────────────────────────────────────────────────────
# Indicator helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, 1e-9)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def compute_macd_hist(close: pd.Series) -> pd.Series:
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return (macd_line - signal_line).fillna(0.0)


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Daily VWAP, reset each calendar day using the timestamp column."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    date_key = pd.to_datetime(df["timestamp"], utc=True).dt.date
    tpv = typical * df["volume"]
    cum_tpv = tpv.groupby(date_key).cumsum()
    cum_vol = df["volume"].groupby(date_key).cumsum()
    return (cum_tpv / cum_vol.replace(0.0, 1e-9)).fillna(df["close"])


def compute_vol_ratio(volume: pd.Series, period: int = VOL_RATIO_PERIOD) -> pd.Series:
    avg = volume.rolling(period, min_periods=1).mean()
    return (volume / avg.replace(0.0, 1e-9)).clip(upper=VOL_RATIO_CAP).fillna(1.0)


def attach_indicators(sdf: pd.DataFrame) -> pd.DataFrame:
    """Add rsi, macd_hist, vwap, vol_ratio columns to a single-symbol 1m DataFrame."""
    sdf = sdf.copy().sort_values("timestamp").reset_index(drop=True)
    sdf["rsi"] = compute_rsi(sdf["close"])
    sdf["macd_hist"] = compute_macd_hist(sdf["close"])
    sdf["vwap"] = compute_vwap(sdf)
    sdf["vol_ratio"] = compute_vol_ratio(sdf["volume"])
    return sdf


@dataclass
class SRDecision:
    symbol: str
    action: str
    price: float
    nearest_support: float | None
    nearest_resistance: float | None
    distance_to_support_pct: float | None
    distance_to_resistance_pct: float | None
    stop_loss: float | None
    take_profit: float | None
    qty: int
    reason: str


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_symbol(x: Any) -> str:
    return str(x or "").upper().strip().replace(".", "-")


def to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def get_feed() -> DataFeed:
    return DataFeed.SIP if ALPACA_DATA_FEED == "sip" else DataFeed.IEX


def data_client() -> None:
    """
    Disabled on purpose.

    This module must not create its own Alpaca data client. The single market-data
    source is stream_confirmed_movement_gate_one_stream.py, which writes:
        system-data/YYYYMMDD/YYYYMMDD_stream_confirmed_bars_1m.csv
    """
    return None


def trading_client() -> TradingClient:
    return TradingClient(
        os.getenv("APCA_API_KEY_ID"),
        os.getenv("APCA_API_SECRET_KEY"),
        paper=True,
    )


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def load_symbols_from_plan() -> list[str]:
    plan = load_json(PREMARKET_TRADE_PLAN_PATH, {})
    symbols = []

    for row in plan.get("daily_trade_plans", []):
        symbol = normalize_symbol(row.get("symbol"))
        if symbol:
            symbols.append(symbol)

    # If the stream-confirm trade plan exists, include those symbols too.
    stream_plan = load_json(OUT_DIR / f"{DATA_TAG}_stream_confirmed_trade_plan.json", [])
    if isinstance(stream_plan, list):
        for row in stream_plan:
            if isinstance(row, dict):
                symbol = normalize_symbol(row.get("symbol"))
                if symbol:
                    symbols.append(symbol)

    # Premarket fallback: if the trade plan hasn't been built yet, use the
    # locked trading universe written by the LLM universe selector.
    if not symbols:
        for candidate in (
            PROJECT_ROOT / f"{DATA_TAG}_locked_trading_universe.txt",
            PROJECT_ROOT / "locked_trading_universe.txt",
        ):
            if candidate.exists():
                try:
                    raw = [normalize_symbol(s) for s in candidate.read_text().splitlines() if s.strip()]
                    symbols = [s for s in raw if s]
                    if symbols:
                        print(f"[sr-rl] premarket symbol fallback: {len(symbols)} symbols from {candidate.name}")
                        break
                except Exception:
                    pass

    return list(dict.fromkeys(symbols))


def market_allows_trading() -> tuple[bool, float, str]:
    ctx = load_json(MARKET_CONTEXT_PATH, {})
    permission = ctx.get("trade_permission", "cautious")
    bias = ctx.get("overall_market_bias", "mixed")

    if permission == "halt_new_buys":
        return False, 0.0, "market_context_halt_new_buys"

    if permission == "cautious" or bias in {"bearish", "risk_off"}:
        return True, 0.5, "market_context_cautious"

    return True, 1.0, "market_context_normal"


def _fetch_historical_1m_alpaca(symbols: list[str], lookback_days: int) -> pd.DataFrame:
    """Pull historical 1-minute bars from Alpaca REST for premarket training."""
    columns = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    api_key = os.getenv("APCA_API_KEY_ID", "")
    secret = os.getenv("APCA_API_SECRET_KEY", "")
    if not api_key or not secret:
        print("[sr-rl] Alpaca credentials missing; cannot fetch historical bars")
        return pd.DataFrame(columns=columns)

    feed = DataFeed.SIP if ALPACA_DATA_FEED == "sip" else DataFeed.IEX
    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret)
    end = datetime.now(timezone.utc)
    # Add extra days to cover weekends / holidays in the lookback window.
    start = end - timedelta(days=lookback_days + 4)

    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=start,
                end=end,
                feed=feed,
            )
            resp = client.get_stock_bars(req)
            df = resp.df
            if df is None or df.empty:
                continue
            df = df.reset_index()
            # Alpaca returns a MultiIndex of (symbol, timestamp); after reset_index
            # the columns are named "symbol" and "timestamp".
            if "symbol" not in df.columns:
                df["symbol"] = symbol
            df = df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]
            frames.append(df)
        except Exception as exc:
            print(f"[sr-rl] historical fetch failed for {symbol}: {exc}")

    if not frames:
        return pd.DataFrame(columns=columns)

    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    return out.dropna(subset=["symbol", "timestamp", "close"]).sort_values(["symbol", "timestamp"])


def _load_shared_1m_bars(symbols: list[str], lookback_days: int) -> pd.DataFrame:
    """
    Read bars from the single stream-confirm cache.

    No Alpaca request is made here. Start stream_confirmed_movement_gate first so
    this file is continuously populated.  When _ALLOW_HISTORICAL_FALLBACK is True
    (set by train_agent) and the cache is absent or too sparse, falls back to
    Alpaca REST history so premarket training can proceed without waiting for
    market open.
    """
    columns = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    if not symbols:
        return pd.DataFrame(columns=columns)
    if not SHARED_BARS_1M_PATH.exists():
        if _ALLOW_HISTORICAL_FALLBACK:
            print(f"[sr-rl] shared cache missing; fetching {lookback_days}d history from Alpaca for {len(symbols)} symbols")
            return _fetch_historical_1m_alpaca(symbols, lookback_days)
        print(f"[sr-rl] shared bar cache missing: {SHARED_BARS_1M_PATH}")
        return pd.DataFrame(columns=columns)

    try:
        df = pd.read_csv(SHARED_BARS_1M_PATH)
    except Exception as exc:
        print(f"[sr-rl] failed to read shared bar cache: {exc}")
        if _ALLOW_HISTORICAL_FALLBACK:
            print(f"[sr-rl] falling back to Alpaca REST history for {len(symbols)} symbols")
            return _fetch_historical_1m_alpaca(symbols, lookback_days)
        return pd.DataFrame(columns=columns)

    if df.empty or not {"symbol", "timestamp", "close"}.issubset(df.columns):
        return pd.DataFrame(columns=columns)

    wanted = {normalize_symbol(s) for s in symbols if normalize_symbol(s)}
    df = df.copy()
    df["symbol"] = df["symbol"].map(normalize_symbol)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df[df["symbol"].isin(wanted)]
    df = df.dropna(subset=["symbol", "timestamp", "close"])

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    df = df[df["timestamp"] >= cutoff]

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = df["close"] if col != "volume" else 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    result = df.dropna(subset=["open", "high", "low", "close"]).sort_values(["symbol", "timestamp"])

    if _ALLOW_HISTORICAL_FALLBACK and len(result) < TRAIN_MIN_ROWS:
        print(
            f"[sr-rl] shared cache too sparse ({len(result)} rows < {TRAIN_MIN_ROWS}); "
            f"fetching {lookback_days}d history from Alpaca for {len(symbols)} symbols"
        )
        return _fetch_historical_1m_alpaca(symbols, lookback_days)

    return result


def _resample_shared_bars(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    columns = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    frames = []
    rule = f"{int(hours)}h"
    for symbol, sdf in df.groupby("symbol", sort=False):
        x = sdf.sort_values("timestamp").set_index("timestamp")
        out = pd.DataFrame(
            {
                "open": x["open"].resample(rule).first(),
                "high": x["high"].resample(rule).max(),
                "low": x["low"].resample(rule).min(),
                "close": x["close"].resample(rule).last(),
                "volume": x["volume"].resample(rule).sum(),
            }
        ).dropna().reset_index()
        out["symbol"] = symbol
        frames.append(out[columns])

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"])


def _fetch_bars(
    symbols: list[str],
    timeframe: "TimeFrame",
    lookback_days: int,
    log_prefix: str = "[sr-rl]",
) -> pd.DataFrame:
    """
    Compatibility wrapper. It no longer fetches Alpaca bars.

    Timeframes are built by resampling the shared 1-minute cache written by the
    stream-confirm module, so the whole system uses one market-data stream.
    """
    # Alpaca TimeFrame objects expose amount/unit inconsistently across versions,
    # so callers use fetch_bars(symbols, hours) for hourly bars and fetch_bars_1m
    # for minute bars. If this wrapper is called directly, default to 1m cache.
    return _load_shared_1m_bars(symbols, lookback_days)


def fetch_bars(symbols: list[str], hours: int) -> pd.DataFrame:
    bars_1m = _load_shared_1m_bars(symbols, LOOKBACK_DAYS)
    return _resample_shared_bars(bars_1m, hours)


def fetch_bars_1m(symbols: list[str]) -> pd.DataFrame:
    return _load_shared_1m_bars(symbols, TRAIN_LOOKBACK_DAYS)


def detect_levels(df: pd.DataFrame) -> tuple[list[float], list[float]]:
    if df.empty or len(df) < LEVEL_WINDOW * 2 + 1:
        return [], []

    highs = df["high"].astype(float).reset_index(drop=True)
    lows = df["low"].astype(float).reset_index(drop=True)

    supports = []
    resistances = []

    for i in range(LEVEL_WINDOW, len(df) - LEVEL_WINDOW):
        high_window = highs.iloc[i - LEVEL_WINDOW:i + LEVEL_WINDOW + 1]
        low_window = lows.iloc[i - LEVEL_WINDOW:i + LEVEL_WINDOW + 1]

        if highs.iloc[i] == high_window.max():
            resistances.append(float(highs.iloc[i]))

        if lows.iloc[i] == low_window.min():
            supports.append(float(lows.iloc[i]))

    return cluster_levels(supports), cluster_levels(resistances)


def cluster_levels(levels: list[float]) -> list[float]:
    clean = sorted([x for x in levels if x > 0])

    if not clean:
        return []

    clusters: list[list[float]] = []

    for level in clean:
        if not clusters:
            clusters.append([level])
            continue

        center = float(np.mean(clusters[-1]))

        if abs(level - center) / max(center, 1e-9) <= LEVEL_CLUSTER_PCT:
            clusters[-1].append(level)
        else:
            clusters.append([level])

    return [round(float(np.mean(c)), 4) for c in clusters]


def combined_levels(symbol: str, bars_1h: pd.DataFrame, bars_4h: pd.DataFrame) -> tuple[list[float], list[float]]:
    s1 = bars_1h[bars_1h["symbol"] == symbol]
    s4 = bars_4h[bars_4h["symbol"] == symbol]

    sup1, res1 = detect_levels(s1)
    sup4, res4 = detect_levels(s4)

    return cluster_levels(sup1 + sup4), cluster_levels(res1 + res4)


def nearest_levels(price: float, supports: list[float], resistances: list[float]) -> tuple[float | None, float | None]:
    below = [x for x in supports if x <= price]
    above = [x for x in resistances if x >= price]

    support = max(below) if below else None
    resistance = min(above) if above else None

    return support, resistance


def latest_price(symbol: str, bars_1h: pd.DataFrame) -> float:
    sdf = bars_1h[bars_1h["symbol"] == symbol].sort_values("timestamp")
    if sdf.empty:
        return 0.0
    return float(sdf.iloc[-1]["close"])


def build_features(
    price: float,
    support: float | None,
    resistance: float | None,
    market_size_mult: float,
    rsi: float = 50.0,
    macd_hist: float = 0.0,
    vwap_dist: float = 0.0,
    vol_ratio: float = 1.0,
) -> np.ndarray:
    if price <= 0:
        return np.zeros(N_FEATURES, dtype=np.float32)

    ds = (price - support) / price if support else 1.0
    dr = (resistance - price) / price if resistance else 1.0
    range_pct = (resistance - support) / price if support and resistance else 0.0

    near_support = float(support is not None and abs(price - support) / price <= ENTRY_DISTANCE_PCT)
    near_resistance = float(resistance is not None and abs(price - resistance) / price <= ENTRY_DISTANCE_PCT)

    # Normalize momentum features to roughly [0, 1] or [-1, 1] scale.
    rsi_norm = rsi / 100.0
    macd_norm = macd_hist / max(abs(price), 1.0)
    # vwap_dist is already fractional; vol_ratio is capped at VOL_RATIO_CAP.
    vol_norm = vol_ratio / VOL_RATIO_CAP

    return np.array(
        [
            price,
            support or 0.0,
            resistance or 0.0,
            ds,
            dr,
            range_pct,
            near_support,
            near_resistance,
            rsi_norm,
            macd_norm,
            vwap_dist,
            vol_norm,
        ],
        dtype=np.float32,
    ) * np.array(
        [0.001, 0.001, 0.001, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        dtype=np.float32,
    )


class SRTradingEnv(gym.Env):
    def __init__(self, rows: pd.DataFrame):
        super().__init__()
        self.rows = rows.reset_index(drop=True)
        self.i = 0
        self.position = 0
        self.entry = 0.0
        self.stop = 0.0
        self.tp = 0.0
        # Reward-shaping state (reset each trade)
        self.max_price: float = 0.0        # peak price while in position (for MFE)
        self.min_price: float = 0.0        # trough price while in position (for MAE)
        self.prev_close: float = 0.0       # last bar's close for bar-over-bar move
        self.bars_in_position: int = 0     # bars held since entry
        self.reversal_penalized: bool = False  # apply reversal penalty only once
        self.episode_bar: int = 0          # bars since last reset (for MAX_EPISODE_BARS)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(N_FEATURES,), dtype=np.float32)

    def _obs(self) -> np.ndarray:
        row = self.rows.iloc[self.i]
        price = to_float(row["close"])
        vwap = to_float(row.get("vwap", price))
        vwap_dist = (price - vwap) / price if price > 0 and vwap > 0 else 0.0
        return build_features(
            price=price,
            support=to_float(row["support"]) or None,
            resistance=to_float(row["resistance"]) or None,
            market_size_mult=1.0,
            rsi=to_float(row.get("rsi", 50.0)),
            macd_hist=to_float(row.get("macd_hist", 0.0)),
            vwap_dist=vwap_dist,
            vol_ratio=to_float(row.get("vol_ratio", 1.0)),
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.i = 0
        self.position = 0
        self.entry = 0.0
        self.stop = 0.0
        self.tp = 0.0
        self.max_price = 0.0
        self.min_price = 0.0
        self.prev_close = 0.0
        self.bars_in_position = 0
        self.reversal_penalized = False
        self.episode_bar = 0
        return self._obs(), {}

    def step(self, action: int):
        row = self.rows.iloc[self.i]
        price      = to_float(row["close"])
        support    = to_float(row["support"])
        resistance = to_float(row["resistance"])
        rsi        = to_float(row.get("rsi", 50.0))
        macd_hist  = to_float(row.get("macd_hist", 0.0))
        vwap       = to_float(row.get("vwap", price))
        vwap_dist  = (price - vwap) / max(price, 1e-9)
        vol_ratio  = to_float(row.get("vol_ratio", 1.0))

        reward = 0.0
        self.episode_bar += 1

        if self.position == 0:
            near_support    = support > 0 and abs(price - support) / price <= SUPPORT_TOUCH_BUFFER_PCT
            near_resistance = resistance > 0 and abs(price - resistance) / price <= RESISTANCE_TOUCH_BUFFER_PCT
            above_resistance = resistance > 0 and price > resistance * (1.0 + BREAKOUT_BUFFER_PCT)

            if action == ACTION_BUY and (near_support or above_resistance):
                self.position = 1
                self.entry = price
                self.stop = (support * (1.0 - STOP_BEYOND_LEVEL_PCT)) if support > 0 else price * (1.0 - STOP_BEYOND_LEVEL_PCT)
                risk = max(self.entry - self.stop, 1e-9)
                self.tp = self.entry + risk * TAKE_PROFIT_R_MULTIPLE
                self.max_price = price
                self.min_price = price
                self.prev_close = price
                self.bars_in_position = 0
                self.reversal_penalized = False

                reward += 0.05  # base entry reward

                # ── Entry quality ──────────────────────────────────────────
                if near_support:
                    reward += SUPPORT_ENTRY_BONUS              # near support — primary long trigger
                if above_resistance:
                    reward += BREAKOUT_ENTRY_BONUS             # breakout above resistance
                if vwap_dist > 0:
                    reward += RECLAIM_ENTRY_BONUS * 0.5        # VWAP reclaim / above VWAP
                if rsi < 40:
                    reward += ENTRY_QUALITY_BONUS              # oversold bounce
                if macd_hist > 0:
                    reward += ENTRY_QUALITY_BONUS * 0.5        # momentum turning positive
                if vol_ratio >= 1.5:
                    reward += ENTRY_QUALITY_BONUS * 0.4        # strong volume surge

                # Bad timing: overbought, below VWAP, or thin volume
                if rsi > 65:
                    reward -= BAD_ENTRY_PENALTY                # chasing overbought
                if vwap_dist < -0.005:
                    reward -= BAD_ENTRY_PENALTY * 0.5          # entering below VWAP
                if vol_ratio < 0.7:
                    reward -= BAD_ENTRY_PENALTY * 0.4          # weak / fading volume

            elif action == ACTION_SELL and near_resistance:
                reward += 0.03
            elif action == ACTION_WAIT:
                reward += 0.01
            else:
                reward -= 0.02

        else:
            self.bars_in_position += 1
            self.max_price = max(self.max_price, price)
            self.min_price = min(self.min_price, price)
            risk = max(self.entry - self.stop, 1e-9)

            # ── Per-bar unrealized drift ────────────────────────────────────
            unrealized_r = (price - self.entry) / risk
            reward += unrealized_r * UNREALIZED_R_SCALE

            # ── Per-bar MAE penalty (depth below entry) ─────────────────────
            mae_pct = max(0.0, (self.entry - self.min_price) / max(self.entry, 1e-9))
            reward -= mae_pct * MAE_PENALTY_MULT * UNREALIZED_R_SCALE

            # ── Violent single-bar move ─────────────────────────────────────
            bar_move = (price - self.prev_close) / max(self.prev_close, 1e-9)
            if bar_move > VIOLENT_MOVE_THRESHOLD:
                reward += VIOLENT_MOVE_BONUS
            elif bar_move < -VIOLENT_MOVE_THRESHOLD:
                reward += VIOLENT_MOVE_PENALTY

            # ── Fizzle penalty ──────────────────────────────────────────────
            flat_threshold = float(os.getenv("SR_RL_FIZZLE_FLAT_PCT", str(FLAT_HOLD_PENALTY_PCT)))
            if self.bars_in_position > FIZZLE_MIN_BARS:
                if abs(price - self.entry) / max(self.entry, 1e-9) < flat_threshold:
                    reward += FIZZLE_PENALTY

            # ── Reversal penalty (once per trade) ───────────────────────────
            if not self.reversal_penalized:
                if self.max_price > self.entry * (1.0 + REVERSAL_UP_PCT):
                    if price < self.entry * (1.0 + REVERSAL_BACK_PCT):
                        giveback_pct = (self.max_price - price) / max(self.max_price, 1e-9)
                        reward += REVERSAL_PENALTY - giveback_pct * GIVEBACK_PENALTY_MULT * 0.01
                        self.reversal_penalized = True

            self.prev_close = price

            # ── Exit conditions ─────────────────────────────────────────────
            if price <= self.stop:
                reward -= 1.0
                self.position = 0

            elif price >= self.tp:
                days_held = self.bars_in_position / max(MULTIDAY_BARS, 1)
                reward += TAKE_PROFIT_R_MULTIPLE + days_held * MULTIDAY_BONUS
                # Resistance exit bonus when TP is near resistance
                if resistance > 0 and abs(price - resistance) / price <= RESISTANCE_TOUCH_BUFFER_PCT * 2:
                    reward += RESISTANCE_EXIT_BONUS * 0.3
                self.position = 0

            elif action == ACTION_SELL:
                r = (price - self.entry) / risk
                reward += r

                # MFE capture bonus: reward proportional to how much of the MFE was kept
                mfe = max(0.0, self.max_price - self.entry)
                if mfe > 0 and r > 0:
                    capture_ratio = min((price - self.entry) / mfe, 1.0)
                    reward += capture_ratio * MFE_CAPTURE_MULT * 0.01

                # Resistance exit bonus
                if resistance > 0 and abs(price - resistance) / max(price, 1e-9) <= RESISTANCE_TOUCH_BUFFER_PCT:
                    reward += RESISTANCE_EXIT_BONUS * 0.5

                if r < 0:
                    saved_r = max(0.0, (self.stop - price) / risk)
                    reward += saved_r * EARLY_CUT_BONUS_SCALE
                else:
                    hold_shortfall = max(0.0, 1.0 - self.bars_in_position / max(MIN_QUALITY_HOLD_BARS, 1))
                    size_shortfall = max(0.0, 1.0 - r / max(PREMATURE_THRESHOLD_R, 1e-9)) if r < PREMATURE_THRESHOLD_R else 0.0
                    penalty_weight = max(hold_shortfall, size_shortfall)
                    if penalty_weight > 0:
                        reward += PREMATURE_EXIT_PENALTY * penalty_weight

                self.position = 0

        self.i += 1
        terminated = self.i >= len(self.rows) - 1
        # Episode length cap: force reset after MAX_EPISODE_BARS bars (close any open position)
        if not terminated and self.episode_bar >= MAX_EPISODE_BARS:
            terminated = True
            if self.position == 1:
                risk = max(self.entry - self.stop, 1e-9)
                r = (price - self.entry) / risk
                reward += r * 0.5  # partial credit for forced close
                self.position = 0
            self.episode_bar = 0
        truncated = False

        return self._obs(), float(reward) * REWARD_SCALE, terminated, truncated, {}


def build_training_rows(symbols: list[str]) -> pd.DataFrame:
    # S/R level detection uses longer-horizon hourly bars for structural context.
    bars_1h = fetch_bars(symbols, 1)
    bars_4h = fetch_bars(symbols, 4)
    # Training rows come from 1-minute bars over the last TRAIN_LOOKBACK_DAYS days.
    bars_1m = fetch_bars_1m(symbols)

    rows = []

    for symbol in symbols:
        if not bars_1m.empty and (bars_1m["symbol"] == symbol).any():
            sdf = bars_1m[bars_1m["symbol"] == symbol].sort_values("timestamp")
        else:
            print(f"[sr-rl] no 1m bars for {symbol}; falling back to 1H")
            sdf = bars_1h[bars_1h["symbol"] == symbol].sort_values("timestamp")

        supports, resistances = combined_levels(symbol, bars_1h, bars_4h)
        sdf = attach_indicators(sdf)

        for _, row in sdf.iterrows():
            price = float(row["close"])
            support, resistance = nearest_levels(price, supports, resistances)
            vwap = float(row.get("vwap", price))
            vwap_dist = (price - vwap) / price if price > 0 and vwap > 0 else 0.0
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": str(row["timestamp"]),
                    "close": price,
                    "support": support or 0.0,
                    "resistance": resistance or 0.0,
                    "rsi": float(row.get("rsi", 50.0)),
                    "macd_hist": float(row.get("macd_hist", 0.0)),
                    "vwap": vwap,
                    "vwap_dist": vwap_dist,
                    "vol_ratio": float(row.get("vol_ratio", 1.0)),
                }
            )

    return pd.DataFrame(rows).dropna()


def _simulate_trades(model: "PPO", rows: pd.DataFrame) -> list[dict]:
    """
    Run the trained model through the training rows in evaluation mode and
    return a list of completed trade records.  Used to compute the metrics
    JSON that the Optuna tuner reads to score each trial.
    """
    if rows.empty:
        return []

    rows_reset = rows.reset_index(drop=True)
    trades: list[dict] = []

    position = 0
    entry = stop = tp = 0.0
    max_price = min_price = prev_close = 0.0
    bars_in = 0
    entry_row: dict = {}

    # Re-create a temporary env just to generate observations.
    tmp_env = SRTradingEnv(rows_reset)
    tmp_env.reset()

    for idx in range(len(rows_reset) - 1):
        tmp_env.i = idx
        obs = tmp_env._obs()
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        row = rows_reset.iloc[idx]
        price     = to_float(row["close"])
        support   = to_float(row["support"])
        resistance = to_float(row["resistance"])

        near_support    = support > 0 and abs(price - support) / price <= SUPPORT_TOUCH_BUFFER_PCT
        above_resistance = resistance > 0 and price > resistance * (1.0 + BREAKOUT_BUFFER_PCT)

        if position == 0:
            if action == ACTION_BUY and (near_support or above_resistance):
                position = 1
                entry = price
                stop = (support * (1.0 - STOP_BEYOND_LEVEL_PCT)) if support > 0 else price * (1.0 - STOP_BEYOND_LEVEL_PCT)
                risk_amt = max(entry - stop, 1e-9)
                tp = entry + risk_amt * TAKE_PROFIT_R_MULTIPLE
                max_price = min_price = prev_close = price
                bars_in = 0
                entry_row = {
                    "entry_price": price,
                    "support": support,
                    "resistance": resistance,
                    "near_support": near_support,
                }
        else:
            bars_in += 1
            max_price = max(max_price, price)
            min_price = min(min_price, price)
            risk_amt = max(entry - stop, 1e-9)
            closed = False
            exit_reason = ""

            if price <= stop:
                exit_reason = "stop"
                closed = True
            elif price >= tp:
                exit_reason = "tp"
                closed = True
            elif action == ACTION_SELL:
                exit_reason = "signal_sell"
                closed = True

            if closed:
                pnl_pct = (price - entry) / entry
                mfe_pct = (max_price - entry) / entry
                mae_pct = (entry - min_price) / entry
                giveback_pct = max(0.0, mfe_pct - pnl_pct)
                capture_ratio = pnl_pct / mfe_pct if mfe_pct > 1e-6 else 0.0
                near_res_exit = resistance > 0 and abs(price - resistance) / max(price, 1e-9) <= RESISTANCE_TOUCH_BUFFER_PCT
                trades.append(
                    {
                        "entry_price": entry,
                        "exit_price": price,
                        "pnl_pct": pnl_pct,
                        "mfe_pct": mfe_pct,
                        "mae_pct": mae_pct,
                        "giveback_pct": giveback_pct,
                        "capture_ratio": max(0.0, min(capture_ratio, 2.0)),
                        "bars_held": bars_in,
                        "exit_reason": exit_reason,
                        "near_support_entry": entry_row.get("near_support", False),
                        "near_resistance_exit": near_res_exit,
                        "is_winner": pnl_pct > 0,
                    }
                )
                position = 0

            prev_close = price

    return trades


def _write_training_metrics(model: "PPO", rows: pd.DataFrame, timesteps: int) -> None:
    """Simulate trades with the trained model and write a metrics JSON for the tuner."""
    METRICS_PATH = OUT_DIR / f"{DATA_TAG}_sr_rl_training_metrics.json"
    trades = _simulate_trades(model, rows)
    n = len(trades)

    if n == 0:
        metrics: dict = {
            "ok": False,
            "rows": len(rows),
            "timesteps": timesteps,
            "trade_count": 0,
            "entry_count": 0,
            "exit_count": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_pnl_pct": 0.0,
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "avg_mfe_pct": 0.0,
            "avg_mae_pct": 0.0,
            "avg_giveback_pct": 0.0,
            "avg_capture_ratio": 0.0,
            "avg_hold_bars": 0.0,
            "support_entry_quality": 0.0,
            "resistance_exit_quality": 0.0,
            "chop_rate": 0.0,
        }
    else:
        pnls = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p <= 0]
        win_rate = len(wins) / n
        profit_factor = sum(wins) / max(sum(losses), 1e-9) if wins else 0.0
        avg_pnl = float(np.mean(pnls))
        total_return = float(np.sum(pnls))

        # Sharpe from per-trade PnL series (annualized loosely)
        pnl_arr = np.array(pnls)
        sharpe = float(pnl_arr.mean() / max(pnl_arr.std(), 1e-9)) if n > 1 else 0.0

        # Max drawdown on cumulative equity curve
        cum = np.cumsum(pnl_arr)
        running_max = np.maximum.accumulate(cum)
        drawdowns = running_max - cum
        max_drawdown = float(drawdowns.max()) if len(drawdowns) else 0.0

        avg_mfe = float(np.mean([t["mfe_pct"] for t in trades]))
        avg_mae = float(np.mean([t["mae_pct"] for t in trades]))
        avg_giveback = float(np.mean([t["giveback_pct"] for t in trades]))
        avg_capture = float(np.mean([t["capture_ratio"] for t in trades]))
        avg_hold = float(np.mean([t["bars_held"] for t in trades]))

        support_entries = sum(1 for t in trades if t["near_support_entry"])
        support_entry_quality = support_entries / n

        res_exits = sum(1 for t in trades if t["near_resistance_exit"])
        resistance_exit_quality = res_exits / n

        chop = sum(1 for t in trades if t["bars_held"] < MIN_QUALITY_HOLD_BARS and not t["is_winner"])
        chop_rate = chop / n

        metrics = {
            "ok": True,
            "rows": len(rows),
            "timesteps": timesteps,
            "trade_count": n,
            "entry_count": n,
            "exit_count": n,
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "avg_pnl_pct": round(avg_pnl, 6),
            "total_return": round(total_return, 6),
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_drawdown, 6),
            "avg_mfe_pct": round(avg_mfe, 6),
            "avg_mae_pct": round(avg_mae, 6),
            "avg_giveback_pct": round(avg_giveback, 6),
            "avg_capture_ratio": round(avg_capture, 4),
            "avg_hold_bars": round(avg_hold, 2),
            "support_entry_quality": round(support_entry_quality, 4),
            "resistance_exit_quality": round(resistance_exit_quality, 4),
            "chop_rate": round(chop_rate, 4),
        }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[sr-rl] metrics → {METRICS_PATH}")
    print(f"[sr-rl] trades={metrics['trade_count']} win_rate={metrics['win_rate']:.2%} "
          f"avg_pnl={metrics['avg_pnl_pct']:.4%} capture={metrics['avg_capture_ratio']:.3f} "
          f"hold={metrics['avg_hold_bars']:.1f}bars")


def train_agent() -> None:
    global _ALLOW_HISTORICAL_FALLBACK
    symbols = load_symbols_from_plan()

    if not symbols:
        raise RuntimeError("No symbols found in premarket_trade_plan.json")

    cache_exists = SHARED_BARS_1M_PATH.exists()
    _ALLOW_HISTORICAL_FALLBACK = not cache_exists
    if _ALLOW_HISTORICAL_FALLBACK:
        print(f"[sr-rl] stream cache absent — will fetch {TRAIN_LOOKBACK_DAYS}d history from Alpaca REST")

    print(f"[sr-rl] training on {len(symbols)} symbols | 1m lookback={TRAIN_LOOKBACK_DAYS}d | timesteps={TRAIN_TIMESTEPS} | historical_fallback={_ALLOW_HISTORICAL_FALLBACK}")
    rows = build_training_rows(symbols)
    print(f"[sr-rl] training rows: {len(rows)} ({len(rows) // max(len(symbols), 1)} avg per symbol)")

    if len(rows) < TRAIN_MIN_ROWS:
        raise RuntimeError(f"Not enough training rows: {len(rows)} < {TRAIN_MIN_ROWS}")

    env = SRTradingEnv(rows)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=PPO_LEARNING_RATE,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        gamma=PPO_GAMMA,
        ent_coef=PPO_ENT_COEF,
    )
    model.learn(total_timesteps=TRAIN_TIMESTEPS)
    _ALLOW_HISTORICAL_FALLBACK = False
    model.save(str(MODEL_PATH))
    print(f"[saved] {MODEL_PATH}")

    _write_training_metrics(model, rows, TRAIN_TIMESTEPS)


def position_size(symbol: str, price: float, stop: float, size_mult: float) -> int:
    acct = trading_client().get_account()
    equity = to_float(getattr(acct, "equity", 0.0))

    risk_budget = equity * RISK_PER_TRADE_PCT * size_mult
    max_notional = equity * MAX_POSITION_PCT * size_mult
    risk_per_share = abs(price - stop)

    if price <= 0 or risk_per_share <= 0:
        return 0

    qty_by_risk = int(risk_budget / risk_per_share)
    qty_by_notional = int(max_notional / price)

    return max(0, min(qty_by_risk, qty_by_notional))


def submit_order(symbol: str, side: str, qty: int) -> None:
    if qty <= 0:
        return

    client = trading_client()

    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
    )

    client.submit_order(order)


def _latest_indicators(symbol: str, bars_1m: pd.DataFrame) -> dict[str, float]:
    """Return the most recent RSI, MACD, VWAP, and vol_ratio for a symbol."""
    sdf = bars_1m[bars_1m["symbol"] == symbol]
    if sdf.empty:
        return {"rsi": 50.0, "macd_hist": 0.0, "vwap": 0.0, "vol_ratio": 1.0}
    sdf = attach_indicators(sdf)
    last = sdf.iloc[-1]
    return {
        "rsi": float(last.get("rsi", 50.0)),
        "macd_hist": float(last.get("macd_hist", 0.0)),
        "vwap": float(last.get("vwap", 0.0)),
        "vol_ratio": float(last.get("vol_ratio", 1.0)),
    }


def live_cycle(model: PPO, execute: bool = False) -> list[dict[str, Any]]:
    can_trade, size_mult, market_reason = market_allows_trading()

    symbols = load_symbols_from_plan()
    bars_1h = fetch_bars(symbols, 1)
    bars_4h = fetch_bars(symbols, 4)
    bars_1m = fetch_bars_1m(symbols)

    decisions = []

    for symbol in symbols:
        price = latest_price(symbol, bars_1h)
        supports, resistances = combined_levels(symbol, bars_1h, bars_4h)
        support, resistance = nearest_levels(price, supports, resistances)

        ind = _latest_indicators(symbol, bars_1m)
        vwap_dist = (price - ind["vwap"]) / price if price > 0 and ind["vwap"] > 0 else 0.0

        obs = build_features(
            price, support, resistance, size_mult,
            rsi=ind["rsi"],
            macd_hist=ind["macd_hist"],
            vwap_dist=vwap_dist,
            vol_ratio=ind["vol_ratio"],
        )
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        dist_support = abs(price - support) / price if support and price > 0 else None
        dist_resistance = abs(price - resistance) / price if resistance and price > 0 else None

        stop = None
        tp = None
        qty = 0
        action_name = "wait"
        reason = market_reason

        if not can_trade:
            action_name = "wait"

        elif action == ACTION_BUY and support and dist_support is not None and dist_support <= ENTRY_DISTANCE_PCT:
            stop = support * (1.0 - STOP_BEYOND_LEVEL_PCT)
            risk = price - stop
            tp = price + risk * TAKE_PROFIT_R_MULTIPLE
            qty = position_size(symbol, price, stop, size_mult)
            action_name = "buy"
            reason = "price_near_support_rl_buy"

            if execute and qty > 0:
                submit_order(symbol, "buy", qty)

        elif action == ACTION_SELL and resistance and dist_resistance is not None and dist_resistance <= ENTRY_DISTANCE_PCT:
            stop = resistance * (1.0 + STOP_BEYOND_LEVEL_PCT)
            risk = stop - price
            tp = price - risk * TAKE_PROFIT_R_MULTIPLE
            action_name = "sell"
            reason = "price_near_resistance_rl_sell"

            # This layer treats SELL as exit/avoid unless you add shorting.
            # submit_order is intentionally not called for short selling.

        decisions.append(
            {
                "timestamp_utc": now_utc(),
                "symbol": symbol,
                "action": action_name,
                "price": round(price, 4),
                "nearest_support": round(support, 4) if support else None,
                "nearest_resistance": round(resistance, 4) if resistance else None,
                "distance_to_support_pct": round(dist_support, 5) if dist_support is not None else None,
                "distance_to_resistance_pct": round(dist_resistance, 5) if dist_resistance is not None else None,
                "stop_loss": round(stop, 4) if stop else None,
                "take_profit": round(tp, 4) if tp else None,
                "qty": qty,
                "reason": reason,
            }
        )

    payload = {
        "created_at_utc": now_utc(),
        "scan_seconds": SCAN_SECONDS,
        "execute_orders": execute,
        "decisions": decisions,
    }

    SIGNALS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[saved] {SIGNALS_PATH}")

    return decisions


def run_live(execute: bool = False) -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Missing model: {MODEL_PATH}. Run: python sr_rl_agent.py --train")

    model = PPO.load(str(MODEL_PATH))

    while True:
        try:
            decisions = live_cycle(model, execute=execute)
            print(json.dumps(decisions[:5], indent=2))
        except Exception as exc:
            print(f"[sr-rl] cycle failed: {exc}")

        time.sleep(SCAN_SECONDS)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually submit BUY orders. Default only writes signals.")
    args = parser.parse_args()

    if args.train:
        train_agent()
    elif args.live:
        run_live(execute=args.execute)
    else:
        print("Use --train or --live")
        
"""
python sr_rl_agent.py --train
python sr_rl_agent.py --live

To actually submit BUY orders:
python sr_rl_agent.py --live --execute
"""