#!/usr/bin/env python3
"""
rl_spx_breakout_trainer.py

Long-only reinforcement-learning breakout planner for SPX-style index trading.

Default trainable proxy:
    SPX logical instrument -> SPY bars

This version trains the bot the same way as the intraday structure trainer:
    0 = HOLD
    1 = ENTER_LONG
    2 = EXIT_LONG
    3 = ADD_LONG

It learns:
    - daily support/resistance context
    - intraday breakout continuation entries
    - longer winner holding
    - dynamic MFE/giveback exits
    - pyramiding into confirmed winners
    - avoiding false breakouts/chop

It does NOT place trades. It writes a plan and training metrics that run.py can use
as a confirmation/timing layer.
"""

from __future__ import annotations

import json
import math
import os
import random
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import gymnasium as gym
import numpy as np
import pandas as pd
from config.env_loader import load_instance_env
from gymnasium import spaces

try:
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except Exception:
    DataFeed = None
    StockHistoricalDataClient = None
    StockBarsRequest = None
    TimeFrame = None

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception:
    PPO = None
    DummyVecEnv = None


ET = ZoneInfo("America/New_York")
TODAY_TAG = datetime.now(ET).strftime("%Y%m%d")
YESTERDAY_TAG = (datetime.now(ET) - timedelta(days=1)).strftime("%Y%m%d")
DATA_TAG = os.getenv("RL_DATA_DATE", os.getenv("SPX_RL_DATA_DATE", TODAY_TAG))


# =============================================================================
# Paths / config
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
load_instance_env()

OUT_DIR = PROJECT_ROOT / "system-data" / f"{DATA_TAG}_SPX"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOGICAL_SYMBOL = os.getenv("SPX_LOGICAL_SYMBOL", "SPX").upper().strip()
TRAIN_SYMBOL = os.getenv("SPX_TRAIN_SYMBOL", "SPY").upper().strip()

LOOKBACK_DAYS = int(os.getenv("SPX_RL_LOOKBACK_DAYS", "730"))
TRAIN_TIMESTEPS = int(os.getenv("SPX_RL_TIMESTEPS", os.getenv("RL_INTRADAY_TIMESTEPS", "300000")))

FEATURE_PATH = Path(os.getenv("SPX_RL_FEATURE_PATH", OUT_DIR / f"{DATA_TAG}_spx_rl_features.parquet"))
MODEL_PATH = Path(os.getenv("SPX_RL_MODEL_PATH", MODEL_DIR / "spx_breakout_ppo.zip"))
PLAN_PATH = Path(os.getenv("SPX_RL_PLAN_PATH", OUT_DIR / f"{DATA_TAG}_spx_intraday_breakout_plan.json"))
METRICS_PATH = OUT_DIR / f"{DATA_TAG}_spx_rl_training_metrics.json"
DIAGNOSTICS_PATH = OUT_DIR / f"{DATA_TAG}_spx_rl_signal_diagnostics.csv"

MARKET_CONTEXT_PATH = OUT_DIR / f"{DATA_TAG}_market_context.json"
PREMARKET_PLAN_PATH = OUT_DIR / f"{DATA_TAG}_premarket_trade_plan.json"

ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex").lower().strip()

MAX_EPISODE_BARS = int(os.getenv("SPX_RL_MAX_EPISODE_BARS", "390"))
LEVEL_WINDOW_DAYS = int(os.getenv("SPX_DAILY_LEVEL_WINDOW_DAYS", "20"))
PIVOT_LOOKBACK = int(os.getenv("SPX_PIVOT_LOOKBACK", "5"))
LEVEL_CLUSTER_ATR_MULT = float(os.getenv("SPX_LEVEL_CLUSTER_ATR_MULT", "0.35"))

ENTRY_NEAR_LEVEL_ATR = float(os.getenv("SPX_ENTRY_NEAR_LEVEL_ATR", "0.30"))
BREAKOUT_CONFIRM_ATR = float(os.getenv("SPX_BREAKOUT_CONFIRM_ATR", "0.12"))
INITIAL_STOP_ATR = float(os.getenv("SPX_INITIAL_STOP_ATR", "0.80"))
TRAIL_ATR = float(os.getenv("SPX_TRAIL_ATR", "1.20"))

COMMISSION_BPS = float(os.getenv("SPX_RL_COMMISSION_BPS", "0.5"))
SLIPPAGE_BPS = float(os.getenv("SPX_RL_SLIPPAGE_BPS", "1.0"))

# PPO hyperparameters.
PPO_LEARNING_RATE = float(os.getenv("SPX_RL_LEARNING_RATE", os.getenv("RL_INTRADAY_LEARNING_RATE", "0.00010")))
PPO_N_STEPS = int(os.getenv("SPX_RL_N_STEPS", os.getenv("RL_INTRADAY_N_STEPS", "512")))
PPO_BATCH_SIZE = int(os.getenv("SPX_RL_BATCH_SIZE", os.getenv("RL_INTRADAY_BATCH_SIZE", "64")))
PPO_GAMMA = float(os.getenv("SPX_RL_GAMMA", os.getenv("RL_INTRADAY_GAMMA", "0.990")))
PPO_ENT_COEF = float(os.getenv("SPX_RL_ENT_COEF", os.getenv("RL_INTRADAY_ENT_COEF", "0.055")))

# Reward shaping / curriculum.
REWARD_SCALE = float(os.getenv("SPX_RL_REWARD_SCALE", "100.0"))
FALSE_BREAKOUT_PENALTY = float(os.getenv("SPX_RL_FALSE_BREAKOUT_PENALTY", "0.35"))
RISK_OFF_PENALTY = float(os.getenv("SPX_RL_RISK_OFF_PENALTY", "0.20"))

FORCE_EXPLORATION_ACTIONS = os.getenv("SPX_RL_FORCE_EXPLORATION_ACTIONS", os.getenv("RL_FORCE_EXPLORATION_ACTIONS", "1")).lower() in {"1", "true", "yes"}
FORCE_EXPLORATION_EPISODE_FRAC = float(os.getenv("SPX_RL_FORCE_EXPLORATION_EPISODE_FRAC", os.getenv("RL_FORCE_EXPLORATION_EPISODE_FRAC", "0.25")))
FORCE_EXPLORATION_PROB = float(os.getenv("SPX_RL_FORCE_EXPLORATION_PROB", os.getenv("RL_FORCE_EXPLORATION_PROB", "0.10")))
FORCE_ADD_LONG_EXPLORATION_PROB = float(os.getenv("SPX_RL_FORCE_ADD_LONG_EXPLORATION_PROB", os.getenv("RL_FORCE_ADD_LONG_EXPLORATION_PROB", "0.16")))

MIN_TRADES_PER_EPISODE = int(os.getenv("SPX_RL_MIN_TRADES_PER_EPISODE", os.getenv("RL_MIN_TRADES_PER_EPISODE", "1")))
NO_TRADE_EPISODE_PENALTY = float(os.getenv("SPX_RL_NO_TRADE_EPISODE_PENALTY", os.getenv("RL_NO_TRADE_EPISODE_PENALTY", "10.0")))
FLAT_HOLD_STEP_PENALTY = float(os.getenv("SPX_RL_FLAT_HOLD_STEP_PENALTY", os.getenv("RL_FLAT_HOLD_STEP_PENALTY", "0.003")))
VALID_ENTRY_REWARD = float(os.getenv("SPX_RL_VALID_ENTRY_REWARD", os.getenv("RL_VALID_ENTRY_REWARD", "3.0")))
FIRST_TRADE_BONUS = float(os.getenv("SPX_RL_FIRST_TRADE_BONUS", os.getenv("RL_FIRST_TRADE_BONUS", "5.0")))

USE_DYNAMIC_GIVEBACK_EXIT = os.getenv("SPX_RL_USE_DYNAMIC_GIVEBACK_EXIT", os.getenv("RL_USE_DYNAMIC_GIVEBACK_EXIT", "1")).lower() in {"1", "true", "yes"}
DYNAMIC_GIVEBACK_MIN_MFE = float(os.getenv("SPX_RL_DYNAMIC_GIVEBACK_MIN_MFE", os.getenv("RL_DYNAMIC_GIVEBACK_MIN_MFE", "0.006")))
DYNAMIC_GIVEBACK_FRACTION = float(os.getenv("SPX_RL_DYNAMIC_GIVEBACK_FRACTION", os.getenv("RL_DYNAMIC_GIVEBACK_FRACTION", "0.60")))
DYNAMIC_GIVEBACK_FLOOR = float(os.getenv("SPX_RL_DYNAMIC_GIVEBACK_FLOOR", os.getenv("RL_DYNAMIC_GIVEBACK_FLOOR", "0.003")))
DYNAMIC_GIVEBACK_MIN_HOLD_BARS = int(os.getenv("SPX_RL_DYNAMIC_GIVEBACK_MIN_HOLD_BARS", os.getenv("RL_DYNAMIC_GIVEBACK_MIN_HOLD_BARS", "6")))

MFE_CAPTURE_REWARD_MULT = float(os.getenv("SPX_RL_MFE_CAPTURE_REWARD_MULT", os.getenv("RL_MFE_CAPTURE_REWARD_MULT", "18.0")))
MFE_CAPTURE_REWARD_CAP = float(os.getenv("SPX_RL_MFE_CAPTURE_REWARD_CAP", os.getenv("RL_MFE_CAPTURE_REWARD_CAP", "8.0")))
MAE_PENALTY_MULT = float(os.getenv("SPX_RL_MAE_PENALTY_MULT", os.getenv("RL_MAE_PENALTY_MULT", "6.0")))
MAE_PENALTY_CAP = float(os.getenv("SPX_RL_MAE_PENALTY_CAP", os.getenv("RL_MAE_PENALTY_CAP", "4.0")))
GIVEBACK_CAPTURE_PENALTY_MULT = float(os.getenv("SPX_RL_GIVEBACK_CAPTURE_PENALTY_MULT", os.getenv("RL_GIVEBACK_CAPTURE_PENALTY_MULT", "10.0")))
GIVEBACK_CAPTURE_PENALTY_CAP = float(os.getenv("SPX_RL_GIVEBACK_CAPTURE_PENALTY_CAP", os.getenv("RL_GIVEBACK_CAPTURE_PENALTY_CAP", "4.0")))

LONG_HOLD_TARGET_BARS = float(os.getenv("SPX_RL_LONG_HOLD_TARGET_BARS", os.getenv("RL_LONG_HOLD_TARGET_BARS", "8")))
AVG_HOLD_REWARD_MULT = float(os.getenv("SPX_RL_AVG_HOLD_REWARD_MULT", os.getenv("RL_AVG_HOLD_REWARD_MULT", "0.75")))
MIN_QUALITY_HOLD_BARS = int(os.getenv("SPX_RL_MIN_QUALITY_HOLD_BARS", os.getenv("RL_MIN_QUALITY_HOLD_BARS", "5")))
EARLY_WINNER_EXIT_PENALTY = float(os.getenv("SPX_RL_EARLY_WINNER_EXIT_PENALTY", os.getenv("RL_EARLY_EXIT_PROFIT_PENALTY", "2.0")))
QUALITY_WINNER_HOLD_BONUS = float(os.getenv("SPX_RL_QUALITY_WINNER_HOLD_BONUS", os.getenv("RL_QUALITY_HOLD_PROFIT_BONUS", "3.0")))

UNREALIZED_WINNER_HOLD_MULT = float(os.getenv("SPX_RL_UNREALIZED_WINNER_HOLD_MULT", os.getenv("RL_HOLD_WINNER_PNL_MULT", "35.0")))
TREND_HOLD_TIME_MULT = float(os.getenv("SPX_RL_TREND_HOLD_TIME_MULT", os.getenv("RL_HOLD_TREND_STRENGTH_MULT", "0.20")))
CONTINUATION_HOLD_MULT = float(os.getenv("SPX_RL_CONTINUATION_HOLD_MULT", os.getenv("RL_HOLD_FUTURE_CONTINUATION_MULT", "160.0")))
CONTINUATION_HOLD_CAP = float(os.getenv("SPX_RL_CONTINUATION_HOLD_CAP", os.getenv("RL_HOLD_CONTINUATION_CAP", "5.0")))

ADD_LONG_ENABLED = os.getenv("SPX_RL_ADD_LONG_ENABLED", os.getenv("RL_ADD_LONG_ENABLED", "1")).lower() in {"1", "true", "yes"}
ADD_LONG_REWARD = float(os.getenv("SPX_RL_ADD_LONG_REWARD", os.getenv("RL_ADD_LONG_REWARD", "8.0")))
ADD_LONG_FRACTION = float(os.getenv("SPX_RL_ADD_LONG_FRACTION", os.getenv("RL_ADD_LONG_FRACTION", "0.35")))
ADD_LONG_MIN_UNREALIZED_PCT = float(os.getenv("SPX_RL_ADD_LONG_MIN_UNREALIZED_PCT", os.getenv("RL_ADD_LONG_MIN_UNREALIZED_PCT", "0.0015")))
ADD_LONG_MIN_HOLD_BARS = int(os.getenv("SPX_RL_ADD_LONG_MIN_HOLD_BARS", os.getenv("RL_ADD_LONG_MIN_HOLD_BARS", "2")))
ADD_LONG_MAX_COUNT = int(os.getenv("SPX_RL_ADD_LONG_MAX_COUNT", os.getenv("RL_ADD_LONG_MAX_COUNT", "2")))
ADD_LONG_FUTURE_REWARD_MULT = float(os.getenv("SPX_RL_ADD_LONG_FUTURE_REWARD_MULT", os.getenv("RL_ADD_LONG_FUTURE_REWARD_MULT", "400.0")))
ADD_LONG_FUTURE_REWARD_CAP = float(os.getenv("SPX_RL_ADD_LONG_FUTURE_REWARD_CAP", os.getenv("RL_ADD_LONG_FUTURE_REWARD_CAP", "4.0")))
BAD_ADD_LONG_PENALTY = float(os.getenv("SPX_RL_BAD_ADD_LONG_PENALTY", os.getenv("RL_BAD_ADD_LONG_PENALTY", "1.5")))
MAX_TOTAL_POSITION_FRACTION = float(os.getenv("SPX_RL_MAX_TOTAL_POSITION_FRACTION", os.getenv("RL_MAX_TOTAL_POSITION_FRACTION", "0.25")))

np.seterr(all="ignore")
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class RLTradePlan:
    logical_symbol: str
    train_symbol: str
    created_at_utc: str
    market_bias: str
    trade_permission: str
    current_price: float
    daily_support: float
    daily_resistance: float
    nearest_support: float
    nearest_resistance: float
    atr_daily: float
    breakout_long_trigger: float
    long_stop: float
    long_initial_target: float
    trailing_atr_multiple: float
    model_action: str
    model_confidence_proxy: float
    feature_snapshot: dict[str, Any]
    notes: list[str]


@dataclass
class TradeEvent:
    timestamp: str
    action: str
    price: float
    position_fraction: float
    realized_pnl: float
    reward: float
    hold_bars: int
    unrealized_pct: float
    mfe_pct: float
    mae_pct: float
    giveback_from_mfe_pct: float
    capture_ratio: float
    reason: str


# =============================================================================
# Utilities
# =============================================================================

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def safe_json(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    return value


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(safe_json(payload), indent=2) + "\n", encoding="utf-8")


# =============================================================================
# Alpaca data
# =============================================================================

def get_alpaca_client() -> StockHistoricalDataClient:
    if StockHistoricalDataClient is None:
        raise RuntimeError("alpaca-py is not installed. Run: pip install alpaca-py")

    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")

    if not key or not secret:
        raise RuntimeError("Missing APCA_API_KEY_ID/APCA_API_SECRET_KEY in .env")

    return StockHistoricalDataClient(key, secret)


def get_feed() -> Any:
    if DataFeed is None:
        return None
    return DataFeed.SIP if ALPACA_DATA_FEED == "sip" else DataFeed.IEX


def fetch_bars(symbol: str, timeframe: Any, start: datetime, end: datetime) -> pd.DataFrame:
    client = get_alpaca_client()

    kwargs = {
        "symbol_or_symbols": symbol,
        "timeframe": timeframe,
        "start": start,
        "end": end,
    }

    feed = get_feed()
    if feed is not None:
        kwargs["feed"] = feed

    bars = client.get_stock_bars(StockBarsRequest(**kwargs)).df

    if bars is None or bars.empty:
        return pd.DataFrame()

    df = bars.reset_index()
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol].copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_training_data() -> dict[str, pd.DataFrame]:
    if TimeFrame is None:
        raise RuntimeError("alpaca-py TimeFrame is unavailable.")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)

    minute = fetch_bars(TRAIN_SYMBOL, TimeFrame.Minute, start, end)
    daily = fetch_bars(TRAIN_SYMBOL, TimeFrame.Day, start, end)

    if minute.empty or daily.empty:
        raise RuntimeError(f"No data returned for {TRAIN_SYMBOL}. Check symbol, feed, and Alpaca credentials.")

    frames = {
        "1m": minute,
        "daily": daily,
    }

    frames["5m"] = resample_ohlcv(minute, "5min")
    frames["15m"] = resample_ohlcv(minute, "15min")
    frames["30m"] = resample_ohlcv(minute, "30min")
    frames["1h"] = resample_ohlcv(minute, "1h")

    return frames


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    x = df.copy()
    x = x.set_index("timestamp").sort_index()

    out = pd.DataFrame(
        {
            "open": x["open"].resample(rule).first(),
            "high": x["high"].resample(rule).max(),
            "low": x["low"].resample(rule).min(),
            "close": x["close"].resample(rule).last(),
            "volume": x["volume"].resample(rule).sum(),
        }
    ).dropna()

    out = out.reset_index()
    out["symbol"] = TRAIN_SYMBOL
    return out


# =============================================================================
# Indicators / features
# =============================================================================

def add_indicators(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    x = df.copy().sort_values("timestamp").reset_index(drop=True)

    close = x["close"].astype(float)
    high = x["high"].astype(float)
    low = x["low"].astype(float)
    volume = x["volume"].astype(float)

    x[f"{prefix}ret_1"] = close.pct_change()
    x[f"{prefix}ret_3"] = close.pct_change(3)
    x[f"{prefix}ret_5"] = close.pct_change(5)
    x[f"{prefix}ret_10"] = close.pct_change(10)
    x[f"{prefix}ret_20"] = close.pct_change(20)

    x[f"{prefix}sma5"] = close.rolling(5).mean()
    x[f"{prefix}sma20"] = close.rolling(20).mean()
    x[f"{prefix}sma50"] = close.rolling(50).mean()
    x[f"{prefix}dist_sma5"] = close / x[f"{prefix}sma5"] - 1.0
    x[f"{prefix}dist_sma20"] = close / x[f"{prefix}sma20"] - 1.0
    x[f"{prefix}dist_sma50"] = close / x[f"{prefix}sma50"] - 1.0

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x[f"{prefix}rsi"] = 100.0 - (100.0 / (1.0 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    x[f"{prefix}macd"] = macd
    x[f"{prefix}macd_signal"] = signal
    x[f"{prefix}macd_hist"] = macd - signal

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    x[f"{prefix}atr"] = tr.rolling(14).mean()
    x[f"{prefix}atr_pct"] = x[f"{prefix}atr"] / close

    typical = (high + low + close) / 3.0
    sma_typical = typical.rolling(20).mean()
    mean_dev = (typical - sma_typical).abs().rolling(20).mean()
    x[f"{prefix}cci"] = (typical - sma_typical) / (0.015 * mean_dev.replace(0, np.nan))

    x[f"{prefix}vol_sma20"] = volume.rolling(20).mean()
    x[f"{prefix}rel_volume"] = volume / x[f"{prefix}vol_sma20"].replace(0, np.nan)

    x[f"{prefix}rolling_high_20"] = high.rolling(20).max()
    x[f"{prefix}rolling_low_20"] = low.rolling(20).min()
    x[f"{prefix}dist_high_20"] = close / x[f"{prefix}rolling_high_20"] - 1.0
    x[f"{prefix}dist_low_20"] = close / x[f"{prefix}rolling_low_20"] - 1.0

    return x.replace([np.inf, -np.inf], np.nan)


def find_pivot_levels(daily: pd.DataFrame) -> pd.DataFrame:
    x = add_indicators(daily, "d_").copy()
    x["date"] = pd.to_datetime(x["timestamp"], utc=True).dt.date

    highs = x["high"].astype(float)
    lows = x["low"].astype(float)

    x["pivot_high"] = highs[
        (highs == highs.rolling(PIVOT_LOOKBACK * 2 + 1, center=True).max())
    ]
    x["pivot_low"] = lows[
        (lows == lows.rolling(PIVOT_LOOKBACK * 2 + 1, center=True).min())
    ]

    supports = []
    resistances = []

    for i in range(len(x)):
        hist = x.iloc[max(0, i - LEVEL_WINDOW_DAYS):i]
        atr = to_float(x.iloc[i].get("d_atr"), 0.0)
        close = to_float(x.iloc[i].get("close"), 0.0)
        cluster_distance = max(atr * LEVEL_CLUSTER_ATR_MULT, close * 0.001)

        raw_supports = [to_float(v) for v in hist["pivot_low"].dropna().tolist()]
        raw_resistances = [to_float(v) for v in hist["pivot_high"].dropna().tolist()]

        supports.append(cluster_nearest(raw_supports, close, cluster_distance, below=True))
        resistances.append(cluster_nearest(raw_resistances, close, cluster_distance, below=False))

    x["daily_support"] = supports
    x["daily_resistance"] = resistances

    x["rolling_support"] = lows.shift(1).rolling(LEVEL_WINDOW_DAYS).min()
    x["rolling_resistance"] = highs.shift(1).rolling(LEVEL_WINDOW_DAYS).max()

    x["daily_support"] = x["daily_support"].fillna(x["rolling_support"])
    x["daily_resistance"] = x["daily_resistance"].fillna(x["rolling_resistance"])

    return x


def cluster_nearest(levels: list[float], price: float, cluster_distance: float, below: bool) -> float | None:
    if not levels or price <= 0:
        return None

    levels = sorted(levels)

    clusters: list[list[float]] = []
    for level in levels:
        if not clusters or abs(np.mean(clusters[-1]) - level) > cluster_distance:
            clusters.append([level])
        else:
            clusters[-1].append(level)

    clustered = [float(np.mean(c)) for c in clusters]

    if below:
        candidates = [x for x in clustered if x <= price]
        return max(candidates) if candidates else None

    candidates = [x for x in clustered if x >= price]
    return min(candidates) if candidates else None


def prepare_features(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    daily_levels = find_pivot_levels(frames["daily"])

    base = add_indicators(frames["1m"], "m1_")
    base["date"] = pd.to_datetime(base["timestamp"], utc=True).dt.date

    merged = pd.merge_asof(
        base.sort_values("timestamp"),
        daily_levels[
            [
                "timestamp",
                "daily_support",
                "daily_resistance",
                "d_atr",
                "d_atr_pct",
                "d_rsi",
                "d_macd_hist",
                "d_cci",
                "d_dist_sma5",
                "d_dist_sma20",
                "d_dist_sma50",
                "d_rel_volume",
            ]
        ].sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    for tf in ["5m", "15m", "30m", "1h"]:
        tfdf = add_indicators(frames[tf], f"{tf}_")
        keep = [
            "timestamp",
            f"{tf}_ret_1",
            f"{tf}_ret_3",
            f"{tf}_ret_5",
            f"{tf}_ret_10",
            f"{tf}_ret_20",
            f"{tf}_dist_sma5",
            f"{tf}_dist_sma20",
            f"{tf}_dist_sma50",
            f"{tf}_rsi",
            f"{tf}_macd_hist",
            f"{tf}_atr_pct",
            f"{tf}_cci",
            f"{tf}_rel_volume",
            f"{tf}_dist_high_20",
            f"{tf}_dist_low_20",
        ]
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            tfdf[keep].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

    close = merged["close"].astype(float)
    support = merged["daily_support"].astype(float)
    resistance = merged["daily_resistance"].astype(float)
    atr = merged["d_atr"].astype(float).replace(0, np.nan)

    merged["dist_to_support_atr"] = (close - support) / atr
    merged["dist_to_resistance_atr"] = (resistance - close) / atr
    merged["above_daily_resistance"] = (close > resistance).astype(float)
    merged["breakout_strength"] = (close - resistance) / atr

    merged["trend_alignment_score"] = (
        (merged["m1_macd_hist"] > 0).astype(float)
        + (merged["5m_macd_hist"] > 0).astype(float)
        + (merged["15m_macd_hist"] > 0).astype(float)
        + (merged["m1_rsi"] >= 50).astype(float)
        + (merged["5m_rsi"] >= 50).astype(float)
        + (merged["d_dist_sma20"] > 0).astype(float)
    ) / 6.0

    merged["breakout_continuation_score"] = (
        (merged["breakout_strength"] > 0).astype(float)
        + (merged["m1_ret_3"] > 0).astype(float)
        + (merged["5m_ret_1"] > 0).astype(float)
        + (merged["m1_rel_volume"] >= 1.0).astype(float)
        + (merged["trend_alignment_score"] >= 0.60).astype(float)
    ) / 5.0

    # Future labels for reward shaping only. They are intentionally excluded
    # from observations.
    for horizon in [1, 3, 5, 10, 15, 20, 30, 45, 60]:
        merged[f"future_ret_{horizon}m"] = close.shift(-horizon) / close - 1.0

    def future_window_max(s: pd.Series, window: int) -> pd.Series:
        return s.shift(-1).iloc[::-1].rolling(window, min_periods=1).max().iloc[::-1]

    def future_window_min(s: pd.Series, window: int) -> pd.Series:
        return s.shift(-1).iloc[::-1].rolling(window, min_periods=1).min().iloc[::-1]

    for horizon in [10, 20, 30, 45, 60]:
        future_high = future_window_max(merged["high"].astype(float), horizon)
        future_low = future_window_min(merged["low"].astype(float), horizon)
        merged[f"future_max_ret_{horizon}m"] = future_high / close - 1.0
        merged[f"future_min_ret_{horizon}m"] = future_low / close - 1.0

    merged = merged.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(FEATURE_PATH, index=False)
    print(f"[saved] {FEATURE_PATH} rows={len(merged)}")

    return merged


# =============================================================================
# RL Environment
# =============================================================================

class SPXBreakoutEnv(gym.Env):
    """
    Long-only actions:
        0 = HOLD / wait
        1 = ENTER_LONG
        2 = EXIT_LONG
        3 = ADD_LONG

    The reward favors:
        - entering confirmed daily-resistance breakouts
        - holding winners through multi-bar continuation
        - adding only into profitable continuation
        - exiting near local highs before giveback
        - avoiding false breakouts/chop
    """

    metadata = {"render_modes": []}

    def __init__(self, features: pd.DataFrame):
        super().__init__()

        self.df = features.reset_index(drop=True).copy()
        self.feature_cols = self._feature_columns()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(len(self.feature_cols) + 8,),
            dtype=np.float32,
        )

        self.idx = 0
        self.start_idx = 0
        self.end_idx = 0

        self.position = 0
        self.entry_price = 0.0
        self.position_fraction = 0.0
        self.high_water = 0.0
        self.low_water = 0.0
        self.bars_held = 0
        self.mfe = 0.0
        self.mae = 0.0
        self.add_long_count = 0
        self.long_entry_count = 0
        self.exit_long_count = 0
        self.realized_pnl = 0.0
        self.trade_events: list[TradeEvent] = []

    def _feature_columns(self) -> list[str]:
        preferred = [
            "m1_ret_1", "m1_ret_3", "m1_ret_5", "m1_ret_10", "m1_ret_20",
            "m1_dist_sma5", "m1_dist_sma20", "m1_dist_sma50",
            "m1_rsi", "m1_macd_hist", "m1_atr_pct", "m1_cci", "m1_rel_volume",
            "5m_ret_1", "5m_ret_3", "5m_ret_5", "5m_ret_10", "5m_ret_20",
            "5m_dist_sma5", "5m_dist_sma20", "5m_dist_sma50",
            "5m_rsi", "5m_macd_hist", "5m_atr_pct", "5m_cci", "5m_rel_volume",
            "15m_ret_1", "15m_ret_3", "15m_ret_5", "15m_ret_10", "15m_ret_20",
            "15m_dist_sma5", "15m_dist_sma20", "15m_dist_sma50",
            "15m_rsi", "15m_macd_hist", "15m_atr_pct", "15m_cci", "15m_rel_volume",
            "30m_ret_1", "30m_ret_3", "30m_ret_5", "30m_ret_10", "30m_ret_20",
            "30m_dist_sma5", "30m_dist_sma20", "30m_dist_sma50",
            "30m_rsi", "30m_macd_hist", "30m_atr_pct", "30m_cci", "30m_rel_volume",
            "1h_ret_1", "1h_ret_3", "1h_ret_5", "1h_ret_10", "1h_ret_20",
            "1h_dist_sma5", "1h_dist_sma20", "1h_dist_sma50",
            "1h_rsi", "1h_macd_hist", "1h_atr_pct", "1h_cci", "1h_rel_volume",
            "d_rsi", "d_macd_hist", "d_cci", "d_dist_sma5", "d_dist_sma20", "d_dist_sma50", "d_rel_volume",
            "dist_to_support_atr", "dist_to_resistance_atr",
            "above_daily_resistance", "breakout_strength",
            "trend_alignment_score", "breakout_continuation_score",
        ]
        return [c for c in preferred if c in self.df.columns]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        max_start = max(1, len(self.df) - MAX_EPISODE_BARS - 2)
        self.start_idx = int(self.np_random.integers(0, max_start))
        self.idx = self.start_idx
        self.end_idx = min(len(self.df) - 2, self.start_idx + MAX_EPISODE_BARS)

        self.position = 0
        self.entry_price = 0.0
        self.position_fraction = 0.0
        self.high_water = 0.0
        self.low_water = 0.0
        self.bars_held = 0
        self.mfe = 0.0
        self.mae = 0.0
        self.add_long_count = 0
        self.long_entry_count = 0
        self.exit_long_count = 0
        self.realized_pnl = 0.0
        self.trade_events = []

        return self._obs(), {}

    def step(self, action: int):
        row = self.df.iloc[self.idx]
        next_row = self.df.iloc[self.idx + 1]

        price = max(to_float(row["close"]), 1e-9)
        next_price = max(to_float(next_row["close"]), 1e-9)

        reward = 0.0
        transaction_cost = (COMMISSION_BPS + SLIPPAGE_BPS) / 10000.0

        episode_limit = max(1, self.end_idx - self.start_idx)

        # Curriculum exploration:
        # - flat: force enter-long sometimes
        # - already long: force ADD_LONG sometimes
        if (
            FORCE_EXPLORATION_ACTIONS
            and self.idx - self.start_idx < int(episode_limit * FORCE_EXPLORATION_EPISODE_FRAC)
        ):
            if self.position == 0 and random.random() < FORCE_EXPLORATION_PROB:
                action = 1
            elif self.position == 1 and random.random() < FORCE_ADD_LONG_EXPLORATION_PROB:
                action = 3

        if self.position == 1:
            self.bars_held += 1
            unrealized = price / self.entry_price - 1.0
            self.mfe = max(self.mfe, unrealized)
            self.mae = min(self.mae, unrealized)

            bar_ret = next_price / price - 1.0
            reward += bar_ret * REWARD_SCALE * max(self.position_fraction, 0.01)

            if unrealized > 0:
                reward += min(unrealized * UNREALIZED_WINNER_HOLD_MULT, 3.0)
                reward += min(self.bars_held, LONG_HOLD_TARGET_BARS) * TREND_HOLD_TIME_MULT * self._trend_strength(row)

            continuation = max(
                to_float(row.get("future_ret_10m")),
                to_float(row.get("future_ret_20m")),
                to_float(row.get("future_ret_30m")),
                0.0,
            )
            if continuation > 0 and self._trend_strength(row) >= 0.50:
                reward += min(continuation * CONTINUATION_HOLD_MULT, CONTINUATION_HOLD_CAP)

            if USE_DYNAMIC_GIVEBACK_EXIT:
                giveback = max(0.0, self.mfe - unrealized)
                allowed_giveback = max(DYNAMIC_GIVEBACK_FLOOR, self.mfe * DYNAMIC_GIVEBACK_FRACTION)
                if (
                    self.bars_held >= DYNAMIC_GIVEBACK_MIN_HOLD_BARS
                    and self.mfe >= DYNAMIC_GIVEBACK_MIN_MFE
                    and giveback >= allowed_giveback
                ):
                    action = 2

        if action == 1:
            reward += self._enter_long(row, price, transaction_cost)

        elif action == 2:
            reward += self._exit_long(row, price, transaction_cost, reason="agent_exit_long")

        elif action == 3:
            reward += self._add_long(row, price, transaction_cost)

        else:
            reward += self._hold_reward(row)

        self.idx += 1
        terminated = self.idx >= self.end_idx
        truncated = False

        if terminated and self.position == 1:
            final_price = max(to_float(self.df.iloc[self.idx]["close"]), 1e-9)
            reward += self._exit_long(self.df.iloc[self.idx], final_price, transaction_cost, reason="episode_end")

        if terminated and self.long_entry_count < MIN_TRADES_PER_EPISODE:
            reward -= NO_TRADE_EPISODE_PENALTY * (MIN_TRADES_PER_EPISODE - self.long_entry_count)

        return self._obs(), float(reward), terminated, truncated, {}

    def _trend_strength(self, row: pd.Series) -> float:
        return float(
            np.clip(
                (
                    to_float(row.get("trend_alignment_score"))
                    + to_float(row.get("breakout_continuation_score"))
                    + (1.0 if to_float(row.get("m1_macd_hist")) > 0 else 0.0)
                    + (1.0 if to_float(row.get("5m_macd_hist")) > 0 else 0.0)
                )
                / 4.0,
                0.0,
                1.0,
            )
        )

    def _entry_quality(self, row: pd.Series) -> float:
        breakout_ok = to_float(row.get("breakout_strength")) >= BREAKOUT_CONFIRM_ATR
        near_resistance = abs(to_float(row.get("dist_to_resistance_atr"))) <= ENTRY_NEAR_LEVEL_ATR
        macd_ok = to_float(row.get("m1_macd_hist")) > 0 and to_float(row.get("5m_macd_hist")) > 0
        rsi_ok = to_float(row.get("m1_rsi")) >= 50
        volume_ok = to_float(row.get("m1_rel_volume"), 1.0) >= 1.0
        return float(np.mean([breakout_ok, near_resistance, macd_ok, rsi_ok, volume_ok]))

    def _enter_long(self, row: pd.Series, price: float, transaction_cost: float) -> float:
        reward = 0.0

        if self.position == 1:
            return self._add_long(row, price, transaction_cost)

        reward -= transaction_cost
        quality = self._entry_quality(row)
        reward += VALID_ENTRY_REWARD * quality

        if self.long_entry_count == 0:
            reward += FIRST_TRADE_BONUS

        if quality < 0.40:
            reward -= FALSE_BREAKOUT_PENALTY

        if to_float(row.get("d_macd_hist")) < 0 or to_float(row.get("d_dist_sma20")) < 0:
            reward -= RISK_OFF_PENALTY

        future = max(to_float(row.get("future_ret_5m")), to_float(row.get("future_ret_10m")), 0.0)
        reward += min(future * 300.0, 3.0)

        self.position = 1
        self.entry_price = price
        self.position_fraction = 0.10
        self.high_water = price
        self.low_water = price
        self.bars_held = 0
        self.mfe = 0.0
        self.mae = 0.0
        self.long_entry_count += 1

        self._record_event(row, "ENTER_LONG", price, reward, "agent_enter_long")
        return reward

    def _add_long(self, row: pd.Series, price: float, transaction_cost: float) -> float:
        if not ADD_LONG_ENABLED:
            return -0.25

        if self.position != 1 or self.entry_price <= 0:
            return -0.50

        unrealized = price / self.entry_price - 1.0
        trend_strength = self._trend_strength(row)
        continuation = max(
            to_float(row.get("future_ret_5m")),
            to_float(row.get("future_ret_10m")),
            to_float(row.get("future_ret_20m")),
            to_float(row.get("future_ret_30m")),
            0.0,
        )

        can_add = (
            unrealized >= ADD_LONG_MIN_UNREALIZED_PCT
            and self.bars_held >= ADD_LONG_MIN_HOLD_BARS
            and self.add_long_count < ADD_LONG_MAX_COUNT
            and self.position_fraction < MAX_TOTAL_POSITION_FRACTION
            and trend_strength >= 0.45
        )

        if not can_add:
            return -BAD_ADD_LONG_PENALTY

        old_fraction = self.position_fraction
        add_fraction = min(ADD_LONG_FRACTION * 0.10, MAX_TOTAL_POSITION_FRACTION - self.position_fraction)
        if add_fraction <= 0:
            return -BAD_ADD_LONG_PENALTY

        new_fraction = old_fraction + add_fraction
        self.entry_price = ((self.entry_price * old_fraction) + (price * add_fraction)) / max(new_fraction, 1e-9)
        self.position_fraction = new_fraction
        self.add_long_count += 1

        reward = ADD_LONG_REWARD
        reward -= transaction_cost
        reward += min(continuation * ADD_LONG_FUTURE_REWARD_MULT, ADD_LONG_FUTURE_REWARD_CAP)
        reward += trend_strength * 2.0

        if continuation <= 0:
            reward -= BAD_ADD_LONG_PENALTY

        self._record_event(row, "ADD_LONG", price, reward, "agent_add_long")
        return reward

    def _exit_long(self, row: pd.Series, price: float, transaction_cost: float, reason: str) -> float:
        if self.position != 1 or self.entry_price <= 0:
            return -0.10

        pnl_pct = price / self.entry_price - 1.0
        realized = pnl_pct * self.position_fraction * 100000.0
        self.realized_pnl += realized

        reward = pnl_pct * REWARD_SCALE * max(self.position_fraction, 0.01)
        reward -= transaction_cost

        future_max = max(
            to_float(row.get("future_max_ret_10m")),
            to_float(row.get("future_max_ret_20m")),
            to_float(row.get("future_max_ret_30m")),
            to_float(row.get("future_max_ret_45m")),
            0.0,
        )
        future_min = min(
            to_float(row.get("future_min_ret_10m")),
            to_float(row.get("future_min_ret_20m")),
            to_float(row.get("future_min_ret_30m")),
            0.0,
        )

        mfe = max(self.mfe, future_max, 0.0)
        mae = min(self.mae, future_min, 0.0)
        giveback = max(0.0, mfe - max(pnl_pct, 0.0))
        capture_ratio = pnl_pct / max(mfe, 1e-6) if pnl_pct > 0 and mfe > 0 else 0.0

        reward += min(max(capture_ratio, 0.0) * MFE_CAPTURE_REWARD_MULT, MFE_CAPTURE_REWARD_CAP)
        reward -= min(abs(mae) * MAE_PENALTY_MULT, MAE_PENALTY_CAP)
        reward -= min(giveback * GIVEBACK_CAPTURE_PENALTY_MULT, GIVEBACK_CAPTURE_PENALTY_CAP)

        if self.bars_held < MIN_QUALITY_HOLD_BARS and pnl_pct > 0:
            reward -= EARLY_WINNER_EXIT_PENALTY

        if self.bars_held >= MIN_QUALITY_HOLD_BARS and pnl_pct > 0:
            reward += QUALITY_WINNER_HOLD_BONUS
            reward += min(self.bars_held / max(LONG_HOLD_TARGET_BARS, 1.0), 1.5) * AVG_HOLD_REWARD_MULT

        self.exit_long_count += 1
        self._record_event(row, "EXIT_LONG", price, reward, reason, realized, mfe, mae, giveback, capture_ratio)

        self.position = 0
        self.entry_price = 0.0
        self.position_fraction = 0.0
        self.high_water = 0.0
        self.low_water = 0.0
        self.bars_held = 0
        self.mfe = 0.0
        self.mae = 0.0
        self.add_long_count = 0

        return reward

    def _hold_reward(self, row: pd.Series) -> float:
        reward = 0.0

        if self.position == 0:
            reward -= FLAT_HOLD_STEP_PENALTY

            if self._entry_quality(row) >= 0.70 and to_float(row.get("future_ret_10m")) > 0.003:
                reward -= 1.0

            return reward

        return reward

    def _record_event(
        self,
        row: pd.Series,
        action: str,
        price: float,
        reward: float,
        reason: str,
        realized_pnl: float = 0.0,
        mfe: float | None = None,
        mae: float | None = None,
        giveback: float | None = None,
        capture_ratio: float = 0.0,
    ) -> None:
        unrealized = 0.0
        if self.position == 1 and self.entry_price > 0:
            unrealized = price / self.entry_price - 1.0

        self.trade_events.append(
            TradeEvent(
                timestamp=str(row.get("timestamp")),
                action=action,
                price=float(price),
                position_fraction=float(self.position_fraction),
                realized_pnl=float(realized_pnl),
                reward=float(reward),
                hold_bars=int(self.bars_held),
                unrealized_pct=float(unrealized),
                mfe_pct=float(self.mfe if mfe is None else mfe),
                mae_pct=float(self.mae if mae is None else mae),
                giveback_from_mfe_pct=float(max(0.0, self.mfe - max(unrealized, 0.0)) if giveback is None else giveback),
                capture_ratio=float(capture_ratio),
                reason=reason,
            )
        )

    def _obs(self) -> np.ndarray:
        row = self.df.iloc[self.idx]
        raw = row[self.feature_cols].astype(float).to_numpy(dtype=np.float32)

        raw = np.nan_to_num(raw, nan=0.0, posinf=10.0, neginf=-10.0)
        raw = np.clip(raw, -10.0, 10.0)

        price = max(to_float(row.get("close")), 1e-9)
        unrealized = 0.0
        giveback = 0.0
        if self.position == 1 and self.entry_price > 0:
            unrealized = price / self.entry_price - 1.0
            giveback = max(0.0, self.mfe - max(unrealized, 0.0))

        pos_state = np.array(
            [
                float(self.position),
                float(unrealized * 100.0),
                float(min(self.bars_held / 30.0, 10.0)),
                float((price - self.entry_price) / max(to_float(row.get("d_atr")), price * 0.002)) if self.entry_price else 0.0,
                float(self.position_fraction),
                float(self.add_long_count),
                float(self.mfe * 100.0),
                float(giveback * 100.0),
            ],
            dtype=np.float32,
        )

        return np.concatenate([raw, pos_state]).astype(np.float32)


# =============================================================================
# Training / diagnostics / inference
# =============================================================================

def collect_diagnostics(env: SPXBreakoutEnv) -> tuple[pd.DataFrame, dict[str, Any]]:
    events = [asdict(e) for e in env.trade_events]
    df = pd.DataFrame(events)

    if df.empty:
        return df, {
            "rows": 0,
            "enter_long_count": 0,
            "add_long_count": 0,
            "exit_long_count": 0,
            "total_realized_pnl": 0.0,
            "trade_quality": {},
        }

    enter_longs = int((df["action"] == "ENTER_LONG").sum())
    add_longs = int((df["action"] == "ADD_LONG").sum())
    exits = df[df["action"] == "EXIT_LONG"].copy()

    total_realized = float(exits["realized_pnl"].sum()) if not exits.empty else 0.0
    wins = exits[exits["realized_pnl"] > 0]
    losses = exits[exits["realized_pnl"] < 0]
    gross_profit = float(wins["realized_pnl"].sum()) if not wins.empty else 0.0
    gross_loss = abs(float(losses["realized_pnl"].sum())) if not losses.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

    trade_quality = {
        "round_trips": int(len(exits)),
        "win_rate": float(len(wins) / max(len(exits), 1)),
        "profit_factor": float(profit_factor),
        "avg_hold_bars": float(exits["hold_bars"].mean()) if not exits.empty else 0.0,
        "avg_mfe_pct": float(exits["mfe_pct"].mean()) if not exits.empty else 0.0,
        "avg_mae_pct": float(exits["mae_pct"].mean()) if not exits.empty else 0.0,
        "avg_giveback_from_mfe_pct": float(exits["giveback_from_mfe_pct"].mean()) if not exits.empty else 0.0,
        "avg_capture_ratio": float(exits["capture_ratio"].mean()) if not exits.empty else 0.0,
        "long_entries": enter_longs,
        "add_longs": add_longs,
        "add_long_count": add_longs,
        "add_long_ratio": float(add_longs / max(enter_longs, 1)),
        "avg_add_long_reward": float(df.loc[df["action"] == "ADD_LONG", "reward"].mean()) if add_longs else 0.0,
        "long_exits": int(len(exits)),
        "short_entries": 0,
        "short_exits": 0,
        "short_entry_ratio": 0.0,
    }

    metrics = {
        "rows": int(len(df)),
        "enter_long_count": enter_longs,
        "add_long_count": add_longs,
        "exit_long_count": int(len(exits)),
        "enter_short_count": 0,
        "exit_short_count": 0,
        "total_realized_pnl": total_realized,
        "trade_quality": trade_quality,
    }

    return df, metrics


def train_model(features: pd.DataFrame | None = None) -> Any:
    if PPO is None or DummyVecEnv is None:
        raise RuntimeError("stable-baselines3 is missing. Run: pip install stable-baselines3 gymnasium")

    if features is None:
        features = load_or_build_features()

    raw_env = SPXBreakoutEnv(features)
    env = DummyVecEnv([lambda: raw_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(MODEL_DIR / "tensorboard_spx_breakout"),
        learning_rate=PPO_LEARNING_RATE,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        gamma=PPO_GAMMA,
        gae_lambda=0.95,
        ent_coef=PPO_ENT_COEF,
        clip_range=0.20,
    )

    model.learn(total_timesteps=TRAIN_TIMESTEPS)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    print(f"[saved] model={MODEL_PATH}")

    diagnostics_df, diagnostic_metrics = collect_diagnostics(raw_env)
    if not diagnostics_df.empty:
        diagnostics_df.to_csv(DIAGNOSTICS_PATH, index=False)
        print(f"[saved] diagnostics={DIAGNOSTICS_PATH}")

    metrics = {
        "created_at_utc": now_utc(),
        "logical_symbol": LOGICAL_SYMBOL,
        "train_symbol": TRAIN_SYMBOL,
        "feature_rows": int(len(features)),
        "model_path": str(MODEL_PATH),
        "feature_path": str(FEATURE_PATH),
        "diagnostics": diagnostic_metrics,
        "config": {
            "data_tag": DATA_TAG,
            "train_timesteps": TRAIN_TIMESTEPS,
            "train_symbol": TRAIN_SYMBOL,
            "logical_symbol": LOGICAL_SYMBOL,
            "action_space": {
                0: "HOLD",
                1: "ENTER_LONG",
                2: "EXIT_LONG",
                3: "ADD_LONG",
            },
            "ppo_learning_rate": PPO_LEARNING_RATE,
            "ppo_n_steps": PPO_N_STEPS,
            "ppo_batch_size": PPO_BATCH_SIZE,
            "ppo_gamma": PPO_GAMMA,
            "ppo_ent_coef": PPO_ENT_COEF,
            "dynamic_giveback_min_mfe": DYNAMIC_GIVEBACK_MIN_MFE,
            "dynamic_giveback_fraction": DYNAMIC_GIVEBACK_FRACTION,
            "dynamic_giveback_floor": DYNAMIC_GIVEBACK_FLOOR,
            "dynamic_giveback_min_hold_bars": DYNAMIC_GIVEBACK_MIN_HOLD_BARS,
            "add_long_enabled": ADD_LONG_ENABLED,
            "add_long_reward": ADD_LONG_REWARD,
            "add_long_fraction": ADD_LONG_FRACTION,
            "add_long_min_unrealized_pct": ADD_LONG_MIN_UNREALIZED_PCT,
            "add_long_min_hold_bars": ADD_LONG_MIN_HOLD_BARS,
            "add_long_max_count": ADD_LONG_MAX_COUNT,
            "add_long_future_reward_mult": ADD_LONG_FUTURE_REWARD_MULT,
            "add_long_future_reward_cap": ADD_LONG_FUTURE_REWARD_CAP,
        },
    }
    write_json(METRICS_PATH, metrics)
    print(f"[saved] metrics={METRICS_PATH}")

    print("\n===== SUMMARY =====")
    print(f"feature_rows={len(features):,}")
    print(f"diagnostic_events={diagnostic_metrics.get('rows', 0):,}")
    print(f"enter_longs={diagnostic_metrics.get('enter_long_count', 0):,}")
    print(f"add_longs={diagnostic_metrics.get('add_long_count', 0):,}")
    print(f"exit_longs={diagnostic_metrics.get('exit_long_count', 0):,}")
    print(f"diagnostic_pnl={diagnostic_metrics.get('total_realized_pnl', 0.0):.2f}")
    tq = diagnostic_metrics.get("trade_quality", {})
    print(f"win_rate={tq.get('win_rate', 0.0):.2%}")
    print(f"profit_factor={tq.get('profit_factor', 0.0):.2f}")
    print(f"avg_hold_bars={tq.get('avg_hold_bars', 0.0):.2f}")
    print(f"avg_capture_ratio={tq.get('avg_capture_ratio', 0.0):.3f}")

    return model


def load_model() -> Any:
    if PPO is None:
        raise RuntimeError("stable-baselines3 is missing. Run: pip install stable-baselines3 gymnasium")
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}. Run with --train first.")
    return PPO.load(str(MODEL_PATH))


def load_or_build_features() -> pd.DataFrame:
    if FEATURE_PATH.exists():
        return pd.read_parquet(FEATURE_PATH)

    frames = fetch_training_data()
    return prepare_features(frames)


def latest_action_and_plan(features: pd.DataFrame | None = None) -> RLTradePlan:
    features = features if features is not None else load_or_build_features()
    model = load_model()

    env = SPXBreakoutEnv(features)
    env.idx = len(features) - 2
    obs = env._obs()

    action, _state = model.predict(obs, deterministic=True)
    action_int = int(action)

    action_name = {
        0: "WAIT",
        1: "LONG_BREAKOUT",
        2: "EXIT_OR_STAY_FLAT",
        3: "ADD_LONG",
    }.get(action_int, "WAIT")

    row = features.iloc[-1]
    price = to_float(row["close"])
    atr = max(to_float(row.get("d_atr")), price * 0.002)

    support = to_float(row.get("daily_support"))
    resistance = to_float(row.get("daily_resistance"))

    market_context = load_json(MARKET_CONTEXT_PATH)
    premarket_plan = load_json(PREMARKET_PLAN_PATH)

    market_bias = str(market_context.get("overall_market_bias", "unknown"))
    trade_permission = str(market_context.get("trade_permission", "unknown"))

    confidence_proxy = compute_confidence_proxy(row, action_int)

    notes = [
        "SPX is represented by the configured training symbol unless your data vendor supports direct SPX bars.",
        "Long-only policy: no short/bearish action is emitted by this trainer.",
        "Daily levels come from pivot highs/lows with rolling high/low fallback.",
        "RL action should be treated as a planning signal, not an unconditional order.",
    ]

    if market_context:
        notes.append("Market context file was detected and included as a gating input.")
    if premarket_plan:
        notes.append("Premarket trade plan file was detected and can be used by the executor as an additional filter.")

    plan = RLTradePlan(
        logical_symbol=LOGICAL_SYMBOL,
        train_symbol=TRAIN_SYMBOL,
        created_at_utc=now_utc(),
        market_bias=market_bias,
        trade_permission=trade_permission,
        current_price=round(price, 4),
        daily_support=round(support, 4),
        daily_resistance=round(resistance, 4),
        nearest_support=round(support, 4),
        nearest_resistance=round(resistance, 4),
        atr_daily=round(atr, 4),
        breakout_long_trigger=round(resistance + atr * BREAKOUT_CONFIRM_ATR, 4),
        long_stop=round(resistance - atr * INITIAL_STOP_ATR, 4),
        long_initial_target=round(resistance + atr * 2.0, 4),
        trailing_atr_multiple=TRAIL_ATR,
        model_action=action_name,
        model_confidence_proxy=round(confidence_proxy, 4),
        feature_snapshot={
            "rsi_1m": round(to_float(row.get("m1_rsi")), 4),
            "rsi_5m": round(to_float(row.get("5m_rsi")), 4),
            "rsi_15m": round(to_float(row.get("15m_rsi")), 4),
            "macd_hist_1m": round(to_float(row.get("m1_macd_hist")), 6),
            "macd_hist_5m": round(to_float(row.get("5m_macd_hist")), 6),
            "macd_hist_15m": round(to_float(row.get("15m_macd_hist")), 6),
            "cci_1m": round(to_float(row.get("m1_cci")), 4),
            "relative_volume_1m": round(to_float(row.get("m1_rel_volume")), 4),
            "dist_to_support_atr": round(to_float(row.get("dist_to_support_atr")), 4),
            "dist_to_resistance_atr": round(to_float(row.get("dist_to_resistance_atr")), 4),
            "trend_alignment_score": round(to_float(row.get("trend_alignment_score")), 4),
            "breakout_continuation_score": round(to_float(row.get("breakout_continuation_score")), 4),
        },
        notes=notes,
    )

    PLAN_PATH.write_text(json.dumps(safe_json(asdict(plan)), indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {PLAN_PATH}")
    return plan


def compute_confidence_proxy(row: pd.Series, action: int) -> float:
    score = 0.0

    if action in {1, 3}:
        score += 0.25 if to_float(row.get("breakout_strength")) > 0 else 0.0
        score += 0.20 if to_float(row.get("m1_macd_hist")) > 0 else 0.0
        score += 0.20 if to_float(row.get("5m_macd_hist")) > 0 else 0.0
        score += 0.15 if to_float(row.get("15m_macd_hist")) > 0 else 0.0
        score += 0.10 if to_float(row.get("m1_rsi")) >= 50 else 0.0
        score += 0.10 if to_float(row.get("m1_rel_volume")) >= 1.0 else 0.0

        if action == 3:
            score = min(1.0, score + 0.10)

    elif action == 2:
        score = 0.50

    else:
        score = 0.35

    return float(max(0.0, min(1.0, score)))


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train/infer long-only SPX breakout RL model.")
    parser.add_argument("--fetch", action="store_true", help="Fetch data and build feature parquet.")
    parser.add_argument("--train", action="store_true", help="Train PPO model.")
    parser.add_argument("--plan", action="store_true", help="Create latest intraday breakout plan from trained model.")
    parser.add_argument("--all", action="store_true", help="Fetch features, train model, then create plan.")
    args = parser.parse_args()

    if not any([args.fetch, args.train, args.plan, args.all]):
        args.all = True

    features = None

    if args.fetch or args.all:
        frames = fetch_training_data()
        features = prepare_features(frames)

    if args.train or args.all:
        if features is None:
            features = load_or_build_features()
        train_model(features)

    if args.plan or args.all:
        plan = latest_action_and_plan(features)
        print(json.dumps(safe_json(asdict(plan)), indent=2))


if __name__ == "__main__":
    main()
