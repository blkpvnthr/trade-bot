#!/usr/bin/env python3
"""
rl_spx_breakout_trainer.py

Reinforcement-learning breakout planner for SPX-style index trading.

Purpose
-------
Train an RL agent on multi-timeframe market data:
    1m, 5m, 15m, 30m, 1h, daily

The model learns to:
    - identify daily support/resistance context
    - plan intraday breakout trades around those levels
    - catch multi-bar continuation moves
    - avoid false breakouts/chop
    - learn dynamic exits instead of relying only on fixed SL/TP

Important practical note
------------------------
Most retail equity APIs do not expose the cash SPX index as normal stock bars.
This module defaults to SPY as the trainable proxy and stores the logical
instrument name as SPX. If your data vendor supports SPX index bars directly,
set SPX_TRAIN_SYMBOL=SPX and wire the fetcher accordingly.

Dependencies
------------
pip install pandas numpy python-dotenv alpaca-py stable-baselines3 gymnasium scikit-learn

Optional:
pip install torch tensorboard

Environment variables
---------------------
APCA_API_KEY_ID / APCA_API_SECRET_KEY
ALPACA_DATA_FEED=iex|sip

SPX_TRAIN_SYMBOL=SPY
SPX_LOGICAL_SYMBOL=SPX
SPX_RL_LOOKBACK_DAYS=730
SPX_RL_TIMESTEPS=300000
SPX_RL_MODEL_PATH=models/spx_breakout_ppo.zip
SPX_RL_FEATURE_PATH=daily_system_outputs/spx_rl_features.parquet
SPX_RL_PLAN_PATH=daily_system_outputs/spx_intraday_breakout_plan.json
"""

from __future__ import annotations

import json
import math
import os
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
YESTERDAY_TAG = (datetime.now(ET) - timedelta(days=1)).strftime("%Y%m%d")
TODAY_TAG = datetime.now(ET).strftime("%Y%m%d")
DATA_TAG = os.getenv("RL_DATA_DATE", TODAY_TAG)

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
TRAIN_TIMESTEPS = int(os.getenv("SPX_RL_TIMESTEPS", "300000"))

FEATURE_PATH = Path(os.getenv("SPX_RL_FEATURE_PATH", OUT_DIR / f"{DATA_TAG}_spx_rl_features.parquet"))
MODEL_PATH = Path(os.getenv("SPX_RL_MODEL_PATH", MODEL_DIR / "spx_breakout_ppo.zip"))
PLAN_PATH = Path(os.getenv("SPX_RL_PLAN_PATH", OUT_DIR / f"{DATA_TAG}_spx_intraday_breakout_plan.json"))

MARKET_CONTEXT_PATH = OUT_DIR / f"{DATA_TAG}_market_context.json"
PREMARKET_PLAN_PATH = OUT_DIR / f"{DATA_TAG}_premarket_trade_plan.json"

ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "sip").lower().strip()

# State/action controls
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

RISK_OFF_PENALTY = float(os.getenv("SPX_RL_RISK_OFF_PENALTY", "0.20"))
FALSE_BREAKOUT_PENALTY = float(os.getenv("SPX_RL_FALSE_BREAKOUT_PENALTY", "0.35"))
HOLDING_REWARD_SCALE = float(os.getenv("SPX_RL_HOLDING_REWARD_SCALE", "10.0"))

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
    breakdown_short_trigger: float
    long_stop: float
    short_stop: float
    long_initial_target: float
    short_initial_target: float
    trailing_atr_multiple: float
    model_action: str
    model_confidence_proxy: float
    feature_snapshot: dict[str, Any]
    notes: list[str]


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

    # Resample 1m bars into higher intraday timeframes.
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
# Indicators
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

    x[f"{prefix}sma5"] = close.rolling(5).mean()
    x[f"{prefix}sma20"] = close.rolling(20).mean()
    x[f"{prefix}dist_sma5"] = close / x[f"{prefix}sma5"] - 1.0
    x[f"{prefix}dist_sma20"] = close / x[f"{prefix}sma20"] - 1.0

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

    return x.replace([np.inf, -np.inf], np.nan)


# =============================================================================
# Daily support/resistance
# =============================================================================

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
        cluster_distance = max(atr * LEVEL_CLUSTER_ATR_MULT, to_float(x.iloc[i].get("close")) * 0.001)

        raw_supports = [to_float(v) for v in hist["pivot_low"].dropna().tolist()]
        raw_resistances = [to_float(v) for v in hist["pivot_high"].dropna().tolist()]

        supports.append(cluster_nearest(raw_supports, to_float(x.iloc[i].get("close")), cluster_distance, below=True))
        resistances.append(cluster_nearest(raw_resistances, to_float(x.iloc[i].get("close")), cluster_distance, below=False))

    x["daily_support"] = supports
    x["daily_resistance"] = resistances

    # Fallback levels if pivot detection is sparse.
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


# =============================================================================
# Multi-timeframe feature assembly
# =============================================================================

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
                "d_rsi",
                "d_macd_hist",
                "d_cci",
                "d_dist_sma5",
                "d_dist_sma20",
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
            f"{tf}_dist_sma5",
            f"{tf}_dist_sma20",
            f"{tf}_rsi",
            f"{tf}_macd_hist",
            f"{tf}_atr_pct",
            f"{tf}_cci",
            f"{tf}_rel_volume",
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
    merged["below_daily_support"] = (close < support).astype(float)
    merged["breakout_strength"] = (close - resistance) / atr
    merged["breakdown_strength"] = (support - close) / atr

    # Future labels are not fed directly to the agent as state, but help diagnostics.
    merged["future_ret_5m"] = close.shift(-5) / close - 1.0
    merged["future_ret_15m"] = close.shift(-15) / close - 1.0
    merged["future_ret_30m"] = close.shift(-30) / close - 1.0

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
    Actions:
        0 = wait / hold current state
        1 = enter long breakout / keep long
        2 = enter short breakdown / keep short
        3 = exit to flat

    The reward favors:
        - entering near confirmed daily S/R breakouts
        - holding through multi-bar continuation
        - exiting before reversal/giveback
        - avoiding false breakouts, chop, and risk-off conditions
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
            shape=(len(self.feature_cols) + 4,),
            dtype=np.float32,
        )

        self.idx = 0
        self.start_idx = 0
        self.end_idx = 0

        self.position = 0  # -1 short, 0 flat, +1 long
        self.entry_price = 0.0
        self.high_water = 0.0
        self.low_water = 0.0
        self.bars_held = 0

    def _feature_columns(self) -> list[str]:
        preferred = [
            "m1_ret_1", "m1_ret_3", "m1_ret_5",
            "m1_dist_sma5", "m1_dist_sma20", "m1_rsi",
            "m1_macd_hist", "m1_atr_pct", "m1_cci", "m1_rel_volume",
            "5m_ret_1", "5m_ret_3", "5m_ret_5", "5m_dist_sma5", "5m_dist_sma20",
            "5m_rsi", "5m_macd_hist", "5m_atr_pct", "5m_cci", "5m_rel_volume",
            "15m_ret_1", "15m_ret_3", "15m_ret_5", "15m_dist_sma5", "15m_dist_sma20",
            "15m_rsi", "15m_macd_hist", "15m_atr_pct", "15m_cci", "15m_rel_volume",
            "30m_ret_1", "30m_ret_3", "30m_ret_5", "30m_dist_sma5", "30m_dist_sma20",
            "30m_rsi", "30m_macd_hist", "30m_atr_pct", "30m_cci", "30m_rel_volume",
            "1h_ret_1", "1h_ret_3", "1h_ret_5", "1h_dist_sma5", "1h_dist_sma20",
            "1h_rsi", "1h_macd_hist", "1h_atr_pct", "1h_cci", "1h_rel_volume",
            "d_rsi", "d_macd_hist", "d_cci", "d_dist_sma5", "d_dist_sma20", "d_rel_volume",
            "dist_to_support_atr", "dist_to_resistance_atr",
            "above_daily_resistance", "below_daily_support",
            "breakout_strength", "breakdown_strength",
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
        self.high_water = 0.0
        self.low_water = 0.0
        self.bars_held = 0

        return self._obs(), {}

    def step(self, action: int):
        row = self.df.iloc[self.idx]
        next_row = self.df.iloc[self.idx + 1]

        price = to_float(row["close"])
        next_price = to_float(next_row["close"])
        atr = max(to_float(row.get("d_atr")), price * 0.002)

        reward = 0.0

        transaction_cost = (COMMISSION_BPS + SLIPPAGE_BPS) / 10000.0

        # Existing position mark-to-market reward.
        if self.position != 0:
            bar_ret = (next_price / price - 1.0) * self.position
            reward += bar_ret * HOLDING_REWARD_SCALE
            self.bars_held += 1

            if self.position > 0:
                self.high_water = max(self.high_water, next_price)
                giveback = (self.high_water - next_price) / atr
                if giveback > TRAIL_ATR:
                    reward -= min(giveback * 0.05, 0.30)

            if self.position < 0:
                self.low_water = min(self.low_water, next_price)
                giveback = (next_price - self.low_water) / atr
                if giveback > TRAIL_ATR:
                    reward -= min(giveback * 0.05, 0.30)

        breakout_ok = (
            to_float(row.get("breakout_strength")) >= BREAKOUT_CONFIRM_ATR
            and to_float(row.get("m1_macd_hist")) > 0
            and to_float(row.get("5m_macd_hist")) > 0
            and to_float(row.get("m1_rsi")) >= 50
        )
        breakdown_ok = (
            to_float(row.get("breakdown_strength")) >= BREAKOUT_CONFIRM_ATR
            and to_float(row.get("m1_macd_hist")) < 0
            and to_float(row.get("5m_macd_hist")) < 0
            and to_float(row.get("m1_rsi")) <= 50
        )

        near_resistance = abs(to_float(row.get("dist_to_resistance_atr"))) <= ENTRY_NEAR_LEVEL_ATR
        near_support = abs(to_float(row.get("dist_to_support_atr"))) <= ENTRY_NEAR_LEVEL_ATR

        # Action handling.
        if action == 1:  # long
            if self.position <= 0:
                reward -= transaction_cost
                if breakout_ok or near_resistance:
                    reward += 0.05
                else:
                    reward -= FALSE_BREAKOUT_PENALTY
                self._enter(position=1, price=price)

        elif action == 2:  # short
            if self.position >= 0:
                reward -= transaction_cost
                if breakdown_ok or near_support:
                    reward += 0.05
                else:
                    reward -= FALSE_BREAKOUT_PENALTY
                self._enter(position=-1, price=price)

        elif action == 3:  # exit
            if self.position != 0:
                pnl = (price / self.entry_price - 1.0) * self.position
                reward += pnl * HOLDING_REWARD_SCALE
                reward -= transaction_cost

                # Encourage profitable exits after multi-bar moves.
                if self.bars_held >= 3 and pnl > 0:
                    reward += min(pnl * 20.0, 0.50)

                self._exit()

        # Risk-off penalty for forcing longs when daily momentum is weak.
        if action == 1 and (
            to_float(row.get("d_macd_hist")) < 0
            or to_float(row.get("d_dist_sma20")) < 0
        ):
            reward -= RISK_OFF_PENALTY

        self.idx += 1
        terminated = self.idx >= self.end_idx
        truncated = False

        if terminated and self.position != 0:
            final_price = to_float(self.df.iloc[self.idx]["close"])
            pnl = (final_price / self.entry_price - 1.0) * self.position
            reward += pnl * HOLDING_REWARD_SCALE
            self._exit()

        return self._obs(), float(reward), terminated, truncated, {}

    def _enter(self, position: int, price: float) -> None:
        self.position = position
        self.entry_price = price
        self.high_water = price
        self.low_water = price
        self.bars_held = 0

    def _exit(self) -> None:
        self.position = 0
        self.entry_price = 0.0
        self.high_water = 0.0
        self.low_water = 0.0
        self.bars_held = 0

    def _obs(self) -> np.ndarray:
        row = self.df.iloc[self.idx]
        raw = row[self.feature_cols].astype(float).to_numpy(dtype=np.float32)

        # Robust clipping keeps PPO stable without expensive scaling objects.
        raw = np.nan_to_num(raw, nan=0.0, posinf=10.0, neginf=-10.0)
        raw = np.clip(raw, -10.0, 10.0)

        price = max(to_float(row.get("close")), 1e-9)
        unrealized = 0.0
        if self.position != 0 and self.entry_price > 0:
            unrealized = (price / self.entry_price - 1.0) * self.position

        pos_state = np.array(
            [
                float(self.position),
                float(unrealized * 100.0),
                float(min(self.bars_held / 30.0, 10.0)),
                float((price - self.entry_price) / max(to_float(row.get("d_atr")), price * 0.002)) if self.entry_price else 0.0,
            ],
            dtype=np.float32,
        )

        return np.concatenate([raw, pos_state]).astype(np.float32)


# =============================================================================
# Training / inference
# =============================================================================

def train_model(features: pd.DataFrame | None = None) -> Any:
    if PPO is None or DummyVecEnv is None:
        raise RuntimeError("stable-baselines3 is missing. Run: pip install stable-baselines3 gymnasium")

    if features is None:
        features = load_or_build_features()

    env = DummyVecEnv([lambda: SPXBreakoutEnv(features)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(MODEL_DIR / "tensorboard_spx_breakout"),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.20,
    )

    model.learn(total_timesteps=TRAIN_TIMESTEPS)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    print(f"[saved] {MODEL_PATH}")
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
        2: "SHORT_BREAKDOWN",
        3: "EXIT_OR_STAY_FLAT",
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
        breakdown_short_trigger=round(support - atr * BREAKOUT_CONFIRM_ATR, 4),
        long_stop=round(resistance - atr * INITIAL_STOP_ATR, 4),
        short_stop=round(support + atr * INITIAL_STOP_ATR, 4),
        long_initial_target=round(resistance + atr * 2.0, 4),
        short_initial_target=round(support - atr * 2.0, 4),
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
        },
        notes=notes,
    )

    PLAN_PATH.write_text(json.dumps(safe_json(asdict(plan)), indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {PLAN_PATH}")
    return plan


def compute_confidence_proxy(row: pd.Series, action: int) -> float:
    """
    Simple transparent confidence proxy.
    PPO itself returns an action, not a calibrated probability.
    This score combines multi-timeframe indicator agreement around S/R.
    """
    score = 0.0

    if action == 1:
        score += 0.25 if to_float(row.get("breakout_strength")) > 0 else 0.0
        score += 0.20 if to_float(row.get("m1_macd_hist")) > 0 else 0.0
        score += 0.20 if to_float(row.get("5m_macd_hist")) > 0 else 0.0
        score += 0.15 if to_float(row.get("15m_macd_hist")) > 0 else 0.0
        score += 0.10 if to_float(row.get("m1_rsi")) >= 50 else 0.0
        score += 0.10 if to_float(row.get("m1_rel_volume")) >= 1.0 else 0.0

    elif action == 2:
        score += 0.25 if to_float(row.get("breakdown_strength")) > 0 else 0.0
        score += 0.20 if to_float(row.get("m1_macd_hist")) < 0 else 0.0
        score += 0.20 if to_float(row.get("5m_macd_hist")) < 0 else 0.0
        score += 0.15 if to_float(row.get("15m_macd_hist")) < 0 else 0.0
        score += 0.10 if to_float(row.get("m1_rsi")) <= 50 else 0.0
        score += 0.10 if to_float(row.get("m1_rel_volume")) >= 1.0 else 0.0

    elif action == 3:
        score = 0.50

    else:
        score = 0.35

    return float(max(0.0, min(1.0, score)))


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train/infer SPX breakout RL model.")
    parser.add_argument("--fetch", action="store_true", help="Fetch data and build feature parquet.")
    parser.add_argument("--train", action="store_true", help="Train PPO model.")
    parser.add_argument("--plan", action="store_true", help="Create latest intraday breakout plan from trained model.")
    parser.add_argument("--all", action="store_true", help="Fetch features, train model, then create plan.")
    args = parser.parse_args()

    if not any([args.fetch, args.train, args.plan, args.all]):
        args.plan = True

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
