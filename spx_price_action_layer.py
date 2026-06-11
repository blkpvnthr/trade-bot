#!/usr/bin/env python3
"""
spx_price_action_layer.py

Dedicated SPX/SPY price-action execution layer.

Purpose
-------
This layer watches SPX-style price action through a tradable proxy, SPY by
default, and looks for order setups around:

    - daily support / resistance
    - opening range breakout
    - VWAP reclaim / VWAP hold
    - 1m / 5m / 15m momentum alignment
    - failed-breakout exits
    - trailing giveback exits

It can run independently from run.py or be launched by run.py as a dedicated
market-index timing layer.

Default behavior:
    python spx_price_action_layer.py

Default tradable symbol:
    SPX_LOGICAL_SYMBOL=SPX
    SPX_TRADE_SYMBOL=SPY

This file submits Alpaca orders when SPX_LAYER_EXECUTE=1.
It defaults to paper trading.
"""

from __future__ import annotations

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

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    gym = None
    spaces = None

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception:
    PPO = None
    DummyVecEnv = None

from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


ET = ZoneInfo("America/New_York")
TODAY_TAG = datetime.now(ET).strftime("%Y%m%d")
DATA_TAG = os.getenv("RUN_DATA_DATE", os.getenv("SPX_LAYER_DATA_DATE", TODAY_TAG))

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_instance_env()

OUT_DIR = PROJECT_ROOT / "system-data" / DATA_TAG
LOG_DIR = PROJECT_ROOT / "logs" / "execution" / DATA_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SIGNAL_PATH = OUT_DIR / f"{DATA_TAG}_spx_price_action_signal.json"
STATE_PATH = OUT_DIR / f"{DATA_TAG}_spx_price_action_state.json"
EVENTS_PATH = LOG_DIR / f"{DATA_TAG}_spx_price_action_events.jsonl"
ORDERS_CSV_PATH = LOG_DIR / f"{DATA_TAG}_spx_price_action_orders.csv"

MARKET_CONTEXT_PATH = OUT_DIR / f"{DATA_TAG}_market_context.json"
SPX_RL_PLAN_PATH = PROJECT_ROOT / "system-data" / f"{DATA_TAG}_SPX" / f"{DATA_TAG}_spx_intraday_breakout_plan.json"

LOGICAL_SYMBOL = os.getenv("SPX_LOGICAL_SYMBOL", "SPX").upper().strip()
TRADE_SYMBOL = os.getenv("SPX_TRADE_SYMBOL", os.getenv("SPX_TRAIN_SYMBOL", "SPY")).upper().strip()

ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "sip").lower().strip()
PAPER = os.getenv("APCA_PAPER", "1").lower() in {"1", "true", "yes"}
EXECUTE_ORDERS = os.getenv("SPX_LAYER_EXECUTE", "1").lower() in {"1", "true", "yes"}

SCAN_SECONDS = int(os.getenv("SPX_LAYER_SCAN_SECONDS", "10"))
LOOKBACK_DAYS = int(os.getenv("SPX_LAYER_LOOKBACK_DAYS", "8"))

MIN_BUYING_POWER = float(os.getenv("SPX_LAYER_MIN_BUYING_POWER", "250"))
MAX_POSITION_DOLLARS = float(os.getenv("SPX_LAYER_MAX_POSITION_DOLLARS", "3500"))
MAX_POSITION_WEIGHT = float(os.getenv("SPX_LAYER_MAX_POSITION_WEIGHT", "0.08"))
MIN_ORDER_DOLLARS = float(os.getenv("SPX_LAYER_MIN_ORDER_DOLLARS", "100"))
BASE_ORDER_DOLLARS = float(os.getenv("SPX_LAYER_BASE_ORDER_DOLLARS", "1500"))

DAILY_LOSS_LIMIT_PCT = float(os.getenv("SPX_LAYER_DAILY_LOSS_LIMIT_PCT", "0.012"))
DAILY_LOSS_LIMIT_DOLLARS = float(os.getenv("SPX_LAYER_DAILY_LOSS_LIMIT_DOLLARS", "750"))

AVOID_FIRST_MINUTES = int(os.getenv("SPX_LAYER_AVOID_FIRST_MINUTES", "5"))
AVOID_LAST_MINUTES = int(os.getenv("SPX_LAYER_AVOID_LAST_MINUTES", "10"))

ORB_MINUTES = int(os.getenv("SPX_LAYER_ORB_MINUTES", "15"))
SR_LOOKBACK_DAYS = int(os.getenv("SPX_LAYER_SR_LOOKBACK_DAYS", "20"))

MIN_SETUP_SCORE = float(os.getenv("SPX_LAYER_MIN_SETUP_SCORE", "0.68"))
MIN_BREAKOUT_SCORE = float(os.getenv("SPX_LAYER_MIN_BREAKOUT_SCORE", "0.72"))
MIN_RECLAIM_SCORE = float(os.getenv("SPX_LAYER_MIN_RECLAIM_SCORE", "0.70"))

STOP_LOSS_PCT = float(os.getenv("SPX_LAYER_STOP_LOSS_PCT", "0.006"))
TAKE_PROFIT_R = float(os.getenv("SPX_LAYER_TAKE_PROFIT_R", "2.0"))
TRAILING_GIVEBACK_PCT = float(os.getenv("SPX_LAYER_TRAILING_GIVEBACK_PCT", "0.0045"))
MIN_TRAIL_PROFIT_PCT = float(os.getenv("SPX_LAYER_MIN_TRAIL_PROFIT_PCT", "0.004"))

ALLOW_MARKET_CONTEXT_CAUTION = os.getenv("SPX_LAYER_ALLOW_CAUTION", "1").lower() in {"1", "true", "yes"}
USE_RL_PLAN_CONFIRMATION = os.getenv("SPX_LAYER_USE_RL_PLAN", "1").lower() in {"1", "true", "yes"}
REQUIRE_RL_FOR_ENTRY = os.getenv("SPX_LAYER_REQUIRE_RL_FOR_ENTRY", "0").lower() in {"1", "true", "yes"}

# One max-return trade per session.
ONE_TRADE_PER_SESSION = os.getenv("SPX_LAYER_ONE_TRADE_PER_SESSION", "1").lower() in {"1", "true", "yes"}
LEARN_MAX_RETURN_TRADE = os.getenv("SPX_LAYER_LEARN_MAX_RETURN_TRADE", "1").lower() in {"1", "true", "yes"}
CANDIDATE_LOG_PATH = OUT_DIR / f"{DATA_TAG}_spx_max_return_candidates.csv"
SESSION_LABELS_PATH = OUT_DIR / f"{DATA_TAG}_spx_max_return_session_labels.csv"
MAX_RETURN_MODEL_PATH = OUT_DIR / f"{DATA_TAG}_spx_max_return_model.json"

# Minimum predicted quality required before spending the only trade of the day.
MAX_RETURN_MIN_SCORE = float(os.getenv("SPX_LAYER_MAX_RETURN_MIN_SCORE", "0.72"))

# Do not spend the one trade too early unless the setup is exceptional.
EARLIEST_ENTRY_MINUTE = int(os.getenv("SPX_LAYER_EARLIEST_ENTRY_MINUTE", "15"))
EARLY_ENTRY_EXCEPTION_SCORE = float(os.getenv("SPX_LAYER_EARLY_ENTRY_EXCEPTION_SCORE", "0.90"))

# If no trade has fired by this time, allow the best available strong setup.
LAST_ENTRY_MINUTE = int(os.getenv("SPX_LAYER_LAST_ENTRY_MINUTE", "240"))

# Oracle labeling horizon used to learn the best intraday entry type.
MAX_RETURN_LOOKAHEAD_BARS = int(os.getenv("SPX_LAYER_MAX_RETURN_LOOKAHEAD_BARS", "60"))
MAX_RETURN_MIN_FUTURE_RETURN = float(os.getenv("SPX_LAYER_MAX_RETURN_MIN_FUTURE_RETURN", "0.003"))

# Reinforcement-learning policy for intraday multibar gain moves.
# Actions:
#   0 = HOLD
#   1 = ENTER_LONG
#   2 = EXIT_LONG
#
# Training features include:
#   1m / 5m / 15m / 1h / 4h trend, momentum, VWAP distance, volume, and structure.
RL_ENABLED = os.getenv("SPX_LAYER_RL_ENABLED", "1").lower() in {"1", "true", "yes"}
RL_TRAIN_ON_START = os.getenv("SPX_LAYER_RL_TRAIN_ON_START", "0").lower() in {"1", "true", "yes"}
RL_REQUIRE_FOR_ENTRY = os.getenv("SPX_LAYER_RL_REQUIRE_FOR_ENTRY", "0").lower() in {"1", "true", "yes"}
RL_REQUIRE_FOR_EXIT = os.getenv("SPX_LAYER_RL_REQUIRE_FOR_EXIT", "0").lower() in {"1", "true", "yes"}

RL_MODEL_PATH = OUT_DIR / f"{DATA_TAG}_spx_multibar_gain_rl_model.zip"
RL_METRICS_PATH = OUT_DIR / f"{DATA_TAG}_spx_multibar_gain_rl_metrics.json"
RL_DATASET_PATH = OUT_DIR / f"{DATA_TAG}_spx_multibar_gain_rl_dataset.csv"
RL_SIGNAL_PATH = OUT_DIR / f"{DATA_TAG}_spx_multibar_gain_rl_signal.json"

RL_TRAIN_LOOKBACK_DAYS = int(os.getenv("SPX_LAYER_RL_TRAIN_LOOKBACK_DAYS", "59"))
RL_TIMESTEPS = int(os.getenv("SPX_LAYER_RL_TIMESTEPS", "25000"))
RL_MAX_EPISODE_BARS = int(os.getenv("SPX_LAYER_RL_MAX_EPISODE_BARS", "180"))

RL_LEARNING_RATE = float(os.getenv("SPX_LAYER_RL_LEARNING_RATE", "0.0001465181270477539"))
RL_GAMMA = float(os.getenv("SPX_LAYER_RL_GAMMA", "0.9886884381471769"))
RL_N_STEPS = int(os.getenv("SPX_LAYER_RL_N_STEPS", "1024"))
RL_BATCH_SIZE = int(os.getenv("SPX_LAYER_RL_BATCH_SIZE", "128"))
RL_ENT_COEF = float(os.getenv("SPX_LAYER_RL_ENT_COEF", "0.00793730366338695"))

RL_MIN_ENTRY_CONFIDENCE = float(os.getenv("SPX_LAYER_RL_MIN_ENTRY_CONFIDENCE", "0.7164247276618194"))
RL_MIN_EXIT_CONFIDENCE = float(os.getenv("SPX_LAYER_RL_MIN_EXIT_CONFIDENCE", "0.4326416408435226"))

RL_REWARD_SCALE = float(os.getenv("SPX_LAYER_RL_REWARD_SCALE", "123.98760355121524"))
RL_ENTRY_BONUS = float(os.getenv("SPX_LAYER_RL_ENTRY_BONUS", "1.0736321062529641"))
RL_BAD_ENTRY_PENALTY = float(os.getenv("SPX_LAYER_RL_BAD_ENTRY_PENALTY", "2.78478083560159"))
RL_EARLY_MULTIBAR_BONUS_MULT = float(os.getenv("SPX_LAYER_RL_EARLY_MULTIBAR_BONUS_MULT", "201.93742669544676"))
RL_HOLD_WINNER_MULT = float(os.getenv("SPX_LAYER_RL_HOLD_WINNER_MULT", "55.993219878576085"))
RL_MFE_CAPTURE_MULT = float(os.getenv("SPX_LAYER_RL_MFE_CAPTURE_MULT", "19.767240684961102"))
RL_GIVEBACK_PENALTY_MULT = float(os.getenv("SPX_LAYER_RL_GIVEBACK_PENALTY_MULT", "25.565381568951203"))
RL_MAE_PENALTY_MULT = float(os.getenv("SPX_LAYER_RL_MAE_PENALTY_MULT", "4.680926975116515"))
RL_MIN_QUALITY_HOLD_BARS = int(os.getenv("SPX_LAYER_RL_MIN_QUALITY_HOLD_BARS", "11"))
RL_FLAT_HOLD_PENALTY = float(os.getenv("SPX_LAYER_RL_FLAT_HOLD_PENALTY", "0.0036799118887083098"))

TIME_IN_FORCE = TimeInForce.DAY

SPX_RL_TUNED_PARAMS = {
    "train_lookback_days": 59,
    "timesteps": 25000,
    "max_episode_bars": 180,
    "learning_rate": 0.0001465181270477539,
    "gamma": 0.9886884381471769,
    "n_steps": 1024,
    "batch_size": 128,
    "ent_coef": 0.00793730366338695,
    "reward_scale": 123.98760355121524,
    "entry_bonus": 1.0736321062529641,
    "bad_entry_penalty": 2.78478083560159,
    "early_multibar_bonus_mult": 201.93742669544676,
    "hold_winner_mult": 55.993219878576085,
    "mfe_capture_mult": 19.767240684961102,
    "giveback_penalty_mult": 25.565381568951203,
    "mae_penalty_mult": 4.680926975116515,
    "min_quality_hold_bars": 11,
    "flat_hold_penalty": 0.0036799118887083098,
    "min_entry_confidence": 0.7164247276618194,
    "min_exit_confidence": 0.4326416408435226
}
SPX_RL_TUNED_STUDY_VALUE = 46.63869029737573
SPX_RL_TUNED_DATA_TAG = "20260514"



@dataclass
class SPXSignal:
    created_at_utc: str
    logical_symbol: str
    trade_symbol: str
    action: str
    setup_type: str
    price: float
    setup_score: float
    breakout_score: float
    reclaim_score: float
    direction_score_1m: float
    direction_score_5m: float
    direction_score_15m: float
    vwap: float
    orb_high: float
    orb_low: float
    daily_support: float
    daily_resistance: float
    stop_loss: float | None
    take_profit: float | None
    qty: int
    notional: float
    max_return_score: float
    session_trade_used: bool
    reason: str


@dataclass
class SPXPositionState:
    symbol: str = ""
    qty: float = 0.0
    avg_entry_price: float = 0.0
    high_water_price: float = 0.0
    entry_time_utc: str = ""
    adds_today: int = 0


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_utc_iso() -> str:
    return now_utc().isoformat()


def f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def i(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


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


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe_json(payload), indent=2) + "\n", encoding="utf-8")


def write_event(event: str, payload: dict[str, Any] | None = None) -> None:
    row = {
        "timestamp_utc": now_utc_iso(),
        "event": event,
        "payload": safe_json(payload or {}),
    }
    with EVENTS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=safe_json) + "\n")


def append_order_row(row: dict[str, Any]) -> None:
    df = pd.DataFrame([row])
    header = not ORDERS_CSV_PATH.exists()
    df.to_csv(ORDERS_CSV_PATH, index=False, mode="a", header=header)


def get_feed() -> DataFeed:
    return DataFeed.SIP if ALPACA_DATA_FEED == "sip" else DataFeed.IEX


def get_credentials() -> tuple[str, str]:
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY")
    return key, secret


def data_client() -> StockHistoricalDataClient:
    key, secret = get_credentials()
    return StockHistoricalDataClient(key, secret)


def trading_client() -> TradingClient:
    key, secret = get_credentials()
    return TradingClient(key, secret, paper=PAPER)


def market_clock_ok(client: TradingClient, allow_for_exit: bool = False) -> tuple[bool, str]:
    try:
        clock = client.get_clock()
    except Exception as exc:
        print(f"[spx-layer] clock fetch failed: {exc}")
        return False, "clock_unavailable"
    now_et = clock.timestamp.astimezone(ET)

    open_time = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    minutes_from_open = (now_et - open_time).total_seconds() / 60.0
    minutes_to_close = (close_time - now_et).total_seconds() / 60.0

    if not clock.is_open:
        return False, "market_closed"

    if not allow_for_exit and minutes_from_open < AVOID_FIRST_MINUTES:
        return False, f"opening_noise_window:{minutes_from_open:.1f}m"

    if not allow_for_exit and minutes_to_close < AVOID_LAST_MINUTES:
        return False, f"near_close_window:{minutes_to_close:.1f}m"

    return True, "market_open_ok"


def account_snapshot(client: TradingClient) -> dict[str, float]:
    try:
        account = client.get_account()
    except Exception as exc:
        print(f"[spx-layer] account snapshot failed: {exc}")
        return {"equity": 0.0, "cash": 0.0, "buying_power": 0.0, "portfolio_value": 0.0}
    return {
        "equity": f(getattr(account, "equity", 0.0)),
        "cash": f(getattr(account, "cash", 0.0)),
        "buying_power": f(getattr(account, "buying_power", 0.0)),
        "portfolio_value": f(getattr(account, "portfolio_value", 0.0)),
    }


def daily_loss_ok(account: dict[str, float]) -> tuple[bool, str]:
    state = read_json(STATE_PATH, {})
    today = datetime.now(ET).date().isoformat()
    equity = account["equity"]

    if state.get("date_et") != today or f(state.get("start_equity")) <= 0:
        state["date_et"] = today
        state["start_equity"] = equity
        write_json(STATE_PATH, state)

    start_equity = max(f(state.get("start_equity"), equity), 1.0)
    pnl = equity - start_equity
    ret = pnl / start_equity
    max_loss = min(abs(DAILY_LOSS_LIMIT_DOLLARS), start_equity * abs(DAILY_LOSS_LIMIT_PCT))

    if pnl <= -max_loss or ret <= -abs(DAILY_LOSS_LIMIT_PCT):
        return False, f"daily_loss_limit:pnl={pnl:.2f}:ret={ret:.2%}:max_loss={max_loss:.2f}"

    return True, f"daily_loss_ok:pnl={pnl:.2f}:ret={ret:.2%}"


def market_context_allows_entry() -> tuple[bool, str]:
    ctx = read_json(MARKET_CONTEXT_PATH, {})
    if not ctx:
        return True, "market_context_missing_allow"

    permission = str(ctx.get("trade_permission", "cautious")).lower().strip()
    bias = str(ctx.get("overall_market_bias", "mixed")).lower().strip()
    risk = str(ctx.get("risk_level", "medium")).lower().strip()

    if permission == "halt_new_buys":
        return False, f"market_context_halt:bias={bias}:risk={risk}"

    if bias in {"bearish", "risk_off"} and risk in {"high", "extreme"}:
        return False, f"market_context_risk_off:bias={bias}:risk={risk}"

    if permission == "cautious" and not ALLOW_MARKET_CONTEXT_CAUTION:
        return False, f"market_context_cautious_block:bias={bias}:risk={risk}"

    return True, f"market_context_ok:permission={permission}:bias={bias}:risk={risk}"


def fetch_bars(symbol: str, timeframe: TimeFrame, lookback_days: int) -> pd.DataFrame:
    client = data_client()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        feed=get_feed(),
    )

    try:
        df = client.get_stock_bars(req).df
    except Exception as exc:
        _preview = str(exc)[:300]
        print(f"[spx] bar fetch failed symbol={symbol} tf={timeframe}: {exc}")
        write_event("external_http_error", {
            "module": "spx_price_action_layer",
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


def fetch_1m_bars(symbol: str) -> pd.DataFrame:
    return fetch_bars(symbol, TimeFrame.Minute, LOOKBACK_DAYS)


def fetch_daily_bars(symbol: str) -> pd.DataFrame:
    return fetch_bars(symbol, TimeFrame.Day, max(SR_LOOKBACK_DAYS + 20, 60))


def resample_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    x = df.copy().set_index("timestamp").sort_index()
    out = pd.DataFrame({
        "open": x["open"].resample(rule).first(),
        "high": x["high"].resample(rule).max(),
        "low": x["low"].resample(rule).min(),
        "close": x["close"].resample(rule).last(),
        "volume": x["volume"].resample(rule).sum(),
    }).dropna().reset_index()
    out["symbol"] = TRADE_SYMBOL
    return out


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
    out["ret_10"] = close.pct_change(10)

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
    cum_vol = vol.replace(0, np.nan).cumsum()
    out["vwap"] = pv.cumsum() / cum_vol

    # Explicit RL training features.
    out["sma5_above_sma20"] = (out["sma5"] > out["sma20"]).astype(float)
    out["sma20_below_sma5"] = (out["sma20"] < out["sma5"]).astype(float)
    out["sma5_cross_up_sma20"] = (
        (out["sma5"] > out["sma20"])
        & (out["sma5"].shift(1) <= out["sma20"].shift(1))
    ).astype(float)
    out["sma5_slope"] = out["sma5"].pct_change()
    out["sma20_slope"] = out["sma20"].pct_change()
    out["macd_hist_slope"] = out["macd_hist"].diff()

    out["rel_volume"] = vol / vol.rolling(20).mean().replace(0, np.nan)
    out["rolling_high_20"] = high.rolling(20).max()
    out["rolling_low_20"] = low.rolling(20).min()

    return out.replace([np.inf, -np.inf], np.nan)


def direction_score(row: pd.Series) -> float:
    parts = [
        1.0 if f(row.get("ret_1")) > 0 else 0.0,
        1.0 if f(row.get("ret_3")) > 0 else 0.0,
        1.0 if f(row.get("macd_hist")) > 0 else 0.0,
        1.0 if f(row.get("rsi14"), 50.0) >= 50.0 else 0.0,
        1.0 if f(row.get("close")) >= f(row.get("vwap"), f(row.get("close"))) else 0.0,
    ]
    return float(np.mean(parts))


def daily_levels(daily: pd.DataFrame, current_price: float) -> tuple[float, float, float]:
    if daily.empty:
        return current_price * 0.995, current_price * 1.005, current_price * 0.01

    x = add_indicators(daily)
    recent = x.tail(SR_LOOKBACK_DAYS).copy()
    atr = f(x["atr14"].iloc[-1], current_price * 0.01)
    support = f(recent["low"].min(), current_price * 0.995)
    resistance = f(recent["high"].max(), current_price * 1.005)
    return support, resistance, max(atr, current_price * 0.002)


def opening_range(df_1m: pd.DataFrame) -> tuple[float, float]:
    if df_1m.empty:
        return 0.0, 0.0

    x = df_1m.copy()
    x["ts_et"] = x["timestamp"].dt.tz_convert(ET)
    today = datetime.now(ET).date()
    session = x[x["ts_et"].dt.date == today].copy()

    if session.empty:
        return 0.0, 0.0

    start = datetime.combine(today, datetime.min.time(), tzinfo=ET).replace(hour=9, minute=30)
    end = start + timedelta(minutes=ORB_MINUTES)

    orb = session[(session["ts_et"] >= start) & (session["ts_et"] < end)].copy()
    if orb.empty:
        return 0.0, 0.0

    return f(orb["high"].max()), f(orb["low"].min())


def latest_features() -> dict[str, Any]:
    bars_1m = add_indicators(fetch_1m_bars(TRADE_SYMBOL))
    if bars_1m.empty or len(bars_1m) < 60:
        raise RuntimeError(f"not enough 1m bars for {TRADE_SYMBOL}")

    bars_5m = add_indicators(resample_bars(bars_1m, "5min"))
    bars_15m = add_indicators(resample_bars(bars_1m, "15min"))
    daily = fetch_daily_bars(TRADE_SYMBOL)

    r1 = bars_1m.iloc[-1]
    r5 = bars_5m.iloc[-1] if not bars_5m.empty else r1
    r15 = bars_15m.iloc[-1] if not bars_15m.empty else r1

    price = f(r1["close"])
    support, resistance, atr_daily = daily_levels(daily, price)
    orb_high, orb_low = opening_range(bars_1m)

    score_1m = direction_score(r1)
    score_5m = direction_score(r5)
    score_15m = direction_score(r15)

    above_vwap = price >= f(r1.get("vwap"), price)
    breakout_orb = orb_high > 0 and price > orb_high
    breakout_daily = resistance > 0 and price > resistance
    reclaim_vwap = above_vwap and f(r1.get("ret_3")) > 0 and f(r1.get("macd_hist")) > 0
    support_bounce = support > 0 and (price - support) / price <= 0.004 and score_1m >= 0.60

    breakout_score = float(np.mean([
        breakout_orb,
        breakout_daily,
        score_1m >= 0.60,
        score_5m >= 0.60,
        score_15m >= 0.55,
        above_vwap,
        f(r1.get("rel_volume"), 1.0) >= 1.0,
    ]))

    reclaim_score = float(np.mean([
        reclaim_vwap,
        score_1m >= 0.60,
        score_5m >= 0.55,
        above_vwap,
        f(r1.get("macd_hist")) > 0,
        f(r1.get("rsi14"), 50.0) >= 50,
    ]))

    setup_score = max(breakout_score, reclaim_score, 0.65 if support_bounce else 0.0)

    return {
        "price": price,
        "row_1m": r1,
        "row_5m": r5,
        "row_15m": r15,
        "support": support,
        "resistance": resistance,
        "atr_daily": atr_daily,
        "orb_high": orb_high,
        "orb_low": orb_low,
        "score_1m": score_1m,
        "score_5m": score_5m,
        "score_15m": score_15m,
        "above_vwap": above_vwap,
        "breakout_score": breakout_score,
        "reclaim_score": reclaim_score,
        "setup_score": setup_score,
        "support_bounce": support_bounce,
    }



def session_minutes_from_open() -> float:
    now_et = datetime.now(ET)
    open_dt = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    return max(0.0, (now_et - open_dt).total_seconds() / 60.0)


def session_trade_used() -> bool:
    state = read_json(STATE_PATH, {})
    today = datetime.now(ET).date().isoformat()

    if state.get("date_et") != today:
        return False

    return bool(state.get("spx_one_trade_used", False))


def mark_session_trade_used(signal_obj: SPXSignal | None = None) -> None:
    state = read_json(STATE_PATH, {})
    state["date_et"] = datetime.now(ET).date().isoformat()
    state["spx_one_trade_used"] = True
    state["spx_one_trade_used_at_utc"] = now_utc_iso()
    if signal_obj is not None:
        state["spx_one_trade_signal"] = asdict(signal_obj)
    write_json(STATE_PATH, state)


def reset_session_if_new_day() -> None:
    state = read_json(STATE_PATH, {})
    today = datetime.now(ET).date().isoformat()
    if state.get("date_et") != today:
        state = {
            "date_et": today,
            "start_equity": state.get("start_equity", 0.0),
            "spx_one_trade_used": False,
            "position_state": asdict(SPXPositionState(symbol=TRADE_SYMBOL)),
        }
        write_json(STATE_PATH, state)


def append_candidate_row(row: dict[str, Any]) -> None:
    df = pd.DataFrame([safe_json(row)])
    header = not CANDIDATE_LOG_PATH.exists()
    df.to_csv(CANDIDATE_LOG_PATH, index=False, mode="a", header=header)


def feature_payload_for_learning(features: dict[str, Any]) -> dict[str, float]:
    return {
        "setup_score": float(features.get("setup_score", 0.0)),
        "breakout_score": float(features.get("breakout_score", 0.0)),
        "reclaim_score": float(features.get("reclaim_score", 0.0)),
        "direction_score_1m": float(features.get("score_1m", 0.0)),
        "direction_score_5m": float(features.get("score_5m", 0.0)),
        "direction_score_15m": float(features.get("score_15m", 0.0)),
        "above_vwap": 1.0 if features.get("above_vwap") else 0.0,
        "support_bounce": 1.0 if features.get("support_bounce") else 0.0,
        "price_vs_vwap": float(
            features["price"] / max(f(features["row_1m"].get("vwap"), features["price"]), 1e-9) - 1.0
        ),
        "price_vs_orb_high": float(
            features["price"] / max(float(features.get("orb_high") or features["price"]), 1e-9) - 1.0
        ),
        "price_vs_daily_resistance": float(
            features["price"] / max(float(features.get("resistance") or features["price"]), 1e-9) - 1.0
        ),
        "rel_volume_1m": f(features["row_1m"].get("rel_volume"), 1.0),
        "rsi_1m": f(features["row_1m"].get("rsi14"), 50.0) / 100.0,
        "macd_hist_1m": f(features["row_1m"].get("macd_hist"), 0.0),
        "minute_from_open": session_minutes_from_open() / 390.0,
    }


def load_max_return_model() -> dict[str, Any]:
    model = read_json(MAX_RETURN_MODEL_PATH, {})
    if not isinstance(model, dict):
        return {}
    return model


def max_return_score(features: dict[str, Any]) -> float:
    """
    Lightweight online ranker for the one trade of the day.

    If a learned model exists, use learned feature weights from prior labeled
    sessions. Otherwise fall back to deterministic price-action quality.
    """
    payload = feature_payload_for_learning(features)
    model = load_max_return_model()
    weights = model.get("weights", {}) if isinstance(model, dict) else {}

    if weights:
        raw = 0.0
        denom = 0.0
        for key, value in payload.items():
            w = f(weights.get(key), 0.0)
            raw += w * f(value)
            denom += abs(w)
        if denom > 0:
            return float(np.clip(0.5 + raw / max(denom, 1e-9), 0.0, 1.0))

    # Fallback score before enough labels exist.
    return float(
        np.clip(
            0.34 * payload["setup_score"]
            + 0.22 * max(payload["breakout_score"], payload["reclaim_score"])
            + 0.16 * payload["direction_score_1m"]
            + 0.14 * payload["direction_score_5m"]
            + 0.10 * payload["direction_score_15m"]
            + 0.04 * min(max(payload["rel_volume_1m"] / 2.0, 0.0), 1.0),
            0.0,
            1.0,
        )
    )


def log_spx_candidate(features: dict[str, Any], action: str, setup_type: str, score: float, reason: str) -> None:
    payload = feature_payload_for_learning(features)
    row = {
        "timestamp_utc": now_utc_iso(),
        "date_et": datetime.now(ET).date().isoformat(),
        "symbol": TRADE_SYMBOL,
        "action": action,
        "setup_type": setup_type,
        "price": float(features["price"]),
        "max_return_score": float(score),
        "reason": reason,
        **payload,
    }
    append_candidate_row(row)


def update_max_return_labels_from_candidates() -> None:
    """
    Labels candidates from the current candidate log using future intraday return.
    This creates a simple supervised target: the candidate that led to the
    highest future return in the session is the one-trade oracle.
    """
    if not LEARN_MAX_RETURN_TRADE or not CANDIDATE_LOG_PATH.exists():
        return

    try:
        cands = pd.read_csv(CANDIDATE_LOG_PATH)
    except Exception:
        return

    if cands.empty or "timestamp_utc" not in cands.columns:
        return

    try:
        bars = add_indicators(fetch_1m_bars(TRADE_SYMBOL))
    except Exception:
        return

    if bars.empty:
        return

    bars = bars[["timestamp", "close", "high", "low"]].copy()
    bars["timestamp_utc"] = pd.to_datetime(bars["timestamp"], utc=True)

    rows = []
    for _, cand in cands.iterrows():
        ts = pd.to_datetime(cand.get("timestamp_utc"), utc=True, errors="coerce")
        if pd.isna(ts):
            continue

        after = bars[bars["timestamp_utc"] >= ts].head(MAX_RETURN_LOOKAHEAD_BARS + 1).copy()
        if len(after) < 3:
            continue

        entry = float(cand.get("price", after["close"].iloc[0]))
        future_high = float(after["high"].max())
        future_low = float(after["low"].min())
        future_close = float(after["close"].iloc[-1])

        future_max_return = future_high / max(entry, 1e-9) - 1.0
        future_min_return = future_low / max(entry, 1e-9) - 1.0
        future_close_return = future_close / max(entry, 1e-9) - 1.0

        out = cand.to_dict()
        out["future_max_return"] = future_max_return
        out["future_min_return"] = future_min_return
        out["future_close_return"] = future_close_return
        out["is_profitable_candidate"] = bool(future_max_return >= MAX_RETURN_MIN_FUTURE_RETURN)
        rows.append(out)

    if not rows:
        return

    labels = pd.DataFrame(rows)
    labels["oracle_rank"] = labels.groupby("date_et")["future_max_return"].rank(ascending=False, method="first")
    labels["is_oracle_trade"] = labels["oracle_rank"] == 1.0
    labels.to_csv(SESSION_LABELS_PATH, index=False)

    train_max_return_model(labels)


def train_max_return_model(labels: pd.DataFrame) -> None:
    """
    Very small transparent online learner:
    compare average features for oracle candidates vs non-oracle candidates and
    store the difference as weights. This avoids adding new dependencies while
    still letting the layer learn from its own sessions.
    """
    feature_cols = [
        "setup_score",
        "breakout_score",
        "reclaim_score",
        "direction_score_1m",
        "direction_score_5m",
        "direction_score_15m",
        "above_vwap",
        "support_bounce",
        "price_vs_vwap",
        "price_vs_orb_high",
        "price_vs_daily_resistance",
        "rel_volume_1m",
        "rsi_1m",
        "macd_hist_1m",
        "minute_from_open",
    ]

    usable = labels.dropna(subset=["is_oracle_trade"]).copy()
    if usable.empty:
        return

    pos = usable[usable["is_oracle_trade"] == True]
    neg = usable[usable["is_oracle_trade"] == False]
    if pos.empty or neg.empty:
        return

    weights: dict[str, float] = {}
    for col in feature_cols:
        if col not in usable.columns:
            continue
        p = pd.to_numeric(pos[col], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()
        n = pd.to_numeric(neg[col], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()
        if pd.isna(p) or pd.isna(n):
            continue
        weights[col] = float(p - n)

    model = {
        "updated_at_utc": now_utc_iso(),
        "symbol": TRADE_SYMBOL,
        "label_rows": int(len(usable)),
        "oracle_rows": int(len(pos)),
        "weights": weights,
        "target": "one_trade_per_session_max_future_return",
        "lookahead_bars": int(MAX_RETURN_LOOKAHEAD_BARS),
    }
    write_json(MAX_RETURN_MODEL_PATH, model)
    write_event("spx_max_return_model_updated", model)


def one_trade_entry_allowed(score: float) -> tuple[bool, str]:
    if not ONE_TRADE_PER_SESSION:
        return True, "one_trade_rule_disabled"

    if session_trade_used():
        return False, "one_trade_already_used_today"

    minute = session_minutes_from_open()

    if minute < EARLIEST_ENTRY_MINUTE and score < EARLY_ENTRY_EXCEPTION_SCORE:
        return False, f"waiting_for_better_setup:minute={minute:.1f}:score={score:.2f}"

    if score >= MAX_RETURN_MIN_SCORE:
        return True, f"max_return_score_ok:{score:.2f}"

    if minute >= LAST_ENTRY_MINUTE and score >= max(0.60, MAX_RETURN_MIN_SCORE - 0.08):
        return True, f"late_session_best_available:{score:.2f}"

    return False, f"max_return_score_low:{score:.2f}"



# =============================================================================
# RL multibar-gain agent
# =============================================================================

RL_FEATURE_COLS = [
    # 1m returns / structure
    "ret_1", "ret_3", "ret_5", "ret_10",
    "dist_sma5", "dist_sma20", "vwap_dist", "rolling_high_dist", "rolling_low_dist",
    "score_1m",

    # 1m RSI / SMA / ATR / MACD
    "rsi14",
    "sma5_above_sma20",
    "sma5_cross_up_sma20",
    "sma20_below_sma5",
    "sma5_slope",
    "sma20_slope",
    "atr14",
    "atr_pct",
    "macd",
    "macd_signal",
    "macd_hist",
    "macd_hist_slope",

    # Volume / early continuation
    "rel_volume",
    "early_multibar_score",

    # 5m
    "score_5m", "ret_3_5m", "rsi14_5m",
    "sma5_above_sma20_5m", "sma5_cross_up_sma20_5m", "sma20_below_sma5_5m",
    "atr14_5m", "atr_pct_5m", "macd_5m", "macd_signal_5m", "macd_hist_5m", "macd_hist_slope_5m",

    # 15m
    "score_15m", "ret_3_15m", "rsi14_15m",
    "sma5_above_sma20_15m", "sma5_cross_up_sma20_15m", "sma20_below_sma5_15m",
    "atr14_15m", "atr_pct_15m", "macd_15m", "macd_signal_15m", "macd_hist_15m", "macd_hist_slope_15m",

    # 1h higher-timeframe context
    "score_1h", "ret_3_1h", "rsi14_1h", "dist_sma20_1h",
    "sma5_above_sma20_1h", "sma5_cross_up_sma20_1h", "sma20_below_sma5_1h",
    "atr14_1h", "atr_pct_1h", "macd_1h", "macd_signal_1h", "macd_hist_1h", "macd_hist_slope_1h",

    # 4h higher-timeframe context
    "score_4h", "ret_3_4h", "rsi14_4h", "dist_sma20_4h",
    "sma5_above_sma20_4h", "sma5_cross_up_sma20_4h", "sma20_below_sma5_4h",
    "atr14_4h", "atr_pct_4h", "macd_4h", "macd_signal_4h", "macd_hist_4h", "macd_hist_slope_4h",
]


def _reverse_rolling_max(series: pd.Series, window: int) -> pd.Series:
    return series.iloc[::-1].rolling(window, min_periods=1).max().iloc[::-1]


def _reverse_rolling_min(series: pd.Series, window: int) -> pd.Series:
    return series.iloc[::-1].rolling(window, min_periods=1).min().iloc[::-1]


def _merge_timeframe_context(base: pd.DataFrame, source: pd.DataFrame, suffix: str) -> pd.DataFrame:
    if source.empty:
        base[f"score_{suffix}"] = base.get("score_1m", 0.0)
        base[f"ret_3_{suffix}"] = 0.0
        base[f"rsi14_{suffix}"] = 50.0
        base[f"dist_sma20_{suffix}"] = 0.0
        base[f"sma5_above_sma20_{suffix}"] = 0.0
        base[f"sma5_cross_up_sma20_{suffix}"] = 0.0
        base[f"sma20_below_sma5_{suffix}"] = 0.0
        base[f"atr14_{suffix}"] = 0.0
        base[f"atr_pct_{suffix}"] = 0.0
        base[f"macd_{suffix}"] = 0.0
        base[f"macd_signal_{suffix}"] = 0.0
        base[f"macd_hist_{suffix}"] = 0.0
        base[f"macd_hist_slope_{suffix}"] = 0.0
        return base

    tf = source.copy()
    tf[f"score_{suffix}"] = tf.apply(direction_score, axis=1)
    keep = [
        "timestamp",
        f"score_{suffix}",
        "ret_3",
        "rsi14",
        "dist_sma20",
        "sma5_above_sma20",
        "sma5_cross_up_sma20",
        "sma20_below_sma5",
        "atr14",
        "atr_pct",
        "macd",
        "macd_signal",
        "macd_hist",
        "macd_hist_slope",
    ]
    tf = tf[keep].rename(
        columns={
            "ret_3": f"ret_3_{suffix}",
            "rsi14": f"rsi14_{suffix}",
            "dist_sma20": f"dist_sma20_{suffix}",
            "sma5_above_sma20": f"sma5_above_sma20_{suffix}",
            "sma5_cross_up_sma20": f"sma5_cross_up_sma20_{suffix}",
            "sma20_below_sma5": f"sma20_below_sma5_{suffix}",
            "atr14": f"atr14_{suffix}",
            "atr_pct": f"atr_pct_{suffix}",
            "macd": f"macd_{suffix}",
            "macd_signal": f"macd_signal_{suffix}",
            "macd_hist": f"macd_hist_{suffix}",
            "macd_hist_slope": f"macd_hist_slope_{suffix}",
        }
    )
    return pd.merge_asof(
        base.sort_values("timestamp"),
        tf.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )


def add_rl_multibar_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Build causal RL features from 1m bars plus 5m/15m/1h/4h context.

    Future columns are used only during training rewards, never in live observation.
    """
    if df_1m.empty:
        return pd.DataFrame()

    bars = add_indicators(df_1m).copy()
    if bars.empty:
        return pd.DataFrame()

    bars["score_1m"] = bars.apply(direction_score, axis=1)

    tf_5m = add_indicators(resample_bars(bars, "5min"))
    tf_15m = add_indicators(resample_bars(bars, "15min"))
    tf_1h = add_indicators(resample_bars(bars, "1h"))
    tf_4h = add_indicators(resample_bars(bars, "4h"))

    bars = _merge_timeframe_context(bars, tf_5m, "5m")
    bars = _merge_timeframe_context(bars, tf_15m, "15m")
    bars = _merge_timeframe_context(bars, tf_1h, "1h")
    bars = _merge_timeframe_context(bars, tf_4h, "4h")

    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)

    bars["vwap_dist"] = close / bars["vwap"].replace(0, np.nan) - 1.0
    bars["rolling_high_dist"] = close / bars["rolling_high_20"].replace(0, np.nan) - 1.0
    bars["rolling_low_dist"] = close / bars["rolling_low_20"].replace(0, np.nan) - 1.0

    ret_3bar = close / close.shift(3) - 1.0
    ret_5bar = close / close.shift(5) - 1.0
    ret_10bar = close / close.shift(10) - 1.0
    vol_accel = bars["volume"].rolling(3).mean() / bars["volume"].rolling(20).mean().replace(0, np.nan)
    macd_up = (
        (bars["macd_hist"] > bars["macd_hist"].shift(1))
        & (bars["macd_hist"].shift(1) > bars["macd_hist"].shift(2))
    )

    # Higher timeframe agreement matters: it keeps the agent from entering
    # 1m noise against 1h/4h trend.
    htf_ok = (
        (bars["score_1h"] >= 0.50).astype(float)
        + (bars["score_4h"] >= 0.50).astype(float)
        + (bars["macd_hist_1h"] >= 0).astype(float)
        + (bars["macd_hist_4h"] >= 0).astype(float)
    ) / 4.0

    bars["early_multibar_score"] = (
        (ret_3bar > 0.0012).astype(float)
        + (ret_5bar > 0.0018).astype(float)
        + (ret_10bar < 0.018).astype(float)
        + (vol_accel > 1.05).astype(float)
        + macd_up.astype(float)
        + (bars["macd_hist"] > 0).astype(float)
        + (bars["macd_hist_slope"] > 0).astype(float)
        + (bars["sma5_above_sma20"] > 0).astype(float)
        + (bars["sma5_cross_up_sma20"] > 0).astype(float)
        + (bars["vwap_dist"] > 0).astype(float)
        + (bars["score_5m"] >= 0.55).astype(float)
        + (bars["score_15m"] >= 0.50).astype(float)
        + (htf_ok >= 0.50).astype(float)
    ) / 13.0

    for horizon in [1, 3, 5, 10, 15, 20, 30, 45, 60]:
        bars[f"future_ret_{horizon}"] = close.shift(-horizon) / close - 1.0

    for horizon in [10, 20, 30, 45, 60]:
        bars[f"future_max_ret_{horizon}"] = _reverse_rolling_max(high.shift(-1), horizon) / close - 1.0
        bars[f"future_min_ret_{horizon}"] = _reverse_rolling_min(low.shift(-1), horizon) / close - 1.0

    bars = bars.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return bars


class SPXMultibarGainEnv(gym.Env if gym is not None else object):
    """
    RL environment for intraday multibar gain entries/exits.

    Actions:
        0 HOLD
        1 ENTER_LONG
        2 EXIT_LONG
    """

    metadata = {"render_modes": []}

    def __init__(self, features: pd.DataFrame):
        if gym is None or spaces is None:
            raise RuntimeError("gymnasium is required")

        super().__init__()
        self.df = features.reset_index(drop=True).copy()
        self.feature_cols = [c for c in RL_FEATURE_COLS if c in self.df.columns]
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(len(self.feature_cols) + 6,),
            dtype=np.float32,
        )
        self.idx = 0
        self.start_idx = 0
        self.end_idx = 0
        self.position = 0
        self.entry_price = 0.0
        self.bars_held = 0
        self.mfe = 0.0
        self.mae = 0.0
        self.realized_pnl = 0.0
        self.trades: list[dict[str, Any]] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        max_start = max(1, len(self.df) - RL_MAX_EPISODE_BARS - 2)
        self.start_idx = int(self.np_random.integers(0, max_start))
        self.idx = self.start_idx
        self.end_idx = min(len(self.df) - 2, self.start_idx + RL_MAX_EPISODE_BARS)
        self.position = 0
        self.entry_price = 0.0
        self.bars_held = 0
        self.mfe = 0.0
        self.mae = 0.0
        self.realized_pnl = 0.0
        self.trades = []
        return self._obs(), {}

    def step(self, action: int):
        row = self.df.iloc[self.idx]
        next_row = self.df.iloc[self.idx + 1]
        price = max(f(row.get("close")), 1e-9)
        next_price = max(f(next_row.get("close")), 1e-9)
        reward = 0.0

        if self.position == 1:
            self.bars_held += 1
            unrealized = price / self.entry_price - 1.0
            self.mfe = max(self.mfe, unrealized)
            self.mae = min(self.mae, unrealized)
            reward += (next_price / price - 1.0) * RL_REWARD_SCALE
            if unrealized > 0:
                reward += min(unrealized * RL_HOLD_WINNER_MULT, 3.0)
            if f(row.get("early_multibar_score")) >= 0.65 and f(row.get("future_ret_5")) > 0:
                reward += min(f(row.get("future_ret_5")) * RL_EARLY_MULTIBAR_BONUS_MULT, 3.0)

        if action == 1:
            reward += self._enter(row, price)
        elif action == 2:
            reward += self._exit(row, price, reason="agent_exit")
        else:
            reward += self._hold(row)

        self.idx += 1
        terminated = self.idx >= self.end_idx
        truncated = False

        if terminated and self.position == 1:
            final_price = max(f(self.df.iloc[self.idx].get("close")), 1e-9)
            reward += self._exit(self.df.iloc[self.idx], final_price, reason="episode_end")

        return self._obs(), float(reward), terminated, truncated, {}

    def _enter(self, row: pd.Series, price: float) -> float:
        if self.position == 1:
            return -0.25

        future_upside = max(f(row.get("future_max_ret_20")), f(row.get("future_max_ret_30")), 0.0)
        future_downside = min(f(row.get("future_min_ret_20")), f(row.get("future_min_ret_30")), 0.0)

        mtf_alignment = np.mean([
            f(row.get("score_1m")) >= 0.55,
            f(row.get("score_5m")) >= 0.55,
            f(row.get("score_15m")) >= 0.50,
            f(row.get("score_1h")) >= 0.50,
            f(row.get("score_4h")) >= 0.50,
            f(row.get("vwap_dist")) > 0,
        ])

        reward = RL_ENTRY_BONUS * float(mtf_alignment)
        reward += min(max(future_upside, 0.0) * RL_EARLY_MULTIBAR_BONUS_MULT, 5.0)
        reward += f(row.get("early_multibar_score")) * 1.5

        if future_upside < 0.002 or future_downside < -0.006:
            reward -= RL_BAD_ENTRY_PENALTY

        self.position = 1
        self.entry_price = price
        self.bars_held = 0
        self.mfe = 0.0
        self.mae = 0.0
        self.trades.append({
            "entry_idx": int(self.idx),
            "entry_time": str(row.get("timestamp")),
            "entry_price": float(price),
            "entry_early_multibar_score": float(f(row.get("early_multibar_score"))),
            "entry_score_1h": float(f(row.get("score_1h"))),
            "entry_score_4h": float(f(row.get("score_4h"))),
        })
        return float(reward)

    def _exit(self, row: pd.Series, price: float, reason: str) -> float:
        if self.position == 0 or self.entry_price <= 0:
            return -0.10

        pnl = price / self.entry_price - 1.0
        capture = pnl / max(self.mfe, 1e-6) if pnl > 0 and self.mfe > 0 else 0.0
        giveback = max(0.0, self.mfe - max(pnl, 0.0))

        reward = pnl * RL_REWARD_SCALE
        reward += min(max(capture, 0.0) * RL_MFE_CAPTURE_MULT, 5.0)
        reward -= min(giveback * RL_GIVEBACK_PENALTY_MULT, 5.0)
        reward -= min(abs(self.mae) * RL_MAE_PENALTY_MULT, 5.0)

        if pnl > 0 and self.bars_held < RL_MIN_QUALITY_HOLD_BARS:
            reward -= 1.0
        if pnl > 0 and self.bars_held >= RL_MIN_QUALITY_HOLD_BARS:
            reward += min(self.bars_held / 10.0, 2.0)

        if self.trades:
            self.trades[-1].update({
                "exit_idx": int(self.idx),
                "exit_time": str(row.get("timestamp")),
                "exit_price": float(price),
                "pnl_pct": float(pnl),
                "mfe_pct": float(self.mfe),
                "mae_pct": float(self.mae),
                "capture_ratio": float(capture),
                "bars_held": int(self.bars_held),
                "reason": reason,
            })

        self.realized_pnl += pnl
        self.position = 0
        self.entry_price = 0.0
        self.bars_held = 0
        self.mfe = 0.0
        self.mae = 0.0
        return float(reward)

    def _hold(self, row: pd.Series) -> float:
        if self.position == 0:
            if (
                f(row.get("early_multibar_score")) >= 0.75
                and f(row.get("future_ret_5")) > 0.002
                and f(row.get("score_1h")) >= 0.50
                and f(row.get("score_4h")) >= 0.50
            ):
                return -0.40
            return -RL_FLAT_HOLD_PENALTY
        return 0.0

    def _obs(self) -> np.ndarray:
        row = self.df.iloc[self.idx]
        x = row[self.feature_cols].astype(float).to_numpy(dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = np.clip(x, -10.0, 10.0)

        price = max(f(row.get("close")), 1e-9)
        unrealized = 0.0
        giveback = 0.0
        if self.position == 1 and self.entry_price > 0:
            unrealized = price / self.entry_price - 1.0
            giveback = max(0.0, self.mfe - max(unrealized, 0.0))

        pos = np.array([
            float(self.position),
            float(unrealized * 100.0),
            float(min(self.bars_held / 30.0, 10.0)),
            float(self.mfe * 100.0),
            float(self.mae * 100.0),
            float(giveback * 100.0),
        ], dtype=np.float32)

        return np.concatenate([x, pos]).astype(np.float32)


def build_rl_training_dataset() -> pd.DataFrame:
    df = fetch_bars(TRADE_SYMBOL, TimeFrame.Minute, RL_TRAIN_LOOKBACK_DAYS)
    features = add_rl_multibar_features(df)
    if not features.empty:
        features.to_csv(RL_DATASET_PATH, index=False)
    return features


def train_rl_multibar_agent() -> dict[str, Any]:
    if not RL_ENABLED:
        return {"ok": False, "reason": "rl_disabled"}
    if PPO is None or DummyVecEnv is None or gym is None:
        return {"ok": False, "reason": "stable_baselines3_or_gymnasium_missing"}

    features = build_rl_training_dataset()
    if features.empty or len(features) < 500:
        return {"ok": False, "reason": f"not_enough_training_rows:{len(features)}"}

    raw_env = SPXMultibarGainEnv(features)
    env = DummyVecEnv([lambda: raw_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=RL_LEARNING_RATE,
        gamma=RL_GAMMA,
        n_steps=RL_N_STEPS,
        batch_size=RL_BATCH_SIZE,
        ent_coef=RL_ENT_COEF,
        gae_lambda=0.95,
        clip_range=0.20,
    )
    model.learn(total_timesteps=RL_TIMESTEPS)
    model.save(str(RL_MODEL_PATH))

    trades = pd.DataFrame(raw_env.trades)
    if not trades.empty:
        pnl = pd.to_numeric(trades.get("pnl_pct"), errors="coerce").fillna(0.0)
        win_rate = float((pnl > 0).mean())
        avg_pnl = float(pnl.mean())
        avg_capture = float(pd.to_numeric(trades.get("capture_ratio"), errors="coerce").fillna(0.0).mean())
        avg_hold = float(pd.to_numeric(trades.get("bars_held"), errors="coerce").fillna(0.0).mean())
    else:
        win_rate = avg_pnl = avg_capture = avg_hold = 0.0

    metrics = {
        "ok": True,
        "created_at_utc": now_utc_iso(),
        "symbol": TRADE_SYMBOL,
        "model_path": str(RL_MODEL_PATH),
        "dataset_path": str(RL_DATASET_PATH),
        "rows": int(len(features)),
        "timesteps": int(RL_TIMESTEPS),
        "trade_count": int(len(trades)),
        "win_rate": win_rate,
        "avg_pnl_pct": avg_pnl,
        "avg_capture_ratio": avg_capture,
        "avg_hold_bars": avg_hold,
        "features_include": [
            "1m", "5m", "15m", "1h", "4h",
            "RSI", "SMA5_cross_up_SMA20", "SMA20_below_SMA5",
            "ATR", "ATR_pct", "MACD", "MACD_signal", "MACD_hist", "MACD_hist_slope",
        ],
        "tuned_params": SPX_RL_TUNED_PARAMS,
        "tuned_study_value": SPX_RL_TUNED_STUDY_VALUE,
        "tuned_data_tag": SPX_RL_TUNED_DATA_TAG,
        "actions": {0: "HOLD", 1: "ENTER_LONG", 2: "EXIT_LONG"},
    }
    write_json(RL_METRICS_PATH, metrics)
    write_event("spx_rl_multibar_agent_trained", metrics)
    return metrics


def load_rl_model() -> Any:
    if not RL_ENABLED or PPO is None or not RL_MODEL_PATH.exists():
        return None
    try:
        return PPO.load(str(RL_MODEL_PATH))
    except Exception as exc:
        write_event("spx_rl_model_load_failed", {"error": str(exc), "model_path": str(RL_MODEL_PATH)})
        return None


def rl_latest_observation() -> tuple[np.ndarray | None, pd.DataFrame]:
    bars_1m = add_indicators(fetch_1m_bars(TRADE_SYMBOL))
    table = add_rl_multibar_features(bars_1m)
    if table.empty or gym is None:
        return None, table

    env = SPXMultibarGainEnv(table)
    env.idx = len(table) - 1
    return env._obs(), table


def rl_multibar_signal(features: dict[str, Any], has_position: bool) -> dict[str, Any]:
    if not RL_ENABLED:
        return {"enabled": False, "action": "DISABLED", "allowed_entry": True, "allowed_exit": True, "confidence": 0.0}

    model = load_rl_model()
    if model is None:
        return {
            "enabled": True,
            "model_exists": False,
            "action": "NO_MODEL",
            "allowed_entry": not RL_REQUIRE_FOR_ENTRY,
            "allowed_exit": not RL_REQUIRE_FOR_EXIT,
            "confidence": 0.0,
            "reason": "rl_model_missing",
        }

    obs, table = rl_latest_observation()
    if obs is None or table.empty:
        return {
            "enabled": True,
            "model_exists": True,
            "action": "NO_OBS",
            "allowed_entry": not RL_REQUIRE_FOR_ENTRY,
            "allowed_exit": not RL_REQUIRE_FOR_EXIT,
            "confidence": 0.0,
            "reason": "rl_observation_unavailable",
        }

    action_int, _ = model.predict(obs, deterministic=True)
    action_int = int(action_int)
    action_name = {0: "HOLD", 1: "ENTER_LONG", 2: "EXIT_LONG"}.get(action_int, "HOLD")
    latest = table.iloc[-1]

    entry_quality = float(np.clip(
        0.25 * f(latest.get("early_multibar_score"))
        + 0.15 * f(latest.get("score_1m"))
        + 0.15 * f(latest.get("score_5m"))
        + 0.15 * f(latest.get("score_15m"))
        + 0.15 * f(latest.get("score_1h"))
        + 0.10 * f(latest.get("score_4h"))
        + 0.05 * (1.0 if f(latest.get("vwap_dist")) > 0 else 0.0),
        0.0,
        1.0,
    ))

    exit_quality = float(np.clip(
        0.24 * (1.0 - f(latest.get("score_1m")))
        + 0.20 * (1.0 - f(latest.get("score_5m")))
        + 0.18 * (1.0 - f(latest.get("score_15m")))
        + 0.16 * (1.0 - f(latest.get("score_1h")))
        + 0.12 * (1.0 - f(latest.get("score_4h")))
        + 0.10 * (1.0 if f(latest.get("vwap_dist")) < 0 else 0.0),
        0.0,
        1.0,
    ))

    confidence = entry_quality if action_name == "ENTER_LONG" else exit_quality if action_name == "EXIT_LONG" else max(entry_quality, exit_quality) * 0.50

    allowed_entry = True
    allowed_exit = True
    if RL_REQUIRE_FOR_ENTRY:
        allowed_entry = action_name == "ENTER_LONG" and confidence >= RL_MIN_ENTRY_CONFIDENCE
    if RL_REQUIRE_FOR_EXIT:
        allowed_exit = action_name == "EXIT_LONG" and confidence >= RL_MIN_EXIT_CONFIDENCE

    payload = {
        "enabled": True,
        "model_exists": True,
        "action": action_name,
        "action_id": action_int,
        "confidence": float(confidence),
        "entry_quality": entry_quality,
        "exit_quality": exit_quality,
        "allowed_entry": bool(allowed_entry),
        "allowed_exit": bool(allowed_exit),
        "early_multibar_score": f(latest.get("early_multibar_score")),
        "score_1m": f(latest.get("score_1m")),
        "score_5m": f(latest.get("score_5m")),
        "score_15m": f(latest.get("score_15m")),
        "score_1h": f(latest.get("score_1h")),
        "score_4h": f(latest.get("score_4h")),
        "vwap_dist": f(latest.get("vwap_dist")),
        "rsi14": f(latest.get("rsi14")),
        "sma5_above_sma20": f(latest.get("sma5_above_sma20")),
        "sma5_cross_up_sma20": f(latest.get("sma5_cross_up_sma20")),
        "sma20_below_sma5": f(latest.get("sma20_below_sma5")),
        "atr14": f(latest.get("atr14")),
        "atr_pct": f(latest.get("atr_pct")),
        "macd": f(latest.get("macd")),
        "macd_signal": f(latest.get("macd_signal")),
        "macd_hist": f(latest.get("macd_hist")),
        "macd_hist_slope": f(latest.get("macd_hist_slope")),
        "reason": f"rl_action:{action_name}:confidence={confidence:.2f}",
    }

    write_json(RL_SIGNAL_PATH, payload)
    write_event("spx_rl_multibar_signal", payload)
    return payload


def get_position(client: TradingClient, symbol: str) -> SPXPositionState:
    try:
        p = client.get_open_position(symbol)
        qty = f(getattr(p, "qty", 0.0))
        avg = f(getattr(p, "avg_entry_price", 0.0))
        current = f(getattr(p, "current_price", avg))
        state = read_json(STATE_PATH, {})
        pstate = state.get("position_state", {})
        high_water = max(f(pstate.get("high_water_price"), current), current)
        return SPXPositionState(
            symbol=symbol,
            qty=qty,
            avg_entry_price=avg,
            high_water_price=high_water,
            entry_time_utc=str(pstate.get("entry_time_utc", "")),
            adds_today=i(pstate.get("adds_today", 0)),
        )
    except Exception:
        return SPXPositionState(symbol=symbol)


def update_position_state(pstate: SPXPositionState) -> None:
    state = read_json(STATE_PATH, {})
    state["date_et"] = datetime.now(ET).date().isoformat()
    state["position_state"] = asdict(pstate)
    write_json(STATE_PATH, state)


def clear_position_state() -> None:
    state = read_json(STATE_PATH, {})
    state["position_state"] = asdict(SPXPositionState(symbol=TRADE_SYMBOL))
    write_json(STATE_PATH, state)


def rl_plan_allows_entry() -> tuple[bool, str]:
    if not USE_RL_PLAN_CONFIRMATION:
        return True, "rl_plan_disabled"

    plan = read_json(SPX_RL_PLAN_PATH, {})
    if not plan:
        return (not REQUIRE_RL_FOR_ENTRY), "rl_plan_missing"

    action = str(plan.get("model_action", "")).upper()
    confidence = f(plan.get("model_confidence_proxy"), 0.0)

    if action in {"LONG_BREAKOUT", "ADD_LONG"} and confidence >= 0.55:
        return True, f"rl_plan_confirms:{action}:conf={confidence:.2f}"

    if REQUIRE_RL_FOR_ENTRY:
        return False, f"rl_plan_blocks:{action}:conf={confidence:.2f}"

    return True, f"rl_plan_not_confirming_but_not_required:{action}:conf={confidence:.2f}"


def build_signal(client: TradingClient) -> SPXSignal:
    reset_session_if_new_day()
    update_max_return_labels_from_candidates()

    features = latest_features()
    price = features["price"]
    pstate = get_position(client, TRADE_SYMBOL)

    stop_loss = None
    take_profit = None
    qty = 0
    notional = 0.0
    action = "WAIT"
    setup_type = "none"
    reason = "no_setup"

    account = account_snapshot(client)
    equity = max(account["equity"], 1.0)
    max_position_dollars = min(MAX_POSITION_DOLLARS, equity * MAX_POSITION_WEIGHT)
    base_notional = min(BASE_ORDER_DOLLARS, max_position_dollars)

    has_position = pstate.qty > 0 and pstate.avg_entry_price > 0
    rl_signal = rl_multibar_signal(features, has_position=has_position)

    if has_position:
        pstate.high_water_price = max(pstate.high_water_price, price)
        update_position_state(pstate)

        pnl_pct = price / pstate.avg_entry_price - 1.0
        high_water_profit = pstate.high_water_price / max(pstate.avg_entry_price, 1e-9) - 1.0
        giveback = high_water_profit - max(pnl_pct, 0.0)

        lost_vwap = price < f(features["row_1m"].get("vwap"), price)
        lost_orb = features["orb_low"] > 0 and price < features["orb_low"]
        direction_weak = (
            features["score_1m"] < 0.40
            or features["score_5m"] < 0.45
            or features["score_15m"] < 0.45
        )

        hard_stop = pnl_pct <= -abs(STOP_LOSS_PCT)
        trail_stop = pnl_pct >= MIN_TRAIL_PROFIT_PCT and giveback >= TRAILING_GIVEBACK_PCT
        rl_exit = bool(
            rl_signal.get("action") == "EXIT_LONG"
            and f(rl_signal.get("confidence")) >= RL_MIN_EXIT_CONFIDENCE
        )
        soft_exit = trail_stop or (lost_vwap and direction_weak) or lost_orb or rl_exit
        rl_allows_exit = bool(rl_signal.get("allowed_exit", True))

        if hard_stop or (soft_exit and rl_allows_exit):
            action = "SELL"
            setup_type = "risk_exit"
            qty = int(abs(pstate.qty))
            notional = qty * price
            reason = (
                f"exit:pnl={pnl_pct:.2%}:giveback={giveback:.2%}:"
                f"hard_stop={hard_stop}:trail={trail_stop}:lost_vwap={lost_vwap}:"
                f"direction_weak={direction_weak}:lost_orb={lost_orb}:"
                f"rl={rl_signal.get('action')}:{f(rl_signal.get('confidence')):.2f}"
            )
        else:
            action = "HOLD"
            setup_type = "hold_winner" if pnl_pct > 0 else "hold_position"
            reason = f"position_ok:pnl={pnl_pct:.2%}:high_water={pstate.high_water_price:.2f}"

    else:
        market_ok, market_reason = market_context_allows_entry()
        rl_ok, rl_reason = rl_plan_allows_entry()
        mr_score = max_return_score(features)
        one_trade_ok, one_trade_reason = one_trade_entry_allowed(mr_score)
        rl_entry_ok = bool(rl_signal.get("allowed_entry", True))
        rl_entry_reason = str(rl_signal.get("reason", "rl_disabled_or_not_required"))

        if account["buying_power"] < MIN_BUYING_POWER:
            reason = f"buying_power_too_low:{account['buying_power']:.2f}"
        elif not market_ok:
            reason = market_reason
        elif not rl_ok:
            reason = rl_reason
        elif not one_trade_ok:
            reason = one_trade_reason
        elif not rl_entry_ok:
            reason = rl_entry_reason
        elif features["breakout_score"] >= MIN_BREAKOUT_SCORE:
            action = "BUY"
            setup_type = "spx_breakout"
            notional = base_notional * min(1.25, 0.75 + features["breakout_score"])
            qty = int(notional // price)
            reason = (
                f"breakout_confirmed:breakout_score={features['breakout_score']:.2f}:"
                f"1m={features['score_1m']:.2f}:5m={features['score_5m']:.2f}:"
                f"15m={features['score_15m']:.2f}:max_return={mr_score:.2f}:"
                f"{market_reason}:{rl_reason}:{one_trade_reason}:{rl_entry_reason}"
            )
        elif features["reclaim_score"] >= MIN_RECLAIM_SCORE:
            action = "BUY"
            setup_type = "vwap_reclaim"
            notional = base_notional
            qty = int(notional // price)
            reason = (
                f"vwap_reclaim_confirmed:reclaim_score={features['reclaim_score']:.2f}:"
                f"1m={features['score_1m']:.2f}:5m={features['score_5m']:.2f}:"
                f"15m={features['score_15m']:.2f}:max_return={mr_score:.2f}:"
                f"{market_reason}:{rl_reason}:{one_trade_reason}:{rl_entry_reason}"
            )
        elif features["support_bounce"] and features["setup_score"] >= MIN_SETUP_SCORE:
            action = "BUY"
            setup_type = "support_bounce"
            notional = base_notional * 0.75
            qty = int(notional // price)
            reason = (
                f"support_bounce_confirmed:setup_score={features['setup_score']:.2f}:"
                f"support={features['support']:.2f}:max_return={mr_score:.2f}:"
                f"{market_reason}:{rl_reason}:{one_trade_reason}:{rl_entry_reason}"
            )

        if action == "BUY":
            stop_loss = price * (1.0 - STOP_LOSS_PCT)
            risk = price - stop_loss
            take_profit = price + risk * TAKE_PROFIT_R
            notional = qty * price

            if qty <= 0 or notional < MIN_ORDER_DOLLARS:
                action = "WAIT"
                setup_type = "order_too_small"
                reason = f"order_too_small:qty={qty}:notional={notional:.2f}"

    try:
        log_spx_candidate(
            features=features,
            action=action,
            setup_type=setup_type,
            score=max_return_score(features),
            reason=reason,
        )
    except Exception as exc:
        write_event("spx_candidate_log_error", {"error": str(exc)})

    return SPXSignal(
        created_at_utc=now_utc_iso(),
        logical_symbol=LOGICAL_SYMBOL,
        trade_symbol=TRADE_SYMBOL,
        action=action,
        setup_type=setup_type,
        price=round(price, 4),
        setup_score=round(features["setup_score"], 4),
        breakout_score=round(features["breakout_score"], 4),
        reclaim_score=round(features["reclaim_score"], 4),
        direction_score_1m=round(features["score_1m"], 4),
        direction_score_5m=round(features["score_5m"], 4),
        direction_score_15m=round(features["score_15m"], 4),
        vwap=round(f(features["row_1m"].get("vwap"), price), 4),
        orb_high=round(features["orb_high"], 4),
        orb_low=round(features["orb_low"], 4),
        daily_support=round(features["support"], 4),
        daily_resistance=round(features["resistance"], 4),
        stop_loss=round(stop_loss, 4) if stop_loss else None,
        take_profit=round(take_profit, 4) if take_profit else None,
        qty=int(qty),
        notional=round(float(notional), 2),
        max_return_score=round(max_return_score(features), 4),
        session_trade_used=bool(session_trade_used()),
        reason=reason,
    )


def submit_signal_order(client: TradingClient, signal: SPXSignal) -> None:
    if signal.action not in {"BUY", "SELL"}:
        return

    if signal.qty <= 0:
        write_event("spx_order_skipped", {"reason": "qty_zero", "signal": asdict(signal)})
        return

    if not EXECUTE_ORDERS:
        print(f"[spx-layer] signal only execute_disabled action={signal.action} qty={signal.qty} {signal.trade_symbol}")
        write_event("spx_signal_only", asdict(signal))
        return

    side = OrderSide.BUY if signal.action == "BUY" else OrderSide.SELL

    req = MarketOrderRequest(
        symbol=signal.trade_symbol,
        qty=int(signal.qty),
        side=side,
        type=OrderType.MARKET,
        time_in_force=TIME_IN_FORCE,
    )

    try:
        order = client.submit_order(req)
        payload = {
            "submitted_at_utc": now_utc_iso(),
            "symbol": signal.trade_symbol,
            "side": signal.action.lower(),
            "qty": signal.qty,
            "notional_estimate": signal.notional,
            "price_reference": signal.price,
            "setup_type": signal.setup_type,
            "reason": signal.reason,
            "rl_signal_path": str(RL_SIGNAL_PATH),
            "order_id": getattr(order, "id", ""),
            "status": str(getattr(order, "status", "")),
        }
        append_order_row(payload)
        write_event("spx_order_submitted", payload)
        print(f"[spx-layer] submitted {signal.action} {signal.qty} {signal.trade_symbol} reason={signal.reason}")

        if signal.action == "BUY":
            mark_session_trade_used(signal)
            update_position_state(
                SPXPositionState(
                    symbol=signal.trade_symbol,
                    qty=signal.qty,
                    avg_entry_price=signal.price,
                    high_water_price=signal.price,
                    entry_time_utc=now_utc_iso(),
                    adds_today=0,
                )
            )
        else:
            clear_position_state()

    except APIError as exc:
        write_event("spx_order_api_error", {"error": str(exc), "signal": asdict(signal)})
        print(f"[spx-layer] order API error: {exc}")
    except Exception as exc:
        write_event("spx_order_error", {"error": str(exc), "signal": asdict(signal)})
        print(f"[spx-layer] order error: {exc}")


_STOP = False


def _handle_stop(signum: int, frame: Any) -> None:
    global _STOP
    _STOP = True
    print("[spx-layer] shutdown requested")


def run_once() -> SPXSignal | None:
    client = trading_client()
    pstate = get_position(client, TRADE_SYMBOL)
    allow_for_exit = pstate.qty > 0

    clock_ok, clock_reason = market_clock_ok(client, allow_for_exit=allow_for_exit)
    account = account_snapshot(client)
    loss_ok, loss_reason = daily_loss_ok(account)

    if not clock_ok:
        print(f"[spx-layer] skip {clock_reason}")
        write_event("spx_cycle_skipped", {"reason": clock_reason})
        return None

    if not loss_ok and not allow_for_exit:
        print(f"[spx-layer] halt {loss_reason}")
        write_event("spx_cycle_halted", {"reason": loss_reason})
        return None

    try:
        signal_obj = build_signal(client)
    except RuntimeError as exc:
        print(f"[spx-layer] build_signal skipped: {exc}")
        write_event("spx_cycle_skipped", {"reason": "bars_unavailable", "error": str(exc)})
        return None
    write_json(SIGNAL_PATH, asdict(signal_obj))
    write_event("spx_signal", asdict(signal_obj))

    print(
        "[spx-layer] "
        f"{signal_obj.action} {signal_obj.trade_symbol} "
        f"setup={signal_obj.setup_type} "
        f"price={signal_obj.price:.2f} "
        f"score={signal_obj.setup_score:.2f} "
        f"qty={signal_obj.qty} "
        f"reason={signal_obj.reason}"
    )

    submit_signal_order(client, signal_obj)
    return signal_obj


def main() -> None:
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    print("===== SPX PRICE ACTION LAYER =====")
    print(f"[config] logical_symbol={LOGICAL_SYMBOL}")
    print(f"[config] trade_symbol={TRADE_SYMBOL}")
    print(f"[config] paper={PAPER}")
    print(f"[config] execute_orders={EXECUTE_ORDERS}")
    print(f"[config] scan_seconds={SCAN_SECONDS}")
    print(f"[config] one_trade_per_session={ONE_TRADE_PER_SESSION}")
    print(f"[config] learn_max_return_trade={LEARN_MAX_RETURN_TRADE}")
    print(f"[config] max_return_min_score={MAX_RETURN_MIN_SCORE}")
    print(f"[paths] signal={SIGNAL_PATH}")
    print(f"[paths] candidate_log={CANDIDATE_LOG_PATH}")
    print(f"[paths] max_return_model={MAX_RETURN_MODEL_PATH}")
    print(f"[paths] events={EVENTS_PATH}")
    print(f"[rl] enabled={RL_ENABLED}")
    print(f"[rl] model={RL_MODEL_PATH}")
    print(f"[rl] dataset={RL_DATASET_PATH}")
    print("[rl] features=1m,5m,15m,1h,4h,RSI,SMA5/SMA20 cross,ATR,MACD")
    print(f"[rl] tuned_study_value={SPX_RL_TUNED_STUDY_VALUE}")

    if RL_TRAIN_ON_START or "--train-rl" in sys.argv:
        print("[rl] training SPX multibar-gain agent...")
        metrics = train_rl_multibar_agent()
        print(f"[rl] training_result={metrics}")

    once = "--once" in sys.argv

    while not _STOP:
        try:
            run_once()
        except Exception as exc:
            print(f"[spx-layer] cycle error: {exc}")
            write_event("spx_cycle_error", {"error": str(exc)})

        if once:
            break

        time.sleep(max(1, SCAN_SECONDS))


if __name__ == "__main__":
    main()
