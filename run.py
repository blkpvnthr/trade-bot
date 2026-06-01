#!/usr/bin/env python3
"""
alpaca_executor.py

Streaming-style paper executor inspired by the ProductionPaperTrader structure.

No dry-run mode.

Default behavior:
    python run.py

This executor is designed to be simple to reason about:
    - polls the existing daily_system_outputs order CSVs
    - checks Alpaca market clock
    - logs account equity curve
    - journals every attempted/placed/skipped trade
    - sells before buys
    - respects cash buffer, max exposure, max position, cooldown, daily loss kill switch
    - avoids duplicate open orders
    - supports market-hours-only execution
    - paper account by default; use --live only intentionally

Default input files:
    daily_system_outputs/daily_walkforward_emulated_rebalance_orders.csv
    daily_system_outputs/daily_walkforward_emulated_exit_orders.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
import re
from dotenv import load_dotenv

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

try:
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, OptionChainRequest
    from alpaca.data.timeframe import TimeFrame
except Exception:  # scanner/options can be disabled if alpaca-data is unavailable
    DataFeed = None
    StockHistoricalDataClient = None
    OptionHistoricalDataClient = None
    StockBarsRequest = None
    OptionChainRequest = None
    TimeFrame = None

try:
    import threading
except ImportError:
    pass

try:
    import msgpack as _msgpack_mod
except ImportError:
    _msgpack_mod = None

try:
    import websocket as _websocket_mod
except ImportError:
    _websocket_mod = None


# =============================================================================
# Paths / environment
# =============================================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

OUT_DIR = PROJECT_ROOT / "daily_system_outputs"
LOG_DIR = PROJECT_ROOT / "logs" / "execution"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_REBALANCE_ORDERS = OUT_DIR / "daily_walkforward_emulated_rebalance_orders.csv"
DEFAULT_EXIT_ORDERS = OUT_DIR / "daily_walkforward_emulated_exit_orders.csv"

JOURNAL_PATH = LOG_DIR / "executor_journal.csv"
EQUITY_CURVE_PATH = LOG_DIR / "executor_equity_curve.csv"
EVENTS_JSONL = LOG_DIR / "execution_events.jsonl"
KILL_SWITCH_PATH = LOG_DIR / "kill_switch_state.json"

# =============================================================================
# Constants
# =============================================================================

POLL_SECONDS = 30
ORDER_WAIT_SECONDS = 20
COOLDOWN_SECONDS = 8

MIN_BUYING_POWER = 250.0
MIN_NOTIONAL_TO_TRADE = 100.0
MIN_WEIGHT_DELTA = 0.0075

MIN_CASH_BUFFER = 1_000.0
MIN_CASH_BUFFER_PCT = 0.01
DEFAULT_CASH_BUFFER_PCT = 0.05

MAX_GROSS_EXPOSURE = 0.92
MAX_POSITION_WEIGHT = 0.25
MAX_SINGLE_ORDER_NOTIONAL_PCT = 0.1
MAX_ORDER_DOLLARS = 15_000.0
MAX_SINGLE_ORDER_QTY = 10_000

DAILY_DRAWDOWN_LIMIT = -0.025
KILL_SWITCH_EQUITY_FLOOR = 95_000.0

MAX_DAILY_LOSS_PCT = 0.02
MAX_DAILY_LOSS_DOLLARS = 2_000.0

HEARTBEAT_SECONDS = 60

# Dynamic account scaling.
# These defaults keep the executor safe on small accounts while still allowing
# larger accounts to use the existing aggressive settings. Runtime scaling uses
# actual Alpaca equity unless --disable-account-scaling is supplied.
ACCOUNT_SCALING_ENABLED = True
SMALL_ACCOUNT_THRESHOLD = 25_000.0
MID_ACCOUNT_THRESHOLD = 50_000.0
LARGE_ACCOUNT_THRESHOLD = 100_000.0
MIN_DYNAMIC_CASH_BUFFER = 250.0
DYNAMIC_CASH_BUFFER_PCT = DEFAULT_CASH_BUFFER_PCT
SMALL_ACCOUNT_MAX_GROSS_EXPOSURE = 0.72
MID_ACCOUNT_MAX_GROSS_EXPOSURE = 0.82
LARGE_ACCOUNT_MAX_GROSS_EXPOSURE = MAX_GROSS_EXPOSURE
SMALL_ACCOUNT_MAX_POSITION_WEIGHT = 0.12
MID_ACCOUNT_MAX_POSITION_WEIGHT = 0.15
LARGE_ACCOUNT_MAX_POSITION_WEIGHT = MAX_POSITION_WEIGHT
SMALL_ACCOUNT_MAX_ORDER_PCT = 0.15
MID_ACCOUNT_MAX_ORDER_PCT = 0.12
LARGE_ACCOUNT_MAX_ORDER_PCT = MAX_SINGLE_ORDER_NOTIONAL_PCT
SMALL_ACCOUNT_DAILY_LOSS_PCT = 0.015
MID_ACCOUNT_DAILY_LOSS_PCT = 0.018
LARGE_ACCOUNT_DAILY_LOSS_PCT = MAX_DAILY_LOSS_PCT
ACCOUNT_KILL_FLOOR_DRAWDOWN_PCT = 0.10
# If the executor was previously run against a much larger/smaller account,
# the saved kill-switch start_equity can make the new account look like it
# suffered a huge daily drawdown. Reset only when the equity gap is clearly an
# account-switch/funding event rather than normal trading PnL.
KILL_STATE_ACCOUNT_SWITCH_RESET_PCT = 0.35


# =============================================================================
# Continuous scanner defaults
# =============================================================================

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SCANNER_STATE_PATH = LOG_DIR / "scanner_state.json"
SCANNER_SCORES_PATH = OUT_DIR / "latest_scanner_scores.csv"
OPTION_BOOSTER_CANDIDATES_PATH = OUT_DIR / "latest_option_booster_candidates.csv"
OPTION_CANDIDATES_PATH = OUT_DIR / "latest_realtime_options_candidates.csv"
SCANNER_STATE_PATH_V2 = LOG_DIR / "scanner_state.json"

DEFAULT_SCAN_UNIVERSE = (
    pd.read_csv("tv_screener_tickers.csv", header=None)
    .iloc[0]
    .loc[lambda s: s != "HPE/PC"]
    .head(30)
    .tolist()
)
#[
 #   "QQQ", "TQQQ", "SQQQ", "SPY", "DIA", "IWM", "XLK", "SOXX", "SMH",
  #  "AMD", "NVDA", "AVGO", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA",
   # "PLTR", "COIN", "MARA", "RIOT", "RGTI", "QBTS", "QUBT", "IONQ", "SOUN",
    #"OKLO", "APLD", "ALAB", "ARM",
#]
# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class BrokerPosition:
    symbol: Optional[str] = None
    qty: float = 0.0
    market_value: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pl: float = 0.0
    unrealized_plpc: float = 0.0


@dataclass
class AccountSnapshot:
    buying_power: float = 0.0
    equity: float = 0.0
    cash: float = 0.0
    portfolio_value: float = 0.0


@dataclass
class ScannerConfig:
    enabled: bool = True
    universe: Optional[list[str]] = None
    use_sp500_fallback: bool = False
    max_universe_symbols: int = 30
    lookback_days: int = 260
    data_feed: str = "iex"

    top_n: int = 8
    min_setup_score: float = 0.62
    min_buy_confidence: float = 0.58
    min_price: float = 2.0
    max_price: float = 500.0
    min_avg_dollar_volume: float = 5_000_000.0

    base_target_weight: float = 0.04
    max_target_weight: float = 0.14
    reversal_target_weight_mult: float = 0.75

    stop_loss_pct: float = -0.035
    stale_position_score_exit: float = 0.45
    profit_lock_trigger_pct: float = 0.06
    profit_giveback_pct: float = 0.025


@dataclass
class OptionsTradingConfig:
    enabled: bool = True
    feed: str = "opra"

    # Position types
    enable_long_calls: bool = True
    enable_long_puts: bool = False

    # DTE
    min_dte: int = 7
    preferred_dte_min: int = 14
    preferred_dte_max: int = 30
    max_dte: int = 45

    # Delta ranges
    call_delta_min: float = 0.35
    call_delta_max: float = 0.65
    call_ideal_delta: float = 0.50
    put_delta_min: float = -0.65
    put_delta_max: float = -0.35
    put_ideal_delta: float = -0.50

    # Hard contract filters
    max_spread_pct: float = 0.12
    min_open_interest: int = 100
    min_volume: int = 20

    # Risk sizing
    risk_per_trade_pct: float = 0.015
    max_options_exposure: float = 0.40
    max_contracts_per_underlying: int = 2
    max_open_option_positions: int = 4

    # Limit order execution
    allow_option_market_orders: bool = False
    limit_entry_improvement: float = 0.01
    limit_exit_offset: float = 0.01

    # Profit/loss
    take_profit_pct: float = 0.60
    hard_stop_pct: float = -0.35
    profit_lock_trigger_pct: float = 0.40
    profit_giveback_pct: float = 0.25

    # Expiration protection
    close_before_expiration_days: int = 3
    block_new_entries_within_expiration_days: int = 5

    # Alpha Vantage sentiment thresholds
    pcr_bullish_threshold: float = 0.75
    pcr_bearish_threshold: float = 1.00
    pcr_contrarian_bullish_threshold: float = 1.50


# Backward-compat alias so old code using OptionsBoosterConfig still works
OptionsBoosterConfig = OptionsTradingConfig


class OptionsContractRanker:
    def __init__(self, config: OptionsTradingConfig):
        self.cfg = config

    def _score_pcr_sentiment(self, pcr: float | None, option_type: str) -> float:
        if pcr is None:
            return 0.5
        cfg = self.cfg
        if option_type == "call":
            if pcr < 0.1:
                return 0.3
            if pcr <= cfg.pcr_bullish_threshold:
                ratio = pcr / cfg.pcr_bullish_threshold
                return 1.0 - 0.15 * ratio
            if pcr >= cfg.pcr_contrarian_bullish_threshold:
                return 0.35
            if pcr >= cfg.pcr_bearish_threshold:
                span = cfg.pcr_contrarian_bullish_threshold - cfg.pcr_bearish_threshold
                if span <= 0:
                    return 0.10
                t = (pcr - cfg.pcr_bearish_threshold) / span
                return 0.15 - 0.10 * t
            span = cfg.pcr_bearish_threshold - cfg.pcr_bullish_threshold
            if span <= 0:
                return 0.5
            t = (pcr - cfg.pcr_bullish_threshold) / span
            return 0.85 - 0.70 * t
        else:
            if pcr is None:
                return 0.5
            if pcr >= cfg.pcr_contrarian_bullish_threshold:
                return 0.6
            if pcr >= cfg.pcr_bearish_threshold:
                span = cfg.pcr_contrarian_bullish_threshold - cfg.pcr_bearish_threshold
                if span <= 0:
                    return 0.85
                t = (pcr - cfg.pcr_bearish_threshold) / span
                return 0.75 + 0.25 * (1.0 - t)
            if pcr <= cfg.pcr_bullish_threshold:
                return 0.10
            span = cfg.pcr_bearish_threshold - cfg.pcr_bullish_threshold
            if span <= 0:
                return 0.4
            t = (pcr - cfg.pcr_bullish_threshold) / span
            return 0.10 + 0.65 * t

    def _score_volume_oi(self, vol_oi_ratio: float | None, volume: float, open_interest: float) -> float:
        if vol_oi_ratio is not None:
            ratio = vol_oi_ratio
        else:
            if open_interest <= 0:
                return 0.15
            ratio = volume / open_interest

        if ratio > 20.0:
            return 0.2
        if ratio > 5.0:
            t = (ratio - 5.0) / 15.0
            return 1.0 - 0.5 * t
        if ratio < 0.05:
            return 0.15
        if 0.10 <= ratio <= 2.0:
            t = (ratio - 0.10) / 1.90
            return 0.65 + 0.35 * t
        if ratio < 0.10:
            t = (ratio - 0.05) / 0.05
            return 0.15 + 0.50 * t
        t = (ratio - 2.0) / 3.0
        return 1.0 - 0.5 * t

    def _score_liquidity(self, volume: float, open_interest: float, bid_size: float, ask_size: float) -> float:
        cfg = self.cfg
        min_vol_10 = max(cfg.min_volume * 10, 1)
        min_oi_10 = max(cfg.min_open_interest * 10, 1)
        vol_score = min(volume / min_vol_10, 1.0)
        oi_score = min(open_interest / min_oi_10, 1.0)
        size_score = min((bid_size + ask_size) / 20.0, 1.0)
        return 0.45 * vol_score + 0.45 * oi_score + 0.10 * size_score

    def _score_delta_fit(self, delta: float, option_type: str) -> float:
        if delta is None or (isinstance(delta, float) and np.isnan(delta)):
            return 0.4
        cfg = self.cfg
        if option_type == "call":
            ideal = cfg.call_ideal_delta
            low = cfg.call_delta_min
            high = cfg.call_delta_max
        else:
            ideal = cfg.put_ideal_delta
            low = cfg.put_delta_min
            high = cfg.put_delta_max
        half_range = (high - low) / 2.0
        if half_range <= 0:
            return 0.4
        score = 1.0 - abs(delta - ideal) / half_range
        return float(np.clip(score, 0.0, 1.0))

    def _score_spread(self, spread_pct: float) -> float:
        cfg = self.cfg
        if cfg.max_spread_pct <= 0:
            return 0.0
        score = 1.0 - spread_pct / cfg.max_spread_pct
        return float(np.clip(score, 0.0, 1.0))

    def _score_dte_fit(self, dte: int) -> float:
        cfg = self.cfg
        pmin = cfg.preferred_dte_min
        pmax = cfg.preferred_dte_max
        dmin = cfg.min_dte
        dmax = cfg.max_dte
        if pmin <= dte <= pmax:
            return 1.0
        if dte < dmin or dte > dmax:
            return 0.0
        if dte < pmin:
            span = pmin - dmin
            if span <= 0:
                return 0.0
            return (dte - dmin) / span
        span = dmax - pmax
        if span <= 0:
            return 0.0
        return (dmax - dte) / span

    def _score_iv_moneyness(self, iv: float, strike: float, underlying_price: float) -> float:
        if iv is None or (isinstance(iv, float) and np.isnan(iv)) or iv <= 0 or iv > 5.0:
            iv_score = 0.15
        elif iv < 0.10:
            iv_score = 0.35
        elif 0.15 <= iv <= 1.5:
            t = (iv - 0.15) / (1.5 - 0.15)
            iv_score = 0.5 + 0.5 * t
        else:
            iv_score = 0.5

        if underlying_price <= 0:
            moneyness_score = 0.5
        else:
            moneyness_raw = abs(strike / underlying_price - 1.0) / 0.15
            moneyness_score = 1.0 - float(np.clip(moneyness_raw, 0.0, 1.0))

        return 0.6 * iv_score + 0.4 * moneyness_score

    def filter_contracts(self, contracts: list[dict]) -> list[dict]:
        cfg = self.cfg
        passing = []
        for c in contracts:
            bid = to_float(c.get("bid"), 0.0)
            ask = to_float(c.get("ask"), 0.0)
            mid = to_float(c.get("mid"), 0.0)
            spread_pct = to_float(c.get("spread_pct"), 999.0)
            oi = to_float(c.get("open_interest"), 0.0)
            vol = to_float(c.get("volume"), 0.0)
            iv = c.get("iv")
            dte = to_int(c.get("dte"), 0)

            if bid <= 0 or ask <= 0 or mid <= 0:
                continue
            if spread_pct > cfg.max_spread_pct:
                continue
            if oi < cfg.min_open_interest:
                continue
            if vol < cfg.min_volume:
                continue
            iv_val = to_float(iv, float("nan"))
            if np.isnan(iv_val) or iv_val <= 0 or iv_val > 5.0:
                continue
            if dte <= cfg.close_before_expiration_days:
                continue
            passing.append(c)
        return passing

    def score_contract(self, contract: dict, underlying_momentum_score: float,
                       pcr: float | None, vol_oi_ratio: float | None) -> float:
        c = contract
        option_type = str(c.get("option_type", "call")).lower()
        delta = to_float(c.get("delta"), float("nan"))
        if delta == 0.0 and "delta" in c and c["delta"] is None:
            delta = float("nan")
        spread_pct = to_float(c.get("spread_pct"), 1.0)
        volume = to_float(c.get("volume"), 0.0)
        oi = to_float(c.get("open_interest"), 0.0)
        bid_size = to_float(c.get("bid_size"), 0.0)
        ask_size = to_float(c.get("ask_size"), 0.0)
        iv = to_float(c.get("iv"), float("nan"))
        strike = to_float(c.get("strike"), 0.0)
        underlying_price = to_float(c.get("underlying_price"), 0.0)
        dte = to_int(c.get("dte"), 0)

        s_momentum = float(np.clip(underlying_momentum_score, 0.0, 1.0))
        s_pcr = self._score_pcr_sentiment(pcr, option_type)
        s_vol_oi = self._score_volume_oi(vol_oi_ratio, volume, oi)
        s_liquidity = self._score_liquidity(volume, oi, bid_size, ask_size)
        s_delta = self._score_delta_fit(delta, option_type)
        s_spread = self._score_spread(spread_pct)
        s_dte = self._score_dte_fit(dte)
        s_iv = self._score_iv_moneyness(iv, strike, underlying_price)

        score = (
            0.20 * s_momentum
            + 0.18 * s_pcr
            + 0.18 * s_vol_oi
            + 0.14 * s_liquidity
            + 0.12 * s_delta
            + 0.08 * s_spread
            + 0.06 * s_dte
            + 0.04 * s_iv
        )
        return float(np.clip(score, 0.0, 1.0))

    def rank(self, contracts: list[dict], underlying_momentum_score: float,
             pcr: float | None, vol_oi_ratio: float | None,
             option_type: str = "call") -> pd.DataFrame:
        filtered = self.filter_contracts(contracts)
        if not filtered:
            return pd.DataFrame()

        rows = []
        for c in filtered:
            score = self.score_contract(c, underlying_momentum_score, pcr, vol_oi_ratio)
            row = dict(c)
            row["contract_score"] = score
            row["put_call_ratio"] = pcr
            row["volume_oi_ratio"] = vol_oi_ratio
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values("contract_score", ascending=False).reset_index(drop=True)
        return df


@dataclass
class ExecutorConfig:
    paper: bool = True
    live: bool = False

    enable_account_scaling: bool = ACCOUNT_SCALING_ENABLED
    account_scale_reference_equity: float = 0.0

    poll_once: bool = False
    poll_seconds: int = POLL_SECONDS

    rebalance_orders_path: Path = DEFAULT_REBALANCE_ORDERS
    exit_orders_path: Path = DEFAULT_EXIT_ORDERS

    require_market_open: bool = True
    wait_for_market_open: bool = False
    allow_closed_market: bool = False
    max_wait_for_open_minutes: int = 180

    cancel_open_orders_first: bool = False
    sell_first: bool = True

    cash_buffer_pct: float = DEFAULT_CASH_BUFFER_PCT
    min_cash_buffer: float = MIN_CASH_BUFFER
    min_buying_power: float = MIN_BUYING_POWER
    min_notional_to_trade: float = MIN_NOTIONAL_TO_TRADE
    min_weight_delta: float = MIN_WEIGHT_DELTA

    max_gross_exposure: float = MAX_GROSS_EXPOSURE
    max_position_weight: float = MAX_POSITION_WEIGHT
    max_single_order_notional_pct: float = MAX_SINGLE_ORDER_NOTIONAL_PCT
    max_order_dollars: float = MAX_ORDER_DOLLARS
    max_single_order_qty: int = MAX_SINGLE_ORDER_QTY

    cooldown_seconds: int = COOLDOWN_SECONDS

    enable_kill_switch: bool = True
    daily_drawdown_limit: float = DAILY_DRAWDOWN_LIMIT
    max_daily_loss_pct: float = MAX_DAILY_LOSS_PCT
    max_daily_loss_dollars: float = MAX_DAILY_LOSS_DOLLARS
    kill_switch_equity_floor: float = KILL_SWITCH_EQUITY_FLOOR

    order_wait_seconds: int = ORDER_WAIT_SECONDS
    time_in_force: TimeInForce = TimeInForce.DAY
    order_tag_prefix: str = "explosive_alpha"

    scanner: ScannerConfig = None
    options_booster: OptionsTradingConfig = None  # OptionsTradingConfig or legacy alias


# =============================================================================
# General helpers
# =============================================================================

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def run_id() -> str:
    return now_utc().strftime("%Y%m%d_%H%M%S")


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def safe_symbol(value: Any) -> str:
    return str(value).upper().strip().replace(".", "-")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    return str(value)


def write_event(event: str, payload: dict[str, Any] | None = None) -> None:
    row = {
        "timestamp_utc": now_utc().isoformat(),
        "event": event,
        "payload": payload or {},
    }
    with EVENTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=json_default) + "\n")


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[saved] {path}")


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=json_default) + "\n", encoding="utf-8")
    print(f"[saved] {path}")


# =============================================================================
# Logging
# =============================================================================

class CsvJournal:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.columns = [
            "timestamp_utc",
            "cycle_id",
            "event",
            "market_open",
            "equity_before",
            "equity_after",
            "cash_before",
            "cash_after",
            "buying_power_before",
            "buying_power_after",
            "gross_exposure_before",
            "gross_exposure_after",
            "trade_symbol",
            "trade_side",
            "trade_qty",
            "trade_price",
            "estimated_notional",
            "target_weight",
            "current_weight",
            "delta_weight",
            "confidence",
            "cash_buffer",
            "max_gross_exposure",
            "max_position_weight",
            "max_order_notional",
            "order_id",
            "order_status",
            "filled_qty",
            "filled_avg_price",
            "slippage_pct",
            "message",
        ]

        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.columns).writeheader()

    def append(self, row: dict[str, object]) -> None:
        safe_row = {k: row.get(k, "") for k in self.columns}
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.columns).writerow(safe_row)


# =============================================================================
# Alpaca broker wrapper
# =============================================================================

class AlpacaBroker:
    def __init__(self, env_path: Path, paper: bool = True):
        load_dotenv(env_path)

        self.api_key = os.getenv("APCA_API_KEY_ID")
        self.secret_key = os.getenv("APCA_API_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise RuntimeError(f"Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in {env_path}")

        self.trading = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=paper,
        )

    def get_account_snapshot(self) -> AccountSnapshot:
        acct = self.trading.get_account()
        return AccountSnapshot(
            buying_power=to_float(getattr(acct, "buying_power", 0.0)),
            equity=to_float(getattr(acct, "equity", 0.0)),
            cash=to_float(getattr(acct, "cash", 0.0)),
            portfolio_value=to_float(getattr(acct, "portfolio_value", 0.0)),
        )

    def get_all_open_positions(self) -> list[BrokerPosition]:
        positions: list[BrokerPosition] = []
        try:
            raw_positions = self.trading.get_all_positions()
        except Exception as exc:
            print(f"[broker] failed to fetch positions: {exc}")
            return positions

        for pos in raw_positions:
            qty = to_float(getattr(pos, "qty", 0.0))
            if qty <= 0:
                continue

            positions.append(
                BrokerPosition(
                    symbol=safe_symbol(getattr(pos, "symbol", "")),
                    qty=qty,
                    market_value=to_float(getattr(pos, "market_value", 0.0)),
                    avg_entry_price=to_float(getattr(pos, "avg_entry_price", 0.0)),
                    current_price=to_float(getattr(pos, "current_price", 0.0)),
                    unrealized_pl=to_float(getattr(pos, "unrealized_pl", 0.0)),
                    unrealized_plpc=to_float(getattr(pos, "unrealized_plpc", 0.0)),
                )
            )

        return positions

    def get_position(self, symbol: str) -> BrokerPosition:
        symbol = safe_symbol(symbol)
        for pos in self.get_all_open_positions():
            if pos.symbol == symbol:
                return pos
        return BrokerPosition(symbol=symbol)

    def list_open_orders(self) -> list:
        try:
            return self.trading.get_orders()
        except Exception:
            return []

    def has_open_order_for_symbol(self, symbol: str) -> bool:
        symbol = safe_symbol(symbol)

        open_statuses = {
            OrderStatus.NEW,
            OrderStatus.ACCEPTED,
            OrderStatus.PENDING_NEW,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.ACCEPTED_FOR_BIDDING,
            OrderStatus.CALCULATED,
        }

        for order in self.list_open_orders():
            try:
                if safe_symbol(order.symbol) == symbol and order.status in open_statuses:
                    return True
            except Exception:
                continue

        return False

    def cancel_open_orders(self) -> None:
        self.trading.cancel_orders()

    def get_clock_info(self) -> dict[str, Any]:
        clock = self.trading.get_clock()
        return {
            "is_open": bool(getattr(clock, "is_open", False)),
            "timestamp": str(getattr(clock, "timestamp", "")),
            "next_open": str(getattr(clock, "next_open", "")),
            "next_close": str(getattr(clock, "next_close", "")),
        }

    def submit_market_order(self, symbol: str, side: str, qty: int, client_order_id: str, tif: TimeInForce):
        order = MarketOrderRequest(
            symbol=safe_symbol(symbol),
            qty=int(qty),
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=tif,
            client_order_id=client_order_id,
        )
        return self.trading.submit_order(order_data=order)

    def submit_market_buy(self, symbol: str, qty: int, client_order_id: str, tif: TimeInForce):
        return self.submit_market_order(symbol=symbol, side="buy", qty=qty, client_order_id=client_order_id, tif=tif)

    def submit_market_sell(self, symbol: str, qty: int, client_order_id: str, tif: TimeInForce):
        return self.submit_market_order(symbol=symbol, side="sell", qty=qty, client_order_id=client_order_id, tif=tif)

    def wait_until_qty_at_or_below(self, symbol: str, target_qty: float, timeout_seconds: int = ORDER_WAIT_SECONDS) -> None:
        started = time.time()
        while time.time() - started < timeout_seconds:
            if self.get_position(symbol).qty <= target_qty:
                return
            time.sleep(0.5)
        raise TimeoutError(f"{symbol} position did not reach target qty <= {target_qty} within timeout")

    def submit_limit_order(self, symbol: str, side: str, qty: int, limit_price: float,
                           client_order_id: str, tif: TimeInForce):
        """Submit a limit order. Options require whole-number qty and no notional."""
        from alpaca.trading.requests import LimitOrderRequest
        order = LimitOrderRequest(
            symbol=safe_symbol(symbol),
            qty=int(qty),
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            type=OrderType.LIMIT,
            time_in_force=tif,
            limit_price=round(float(limit_price), 2),
            client_order_id=client_order_id,
        )
        return self.trading.submit_order(order_data=order)

    def submit_limit_buy(self, symbol: str, qty: int, limit_price: float,
                         client_order_id: str, tif: TimeInForce):
        return self.submit_limit_order(symbol, "buy", qty, limit_price, client_order_id, tif)

    def submit_limit_sell(self, symbol: str, qty: int, limit_price: float,
                          client_order_id: str, tif: TimeInForce):
        return self.submit_limit_order(symbol, "sell", qty, limit_price, client_order_id, tif)

    def verify_options_eligibility(self) -> dict:
        """Check if account has options trading approval (level >= 1 = long calls/puts)."""
        try:
            acct = self.trading.get_account()
            level = to_int(getattr(acct, "options_approved_level", 0), 0)
            return {
                "eligible": level >= 1,
                "level": level,
                "details": str(getattr(acct, "options_approved_level", "unknown")),
            }
        except Exception as exc:
            print(f"[broker] options eligibility check failed: {exc}")
            return {"eligible": False, "level": 0, "details": str(exc)}

    def cancel_order_by_id(self, order_id: str) -> bool:
        """Cancel a specific order by ID. Returns True if successful."""
        try:
            self.trading.cancel_order_by_id(order_id)
            return True
        except Exception as exc:
            print(f"[broker] cancel_order_by_id {order_id} failed: {exc}")
            return False

    def wait_for_position(self, symbol: str, timeout_seconds: int = ORDER_WAIT_SECONDS) -> BrokerPosition:
        started = time.time()
        while time.time() - started < timeout_seconds:
            pos = self.get_position(symbol)
            if pos.qty > 0:
                return pos
            time.sleep(0.5)
        return BrokerPosition(symbol=symbol)

    def get_order_by_id(self, order_id: str):
        return self.trading.get_order_by_id(order_id)


class AlphaVantageOptionsSentimentClient:
    BASE_URL = "https://www.alphavantage.co/query"
    RATE_LIMIT = 5
    RATE_WINDOW = 60.0
    TTL = 300.0

    def __init__(self, api_key: str) -> None:
        from collections import deque
        self._api_key = api_key
        self._call_times: deque = deque()
        self._cache: dict[str, tuple[float, Any]] = {}

    def _rate_limit(self) -> None:
        now = time.monotonic()
        while self._call_times and now - self._call_times[0] >= self.RATE_WINDOW:
            self._call_times.popleft()
        if len(self._call_times) >= self.RATE_LIMIT:
            oldest = self._call_times[0]
            sleep_for = self.RATE_WINDOW - (now - oldest)
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            while self._call_times and now - self._call_times[0] >= self.RATE_WINDOW:
                self._call_times.popleft()
        self._call_times.append(time.monotonic())

    def _fetch(self, function: str, symbol: str, retries: int = 3) -> dict | None:
        cache_key = f"{function}:{symbol}"
        now = time.time()
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if now - ts < self.TTL:
                return data
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self._api_key,
        }
        for attempt in range(retries):
            try:
                self._rate_limit()
                resp = requests.get(self.BASE_URL, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                for bad_key in ("Error Message", "Note", "Information"):
                    if bad_key in data:
                        return None
                self._cache[cache_key] = (time.time(), data)
                return data
            except Exception:
                backoff = (2 ** attempt) * 0.5
                time.sleep(backoff)
        return None

    def _get_latest_value(self, data: dict, *keys: str) -> float | None:
        if not isinstance(data, dict):
            return None
        records = None
        for v in data.values():
            if isinstance(v, list) and len(v) > 0:
                records = v
                break
            if isinstance(v, dict):
                for inner_v in v.values():
                    if isinstance(inner_v, list) and len(inner_v) > 0:
                        records = inner_v
                        break
            if records is not None:
                break
        if records is None:
            record = data
        else:
            record = records[0]
        if not isinstance(record, dict):
            return None
        for key in keys:
            if key in record:
                val = to_float(record[key], default=None)
                if val is not None:
                    return val
        return None

    def get_put_call_ratio(self, symbol: str) -> float | None:
        try:
            data = self._fetch("REALTIME_PUT_CALL_RATIO", symbol)
            if data is None:
                return None
            return self._get_latest_value(
                data,
                "put_call_ratio",
                "value",
                "put_call_ratio_60day",
            )
        except Exception:
            return None

    def get_volume_oi_ratio(self, symbol: str) -> float | None:
        try:
            data = self._fetch("REALTIME_VOLUME_OPEN_INTEREST_RATIO", symbol)
            if data is None:
                return None
            return self._get_latest_value(
                data,
                "volume_open_interest_ratio",
                "value",
            )
        except Exception:
            return None


class AlpacaOptionChainDiscovery:
    """Discovers option contracts via Alpaca REST API."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self._data_url = "https://data.alpaca.markets"
        self._headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
            "accept": "application/json",
        }

    def get_contracts(
        self,
        underlying: str,
        expiration_date_gte: str | None = None,
        expiration_date_lte: str | None = None,
        option_type: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        params: dict[str, Any] = {
            "underlying_symbols": underlying,
            "status": "active",
            "limit": limit,
        }
        if expiration_date_gte:
            params["expiration_date_gte"] = expiration_date_gte
        if expiration_date_lte:
            params["expiration_date_lte"] = expiration_date_lte
        if option_type:
            params["type"] = option_type
        try:
            resp = requests.get(
                f"{self.base_url}/v2/options/contracts",
                headers=self._headers,
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("option_contracts", [])
        except Exception as exc:
            print(f"[AlpacaOptionChainDiscovery.get_contracts] error: {exc}")
            return []

    def get_snapshots_bulk(self, symbols: list[str], feed: str = "indicative") -> dict[str, dict]:
        results: dict[str, dict] = {}
        chunk_size = 50
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i : i + chunk_size]
            try:
                resp = requests.get(
                    f"{self._data_url}/v1beta1/options/snapshots",
                    headers=self._headers,
                    params={"symbols": ",".join(chunk), "feed": feed},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                snapshots = data.get("snapshots", data)
                if isinstance(snapshots, dict):
                    results.update(snapshots)
            except Exception as exc:
                print(f"[AlpacaOptionChainDiscovery.get_snapshots_bulk] chunk error: {exc}")
        return results

    def discover_contracts_for_underlying(
        self,
        underlying: str,
        min_dte: int = 7,
        max_dte: int = 45,
        option_type: str | None = None,
        feed: str = "indicative",
    ) -> list[dict]:
        try:
            today = now_utc().date()
            exp_gte = (today + timedelta(days=min_dte)).isoformat()
            exp_lte = (today + timedelta(days=max_dte)).isoformat()
        except Exception as exc:
            print(f"[discover_contracts_for_underlying] date error: {exc}")
            return []

        try:
            contracts = self.get_contracts(
                underlying=underlying,
                expiration_date_gte=exp_gte,
                expiration_date_lte=exp_lte,
                option_type=option_type,
            )
        except Exception as exc:
            print(f"[discover_contracts_for_underlying] get_contracts error: {exc}")
            return []

        try:
            symbols = [c.get("symbol", "") for c in contracts if c.get("symbol")]
        except Exception as exc:
            print(f"[discover_contracts_for_underlying] symbol extraction error: {exc}")
            return []

        try:
            snapshots = self.get_snapshots_bulk(symbols, feed=feed) if symbols else {}
        except Exception as exc:
            print(f"[discover_contracts_for_underlying] get_snapshots_bulk error: {exc}")
            snapshots = {}

        results: list[dict] = []
        today = now_utc().date()
        for contract in contracts:
            sym = contract.get("symbol", "")
            if not sym:
                continue
            try:
                parsed = parse_occ_option_symbol(sym) or {}
                snapshot = snapshots.get(sym, {})
                metrics = _option_quote_metrics(snapshot)
                exp_str = parsed.get("expiration") or contract.get("expiration_date", "")
                try:
                    from datetime import date as _date
                    if exp_str:
                        exp_date = _date.fromisoformat(str(exp_str)[:10])
                        dte = (exp_date - today).days
                    else:
                        dte = 0
                except Exception:
                    dte = 0
                results.append(
                    {
                        "symbol": sym,
                        "underlying": parsed.get("underlying") or contract.get("underlying_symbol", underlying),
                        "expiration": str(exp_str)[:10] if exp_str else "",
                        "dte": dte,
                        "strike": to_float(parsed.get("strike") or contract.get("strike_price"), 0.0),
                        "type": parsed.get("option_type") or contract.get("type", ""),
                        "bid": metrics.get("bid", 0.0),
                        "ask": metrics.get("ask", 0.0),
                        "bid_size": to_int(snapshot.get("latestQuote", {}).get("bs", 0), 0),
                        "ask_size": to_int(snapshot.get("latestQuote", {}).get("as", 0), 0),
                        "mid": metrics.get("mid", 0.0),
                        "spread_pct": metrics.get("spread_pct", 0.0),
                        "delta": metrics.get("delta", 0.0),
                        "gamma": metrics.get("gamma", 0.0),
                        "iv": metrics.get("iv", 0.0),
                        "volume": metrics.get("volume", 0),
                        "open_interest": metrics.get("open_interest", 0),
                    }
                )
            except Exception as exc:
                print(f"[discover_contracts_for_underlying] contract processing error for {sym}: {exc}")
                continue
        return results


class AlpacaRealtimeOptionStream:
    """Thread-safe WebSocket stream for Alpaca options data using MsgPack.
    NEVER subscribes to wildcard (*)."""

    def __init__(self, api_key: str, secret_key: str, feed: str = "opra"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.feed = feed
        self.url = f"wss://stream.data.alpaca.markets/v1beta1/{feed}"
        self._quotes: dict[str, dict] = {}
        self._trades: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._ws = None
        self._thread: threading.Thread | None = None
        self._subscribed: set[str] = set()
        self._running = False

    def _on_open(self, ws):
        if _msgpack_mod is None:
            print("[stream] msgpack not available, cannot send auth")
            return
        try:
            auth_msg = _msgpack_mod.packb(
                {"action": "auth", "key": self.api_key, "secret": self.secret_key},
                use_bin_type=True,
            )
            ws.send(auth_msg, opcode=0x2)
        except Exception as exc:
            print(f"[stream] auth send error: {exc}")

    def _on_message(self, _ws, message):
        if _msgpack_mod is None:
            return
        try:
            msgs = _msgpack_mod.unpackb(message, raw=False)
            if not isinstance(msgs, list):
                msgs = [msgs]
            with self._lock:
                for msg in msgs:
                    if not isinstance(msg, dict):
                        continue
                    msg_type = msg.get("T")
                    sym = msg.get("S", "")
                    if not sym:
                        continue
                    if msg_type == "q":
                        self._quotes[sym] = {
                            "bid": to_float(msg.get("bp"), 0.0),
                            "ask": to_float(msg.get("ap"), 0.0),
                            "bid_size": to_int(msg.get("bs"), 0),
                            "ask_size": to_int(msg.get("as"), 0),
                            "timestamp": msg.get("t"),
                        }
                    elif msg_type == "t":
                        self._trades[sym] = {
                            "price": to_float(msg.get("p"), 0.0),
                            "size": to_int(msg.get("s"), 0),
                            "exchange": msg.get("x", ""),
                            "condition": msg.get("c", ""),
                            "timestamp": msg.get("t"),
                        }
        except Exception as exc:
            print(f"[stream] message parse error: {exc}")

    def _on_error(self, _ws, error):
        print(f"[stream] error: {error}")

    def _on_close(self, _ws, close_status_code, close_msg):
        print(f"[stream] closed: {close_status_code} {close_msg}")
        self._running = False

    def start(self) -> bool:
        if _msgpack_mod is None:
            print("[stream] msgpack not installed — cannot start stream")
            return False
        if _websocket_mod is None:
            print("[stream] websocket-client not installed — cannot start stream")
            return False
        self._running = True
        self._ws = _websocket_mod.WebSocketApp(
            self.url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._thread = threading.Thread(
            target=self._ws.run_forever,
            daemon=True,
            name="alpaca-option-stream",
        )
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception as exc:
                print(f"[stream] stop error: {exc}")

    def subscribe(self, symbols: list[str]):
        if _msgpack_mod is None or self._ws is None:
            return
        new_syms = [s for s in symbols if s and s != "*" and s not in self._subscribed]
        if not new_syms:
            return
        try:
            sub_msg = _msgpack_mod.packb(
                {"action": "subscribe", "quotes": new_syms, "trades": new_syms},
                use_bin_type=True,
            )
            self._ws.send(sub_msg, opcode=0x2)
            self._subscribed.update(new_syms)
        except Exception as exc:
            print(f"[stream] subscribe error: {exc}")

    def get_quote(self, symbol: str) -> dict | None:
        with self._lock:
            return self._quotes.get(symbol)

    def get_trade(self, symbol: str) -> dict | None:
        with self._lock:
            return self._trades.get(symbol)

    def get_mid(self, symbol: str) -> float | None:
        with self._lock:
            q = self._quotes.get(symbol)
        if q is None:
            return None
        bid = q.get("bid")
        ask = q.get("ask")
        if bid is None or ask is None:
            return None
        try:
            return (float(bid) + float(ask)) / 2.0
        except Exception:
            return None


# =============================================================================
# Continuous setup scanner
# =============================================================================

def _web_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }


def normalize_symbol(symbol: Any) -> str:
    return safe_symbol(symbol)


def get_sp500_fallback_symbols() -> list[str]:
    try:
        resp = requests.get(WIKI_SP500_URL, headers=_web_headers(), timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        if not tables or "Symbol" not in tables[0].columns:
            raise RuntimeError("Could not parse S&P 500 symbols from Wikipedia.")
        return sorted(set(normalize_symbol(x) for x in tables[0]["Symbol"].astype(str)))
    except Exception as exc:
        print(f"[scanner] S&P 500 fallback failed: {exc}; using DEFAULT_SCAN_UNIVERSE")
        return list(DEFAULT_SCAN_UNIVERSE)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()



OCC_OPTION_SYMBOL_RE = re.compile(r"^([A-Z][A-Z0-9-]*?)(\d{6})([CP])(\d{8})$")


def parse_occ_option_symbol(symbol: Any) -> dict[str, Any] | None:
    """Parse standard OCC/Alpaca option symbols such as AAPL260116C00200000."""
    raw = safe_symbol(symbol)
    match = OCC_OPTION_SYMBOL_RE.match(raw)
    if not match:
        return None

    underlying, yymmdd, cp, strike_raw = match.groups()
    try:
        expiration = datetime.strptime(yymmdd, "%y%m%d").date()
        strike = int(strike_raw) / 1000.0
    except Exception:
        return None

    return {
        "symbol": raw,
        "underlying": underlying,
        "expiration": expiration,
        "type": "call" if cp == "C" else "put",
        "strike": strike,
    }


def _obj_get(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _option_quote_metrics(snapshot: Any) -> dict[str, float]:
    quote = _obj_get(snapshot, "latest_quote", "latestQuote", default=None)
    greeks = _obj_get(snapshot, "greeks", default=None)

    bid = to_float(_obj_get(quote, "bid_price", "bp", "bid", default=0.0))
    ask = to_float(_obj_get(quote, "ask_price", "ap", "ask", default=0.0))
    bid_size = to_float(_obj_get(quote, "bid_size", "bs", default=0.0))
    ask_size = to_float(_obj_get(quote, "ask_size", "as", default=0.0))
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
    spread_pct = (ask - bid) / mid if mid > 0 else 999.0

    delta = to_float(_obj_get(greeks, "delta", default=np.nan), np.nan)
    gamma = to_float(_obj_get(greeks, "gamma", default=np.nan), np.nan)
    iv = to_float(_obj_get(snapshot, "implied_volatility", "iv", default=np.nan), np.nan)
    volume = to_float(_obj_get(snapshot, "volume", default=0.0))
    open_interest = to_float(_obj_get(snapshot, "open_interest", "openInterest", default=0.0))

    return {
        "bid": bid,
        "ask": ask,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "mid": mid,
        "spread_pct": spread_pct,
        "delta": delta,
        "gamma": gamma,
        "iv": iv,
        "volume": volume,
        "open_interest": open_interest,
    }


class MomentumSetupScanner:
    """Builds fresh rebalance/exit CSVs before the executor submits orders."""

    def __init__(self, broker: AlpacaBroker, config: ScannerConfig, executor_config: ExecutorConfig):
        self.broker = broker
        self.config = config
        self.executor_config = executor_config

        if StockHistoricalDataClient is None:
            raise RuntimeError("alpaca-py data classes are unavailable. Install alpaca-py or disable scanner.")

        self.data = StockHistoricalDataClient(broker.api_key, broker.secret_key)
        self.options_config: OptionsTradingConfig = (
            executor_config.options_booster or OptionsTradingConfig(enabled=False)
        )
        self.option_data = (
            OptionHistoricalDataClient(broker.api_key, broker.secret_key)
            if OptionHistoricalDataClient is not None
            else None
        )
        self.chain_discovery = AlpacaOptionChainDiscovery(
            api_key=broker.api_key,
            secret_key=broker.secret_key,
            paper=executor_config.paper,
        )
        self.ranker = OptionsContractRanker(self.options_config)
        _av_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self.av_client: AlphaVantageOptionsSentimentClient | None = (
            AlphaVantageOptionsSentimentClient(_av_key) if _av_key else None
        )
        self.option_stream: AlpacaRealtimeOptionStream | None = None

    def _load_state(self) -> dict[str, Any]:
        if not SCANNER_STATE_PATH.exists():
            return {"high_plpc": {}}
        try:
            return json.loads(SCANNER_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"high_plpc": {}}

    def _save_state(self, state: dict[str, Any]) -> None:
        SCANNER_STATE_PATH.write_text(json.dumps(state, indent=2, default=json_default) + "\n", encoding="utf-8")

    def _universe(self) -> list[str]:
        if self.config.universe:
            symbols = self.config.universe
        elif self.config.use_sp500_fallback:
            symbols = get_sp500_fallback_symbols()
        else:
            symbols = DEFAULT_SCAN_UNIVERSE

        out: list[str] = []
        for sym in symbols:
            sym = normalize_symbol(sym)
            if sym and sym not in out:
                out.append(sym)
        return out[: self.config.max_universe_symbols]

    def _feed(self):
        if DataFeed is None:
            return None
        return DataFeed.IEX if str(self.config.data_feed).lower() == "iex" else DataFeed.SIP

    def _download_bars(self, symbols: list[str]) -> pd.DataFrame:
        end = now_utc()
        start = end - timedelta(days=max(30, self.config.lookback_days))
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=self._feed(),
        )
        bars = self.data.get_stock_bars(req).df
        if bars is None or bars.empty:
            return pd.DataFrame()

        df = bars.reset_index()
        rename = {"level_0": "symbol", "level_1": "timestamp"}
        df = df.rename(columns=rename)
        if "symbol" not in df.columns and "Symbol" in df.columns:
            df = df.rename(columns={"Symbol": "symbol"})
        if "timestamp" not in df.columns and "Timestamp" in df.columns:
            df = df.rename(columns={"Timestamp": "timestamp"})

        df["symbol"] = df["symbol"].map(normalize_symbol)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        return df.dropna(subset=["symbol", "timestamp", "close"]).sort_values(["symbol", "timestamp"])

    def _build_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for _symbol, g in bars.groupby("symbol", sort=False):
            g = g.sort_values("timestamp").copy()
            c = g["close"].astype(float)
            v = g["volume"].astype(float)

            g["ret_1d"] = c.pct_change(1, fill_method=None)
            g["ret_3d"] = c.pct_change(3, fill_method=None)
            g["ret_5d"] = c.pct_change(5, fill_method=None)
            g["ret_10d"] = c.pct_change(10, fill_method=None)
            g["sma5"] = c.rolling(5).mean()
            g["sma20"] = c.rolling(20).mean()
            g["sma50"] = c.rolling(50).mean()
            g["sma200"] = c.rolling(200).mean()
            g["rsi14"] = _rsi(c, 14)
            g["atr14"] = _atr(g, 14)
            g["atr_pct"] = g["atr14"] / c.replace(0, np.nan)
            g["vol20"] = v.rolling(20).mean()
            g["dollar_volume20"] = (c * v).rolling(20).mean()
            g["rel_volume"] = v / g["vol20"].replace(0, np.nan)
            mid = g["sma20"]
            sd = c.rolling(20).std()
            g["bb_low"] = mid - 2 * sd
            g["bb_high"] = mid + 2 * sd
            g["bb_pos"] = (c - g["bb_low"]) / (g["bb_high"] - g["bb_low"]).replace(0, np.nan)
            g["high20"] = c.rolling(20).max()
            g["drawdown_20h"] = c / g["high20"].replace(0, np.nan) - 1.0
            g["trend_stack"] = ((g["sma5"] > g["sma20"]) & (g["sma50"] > g["sma200"])).astype(float)
            g["above_sma20"] = (c > g["sma20"]).astype(float)
            g["above_prev_high"] = (c > g["high"].shift(1)).astype(float)
            g["lower_band_reclaim"] = ((c.shift(1) < g["bb_low"].shift(1)) & (c > g["bb_low"]) & (g["ret_1d"] > 0)).astype(float)
            rows.append(g)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def _score_latest(self, features: pd.DataFrame) -> pd.DataFrame:
        latest = features.sort_values("timestamp").groupby("symbol", as_index=False).tail(1).copy()
        if latest.empty:
            return latest

        latest = latest[
            (latest["close"] >= self.config.min_price)
            & (latest["close"] <= self.config.max_price)
            & (latest["dollar_volume20"] >= self.config.min_avg_dollar_volume)
        ].copy()
        if latest.empty:
            return latest

        def clip01(s: pd.Series, lo: float, hi: float) -> pd.Series:
            return ((s - lo) / max(hi - lo, 1e-9)).clip(0, 1)

        momentum = (
            0.20 * clip01(latest["ret_5d"], -0.02, 0.10)
            + 0.15 * clip01(latest["ret_10d"], -0.03, 0.18)
            + 0.15 * latest["trend_stack"].fillna(0)
            + 0.12 * latest["above_sma20"].fillna(0)
            + 0.10 * latest["above_prev_high"].fillna(0)
            + 0.10 * clip01(latest["rel_volume"], 0.8, 2.5)
            + 0.10 * clip01(latest["atr_pct"], 0.015, 0.09)
            + 0.08 * clip01(latest["bb_pos"], 0.45, 0.95)
        )

        reversal = (
            0.22 * (1 - clip01(latest["rsi14"], 30, 55))
            + 0.20 * latest["lower_band_reclaim"].fillna(0)
            + 0.16 * clip01(latest["ret_1d"], -0.01, 0.05)
            + 0.14 * clip01(latest["rel_volume"], 0.8, 2.8)
            + 0.14 * (1 - clip01(latest["bb_pos"], 0.15, 0.55))
            + 0.14 * (1 - clip01(latest["drawdown_20h"], -0.18, -0.03))
        )

        latest["momentum_score"] = momentum.clip(0, 1)
        latest["reversal_score"] = reversal.clip(0, 1)
        latest["setup_type"] = np.where(latest["momentum_score"] >= latest["reversal_score"], "momentum", "reversal")
        latest["setup_score"] = latest[["momentum_score", "reversal_score"]].max(axis=1)
        latest["trade_confidence"] = latest["setup_score"].clip(0, 1)
        latest = latest.sort_values(["setup_score", "dollar_volume20"], ascending=[False, False])
        return latest.reset_index(drop=True)

    def _update_high_water_and_exits(self, scores: pd.DataFrame, equity: float) -> pd.DataFrame:
        state = self._load_state()
        high_plpc: dict[str, float] = {k: to_float(v) for k, v in state.get("high_plpc", {}).items()}
        score_by_symbol = dict(zip(scores.get("symbol", []), scores.get("setup_score", [])))
        top_symbols = set(scores.head(self.config.top_n).get("symbol", []))

        rows: list[dict[str, Any]] = []
        for pos in self.broker.get_all_open_positions():
            symbol = normalize_symbol(pos.symbol)
            if pos.qty <= 0:
                continue

            old_high = high_plpc.get(symbol, pos.unrealized_plpc)
            new_high = max(old_high, pos.unrealized_plpc)
            high_plpc[symbol] = new_high

            reason = ""
            if pos.unrealized_plpc <= self.config.stop_loss_pct:
                reason = "scanner_stop_loss_negative_position"
            elif new_high >= self.config.profit_lock_trigger_pct and pos.unrealized_plpc <= new_high - self.config.profit_giveback_pct:
                reason = "scanner_profit_lock_giveback_exit"
            elif symbol not in top_symbols and to_float(score_by_symbol.get(symbol), 0.0) < self.config.stale_position_score_exit:
                reason = "scanner_stale_position_no_longer_high_conviction"

            if reason:
                rows.append(
                    {
                        "symbol": symbol,
                        "side": "sell",
                        "delta_shares": int(pos.qty),
                        "delta_dollars": float(pos.market_value),
                        "price": float(pos.current_price),
                        "reason": reason,
                        "target_weight": 0.0,
                        "current_weight": float(pos.market_value) / max(equity, 1e-9),
                        "delta_weight": -float(pos.market_value) / max(equity, 1e-9),
                        "trade_confidence": 1.0,
                    }
                )
                high_plpc.pop(symbol, None)

        state["high_plpc"] = high_plpc
        state["updated_at"] = now_utc().isoformat()
        self._save_state(state)
        return pd.DataFrame(rows)


    def _existing_option_underlyings(self) -> set[str]:
        underlyings: set[str] = set()
        for pos in self.broker.get_all_open_positions():
            parsed = parse_occ_option_symbol(pos.symbol)
            if parsed and pos.qty > 0:
                underlyings.add(str(parsed["underlying"]))
        return underlyings

    def _existing_option_positions_by_underlying(self) -> dict[str, list[BrokerPosition]]:
        result: dict[str, list[BrokerPosition]] = {}
        for pos in self.broker.get_all_open_positions():
            parsed = parse_occ_option_symbol(pos.symbol)
            if parsed and pos.qty > 0:
                u = str(parsed["underlying"])
                result.setdefault(u, []).append(pos)
        return result

    def _options_exposure(self, equity: float) -> float:
        total_mv = sum(
            pos.market_value
            for pos in self.broker.get_all_open_positions()
            if parse_occ_option_symbol(pos.symbol) and pos.qty > 0
        )
        return total_mv / max(equity, 1e-9)

    def _open_option_position_count(self) -> int:
        count = 0
        for pos in self.broker.get_all_open_positions():
            parsed = parse_occ_option_symbol(pos.symbol)
            if parsed and pos.qty > 0:
                count += 1
        return count

    def _select_best_contract(
        self,
        underlying: str,
        underlying_price: float,
        momentum_score: float,
        option_type: str = "call",
        pcr: float | None = None,
        vol_oi_ratio: float | None = None,
    ) -> dict[str, Any] | None:
        cfg = self.options_config
        try:
            contracts = self.chain_discovery.discover_contracts_for_underlying(
                underlying=underlying,
                min_dte=cfg.min_dte,
                max_dte=cfg.max_dte,
                option_type=option_type,
                feed=cfg.feed,
            )
        except Exception as exc:
            print(f"[options] chain discovery failed for {underlying}: {exc}")
            return None

        if not contracts:
            return None

        for c in contracts:
            c["underlying_price"] = underlying_price
            if self.option_stream is not None:
                live_quote = self.option_stream.get_quote(c.get("symbol", ""))
                if live_quote and live_quote.get("bid", 0) > 0 and live_quote.get("ask", 0) > 0:
                    c["bid"] = live_quote["bid"]
                    c["ask"] = live_quote["ask"]
                    c["bid_size"] = live_quote.get("bid_size", c.get("bid_size", 0))
                    c["ask_size"] = live_quote.get("ask_size", c.get("ask_size", 0))
                    c["mid"] = (c["bid"] + c["ask"]) / 2.0
                    c["spread_pct"] = (c["ask"] - c["bid"]) / max(c["mid"], 1e-9)

        ranked = self.ranker.rank(
            contracts=contracts,
            underlying_momentum_score=momentum_score,
            pcr=pcr,
            vol_oi_ratio=vol_oi_ratio,
            option_type=option_type,
        )

        if ranked.empty:
            return None

        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            ranked.to_csv(OPTION_CANDIDATES_PATH, index=False)
        except Exception:
            pass

        return ranked.iloc[0].to_dict()

    def _build_options_trading_orders(self, scores: pd.DataFrame, equity: float) -> pd.DataFrame:
        """Options-first entry orders. No underlying stock position required."""
        cfg = self.options_config
        if not cfg.enabled:
            return pd.DataFrame()
        if scores.empty:
            return pd.DataFrame()
        if self._open_option_position_count() >= cfg.max_open_option_positions:
            print("[options] max open option positions reached")
            return pd.DataFrame()
        if self._options_exposure(equity) >= cfg.max_options_exposure:
            print(f"[options] options exposure >= {cfg.max_options_exposure:.0%} limit")
            return pd.DataFrame()

        existing_by_underlying = self._existing_option_positions_by_underlying()
        rows: list[dict[str, Any]] = []

        candidate_scores = scores.head(cfg.max_open_option_positions * 2).copy()
        candidate_scores["symbol"] = candidate_scores["symbol"].map(normalize_symbol)
        candidate_scores = candidate_scores.sort_values("setup_score", ascending=False)

        for _, r in candidate_scores.iterrows():
            if self._open_option_position_count() >= cfg.max_open_option_positions:
                break

            underlying = normalize_symbol(r["symbol"])
            if underlying in existing_by_underlying:
                continue

            underlying_price = to_float(r.get("close", 0.0))
            momentum_score = to_float(r.get("momentum_score", r.get("setup_score", 0.0)))
            reversal_score = to_float(r.get("reversal_score", 0.0))

            pcr: float | None = None
            vol_oi_ratio: float | None = None
            if self.av_client is not None:
                try:
                    pcr = self.av_client.get_put_call_ratio(underlying)
                    vol_oi_ratio = self.av_client.get_volume_oi_ratio(underlying)
                except Exception:
                    pass

            trade_calls = cfg.enable_long_calls
            trade_puts = cfg.enable_long_puts

            if pcr is not None:
                if cfg.pcr_bearish_threshold <= pcr < cfg.pcr_contrarian_bullish_threshold:
                    trade_calls = False
                if pcr <= cfg.pcr_bullish_threshold:
                    trade_puts = False
            elif reversal_score > momentum_score * 1.2:
                trade_calls = False

            for opt_type in (["call"] if trade_calls else []) + (["put"] if trade_puts else []):
                if self._open_option_position_count() >= cfg.max_open_option_positions:
                    break
                if underlying in self._existing_option_positions_by_underlying():
                    break

                best = self._select_best_contract(
                    underlying=underlying,
                    underlying_price=underlying_price,
                    momentum_score=momentum_score,
                    option_type=opt_type,
                    pcr=pcr,
                    vol_oi_ratio=vol_oi_ratio,
                )
                if not best:
                    continue

                mid = to_float(best.get("mid", 0.0))
                if mid <= 0:
                    continue

                contract_notional = mid * 100.0
                risk_dollars = equity * cfg.risk_per_trade_pct
                n_contracts = max(1, int(risk_dollars // max(contract_notional, 1e-9)))
                n_contracts = min(n_contracts, cfg.max_contracts_per_underlying)

                exposure_room = (cfg.max_options_exposure - self._options_exposure(equity)) * equity
                if contract_notional * n_contracts > exposure_room:
                    n_contracts = int(exposure_room // max(contract_notional, 1e-9))
                if n_contracts <= 0:
                    continue

                option_symbol = str(best.get("symbol", ""))
                total_notional = contract_notional * n_contracts
                rows.append({
                    "symbol": option_symbol,
                    "side": "buy",
                    "delta_shares": n_contracts,
                    "delta_dollars": total_notional,
                    "price": mid,
                    "reason": f"options_trading_long_{opt_type}",
                    "target_weight": total_notional / max(equity, 1e-9),
                    "current_weight": 0.0,
                    "delta_weight": total_notional / max(equity, 1e-9),
                    "trade_confidence": momentum_score,
                    "option_type": opt_type,
                    "underlying_symbol": underlying,
                    "contract_score": to_float(best.get("contract_score", 0.0)),
                    "option_delta": best.get("delta"),
                    "option_dte": best.get("dte"),
                    "option_strike": best.get("strike"),
                    "option_spread_pct": best.get("spread_pct"),
                    "option_iv": best.get("iv"),
                    "put_call_ratio": pcr,
                    "volume_oi_ratio": vol_oi_ratio,
                    "is_option": True,
                })

        if rows:
            print(f"[options] generated trading orders={len(rows)}")
        return pd.DataFrame(rows)

    def _build_option_exit_orders(self, equity: float) -> pd.DataFrame:
        """Exit orders for all open long option positions (calls and puts)."""
        cfg = self.options_config
        if not cfg.enabled:
            return pd.DataFrame()

        state = self._load_state()
        high_plpc: dict[str, float] = {k: to_float(v) for k, v in state.get("option_high_plpc", {}).items()}
        rows: list[dict[str, Any]] = []

        for pos in self.broker.get_all_open_positions():
            parsed = parse_occ_option_symbol(pos.symbol)
            if not parsed or pos.qty <= 0:
                continue

            option_type = str(parsed.get("type", "call"))
            symbol = safe_symbol(pos.symbol)
            old_high = high_plpc.get(symbol, pos.unrealized_plpc)
            new_high = max(old_high, pos.unrealized_plpc)
            high_plpc[symbol] = new_high

            today = now_utc().date()
            dte = (parsed["expiration"] - today).days

            reason = ""
            if dte <= cfg.close_before_expiration_days:
                reason = f"option_{option_type}_expiration_protection"
            elif pos.unrealized_plpc <= cfg.hard_stop_pct:
                reason = f"option_{option_type}_hard_stop"
            elif pos.unrealized_plpc >= cfg.take_profit_pct:
                reason = f"option_{option_type}_take_profit"
            elif new_high >= cfg.profit_lock_trigger_pct and pos.unrealized_plpc <= new_high - cfg.profit_giveback_pct:
                reason = f"option_{option_type}_profit_lock_giveback"

            if reason:
                rows.append({
                    "symbol": symbol,
                    "side": "sell",
                    "delta_shares": int(pos.qty),
                    "delta_dollars": float(pos.market_value),
                    "price": float(pos.current_price),
                    "reason": reason,
                    "target_weight": 0.0,
                    "current_weight": float(pos.market_value) / max(equity, 1e-9),
                    "delta_weight": -float(pos.market_value) / max(equity, 1e-9),
                    "trade_confidence": 1.0,
                    "underlying_symbol": parsed.get("underlying"),
                    "option_type": option_type,
                    "option_plpc": float(pos.unrealized_plpc),
                    "option_high_plpc": float(new_high),
                    "option_expiration": parsed["expiration"].isoformat(),
                    "option_dte": int(dte),
                    "is_option": True,
                })
                high_plpc.pop(symbol, None)

        state["option_high_plpc"] = high_plpc
        state["updated_at"] = now_utc().isoformat()
        self._save_state(state)
        return pd.DataFrame(rows)

    def build_orders(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        symbols = self._universe()
        print(f"[scanner] scanning universe symbols={len(symbols)}")
        bars = self._download_bars(symbols)
        if bars.empty:
            print("[scanner] no bars downloaded; keeping existing order files")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        features = self._build_features(bars)
        scores = self._score_latest(features)
        if scores.empty:
            print("[scanner] no symbols passed scanner liquidity/price filters")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        scores.to_csv(SCANNER_SCORES_PATH, index=False)
        print("[scanner] top setups:")
        show_cols = ["symbol", "close", "setup_type", "setup_score", "momentum_score", "reversal_score", "rsi14", "ret_5d", "rel_volume"]
        print(scores[[c for c in show_cols if c in scores.columns]].head(self.config.top_n).to_string(index=False))

        acct = self.broker.get_account_snapshot()
        equity = max(float(acct.equity), 1.0)
        positions = {normalize_symbol(p.symbol): p for p in self.broker.get_all_open_positions()}

        candidates = scores[
            (scores["setup_score"] >= self.config.min_setup_score)
            & (scores["trade_confidence"] >= self.config.min_buy_confidence)
        ].head(self.config.top_n).copy()

        rows: list[dict[str, Any]] = []
        for _, r in candidates.iterrows():
            symbol = normalize_symbol(r["symbol"])
            price = to_float(r["close"])
            if price <= 0:
                continue

            current_mv = positions.get(symbol).market_value if symbol in positions else 0.0
            current_weight = float(current_mv) / equity
            type_mult = self.config.reversal_target_weight_mult if r["setup_type"] == "reversal" else 1.0
            target_weight = min(
                self.config.max_target_weight,
                self.config.base_target_weight * (0.50 + 1.50 * float(r["setup_score"])) * type_mult,
                self.executor_config.max_position_weight,
            )
            delta_weight = target_weight - current_weight
            delta_dollars = delta_weight * equity
            if delta_dollars < self.executor_config.min_notional_to_trade:
                continue

            rows.append(
                {
                    "symbol": symbol,
                    "side": "buy",
                    "delta_shares": int(delta_dollars // price),
                    "delta_dollars": float(delta_dollars),
                    "price": price,
                    "reason": f"scanner_{r['setup_type']}_high_conviction_large_gain_candidate",
                    "target_weight": target_weight,
                    "current_weight": current_weight,
                    "delta_weight": delta_weight,
                    "trade_confidence": float(r["trade_confidence"]),
                    "confidence_score": float(r["setup_score"]),
                    "momentum_score": float(r["momentum_score"]),
                    "reversal_score": float(r["reversal_score"]),
                    "rsi14": to_float(r.get("rsi14")),
                    "ret_5d": to_float(r.get("ret_5d")),
                    "rel_volume": to_float(r.get("rel_volume")),
                }
            )

        rebalance = pd.DataFrame(rows)
        option_orders = self._build_options_trading_orders(scores, equity)
        if not option_orders.empty:
            rebalance = pd.concat([rebalance, option_orders], ignore_index=True)

        exits = self._update_high_water_and_exits(scores, equity)
        option_exits = self._build_option_exit_orders(equity)
        if not option_exits.empty:
            exits = pd.concat([exits, option_exits], ignore_index=True)

        rebalance.to_csv(DEFAULT_REBALANCE_ORDERS, index=False)
        exits.to_csv(DEFAULT_EXIT_ORDERS, index=False)
        print(f"[scanner] wrote rebalance_orders={DEFAULT_REBALANCE_ORDERS} rows={len(rebalance)}")
        print(f"[scanner] wrote exit_orders={DEFAULT_EXIT_ORDERS} rows={len(exits)}")
        return rebalance, exits, scores

# =============================================================================
# Executor
# =============================================================================


def _account_size_band(equity: float) -> str:
    if equity < SMALL_ACCOUNT_THRESHOLD:
        return "SMALL"
    if equity < MID_ACCOUNT_THRESHOLD:
        return "MID"
    if equity < LARGE_ACCOUNT_THRESHOLD:
        return "LARGE"
    return "INSTITUTIONAL"


def apply_dynamic_account_scaling(config: ExecutorConfig, equity: float) -> ExecutorConfig:
    """Clamp execution risk limits to the live Alpaca account size.

    This function is intentionally conservative: it only tightens dangerous
    limits. User-supplied CLI values still work, but account scaling prevents a
    small account from inheriting $100K-sized risk limits such as $15K orders or
    a $95K kill-switch floor.
    """
    equity = max(float(equity or 0.0), 1.0)

    if not config.enable_account_scaling:
        return config

    if config.account_scale_reference_equity <= 0:
        config.account_scale_reference_equity = equity

    band = _account_size_band(equity)

    if band == "SMALL":
        max_gross = SMALL_ACCOUNT_MAX_GROSS_EXPOSURE
        max_pos = SMALL_ACCOUNT_MAX_POSITION_WEIGHT
        max_order_pct = SMALL_ACCOUNT_MAX_ORDER_PCT
        max_daily_loss_pct = SMALL_ACCOUNT_DAILY_LOSS_PCT
        scanner_base_weight = 0.035
        scanner_max_weight = 0.10
        scanner_top_n = 5
        min_notional_pct = 0.0075
        min_bp_pct = 0.015
    elif band == "MID":
        max_gross = MID_ACCOUNT_MAX_GROSS_EXPOSURE
        max_pos = MID_ACCOUNT_MAX_POSITION_WEIGHT
        max_order_pct = MID_ACCOUNT_MAX_ORDER_PCT
        max_daily_loss_pct = MID_ACCOUNT_DAILY_LOSS_PCT
        scanner_base_weight = 0.04
        scanner_max_weight = 0.12
        scanner_top_n = 6
        min_notional_pct = 0.0060
        min_bp_pct = 0.0125
    else:
        max_gross = LARGE_ACCOUNT_MAX_GROSS_EXPOSURE
        max_pos = LARGE_ACCOUNT_MAX_POSITION_WEIGHT
        max_order_pct = LARGE_ACCOUNT_MAX_ORDER_PCT
        max_daily_loss_pct = LARGE_ACCOUNT_DAILY_LOSS_PCT
        scanner_base_weight = 0.04
        scanner_max_weight = 0.14
        scanner_top_n = config.scanner.top_n if config.scanner else 8
        min_notional_pct = 0.0040
        min_bp_pct = 0.0100

    config.cash_buffer_pct = min(config.cash_buffer_pct, DYNAMIC_CASH_BUFFER_PCT)
    config.min_cash_buffer = min(
        config.min_cash_buffer,
        max(MIN_DYNAMIC_CASH_BUFFER, equity * 0.04),
    )
    config.min_buying_power = min(
        config.min_buying_power,
        max(100.0, equity * min_bp_pct),
    )
    config.min_notional_to_trade = min(
        config.min_notional_to_trade,
        max(35.0, equity * min_notional_pct),
    )

    config.max_gross_exposure = min(config.max_gross_exposure, max_gross)
    config.max_position_weight = min(config.max_position_weight, max_pos)
    config.max_single_order_notional_pct = min(config.max_single_order_notional_pct, max_order_pct)
    config.max_order_dollars = min(config.max_order_dollars, max(100.0, equity * max_order_pct))

    config.max_daily_loss_pct = min(config.max_daily_loss_pct, max_daily_loss_pct)
    config.max_daily_loss_dollars = min(
        config.max_daily_loss_dollars,
        max(125.0, equity * config.max_daily_loss_pct),
    )

    # Reference-equity floor prevents the old hard-coded $95K floor from
    # instantly blocking smaller accounts, while still creating a hard account
    # drawdown stop from the first observed equity in this process.
    config.kill_switch_equity_floor = min(
        config.kill_switch_equity_floor,
        config.account_scale_reference_equity * (1.0 - ACCOUNT_KILL_FLOOR_DRAWDOWN_PCT),
    )

    if config.scanner is not None:
        config.scanner.top_n = min(config.scanner.top_n, scanner_top_n)
        config.scanner.base_target_weight = min(config.scanner.base_target_weight, scanner_base_weight)
        config.scanner.max_target_weight = min(
            config.scanner.max_target_weight,
            scanner_max_weight,
            config.max_position_weight,
        )

    print(
        "[account-scaling] "
        f"band={band} equity=${equity:,.2f} "
        f"cash_buffer=${max(config.min_cash_buffer, equity * config.cash_buffer_pct):,.2f} "
        f"max_gross={config.max_gross_exposure:.2%} "
        f"max_position={config.max_position_weight:.2%} "
        f"max_order=${config.max_order_dollars:,.2f} "
        f"daily_loss_limit=${config.max_daily_loss_dollars:,.2f}/{config.max_daily_loss_pct:.2%} "
        f"kill_floor=${config.kill_switch_equity_floor:,.2f}"
    )
    write_event(
        "account_scaling_applied",
        {
            "band": band,
            "equity": equity,
            "reference_equity": config.account_scale_reference_equity,
            "cash_buffer_pct": config.cash_buffer_pct,
            "min_cash_buffer": config.min_cash_buffer,
            "min_buying_power": config.min_buying_power,
            "min_notional_to_trade": config.min_notional_to_trade,
            "max_gross_exposure": config.max_gross_exposure,
            "max_position_weight": config.max_position_weight,
            "max_single_order_notional_pct": config.max_single_order_notional_pct,
            "max_order_dollars": config.max_order_dollars,
            "max_daily_loss_pct": config.max_daily_loss_pct,
            "max_daily_loss_dollars": config.max_daily_loss_dollars,
            "kill_switch_equity_floor": config.kill_switch_equity_floor,
        },
    )
    return config


class ProductionOrderExecutor:
    def __init__(self, config: ExecutorConfig):
        self.config = config
        self.journal = CsvJournal(JOURNAL_PATH)
        self.broker = AlpacaBroker(ENV_PATH, paper=config.paper)
        try:
            acct = self.broker.get_account_snapshot()
            self.config = apply_dynamic_account_scaling(self.config, acct.equity)
        except Exception as exc:
            print(f"[account-scaling] unable to fetch account equity during init: {exc}")
        if self.config.scanner is None:
            self.config.scanner = ScannerConfig()
        self.scanner = MomentumSetupScanner(self.broker, self.config.scanner, self.config) if self.config.scanner.enabled else None

        self.last_order_time: float = 0.0
        self.last_heartbeat_time: float = 0.0
        self.account_history: list[dict[str, float | str]] = []

    # -------------------------------------------------------------------------
    # State helpers
    # -------------------------------------------------------------------------

    def _in_cooldown(self) -> bool:
        return (time.time() - self.last_order_time) < self.config.cooldown_seconds

    def _mark_order(self) -> None:
        self.last_order_time = time.time()

    def _record_account_snapshot(self, label: str, cycle_id: str) -> dict[str, float | str]:
        acct = self.broker.get_account_snapshot()
        snapshot = {
            "label": label,
            "cycle_id": cycle_id,
            "timestamp_utc": now_utc().isoformat(),
            "buying_power": float(acct.buying_power),
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "portfolio_value": float(acct.portfolio_value),
        }
        self.account_history.append(snapshot)
        return snapshot

    def _write_account_equity_curve(self) -> None:
        if self.account_history:
            pd.DataFrame(self.account_history).to_csv(EQUITY_CURVE_PATH, index=False)

    def _positions_by_symbol(self) -> dict[str, BrokerPosition]:
        return {safe_symbol(p.symbol): p for p in self.broker.get_all_open_positions() if p.symbol}

    def _gross_exposure(self, equity: float) -> float:
        if equity <= 0:
            return 0.0
        return float(
            sum(max(0.0, float(p.market_value)) for p in self.broker.get_all_open_positions())
            / equity
        )

    def _current_weight(self, symbol: str, equity: float) -> float:
        if equity <= 0:
            return 0.0
        pos = self._positions_by_symbol().get(safe_symbol(symbol))
        if pos is None:
            return 0.0
        return float(pos.market_value) / equity

    def _cash_buffer(self, equity: float) -> float:
        return max(self.config.min_cash_buffer, equity * self.config.cash_buffer_pct)

    def _max_order_notional(self, equity: float) -> float:
        return min(
            self.config.max_order_dollars,
            equity * self.config.max_single_order_notional_pct,
        )

    # -------------------------------------------------------------------------
    # Market clock / kill switch
    # -------------------------------------------------------------------------

    def _market_is_open_or_wait(self) -> bool:
        if self.config.allow_closed_market or not self.config.require_market_open:
            print("[clock] market-open guard disabled.")
            return True

        info = self.broker.get_clock_info()
        print(
            "[clock] "
            f"is_open={info['is_open']} "
            f"timestamp={info['timestamp']} "
            f"next_open={info['next_open']} "
            f"next_close={info['next_close']}"
        )
        write_event("market_clock_checked", info)

        if info["is_open"]:
            return True

        if not self.config.wait_for_market_open:
            print("[clock] market is closed. Execution skipped.")
            return False

        started = time.time()
        max_wait = self.config.max_wait_for_open_minutes * 60

        while time.time() - started < max_wait:
            time.sleep(max(5, self.config.poll_seconds))
            info = self.broker.get_clock_info()
            print(f"[clock] waiting | is_open={info['is_open']} next_open={info['next_open']}")
            write_event("market_clock_wait_poll", info)
            if info["is_open"]:
                return True

        print("[clock] market did not open within wait limit. Execution skipped.")
        return False

    def _read_kill_state(self) -> dict[str, Any]:
        if not KILL_SWITCH_PATH.exists():
            return {}
        try:
            return json.loads(KILL_SWITCH_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_kill_state(self, state: dict[str, Any]) -> None:
        KILL_SWITCH_PATH.write_text(json.dumps(state, indent=2, default=json_default) + "\n", encoding="utf-8")

    def _new_kill_state(self, equity: float, reason: str = "") -> dict[str, Any]:
        return {
            "date": now_utc().date().isoformat(),
            "start_equity": float(equity),
            "reference_equity": float(getattr(self.config, "account_scale_reference_equity", equity) or equity),
            "kill_triggered": False,
            "triggered_at": "",
            "reason": "",
            "reset_reason": reason,
            "updated_at": now_utc().isoformat(),
        }

    def _kill_switch_allows_execution(self, equity: float) -> tuple[bool, str, float]:
        if not self.config.enable_kill_switch:
            return True, "kill_switch_disabled", 0.0

        equity = max(float(equity or 0.0), 1.0)
        today = now_utc().date().isoformat()
        state = self._read_kill_state()

        if state.get("date") != today:
            state = self._new_kill_state(equity, reason="new_trading_day")
            self._write_kill_state(state)
        else:
            saved_start_equity = to_float(state.get("start_equity"), equity)
            if saved_start_equity > 0:
                equity_gap_pct = abs(equity / saved_start_equity - 1.0)
            else:
                equity_gap_pct = 0.0

            # This protects account switches/deposits/withdrawals. Example:
            # old kill state start_equity=$100,000, new account equity=$13,500.
            # Without this reset, the new account falsely looks down -86.5%.
            if equity_gap_pct >= KILL_STATE_ACCOUNT_SWITCH_RESET_PCT:
                reset_payload = {
                    "old_start_equity": saved_start_equity,
                    "new_equity": equity,
                    "equity_gap_pct": equity_gap_pct,
                    "old_state": state,
                }
                print(
                    "[kill-switch] resetting stale state after account equity change | "
                    f"old_start=${saved_start_equity:,.2f} new_equity=${equity:,.2f} "
                    f"gap={equity_gap_pct:.2%}"
                )
                write_event("kill_switch_state_reset_account_change", reset_payload)
                state = self._new_kill_state(equity, reason="account_equity_changed_substantially")
                self._write_kill_state(state)

        start_equity = to_float(state.get("start_equity"), equity)
        pnl = equity - start_equity
        dd_pct = pnl / max(start_equity, 1e-9)

        reason = "ok"
        blocked = False

        if state.get("kill_triggered"):
            blocked = True
            reason = str(state.get("reason", "kill_switch_already_triggered"))

        if equity <= self.config.kill_switch_equity_floor:
            blocked = True
            reason = "kill_switch_equity_floor"

        if dd_pct <= self.config.daily_drawdown_limit:
            blocked = True
            reason = "daily_drawdown_limit"

        if pnl <= -abs(self.config.max_daily_loss_dollars):
            blocked = True
            reason = "max_daily_loss_dollars"

        if (-dd_pct) >= abs(self.config.max_daily_loss_pct):
            blocked = True
            reason = "max_daily_loss_pct"

        print(
            f"[kill-switch] start_equity=${start_equity:,.2f} equity=${equity:,.2f} "
            f"daily_pnl=${pnl:,.2f} daily_return={dd_pct:.2%} status={reason}"
        )

        if blocked:
            state.update(
                {
                    "kill_triggered": True,
                    "triggered_at": now_utc().isoformat(),
                    "reason": reason,
                    "current_equity": equity,
                    "daily_pnl": pnl,
                    "daily_return": dd_pct,
                }
            )
            self._write_kill_state(state)
            write_event("kill_switch_triggered", state)
            return False, reason, dd_pct

        return True, reason, dd_pct

    # -------------------------------------------------------------------------
    # Order loading / normalization
    # -------------------------------------------------------------------------

    def _safe_read_order_csv(self, path: Path, required: bool = False) -> pd.DataFrame:
        # Read an order CSV safely.
        # Empty CSVs are common when there are no exit orders. Pandas raises
        # EmptyDataError on a zero-byte file, so return an empty DataFrame instead.
        if not path.exists():
            if required:
                raise RuntimeError(f"Required order file not found: {path}. Run trade.py first.")
            return pd.DataFrame()

        try:
            if path.stat().st_size == 0:
                return pd.DataFrame()
        except Exception:
            pass

        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def _load_orders(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        rebalance = self._safe_read_order_csv(
            self.config.rebalance_orders_path,
            required=True,
        )

        exits = self._safe_read_order_csv(
            self.config.exit_orders_path,
            required=False,
        )

        return rebalance, exits

    def _normalize_order_frame(self, orders: pd.DataFrame | None, default_reason: str = "") -> pd.DataFrame:
        cols = [
            "symbol",
            "side",
            "delta_shares",
            "delta_dollars",
            "price",
            "reason",
            "target_weight",
            "current_weight",
            "delta_weight",
            "trade_confidence",
            "confidence",
            "confidence_score",
            "continuation_prob",
            "continuation_probability",
            "explosive_alpha_probability",
        ]

        if orders is None or orders.empty:
            return pd.DataFrame(columns=cols)

        out = orders.copy()

        defaults = {
            "symbol": "",
            "side": "",
            "delta_shares": 0.0,
            "delta_dollars": 0.0,
            "price": 0.0,
            "reason": default_reason,
            "target_weight": np.nan,
            "current_weight": np.nan,
            "delta_weight": np.nan,
            "trade_confidence": np.nan,
            "confidence": np.nan,
            "confidence_score": np.nan,
            "continuation_prob": np.nan,
            "continuation_probability": np.nan,
            "explosive_alpha_probability": np.nan,
        }

        for col, default in defaults.items():
            if col not in out.columns:
                out[col] = default

        out["symbol"] = out["symbol"].map(safe_symbol)
        out["side"] = out["side"].astype(str).str.lower().str.strip()

        numeric_cols = [
            "delta_shares",
            "delta_dollars",
            "price",
            "target_weight",
            "current_weight",
            "delta_weight",
            "trade_confidence",
            "confidence",
            "confidence_score",
            "continuation_prob",
            "continuation_probability",
            "explosive_alpha_probability",
        ]

        for col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        out["reason"] = out["reason"].fillna(default_reason).astype(str)
        out = out[(out["symbol"] != "") & out["side"].isin(["buy", "sell"])].copy()

        return out.reset_index(drop=True)

    def _confidence_for_row(self, row: pd.Series) -> float:
        for col in [
            "trade_confidence",
            "confidence",
            "confidence_score",
            "continuation_prob",
            "continuation_probability",
            "explosive_alpha_probability",
        ]:
            value = row.get(col, np.nan)
            if pd.notna(value):
                val = to_float(value, 0.5)
                if val > 1:
                    val /= 100.0
                return float(np.clip(val, 0.0, 1.0))
        return 0.5

    def _confidence_multiplier(self, confidence: float) -> float:
        return float(np.clip(0.50 + confidence, 0.35, 1.25))

    def _build_order_plan(self, rebalance: pd.DataFrame, exits: pd.DataFrame, equity: float) -> pd.DataFrame:
        rb = self._normalize_order_frame(rebalance)
        ex = self._normalize_order_frame(exits, default_reason="removed_from_high_conviction_universe")

        plan = pd.concat([ex, rb], ignore_index=True)

        if plan.empty:
            return pd.DataFrame()

        plan["abs_delta_shares"] = plan["delta_shares"].abs()
        plan["abs_delta_dollars"] = plan["delta_dollars"].abs()

        plan = plan[
            (plan["side"] == "sell")
            | (plan["abs_delta_dollars"] >= self.config.min_notional_to_trade)
        ].copy()

        if plan.empty:
            return pd.DataFrame()

        plan["confidence_used"] = plan.apply(self._confidence_for_row, axis=1)
        plan["confidence_multiplier"] = plan["confidence_used"].map(self._confidence_multiplier)

        buy_mask = plan["side"] == "buy"
        option_booster_mask = plan.get("reason", pd.Series("", index=plan.index)).astype(str).str.contains("option_booster", case=False, na=False)
        # Stock orders get adaptive confidence scaling. Option booster orders are
        # already sized by explicit premium-at-risk, so do not multiply them again.
        plan.loc[buy_mask & ~option_booster_mask, "abs_delta_dollars"] *= plan.loc[buy_mask & ~option_booster_mask, "confidence_multiplier"]

        plan["order_qty"] = plan["abs_delta_shares"].fillna(0).astype(int)

        missing_qty = (plan["order_qty"] <= 0) & (plan["price"] > 0)
        plan.loc[missing_qty, "order_qty"] = (
            plan.loc[missing_qty, "abs_delta_dollars"] / plan.loc[missing_qty, "price"]
        ).fillna(0).astype(int)

        max_order_notional = self._max_order_notional(equity)

        plan.loc[buy_mask, "capped_dollars"] = plan.loc[buy_mask, "abs_delta_dollars"].clip(
            upper=max_order_notional
        )
        plan.loc[~buy_mask, "capped_dollars"] = plan.loc[~buy_mask, "abs_delta_dollars"]

        plan.loc[buy_mask & (plan["price"] > 0), "order_qty"] = (
            plan.loc[buy_mask & (plan["price"] > 0), "capped_dollars"]
            / plan.loc[buy_mask & (plan["price"] > 0), "price"]
        ).fillna(0).astype(int)

        plan = plan[plan["order_qty"] > 0].copy()
        plan = plan[plan["order_qty"] <= self.config.max_single_order_qty].copy()

        if plan.empty:
            return pd.DataFrame()

        sell_first = getattr(self.config, "sell_first", True)
        side_rank = {"sell": 0, "buy": 1} if sell_first else {"buy": 0, "sell": 1}
        plan["side_rank"] = plan["side"].map(side_rank).fillna(9)
        plan = plan.sort_values(["side_rank", "abs_delta_dollars"], ascending=[True, False]).reset_index(drop=True)

        stamp = run_id()
        plan["client_order_id"] = [
            f"{self.config.order_tag_prefix}_{stamp}_{idx}_{row.symbol}_{row.side}"[:48]
            for idx, row in plan.iterrows()
        ]
        plan["planned_at"] = now_utc().isoformat()

        return plan

    # -------------------------------------------------------------------------
    # Submission / fills
    # -------------------------------------------------------------------------

    def _poll_fill(self, order_id: str, planned_price: float) -> dict[str, Any]:
        last_order = None

        for _ in range(6):
            try:
                last_order = self.broker.get_order_by_id(order_id)
            except Exception as exc:
                return {
                    "fill_status": "fill_lookup_error",
                    "filled_qty": 0.0,
                    "filled_avg_price": 0.0,
                    "slippage_pct": 0.0,
                    "message": str(exc),
                }

            status = str(getattr(last_order, "status", ""))
            if status.lower() in {"filled", "partially_filled", "canceled", "rejected", "expired"}:
                break

            time.sleep(2.0)

        if last_order is None:
            return {
                "fill_status": "unknown",
                "filled_qty": 0.0,
                "filled_avg_price": 0.0,
                "slippage_pct": 0.0,
            }

        filled_qty = to_float(getattr(last_order, "filled_qty", 0.0))
        filled_avg_price = to_float(getattr(last_order, "filled_avg_price", 0.0))
        status = str(getattr(last_order, "status", ""))

        if planned_price > 0 and filled_avg_price > 0:
            slippage_pct = filled_avg_price / planned_price - 1.0
        else:
            slippage_pct = 0.0

        return {
            "fill_status": status,
            "filled_qty": filled_qty,
            "filled_avg_price": filled_avg_price,
            "slippage_pct": slippage_pct,
        }

    def _journal_order(
        self,
        cycle_id: str,
        event: str,
        order_row: pd.Series | dict[str, Any],
        acct_before: dict[str, float | str],
        acct_after: dict[str, float | str],
        message: str,
        order_id: str = "",
        order_status: str = "",
        fill_info: dict[str, Any] | None = None,
    ) -> None:
        fill_info = fill_info or {}

        if isinstance(order_row, pd.Series):
            row = order_row.to_dict()
        else:
            row = dict(order_row)

        self.journal.append(
            {
                "timestamp_utc": now_utc().isoformat(),
                "cycle_id": cycle_id,
                "event": event,
                "market_open": True,
                "equity_before": acct_before.get("equity", ""),
                "equity_after": acct_after.get("equity", ""),
                "cash_before": acct_before.get("cash", ""),
                "cash_after": acct_after.get("cash", ""),
                "buying_power_before": acct_before.get("buying_power", ""),
                "buying_power_after": acct_after.get("buying_power", ""),
                "gross_exposure_before": acct_before.get("gross_exposure", ""),
                "gross_exposure_after": acct_after.get("gross_exposure", ""),
                "trade_symbol": row.get("symbol", ""),
                "trade_side": str(row.get("side", "")).upper(),
                "trade_qty": row.get("order_qty", ""),
                "trade_price": row.get("price", ""),
                "estimated_notional": row.get("capped_dollars", row.get("abs_delta_dollars", "")),
                "target_weight": row.get("target_weight", ""),
                "current_weight": row.get("current_weight", ""),
                "delta_weight": row.get("delta_weight", ""),
                "confidence": row.get("confidence_used", ""),
                "cash_buffer": self._cash_buffer(float(acct_before.get("equity", 0.0))),
                "max_gross_exposure": self.config.max_gross_exposure,
                "max_position_weight": self.config.max_position_weight,
                "max_order_notional": self._max_order_notional(float(acct_before.get("equity", 0.0))),
                "order_id": order_id,
                "order_status": order_status,
                "filled_qty": fill_info.get("filled_qty", ""),
                "filled_avg_price": fill_info.get("filled_avg_price", ""),
                "slippage_pct": fill_info.get("slippage_pct", ""),
                "message": message,
            }
        )

    def _execute_order_plan(self, plan: pd.DataFrame, cycle_id: str) -> pd.DataFrame:
        if plan.empty:
            return pd.DataFrame()

        results: list[dict[str, Any]] = []

        for _, row in plan.iterrows():
            symbol = safe_symbol(row.get("symbol"))
            side = str(row.get("side")).lower()
            qty = to_int(row.get("order_qty"))
            price = to_float(row.get("price"))
            est_notional = to_float(row.get("capped_dollars", row.get("abs_delta_dollars", 0.0)))
            reason = str(row.get("reason", ""))
            client_order_id = str(row.get("client_order_id"))

            acct_before = self._record_account_snapshot("before_order", cycle_id)
            equity = float(acct_before["equity"])
            cash = float(acct_before["cash"])
            buying_power = float(acct_before["buying_power"])
            gross_before = self._gross_exposure(equity)
            acct_before["gross_exposure"] = gross_before

            if qty <= 0:
                result = {
                    "timestamp_utc": now_utc().isoformat(),
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "status": "skipped",
                    "message": "qty <= 0",
                }
                results.append(result)
                acct_after = self._record_account_snapshot("after_skip", cycle_id)
                acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
                self._journal_order(cycle_id, "SKIP", row, acct_before, acct_after, "qty <= 0")
                continue

            if self.broker.has_open_order_for_symbol(symbol):
                result = {
                    "timestamp_utc": now_utc().isoformat(),
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "status": "skipped_open_order",
                    "message": "existing open order for symbol",
                }
                results.append(result)
                acct_after = self._record_account_snapshot("after_skip", cycle_id)
                acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
                self._journal_order(cycle_id, "SKIP", row, acct_before, acct_after, "existing open order")
                continue

            if side == "buy":
                projected_cash = cash - est_notional
                cash_buffer = self._cash_buffer(equity)
                max_order_notional = self._max_order_notional(equity)
                gross_room = max(0.0, (self.config.max_gross_exposure - gross_before) * equity)

                if buying_power < self.config.min_buying_power:
                    message = "buying_power_below_minimum"
                    print(f"[skip] BUY {qty} {symbol}: {message}")
                    result = {
                        "timestamp_utc": now_utc().isoformat(),
                        "symbol": symbol,
                        "side": side,
                        "qty": qty,
                        "status": "skipped_buying_power",
                        "message": message,
                    }
                    results.append(result)
                    acct_after = self._record_account_snapshot("after_skip", cycle_id)
                    acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
                    self._journal_order(cycle_id, "SKIP", row, acct_before, acct_after, message)
                    continue

                if projected_cash < cash_buffer:
                    message = (
                        f"cash_buffer_violation cash={cash:.2f} projected={projected_cash:.2f} "
                        f"required={cash_buffer:.2f}"
                    )
                    print(f"[skip] BUY {qty} {symbol}: {message}")
                    result = {
                        "timestamp_utc": now_utc().isoformat(),
                        "symbol": symbol,
                        "side": side,
                        "qty": qty,
                        "status": "skipped_cash_buffer",
                        "message": message,
                    }
                    results.append(result)
                    acct_after = self._record_account_snapshot("after_skip", cycle_id)
                    acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
                    self._journal_order(cycle_id, "SKIP", row, acct_before, acct_after, message)
                    continue

                if est_notional > max_order_notional:
                    new_qty = int(max_order_notional // price) if price > 0 else 0
                    print(f"[clip] BUY {symbol}: qty {qty} -> {new_qty} due to max_order_notional")
                    qty = new_qty
                    est_notional = qty * price

                if est_notional > gross_room:
                    new_qty = int(gross_room // price) if price > 0 else 0
                    print(f"[clip] BUY {symbol}: qty {qty} -> {new_qty} due to gross exposure room")
                    qty = new_qty
                    est_notional = qty * price

                if qty <= 0 or est_notional < self.config.min_notional_to_trade:
                    message = "buy_clipped_below_min_notional"
                    print(f"[skip] BUY {symbol}: {message}")
                    result = {
                        "timestamp_utc": now_utc().isoformat(),
                        "symbol": symbol,
                        "side": side,
                        "qty": qty,
                        "status": "skipped_min_notional",
                        "message": message,
                    }
                    results.append(result)
                    acct_after = self._record_account_snapshot("after_skip", cycle_id)
                    acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
                    self._journal_order(cycle_id, "SKIP", row, acct_before, acct_after, message)
                    continue

            try:
                is_option = parse_occ_option_symbol(symbol) is not None
                opts_cfg: OptionsTradingConfig = (
                    self.config.options_booster or OptionsTradingConfig(enabled=False)
                )
                if is_option and not opts_cfg.allow_option_market_orders:
                    if side == "buy":
                        limit_price = round(price + opts_cfg.limit_entry_improvement, 2)
                        submitted = self.broker.submit_limit_buy(
                            symbol, qty, limit_price, client_order_id, self.config.time_in_force
                        )
                    else:
                        limit_price = round(max(price - opts_cfg.limit_exit_offset, 0.01), 2)
                        submitted = self.broker.submit_limit_sell(
                            symbol, qty, limit_price, client_order_id, self.config.time_in_force
                        )
                elif side == "buy":
                    submitted = self.broker.submit_market_buy(symbol, qty, client_order_id, self.config.time_in_force)
                else:
                    submitted = self.broker.submit_market_sell(symbol, qty, client_order_id, self.config.time_in_force)

                order_id = str(getattr(submitted, "id", ""))
                status = str(getattr(submitted, "status", "submitted"))

                print(f"[submitted] {side.upper()} {qty} {symbol} status={status} id={order_id}")
                self._mark_order()

                fill_info = self._poll_fill(order_id, price)
                acct_after = self._record_account_snapshot("after_order", cycle_id)
                acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))

                result = {
                    "timestamp_utc": now_utc().isoformat(),
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "estimated_notional": est_notional,
                    "price": price,
                    "reason": reason,
                    "status": status,
                    "order_id": order_id,
                    **fill_info,
                }
                results.append(result)

                self._journal_order(
                    cycle_id=cycle_id,
                    event="ORDER",
                    order_row={**row.to_dict(), "order_qty": qty, "capped_dollars": est_notional},
                    acct_before=acct_before,
                    acct_after=acct_after,
                    message=reason or "submitted",
                    order_id=order_id,
                    order_status=status,
                    fill_info=fill_info,
                )

                write_event("order_submitted", result)
                time.sleep(0.35)

            except APIError as exc:
                message = str(exc)
                print(f"[error] APIError {side.upper()} {qty} {symbol}: {message}")
                acct_after = self._record_account_snapshot("after_error", cycle_id)
                acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))

                result = {
                    "timestamp_utc": now_utc().isoformat(),
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "status": "api_error",
                    "message": message,
                    "order_id": "",
                }
                results.append(result)
                self._journal_order(cycle_id, "ERROR", row, acct_before, acct_after, message)
                write_event("order_api_error", result)

            except Exception as exc:
                message = str(exc)
                print(f"[error] submit failed {side.upper()} {qty} {symbol}: {message}")
                acct_after = self._record_account_snapshot("after_error", cycle_id)
                acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))

                result = {
                    "timestamp_utc": now_utc().isoformat(),
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "status": "error",
                    "message": message,
                    "order_id": "",
                }
                results.append(result)
                self._journal_order(cycle_id, "ERROR", row, acct_before, acct_after, message)
                write_event("order_error", result)

        return pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # Compatibility wrappers
    # -------------------------------------------------------------------------

    def run_once_from_frames(self, rebalance_orders: pd.DataFrame, exit_orders: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, AccountSnapshot, pd.DataFrame]:
        cycle_id = run_id()

        acct_start = self._record_account_snapshot("cycle_start", cycle_id)
        equity = float(acct_start["equity"])
        self.config = apply_dynamic_account_scaling(self.config, equity)
        cash = float(acct_start["cash"])
        gross = self._gross_exposure(equity)
        acct_start["gross_exposure"] = gross

        positions_before = pd.DataFrame([asdict(p) for p in self.broker.get_all_open_positions()])

        if not self._market_is_open_or_wait():
            results = pd.DataFrame(
                [
                    {
                        "timestamp_utc": now_utc().isoformat(),
                        "status": "skipped_market_closed",
                        "message": "market is closed",
                    }
                ]
            )
            save_df(pd.DataFrame(), LOG_DIR / f"order_plan_{cycle_id}.csv")
            save_df(results, LOG_DIR / f"order_results_{cycle_id}.csv")
            self._write_account_equity_curve()
            return pd.DataFrame(), results, self.broker.get_account_snapshot(), positions_before

        allowed, kill_reason, daily_return = self._kill_switch_allows_execution(equity)
        if not allowed:
            print(f"[kill-switch] execution blocked for new buys: {kill_reason}")

            # Even when the kill switch blocks new risk, allow SELL orders so the
            # system can reduce exposure, honor exits, and clean up positions.
            kill_plan = self._build_order_plan(
                rebalance_orders,
                exit_orders if exit_orders is not None else pd.DataFrame(),
                equity,
            )
            sell_plan = kill_plan[kill_plan["side"] == "sell"].copy() if not kill_plan.empty else pd.DataFrame()

            if not sell_plan.empty:
                print(f"[kill-switch] executing sell-only risk reduction orders={len(sell_plan)}")
                save_df(sell_plan, LOG_DIR / f"order_plan_{cycle_id}.csv")
                results = self._execute_order_plan(sell_plan, cycle_id)
                save_df(results, LOG_DIR / f"order_results_{cycle_id}.csv")
                self._write_account_equity_curve()
                return sell_plan, results, self.broker.get_account_snapshot(), positions_before

            results = pd.DataFrame(
                [
                    {
                        "timestamp_utc": now_utc().isoformat(),
                        "status": "skipped_kill_switch",
                        "message": kill_reason,
                        "daily_return": daily_return,
                    }
                ]
            )
            save_df(pd.DataFrame(), LOG_DIR / f"order_plan_{cycle_id}.csv")
            save_df(results, LOG_DIR / f"order_results_{cycle_id}.csv")
            self._write_account_equity_curve()
            return pd.DataFrame(), results, self.broker.get_account_snapshot(), positions_before

        if self._in_cooldown():
            print("[cooldown] execution blocked by cooldown")
            results = pd.DataFrame(
                [
                    {
                        "timestamp_utc": now_utc().isoformat(),
                        "status": "skipped_cooldown",
                        "message": "cooldown active",
                    }
                ]
            )
            save_df(pd.DataFrame(), LOG_DIR / f"order_plan_{cycle_id}.csv")
            save_df(results, LOG_DIR / f"order_results_{cycle_id}.csv")
            self._write_account_equity_curve()
            return pd.DataFrame(), results, self.broker.get_account_snapshot(), positions_before

        if self.config.cancel_open_orders_first:
            print("[orders] canceling open orders")
            self.broker.cancel_open_orders()

        plan = self._build_order_plan(rebalance_orders, exit_orders if exit_orders is not None else pd.DataFrame(), equity)

        if plan.empty:
            print("[plan] no valid orders")
            results = pd.DataFrame()
            save_df(plan, LOG_DIR / f"order_plan_{cycle_id}.csv")
            save_df(results, LOG_DIR / f"order_results_{cycle_id}.csv")
            self._write_account_equity_curve()
            return plan, results, self.broker.get_account_snapshot(), positions_before

        print("\n===== ORDER PLAN =====")
        display_cols = [
            "symbol",
            "side",
            "order_qty",
            "price",
            "abs_delta_dollars",
            "capped_dollars",
            "confidence_used",
            "confidence_multiplier",
            "reason",
        ]
        print(plan[[c for c in display_cols if c in plan.columns]].to_string(index=False))

        save_df(plan, LOG_DIR / f"order_plan_{cycle_id}.csv")

        results = self._execute_order_plan(plan, cycle_id)

        save_df(results, LOG_DIR / f"order_results_{cycle_id}.csv")

        positions_after = pd.DataFrame([asdict(p) for p in self.broker.get_all_open_positions()])
        save_df(positions_after, LOG_DIR / f"positions_after_{cycle_id}.csv")

        acct_after = self._record_account_snapshot("cycle_end", cycle_id)
        acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
        self._write_account_equity_curve()

        summary = {
            "cycle_id": cycle_id,
            "timestamp_utc": now_utc().isoformat(),
            "paper": self.config.paper,
            "mode": "EXECUTION",
            "equity_before": equity,
            "cash_before": cash,
            "gross_exposure_before": gross,
            "equity_after": acct_after["equity"],
            "cash_after": acct_after["cash"],
            "gross_exposure_after": acct_after["gross_exposure"],
            "orders_planned": len(plan),
            "results_rows": len(results),
        }
        save_json(summary, LOG_DIR / f"execution_summary_{cycle_id}.json")

        return plan, results, self.broker.get_account_snapshot(), positions_before

    def run_cycle_from_files(self) -> tuple[pd.DataFrame, pd.DataFrame, AccountSnapshot, pd.DataFrame]:
        if self.scanner is not None:
            try:
                rebalance, exits, _ = self.scanner.build_orders()
                if not rebalance.empty or not exits.empty:
                    return self.run_once_from_frames(rebalance, exits)
            except Exception as exc:
                print(f"[scanner] cycle failed; falling back to existing order files: {exc}")
                write_event("scanner_cycle_error", {"error": str(exc)})

        print(f"[executor] using rebalance_orders={self.config.rebalance_orders_path}")
        print(f"[executor] using exit_orders={self.config.exit_orders_path if self.config.exit_orders_path.exists() else 'NONE'}")

        rebalance, exits = self._load_orders()
        return self.run_once_from_frames(rebalance, exits)

    def _print_heartbeat(self) -> None:
        now_ts = time.time()
        if now_ts - self.last_heartbeat_time < HEARTBEAT_SECONDS:
            return

        acct = self.broker.get_account_snapshot()
        gross = self._gross_exposure(acct.equity)
        print(
            f"[executor] heartbeat | equity=${acct.equity:,.2f} | "
            f"cash=${acct.cash:,.2f} | buying_power=${acct.buying_power:,.2f} | "
            f"gross={gross:.2%}"
        )
        self.last_heartbeat_time = now_ts

    def run(self) -> None:
        print("[executor] production order executor starting...")
        print(f"[executor] journal={JOURNAL_PATH}")
        print(f"[executor] equity_curve={EQUITY_CURVE_PATH}")
        print(f"[executor] paper={self.config.paper}")
        print(f"[executor] account_scaling={self.config.enable_account_scaling}")
        print(f"[executor] options_mode={bool(self.config.options_booster and self.config.options_booster.enabled)}")
        print("[executor] mode=EXECUTION")
        print("[executor] run_policy=FOREVER_UNTIL_INTERRUPTED")

        if self.config.poll_once:
            self.run_cycle_from_files()
            return

        while True:
            try:
                self.run_cycle_from_files()
                self._print_heartbeat()
            except KeyboardInterrupt:
                print("[executor] interrupted by user")
                raise
            except Exception as exc:
                print(f"[executor] cycle error: {exc}")
                write_event("cycle_error", {"error": str(exc)})

            # Important: closed market, empty order files, no valid orders, or errors
            # must not terminate the process. Sleep and poll again.
            time.sleep(max(1, int(self.config.poll_seconds)))


# =============================================================================
# Backward-compatible module functions
# =============================================================================

def get_trading_client(config: ExecutorConfig | None = None) -> TradingClient:
    config = config or ExecutorConfig()
    return AlpacaBroker(ENV_PATH, paper=config.paper).trading


def get_account_snapshot(client: TradingClient) -> AccountSnapshot:
    acct = client.get_account()
    return AccountSnapshot(
        buying_power=to_float(getattr(acct, "buying_power", 0.0)),
        equity=to_float(getattr(acct, "equity", 0.0)),
        cash=to_float(getattr(acct, "cash", 0.0)),
        portfolio_value=to_float(getattr(acct, "portfolio_value", 0.0)),
    )


def get_positions_df(client: TradingClient) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    try:
        positions = client.get_all_positions()
    except Exception:
        positions = []

    for p in positions:
        rows.append(
            {
                "symbol": safe_symbol(getattr(p, "symbol", "")),
                "qty": to_float(getattr(p, "qty", 0.0)),
                "avg_entry_price": to_float(getattr(p, "avg_entry_price", 0.0)),
                "current_price": to_float(getattr(p, "current_price", 0.0)),
                "market_value": to_float(getattr(p, "market_value", 0.0)),
                "cost_basis": to_float(getattr(p, "cost_basis", 0.0)),
                "unrealized_pl": to_float(getattr(p, "unrealized_pl", 0.0)),
                "unrealized_plpc": to_float(getattr(p, "unrealized_plpc", 0.0)),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "symbol",
            "qty",
            "avg_entry_price",
            "current_price",
            "market_value",
            "cost_basis",
            "unrealized_pl",
            "unrealized_plpc",
        ],
    )


def run_execution(
    rebalance_orders: pd.DataFrame,
    exit_orders: pd.DataFrame | None = None,
    config: ExecutorConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, AccountSnapshot, pd.DataFrame]:
    config = config or ExecutorConfig(poll_once=True)
    config.poll_once = True
    executor = ProductionOrderExecutor(config)
    return executor.run_once_from_frames(rebalance_orders, exit_orders if exit_orders is not None else pd.DataFrame())


def save_execution_results(
    order_plan: pd.DataFrame,
    results: pd.DataFrame,
    output_dir: Path | str | None = None,
    run_stamp: str | None = None,
) -> tuple[Path, Path]:
    output_path = Path(output_dir) if output_dir is not None else LOG_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    stamp = run_stamp or run_id()

    order_plan_path = output_path / f"order_plan_{stamp}.csv"
    results_path = output_path / f"order_results_{stamp}.csv"

    save_df(order_plan if order_plan is not None else pd.DataFrame(), order_plan_path)
    save_df(results if results is not None else pd.DataFrame(), results_path)

    return order_plan_path, results_path


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production order executor.")

    parser.add_argument("--rebalance-orders", default=str(DEFAULT_REBALANCE_ORDERS))
    parser.add_argument("--exit-orders", default=str(DEFAULT_EXIT_ORDERS))

    parser.add_argument("--live", action="store_true")
    parser.add_argument("--disable-account-scaling", action="store_true", help="Do not clamp risk limits to live account equity.")

    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    parser.add_argument("--cash-buffer", type=float, default=DEFAULT_CASH_BUFFER_PCT)
    parser.add_argument("--max-gross-exposure", type=float, default=MAX_GROSS_EXPOSURE)
    parser.add_argument("--max-position-weight", type=float, default=MAX_POSITION_WEIGHT)
    parser.add_argument("--max-order-dollars", type=float, default=MAX_ORDER_DOLLARS)
    parser.add_argument("--max-single-order-qty", type=int, default=MAX_SINGLE_ORDER_QTY)
    parser.add_argument("--min-notional-to-trade", type=float, default=MIN_NOTIONAL_TO_TRADE)

    parser.add_argument("--allow-closed-market", action="store_true")
    parser.add_argument("--wait-for-open", action="store_true")
    parser.add_argument("--cancel-open-orders-first", action="store_true")
    parser.add_argument("--buy-first", action="store_true", help="Execute buys before sells. Default is sell-first.")

    parser.add_argument("--disable-kill-switch", action="store_true")
    parser.add_argument("--max-daily-loss-pct", type=float, default=MAX_DAILY_LOSS_PCT)
    parser.add_argument("--max-daily-loss-dollars", type=float, default=MAX_DAILY_LOSS_DOLLARS)
    parser.add_argument("--kill-switch-equity-floor", type=float, default=KILL_SWITCH_EQUITY_FLOOR)
    parser.add_argument("--loop", action="store_true", help="Run continuously. Default is one cycle.")
    parser.add_argument("--once", action="store_true", help="Run one cycle then exit. Default is to run forever.")

    parser.add_argument("--disable-scanner", action="store_true", help="Disable internal universe scanner and only read order CSVs.")
    parser.add_argument("--universe", default="", help="Comma-separated universe. Default uses DEFAULT_SCAN_UNIVERSE.")
    parser.add_argument("--use-sp500-fallback", action="store_true", help="Scan S&P 500 symbols from Wikipedia, capped by --max-universe-symbols.")
    parser.add_argument("--max-universe-symbols", type=int, default=30)
    parser.add_argument("--scanner-top-n", type=int, default=8)
    parser.add_argument("--min-setup-score", type=float, default=0.62)
    parser.add_argument("--min-buy-confidence", type=float, default=0.58)
    parser.add_argument("--min-avg-dollar-volume", type=float, default=5_000_000.0)
    parser.add_argument("--scanner-max-target-weight", type=float, default=0.14)
    parser.add_argument("--scanner-stop-loss", type=float, default=-0.035)
    parser.add_argument("--profit-lock-trigger", type=float, default=0.06)
    parser.add_argument("--profit-giveback", type=float, default=0.025)

    # ── Options-first mode ────────────────────────────────────────────────────
    parser.add_argument("--enable-options-booster", action="store_true",
                        help="Legacy alias. Enables options-first trading mode.")
    parser.add_argument("--enable-options-mode", action="store_true",
                        help="Enable options-first trading mode (replaces legacy call booster).")
    parser.add_argument("--enable-long-puts", action="store_true",
                        help="Allow long put entries in addition to long calls.")
    parser.add_argument("--option-feed", default="opra",
                        help="Alpaca data feed for options: opra (default) or indicative.")
    parser.add_argument("--option-min-dte", type=int, default=7,
                        help="Minimum DTE to consider a contract.")
    parser.add_argument("--option-max-dte", type=int, default=45,
                        help="Maximum DTE to consider a contract.")
    parser.add_argument("--option-call-delta-min", type=float, default=0.35)
    parser.add_argument("--option-call-delta-max", type=float, default=0.65)
    parser.add_argument("--option-put-delta-min", type=float, default=-0.65)
    parser.add_argument("--option-put-delta-max", type=float, default=-0.35)
    parser.add_argument("--option-max-spread-pct", type=float, default=0.12,
                        help="Max bid/ask spread as fraction of mid (0.12 = 12%%).")
    parser.add_argument("--option-min-volume", type=int, default=20)
    parser.add_argument("--option-min-oi", type=int, default=100,
                        help="Minimum open interest to consider a contract.")
    parser.add_argument("--option-risk-pct", type=float, default=0.015,
                        help="Risk per trade as fraction of equity (0.015 = 1.5%%).")
    parser.add_argument("--option-max-exposure", type=float, default=0.40,
                        help="Max total options exposure as fraction of equity.")
    parser.add_argument("--option-max-positions", type=int, default=4)
    parser.add_argument("--option-max-contracts", type=int, default=2)
    parser.add_argument("--option-take-profit", type=float, default=0.60)
    parser.add_argument("--option-stop-loss", type=float, default=-0.35)
    parser.add_argument("--allow-option-market-orders", action="store_true",
                        help="Use market orders for options (default: limit at midpoint).")
    parser.add_argument("--option-limit-entry-improvement", type=float, default=0.01,
                        help="Dollars above mid to improve limit entry price.")
    parser.add_argument("--option-limit-exit-offset", type=float, default=0.01,
                        help="Dollars below mid to accept on limit exit.")
    parser.add_argument("--option-close-before-expiration-days", type=int, default=3,
                        help="Force-close option positions when DTE <= this value.")
    parser.add_argument("--option-block-new-entries-within-expiration-days", type=int, default=5,
                        help="Do not open new options this close to expiration.")
    # Legacy args retained for backward compat
    parser.add_argument("--option-min-confidence", type=float, default=0.68)
    parser.add_argument("--option-min-underlying-plpc", type=float, default=0.003)
    parser.add_argument("--option-max-risk-pct", type=float, default=0.010)
    parser.add_argument("--option-max-risk-dollars", type=float, default=250.0)
    parser.add_argument("--option-max-open-positions", type=int, default=4)

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExecutorConfig:
    scanner_universe = [normalize_symbol(x) for x in str(args.universe).split(",") if normalize_symbol(x)] or None
    scanner_config = ScannerConfig(
        enabled=not args.disable_scanner,
        universe=scanner_universe,
        use_sp500_fallback=args.use_sp500_fallback,
        max_universe_symbols=args.max_universe_symbols,
        top_n=args.scanner_top_n,
        min_setup_score=args.min_setup_score,
        min_buy_confidence=args.min_buy_confidence,
        min_avg_dollar_volume=args.min_avg_dollar_volume,
        max_target_weight=args.scanner_max_target_weight,
        stop_loss_pct=args.scanner_stop_loss,
        profit_lock_trigger_pct=args.profit_lock_trigger,
        profit_giveback_pct=args.profit_giveback,
    )
    options_config = OptionsTradingConfig(
        enabled=args.enable_options_mode or args.enable_options_booster,
        feed=args.option_feed,
        enable_long_calls=True,
        enable_long_puts=args.enable_long_puts,
        min_dte=args.option_min_dte,
        max_dte=args.option_max_dte,
        call_delta_min=args.option_call_delta_min,
        call_delta_max=args.option_call_delta_max,
        put_delta_min=args.option_put_delta_min,
        put_delta_max=args.option_put_delta_max,
        max_spread_pct=args.option_max_spread_pct,
        min_open_interest=args.option_min_oi,
        min_volume=args.option_min_volume,
        risk_per_trade_pct=args.option_risk_pct,
        max_options_exposure=args.option_max_exposure,
        max_contracts_per_underlying=args.option_max_contracts,
        max_open_option_positions=args.option_max_positions,
        allow_option_market_orders=args.allow_option_market_orders,
        limit_entry_improvement=args.option_limit_entry_improvement,
        limit_exit_offset=args.option_limit_exit_offset,
        take_profit_pct=args.option_take_profit,
        hard_stop_pct=args.option_stop_loss,
        close_before_expiration_days=args.option_close_before_expiration_days,
        block_new_entries_within_expiration_days=args.option_block_new_entries_within_expiration_days,
    )

    return ExecutorConfig(
        paper=not args.live,
        live=args.live,
        enable_account_scaling=not args.disable_account_scaling,
        poll_once=args.once,
        poll_seconds=args.poll_seconds,
        rebalance_orders_path=Path(args.rebalance_orders),
        exit_orders_path=Path(args.exit_orders),
        cash_buffer_pct=args.cash_buffer,
        max_gross_exposure=args.max_gross_exposure,
        max_position_weight=args.max_position_weight,
        max_order_dollars=args.max_order_dollars,
        max_single_order_qty=args.max_single_order_qty,
        min_notional_to_trade=args.min_notional_to_trade,
        allow_closed_market=args.allow_closed_market,
        require_market_open=not args.allow_closed_market,
        wait_for_market_open=args.wait_for_open,
        cancel_open_orders_first=args.cancel_open_orders_first,
        sell_first=not args.buy_first,
        enable_kill_switch=not args.disable_kill_switch,
        max_daily_loss_pct=args.max_daily_loss_pct,
        max_daily_loss_dollars=args.max_daily_loss_dollars,
        kill_switch_equity_floor=args.kill_switch_equity_floor,
        scanner=scanner_config,
        options_booster=options_config,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    trader = ProductionOrderExecutor(config)
    trader.run()


if __name__ == "__main__":
    main()
