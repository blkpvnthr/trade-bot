#!/usr/bin/env python3
"""
run.py

Clean 13K-account Alpaca paper executor with built-in momentum/reversal scanner.

Default behavior:
    python run.py

What it does:
    - loads Alpaca credentials from .env
    - scans a small default universe using Alpaca daily bars
    - writes rebalance/exit CSVs into daily_system_outputs/
    - checks Alpaca market clock
    - applies small-account risk controls
    - sells before buys
    - enforces cash buffer, gross exposure, position size, order size, cooldown
    - applies daily loss kill switch and absolute equity floor
    - avoids duplicate open orders
    - logs plans, results, account snapshots, journal rows, and JSONL events

Designed for roughly a $13,000 paper account.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

try:
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except Exception:
    DataFeed = None
    StockHistoricalDataClient = None
    StockBarsRequest = None
    TimeFrame = None


# =============================================================================
# Paths / environment
# =============================================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

OUT_DIR = PROJECT_ROOT / "daily_system_outputs"
LOG_DIR = PROJECT_ROOT / "logs" / "execution"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_REBALANCE_ORDERS = OUT_DIR / "daily_walkforward_emulated_rebalance_orders.csv"
DEFAULT_EXIT_ORDERS = OUT_DIR / "daily_walkforward_emulated_exit_orders.csv"

JOURNAL_PATH = LOG_DIR / "executor_journal.csv"
EQUITY_CURVE_PATH = LOG_DIR / "executor_equity_curve.csv"
EVENTS_JSONL = LOG_DIR / "execution_events.jsonl"
KILL_SWITCH_PATH = LOG_DIR / "kill_switch_state.json"
SCANNER_STATE_PATH = LOG_DIR / "scanner_state.json"
SCANNER_SCORES_PATH = OUT_DIR / "latest_scanner_scores.csv"
NO_REENTRY_STATE_PATH = LOG_DIR / "no_reentry_state.json"


# =============================================================================
# Dynamic account-scaling defaults
# =============================================================================

POLL_SECONDS = 30
ORDER_WAIT_SECONDS = 20
COOLDOWN_SECONDS = 15
HEARTBEAT_SECONDS = 60

MIN_BUYING_POWER = 250.0
MIN_NOTIONAL_TO_TRADE = 75.0
MIN_WEIGHT_DELTA = 0.005

# Fallback defaults. Actual risk is dynamically scaled from account equity.
MIN_CASH_BUFFER = 1_500.0
DEFAULT_CASH_BUFFER_PCT = 0.12
MAX_GROSS_EXPOSURE = 0.72
MAX_POSITION_WEIGHT = 0.12
MAX_SINGLE_ORDER_NOTIONAL_PCT = 0.045
MAX_ORDER_DOLLARS = 1_500.0
MAX_SINGLE_ORDER_QTY = 25_000

DAILY_DRAWDOWN_LIMIT = -0.015
KILL_SWITCH_EQUITY_FLOOR = 11_700.0
MAX_DAILY_LOSS_PCT = 0.015
MAX_DAILY_LOSS_DOLLARS = 200.0

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

DEFAULT_SCAN_UNIVERSE = [
    "QQQ", "TQQQ", "SQQQ", "SPY", "DIA", "IWM", "XLK", "SOXX", "SMH",
    "AMD", "NVDA", "AVGO", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA",
    "PLTR", "COIN", "MARA", "RIOT", "RGTI", "QBTS", "QUBT", "IONQ", "SOUN",
    "OKLO", "APLD", "ALAB", "ARM",
]


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
class AccountRiskProfile:
    band: str
    equity: float
    cash_buffer_pct: float
    min_cash_buffer: float
    max_gross_exposure: float
    max_position_weight: float
    max_single_order_notional_pct: float
    max_order_dollars: float
    max_daily_loss_pct: float
    max_daily_loss_dollars: float
    daily_drawdown_limit: float
    kill_switch_equity_floor: float
    min_notional_to_trade: float


@dataclass
class ScannerConfig:
    enabled: bool = True
    universe: Optional[list[str]] = None
    use_sp500_fallback: bool = False
    max_universe_symbols: int = 30
    lookback_days: int = 260
    data_feed: str = "iex"

    # Broad-market mode. If Wikipedia retrieval fails, falls back to DEFAULT_SCAN_UNIVERSE.
    # Capped to avoid very large one-shot Alpaca requests.
    broad_market_scan: bool = True

    top_n: int = 30
    min_setup_score: float = 0.62
    min_buy_confidence: float = 0.58
    min_price: float = 2.0
    max_price: float = 500.0
    min_avg_dollar_volume: float = 5_000_000.0

    base_target_weight: float = 0.03
    max_target_weight: float = 0.10
    reversal_target_weight_mult: float = 0.75

    stop_loss_pct: float = -0.025
    stale_position_score_exit: float = 0.45
    profit_lock_trigger_pct: float = 0.04
    profit_giveback_pct: float = 0.015

    require_positive_daily_move: bool = True
    avoid_bearish_regime_for_longs: bool = True
    allow_relaxed_fallback: bool = True
    fallback_min_setup_score: float = 0.52
    fallback_min_buy_confidence: float = 0.50
    max_overbought_rsi: float = 78.0
    min_regime_score_for_longs: float = 0.40

    # New-opportunity scanner: looks for stocks already green on the day
    # with positive momentum or confirmed breakout/reclaim behavior.
    opportunity_scan_enabled: bool = True
    min_daily_gain_pct: float = 0.005
    min_opportunity_score: float = 0.60
    min_breakout_confirmation_score: float = 0.55
    require_positive_momentum_or_breakout: bool = True

    # Intraday entry-analysis layer.
    analyze_intraday_entries: bool = True
    intraday_lookback_days: int = 5
    entry_near_support_buffer_pct: float = 0.003
    entry_breakout_buffer_pct: float = 0.002
    max_entry_chase_pct: float = 0.018


@dataclass
class ExecutorConfig:
    paper: bool = True
    live: bool = False

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
    order_tag_prefix: str = "qq_13k"

    scanner: ScannerConfig = field(default_factory=ScannerConfig)


# =============================================================================
# Helpers
# =============================================================================

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def run_id() -> str:
    return now_utc().strftime("%Y%m%d_%H%M%S")


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def safe_symbol(value: Any) -> str:
    symbol = str(value or "").upper().strip().replace(".", "-")
    return re.sub(r"[^A-Z0-9\-]", "", symbol)


def normalize_symbol(symbol: Any) -> str:
    return safe_symbol(symbol)


def is_simple_alpaca_symbol(symbol: Any) -> bool:
    """
    Keep only simple Alpaca-friendly stock symbols.

    Excludes class-share / special symbols such as BF.B, BRK.B, BF-B,
    warrants, units, preferreds, and other symbols that commonly break
    Alpaca historical bar requests.
    """
    raw = str(symbol or "").upper().strip()
    normalized = safe_symbol(raw)
    if not normalized:
        return False
    if "." in raw or "/" in raw or "^" in raw or " " in raw:
        return False
    if "-" in normalized:
        return False
    return bool(re.fullmatch(r"[A-Z]{1,5}", normalized))


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    return str(value)


def build_account_risk_profile(equity: float) -> AccountRiskProfile:
    """Scale risk controls from a small account to roughly $1M automatically."""
    equity = max(float(equity), 1.0)

    if equity < 25_000:
        band = "MICRO_UNDER_PDT"
        cash_buffer_pct = 0.12
        min_cash_buffer = max(1_500.0, equity * 0.10)
        max_gross_exposure = 0.72
        max_position_weight = 0.12
        max_single_order_notional_pct = 0.045
        max_order_dollars = min(1_500.0, equity * 0.12)
        max_daily_loss_pct = 0.015
        max_daily_loss_dollars = min(250.0, equity * 0.018)
        daily_drawdown_limit = -0.015
        kill_floor_drawdown = 0.10
        min_notional = 75.0
    elif equity < 100_000:
        band = "SMALL"
        cash_buffer_pct = 0.10
        min_cash_buffer = max(2_500.0, equity * 0.07)
        max_gross_exposure = 0.80
        max_position_weight = 0.14
        max_single_order_notional_pct = 0.055
        max_order_dollars = min(5_000.0, equity * 0.08)
        max_daily_loss_pct = 0.018
        max_daily_loss_dollars = min(1_250.0, equity * 0.018)
        daily_drawdown_limit = -0.018
        kill_floor_drawdown = 0.10
        min_notional = 100.0
    elif equity < 250_000:
        band = "GROWTH"
        cash_buffer_pct = 0.08
        min_cash_buffer = max(7_500.0, equity * 0.05)
        max_gross_exposure = 0.88
        max_position_weight = 0.16
        max_single_order_notional_pct = 0.06
        max_order_dollars = min(15_000.0, equity * 0.07)
        max_daily_loss_pct = 0.018
        max_daily_loss_dollars = min(3_000.0, equity * 0.018)
        daily_drawdown_limit = -0.018
        kill_floor_drawdown = 0.09
        min_notional = 250.0
    elif equity < 500_000:
        band = "PRO"
        cash_buffer_pct = 0.07
        min_cash_buffer = max(15_000.0, equity * 0.045)
        max_gross_exposure = 0.92
        max_position_weight = 0.16
        max_single_order_notional_pct = 0.055
        max_order_dollars = min(25_000.0, equity * 0.06)
        max_daily_loss_pct = 0.016
        max_daily_loss_dollars = min(5_000.0, equity * 0.016)
        daily_drawdown_limit = -0.016
        kill_floor_drawdown = 0.08
        min_notional = 500.0
    else:
        band = "INSTITUTIONAL_1M"
        cash_buffer_pct = 0.06
        min_cash_buffer = max(35_000.0, equity * 0.04)
        max_gross_exposure = 0.95
        max_position_weight = 0.14
        max_single_order_notional_pct = 0.04
        max_order_dollars = min(40_000.0, equity * 0.045)
        max_daily_loss_pct = 0.0125
        max_daily_loss_dollars = min(10_000.0, equity * 0.0125)
        daily_drawdown_limit = -0.0125
        kill_floor_drawdown = 0.07
        min_notional = 1_000.0

    return AccountRiskProfile(
        band=band,
        equity=equity,
        cash_buffer_pct=cash_buffer_pct,
        min_cash_buffer=min_cash_buffer,
        max_gross_exposure=max_gross_exposure,
        max_position_weight=max_position_weight,
        max_single_order_notional_pct=max_single_order_notional_pct,
        max_order_dollars=max_order_dollars,
        max_daily_loss_pct=max_daily_loss_pct,
        max_daily_loss_dollars=max_daily_loss_dollars,
        daily_drawdown_limit=daily_drawdown_limit,
        kill_switch_equity_floor=equity * (1.0 - kill_floor_drawdown),
        min_notional_to_trade=min_notional,
    )


def write_event(event: str, payload: Optional[dict[str, Any]] = None) -> None:
    row = {
        "timestamp_utc": now_utc().isoformat(),
        "event": event,
        "payload": payload or {},
    }
    with EVENTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=json_default) + "\n")


def save_df(df: Optional[pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (df if df is not None else pd.DataFrame()).to_csv(path, index=False)
    print(f"[saved] {path}")


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=json_default) + "\n", encoding="utf-8")
    print(f"[saved] {path}")


def local_trading_day() -> str:
    """Trading-day key used for same-day no-reentry state."""
    return datetime.now().date().isoformat()


def read_json_file(path: Path, default: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if not path.exists():
        return dict(default or {})
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(default or {})


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")


def load_no_reentry_symbols() -> set[str]:
    """Symbols exited today that must not be re-entered until the next trading day."""
    today = local_trading_day()
    state = read_json_file(NO_REENTRY_STATE_PATH, {"date": today, "symbols": {}})
    if state.get("date") != today:
        state = {"date": today, "symbols": {}}
        write_json_file(NO_REENTRY_STATE_PATH, state)
    symbols = state.get("symbols", {})
    if isinstance(symbols, dict):
        return {safe_symbol(sym) for sym in symbols.keys() if safe_symbol(sym)}
    if isinstance(symbols, list):
        return {safe_symbol(sym) for sym in symbols if safe_symbol(sym)}
    return set()


def mark_no_reentry_symbol(symbol: str, reason: str = "exit") -> None:
    """Persist that a symbol was exited today and cannot be bought again today."""
    symbol = safe_symbol(symbol)
    if not symbol:
        return
    today = local_trading_day()
    state = read_json_file(NO_REENTRY_STATE_PATH, {"date": today, "symbols": {}})
    if state.get("date") != today:
        state = {"date": today, "symbols": {}}

    symbols = state.get("symbols", {})
    if not isinstance(symbols, dict):
        symbols = {}

    symbols[symbol] = {
        "symbol": symbol,
        "marked_at": now_utc().isoformat(),
        "reason": reason,
    }
    state["symbols"] = symbols
    write_json_file(NO_REENTRY_STATE_PATH, state)
    write_event("no_reentry_marked", {"symbol": symbol, "reason": reason, "date": today})


# =============================================================================
# CSV journal
# =============================================================================

class CsvJournal:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = [
            "timestamp_utc", "cycle_id", "event", "market_open",
            "equity_before", "equity_after", "cash_before", "cash_after",
            "buying_power_before", "buying_power_after",
            "gross_exposure_before", "gross_exposure_after",
            "trade_symbol", "trade_side", "trade_qty", "trade_price",
            "estimated_notional", "target_weight", "current_weight", "delta_weight",
            "confidence", "cash_buffer", "max_gross_exposure", "max_position_weight",
            "max_order_notional", "order_id", "order_status", "filled_qty",
            "filled_avg_price", "slippage_pct", "message",
        ]
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.columns).writeheader()

    def append(self, row: dict[str, Any]) -> None:
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
        try:
            raw_positions = self.trading.get_all_positions()
        except Exception as exc:
            print(f"[broker] failed to fetch positions: {exc}")
            return []

        positions: list[BrokerPosition] = []
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

    def list_open_orders(self) -> list[Any]:
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
                if safe_symbol(getattr(order, "symbol", "")) == symbol and order.status in open_statuses:
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
        return self.submit_market_order(symbol, "buy", qty, client_order_id, tif)

    def submit_market_sell(self, symbol: str, qty: int, client_order_id: str, tif: TimeInForce):
        return self.submit_market_order(symbol, "sell", qty, client_order_id, tif)

    def get_order_by_id(self, order_id: str):
        return self.trading.get_order_by_id(order_id)


# =============================================================================
# Scanner helpers
# =============================================================================

def _web_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }


def get_sp500_fallback_symbols() -> list[str]:
    try:
        resp = requests.get(WIKI_SP500_URL, headers=_web_headers(), timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        if not tables or "Symbol" not in tables[0].columns:
            raise RuntimeError("Could not parse S&P 500 symbols from Wikipedia.")
        symbols = []
        for raw in tables[0]["Symbol"].astype(str):
            if is_simple_alpaca_symbol(raw):
                symbols.append(normalize_symbol(raw))
        return sorted(set(symbols))
    except Exception as exc:
        print(f"[scanner] S&P 500 fallback failed: {exc}; using default universe")
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


class MomentumSetupScanner:
    def __init__(self, broker: AlpacaBroker, config: ScannerConfig, executor_config: ExecutorConfig):
        self.broker = broker
        self.config = config
        self.executor_config = executor_config

        if StockHistoricalDataClient is None:
            raise RuntimeError("alpaca-py data classes are unavailable. Install alpaca-py or disable scanner.")

        self.data = StockHistoricalDataClient(broker.api_key, broker.secret_key)

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
            raw_symbols = self.config.universe
        elif self.config.use_sp500_fallback or self.config.broad_market_scan:
            raw_symbols = get_sp500_fallback_symbols()
        else:
            raw_symbols = DEFAULT_SCAN_UNIVERSE

        symbols: list[str] = []
        rejected: list[str] = []
        for raw in raw_symbols:
            if not is_simple_alpaca_symbol(raw):
                rejected.append(str(raw))
                continue
            symbol = normalize_symbol(raw)
            if symbol and symbol not in symbols:
                symbols.append(symbol)

        if rejected:
            print(f"[scanner] rejected non-simple Alpaca symbols={len(rejected)} examples={rejected[:8]}")
        return symbols[: self.config.max_universe_symbols]

    def _feed(self):
        if DataFeed is None:
            return None
        return DataFeed.IEX if str(self.config.data_feed).lower() == "iex" else DataFeed.SIP

    def _download_bars(self, symbols: list[str]) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()

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

        df = bars.reset_index().rename(columns={"level_0": "symbol", "level_1": "timestamp"})
        if "symbol" not in df.columns and "Symbol" in df.columns:
            df = df.rename(columns={"Symbol": "symbol"})
        if "timestamp" not in df.columns and "Timestamp" in df.columns:
            df = df.rename(columns={"Timestamp": "timestamp"})

        df["symbol"] = df["symbol"].map(normalize_symbol)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        return df.dropna(subset=["symbol", "timestamp", "close"]).sort_values(["symbol", "timestamp"])

    def _download_intraday_bars(self, symbols: list[str], minutes: int) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()
        if TimeFrameUnit is None:
            return pd.DataFrame()

        end = now_utc()
        start = end - timedelta(days=max(1, self.config.intraday_lookback_days))
        timeframe = TimeFrame(minutes, TimeFrameUnit.Minute)

        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=self._feed(),
        )

        try:
            bars = self.data.get_stock_bars(req).df
        except Exception as exc:
            print(f"[intraday] failed to download {minutes}Min bars: {exc}")
            return pd.DataFrame()

        if bars is None or bars.empty:
            return pd.DataFrame()

        df = bars.reset_index().rename(columns={"level_0": "symbol", "level_1": "timestamp"})
        if "symbol" not in df.columns and "Symbol" in df.columns:
            df = df.rename(columns={"Symbol": "symbol"})
        if "timestamp" not in df.columns and "Timestamp" in df.columns:
            df = df.rename(columns={"Timestamp": "timestamp"})

        df["symbol"] = df["symbol"].map(normalize_symbol)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["timeframe_minutes"] = minutes
        return df.dropna(subset=["symbol", "timestamp", "close"]).sort_values(["symbol", "timestamp"])

    def _build_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []

        for symbol, g in bars.groupby("symbol", sort=False):
            g = g.sort_values("timestamp").copy()
            c = g["close"].astype(float)
            v = g["volume"].astype(float)

            g["ret_1d"] = c.pct_change(1, fill_method=None)
            g["ret_3d"] = c.pct_change(3, fill_method=None)
            g["ret_5d"] = c.pct_change(5, fill_method=None)
            g["ret_10d"] = c.pct_change(10, fill_method=None)
            g["ret_20d"] = c.pct_change(20, fill_method=None)
            g["positive_daily_move"] = (g["ret_1d"] > 0).astype(float)
            g["sma5"] = c.rolling(5).mean()
            g["sma20"] = c.rolling(20).mean()
            g["sma50"] = c.rolling(50).mean()
            g["sma200"] = c.rolling(200).mean()
            g["sma5_cross_sma20"] = ((g["sma5"] > g["sma20"]) & (g["sma5"].shift(1) <= g["sma20"].shift(1))).astype(float)
            g["sma20_cross_sma50"] = ((g["sma20"] > g["sma50"]) & (g["sma20"].shift(1) <= g["sma50"].shift(1))).astype(float)
            g["sma50_cross_sma200"] = ((g["sma50"] > g["sma200"]) & (g["sma50"].shift(1) <= g["sma200"].shift(1))).astype(float)
            g["rsi14"] = _rsi(c, 14)

            ema12 = c.ewm(span=12, adjust=False).mean()
            ema26 = c.ewm(span=26, adjust=False).mean()
            g["macd"] = ema12 - ema26
            g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
            g["macd_hist"] = g["macd"] - g["macd_signal"]
            g["macd_hist_slope"] = g["macd_hist"].diff()
            g["macd_bullish"] = ((g["macd"] > g["macd_signal"]) & (g["macd_hist"] > 0)).astype(float)
            g["atr14"] = _atr(g, 14)
            g["atr_pct"] = g["atr14"] / c.replace(0, np.nan)
            g["rolling_std10"] = g["ret_1d"].rolling(10).std()
            g["rolling_std20"] = g["ret_1d"].rolling(20).std()
            g["vol20"] = v.rolling(20).mean()
            g["dollar_volume20"] = (c * v).rolling(20).mean()
            g["rel_volume"] = v / g["vol20"].replace(0, np.nan)

            mid = g["sma20"]
            sd = c.rolling(20).std()
            g["bb_low"] = mid - 2 * sd
            g["bb_high"] = mid + 2 * sd
            g["bb_pos"] = (c - g["bb_low"]) / (g["bb_high"] - g["bb_low"]).replace(0, np.nan)
            g["bb_breakout"] = ((c > g["bb_high"]) & (g["ret_1d"] > 0)).astype(float)
            g["bb_rejection"] = ((c < g["bb_high"]) & (c.shift(1) > g["bb_high"].shift(1)) & (g["ret_1d"] < 0)).astype(float)

            g["high20"] = c.rolling(20).max()
            g["drawdown_20h"] = c / g["high20"].replace(0, np.nan) - 1.0
            g["trend_stack"] = ((g["sma5"] > g["sma20"]) & (g["sma50"] > g["sma200"])).astype(float)
            g["above_sma20"] = (c > g["sma20"]).astype(float)
            g["above_prev_high"] = (c > g["high"].shift(1)).astype(float)
            g["lower_band_reclaim"] = ((c.shift(1) < g["bb_low"].shift(1)) & (c > g["bb_low"]) & (g["ret_1d"] > 0)).astype(float)
            g["overbought_rsi"] = (g["rsi14"] >= self.config.max_overbought_rsi).astype(float)
            g["weak_macd"] = ((g["macd_hist"] < 0) | (g["macd_hist_slope"] < 0)).astype(float)
            rows.append(g)

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def _pivot_levels(self, df: pd.DataFrame) -> tuple[list[float], list[float]]:
        """Extract simple pivot support/resistance levels from intraday bars."""
        if df.empty or len(df) < 10:
            return [], []

        g = df.sort_values("timestamp").copy()
        highs = g["high"].astype(float).reset_index(drop=True)
        lows = g["low"].astype(float).reset_index(drop=True)

        resistance: list[float] = []
        support: list[float] = []

        for i in range(2, len(g) - 2):
            h = highs.iloc[i]
            l = lows.iloc[i]
            if h >= highs.iloc[i - 2:i + 3].max():
                resistance.append(float(h))
            if l <= lows.iloc[i - 2:i + 3].min():
                support.append(float(l))

        # Include recent range extremes so the layer still works when pivots are sparse.
        support.extend([float(lows.tail(30).min()), float(lows.tail(60).min())])
        resistance.extend([float(highs.tail(30).max()), float(highs.tail(60).max())])

        return support, resistance

    def _cluster_price_levels(self, levels: list[float], tolerance_pct: float = 0.003) -> list[float]:
        """Cluster nearby price levels into cleaner S/R zones."""
        clean = sorted([float(x) for x in levels if np.isfinite(x) and x > 0])
        if not clean:
            return []

        clusters: list[list[float]] = []
        for level in clean:
            if not clusters:
                clusters.append([level])
                continue
            center = float(np.mean(clusters[-1]))
            if abs(level - center) / max(center, 1e-9) <= tolerance_pct:
                clusters[-1].append(level)
            else:
                clusters.append([level])

        return [float(np.mean(c)) for c in clusters]

    def _nearest_levels(self, current_price: float, support: list[float], resistance: list[float]) -> tuple[float, float]:
        supports_below = [x for x in support if x <= current_price]
        resistance_above = [x for x in resistance if x >= current_price]
        nearest_support = max(supports_below) if supports_below else (max(support) if support else 0.0)
        nearest_resistance = min(resistance_above) if resistance_above else (min(resistance) if resistance else 0.0)
        return float(nearest_support or 0.0), float(nearest_resistance or 0.0)

    def _entry_from_intraday_context(
        self,
        symbol: str,
        current_price: float,
        setup_type: str,
        is_breakout: bool,
        bars_1m: pd.DataFrame,
        bars_5m: pd.DataFrame,
        bars_15m: pd.DataFrame,
    ) -> dict[str, Any]:
        """Analyze 1m/5m/15m bars to estimate support, resistance, and a reasonable entry."""
        symbol = normalize_symbol(symbol)
        frames = {
            "1m": bars_1m[bars_1m["symbol"] == symbol].copy() if not bars_1m.empty else pd.DataFrame(),
            "5m": bars_5m[bars_5m["symbol"] == symbol].copy() if not bars_5m.empty else pd.DataFrame(),
            "15m": bars_15m[bars_15m["symbol"] == symbol].copy() if not bars_15m.empty else pd.DataFrame(),
        }

        all_support: list[float] = []
        all_resistance: list[float] = []
        timeframe_notes: dict[str, Any] = {}

        for label, frame in frames.items():
            if frame.empty:
                continue
            support, resistance = self._pivot_levels(frame)
            support = self._cluster_price_levels(support)
            resistance = self._cluster_price_levels(resistance)
            all_support.extend(support)
            all_resistance.extend(resistance)
            timeframe_notes[label] = {
                "bars": int(len(frame)),
                "support_levels": support[-5:],
                "resistance_levels": resistance[:5],
            }

        support_levels = self._cluster_price_levels(all_support)
        resistance_levels = self._cluster_price_levels(all_resistance)
        nearest_support, nearest_resistance = self._nearest_levels(current_price, support_levels, resistance_levels)

        if current_price <= 0:
            return {
                "entry_price": 0.0,
                "entry_type": "invalid_price",
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance,
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "risk_reward_ratio": 0.0,
                "stop_price": 0.0,
                "entry_notes": "invalid current price",
                "intraday_context": timeframe_notes,
            }

        support_entry = nearest_support * (1.0 + self.config.entry_near_support_buffer_pct) if nearest_support > 0 else current_price
        breakout_entry = nearest_resistance * (1.0 + self.config.entry_breakout_buffer_pct) if nearest_resistance > 0 else current_price
        max_chase_entry = current_price * (1.0 + self.config.max_entry_chase_pct)

        if is_breakout or setup_type == "momentum":
            entry_price = min(max(current_price, breakout_entry), max_chase_entry) if nearest_resistance > current_price else current_price
            entry_type = "breakout_confirmation_entry" if nearest_resistance > current_price else "market_momentum_entry"
        else:
            entry_price = min(current_price, support_entry) if nearest_support > 0 else current_price
            entry_type = "support_reclaim_entry"

        stop_price = nearest_support * 0.992 if nearest_support > 0 else current_price * 0.975
        target_price = nearest_resistance if nearest_resistance > entry_price else entry_price * 1.035
        risk = max(entry_price - stop_price, 0.01)
        reward = max(target_price - entry_price, 0.0)
        rr = reward / risk if risk > 0 else 0.0

        return {
            "entry_price": float(entry_price),
            "entry_type": entry_type,
            "nearest_support": float(nearest_support),
            "nearest_resistance": float(nearest_resistance),
            "support_levels": support_levels[-8:],
            "resistance_levels": resistance_levels[:8],
            "risk_reward_ratio": float(rr),
            "stop_price": float(stop_price),
            "target_price": float(target_price),
            "entry_notes": f"{entry_type}; support={nearest_support:.2f}; resistance={nearest_resistance:.2f}; rr={rr:.2f}",
            "intraday_context": timeframe_notes,
        }

    def _annotate_intraday_entries(self, candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty or not self.config.analyze_intraday_entries:
            return candidates

        symbols = [normalize_symbol(x) for x in candidates["symbol"].tolist() if normalize_symbol(x)]
        if not symbols:
            return candidates

        print(f"[intraday] analyzing 1m/5m/15m support-resistance for candidates={len(symbols)}")
        bars_1m = self._download_intraday_bars(symbols, 1)
        bars_5m = self._download_intraday_bars(symbols, 5)
        bars_15m = self._download_intraday_bars(symbols, 15)

        rows: list[dict[str, Any]] = []
        for _, row in candidates.iterrows():
            symbol = normalize_symbol(row["symbol"])
            current_price = to_float(row.get("close"))
            is_breakout = bool(to_float(row.get("confirmed_breakout_score")) >= self.config.min_breakout_confirmation_score)
            context = self._entry_from_intraday_context(
                symbol=symbol,
                current_price=current_price,
                setup_type=str(row.get("setup_type", "")),
                is_breakout=is_breakout,
                bars_1m=bars_1m,
                bars_5m=bars_5m,
                bars_15m=bars_15m,
            )
            merged = row.to_dict()
            merged.update(context)
            rows.append(merged)

        out = pd.DataFrame(rows)
        if not out.empty:
            print("[intraday] entry analysis:")
            show = ["symbol", "close", "entry_price", "entry_type", "nearest_support", "nearest_resistance", "risk_reward_ratio"]
            print(out[[c for c in show if c in out.columns]].to_string(index=False))
        return out

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

        # Market regime proxy from SPY/QQQ if present, otherwise neutral.
        benchmark = latest[latest["symbol"].isin(["SPY", "QQQ"])].copy()
        if benchmark.empty:
            regime_score = 0.55
            regime_label = "NEUTRAL_UNKNOWN"
        else:
            bench_momentum = (
                0.35 * clip01(benchmark["ret_5d"], -0.03, 0.06)
                + 0.25 * clip01(benchmark["ret_10d"], -0.04, 0.09)
                + 0.20 * benchmark["above_sma20"].fillna(0)
                + 0.20 * benchmark["trend_stack"].fillna(0)
            ).mean()
            regime_score = float(np.clip(bench_momentum, 0, 1))
            if regime_score >= 0.62:
                regime_label = "BULLISH"
            elif regime_score <= 0.38:
                regime_label = "BEARISH"
            else:
                regime_label = "NEUTRAL"

        latest["regime_score"] = regime_score
        latest["regime_label"] = regime_label

        trend_score = (
            0.24 * latest["trend_stack"].fillna(0)
            + 0.16 * latest["above_sma20"].fillna(0)
            + 0.12 * latest["above_prev_high"].fillna(0)
            + 0.12 * latest["sma5_cross_sma20"].fillna(0)
            + 0.10 * latest["sma20_cross_sma50"].fillna(0)
            + 0.08 * latest["sma50_cross_sma200"].fillna(0)
            + 0.10 * clip01(latest["bb_pos"], 0.45, 0.95)
            + 0.08 * latest["bb_breakout"].fillna(0)
        ).clip(0, 1)

        momentum_score = (
            0.18 * clip01(latest["ret_1d"], -0.01, 0.05)
            + 0.22 * clip01(latest["ret_5d"], -0.02, 0.10)
            + 0.18 * clip01(latest["ret_10d"], -0.03, 0.18)
            + 0.12 * clip01(latest["ret_20d"], -0.05, 0.25)
            + 0.12 * clip01(latest["rsi14"], 45, 72)
            + 0.10 * latest["macd_bullish"].fillna(0)
            + 0.08 * clip01(latest["macd_hist_slope"], -0.02, 0.05)
        ).clip(0, 1)

        liquidity_score = (
            0.55 * clip01(latest["dollar_volume20"], self.config.min_avg_dollar_volume, self.config.min_avg_dollar_volume * 20)
            + 0.45 * clip01(latest["rel_volume"], 0.8, 3.0)
        ).clip(0, 1)

        volatility_score = (
            0.50 * clip01(latest["atr_pct"], 0.01, 0.08)
            + 0.25 * (1 - clip01(latest["rolling_std10"], 0.015, 0.10))
            + 0.25 * (1 - clip01(latest["rolling_std20"], 0.015, 0.10))
        ).clip(0, 1)

        risk_penalty = (
            0.35 * latest["overbought_rsi"].fillna(0)
            + 0.25 * latest["bb_rejection"].fillna(0)
            + 0.25 * latest["weak_macd"].fillna(0)
            + 0.15 * (1 - latest["positive_daily_move"].fillna(0))
        ).clip(0, 1)

        reversal_score = (
            0.24 * (1 - clip01(latest["rsi14"], 30, 55))
            + 0.22 * latest["lower_band_reclaim"].fillna(0)
            + 0.18 * clip01(latest["ret_1d"], -0.01, 0.05)
            + 0.16 * clip01(latest["rel_volume"], 0.8, 2.8)
            + 0.10 * (1 - clip01(latest["bb_pos"], 0.15, 0.55))
            + 0.10 * (1 - clip01(latest["drawdown_20h"], -0.18, -0.03))
        ).clip(0, 1)

        # Continuation probability proxy. If a real ML model is later plugged in,
        # replace this column with model.predict_proba outputs.
        ml_probability = (
            0.30 * trend_score
            + 0.25 * momentum_score
            + 0.15 * liquidity_score
            + 0.10 * volatility_score
            + 0.15 * regime_score
            + 0.05 * latest["positive_daily_move"].fillna(0)
            - 0.22 * risk_penalty
        ).clip(0.01, 0.99)

        confidence_score = (
            0.35 * ml_probability
            + 0.20 * trend_score
            + 0.20 * momentum_score
            + 0.10 * liquidity_score
            + 0.10 * regime_score
            + 0.05 * volatility_score
            - 0.20 * risk_penalty
        ).clip(0, 1)

        latest["trend_score"] = trend_score
        latest["momentum_score"] = momentum_score
        latest["liquidity_score"] = liquidity_score
        latest["volatility_score"] = volatility_score
        latest["risk_penalty"] = risk_penalty
        latest["reversal_score"] = reversal_score
        latest["continuation_probability"] = ml_probability
        latest["ml_probability"] = ml_probability
        latest["confidence_score"] = confidence_score
        latest["setup_type"] = np.where(latest["momentum_score"] >= latest["reversal_score"], "momentum", "reversal")
        latest["setup_score"] = latest[["confidence_score", "momentum_score", "reversal_score"]].max(axis=1)
        latest["trade_confidence"] = latest["confidence_score"].clip(0, 1)

        # New-opportunity scanner layer.
        # This is intentionally rule-based so it can run without a trained model:
        # +P&L for the day, positive momentum, or confirmed breakout/reclaim.
        confirmed_breakout_score = (
            0.28 * latest["above_prev_high"].fillna(0)
            + 0.18 * latest["bb_breakout"].fillna(0)
            + 0.16 * latest["sma5_cross_sma20"].fillna(0)
            + 0.14 * latest["macd_bullish"].fillna(0)
            + 0.12 * clip01(latest["rel_volume"], 1.0, 3.0)
            + 0.12 * clip01(latest["ret_1d"], 0.0025, 0.06)
        ).clip(0, 1)

        positive_momentum_score = (
            0.24 * clip01(latest["ret_1d"], 0.0, 0.05)
            + 0.22 * clip01(latest["ret_5d"], 0.0, 0.12)
            + 0.16 * clip01(latest["ret_10d"], 0.0, 0.18)
            + 0.14 * latest["above_sma20"].fillna(0)
            + 0.12 * latest["macd_bullish"].fillna(0)
            + 0.12 * clip01(latest["rsi14"], 50, 72)
        ).clip(0, 1)

        opportunity_score = (
            0.35 * positive_momentum_score
            + 0.30 * confirmed_breakout_score
            + 0.15 * liquidity_score
            + 0.10 * regime_score
            + 0.10 * latest["positive_daily_move"].fillna(0)
            - 0.20 * risk_penalty
        ).clip(0, 1)

        latest["confirmed_breakout_score"] = confirmed_breakout_score
        latest["positive_momentum_score"] = positive_momentum_score
        latest["opportunity_score"] = opportunity_score
        latest["is_positive_day_opportunity"] = (
            (latest["ret_1d"] >= self.config.min_daily_gain_pct)
            & (
                (positive_momentum_score >= self.config.min_opportunity_score)
                | (confirmed_breakout_score >= self.config.min_breakout_confirmation_score)
            )
            & (risk_penalty <= 0.60)
            & (latest["dollar_volume20"] >= self.config.min_avg_dollar_volume)
        )

        return latest.sort_values(
            ["is_positive_day_opportunity", "opportunity_score", "trade_confidence", "setup_score", "continuation_probability", "dollar_volume20"],
            ascending=[False, False, False, False, False, False],
        ).reset_index(drop=True)

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
                reason = "scanner_stop_loss"
            elif new_high >= self.config.profit_lock_trigger_pct and pos.unrealized_plpc <= new_high - self.config.profit_giveback_pct:
                reason = "scanner_profit_giveback"
            elif symbol not in top_symbols and to_float(score_by_symbol.get(symbol), 0.0) < self.config.stale_position_score_exit:
                reason = "scanner_stale_position"

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

    def build_orders(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        symbols = self._universe()
        print(f"[scanner] scanning universe symbols={len(symbols)}")

        bars = self._download_bars(symbols)
        if bars.empty:
            print("[scanner] no bars downloaded")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        features = self._build_features(bars)
        scores = self._score_latest(features)
        if scores.empty:
            print("[scanner] no symbols passed scanner filters")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        scores.to_csv(SCANNER_SCORES_PATH, index=False)
        print("[scanner] top setups:")
        show_cols = [
            "symbol", "close", "regime_label", "setup_type", "is_positive_day_opportunity",
            "opportunity_score", "positive_momentum_score", "confirmed_breakout_score",
            "trade_confidence", "continuation_probability", "setup_score", "trend_score",
            "momentum_score", "liquidity_score", "risk_penalty", "rsi14", "ret_1d",
            "ret_5d", "rel_volume",
        ]
        print(scores[[c for c in show_cols if c in scores.columns]].head(self.config.top_n).to_string(index=False))

        acct = self.broker.get_account_snapshot()
        equity = max(float(acct.equity), 1.0)
        positions = {normalize_symbol(p.symbol): p for p in self.broker.get_all_open_positions()}

        strict_mask = (
            (scores["setup_score"] >= self.config.min_setup_score)
            & (scores["trade_confidence"] >= self.config.min_buy_confidence)
            & (scores["dollar_volume20"] >= self.config.min_avg_dollar_volume)
            & (scores["risk_penalty"] <= 0.55)
            & (scores["rsi14"] <= self.config.max_overbought_rsi)
        )

        if self.config.opportunity_scan_enabled:
            opportunity_mask = (
                scores["is_positive_day_opportunity"].fillna(False).astype(bool)
                & (scores["ret_1d"] >= self.config.min_daily_gain_pct)
                & (scores["opportunity_score"] >= self.config.min_opportunity_score)
            )
            if self.config.require_positive_momentum_or_breakout:
                opportunity_mask &= (
                    (scores["positive_momentum_score"] >= self.config.min_opportunity_score)
                    | (scores["confirmed_breakout_score"] >= self.config.min_breakout_confirmation_score)
                )
            strict_mask |= opportunity_mask

        if self.config.require_positive_daily_move:
            strict_mask &= scores["positive_daily_move"].fillna(0).astype(bool)

        if self.config.avoid_bearish_regime_for_longs:
            strict_mask &= scores["regime_score"] >= self.config.min_regime_score_for_longs
            strict_mask &= scores["regime_label"] != "BEARISH"

        candidates = scores[strict_mask].head(self.config.top_n).copy()

        if candidates.empty and self.config.allow_relaxed_fallback:
            print("[scanner] strict filters returned no candidates; applying relaxed fallback filters")
            relaxed_mask = (
                (scores["setup_score"] >= self.config.fallback_min_setup_score)
                & (scores["trade_confidence"] >= self.config.fallback_min_buy_confidence)
                & (scores["dollar_volume20"] >= self.config.min_avg_dollar_volume)
                & (scores["risk_penalty"] <= 0.70)
                & (scores["rsi14"] <= min(85.0, self.config.max_overbought_rsi + 7.0))
                & (scores["regime_score"] >= 0.30)
            )
            candidates = scores[relaxed_mask].head(self.config.top_n).copy()

        candidates = self._annotate_intraday_entries(candidates)

        no_reentry_symbols = load_no_reentry_symbols()
        if no_reentry_symbols:
            before = len(candidates)
            candidates = candidates[~candidates["symbol"].map(normalize_symbol).isin(no_reentry_symbols)].copy()
            skipped = before - len(candidates)
            if skipped:
                print(f"[scanner] skipped {skipped} same-day no-reentry candidate(s): {sorted(no_reentry_symbols)}")

        rows: list[dict[str, Any]] = []
        for _, r in candidates.iterrows():
            symbol = normalize_symbol(r["symbol"])
            price = to_float(r.get("entry_price"), to_float(r["close"]))
            market_price = to_float(r["close"])
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

            if delta_weight < self.executor_config.min_weight_delta:
                continue

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
                    "market_price": market_price,
                    "entry_price": to_float(r.get("entry_price"), price),
                    "entry_type": str(r.get("entry_type", "")),
                    "nearest_support": to_float(r.get("nearest_support")),
                    "nearest_resistance": to_float(r.get("nearest_resistance")),
                    "stop_price": to_float(r.get("stop_price")),
                    "target_price": to_float(r.get("target_price")),
                    "risk_reward_ratio": to_float(r.get("risk_reward_ratio")),
                    "entry_notes": str(r.get("entry_notes", "")),
                    "support_levels": json.dumps(r.get("support_levels", []), default=json_default),
                    "resistance_levels": json.dumps(r.get("resistance_levels", []), default=json_default),
                    "reason": f"scanner_{r['setup_type']}_candidate",
                    "target_weight": target_weight,
                    "current_weight": current_weight,
                    "delta_weight": delta_weight,
                    "trade_confidence": float(r["trade_confidence"]),
                    "confidence_score": float(r["confidence_score"]),
                    "continuation_probability": float(r["continuation_probability"]),
                    "ml_probability": float(r["ml_probability"]),
                    "setup_score": float(r["setup_score"]),
                    "trend_score": float(r["trend_score"]),
                    "momentum_score": float(r["momentum_score"]),
                    "liquidity_score": float(r["liquidity_score"]),
                    "volatility_score": float(r["volatility_score"]),
                    "opportunity_score": float(r["opportunity_score"]),
                    "positive_momentum_score": float(r["positive_momentum_score"]),
                    "confirmed_breakout_score": float(r["confirmed_breakout_score"]),
                    "is_positive_day_opportunity": bool(r["is_positive_day_opportunity"]),
                    "risk_penalty": float(r["risk_penalty"]),
                    "regime_score": float(r["regime_score"]),
                    "regime_label": str(r["regime_label"]),
                    "reversal_score": float(r["reversal_score"]),
                    "rsi14": to_float(r.get("rsi14")),
                    "ret_1d": to_float(r.get("ret_1d")),
                    "ret_5d": to_float(r.get("ret_5d")),
                    "ret_10d": to_float(r.get("ret_10d")),
                    "rel_volume": to_float(r.get("rel_volume")),
                    "macd": to_float(r.get("macd")),
                    "macd_signal": to_float(r.get("macd_signal")),
                    "macd_hist": to_float(r.get("macd_hist")),
                    "rolling_std10": to_float(r.get("rolling_std10")),
                    "rolling_std20": to_float(r.get("rolling_std20")),
                }
            )

        rebalance = pd.DataFrame(rows)
        exits = self._update_high_water_and_exits(scores, equity)

        rebalance.to_csv(DEFAULT_REBALANCE_ORDERS, index=False)
        exits.to_csv(DEFAULT_EXIT_ORDERS, index=False)
        print(f"[scanner] wrote rebalance_orders={DEFAULT_REBALANCE_ORDERS} rows={len(rebalance)}")
        print(f"[scanner] wrote exit_orders={DEFAULT_EXIT_ORDERS} rows={len(exits)}")
        return rebalance, exits, scores


# =============================================================================
# Executor
# =============================================================================

class ProductionOrderExecutor:
    def __init__(self, config: ExecutorConfig):
        self.config = config
        self.journal = CsvJournal(JOURNAL_PATH)
        self.broker = AlpacaBroker(ENV_PATH, paper=config.paper)
        self.scanner = MomentumSetupScanner(self.broker, self.config.scanner, self.config) if self.config.scanner.enabled else None
        self.last_order_time = 0.0
        self.last_heartbeat_time = 0.0
        self.account_history: list[dict[str, Any]] = []

    def _in_cooldown(self) -> bool:
        return (time.time() - self.last_order_time) < self.config.cooldown_seconds

    def _mark_order(self) -> None:
        self.last_order_time = time.time()

    def _record_account_snapshot(self, label: str, cycle_id: str) -> dict[str, Any]:
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
        return sum(max(0.0, float(p.market_value)) for p in self.broker.get_all_open_positions()) / equity

    def _risk_profile(self, equity: float) -> AccountRiskProfile:
        return build_account_risk_profile(equity)

    def _cash_buffer(self, equity: float) -> float:
        profile = self._risk_profile(equity)
        return max(profile.min_cash_buffer, equity * profile.cash_buffer_pct)

    def _max_order_notional(self, equity: float) -> float:
        profile = self._risk_profile(equity)
        return min(profile.max_order_dollars, equity * profile.max_single_order_notional_pct)

    def _market_is_open_or_wait(self) -> bool:
        if self.config.allow_closed_market or not self.config.require_market_open:
            print("[clock] market-open guard disabled")
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
            print("[clock] market is closed. Execution skipped")
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

        print("[clock] market did not open within wait limit. Execution skipped")
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

    def _kill_switch_allows_execution(self, equity: float) -> tuple[bool, str, float]:
        if not self.config.enable_kill_switch:
            return True, "kill_switch_disabled", 0.0

        today = now_utc().date().isoformat()
        state = self._read_kill_state()

        if state.get("date") != today:
            state = {
                "date": today,
                "start_equity": equity,
                "kill_triggered": False,
                "triggered_at": "",
                "reason": "",
            }
            self._write_kill_state(state)

        start_equity = to_float(state.get("start_equity"), equity)
        profile = self._risk_profile(start_equity)

        # If an older run latched a kill switch while using stale settings,
        # clear it when the account is currently above the active dynamic floor.
        # Real daily drawdown/loss triggers remain latched for the trading day.
        if (
            state.get("kill_triggered")
            and str(state.get("reason", "")) == "kill_switch_equity_floor"
            and equity > profile.kill_switch_equity_floor
        ):
            state.update(
                {
                    "kill_triggered": False,
                    "triggered_at": "",
                    "reason": "",
                    "cleared_at": now_utc().isoformat(),
                    "clear_reason": "equity_above_dynamic_floor",
                }
            )
            self._write_kill_state(state)
            write_event("kill_switch_cleared", state)

        pnl = equity - start_equity
        dd_pct = pnl / max(start_equity, 1e-9)

        reason = "ok"
        blocked = False

        if state.get("kill_triggered"):
            blocked = True
            reason = str(state.get("reason", "kill_switch_already_triggered"))
        elif equity <= profile.kill_switch_equity_floor:
            blocked = True
            reason = "kill_switch_equity_floor"
        elif dd_pct <= profile.daily_drawdown_limit:
            blocked = True
            reason = "daily_drawdown_limit"
        elif pnl <= -abs(profile.max_daily_loss_dollars):
            blocked = True
            reason = "max_daily_loss_dollars"
        elif (-dd_pct) >= abs(profile.max_daily_loss_pct):
            blocked = True
            reason = "max_daily_loss_pct"

        print(
            f"[kill-switch] band={profile.band} start_equity=${start_equity:,.2f} equity=${equity:,.2f} "
            f"floor=${profile.kill_switch_equity_floor:,.2f} daily_pnl=${pnl:,.2f} "
            f"daily_return={dd_pct:.2%} status={reason}"
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

    def _safe_read_order_csv(self, path: Path, required: bool = False) -> pd.DataFrame:
        if not path.exists():
            if required:
                raise RuntimeError(f"Required order file not found: {path}")
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
        return (
            self._safe_read_order_csv(self.config.rebalance_orders_path, required=True),
            self._safe_read_order_csv(self.config.exit_orders_path, required=False),
        )

    def _normalize_order_frame(self, orders: Optional[pd.DataFrame], default_reason: str = "") -> pd.DataFrame:
        cols = [
            "symbol", "side", "delta_shares", "delta_dollars", "price", "reason",
            "target_weight", "current_weight", "delta_weight", "trade_confidence",
            "confidence", "confidence_score", "continuation_prob", "continuation_probability",
            "explosive_alpha_probability",
        ]
        if orders is None or orders.empty:
            return pd.DataFrame(columns=cols)

        out = orders.copy()
        for col in cols:
            if col not in out.columns:
                out[col] = default_reason if col == "reason" else np.nan

        out["symbol"] = out["symbol"].map(safe_symbol)
        out["side"] = out["side"].astype(str).str.lower().str.strip()
        out["reason"] = out["reason"].fillna(default_reason).astype(str)

        numeric_cols = [c for c in cols if c not in {"symbol", "side", "reason"}]
        for col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        return out[(out["symbol"] != "") & out["side"].isin(["buy", "sell"])].reset_index(drop=True)

    def _confidence_for_row(self, row: pd.Series) -> float:
        for col in [
            "trade_confidence", "confidence", "confidence_score", "continuation_prob",
            "continuation_probability", "explosive_alpha_probability",
        ]:
            value = row.get(col, np.nan)
            if pd.notna(value):
                val = to_float(value, 0.5)
                if val > 1:
                    val /= 100.0
                return float(np.clip(val, 0.0, 1.0))
        return 0.5

    def _confidence_multiplier(self, confidence: float) -> float:
        return float(np.clip(0.50 + confidence, 0.35, 1.15))

    def _build_order_plan(self, rebalance: pd.DataFrame, exits: pd.DataFrame, equity: float) -> pd.DataFrame:
        rb = self._normalize_order_frame(rebalance)
        ex = self._normalize_order_frame(exits, default_reason="exit_signal")
        plan = pd.concat([ex, rb], ignore_index=True)

        if plan.empty:
            return pd.DataFrame()

        # Never re-enter a symbol on the same day it was exited.
        # This applies both to symbols exited earlier today and to symbols that
        # have a sell order in the current plan.
        no_reentry_symbols = load_no_reentry_symbols()
        same_cycle_exit_symbols = set(plan.loc[plan["side"] == "sell", "symbol"].map(safe_symbol))
        blocked_buy_symbols = {sym for sym in no_reentry_symbols.union(same_cycle_exit_symbols) if sym}
        if blocked_buy_symbols:
            buy_block_mask = (plan["side"] == "buy") & plan["symbol"].map(safe_symbol).isin(blocked_buy_symbols)
            blocked_count = int(buy_block_mask.sum())
            if blocked_count:
                print(f"[no-reentry] removed {blocked_count} buy order(s) for same-day exited symbols: {sorted(blocked_buy_symbols)}")
                plan = plan[~buy_block_mask].copy()
                if plan.empty:
                    return pd.DataFrame()

        plan["abs_delta_shares"] = plan["delta_shares"].abs().fillna(0)
        plan["abs_delta_dollars"] = plan["delta_dollars"].abs().fillna(0)
        plan = plan[(plan["side"] == "sell") | (plan["abs_delta_dollars"] >= self.config.min_notional_to_trade)].copy()
        if plan.empty:
            return pd.DataFrame()

        plan["confidence_used"] = plan.apply(self._confidence_for_row, axis=1)
        plan["confidence_multiplier"] = plan["confidence_used"].map(self._confidence_multiplier)

        buy_mask = plan["side"] == "buy"
        plan.loc[buy_mask, "abs_delta_dollars"] *= plan.loc[buy_mask, "confidence_multiplier"]

        max_order_notional = self._max_order_notional(equity)
        plan.loc[buy_mask, "capped_dollars"] = plan.loc[buy_mask, "abs_delta_dollars"].clip(upper=max_order_notional)
        plan.loc[~buy_mask, "capped_dollars"] = plan.loc[~buy_mask, "abs_delta_dollars"]

        plan["order_qty"] = plan["abs_delta_shares"].fillna(0).astype(int)
        qty_from_dollars = (plan["order_qty"] <= 0) & (plan["price"] > 0)
        plan.loc[qty_from_dollars, "order_qty"] = (
            plan.loc[qty_from_dollars, "capped_dollars"] / plan.loc[qty_from_dollars, "price"]
        ).fillna(0).astype(int)

        plan = plan[(plan["order_qty"] > 0) & (plan["order_qty"] <= self.config.max_single_order_qty)].copy()
        if plan.empty:
            return pd.DataFrame()

        side_rank = {"sell": 0, "buy": 1} if self.config.sell_first else {"buy": 0, "sell": 1}
        plan["side_rank"] = plan["side"].map(side_rank).fillna(9)
        plan = plan.sort_values(["side_rank", "capped_dollars"], ascending=[True, False]).reset_index(drop=True)

        stamp = run_id()
        plan["client_order_id"] = [
            f"{self.config.order_tag_prefix}_{stamp}_{i}_{row.symbol}_{row.side}"[:48]
            for i, row in plan.iterrows()
        ]
        plan["planned_at"] = now_utc().isoformat()
        return plan

    def _poll_fill(self, order_id: str, planned_price: float) -> dict[str, Any]:
        last_order = None
        for _ in range(6):
            try:
                last_order = self.broker.get_order_by_id(order_id)
            except Exception as exc:
                return {"fill_status": "fill_lookup_error", "filled_qty": 0.0, "filled_avg_price": 0.0, "slippage_pct": 0.0, "message": str(exc)}

            status = str(getattr(last_order, "status", "")).lower()
            if status in {"filled", "partially_filled", "canceled", "rejected", "expired"}:
                break
            time.sleep(2.0)

        if last_order is None:
            return {"fill_status": "unknown", "filled_qty": 0.0, "filled_avg_price": 0.0, "slippage_pct": 0.0}

        filled_qty = to_float(getattr(last_order, "filled_qty", 0.0))
        filled_avg_price = to_float(getattr(last_order, "filled_avg_price", 0.0))
        status = str(getattr(last_order, "status", ""))
        slippage_pct = filled_avg_price / planned_price - 1.0 if planned_price > 0 and filled_avg_price > 0 else 0.0

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
        acct_before: dict[str, Any],
        acct_after: dict[str, Any],
        message: str,
        order_id: str = "",
        order_status: str = "",
        fill_info: Optional[dict[str, Any]] = None,
    ) -> None:
        row = order_row.to_dict() if isinstance(order_row, pd.Series) else dict(order_row)
        fill_info = fill_info or {}
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
        results: list[dict[str, Any]] = []
        if plan.empty:
            return pd.DataFrame()

        for _, row in plan.iterrows():
            symbol = safe_symbol(row.get("symbol"))
            side = str(row.get("side", "")).lower()
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

            skip_message = ""

            if qty <= 0:
                skip_message = "qty <= 0"
            elif self.broker.has_open_order_for_symbol(symbol):
                skip_message = "existing open order for symbol"

            if not skip_message and side == "buy":
                cash_buffer = self._cash_buffer(equity)
                max_order_notional = self._max_order_notional(equity)
                profile = self._risk_profile(equity)
                gross_room = max(0.0, (profile.max_gross_exposure - gross_before) * equity)

                if buying_power < self.config.min_buying_power:
                    skip_message = "buying_power_below_minimum"
                elif cash - est_notional < cash_buffer:
                    skip_message = f"cash_buffer_violation cash={cash:.2f} projected={cash - est_notional:.2f} required={cash_buffer:.2f}"
                else:
                    if est_notional > max_order_notional:
                        qty = int(max_order_notional // price) if price > 0 else 0
                        est_notional = qty * price
                        print(f"[clip] BUY {symbol}: clipped to qty={qty} by max_order_notional")
                    if est_notional > gross_room:
                        qty = int(gross_room // price) if price > 0 else 0
                        est_notional = qty * price
                        print(f"[clip] BUY {symbol}: clipped to qty={qty} by gross exposure room")
                    if qty <= 0 or est_notional < profile.min_notional_to_trade:
                        skip_message = "buy_clipped_below_min_notional"

            if skip_message:
                print(f"[skip] {side.upper()} {qty} {symbol}: {skip_message}")
                result = {
                    "timestamp_utc": now_utc().isoformat(),
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "status": "skipped",
                    "message": skip_message,
                }
                results.append(result)
                acct_after = self._record_account_snapshot("after_skip", cycle_id)
                acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
                self._journal_order(cycle_id, "SKIP", row, acct_before, acct_after, skip_message)
                continue

            try:
                if side == "buy":
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
                    cycle_id,
                    "ORDER",
                    {**row.to_dict(), "order_qty": qty, "capped_dollars": est_notional},
                    acct_before,
                    acct_after,
                    reason or "submitted",
                    order_id,
                    status,
                    fill_info,
                )
                write_event("order_submitted", result)
                if side == "sell":
                    fill_status = str(fill_info.get("fill_status", status)).lower()
                    if fill_status in {"filled", "partially_filled", "submitted", "accepted", "new"} or to_float(fill_info.get("filled_qty", 0.0)) > 0:
                        mark_no_reentry_symbol(symbol, reason=reason or "sell_order_submitted")
                time.sleep(0.35)

            except APIError as exc:
                message = str(exc)
                print(f"[error] APIError {side.upper()} {qty} {symbol}: {message}")
                acct_after = self._record_account_snapshot("after_error", cycle_id)
                acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
                result = {"timestamp_utc": now_utc().isoformat(), "symbol": symbol, "side": side, "qty": qty, "status": "api_error", "message": message, "order_id": ""}
                results.append(result)
                self._journal_order(cycle_id, "ERROR", row, acct_before, acct_after, message)
                write_event("order_api_error", result)

            except Exception as exc:
                message = str(exc)
                print(f"[error] submit failed {side.upper()} {qty} {symbol}: {message}")
                acct_after = self._record_account_snapshot("after_error", cycle_id)
                acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
                result = {"timestamp_utc": now_utc().isoformat(), "symbol": symbol, "side": side, "qty": qty, "status": "error", "message": message, "order_id": ""}
                results.append(result)
                self._journal_order(cycle_id, "ERROR", row, acct_before, acct_after, message)
                write_event("order_error", result)

        return pd.DataFrame(results)

    def run_once_from_frames(self, rebalance_orders: pd.DataFrame, exit_orders: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, pd.DataFrame, AccountSnapshot, pd.DataFrame]:
        cycle_id = run_id()
        acct_start = self._record_account_snapshot("cycle_start", cycle_id)
        equity = float(acct_start["equity"])
        cash = float(acct_start["cash"])
        gross = self._gross_exposure(equity)
        acct_start["gross_exposure"] = gross
        positions_before = pd.DataFrame([asdict(p) for p in self.broker.get_all_open_positions()])

        if not self._market_is_open_or_wait():
            results = pd.DataFrame([{"timestamp_utc": now_utc().isoformat(), "status": "skipped_market_closed", "message": "market is closed"}])
            save_df(pd.DataFrame(), LOG_DIR / f"order_plan_{cycle_id}.csv")
            save_df(results, LOG_DIR / f"order_results_{cycle_id}.csv")
            self._write_account_equity_curve()
            return pd.DataFrame(), results, self.broker.get_account_snapshot(), positions_before

        allowed, kill_reason, daily_return = self._kill_switch_allows_execution(equity)
        if not allowed:
            print(f"[kill-switch] execution blocked: {kill_reason}")
            results = pd.DataFrame([{"timestamp_utc": now_utc().isoformat(), "status": "skipped_kill_switch", "message": kill_reason, "daily_return": daily_return}])
            save_df(pd.DataFrame(), LOG_DIR / f"order_plan_{cycle_id}.csv")
            save_df(results, LOG_DIR / f"order_results_{cycle_id}.csv")
            self._write_account_equity_curve()
            return pd.DataFrame(), results, self.broker.get_account_snapshot(), positions_before

        if self._in_cooldown():
            print("[cooldown] execution blocked by cooldown")
            results = pd.DataFrame([{"timestamp_utc": now_utc().isoformat(), "status": "skipped_cooldown", "message": "cooldown active"}])
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
            save_df(plan, LOG_DIR / f"order_plan_{cycle_id}.csv")
            save_df(pd.DataFrame(), LOG_DIR / f"order_results_{cycle_id}.csv")
            self._write_account_equity_curve()
            return plan, pd.DataFrame(), self.broker.get_account_snapshot(), positions_before

        print("\n===== ORDER PLAN =====")
        display_cols = ["symbol", "side", "order_qty", "price", "abs_delta_dollars", "capped_dollars", "confidence_used", "confidence_multiplier", "reason"]
        print(plan[[c for c in display_cols if c in plan.columns]].to_string(index=False))
        save_df(plan, LOG_DIR / f"order_plan_{cycle_id}.csv")

        results = self._execute_order_plan(plan, cycle_id)
        save_df(results, LOG_DIR / f"order_results_{cycle_id}.csv")

        positions_after = pd.DataFrame([asdict(p) for p in self.broker.get_all_open_positions()])
        save_df(positions_after, LOG_DIR / f"positions_after_{cycle_id}.csv")

        acct_after = self._record_account_snapshot("cycle_end", cycle_id)
        acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
        self._write_account_equity_curve()

        save_json(
            {
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
            },
            LOG_DIR / f"execution_summary_{cycle_id}.json",
        )

        return plan, results, self.broker.get_account_snapshot(), positions_before

    def run_cycle_from_files(self) -> tuple[pd.DataFrame, pd.DataFrame, AccountSnapshot, pd.DataFrame]:
        if self.scanner is not None:
            try:
                rebalance, exits, _scores = self.scanner.build_orders()
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
            f"cash=${acct.cash:,.2f} | buying_power=${acct.buying_power:,.2f} | gross={gross:.2%}"
        )
        self.last_heartbeat_time = now_ts

    def run(self) -> None:
        print("[executor] 13K account executor starting")
        print(f"[executor] journal={JOURNAL_PATH}")
        print(f"[executor] equity_curve={EQUITY_CURVE_PATH}")
        print(f"[executor] paper={self.config.paper}")
        try:
            acct = self.broker.get_account_snapshot()
            profile = self._risk_profile(acct.equity)
            print(f"[executor] account_band={profile.band}")
            print(f"[executor] equity=${acct.equity:,.2f}")
            print(f"[executor] cash_buffer_pct={profile.cash_buffer_pct:.2%}")
            print(f"[executor] min_cash_buffer=${profile.min_cash_buffer:,.2f}")
            print(f"[executor] max_gross_exposure={profile.max_gross_exposure:.2%}")
            print(f"[executor] max_position_weight={profile.max_position_weight:.2%}")
            print(f"[executor] max_order_dollars=${profile.max_order_dollars:,.2f}")
            print(f"[executor] max_daily_loss=${profile.max_daily_loss_dollars:,.2f}/{profile.max_daily_loss_pct:.2%}")
            print(f"[executor] dynamic_kill_floor=${profile.kill_switch_equity_floor:,.2f}")
        except Exception as exc:
            print(f"[executor] risk profile unavailable at startup: {exc}")

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
            time.sleep(max(1, int(self.config.poll_seconds)))


# =============================================================================
# Backward-compatible module functions
# =============================================================================

def get_trading_client(config: Optional[ExecutorConfig] = None) -> TradingClient:
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
    return pd.DataFrame(rows)


def run_execution(
    rebalance_orders: pd.DataFrame,
    exit_orders: Optional[pd.DataFrame] = None,
    config: Optional[ExecutorConfig] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, AccountSnapshot, pd.DataFrame]:
    config = config or ExecutorConfig(poll_once=True)
    config.poll_once = True
    executor = ProductionOrderExecutor(config)
    return executor.run_once_from_frames(rebalance_orders, exit_orders if exit_orders is not None else pd.DataFrame())


def save_execution_results(
    order_plan: pd.DataFrame,
    results: pd.DataFrame,
    output_dir: Optional[Path | str] = None,
    run_stamp: Optional[str] = None,
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
# Zero-argument startup
# =============================================================================

def build_default_config() -> ExecutorConfig:
    """
    Default production-style paper configuration for a roughly $13K account.

    Run with:
        python run.py

    No command-line arguments are required.
    """
    scanner_config = ScannerConfig(
        enabled=True,
        universe=None,
        use_sp500_fallback=True,
        max_universe_symbols=500,
        top_n=30,
        min_setup_score=0.62,
        min_buy_confidence=0.58,
        min_avg_dollar_volume=5_000_000.0,
        opportunity_scan_enabled=True,
        min_daily_gain_pct=0.005,
        min_opportunity_score=0.60,
        min_breakout_confirmation_score=0.55,
        require_positive_momentum_or_breakout=True,
        base_target_weight=0.03,
        max_target_weight=0.10,
        broad_market_scan=True,
        stop_loss_pct=-0.025,
        profit_lock_trigger_pct=0.04,
        profit_giveback_pct=0.015,
    )

    return ExecutorConfig(
        paper=True,
        live=False,
        poll_once=False,
        poll_seconds=POLL_SECONDS,
        rebalance_orders_path=DEFAULT_REBALANCE_ORDERS,
        exit_orders_path=DEFAULT_EXIT_ORDERS,
        require_market_open=True,
        wait_for_market_open=False,
        allow_closed_market=False,
        cancel_open_orders_first=False,
        sell_first=True,
        cash_buffer_pct=DEFAULT_CASH_BUFFER_PCT,
        min_cash_buffer=MIN_CASH_BUFFER,
        min_buying_power=MIN_BUYING_POWER,
        min_notional_to_trade=MIN_NOTIONAL_TO_TRADE,
        min_weight_delta=MIN_WEIGHT_DELTA,
        max_gross_exposure=MAX_GROSS_EXPOSURE,
        max_position_weight=MAX_POSITION_WEIGHT,
        max_single_order_notional_pct=MAX_SINGLE_ORDER_NOTIONAL_PCT,
        max_order_dollars=MAX_ORDER_DOLLARS,
        max_single_order_qty=MAX_SINGLE_ORDER_QTY,
        cooldown_seconds=COOLDOWN_SECONDS,
        enable_kill_switch=True,
        daily_drawdown_limit=DAILY_DRAWDOWN_LIMIT,
        max_daily_loss_pct=MAX_DAILY_LOSS_PCT,
        max_daily_loss_dollars=MAX_DAILY_LOSS_DOLLARS,
        kill_switch_equity_floor=KILL_SWITCH_EQUITY_FLOOR,
        order_wait_seconds=ORDER_WAIT_SECONDS,
        time_in_force=TimeInForce.DAY,
        order_tag_prefix="qq_13k",
        scanner=scanner_config,
    )


def main() -> None:
    config = build_default_config()
    executor = ProductionOrderExecutor(config)
    executor.run()


if __name__ == "__main__":
    main()
