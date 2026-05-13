#!/usr/bin/env python3
"""
run.py

Clean 13K-account Alpaca paper executor with built-in momentum/reversal scanner.

Default behavior:
    python run.py

What it does:
    - loads Alpaca credentials from .env
    - refreshes llm_sentiment_universe_selector.py at the start of the trading day
    - starts the session-planning pipeline at/after 4:00 AM ET
    - builds market_context.json from market_context_builder.py
    - builds premarket_trade_plan.json from premarket_trade_planner.py
    - builds regime_position_plan.csv from production_regime_sizing_model.py
    - trains/refreshes the intraday RL timing model from rl_intraday_structure_trainer.py
    - gates entries/exits with the RL timing policy before Alpaca execution
    - scans the LLM-selected universe using Alpaca daily/intraday bars when regime plan is unavailable
    - uses market context + premarket trade plan as buy-side execution gates
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
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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
POSITION_MONITOR_PATH = LOG_DIR / "active_position_monitor.csv"

LLM_SENTIMENT_UNIVERSE_PATH = OUT_DIR / "latest_llm_sentiment_universe.json"
LLM_SENTIMENT_DISCOVERY_PATH = OUT_DIR / "latest_llm_candidate_discovery.json"
LOCKED_UNIVERSE_PATH = PROJECT_ROOT / "locked_trading_universe.txt"
LLM_SENTIMENT_SELECTOR_PATH = PROJECT_ROOT / "llm_sentiment_universe_selector.py"
MARKET_CONTEXT_BUILDER_PATH = PROJECT_ROOT / "market_context_builder.py"
PREMARKET_TRADE_PLANNER_PATH = PROJECT_ROOT / "premarket_trade_planner.py"
PRODUCTION_REGIME_SIZING_MODEL_PATH = PROJECT_ROOT / "production_regime_sizing_model.py"
RL_INTRADAY_STRUCTURE_TRAINER_PATH = PROJECT_ROOT / "rl_intraday_structure_trainer.py"
OPTIONS_BOOSTER_PATH = PROJECT_ROOT / "options_booster.py"

MARKET_CONTEXT_PATH = OUT_DIR / "market_context.json"
PREMARKET_TRADE_PLAN_PATH = OUT_DIR / "premarket_trade_plan.json"
INTRADAY_STRUCTURE_MAP_PATH = OUT_DIR / "intraday_structure_map.json"
INTRADAY_BARS_1M_PATH = OUT_DIR / "intraday_bars_1m.csv"
REGIME_POSITION_PLAN_PATH = OUT_DIR / "regime_position_plan.csv"
REGIME_UNIVERSE_PATH = OUT_DIR / "regime_universe_30.csv"
REGIME_ACTIONS_PATH = OUT_DIR / "regime_actions.csv"
RL_MODEL_PATH = OUT_DIR / "rl_intraday_structure_model.zip"
RL_SIGNAL_LOG_PATH = LOG_DIR / "rl_signal_log.csv"
SESSION_PREP_STATE_PATH = LOG_DIR / "session_planning_state.json"
OPTION_BOOSTER_CANDIDATES_PATH = OUT_DIR / "latest_option_booster_candidates.csv"

ET = ZoneInfo("America/New_York")


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

    # Active-position monitoring. Every cycle, held symbols are checked on
    # 1m/5m/15m bars to ensure they are still moving in the right direction.
    active_position_monitor_enabled: bool = True
    active_monitor_lookback_days: int = 2
    # Momentum systems should prune quickly. These thresholds are intentionally
    # stricter than a swing-trading hold model.
    min_hold_direction_score: float = 0.55
    hard_exit_direction_score: float = 0.40
    intraday_trailing_stop_pct: float = 0.010
    failed_breakout_loss_pct: float = -0.004
    failed_breakout_buffer_pct: float = 0.002
    monitor_exit_on_vwap_loss: bool = True
    monitor_exit_on_lower_high_lower_low: bool = True
    monitor_exit_on_failed_breakout: bool = True

    # Final pre-execution direction gate.
    # Buys must pass this immediately before order submission.
    # Sells/exits bypass this so risk can always be reduced.
    require_direction_confirmation_before_buy: bool = True
    min_pretrade_direction_score: float = 0.62
    min_pretrade_1m_score: float = 0.52
    min_pretrade_5m_score: float = 0.55
    min_pretrade_15m_score: float = 0.50
    require_price_above_vwap_for_buy: bool = True
    require_no_lower_low_for_buy: bool = True
    max_pretrade_pullback_pct: float = -0.006

    # ML next-bar direction prediction.
    ml_next_bar_enabled: bool = True
    ml_next_bar_timeframe_minutes: int = 5
    ml_next_bar_lookback_days: int = 20
    ml_next_bar_min_train_rows: int = 120
    ml_next_bar_min_buy_probability: float = 0.58
    ml_next_bar_min_pyramid_probability: float = 0.62
    ml_next_bar_max_exit_probability: float = 0.42

    # Pyramiding / loser scale-out behavior.
    pyramid_winners_enabled: bool = True
    pyramid_min_unrealized_plpc: float = 0.012
    pyramid_min_hold_direction_score: float = 0.62
    pyramid_add_weight: float = 0.025
    pyramid_max_adds_per_symbol_per_day: int = 2

    scale_out_losers_enabled: bool = True
    scale_out_loss_pct: float = -0.006
    scale_out_direction_score: float = 0.50
    scale_out_fraction: float = 0.50
    loser_full_exit_loss_pct: float = -0.015
    loser_full_exit_direction_score: float = 0.38

    # Universe source.
    # Default is LLM sentiment universe from llm_sentiment_universe_selector.py.
    use_llm_sentiment_universe: bool = True
    refresh_llm_sentiment_universe: bool = True
    min_llm_universe_confidence: float = 0.0

    # Morning orchestration / macro gate.
    market_context_enabled: bool = True
    refresh_market_context: bool = True
    halt_new_buys_on_market_context: bool = True
    allow_cautious_market_buys: bool = True

    # Premarket trade-plan gate.
    premarket_plan_enabled: bool = True
    refresh_premarket_trade_plan: bool = True
    require_valid_premarket_plan_for_buys: bool = False
    prefer_premarket_valid_symbols: bool = True

    # 4:00 AM session-planning pipeline.
    session_planning_enabled: bool = True
    session_planning_start_hour_et: int = 4
    session_planning_start_minute_et: int = 0
    session_planning_force_each_startup: bool = False
    run_market_context_builder: bool = True
    run_premarket_trade_planner: bool = True
    run_production_regime_sizing_model: bool = True
    run_rl_intraday_structure_trainer: bool = True
    session_planning_timeout_seconds: int = 900

    # Regime sizing + RL timing integration.
    use_regime_position_plan: bool = True
    prefer_regime_position_plan: bool = True
    use_rl_timing_gate: bool = True
    rl_min_enter_long_confidence: float = 0.65
    rl_min_exit_confidence: float = 0.55
    rl_min_enter_short_confidence: float = 0.70
    rl_gate_buys_only: bool = False
    rl_allow_hold_when_model_missing: bool = True
    rl_latest_bar_timeframe_minutes: int = 1
    rl_latest_bar_lookback_days: int = 5

    # Options booster handoff.
    options_booster_enabled: bool = True
    options_booster_confidence_threshold: float = 0.75
    options_booster_min_position_plpc: float = 0.0056
    options_booster_target_dte: int = 30
    options_booster_min_dte: int = 28
    options_booster_max_dte: int = 32
    options_booster_limit_price_source: str = "bid"

    # ML-style call selection: rank option contracts using Greeks + liquidity + TA.
    # This is a deterministic model/ranker by default; if options_booster.py exposes
    # richer chain data, these fields let that module select the best contract.
    options_ml_contract_selection_enabled: bool = True
    options_target_delta: float = 0.52
    options_min_delta: float = 0.40
    options_max_delta: float = 0.65
    options_max_spread_pct: float = 0.15
    options_min_open_interest: int = 50
    options_min_volume: int = 5
    options_max_theta_abs: float = 0.12
    options_min_gamma: float = 0.005
    options_max_iv_rank: float = 0.85
    options_greek_score_weight: float = 0.35
    options_technical_score_weight: float = 0.35
    options_liquidity_score_weight: float = 0.20
    options_dte_score_weight: float = 0.10
    options_booster_require_technical_confirmation: bool = True
    options_booster_min_direction_score: float = 0.64
    options_booster_min_ml_next_bar_probability: float = 0.60
    options_booster_once_per_cycle: bool = True
    options_booster_timeout_seconds: int = 180


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


def run_python_script(path: Path, args: Optional[list[str]] = None, timeout_seconds: int = 180) -> tuple[bool, str]:
    if not path.exists():
        return False, f"script_not_found: {path}"

    cmd = [sys.executable, str(path)] + list(args or [])
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        output = "\n".join(x for x in [proc.stdout.strip(), proc.stderr.strip()] if x)
        return proc.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"timeout_after_{timeout_seconds}_seconds"
    except Exception as exc:
        return False, str(exc)


def refresh_llm_sentiment_universe(timeout_seconds: int = 180) -> None:
    """Refresh latest_llm_sentiment_universe.json and locked_trading_universe.txt."""
    try:
        from llm_sentiment_universe_selector import build_sentiment_universe, write_outputs

        payload = build_sentiment_universe()
        write_outputs(payload)
        print(f"[llm-universe] refreshed via import symbols={len(payload.get('symbols', []))}")
        write_event("llm_universe_refreshed", {"symbols": payload.get("symbols", [])})
        return
    except Exception as exc:
        print(f"[llm-universe] import refresh failed; trying subprocess: {exc}")

    ok, output = run_python_script(LLM_SENTIMENT_SELECTOR_PATH, timeout_seconds=timeout_seconds)
    if ok:
        print("[llm-universe] refreshed via subprocess")
        if output:
            print(output[-2000:])
        write_event("llm_universe_refreshed_subprocess", {"output_tail": output[-2000:]})
    else:
        print(f"[llm-universe] refresh failed: {output}")
        write_event("llm_universe_refresh_failed", {"error": output})



def refresh_market_context(timeout_seconds: int = 240) -> dict[str, Any]:
    """Build daily market_context.json using market_context_builder.py."""
    try:
        from market_context_builder import build_market_context

        payload = build_market_context()
        print(
            "[market-context] refreshed via import "
            f"bias={payload.get('overall_market_bias')} "
            f"permission={payload.get('trade_permission')} "
            f"risk={payload.get('risk_level')}"
        )
        write_event(
            "market_context_refreshed",
            {
                "overall_market_bias": payload.get("overall_market_bias"),
                "trade_permission": payload.get("trade_permission"),
                "risk_level": payload.get("risk_level"),
            },
        )
        return payload
    except Exception as exc:
        print(f"[market-context] import refresh failed; trying subprocess: {exc}")

    ok, output = run_python_script(MARKET_CONTEXT_BUILDER_PATH, timeout_seconds=timeout_seconds)
    if ok:
        print("[market-context] refreshed via subprocess")
        if output:
            print(output[-2000:])
        write_event("market_context_refreshed_subprocess", {"output_tail": output[-2000:]})
    else:
        print(f"[market-context] refresh failed: {output}")
        write_event("market_context_refresh_failed", {"error": output})

    return read_json_file(MARKET_CONTEXT_PATH, {})


def refresh_premarket_trade_plan(timeout_seconds: int = 240) -> dict[str, Any]:
    """Build premarket_trade_plan.json using premarket_trade_planner.py."""
    try:
        from premarket_trade_planner import build_premarket_trade_plan

        payload = build_premarket_trade_plan()
        print(
            "[premarket-plan] refreshed via import "
            f"symbols={payload.get('symbol_count', 0)} valid={payload.get('valid_plan_count', 0)}"
        )
        write_event(
            "premarket_trade_plan_refreshed",
            {
                "symbol_count": payload.get("symbol_count"),
                "valid_plan_count": payload.get("valid_plan_count"),
                "valid_symbols": payload.get("valid_symbols", []),
            },
        )
        return payload
    except Exception as exc:
        print(f"[premarket-plan] import refresh failed; trying subprocess: {exc}")

    ok, output = run_python_script(PREMARKET_TRADE_PLANNER_PATH, timeout_seconds=timeout_seconds)
    if ok:
        print("[premarket-plan] refreshed via subprocess")
        if output:
            print(output[-2000:])
        write_event("premarket_trade_plan_refreshed_subprocess", {"output_tail": output[-2000:]})
    else:
        print(f"[premarket-plan] refresh failed: {output}")
        write_event("premarket_trade_plan_refresh_failed", {"error": output})

    return read_json_file(PREMARKET_TRADE_PLAN_PATH, {})


def load_market_context(refresh: bool = False) -> dict[str, Any]:
    if refresh or not MARKET_CONTEXT_PATH.exists():
        return refresh_market_context()
    return read_json_file(MARKET_CONTEXT_PATH, {})


def load_premarket_trade_plan(refresh: bool = False) -> dict[str, Any]:
    if refresh or not PREMARKET_TRADE_PLAN_PATH.exists():
        return refresh_premarket_trade_plan()
    return read_json_file(PREMARKET_TRADE_PLAN_PATH, {})




def session_planning_start_dt_et(hour: int = 4, minute: int = 0) -> datetime:
    now = datetime.now(ET)
    return now.replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)


def session_planning_window_is_open(config: ScannerConfig) -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    return now >= session_planning_start_dt_et(
        config.session_planning_start_hour_et,
        config.session_planning_start_minute_et,
    )


def wait_until_session_planning_window(config: ScannerConfig) -> None:
    if not config.session_planning_enabled:
        return
    if session_planning_window_is_open(config):
        return

    start_dt = session_planning_start_dt_et(
        config.session_planning_start_hour_et,
        config.session_planning_start_minute_et,
    )

    while datetime.now(ET) < start_dt:
        seconds_left = max(1.0, (start_dt - datetime.now(ET)).total_seconds())
        print(f"[session-prep] waiting until {start_dt.strftime('%H:%M')} ET; {int(seconds_left // 60)} minute(s) left")
        time.sleep(min(300, max(5, seconds_left)))


def _read_session_prep_state() -> dict[str, Any]:
    return read_json_file(SESSION_PREP_STATE_PATH, {"date_et": "", "steps": {}})


def _write_session_prep_state(state: dict[str, Any]) -> None:
    write_json_file(SESSION_PREP_STATE_PATH, state)


def _session_step_already_done(state: dict[str, Any], date_et: str, step_name: str) -> bool:
    if state.get("date_et") != date_et:
        return False
    step = state.get("steps", {}).get(step_name, {})
    return bool(step.get("ok"))


def _record_session_step(state: dict[str, Any], date_et: str, step_name: str, ok: bool, output: str = "") -> None:
    state["date_et"] = date_et
    steps = state.setdefault("steps", {})
    steps[step_name] = {
        "ok": bool(ok),
        "completed_at_utc": now_utc().isoformat(),
        "output_tail": str(output or "")[-3000:],
    }
    _write_session_prep_state(state)
    write_event(
        "session_planning_step_completed" if ok else "session_planning_step_failed",
        {"step": step_name, "ok": bool(ok), "output_tail": str(output or "")[-1200:]},
    )


def refresh_production_regime_sizing_model(timeout_seconds: int = 900) -> dict[str, Any]:
    """Build regime_position_plan.csv using production_regime_sizing_model.py."""
    try:
        from production_regime_sizing_model import main as regime_main

        result = regime_main()
        print("[regime-model] refreshed via import")
        write_event("regime_model_refreshed", {"method": "import", "result": str(result)[:500]})
    except Exception as exc:
        print(f"[regime-model] import refresh failed; trying subprocess: {exc}")
        ok, output = run_python_script(PRODUCTION_REGIME_SIZING_MODEL_PATH, timeout_seconds=timeout_seconds)
        if ok:
            print("[regime-model] refreshed via subprocess")
            if output:
                print(output[-2000:])
            write_event("regime_model_refreshed_subprocess", {"output_tail": output[-2000:]})
        else:
            print(f"[regime-model] refresh failed: {output}")
            write_event("regime_model_refresh_failed", {"error": output})

    return {
        "regime_position_plan_exists": REGIME_POSITION_PLAN_PATH.exists(),
        "regime_universe_exists": REGIME_UNIVERSE_PATH.exists(),
        "regime_actions_exists": REGIME_ACTIONS_PATH.exists(),
    }


def refresh_rl_intraday_structure_model(timeout_seconds: int = 900) -> dict[str, Any]:
    """Train/refresh rl_intraday_structure_model.zip using rl_intraday_structure_trainer.py."""
    ok, output = run_python_script(RL_INTRADAY_STRUCTURE_TRAINER_PATH, timeout_seconds=timeout_seconds)
    if ok:
        print("[rl-trainer] refreshed via subprocess")
        if output:
            print(output[-2500:])
        write_event("rl_intraday_model_refreshed", {"output_tail": output[-2500:]})
    else:
        print(f"[rl-trainer] refresh failed: {output}")
        write_event("rl_intraday_model_refresh_failed", {"error": output})

    return {
        "ok": ok,
        "model_exists": RL_MODEL_PATH.exists(),
        "metrics_exists": (OUT_DIR / "rl_intraday_training_metrics.json").exists(),
        "output_tail": output[-2000:] if output else "",
    }


def run_session_planning_pipeline(config: ScannerConfig, force: bool = False) -> dict[str, Any]:
    """
    Runs once per ET trading day, starting no earlier than 4:00 AM ET by default.

    Order:
        1. llm_sentiment_universe_selector.py
        2. market_context_builder.py
        3. premarket_trade_planner.py
        4. production_regime_sizing_model.py
        5. rl_intraday_structure_trainer.py
    """
    if not config.session_planning_enabled:
        return {"skipped": True, "reason": "session_planning_disabled"}

    wait_until_session_planning_window(config)

    date_et = datetime.now(ET).date().isoformat()
    state = _read_session_prep_state()
    if state.get("date_et") != date_et:
        state = {"date_et": date_et, "started_at_utc": now_utc().isoformat(), "steps": {}}
        _write_session_prep_state(state)

    if not force and not config.session_planning_force_each_startup and state.get("completed") is True:
        print(f"[session-prep] already completed for {date_et}")
        return state

    print(f"[session-prep] starting daily planning pipeline for {date_et}")
    write_event("session_planning_started", {"date_et": date_et})

    if config.use_llm_sentiment_universe and config.refresh_llm_sentiment_universe:
        step = "llm_sentiment_universe_selector"
        if force or not _session_step_already_done(state, date_et, step):
            try:
                refresh_llm_sentiment_universe(timeout_seconds=config.session_planning_timeout_seconds)
                _record_session_step(state, date_et, step, True, "refreshed")
            except Exception as exc:
                _record_session_step(state, date_et, step, False, str(exc))

    if config.run_market_context_builder and config.market_context_enabled:
        step = "market_context_builder"
        if force or not _session_step_already_done(state, date_et, step):
            try:
                payload = refresh_market_context(timeout_seconds=config.session_planning_timeout_seconds)
                _record_session_step(state, date_et, step, bool(payload), json.dumps(payload, default=json_default)[:3000])
            except Exception as exc:
                _record_session_step(state, date_et, step, False, str(exc))

    if config.run_premarket_trade_planner and config.premarket_plan_enabled:
        step = "premarket_trade_planner"
        if force or not _session_step_already_done(state, date_et, step):
            try:
                payload = refresh_premarket_trade_plan(timeout_seconds=config.session_planning_timeout_seconds)
                _record_session_step(state, date_et, step, bool(payload), json.dumps(payload, default=json_default)[:3000])
            except Exception as exc:
                _record_session_step(state, date_et, step, False, str(exc))

    if config.run_production_regime_sizing_model:
        step = "production_regime_sizing_model"
        if force or not _session_step_already_done(state, date_et, step):
            try:
                payload = refresh_production_regime_sizing_model(timeout_seconds=config.session_planning_timeout_seconds)
                _record_session_step(state, date_et, step, bool(payload.get("regime_position_plan_exists")), json.dumps(payload, default=json_default))
            except Exception as exc:
                _record_session_step(state, date_et, step, False, str(exc))

    if config.run_rl_intraday_structure_trainer:
        step = "rl_intraday_structure_trainer"
        if force or not _session_step_already_done(state, date_et, step):
            try:
                payload = refresh_rl_intraday_structure_model(timeout_seconds=config.session_planning_timeout_seconds)
                # Do not mark the whole session failed just because the model already exists but retraining failed.
                ok = bool(payload.get("ok") or payload.get("model_exists"))
                _record_session_step(state, date_et, step, ok, json.dumps(payload, default=json_default))
            except Exception as exc:
                _record_session_step(state, date_et, step, RL_MODEL_PATH.exists(), str(exc))

    state = _read_session_prep_state()
    state["completed"] = True
    state["completed_at_utc"] = now_utc().isoformat()
    _write_session_prep_state(state)
    write_event("session_planning_completed", state)
    print(f"[session-prep] completed for {date_et}")
    return state


def load_regime_position_plan() -> pd.DataFrame:
    if not REGIME_POSITION_PLAN_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(REGIME_POSITION_PLAN_PATH)
    except Exception as exc:
        print(f"[regime-plan] failed to read {REGIME_POSITION_PLAN_PATH}: {exc}")
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    if "symbol" not in df.columns:
        print(f"[regime-plan] missing symbol column: {REGIME_POSITION_PLAN_PATH}")
        return pd.DataFrame()
    df = df.copy()
    df["symbol"] = df["symbol"].map(normalize_symbol)
    return df[df["symbol"] != ""].reset_index(drop=True)


def _first_existing_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _regime_action_value(row: pd.Series) -> str:
    for col in ["trade_action", "action", "regime_action", "recommended_action", "signal"]:
        if col in row.index and pd.notna(row.get(col)):
            return str(row.get(col, "")).upper().strip()
    return ""


def _regime_target_weight(row: pd.Series, default: float = 0.0) -> float:
    for col in ["target_weight", "weight", "recommended_weight", "target_alloc", "allocation"]:
        if col in row.index and pd.notna(row.get(col)):
            val = to_float(row.get(col), default)
            if val > 1.0:
                val /= 100.0
            return float(np.clip(val, 0.0, 1.0))
    return default


def _regime_confidence(row: pd.Series, default: float = 0.55) -> float:
    for col in ["confidence", "trade_confidence", "confidence_score", "probability", "model_probability", "buy_probability"]:
        if col in row.index and pd.notna(row.get(col)):
            val = to_float(row.get(col), default)
            if val > 1.0:
                val /= 100.0
            return float(np.clip(val, 0.0, 1.0))
    return default


def _regime_price(row: pd.Series, fallback: float = 0.0) -> float:
    for col in ["price", "last_price", "close", "entry_price", "current_price"]:
        if col in row.index and pd.notna(row.get(col)):
            val = to_float(row.get(col), fallback)
            if val > 0:
                return val
    return fallback


def market_context_blocks_new_buys(context: dict[str, Any], allow_cautious: bool = True) -> tuple[bool, str]:
    if not context:
        return False, "market_context_missing"

    permission = str(context.get("trade_permission", "cautious")).lower().strip()
    bias = str(context.get("overall_market_bias", "mixed")).lower().strip()
    risk = str(context.get("risk_level", "medium")).lower().strip()

    if permission == "halt_new_buys":
        return True, f"market_context_halt_new_buys:bias={bias}:risk={risk}"
    if not allow_cautious and permission == "cautious":
        return True, f"market_context_cautious_block:bias={bias}:risk={risk}"
    if bias in {"risk_off", "bearish"} and risk == "high":
        return True, f"market_context_risk_off_high_risk:permission={permission}"

    return False, f"market_context_allows_buys:permission={permission}:bias={bias}:risk={risk}"


def valid_symbols_from_premarket_plan(payload: dict[str, Any]) -> set[str]:
    symbols = payload.get("valid_symbols", []) if isinstance(payload, dict) else []
    return {normalize_symbol(s) for s in symbols if normalize_symbol(s)}

def load_llm_sentiment_universe(refresh: bool = True, min_confidence: float = 0.0) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Load symbols and per-symbol metadata from llm_sentiment_universe_selector.py output."""
    if refresh:
        refresh_llm_sentiment_universe()

    payload: dict[str, Any] = {}
    if LLM_SENTIMENT_UNIVERSE_PATH.exists():
        payload = read_json_file(LLM_SENTIMENT_UNIVERSE_PATH, {})
    else:
        print(f"[llm-universe] missing {LLM_SENTIMENT_UNIVERSE_PATH}; falling back to locked universe/default list")

    metadata: dict[str, dict[str, Any]] = {}
    symbols: list[str] = []

    selections = payload.get("selections", []) if isinstance(payload, dict) else []
    if isinstance(selections, list):
        for item in selections:
            if not isinstance(item, dict):
                continue
            symbol = normalize_symbol(item.get("symbol"))
            confidence = to_float(item.get("confidence"), 0.0)
            if not symbol or confidence < min_confidence:
                continue
            metadata[symbol] = {
                "llm_rank": to_int(item.get("rank"), 999),
                "llm_confidence": confidence,
                "llm_catalyst_type": str(item.get("catalyst_type", "")),
                "llm_reason": str(item.get("reason", ""))[:700],
                "llm_market_tone": str(payload.get("market_tone", "")),
                "llm_summary": str(payload.get("summary", ""))[:700],
            }
            symbols.append(symbol)

    if not symbols and isinstance(payload.get("symbols"), list):
        symbols = [normalize_symbol(x) for x in payload.get("symbols", []) if normalize_symbol(x)]
        metadata = {s: {"llm_confidence": 0.50, "llm_rank": i + 1} for i, s in enumerate(symbols)}

    if not symbols and LOCKED_UNIVERSE_PATH.exists():
        symbols = [normalize_symbol(x) for x in LOCKED_UNIVERSE_PATH.read_text().splitlines() if normalize_symbol(x)]
        metadata = {s: {"llm_confidence": 0.50, "llm_rank": i + 1, "llm_source": "locked_trading_universe"} for i, s in enumerate(symbols)}

    symbols = [s for s in symbols if is_simple_alpaca_symbol(s)]
    symbols = list(dict.fromkeys(symbols))

    if symbols:
        print(f"[llm-universe] loaded symbols={len(symbols)} source={LLM_SENTIMENT_UNIVERSE_PATH}")
    else:
        print("[llm-universe] no valid LLM symbols loaded; using DEFAULT_SCAN_UNIVERSE")
        symbols = list(DEFAULT_SCAN_UNIVERSE)
        metadata = {s: {"llm_confidence": 0.50, "llm_rank": i + 1, "llm_source": "default_fallback"} for i, s in enumerate(symbols)}

    return symbols, metadata


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


def load_pyramid_state() -> dict[str, Any]:
    today = local_trading_day()
    state = read_json_file(LOG_DIR / "pyramid_state.json", {"date": today, "symbols": {}})
    if state.get("date") != today:
        state = {"date": today, "symbols": {}}
        write_json_file(LOG_DIR / "pyramid_state.json", state)
    if not isinstance(state.get("symbols"), dict):
        state["symbols"] = {}
    return state


def save_pyramid_state(state: dict[str, Any]) -> None:
    write_json_file(LOG_DIR / "pyramid_state.json", state)


def pyramid_add_count(symbol: str) -> int:
    state = load_pyramid_state()
    return int(to_int(state.get("symbols", {}).get(safe_symbol(symbol), {}).get("adds", 0)))


def mark_pyramid_add(symbol: str, reason: str = "pyramid_add") -> None:
    symbol = safe_symbol(symbol)
    if not symbol:
        return
    state = load_pyramid_state()
    symbols = state.setdefault("symbols", {})
    current = symbols.get(symbol, {}) if isinstance(symbols.get(symbol, {}), dict) else {}
    current["adds"] = to_int(current.get("adds", 0)) + 1
    current["last_added_at"] = now_utc().isoformat()
    current["reason"] = reason
    symbols[symbol] = current
    state["symbols"] = symbols
    save_pyramid_state(state)
    write_event("pyramid_add_marked", {"symbol": symbol, "adds": current["adds"], "reason": reason})


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
        self.llm_metadata_by_symbol: dict[str, dict[str, Any]] = {}

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
            self.llm_metadata_by_symbol = {}
        elif self.config.use_llm_sentiment_universe:
            raw_symbols, self.llm_metadata_by_symbol = load_llm_sentiment_universe(
                refresh=self.config.refresh_llm_sentiment_universe,
                min_confidence=self.config.min_llm_universe_confidence,
            )
        elif self.config.use_sp500_fallback or self.config.broad_market_scan:
            raw_symbols = get_sp500_fallback_symbols()
            self.llm_metadata_by_symbol = {}
        else:
            raw_symbols = DEFAULT_SCAN_UNIVERSE
            self.llm_metadata_by_symbol = {}

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

    def _intraday_direction_metrics(self, symbol: str, frame: pd.DataFrame, minutes: int, avg_entry_price: float) -> dict[str, Any]:
        if frame.empty or len(frame) < 5:
            return {
                f"{minutes}m_direction_score": 0.5,
                f"{minutes}m_ret": 0.0,
                f"{minutes}m_vwap": 0.0,
                f"{minutes}m_above_vwap": True,
                f"{minutes}m_lower_highs": False,
                f"{minutes}m_lower_lows": False,
                f"{minutes}m_below_fast_sma": False,
                f"{minutes}m_below_slow_sma": False,
                f"{minutes}m_failed_reclaim": False,
            }

        g = frame.sort_values("timestamp").copy()
        close = g["close"].astype(float)
        high = g["high"].astype(float)
        low = g["low"].astype(float)
        volume = g["volume"].astype(float).replace(0, np.nan)

        last_close = float(close.iloc[-1])
        ret_n = last_close / max(float(close.iloc[0]), 1e-9) - 1.0
        ret_recent = last_close / max(float(close.iloc[-min(len(close), 6)]), 1e-9) - 1.0
        sma_fast = float(close.rolling(min(5, len(close))).mean().iloc[-1])
        sma_slow = float(close.rolling(min(15, len(close))).mean().iloc[-1])
        vwap = float((close * volume).sum() / max(volume.sum(), 1e-9)) if volume.notna().any() else last_close
        above_vwap = last_close >= vwap
        above_entry = last_close >= avg_entry_price if avg_entry_price > 0 else True
        below_fast_sma = last_close < sma_fast
        below_slow_sma = last_close < sma_slow

        recent_highs = high.tail(min(10, len(high))).reset_index(drop=True)
        recent_lows = low.tail(min(10, len(low))).reset_index(drop=True)
        lower_highs = bool(len(recent_highs) >= 4 and recent_highs.iloc[-1] < recent_highs.iloc[-3])
        lower_lows = bool(len(recent_lows) >= 4 and recent_lows.iloc[-1] < recent_lows.iloc[-3])
        failed_reclaim = bool(len(close) >= 4 and close.iloc[-1] < close.iloc[-2] and high.iloc[-1] <= high.iloc[-2])

        direction_score = (
            0.22 * np.clip((ret_n + 0.015) / 0.055, 0, 1)
            + 0.20 * np.clip((ret_recent + 0.006) / 0.030, 0, 1)
            + 0.18 * float(sma_fast >= sma_slow)
            + 0.14 * float(above_vwap)
            + 0.10 * float(above_entry)
            + 0.08 * float(not lower_lows)
            + 0.08 * float(not below_slow_sma)
            - 0.16 * float(lower_highs and lower_lows)
            - 0.10 * float(below_fast_sma and below_slow_sma)
            - 0.08 * float(failed_reclaim)
        )
        direction_score = float(np.clip(direction_score, 0, 1))

        return {
            f"{minutes}m_direction_score": direction_score,
            f"{minutes}m_ret": float(ret_n),
            f"{minutes}m_recent_ret": float(ret_recent),
            f"{minutes}m_vwap": vwap,
            f"{minutes}m_above_vwap": bool(above_vwap),
            f"{minutes}m_above_entry": bool(above_entry),
            f"{minutes}m_sma_fast": float(sma_fast),
            f"{minutes}m_sma_slow": float(sma_slow),
            f"{minutes}m_lower_highs": bool(lower_highs),
            f"{minutes}m_lower_lows": bool(lower_lows),
            f"{minutes}m_below_fast_sma": bool(below_fast_sma),
            f"{minutes}m_below_slow_sma": bool(below_slow_sma),
            f"{minutes}m_failed_reclaim": bool(failed_reclaim),
            f"{minutes}m_last_close": last_close,
        }

    def _build_next_bar_ml_dataset(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars.empty:
            return pd.DataFrame()

        rows: list[pd.DataFrame] = []
        for symbol, g in bars.groupby("symbol", sort=False):
            g = g.sort_values("timestamp").copy()
            c = g["close"].astype(float)
            h = g["high"].astype(float)
            l = g["low"].astype(float)
            v = g["volume"].astype(float)

            g["ret_1"] = c.pct_change(1, fill_method=None)
            g["ret_2"] = c.pct_change(2, fill_method=None)
            g["ret_3"] = c.pct_change(3, fill_method=None)
            g["ret_6"] = c.pct_change(6, fill_method=None)
            g["range_pct"] = (h - l) / c.replace(0, np.nan)
            g["close_location"] = (c - l) / (h - l).replace(0, np.nan)
            g["vol_ratio"] = v / v.rolling(20).mean().replace(0, np.nan)
            g["sma5_dist"] = c / c.rolling(5).mean().replace(0, np.nan) - 1.0
            g["sma15_dist"] = c / c.rolling(15).mean().replace(0, np.nan) - 1.0
            g["sma30_dist"] = c / c.rolling(30).mean().replace(0, np.nan) - 1.0
            g["rsi14"] = _rsi(c, 14)
            ema12 = c.ewm(span=12, adjust=False).mean()
            ema26 = c.ewm(span=26, adjust=False).mean()
            g["macd"] = ema12 - ema26
            g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
            g["macd_hist"] = g["macd"] - g["macd_signal"]
            g["next_bar_up"] = (c.shift(-1) > c).astype(int)
            rows.append(g)

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def predict_next_bar_direction(self, symbols: list[str]) -> pd.DataFrame:
        """Train a local intraday model and predict next-bar up probability per symbol."""
        symbols = [normalize_symbol(s) for s in symbols if normalize_symbol(s)]
        if not symbols or not self.config.ml_next_bar_enabled:
            return pd.DataFrame({"symbol": symbols, "ml_next_bar_probability": 0.50, "ml_next_bar_confirmed": True})

        old_lookback = self.config.intraday_lookback_days
        self.config.intraday_lookback_days = max(old_lookback, self.config.ml_next_bar_lookback_days)
        try:
            bars = self._download_intraday_bars(symbols, self.config.ml_next_bar_timeframe_minutes)
        finally:
            self.config.intraday_lookback_days = old_lookback

        dataset = self._build_next_bar_ml_dataset(bars)
        if dataset.empty:
            return pd.DataFrame({"symbol": symbols, "ml_next_bar_probability": 0.50, "ml_next_bar_confirmed": False, "ml_next_bar_reason": "no_intraday_training_data"})

        feature_cols = [
            "ret_1", "ret_2", "ret_3", "ret_6", "range_pct", "close_location",
            "vol_ratio", "sma5_dist", "sma15_dist", "sma30_dist", "rsi14",
            "macd", "macd_signal", "macd_hist",
        ]

        train = dataset.dropna(subset=["next_bar_up"]).copy()
        train = train.iloc[:-len(symbols)] if len(train) > len(symbols) else train
        train = train.dropna(how="all", subset=feature_cols)

        if len(train) < self.config.ml_next_bar_min_train_rows or train["next_bar_up"].nunique() < 2:
            latest = dataset.sort_values("timestamp").groupby("symbol", as_index=False).tail(1).copy()
            latest["ml_next_bar_probability"] = 0.50
            latest["ml_next_bar_confirmed"] = False
            latest["ml_next_bar_reason"] = "insufficient_training_rows"
            return latest[["symbol", "ml_next_bar_probability", "ml_next_bar_confirmed", "ml_next_bar_reason"]]

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, max_leaf_nodes=15, random_state=42)),
            ]
        )
        model.fit(train[feature_cols], train["next_bar_up"].astype(int))

        latest = dataset.sort_values("timestamp").groupby("symbol", as_index=False).tail(1).copy()
        probs = model.predict_proba(latest[feature_cols])[:, 1]
        latest["ml_next_bar_probability"] = probs
        latest["ml_next_bar_confirmed"] = latest["ml_next_bar_probability"] >= self.config.ml_next_bar_min_buy_probability
        latest["ml_next_bar_reason"] = np.where(latest["ml_next_bar_confirmed"], "next_bar_up_confirmed", "next_bar_probability_too_low")

        out = latest[["symbol", "ml_next_bar_probability", "ml_next_bar_confirmed", "ml_next_bar_reason"]].copy()
        print("[ml-next-bar] predictions:")
        print(out.to_string(index=False))
        return out

    def confirm_buy_direction(self, symbols: list[str]) -> pd.DataFrame:
        """Final directional confirmation before buy execution.

        Uses fresh 1m/5m/15m bars. A buy must show positive/constructive
        direction across timeframes, remain above VWAP when required, avoid
        lower-low deterioration, and avoid sharp immediate pullbacks.
        """
        symbols = [normalize_symbol(s) for s in symbols if normalize_symbol(s)]
        if not symbols or not self.config.require_direction_confirmation_before_buy:
            return pd.DataFrame({"symbol": symbols, "direction_confirmed": True})

        bars_1m = self._download_intraday_bars(symbols, 1)
        bars_5m = self._download_intraday_bars(symbols, 5)
        bars_15m = self._download_intraday_bars(symbols, 15)

        ml_df = self.predict_next_bar_direction(symbols)
        ml_by_symbol = {safe_symbol(r["symbol"]): r for _, r in ml_df.iterrows()} if not ml_df.empty else {}

        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            frames = {
                1: bars_1m[bars_1m["symbol"] == symbol].copy() if not bars_1m.empty else pd.DataFrame(),
                5: bars_5m[bars_5m["symbol"] == symbol].copy() if not bars_5m.empty else pd.DataFrame(),
                15: bars_15m[bars_15m["symbol"] == symbol].copy() if not bars_15m.empty else pd.DataFrame(),
            }

            metrics: dict[str, Any] = {"symbol": symbol}
            for minutes, frame in frames.items():
                metrics.update(self._intraday_direction_metrics(symbol, frame, minutes, 0.0))

            combined = float(np.clip(
                0.35 * to_float(metrics.get("1m_direction_score"), 0.5)
                + 0.40 * to_float(metrics.get("5m_direction_score"), 0.5)
                + 0.25 * to_float(metrics.get("15m_direction_score"), 0.5),
                0,
                1,
            ))

            last_1m = to_float(metrics.get("1m_last_close"), 0.0)
            ret_1m = to_float(metrics.get("1m_recent_ret"), 0.0)
            above_vwap_ok = True
            lower_low_ok = True

            if self.config.require_price_above_vwap_for_buy:
                above_vwap_ok = bool(metrics.get("1m_above_vwap", True)) and bool(metrics.get("5m_above_vwap", True))

            if self.config.require_no_lower_low_for_buy:
                lower_low_ok = not bool(metrics.get("1m_lower_lows", False)) and not bool(metrics.get("5m_lower_lows", False))

            pullback_ok = ret_1m >= self.config.max_pretrade_pullback_pct

            ml_info = ml_by_symbol.get(symbol, {})
            ml_prob = to_float(ml_info.get("ml_next_bar_probability"), 0.50)
            ml_confirmed = bool(ml_info.get("ml_next_bar_confirmed", True))

            confirmed = (
                combined >= self.config.min_pretrade_direction_score
                and to_float(metrics.get("1m_direction_score"), 0.5) >= self.config.min_pretrade_1m_score
                and to_float(metrics.get("5m_direction_score"), 0.5) >= self.config.min_pretrade_5m_score
                and to_float(metrics.get("15m_direction_score"), 0.5) >= self.config.min_pretrade_15m_score
                and above_vwap_ok
                and lower_low_ok
                and pullback_ok
                and last_1m > 0
                and ml_confirmed
            )

            reason = "confirmed"
            if not confirmed:
                failed = []
                if combined < self.config.min_pretrade_direction_score:
                    failed.append("combined_direction_weak")
                if not above_vwap_ok:
                    failed.append("below_vwap")
                if not lower_low_ok:
                    failed.append("lower_low_structure")
                if not pullback_ok:
                    failed.append("active_pullback")
                if last_1m <= 0:
                    failed.append("missing_intraday_price")
                if not ml_confirmed:
                    failed.append(f"ml_next_bar_prob_low:{ml_prob:.2f}")
                reason = ",".join(failed) or "direction_not_confirmed"

            rows.append({
                **metrics,
                "direction_score": combined,
                "ml_next_bar_probability": ml_prob,
                "ml_next_bar_confirmed": ml_confirmed,
                "direction_confirmed": bool(confirmed),
                "direction_reject_reason": reason,
            })

        out = pd.DataFrame(rows)
        if not out.empty:
            show = ["symbol", "direction_confirmed", "direction_score", "ml_next_bar_probability", "1m_direction_score", "5m_direction_score", "15m_direction_score", "direction_reject_reason"]
            print("[direction] pretrade confirmation:")
            print(out[[c for c in show if c in out.columns]].to_string(index=False))
        return out

    def monitor_active_positions(self, equity: float) -> pd.DataFrame:
        """Generate protective sell orders when active holdings lose intraday confirmation."""
        if not self.config.active_position_monitor_enabled:
            return pd.DataFrame()

        positions = self.broker.get_all_open_positions()
        if not positions:
            return pd.DataFrame()

        symbols = [normalize_symbol(p.symbol) for p in positions if p.symbol]
        print(f"[monitor] checking active positions with 1m/5m/15m bars symbols={symbols}")

        bars_1m = self._download_intraday_bars(symbols, 1)
        bars_5m = self._download_intraday_bars(symbols, 5)
        bars_15m = self._download_intraday_bars(symbols, 15)

        monitor_rows: list[dict[str, Any]] = []
        exit_rows: list[dict[str, Any]] = []

        for pos in positions:
            symbol = normalize_symbol(pos.symbol)
            avg_entry = float(pos.avg_entry_price or 0.0)
            current_price = float(pos.current_price or 0.0)
            unrealized_plpc = float(pos.unrealized_plpc or 0.0)

            frames = {
                1: bars_1m[bars_1m["symbol"] == symbol].copy() if not bars_1m.empty else pd.DataFrame(),
                5: bars_5m[bars_5m["symbol"] == symbol].copy() if not bars_5m.empty else pd.DataFrame(),
                15: bars_15m[bars_15m["symbol"] == symbol].copy() if not bars_15m.empty else pd.DataFrame(),
            }

            metrics: dict[str, Any] = {
                "timestamp_utc": now_utc().isoformat(),
                "symbol": symbol,
                "qty": pos.qty,
                "current_price": current_price,
                "avg_entry_price": avg_entry,
                "unrealized_plpc": unrealized_plpc,
                "market_value": pos.market_value,
            }

            for minutes, frame in frames.items():
                metrics.update(self._intraday_direction_metrics(symbol, frame, minutes, avg_entry))

            direction_score = float(np.clip(
                0.40 * to_float(metrics.get("1m_direction_score"), 0.5)
                + 0.35 * to_float(metrics.get("5m_direction_score"), 0.5)
                + 0.25 * to_float(metrics.get("15m_direction_score"), 0.5),
                0,
                1,
            ))
            metrics["hold_direction_score"] = direction_score

            lost_vwap = (
                self.config.monitor_exit_on_vwap_loss
                and not bool(metrics.get("1m_above_vwap", True))
                and not bool(metrics.get("5m_above_vwap", True))
            )
            structure_break = (
                self.config.monitor_exit_on_lower_high_lower_low
                and bool(metrics.get("5m_lower_highs", False))
                and bool(metrics.get("5m_lower_lows", False))
            )
            micro_structure_break = (
                bool(metrics.get("1m_lower_highs", False))
                and bool(metrics.get("1m_lower_lows", False))
                and bool(metrics.get("1m_below_slow_sma", False))
            )
            failed_reclaim = bool(metrics.get("1m_failed_reclaim", False)) and bool(metrics.get("5m_failed_reclaim", False))
            below_fast_and_slow = bool(metrics.get("1m_below_fast_sma", False)) and bool(metrics.get("5m_below_slow_sma", False))
            trailing_stop_hit = unrealized_plpc <= -abs(self.config.intraday_trailing_stop_pct)
            failed_breakout_loss = unrealized_plpc <= self.config.failed_breakout_loss_pct
            failed_breakout_price = avg_entry > 0 and current_price < avg_entry * (1.0 - self.config.failed_breakout_buffer_pct)

            exit_reason = ""
            if direction_score <= self.config.hard_exit_direction_score:
                exit_reason = "monitor_hard_direction_failure"
            elif self.config.monitor_exit_on_failed_breakout and failed_breakout_loss and failed_breakout_price:
                exit_reason = "monitor_failed_breakout_cut"
            elif trailing_stop_hit:
                exit_reason = "monitor_intraday_trailing_stop"
            elif lost_vwap and structure_break:
                exit_reason = "monitor_vwap_and_structure_break"
            elif direction_score < self.config.min_hold_direction_score and (lost_vwap or structure_break or micro_structure_break):
                exit_reason = "monitor_direction_breakdown"
            elif direction_score < self.config.min_hold_direction_score and failed_reclaim and below_fast_and_slow:
                exit_reason = "monitor_failed_reclaim"

            metrics["lost_vwap"] = bool(lost_vwap)
            metrics["structure_break"] = bool(structure_break)
            metrics["micro_structure_break"] = bool(micro_structure_break)
            metrics["failed_reclaim"] = bool(failed_reclaim)
            metrics["below_fast_and_slow"] = bool(below_fast_and_slow)
            metrics["failed_breakout_loss"] = bool(failed_breakout_loss)
            metrics["failed_breakout_price"] = bool(failed_breakout_price)
            metrics["exit_reason"] = exit_reason
            monitor_rows.append(metrics)

            if not exit_reason and self.config.scale_out_losers_enabled:
                if unrealized_plpc <= self.config.loser_full_exit_loss_pct and direction_score <= self.config.min_hold_direction_score:
                    exit_reason = "loser_full_exit_loss_and_weak_direction"
                elif unrealized_plpc <= self.config.scale_out_loss_pct and direction_score <= self.config.scale_out_direction_score:
                    exit_reason = "loser_scale_out_weak_direction"

            if exit_reason:
                if exit_reason.startswith("loser_scale_out"):
                    sell_qty = max(1, int(float(pos.qty) * self.config.scale_out_fraction))
                    target_weight = max(0.0, (float(pos.market_value) - sell_qty * current_price) / max(equity, 1e-9))
                else:
                    sell_qty = int(pos.qty)
                    target_weight = 0.0

                exit_rows.append(
                    {
                        "symbol": symbol,
                        "side": "sell",
                        "delta_shares": sell_qty,
                        "delta_dollars": float(sell_qty * current_price),
                        "price": current_price,
                        "reason": exit_reason,
                        "target_weight": target_weight,
                        "current_weight": float(pos.market_value) / max(equity, 1e-9),
                        "delta_weight": -float(sell_qty * current_price) / max(equity, 1e-9),
                        "trade_confidence": 1.0,
                        "hold_direction_score": direction_score,
                    }
                )

        monitor = pd.DataFrame(monitor_rows)
        if not monitor.empty:
            monitor.to_csv(POSITION_MONITOR_PATH, index=False)
            show = [
                "symbol", "unrealized_plpc", "hold_direction_score",
                "1m_direction_score", "5m_direction_score", "15m_direction_score",
                "lost_vwap", "structure_break", "micro_structure_break",
                "failed_reclaim", "failed_breakout_loss", "exit_reason",
            ]
            print("[monitor] active position direction check:")
            print(monitor[[c for c in show if c in monitor.columns]].to_string(index=False))

        return pd.DataFrame(exit_rows)

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
        if not scores.empty and self.llm_metadata_by_symbol:
            meta = pd.DataFrame(
                [{"symbol": sym, **data} for sym, data in self.llm_metadata_by_symbol.items()]
            )
            if not meta.empty:
                scores = scores.merge(meta, on="symbol", how="left")
                if "llm_confidence" in scores.columns:
                    scores["llm_confidence"] = pd.to_numeric(scores["llm_confidence"], errors="coerce").fillna(0.50)
                    scores["trade_confidence"] = (0.70 * scores["trade_confidence"] + 0.30 * scores["llm_confidence"]).clip(0, 1)
                    scores["confidence_score"] = (0.70 * scores["confidence_score"] + 0.30 * scores["llm_confidence"]).clip(0, 1)
                    scores["setup_score"] = scores[["setup_score", "trade_confidence", "confidence_score"]].max(axis=1)
                    scores = scores.sort_values(
                        ["trade_confidence", "setup_score", "continuation_probability", "dollar_volume20"],
                        ascending=[False, False, False, False],
                    ).reset_index(drop=True)
        if scores.empty:
            print("[scanner] no symbols passed scanner filters")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        scores.to_csv(SCANNER_SCORES_PATH, index=False)
        print("[scanner] top setups:")
        show_cols = [
            "symbol", "close", "llm_confidence", "llm_catalyst_type", "regime_label", "setup_type", "is_positive_day_opportunity",
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

        if not candidates.empty and self.config.ml_next_bar_enabled:
            ml_preds = self.predict_next_bar_direction(candidates["symbol"].tolist())
            if not ml_preds.empty:
                candidates = candidates.merge(ml_preds, on="symbol", how="left")
                candidates["ml_next_bar_probability"] = pd.to_numeric(candidates["ml_next_bar_probability"], errors="coerce").fillna(0.50)
                candidates["trade_confidence"] = (0.75 * candidates["trade_confidence"] + 0.25 * candidates["ml_next_bar_probability"]).clip(0, 1)
                candidates = candidates[candidates["ml_next_bar_probability"] >= self.config.ml_next_bar_min_buy_probability].copy()
                if candidates.empty:
                    print("[ml-next-bar] all candidates rejected by next-bar direction filter")

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
                    "ml_next_bar_probability": to_float(r.get("ml_next_bar_probability"), 0.50),
                    "ml_next_bar_confirmed": bool(r.get("ml_next_bar_confirmed", False)),
                    "llm_confidence": to_float(r.get("llm_confidence"), 0.0),
                    "llm_rank": to_int(r.get("llm_rank"), 999),
                    "llm_catalyst_type": str(r.get("llm_catalyst_type", "")),
                    "llm_reason": str(r.get("llm_reason", "")),
                    "llm_market_tone": str(r.get("llm_market_tone", "")),
                    "llm_summary": str(r.get("llm_summary", "")),
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
        rebalance = self._add_pyramid_orders(rebalance, scores, equity)
        self._trigger_options_booster_if_needed(rebalance)
        exits = self._update_high_water_and_exits(scores, equity)

        rebalance.to_csv(DEFAULT_REBALANCE_ORDERS, index=False)
        exits.to_csv(DEFAULT_EXIT_ORDERS, index=False)
        print(f"[scanner] wrote rebalance_orders={DEFAULT_REBALANCE_ORDERS} rows={len(rebalance)}")
        print(f"[scanner] wrote exit_orders={DEFAULT_EXIT_ORDERS} rows={len(exits)}")
        return rebalance, exits, scores

    def _add_pyramid_orders(self, rebalance: pd.DataFrame, scores: pd.DataFrame, equity: float) -> pd.DataFrame:
        """Add incremental buy orders for existing winners that are still confirmed."""
        if not self.config.pyramid_winners_enabled:
            return rebalance

        positions = self.broker.get_all_open_positions()
        if not positions or scores.empty:
            return rebalance

        score_by_symbol = scores.set_index("symbol").to_dict("index") if "symbol" in scores.columns else {}
        rows: list[dict[str, Any]] = []

        for pos in positions:
            symbol = normalize_symbol(pos.symbol)
            if not symbol or symbol not in score_by_symbol:
                continue
            if symbol in load_no_reentry_symbols():
                continue
            if pyramid_add_count(symbol) >= self.config.pyramid_max_adds_per_symbol_per_day:
                continue

            s = score_by_symbol[symbol]
            confidence = to_float(s.get("trade_confidence"), 0.0)
            direction_score = to_float(s.get("positive_momentum_score"), to_float(s.get("momentum_score"), 0.0))
            current_price = to_float(s.get("entry_price"), to_float(s.get("close"), pos.current_price))
            current_weight = float(pos.market_value) / max(equity, 1e-9)

            ml_prob = to_float(s.get("ml_next_bar_probability"), 0.50)

            if (
                pos.unrealized_plpc >= self.config.pyramid_min_unrealized_plpc
                and direction_score >= self.config.pyramid_min_hold_direction_score
                and ml_prob >= self.config.ml_next_bar_min_pyramid_probability
                and confidence >= self.config.min_buy_confidence
                and current_weight < min(self.config.max_target_weight, build_account_risk_profile(equity).max_position_weight)
            ):
                add_weight = min(
                    self.config.pyramid_add_weight,
                    min(self.config.max_target_weight, build_account_risk_profile(equity).max_position_weight) - current_weight,
                )
                delta_dollars = add_weight * equity
                if delta_dollars >= build_account_risk_profile(equity).min_notional_to_trade and current_price > 0:
                    rows.append({
                        "symbol": symbol,
                        "side": "buy",
                        "delta_shares": int(delta_dollars // current_price),
                        "delta_dollars": float(delta_dollars),
                        "price": current_price,
                        "market_price": to_float(s.get("close"), current_price),
                        "entry_price": current_price,
                        "reason": "pyramid_winner_confirmed_direction",
                        "target_weight": current_weight + add_weight,
                        "current_weight": current_weight,
                        "delta_weight": add_weight,
                        "trade_confidence": confidence,
                        "confidence_score": to_float(s.get("confidence_score"), confidence),
                        "continuation_probability": to_float(s.get("continuation_probability"), confidence),
                        "momentum_score": to_float(s.get("momentum_score"), direction_score),
                        "positive_momentum_score": direction_score,
                        "ml_next_bar_probability": ml_prob,
                        "unrealized_plpc": float(pos.unrealized_plpc),
                    })
                    mark_pyramid_add(symbol, reason="pyramid_winner_confirmed_direction")

        if rows:
            pyramid_df = pd.DataFrame(rows)
            print(f"[pyramid] adding {len(pyramid_df)} winner scale-in order(s)")
            show = ["symbol", "unrealized_plpc", "delta_dollars", "target_weight", "trade_confidence", "reason"]
            print(pyramid_df[[c for c in show if c in pyramid_df.columns]].to_string(index=False))
            return pd.concat([rebalance, pyramid_df], ignore_index=True)

        return rebalance

    def _trigger_options_booster_if_needed(self, candidates: pd.DataFrame) -> None:
        """Hand high-confidence candidates and profitable active holdings to options_booster.py.

        Rule:
            If an actively held stock is up at least +0.56%, request a long-call
            booster around 30 DTE using bid-price limit intent.
        """
        if not self.config.options_booster_enabled:
            return
        if not OPTIONS_BOOSTER_PATH.exists():
            print(f"[options] options_booster.py not found at {OPTIONS_BOOSTER_PATH}; skipping")
            return

        frames: list[pd.DataFrame] = []

        if candidates is not None and not candidates.empty and "trade_confidence" in candidates.columns:
            df = candidates.copy()
            df["trade_confidence"] = pd.to_numeric(df["trade_confidence"], errors="coerce").fillna(0.0)
            high = df[df["trade_confidence"] >= self.config.options_booster_confidence_threshold].copy()
            if not high.empty:
                high["booster_trigger"] = "high_stock_confidence"
                frames.append(high)

        position_rows: list[dict[str, Any]] = []
        active_positions = self.broker.get_all_open_positions()
        position_symbols_for_ta = [
            normalize_symbol(p.symbol)
            for p in active_positions
            if p.symbol and p.qty > 0 and float(p.unrealized_plpc or 0.0) >= self.config.options_booster_min_position_plpc
        ]
        ta_by_symbol: dict[str, dict[str, Any]] = {}
        if self.config.options_booster_require_technical_confirmation and position_symbols_for_ta:
            ta_df = self.confirm_buy_direction(position_symbols_for_ta)
            if not ta_df.empty:
                ta_by_symbol = {safe_symbol(r["symbol"]): r.to_dict() for _, r in ta_df.iterrows()}

        for pos in active_positions:
            parsed = parse_occ_option_symbol(pos.symbol) if "parse_occ_option_symbol" in globals() else None
            if parsed:
                continue
            symbol = normalize_symbol(pos.symbol)
            if not symbol or pos.qty <= 0:
                continue
            if float(pos.unrealized_plpc or 0.0) < self.config.options_booster_min_position_plpc:
                continue

            ta = ta_by_symbol.get(symbol, {})
            if self.config.options_booster_require_technical_confirmation:
                direction_confirmed = bool(ta.get("direction_confirmed", False))
                direction_score = to_float(ta.get("direction_score"), 0.0)
                ml_prob = to_float(ta.get("ml_next_bar_probability"), 0.0)
                if (
                    not direction_confirmed
                    or direction_score < self.config.options_booster_min_direction_score
                    or ml_prob < self.config.options_booster_min_ml_next_bar_probability
                ):
                    print(
                        f"[options] skip {symbol}: technical direction not confirmed "
                        f"direction_score={direction_score:.2f} ml_next_bar={ml_prob:.2f} "
                        f"reason={ta.get('direction_reject_reason', 'not_confirmed')}"
                    )
                    continue
            else:
                direction_score = 1.0
                ml_prob = 1.0

            technical_score = float(np.clip(
                0.55 * direction_score
                + 0.35 * ml_prob
                + 0.10 * min(max(float(pos.unrealized_plpc or 0.0) / 0.03, 0.0), 1.0),
                0,
                1,
            ))

            position_rows.append(
                {
                    "symbol": symbol,
                    "underlying_symbol": symbol,
                    "close": float(pos.current_price or 0.0),
                    "price": float(pos.current_price or 0.0),
                    "trade_confidence": technical_score,
                    "continuation_confidence": technical_score,
                    "setup_score": technical_score,
                    "momentum_score": direction_score,
                    "setup_type": "momentum",
                    "unrealized_plpc": float(pos.unrealized_plpc or 0.0),
                    "position_qty": float(pos.qty),
                    "booster_trigger": "position_up_0_56pct_with_ml_ta_confirmation",
                    "target_dte": self.config.options_booster_target_dte,
                    "min_dte": self.config.options_booster_min_dte,
                    "max_dte": self.config.options_booster_max_dte,
                    "limit_price_source": self.config.options_booster_limit_price_source,
                    "desired_order_type": "ml_ranked_limit_buy_call_at_bid",
                    "technical_direction_confirmed": True,
                    "technical_direction_score": direction_score,
                    "ml_next_bar_probability": ml_prob,
                    "technical_score": technical_score,
                    "use_ml_contract_selection": self.config.options_ml_contract_selection_enabled,
                    "target_delta": self.config.options_target_delta,
                    "min_delta": self.config.options_min_delta,
                    "max_delta": self.config.options_max_delta,
                    "max_spread_pct": self.config.options_max_spread_pct,
                    "min_open_interest": self.config.options_min_open_interest,
                    "min_volume": self.config.options_min_volume,
                    "max_theta_abs": self.config.options_max_theta_abs,
                    "min_gamma": self.config.options_min_gamma,
                    "max_iv_rank": self.config.options_max_iv_rank,
                    "greek_score_weight": self.config.options_greek_score_weight,
                    "technical_score_weight": self.config.options_technical_score_weight,
                    "liquidity_score_weight": self.config.options_liquidity_score_weight,
                    "dte_score_weight": self.config.options_dte_score_weight,
                }
            )

        if position_rows:
            frames.append(pd.DataFrame(position_rows))

        if not frames:
            print(
                f"[options] no confidence candidates and no active positions above "
                f"+{self.config.options_booster_min_position_plpc:.2%}; booster not triggered"
            )
            return

        high = pd.concat(frames, ignore_index=True)
        high["symbol"] = high["symbol"].map(normalize_symbol)
        high = high.drop_duplicates("symbol", keep="first").reset_index(drop=True)
        high["continuation_confidence"] = pd.to_numeric(
            high.get("continuation_confidence", high.get("trade_confidence", 1.0)),
            errors="coerce",
        ).fillna(high.get("trade_confidence", 1.0))
        high["underlying_symbol"] = high["symbol"].map(normalize_symbol)
        high["source"] = "run_py_position_profit_options_booster"
        high["created_at_utc"] = now_utc().isoformat()
        high["target_dte"] = self.config.options_booster_target_dte
        high["min_dte"] = self.config.options_booster_min_dte
        high["max_dte"] = self.config.options_booster_max_dte
        high["limit_price_source"] = self.config.options_booster_limit_price_source
        high["desired_order_type"] = "ml_ranked_limit_buy_call_at_bid"
        high["use_ml_contract_selection"] = self.config.options_ml_contract_selection_enabled
        high["target_delta"] = self.config.options_target_delta
        high["min_delta"] = self.config.options_min_delta
        high["max_delta"] = self.config.options_max_delta
        high["max_spread_pct"] = self.config.options_max_spread_pct
        high["min_open_interest"] = self.config.options_min_open_interest
        high["min_volume"] = self.config.options_min_volume
        high["max_theta_abs"] = self.config.options_max_theta_abs
        high["min_gamma"] = self.config.options_min_gamma
        high["max_iv_rank"] = self.config.options_max_iv_rank
        high["greek_score_weight"] = self.config.options_greek_score_weight
        high["technical_score_weight"] = self.config.options_technical_score_weight
        high["liquidity_score_weight"] = self.config.options_liquidity_score_weight
        high["dte_score_weight"] = self.config.options_dte_score_weight
        high.to_csv(OPTION_BOOSTER_CANDIDATES_PATH, index=False)

        print(
            f"[options] triggering options_booster.py for {len(high)} candidate(s); "
            f"position trigger=+{self.config.options_booster_min_position_plpc:.2%}, "
            f"target_dte={self.config.options_booster_target_dte}, "
            f"contract_selection=ML_GREEKS_TA_RANKER, limit_source=bid"
        )
        print(high[[c for c in ["symbol", "trade_confidence", "continuation_confidence", "unrealized_plpc", "technical_direction_score", "ml_next_bar_probability", "target_delta", "min_delta", "max_delta", "booster_trigger", "target_dte", "limit_price_source"] if c in high.columns]].to_string(index=False))

        args = [
            "--once",
            "--enable-options-booster",
            "--disable-scanner",
            "--option-min-confidence",
            "0.0",
            "--option-min-underlying-plpc",
            str(self.config.options_booster_min_position_plpc),
            "--option-min-dte",
            str(self.config.options_booster_min_dte),
            "--option-max-dte",
            str(self.config.options_booster_max_dte),
            "--option-target-delta",
            str(self.config.options_target_delta),
            "--option-min-delta",
            str(self.config.options_min_delta),
            "--option-max-delta",
            str(self.config.options_max_delta),
            "--option-max-spread-pct",
            str(self.config.options_max_spread_pct),
            "--option-min-open-interest",
            str(self.config.options_min_open_interest),
            "--option-min-volume",
            str(self.config.options_min_volume),
            "--option-limit-price-source",
            self.config.options_booster_limit_price_source,
        ]
        ok, output = run_python_script(
            OPTIONS_BOOSTER_PATH,
            args=args,
            timeout_seconds=self.config.options_booster_timeout_seconds,
        )
        if ok:
            print("[options] options_booster.py completed")
            if output:
                print(output[-2000:])
            write_event("options_booster_triggered", {"rows": len(high), "output_tail": output[-2000:]})
        else:
            print(f"[options] options_booster.py failed: {output}")
            write_event("options_booster_failed", {"rows": len(high), "error": output})


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
        self._last_morning_refresh_date: str | None = None

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
        # This guard is BUY-ONLY. SELL/EXIT orders must always remain allowed.
        no_reentry_symbols = load_no_reentry_symbols()
        same_cycle_exit_symbols = set(plan.loc[plan["side"] == "sell", "symbol"].map(safe_symbol))
        blocked_buy_symbols = {sym for sym in no_reentry_symbols.union(same_cycle_exit_symbols) if sym}
        if blocked_buy_symbols:
            buy_block_mask = (plan["side"] == "buy") & plan["symbol"].map(safe_symbol).isin(blocked_buy_symbols)
            blocked_count = int(buy_block_mask.sum())
            if blocked_count:
                print(f"[no-reentry] removed {blocked_count} BUY order(s); EXIT orders remain allowed: {sorted(blocked_buy_symbols)}")
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

        direction_by_symbol: dict[str, dict[str, Any]] = {}
        if self.scanner is not None and self.config.scanner.require_direction_confirmation_before_buy:
            buy_symbols = sorted(set(plan.loc[plan["side"] == "buy", "symbol"].map(safe_symbol)))
            if buy_symbols:
                direction_df = self.scanner.confirm_buy_direction(buy_symbols)
                if not direction_df.empty:
                    direction_by_symbol = {
                        safe_symbol(row["symbol"]): row.to_dict()
                        for _, row in direction_df.iterrows()
                    }

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

            if not skip_message and side == "buy" and direction_by_symbol:
                direction = direction_by_symbol.get(symbol, {})
                if not bool(direction.get("direction_confirmed", False)):
                    skip_message = f"price_direction_not_confirmed:{direction.get('direction_reject_reason', 'unknown')}"

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

        if self.config.cancel_open_orders_first:
            print("[orders] canceling open orders")
            self.broker.cancel_open_orders()

        plan = self._build_order_plan(rebalance_orders, exit_orders if exit_orders is not None else pd.DataFrame(), equity)

        # Cooldown is buy-only. Exits are always allowed so risk can be reduced immediately.
        if self._in_cooldown() and not plan.empty:
            sells = plan[plan["side"] == "sell"].copy()
            buys = plan[plan["side"] == "buy"].copy()
            if not sells.empty:
                print(f"[cooldown] active; allowing {len(sells)} exit order(s), deferring {len(buys)} buy order(s)")
                plan = sells
            else:
                print("[cooldown] active; no exits to process, buys deferred")
                results = pd.DataFrame([{"timestamp_utc": now_utc().isoformat(), "status": "skipped_cooldown", "message": "cooldown active; buys deferred"}])
                save_df(pd.DataFrame(), LOG_DIR / f"order_plan_{cycle_id}.csv")
                save_df(results, LOG_DIR / f"order_results_{cycle_id}.csv")
                self._write_account_equity_curve()
                return pd.DataFrame(), results, self.broker.get_account_snapshot(), positions_before
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

    def _regime_plan_to_order_frames(self, regime_plan: pd.DataFrame, equity: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convert regime_position_plan.csv into executor-compatible rebalance/exit frames."""
        if regime_plan is None or regime_plan.empty:
            return pd.DataFrame(), pd.DataFrame()

        positions = self._positions_by_symbol()
        buys: list[dict[str, Any]] = []
        exits: list[dict[str, Any]] = []

        for _, row in regime_plan.iterrows():
            symbol = safe_symbol(row.get("symbol"))
            if not symbol:
                continue

            action = _regime_action_value(row)
            target_weight = _regime_target_weight(row, 0.0)
            confidence = _regime_confidence(row, 0.55)
            pos = positions.get(symbol, BrokerPosition(symbol=symbol))
            current_value = max(0.0, float(pos.market_value or 0.0))
            current_weight = current_value / max(equity, 1.0)
            price = _regime_price(row, float(pos.current_price or pos.avg_entry_price or 0.0))
            if price <= 0:
                # The actual executor can still size from shares if supplied, but regime plans usually need a price.
                print(f"[regime-plan] skipping {symbol}: no usable price")
                continue

            reason = f"regime_model:{action or 'NO_ACTION'}"

            if action in {"LIQUIDATE", "EXIT", "EXIT_LONG", "SELL", "CLOSE", "CLOSE_LONG", "REDUCE_TO_ZERO"}:
                qty = int(max(0, float(pos.qty or 0)))
                if qty > 0:
                    exits.append(
                        {
                            "symbol": symbol,
                            "side": "sell",
                            "delta_shares": qty,
                            "delta_dollars": qty * price,
                            "price": price,
                            "reason": reason,
                            "target_weight": 0.0,
                            "current_weight": current_weight,
                            "delta_weight": -current_weight,
                            "trade_confidence": confidence,
                        }
                    )
                continue

            if action in {"OPEN_LONG", "BUY", "HOLD_LONG", "ADD_LONG", "CONTINUE_HOLD_WAIT_CONFIRMATION", "REBALANCE_LONG"} and target_weight > 0:
                target_value = equity * target_weight
                delta_dollars = target_value - current_value
                if delta_dollars >= self.config.min_notional_to_trade:
                    buys.append(
                        {
                            "symbol": symbol,
                            "side": "buy",
                            "delta_shares": max(0, int(delta_dollars / max(price, 0.01))),
                            "delta_dollars": delta_dollars,
                            "price": price,
                            "reason": reason,
                            "target_weight": target_weight,
                            "current_weight": current_weight,
                            "delta_weight": target_weight - current_weight,
                            "trade_confidence": confidence,
                        }
                    )
                elif delta_dollars <= -self.config.min_notional_to_trade and float(pos.qty or 0) > 0:
                    reduce_qty = int(min(float(pos.qty), abs(delta_dollars) / max(price, 0.01)))
                    if reduce_qty > 0:
                        exits.append(
                            {
                                "symbol": symbol,
                                "side": "sell",
                                "delta_shares": reduce_qty,
                                "delta_dollars": reduce_qty * price,
                                "price": price,
                                "reason": reason + ":reduce_to_target",
                                "target_weight": target_weight,
                                "current_weight": current_weight,
                                "delta_weight": target_weight - current_weight,
                                "trade_confidence": confidence,
                            }
                        )

        rebalance = pd.DataFrame(buys)
        exits_df = pd.DataFrame(exits)
        if not rebalance.empty:
            save_df(rebalance, self.config.rebalance_orders_path)
        if not exits_df.empty:
            save_df(exits_df, self.config.exit_orders_path)
        print(f"[regime-plan] converted to buys={len(rebalance)} exits={len(exits_df)}")
        return rebalance, exits_df

    def _load_rl_structure_map(self) -> dict[str, Any]:
        if not INTRADAY_STRUCTURE_MAP_PATH.exists():
            return {}
        return read_json_file(INTRADAY_STRUCTURE_MAP_PATH, {})

    def _download_latest_bars_for_rl(self, symbols: list[str]) -> pd.DataFrame:
        symbols = [safe_symbol(s) for s in symbols if safe_symbol(s)]
        if not symbols or self.scanner is None:
            if INTRADAY_BARS_1M_PATH.exists():
                try:
                    return pd.read_csv(INTRADAY_BARS_1M_PATH)
                except Exception:
                    return pd.DataFrame()
            return pd.DataFrame()

        old_lookback = self.scanner.config.intraday_lookback_days
        try:
            self.scanner.config.intraday_lookback_days = max(
                old_lookback,
                int(self.config.scanner.rl_latest_bar_lookback_days),
            )
            bars = self.scanner._download_intraday_bars(symbols, int(self.config.scanner.rl_latest_bar_timeframe_minutes))
            if not bars.empty:
                save_df(bars, INTRADAY_BARS_1M_PATH)
            return bars
        finally:
            self.scanner.config.intraday_lookback_days = old_lookback

    def _predict_rl_signal(self, symbol: str, latest_bars: pd.DataFrame, structure_map: dict[str, Any]) -> dict[str, Any]:
        symbol = safe_symbol(symbol)
        if not self.config.scanner.use_rl_timing_gate:
            return {"symbol": symbol, "action": "HOLD", "confidence": 0.0, "reason": "rl_gate_disabled"}
        if not RL_INTRADAY_STRUCTURE_TRAINER_PATH.exists():
            return {"symbol": symbol, "action": "HOLD", "confidence": 0.0, "reason": "rl_trainer_missing"}
        if not RL_MODEL_PATH.exists():
            return {"symbol": symbol, "action": "HOLD", "confidence": 0.0, "reason": "rl_model_missing"}
        if latest_bars.empty or not structure_map:
            return {"symbol": symbol, "action": "HOLD", "confidence": 0.0, "reason": "rl_inputs_missing"}

        try:
            from rl_intraday_structure_trainer import predict_live_action

            signal = predict_live_action(
                model_path=RL_MODEL_PATH,
                latest_bars=latest_bars,
                structure_map=structure_map,
                symbol=symbol,
            )
            signal["symbol"] = symbol
            return signal
        except Exception as exc:
            return {"symbol": symbol, "action": "HOLD", "confidence": 0.0, "reason": f"rl_error:{exc}"}

    def _apply_rl_timing_gate(self, rebalance: pd.DataFrame, exits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Use rl_intraday_structure_trainer.py as timing confirmation.

        BUY rows require ENTER_LONG confidence.
        Existing exits always remain allowed, and RL EXIT signals can add protective exits.
        """
        if not self.config.scanner.use_rl_timing_gate:
            return rebalance, exits

        symbols = []
        if rebalance is not None and not rebalance.empty and "symbol" in rebalance.columns:
            symbols.extend(rebalance["symbol"].map(safe_symbol).tolist())
        positions = self.broker.get_all_open_positions()
        symbols.extend([safe_symbol(p.symbol) for p in positions if p.symbol])
        symbols = sorted(set(s for s in symbols if s))

        if not symbols:
            return rebalance, exits

        latest_bars = self._download_latest_bars_for_rl(symbols)
        structure_map = self._load_rl_structure_map()
        signals: list[dict[str, Any]] = []

        # Gate buys.
        gated_rebalance = rebalance.copy() if rebalance is not None else pd.DataFrame()
        if not gated_rebalance.empty:
            keep_mask = []
            for _, row in gated_rebalance.iterrows():
                symbol = safe_symbol(row.get("symbol"))
                signal = self._predict_rl_signal(symbol, latest_bars, structure_map)
                signals.append(signal)
                action = str(signal.get("action", "HOLD")).upper()
                confidence = to_float(signal.get("confidence"), 0.0)
                allow = action == "ENTER_LONG" and confidence >= self.config.scanner.rl_min_enter_long_confidence

                if not allow and self.config.scanner.rl_allow_hold_when_model_missing:
                    reason = str(signal.get("reason", ""))
                    allow = reason in {"model_not_found", "rl_model_missing", "rl_trainer_missing", "rl_inputs_missing", "no_latest_bars"}

                keep_mask.append(bool(allow))
                if not allow:
                    print(f"[rl-gate] blocked BUY {symbol}: action={action} confidence={confidence:.2f} reason={signal.get('reason')} mask={signal.get('mask_reason')}")
                    write_event("rl_gate_blocked_buy", signal)
                else:
                    gated_rebalance.loc[gated_rebalance.index[len(keep_mask) - 1], "reason"] = (
                        str(row.get("reason", "")) + f":rl_{action}_{confidence:.2f}"
                    )
                    gated_rebalance.loc[gated_rebalance.index[len(keep_mask) - 1], "rl_action"] = action
                    gated_rebalance.loc[gated_rebalance.index[len(keep_mask) - 1], "rl_confidence"] = confidence

            gated_rebalance = gated_rebalance.loc[keep_mask].copy() if keep_mask else gated_rebalance.iloc[0:0].copy()

        # Add RL protective exits for held positions.
        rl_exit_rows: list[dict[str, Any]] = []
        existing_exit_symbols = set()
        if exits is not None and not exits.empty and "symbol" in exits.columns:
            existing_exit_symbols = set(exits["symbol"].map(safe_symbol))

        for pos in positions:
            symbol = safe_symbol(pos.symbol)
            if not symbol or symbol in existing_exit_symbols:
                continue
            signal = self._predict_rl_signal(symbol, latest_bars, structure_map)
            signals.append(signal)
            action = str(signal.get("action", "HOLD")).upper()
            confidence = to_float(signal.get("confidence"), 0.0)
            if action in {"EXIT_LONG", "EXIT_SHORT", "SELL"} and confidence >= self.config.scanner.rl_min_exit_confidence:
                price = float(pos.current_price or pos.avg_entry_price or 0.0)
                qty = int(max(0, float(pos.qty or 0)))
                if price > 0 and qty > 0:
                    rl_exit_rows.append(
                        {
                            "symbol": symbol,
                            "side": "sell",
                            "delta_shares": qty,
                            "delta_dollars": qty * price,
                            "price": price,
                            "reason": f"rl_timing_exit:{action}:{confidence:.2f}",
                            "target_weight": 0.0,
                            "current_weight": 0.0,
                            "delta_weight": 0.0,
                            "trade_confidence": confidence,
                        }
                    )
                    print(f"[rl-gate] added protective exit {symbol}: action={action} confidence={confidence:.2f}")
                    write_event("rl_gate_added_exit", signal)

        if signals:
            signal_df = pd.DataFrame(signals)
            if RL_SIGNAL_LOG_PATH.exists():
                try:
                    prior = pd.read_csv(RL_SIGNAL_LOG_PATH)
                    signal_df = pd.concat([prior, signal_df], ignore_index=True)
                except Exception:
                    pass
            save_df(signal_df.tail(5000), RL_SIGNAL_LOG_PATH)

        if rl_exit_rows:
            exits = pd.concat([exits if exits is not None else pd.DataFrame(), pd.DataFrame(rl_exit_rows)], ignore_index=True)

        return gated_rebalance, exits if exits is not None else pd.DataFrame()


    def _prepare_morning_inputs_once(self) -> None:
        today = datetime.now(ET).date().isoformat()
        if self._last_morning_refresh_date == today:
            # Avoid rebuilding expensive OpenAI/Yahoo/SerpApi inputs every executor loop.
            self.config.scanner.refresh_llm_sentiment_universe = False
            self.config.scanner.refresh_market_context = False
            self.config.scanner.refresh_premarket_trade_plan = False
            return

        print(f"[morning] preparing daily inputs for {today}")
        run_session_planning_pipeline(self.config.scanner, force=False)

        self._last_morning_refresh_date = today
        self.config.scanner.refresh_llm_sentiment_universe = False
        self.config.scanner.refresh_market_context = False
        self.config.scanner.refresh_premarket_trade_plan = False

    def run_cycle_from_files(self) -> tuple[pd.DataFrame, pd.DataFrame, AccountSnapshot, pd.DataFrame]:
        # Do not spend time refreshing/scanning before the market is executable.
        # Exits remain allowed once the market is open; buy-side gates are applied below.
        if not self._market_is_open_or_wait():
            cycle_id = run_id()
            results = pd.DataFrame([{"timestamp_utc": now_utc().isoformat(), "status": "skipped_market_closed", "message": "market is closed"}])
            save_df(pd.DataFrame(), LOG_DIR / f"order_plan_{cycle_id}.csv")
            save_df(results, LOG_DIR / f"order_results_{cycle_id}.csv")
            self._write_account_equity_curve()
            return pd.DataFrame(), results, self.broker.get_account_snapshot(), pd.DataFrame([asdict(p) for p in self.broker.get_all_open_positions()])

        if self.scanner is not None:
            try:
                self._prepare_morning_inputs_once()
                acct = self.broker.get_account_snapshot()
                equity = max(float(acct.equity), 1.0)

                market_context: dict[str, Any] = {}
                premarket_plan: dict[str, Any] = {}

                if self.config.scanner.market_context_enabled:
                    market_context = load_market_context(refresh=self.config.scanner.refresh_market_context)
                    blocked, reason = market_context_blocks_new_buys(
                        market_context,
                        allow_cautious=self.config.scanner.allow_cautious_market_buys,
                    )
                    print(f"[market-context] {reason}")
                    write_event("market_context_gate", {"blocked": blocked, "reason": reason})
                else:
                    blocked, reason = False, "market_context_disabled"

                if self.config.scanner.premarket_plan_enabled:
                    premarket_plan = load_premarket_trade_plan(refresh=self.config.scanner.refresh_premarket_trade_plan)

                # Run active-position monitor first so exits are generated before any new buys.
                monitor_exits = self.scanner.monitor_active_positions(equity)

                regime_plan = load_regime_position_plan() if self.config.scanner.use_regime_position_plan else pd.DataFrame()
                if self.config.scanner.prefer_regime_position_plan and not regime_plan.empty:
                    print(f"[regime-plan] using {REGIME_POSITION_PLAN_PATH} rows={len(regime_plan)}")
                    rebalance, scanner_exits = self._regime_plan_to_order_frames(regime_plan, equity)
                    _scores = pd.DataFrame()
                else:
                    rebalance, scanner_exits, _scores = self.scanner.build_orders()

                exits = pd.concat([monitor_exits, scanner_exits], ignore_index=True)

                # Market-context risk gate is BUY-ONLY. Never block exits.
                if self.config.scanner.halt_new_buys_on_market_context and blocked and not rebalance.empty:
                    print(f"[market-context] removing {len(rebalance)} buy candidate(s): {reason}")
                    write_event("market_context_removed_buys", {"rows": len(rebalance), "reason": reason})
                    rebalance = rebalance.iloc[0:0].copy()

                # Premarket plan gate is also BUY-ONLY. It optionally restricts buys to valid premarket setups.
                valid_pm_symbols = valid_symbols_from_premarket_plan(premarket_plan)
                if self.config.scanner.premarket_plan_enabled and self.config.scanner.prefer_premarket_valid_symbols and valid_pm_symbols and not rebalance.empty:
                    before = len(rebalance)
                    rebalance = rebalance[rebalance["symbol"].map(normalize_symbol).isin(valid_pm_symbols)].copy()
                    removed = before - len(rebalance)
                    if removed:
                        print(f"[premarket-plan] kept {len(rebalance)} valid premarket setup buy(s); removed {removed}")
                        write_event("premarket_plan_filtered_buys", {"kept": len(rebalance), "removed": removed, "valid_symbols": sorted(valid_pm_symbols)})
                elif self.config.scanner.premarket_plan_enabled and self.config.scanner.require_valid_premarket_plan_for_buys and not valid_pm_symbols and not rebalance.empty:
                    print("[premarket-plan] no valid premarket setups; removing all buy candidates")
                    write_event("premarket_plan_removed_all_buys", {"rows": len(rebalance), "reason": "no_valid_premarket_symbols"})
                    rebalance = rebalance.iloc[0:0].copy()

                # RL timing gate is applied after scanner/regime and buy-only context filters.
                # Exits remain allowed, and the RL layer may add additional protective exits.
                rebalance, exits = self._apply_rl_timing_gate(rebalance, exits)

                if not monitor_exits.empty:
                    exited_today = sorted(set(monitor_exits["symbol"].map(safe_symbol)))
                    print(f"[monitor] generated protective exits; same-day re-entry blocked for {exited_today}")

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
        if self.config.scanner:
            print(f"[executor] universe_source={'LLM_SENTIMENT' if self.config.scanner.use_llm_sentiment_universe else 'SCANNER_DEFAULT'}")
            print(f"[executor] options_booster_enabled={self.config.scanner.options_booster_enabled}")
            print(f"[executor] options_booster_threshold={self.config.scanner.options_booster_confidence_threshold:.0%}")
            print(f"[executor] market_context_enabled={self.config.scanner.market_context_enabled}")
            print(f"[executor] premarket_plan_enabled={self.config.scanner.premarket_plan_enabled}")
            print(f"[executor] session_planning_enabled={self.config.scanner.session_planning_enabled}")
            print(f"[executor] session_planning_start={self.config.scanner.session_planning_start_hour_et:02d}:{self.config.scanner.session_planning_start_minute_et:02d} ET")
            print(f"[executor] regime_plan_enabled={self.config.scanner.use_regime_position_plan}")
            print(f"[executor] rl_timing_gate_enabled={self.config.scanner.use_rl_timing_gate}")
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
        use_sp500_fallback=False,
        max_universe_symbols=500,
        top_n=30,
        min_setup_score=0.62,
        min_buy_confidence=0.58,
        min_avg_dollar_volume=5_000_000.0,
        use_llm_sentiment_universe=True,
        refresh_llm_sentiment_universe=True,
        min_llm_universe_confidence=0.0,
        options_booster_enabled=True,
        options_booster_confidence_threshold=0.75,
        opportunity_scan_enabled=True,
        min_daily_gain_pct=0.005,
        min_opportunity_score=0.60,
        min_breakout_confirmation_score=0.55,
        require_positive_momentum_or_breakout=True,
        base_target_weight=0.03,
        max_target_weight=0.10,
        broad_market_scan=False,
        stop_loss_pct=-0.025,
        profit_lock_trigger_pct=0.04,
        profit_giveback_pct=0.015,
        session_planning_enabled=True,
        session_planning_start_hour_et=4,
        session_planning_start_minute_et=0,
        run_market_context_builder=True,
        run_premarket_trade_planner=True,
        run_production_regime_sizing_model=True,
        run_rl_intraday_structure_trainer=True,
        use_regime_position_plan=True,
        prefer_regime_position_plan=True,
        use_rl_timing_gate=True,
        rl_min_enter_long_confidence=0.65,
        rl_min_exit_confidence=0.55,
        rl_min_enter_short_confidence=0.70,
    )

    return ExecutorConfig(
        paper=True,
        live=False,
        poll_once=False,
        poll_seconds=POLL_SECONDS,
        rebalance_orders_path=DEFAULT_REBALANCE_ORDERS,
        exit_orders_path=DEFAULT_EXIT_ORDERS,
        require_market_open=True,
        wait_for_market_open=True,
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
