#!/usr/bin/env python3
"""
alpaca_executor.py

Streaming-style paper executor inspired by the ProductionPaperTrader structure.

No dry-run mode.

Default behavior:
    python run_integrated.py

No arguments are required. By default this runs continuously in paper mode,
scans a built-in symbol universe directly from Alpaca daily bars, applies
market leadership/regime-aware risk controls, and enables the long-call-only
options booster.

This executor is designed to be simple to reason about:
    - scans symbols directly using Alpaca historical daily bars
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
    from llm_sentiment_universe_selector import build_sentiment_universe
except Exception:
    build_sentiment_universe = None
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
    from portfolio_sentiment_monitor import (
        SentimentMonitorConfig,
        run_portfolio_sentiment_monitor,
    )
except Exception:  # portfolio sentiment is optional and degrades safely
    SentimentMonitorConfig = None
    run_portfolio_sentiment_monitor = None


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


LLM_SENTIMENT_UNIVERSE_JSON_PATH = OUT_DIR / "latest_llm_sentiment_universe.json"

DEFAULT_REBALANCE_ORDERS = OUT_DIR / "daily_walkforward_emulated_rebalance_orders.csv"
DEFAULT_EXIT_ORDERS = OUT_DIR / "daily_walkforward_emulated_exit_orders.csv"

JOURNAL_PATH = LOG_DIR / "executor_journal.csv"
EQUITY_CURVE_PATH = LOG_DIR / "executor_equity_curve.csv"
EVENTS_JSONL = LOG_DIR / "execution_events.jsonl"
KILL_SWITCH_PATH = LOG_DIR / "kill_switch_state.json"
PDT_STATE_PATH = LOG_DIR / "pdt_state.json"

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
DEFAULT_CASH_BUFFER_PCT = 0.08

MAX_GROSS_EXPOSURE = 0.92
MAX_POSITION_WEIGHT = 0.18
MAX_SINGLE_ORDER_NOTIONAL_PCT = 0.08
MAX_ORDER_DOLLARS = 15_000.0
MAX_SINGLE_ORDER_QTY = 10_000

DAILY_DRAWDOWN_LIMIT = -0.025
KILL_SWITCH_EQUITY_FLOOR = 95_000.0

MAX_DAILY_LOSS_PCT = 0.02
MAX_DAILY_LOSS_DOLLARS = 2_000.0

HEARTBEAT_SECONDS = 60

# PDT guard rails. US pattern day trader rules generally become important below
# $25K equity. These controls are intentionally conservative and are enforced
# locally before submitting orders. They do not replace broker-side rules.
PDT_EQUITY_THRESHOLD = 25_000.0
PDT_ROLLING_WINDOW_DAYS = 5
PDT_MAX_DAY_TRADES = 3
PDT_RESERVE_DAY_TRADES = 1
PDT_BLOCK_SAME_DAY_EXITS_UNDER_25K = True
PDT_BLOCK_NEW_BUYS_WHEN_NEAR_LIMIT = True
PDT_STRICT_UNDER_25K = False
PDT_ALLOW_EMERGENCY_SELLS = True

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
SMALL_ACCOUNT_MAX_ORDER_PCT = 0.12
MID_ACCOUNT_MAX_ORDER_PCT = 0.10
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
LOCKED_UNIVERSE_PATH = PROJECT_ROOT / "locked_trading_universe.txt"
LEADERSHIP_SCORES_PATH = OUT_DIR / "latest_market_leadership_scores.csv"
LEADERSHIP_STATE_PATH = LOG_DIR / "market_leadership_state.json"
LATEST_POSITION_SENTIMENT_PATH = OUT_DIR / "latest_position_sentiment.csv"
SENTIMENT_ADJUSTED_REBALANCE_ORDERS_PATH = OUT_DIR / "sentiment_adjusted_rebalance_orders.csv"
SENTIMENT_ADJUSTED_EXIT_ORDERS_PATH = OUT_DIR / "sentiment_adjusted_exit_orders.csv"
REGIME_STATE_PATH = LOG_DIR / "market_regime_state.json"
REGIME_SCORES_PATH = OUT_DIR / "latest_market_regime_scores.csv"

# Inverse ETFs are included as normal long-only tradable symbols.
# The executor never shorts. It may buy inverse ETFs only as long positions when
# their own scanner score/volume/confidence passes the same filters as any other
# symbol. Inverse ETF activation is NOT conditioned on bearish regimes.
INVERSE_ETF_UNIVERSE = [
    "SQQQ", "SH", "PSQ", "DOG", "RWM", "SDS", "QID", "TZA", "SPXU", "SDOW",
]
INVERSE_ETF_SET = set(INVERSE_ETF_UNIVERSE)
REGIME_REQUIRED_SYMBOLS = ["SPY", "QQQ", "VIXY"]

DEFAULT_SCAN_UNIVERSE = [
    # Core indexes / ETFs / inverse ETFs
    "SPY", "QQQ", "DIA", "IWM", "XLK", "SOXX", "SMH", "ARKK", "TQQQ", "SQQQ",
    "SH", "PSQ", "DOG", "RWM", "SDS", "QID", "TZA", "SPXU", "SDOW",

    # Mega-cap and tech leadership
    "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "ARM", "AMZN", "META", "GOOGL",
    "GOOG", "TSLA", "NFLX", "ORCL", "CRM", "ADBE", "INTC", "MU", "QCOM",
    "ASML", "TSM", "SMCI", "DELL", "HPE",

    # AI / data / software / infra
    "PLTR", "SOUN", "AI", "BBAI", "PATH", "SNOW", "DDOG", "NET", "CRWD",
    "PANW", "MDB", "ESTC", "C3AI",

    # Crypto miners / crypto beta
    "COIN", "MARA", "RIOT", "CLSK", "HUT", "BTBT", "IREN", "CIFR", "WULF",
    "BITF", "HIVE",

    # Quantum / next-gen compute / high-beta innovation
    "IONQ", "RGTI", "QBTS", "QUBT", "ARQQ", "QMCO", "LAES",

    # Energy / nuclear / power infra often tied to AI compute demand
    "OKLO", "SMR", "CEG", "VST", "GEV", "ETN", "PWR", "APLD", "ALAB",
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
class LeadershipConfig:
    enabled: bool = True
    benchmark: str = "QQQ"
    lags: tuple[int, ...] = (1, 2, 3, 5)
    lookback: int = 120
    min_obs: int = 60
    min_lead_score: float = 0.05
    leadership_weight: float = 0.18
    require_positive_leadership: bool = False
    divergence_penalty: float = 0.08


@dataclass
class RegimeConfig:
    enabled: bool = True
    spy_symbol: str = "SPY"
    qqq_symbol: str = "QQQ"
    vix_proxy_symbol: str = "VIXY"
    lookback_days: int = 80

    risk_on_gross: float = 0.92
    volatile_risk_on_gross: float = 0.72
    chop_gross: float = 0.50
    defensive_gross: float = 0.25
    bear_inverse_gross: float = 0.35

    risk_on_position_weight: float = 0.18
    volatile_position_weight: float = 0.12
    chop_position_weight: float = 0.08
    defensive_position_weight: float = 0.05
    bear_inverse_position_weight: float = 0.10

    risk_on_score_bonus: float = 0.04
    volatile_score_penalty: float = 0.04
    chop_score_penalty: float = 0.08
    defensive_score_penalty: float = 0.14
    bear_inverse_score_bonus: float = 0.08

    # These remain for backward CLI compatibility, but defaults keep inverse
    # ETFs active as ordinary scanner symbols in every regime.
    allow_inverse_in_bear: bool = True
    block_inverse_when_not_bear: bool = False
    bear_market_only_inverse_routing: bool = False


@dataclass
class MarketRegimeSnapshot:
    state: str = "CHOP_NEUTRAL"
    risk_state: str = "NEUTRAL"
    max_gross_exposure: float = 0.50
    max_position_weight: float = 0.08
    setup_score_adjustment: float = -0.08
    allow_long_stocks: bool = True
    allow_inverse_etfs: bool = True
    inverse_only: bool = False
    allow_options_booster: bool = False
    spy_ret_5d: float = 0.0
    spy_ret_20d: float = 0.0
    qqq_ret_5d: float = 0.0
    qqq_ret_20d: float = 0.0
    vix_ret_5d: float = 0.0
    vix_ret_20d: float = 0.0
    realized_vol_20d: float = 0.0
    details: str = ""


class MarketRegimeEngine:
    """Classifies broad market direction using SPY/QQQ plus a VIX proxy.

    Alpaca stock data generally does not expose the cash VIX index as a normal
    tradable symbol through the stock bars endpoint, so VIXY is used by default
    as the volatility proxy. You can swap this with --regime-vix-symbol.
    """

    def __init__(self, config: RegimeConfig = RegimeConfig()):
        self.config = config

    def _latest_metrics(self, close_df: pd.DataFrame) -> dict[str, float]:
        cfg = self.config

        def ret(symbol: str, bars: int) -> float:
            if symbol not in close_df.columns or len(close_df[symbol].dropna()) <= bars:
                return 0.0
            s = close_df[symbol].dropna()
            return float(s.iloc[-1] / max(s.iloc[-bars], 1e-9) - 1.0)

        q = close_df[cfg.qqq_symbol].dropna() if cfg.qqq_symbol in close_df.columns else pd.Series(dtype=float)
        realized_vol = float(q.pct_change(fill_method=None).tail(20).std() * np.sqrt(252)) if len(q) >= 25 else 0.0

        return {
            "spy_ret_5d": ret(cfg.spy_symbol, 5),
            "spy_ret_20d": ret(cfg.spy_symbol, 20),
            "qqq_ret_5d": ret(cfg.qqq_symbol, 5),
            "qqq_ret_20d": ret(cfg.qqq_symbol, 20),
            "vix_ret_5d": ret(cfg.vix_proxy_symbol, 5),
            "vix_ret_20d": ret(cfg.vix_proxy_symbol, 20),
            "realized_vol_20d": realized_vol,
        }

    def classify(self, close_df: pd.DataFrame) -> MarketRegimeSnapshot:
        cfg = self.config
        if not cfg.enabled or close_df.empty:
            return MarketRegimeSnapshot(state="REGIME_DISABLED", risk_state="RISK_ON", max_gross_exposure=cfg.risk_on_gross, max_position_weight=cfg.risk_on_position_weight, setup_score_adjustment=0.0, allow_long_stocks=True, allow_inverse_etfs=True, inverse_only=False, allow_options_booster=True, details="regime engine disabled or no bars")

        metrics = self._latest_metrics(close_df)
        spy5 = metrics["spy_ret_5d"]
        spy20 = metrics["spy_ret_20d"]
        qqq5 = metrics["qqq_ret_5d"]
        qqq20 = metrics["qqq_ret_20d"]
        vix5 = metrics["vix_ret_5d"]
        vix20 = metrics["vix_ret_20d"]
        rv20 = metrics["realized_vol_20d"]

        trend_positive = spy20 > 0.015 and qqq20 > 0.015 and spy5 > -0.015 and qqq5 > -0.015
        trend_negative = spy20 < -0.025 and qqq20 < -0.025
        volatility_spiking = vix5 > 0.08 or vix20 > 0.15 or rv20 > 0.32
        volatility_elevated = vix5 > 0.03 or vix20 > 0.08 or rv20 > 0.24
        short_term_breakdown = spy5 < -0.025 and qqq5 < -0.025

        if trend_negative and (volatility_elevated or short_term_breakdown):
            state = "BEAR_INVERSE_ONLY"
            return MarketRegimeSnapshot(
                state=state, risk_state="RISK_OFF", max_gross_exposure=cfg.bear_inverse_gross,
                max_position_weight=cfg.bear_inverse_position_weight, setup_score_adjustment=cfg.bear_inverse_score_bonus,
                allow_long_stocks=False, allow_inverse_etfs=cfg.allow_inverse_in_bear, inverse_only=cfg.bear_market_only_inverse_routing,
                allow_options_booster=False, details="SPY/QQQ trend negative; inverse ETF routing active", **metrics
            )

        if trend_negative or (short_term_breakdown and volatility_spiking):
            state = "DEFENSIVE_RISK_OFF"
            return MarketRegimeSnapshot(
                state=state, risk_state="RISK_OFF", max_gross_exposure=cfg.defensive_gross,
                max_position_weight=cfg.defensive_position_weight, setup_score_adjustment=-abs(cfg.defensive_score_penalty),
                allow_long_stocks=True, allow_inverse_etfs=True, inverse_only=False, allow_options_booster=False,
                details="Market breakdown/volatility spike; new long risk disabled", **metrics
            )

        if trend_positive and volatility_spiking:
            state = "VOLATILE_RISK_ON"
            return MarketRegimeSnapshot(
                state=state, risk_state="RISK_ON_VOLATILE", max_gross_exposure=cfg.volatile_risk_on_gross,
                max_position_weight=cfg.volatile_position_weight, setup_score_adjustment=-abs(cfg.volatile_score_penalty),
                allow_long_stocks=True, allow_inverse_etfs=True, inverse_only=False, allow_options_booster=True,
                details="Trend positive but volatility elevated; reduced gross exposure", **metrics
            )

        if trend_positive and not volatility_elevated:
            state = "TREND_RISK_ON"
            return MarketRegimeSnapshot(
                state=state, risk_state="RISK_ON", max_gross_exposure=cfg.risk_on_gross,
                max_position_weight=cfg.risk_on_position_weight, setup_score_adjustment=cfg.risk_on_score_bonus,
                allow_long_stocks=True, allow_inverse_etfs=True, inverse_only=False, allow_options_booster=True,
                details="SPY/QQQ trend positive and volatility contained", **metrics
            )

        state = "CHOP_NEUTRAL"
        return MarketRegimeSnapshot(
            state=state, risk_state="NEUTRAL", max_gross_exposure=cfg.chop_gross,
            max_position_weight=cfg.chop_position_weight, setup_score_adjustment=-abs(cfg.chop_score_penalty),
            allow_long_stocks=True, allow_inverse_etfs=True, inverse_only=False, allow_options_booster=False,
            details="Mixed SPY/QQQ direction or elevated volatility; reduced participation", **metrics
        )


class MarketLeadershipEngine:
    """Scores symbols by whether their returns lead the benchmark's future returns.

    close_df columns must be symbols and should include the configured benchmark.
    The score is intentionally conservative and can be used as a ranking boost,
    confidence confirmation, or divergence warning for the scanner.
    """

    def __init__(self, config: LeadershipConfig = LeadershipConfig()):
        self.config = config

    def compute_returns(self, close_df: pd.DataFrame) -> pd.DataFrame:
        return close_df.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    def score_leaders(self, close_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        if close_df.empty or cfg.benchmark not in close_df.columns:
            return pd.DataFrame()

        returns = self.compute_returns(close_df).tail(cfg.lookback)
        benchmark_ret = returns[cfg.benchmark]
        rows: list[dict[str, Any]] = []

        for symbol in returns.columns:
            if symbol == cfg.benchmark:
                continue

            symbol_ret = returns[symbol]
            lag_scores: list[float] = []
            for lag in cfg.lags:
                future_benchmark = benchmark_ret.shift(-lag)
                aligned = pd.concat([symbol_ret, future_benchmark], axis=1).dropna()
                if len(aligned) < cfg.min_obs:
                    continue
                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                if pd.notna(corr):
                    lag_scores.append(float(corr))

            if not lag_scores:
                continue

            recent_rs = close_df[symbol] / close_df[cfg.benchmark].replace(0, np.nan)
            rs_slope = float(np.clip(np.nan_to_num(recent_rs.tail(20).pct_change(fill_method=None).mean()), -0.05, 0.05))
            volatility_impulse = float(np.clip(np.nan_to_num(abs(symbol_ret.tail(10)).mean()), 0.0, 0.05))
            best_lag_corr = float(np.nanmax(lag_scores))
            avg_lag_corr = float(np.nanmean(lag_scores))

            lead_score = (
                0.70 * best_lag_corr
                + 0.20 * rs_slope
                + 0.10 * volatility_impulse
            )

            rows.append({
                "symbol": normalize_symbol(symbol),
                "lead_score": float(lead_score),
                "best_lag_corr": best_lag_corr,
                "avg_lag_corr": avg_lag_corr,
                "rs_slope": rs_slope,
                "volatility_impulse": volatility_impulse,
            })

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("lead_score", ascending=False).reset_index(drop=True)

    def market_alignment(self, close_df: pd.DataFrame, leader_scores: pd.DataFrame, top_n: int = 8) -> dict[str, Any]:
        cfg = self.config
        if close_df.empty or leader_scores.empty or cfg.benchmark not in close_df.columns:
            return {"leader_alignment": 0.0, "benchmark_momentum": 0.0, "divergence_warning": False}

        top = leader_scores.head(max(1, top_n))["symbol"].tolist()
        ret = self.compute_returns(close_df)
        available = [s for s in top if s in ret.columns]
        if not available:
            return {"leader_alignment": 0.0, "benchmark_momentum": 0.0, "divergence_warning": False}

        leader_alignment = float(np.nan_to_num(ret[available].tail(5).mean(axis=1).mean()))
        benchmark_momentum = float(np.nan_to_num(ret[cfg.benchmark].tail(5).mean()))
        divergence_warning = bool(benchmark_momentum > 0 and leader_alignment < 0)
        return {
            "leader_alignment": leader_alignment,
            "benchmark_momentum": benchmark_momentum,
            "divergence_warning": divergence_warning,
        }


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

    leadership: LeadershipConfig = None
    regime: RegimeConfig = None


@dataclass
class OptionsBoosterConfig:
    # Long-call-only booster. This never shorts options. It only buys calls after
    # the underlying stock position is already open and profitable.
    enabled: bool = True
    feed: str = "indicative"

    min_continuation_confidence: float = 0.68
    min_underlying_unrealized_plpc: float = 0.003
    require_momentum_setup: bool = True

    min_dte: int = 14
    max_dte: int = 45
    target_delta_min: float = 0.40
    target_delta_max: float = 0.65
    ideal_delta: float = 0.52

    max_spread_pct: float = 0.15
    min_open_interest: int = 50
    min_volume: int = 5
    max_contract_mid_price: float = 5.00

    max_risk_per_trade_pct: float = 0.010
    max_risk_per_trade_dollars: float = 250.0
    max_contracts_per_underlying: int = 2
    max_open_option_positions: int = 4

    take_profit_pct: float = 0.60
    hard_stop_pct: float = -0.35
    profit_lock_trigger_pct: float = 0.40
    profit_giveback_pct: float = 0.25

    # Expiration protection. Long calls are closed before expiration so the
    # account does not carry exercise/assignment-style settlement risk into
    # expiration. New entries also require enough remaining DTE after this
    # buffer.
    close_before_expiration_days: int = 3
    block_new_entries_within_expiration_days: int = 5


@dataclass
class PortfolioSentimentOverlayConfig:
    # OpenAI-backed monitor for currently held portfolio constituents only.
    # This never expands the universe and never creates new entry symbols.
    enabled: bool = True
    use_openai: bool = True
    model: str = os.getenv("OPENAI_SENTIMENT_MODEL", "gpt-5.4-mini")
    max_headlines_per_symbol: int = 8

    # Conservative risk overlay thresholds. These are intentionally strict so
    # news/sentiment can reduce risk but cannot bypass normal scanner/executor rules.
    reduce_score_threshold: float = -0.55
    exit_score_threshold: float = -0.75
    min_confidence_for_action: float = 0.70
    bullish_hold_threshold: float = 0.50

    watch_multiplier: float = 0.85
    reduce_multiplier: float = 0.50
    exit_multiplier: float = 0.00
    bullish_hold_multiplier: float = 1.05
    max_bullish_multiplier: float = 1.10

    # Reduce means sell this fraction of the current stock position. Exit means
    # sell the full current stock/option position. Watch only blocks/scales adds.
    reduce_sell_fraction: float = 0.50

    # If True, a bearish stock sentiment reading also blocks new long-call
    # option booster entries for that underlying.
    gate_options_booster: bool = True


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

    # These are recalculated by account scaling every cycle, then clamped by
    # the current market regime. They let risk-on exposure recover after a
    # previous risk-off cycle without restarting the process.
    account_scaled_max_gross_exposure: float = MAX_GROSS_EXPOSURE
    account_scaled_max_position_weight: float = MAX_POSITION_WEIGHT
    account_scaled_max_single_order_notional_pct: float = MAX_SINGLE_ORDER_NOTIONAL_PCT
    account_scaled_max_order_dollars: float = MAX_ORDER_DOLLARS

    cooldown_seconds: int = COOLDOWN_SECONDS

    enable_kill_switch: bool = True
    daily_drawdown_limit: float = DAILY_DRAWDOWN_LIMIT
    max_daily_loss_pct: float = MAX_DAILY_LOSS_PCT
    max_daily_loss_dollars: float = MAX_DAILY_LOSS_DOLLARS
    kill_switch_equity_floor: float = KILL_SWITCH_EQUITY_FLOOR

    enable_pdt_guard: bool = True
    pdt_equity_threshold: float = PDT_EQUITY_THRESHOLD
    pdt_rolling_window_days: int = PDT_ROLLING_WINDOW_DAYS
    pdt_max_day_trades: int = PDT_MAX_DAY_TRADES
    pdt_reserve_day_trades: int = PDT_RESERVE_DAY_TRADES
    pdt_block_same_day_exits_under_25k: bool = PDT_BLOCK_SAME_DAY_EXITS_UNDER_25K
    pdt_block_new_buys_when_near_limit: bool = PDT_BLOCK_NEW_BUYS_WHEN_NEAR_LIMIT
    pdt_strict_under_25k: bool = PDT_STRICT_UNDER_25K
    pdt_allow_emergency_sells: bool = PDT_ALLOW_EMERGENCY_SELLS

    order_wait_seconds: int = ORDER_WAIT_SECONDS
    time_in_force: TimeInForce = TimeInForce.DAY
    order_tag_prefix: str = "explosive_alpha"

    scanner: ScannerConfig = None
    options_booster: OptionsBoosterConfig = None
    portfolio_sentiment: PortfolioSentimentOverlayConfig = None


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

    def get_account_flags(self) -> dict[str, Any]:
        try:
            acct = self.trading.get_account()
        except Exception as exc:
            return {"lookup_error": str(exc)}

        return {
            "pattern_day_trader": bool(getattr(acct, "pattern_day_trader", False)),
            "trading_blocked": bool(getattr(acct, "trading_blocked", False)),
            "account_blocked": bool(getattr(acct, "account_blocked", False)),
            "daytrade_count": to_int(getattr(acct, "daytrade_count", 0), 0),
            "multiplier": to_float(getattr(acct, "multiplier", 0.0), 0.0),
            "status": str(getattr(acct, "status", "")),
        }

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


def load_llm_sentiment_universe(
    json_path: Path = LLM_SENTIMENT_UNIVERSE_JSON_PATH,
    locked_path: Path = LOCKED_UNIVERSE_PATH,
    max_symbols: int = 30,
    max_age_hours: float = 16.0,
) -> list[str]:
    """
    Preferred universe loader.

    Priority:
        1. daily_system_outputs/latest_llm_sentiment_universe.json
        2. locked_trading_universe.txt
        3. DEFAULT_SCAN_UNIVERSE

    Expected JSON shape:
        {
          "symbols": ["NVDA", "PLTR", ...],
          "selections": [
            {"symbol": "NVDA", "reason": "..."}
          ]
        }
    """
    def valid(sym: Any) -> str:
        s = normalize_symbol(sym)
        if not s:
            return ""
        if not re.fullmatch(r"[A-Z][A-Z0-9-]{0,9}", s):
            return ""
        return s

    def dedupe(symbols: list[Any]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()

        for raw in symbols:
            sym = valid(raw)
            if not sym or sym in seen:
                continue
            seen.add(sym)
            out.append(sym)

            if len(out) >= max_symbols:
                break

        return out

    if json_path.exists():
        try:
            age_hours = (time.time() - json_path.stat().st_mtime) / 3600.0

            if age_hours <= max_age_hours:
                data = json.loads(json_path.read_text(encoding="utf-8"))

                raw_symbols: list[Any] = []

                if isinstance(data.get("symbols"), list):
                    raw_symbols.extend(data["symbols"])

                if isinstance(data.get("selections"), list):
                    for row in data["selections"]:
                        if isinstance(row, dict):
                            raw_symbols.append(row.get("symbol") or row.get("ticker"))

                symbols = dedupe(raw_symbols)

                if symbols:
                    print(
                        "[universe] loaded LLM sentiment universe "
                        f"symbols={len(symbols)} path={json_path} age_hours={age_hours:.2f}"
                    )

                    write_event(
                        "llm_sentiment_universe_loaded",
                        {
                            "path": str(json_path),
                            "age_hours": age_hours,
                            "symbols": symbols,
                            "market_tone": data.get("market_tone", ""),
                            "summary": data.get("summary", ""),
                        },
                    )

                    return symbols

                print(f"[universe] LLM JSON existed but had no valid symbols: {json_path}")

            else:
                print(
                    "[universe] LLM sentiment universe is stale; "
                    f"path={json_path} age_hours={age_hours:.2f} max_age_hours={max_age_hours:.2f}"
                )

        except Exception as exc:
            print(f"[universe] failed to load LLM sentiment universe: {exc}")

    # Backward-compatible fallback.
    if locked_path.exists():
        raw = locked_path.read_text(encoding="utf-8", errors="ignore")
        tokens = re.split(r"[\s,]+", raw)
        symbols = dedupe(tokens)

        if symbols:
            print(f"[universe] loaded locked universe symbols={len(symbols)} path={locked_path}")
            return symbols

    print("[universe] using DEFAULT_SCAN_UNIVERSE fallback")
    return list(DEFAULT_SCAN_UNIVERSE)[:max_symbols]


def load_locked_universe(path: Path = LOCKED_UNIVERSE_PATH, max_symbols: int = 30) -> list[str]:
    """
    Backward-compatible wrapper.
    Existing run.py code can keep calling load_locked_universe().
    """
    return load_llm_sentiment_universe(
        json_path=LLM_SENTIMENT_UNIVERSE_JSON_PATH,
        locked_path=path,
        max_symbols=max_symbols,
    )


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
        self.options_config = executor_config.options_booster or OptionsBoosterConfig(enabled=False)
        self.leadership_config = config.leadership or LeadershipConfig(enabled=False)
        self.leadership_engine = MarketLeadershipEngine(self.leadership_config) if self.leadership_config.enabled else None
        self.regime_config = config.regime or RegimeConfig(enabled=True)
        self.regime_engine = MarketRegimeEngine(self.regime_config) if self.regime_config.enabled else None
        self.latest_regime = MarketRegimeSnapshot()
        self.option_data = (
            OptionHistoricalDataClient(broker.api_key, broker.secret_key)
            if OptionHistoricalDataClient is not None
            else None
        )

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
            symbols = load_llm_sentiment_universe(
                json_path=LLM_SENTIMENT_UNIVERSE_JSON_PATH,
                locked_path=LOCKED_UNIVERSE_PATH,
                max_symbols=self.config.max_universe_symbols,
            )

        out: list[str] = []

        for sym in symbols:
            sym = normalize_symbol(sym)
            if not sym:
                continue
            if not re.fullmatch(r"[A-Z][A-Z0-9-]{0,9}", sym):
                continue
            if sym not in out:
                out.append(sym)

        if not out:
            out = list(DEFAULT_SCAN_UNIVERSE)[: self.config.max_universe_symbols]

        print(f"[universe] active monitor universe symbols={len(out)} source=llm_sentiment_json")
        return out[: self.config.max_universe_symbols]

    def _regime_required_symbols(self) -> list[str]:
        cfg = self.regime_config
        if not cfg.enabled:
            return []
        return [normalize_symbol(cfg.spy_symbol), normalize_symbol(cfg.qqq_symbol), normalize_symbol(cfg.vix_proxy_symbol)]

    def _route_universe_for_regime(self, base_symbols: list[str], regime: MarketRegimeSnapshot | None) -> list[str]:
        """Return the direct scan universe.

        Regime is used for exposure/position sizing and score adjustment, but it
        does not activate, deactivate, force, or block inverse ETFs. Inverse ETFs
        are treated like every other symbol and must earn their place via the
        scanner's own momentum/reversal/confidence filters.
        """
        base = [normalize_symbol(s) for s in base_symbols if normalize_symbol(s)]
        routed = list(dict.fromkeys(base + self._regime_required_symbols()))
        print(
            "[scanner-routing] direct_scan=True "
            f"state={getattr(regime, 'state', 'UNKNOWN')} "
            f"inverse_etfs_active_by_score=True symbols={len(routed)}"
        )
        return routed[: max(self.config.max_universe_symbols, len(self._regime_required_symbols()))]

    def _classify_market_regime(self, bars: pd.DataFrame) -> MarketRegimeSnapshot:
        if self.regime_engine is None:
            return MarketRegimeSnapshot(state="REGIME_DISABLED", risk_state="RISK_ON", max_gross_exposure=self.executor_config.max_gross_exposure, max_position_weight=self.executor_config.max_position_weight, setup_score_adjustment=0.0, allow_long_stocks=True, allow_inverse_etfs=True, inverse_only=False, allow_options_booster=True)

        close_df = self._close_matrix(bars)
        regime = self.regime_engine.classify(close_df)
        self.latest_regime = regime
        save_json(asdict(regime), REGIME_STATE_PATH)
        pd.DataFrame([asdict(regime)]).to_csv(REGIME_SCORES_PATH, index=False)

        # Dynamic gross/position exposure switching. Account scaling is applied
        # each cycle before the scanner runs; these effective limits clamp the
        # account-scaled limits by the active regime.
        self.executor_config.max_gross_exposure = min(
            self.executor_config.account_scaled_max_gross_exposure,
            regime.max_gross_exposure,
        )
        self.executor_config.max_position_weight = min(
            self.executor_config.account_scaled_max_position_weight,
            regime.max_position_weight,
        )
        self.executor_config.max_single_order_notional_pct = min(
            self.executor_config.account_scaled_max_single_order_notional_pct,
            max(regime.max_position_weight, 0.01),
        )
        self.executor_config.max_order_dollars = min(
            self.executor_config.account_scaled_max_order_dollars,
            max(100.0, regime.max_position_weight * max(self.broker.get_account_snapshot().equity, 1.0)),
        )

        print(
            "[regime] "
            f"state={regime.state} risk_state={regime.risk_state} "
            f"SPY5={regime.spy_ret_5d:.2%} SPY20={regime.spy_ret_20d:.2%} "
            f"QQQ5={regime.qqq_ret_5d:.2%} QQQ20={regime.qqq_ret_20d:.2%} "
            f"VIX5={regime.vix_ret_5d:.2%} RV20={regime.realized_vol_20d:.2%} "
            f"max_gross={self.executor_config.max_gross_exposure:.2%} "
            f"max_position={self.executor_config.max_position_weight:.2%} "
            f"inverse_only={regime.inverse_only}"
        )
        return regime

    def _apply_regime_to_scores(self, scores: pd.DataFrame, regime: MarketRegimeSnapshot) -> pd.DataFrame:
        """Annotate scores and apply regime score pressure without symbol gating.

        Regime controls risk budget and confidence pressure. It does not remove
        inverse ETFs, does not force inverse-only routing, and does not block
        regular equities from being considered.
        """
        if scores.empty or not self.regime_config.enabled:
            return scores

        out = scores.copy()
        out["symbol"] = out["symbol"].map(normalize_symbol)
        out["market_regime"] = regime.state
        out["risk_state"] = regime.risk_state
        out["regime_setup_adjustment"] = regime.setup_score_adjustment
        out["regime_max_gross_exposure"] = regime.max_gross_exposure
        out["regime_max_position_weight"] = regime.max_position_weight
        out["regime_inverse_only"] = False
        out["is_inverse_etf"] = out["symbol"].isin(INVERSE_ETF_SET)

        adj = float(regime.setup_score_adjustment)
        out["setup_score"] = (out["setup_score"] + adj).clip(0.0, 1.0)
        out["trade_confidence"] = (out["trade_confidence"] + adj * 0.75).clip(0.0, 1.0)

        return out.sort_values(["setup_score", "dollar_volume20"], ascending=[False, False]).reset_index(drop=True)

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
        for symbol, g in bars.groupby("symbol", sort=False):
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

    def _close_matrix(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars.empty:
            return pd.DataFrame()
        close_df = bars.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
        close_df = close_df.sort_index().ffill().dropna(axis=1, how="all")
        close_df.columns = [normalize_symbol(c) for c in close_df.columns]
        return close_df

    def _apply_market_leadership(self, scores: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
        cfg = self.leadership_config
        if not cfg.enabled or self.leadership_engine is None or scores.empty:
            return scores

        close_df = self._close_matrix(bars)
        if cfg.benchmark not in close_df.columns:
            print(f"[leadership] benchmark {cfg.benchmark} missing from bars; skipping leadership scoring")
            return scores

        leader_scores = self.leadership_engine.score_leaders(close_df)
        if leader_scores.empty:
            print("[leadership] no valid leader scores; skipping leadership adjustment")
            return scores

        LEADERSHIP_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
        leader_scores.to_csv(LEADERSHIP_SCORES_PATH, index=False)

        alignment = self.leadership_engine.market_alignment(close_df, leader_scores, top_n=self.config.top_n)
        save_json(alignment, LEADERSHIP_STATE_PATH)

        out = scores.merge(leader_scores, on="symbol", how="left")
        out["lead_score"] = out["lead_score"].fillna(0.0)
        out["best_lag_corr"] = out["best_lag_corr"].fillna(0.0)
        out["avg_lag_corr"] = out["avg_lag_corr"].fillna(0.0)
        out["rs_slope"] = out["rs_slope"].fillna(0.0)
        out["volatility_impulse"] = out["volatility_impulse"].fillna(0.0)

        # Convert raw lead score into a bounded confirmation term. Positive
        # leaders boost setup confidence; weak/negative leaders are penalized.
        leadership_confirm = ((out["lead_score"] - cfg.min_lead_score) / 0.35).clip(-1.0, 1.0)
        out["leadership_confirm"] = leadership_confirm
        out["raw_setup_score"] = out["setup_score"]
        out["raw_trade_confidence"] = out["trade_confidence"]
        out["leadership_divergence_warning"] = bool(alignment.get("divergence_warning", False))

        out["setup_score"] = (out["setup_score"] + cfg.leadership_weight * leadership_confirm).clip(0.0, 1.0)
        out["trade_confidence"] = (out["trade_confidence"] + (cfg.leadership_weight * 0.75) * leadership_confirm).clip(0.0, 1.0)

        if alignment.get("divergence_warning", False):
            out["setup_score"] = (out["setup_score"] - cfg.divergence_penalty).clip(0.0, 1.0)
            out["trade_confidence"] = (out["trade_confidence"] - cfg.divergence_penalty).clip(0.0, 1.0)

        if cfg.require_positive_leadership:
            out = out[out["lead_score"] >= cfg.min_lead_score].copy()

        out = out.sort_values(["setup_score", "lead_score", "dollar_volume20"], ascending=[False, False, False])
        print(
            "[leadership] "
            f"benchmark={cfg.benchmark} leaders={len(leader_scores)} "
            f"alignment={alignment.get('leader_alignment', 0.0):.4f} "
            f"benchmark_momentum={alignment.get('benchmark_momentum', 0.0):.4f} "
            f"divergence={alignment.get('divergence_warning', False)}"
        )
        return out.reset_index(drop=True)

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
            if parsed and parsed.get("type") == "call" and pos.qty > 0:
                underlyings.add(str(parsed["underlying"]))
        return underlyings

    def _open_option_position_count(self) -> int:
        count = 0
        for pos in self.broker.get_all_open_positions():
            parsed = parse_occ_option_symbol(pos.symbol)
            if parsed and pos.qty > 0:
                count += 1
        return count

    def _get_call_chain_snapshots(self, underlying: str) -> dict[str, Any]:
        if self.option_data is None or OptionChainRequest is None:
            return {}

        try:
            req = OptionChainRequest(underlying_symbol=underlying)
            chain = self.option_data.get_option_chain(req)
            if isinstance(chain, dict):
                return chain
            if hasattr(chain, "data") and isinstance(chain.data, dict):
                return chain.data
            return dict(chain) if chain is not None else {}
        except Exception as exc:
            print(f"[options] failed to fetch option chain for {underlying}: {exc}")
            write_event("option_chain_fetch_error", {"underlying": underlying, "error": str(exc)})
            return {}

    def _select_best_call_contract(self, underlying: str, underlying_price: float, confidence: float) -> dict[str, Any] | None:
        cfg = self.options_config
        chain = self._get_call_chain_snapshots(underlying)
        if not chain:
            return None

        today = now_utc().date()
        rows: list[dict[str, Any]] = []
        for contract_symbol, snapshot in chain.items():
            parsed = parse_occ_option_symbol(contract_symbol)
            if not parsed or parsed.get("type") != "call":
                continue

            dte = (parsed["expiration"] - today).days
            min_entry_dte = max(cfg.min_dte, cfg.block_new_entries_within_expiration_days + 1)
            if dte < min_entry_dte or dte > cfg.max_dte:
                continue

            metrics = _option_quote_metrics(snapshot)
            mid = metrics["mid"]
            if mid <= 0 or mid > cfg.max_contract_mid_price:
                continue
            if metrics["spread_pct"] > cfg.max_spread_pct:
                continue
            if metrics["volume"] < cfg.min_volume:
                continue
            if metrics["open_interest"] < cfg.min_open_interest:
                continue

            delta = metrics["delta"]
            if np.isnan(delta):
                # Fall back to moneyness if greeks are unavailable.
                moneyness = parsed["strike"] / max(underlying_price, 1e-9)
                if not (0.95 <= moneyness <= 1.08):
                    continue
                delta_fit = 0.55
            else:
                if delta < cfg.target_delta_min or delta > cfg.target_delta_max:
                    continue
                delta_fit = 1.0 - min(abs(delta - cfg.ideal_delta) / max(cfg.target_delta_max - cfg.target_delta_min, 1e-9), 1.0)

            dte_mid = (cfg.min_dte + cfg.max_dte) / 2.0
            dte_fit = 1.0 - min(abs(dte - dte_mid) / max((cfg.max_dte - cfg.min_dte) / 2.0, 1e-9), 1.0)
            spread_score = 1.0 - min(metrics["spread_pct"] / max(cfg.max_spread_pct, 1e-9), 1.0)
            liquidity_score = min(metrics["volume"] / max(cfg.min_volume * 10, 1), 1.0) * 0.45 + min(metrics["open_interest"] / max(cfg.min_open_interest * 10, 1), 1.0) * 0.55
            moneyness_score = 1.0 - min(abs(parsed["strike"] / max(underlying_price, 1e-9) - 1.0) / 0.12, 1.0)

            option_score = (
                0.30 * float(confidence)
                + 0.20 * liquidity_score
                + 0.18 * delta_fit
                + 0.14 * spread_score
                + 0.10 * dte_fit
                + 0.08 * moneyness_score
            )

            rows.append({
                "option_symbol": safe_symbol(contract_symbol),
                "underlying": underlying,
                "underlying_price": underlying_price,
                "expiration": parsed["expiration"].isoformat(),
                "dte": dte,
                "strike": parsed["strike"],
                "bid": metrics["bid"],
                "ask": metrics["ask"],
                "mid": mid,
                "spread_pct": metrics["spread_pct"],
                "delta": delta,
                "gamma": metrics["gamma"],
                "iv": metrics["iv"],
                "volume": metrics["volume"],
                "open_interest": metrics["open_interest"],
                "option_score": option_score,
            })

        if not rows:
            return None

        ranked = pd.DataFrame(rows).sort_values("option_score", ascending=False).reset_index(drop=True)
        ranked.to_csv(OPTION_BOOSTER_CANDIDATES_PATH, index=False)
        return ranked.iloc[0].to_dict()

    def _build_option_booster_orders(self, scores: pd.DataFrame, equity: float) -> pd.DataFrame:
        cfg = self.options_config
        if not cfg.enabled:
            return pd.DataFrame()
        if self.option_data is None:
            print("[options] booster enabled but OptionHistoricalDataClient is unavailable; skipping")
            return pd.DataFrame()
        if scores.empty:
            return pd.DataFrame()

        positions = {normalize_symbol(p.symbol): p for p in self.broker.get_all_open_positions()}
        existing_option_underlyings = self._existing_option_underlyings()
        if self._open_option_position_count() >= cfg.max_open_option_positions:
            print("[options] max open option positions reached; skipping call booster")
            return pd.DataFrame()

        latest = scores.copy()
        latest["symbol"] = latest["symbol"].map(normalize_symbol)
        latest["continuation_confidence"] = latest[[c for c in ["trade_confidence", "setup_score", "momentum_score"] if c in latest.columns]].max(axis=1)

        candidates = latest[latest["continuation_confidence"] >= cfg.min_continuation_confidence].copy()
        if cfg.require_momentum_setup and "setup_type" in candidates.columns:
            candidates = candidates[candidates["setup_type"].astype(str).str.lower() == "momentum"].copy()

        rows: list[dict[str, Any]] = []
        for _, r in candidates.sort_values("continuation_confidence", ascending=False).iterrows():
            underlying = normalize_symbol(r["symbol"])
            if underlying in existing_option_underlyings:
                continue
            pos = positions.get(underlying)
            if pos is None or pos.qty <= 0:
                continue
            if pos.unrealized_plpc < cfg.min_underlying_unrealized_plpc:
                continue

            underlying_price = to_float(r.get("close"), pos.current_price)
            if underlying_price <= 0:
                underlying_price = pos.current_price
            if underlying_price <= 0:
                continue

            best = self._select_best_call_contract(
                underlying=underlying,
                underlying_price=underlying_price,
                confidence=float(r["continuation_confidence"]),
            )
            if not best:
                continue

            risk_budget = min(equity * cfg.max_risk_per_trade_pct, cfg.max_risk_per_trade_dollars)
            contract_notional = float(best["mid"]) * 100.0
            contracts = int(risk_budget // max(contract_notional, 1e-9))
            contracts = min(contracts, cfg.max_contracts_per_underlying)
            if contracts <= 0:
                print(
                    f"[options] skip {underlying}: selected call {best['option_symbol']} "
                    f"cost=${contract_notional:.2f} exceeds risk_budget=${risk_budget:.2f}"
                )
                continue

            rows.append({
                "symbol": best["option_symbol"],
                "side": "buy",
                "delta_shares": contracts,
                "delta_dollars": contract_notional * contracts,
                "price": float(best["mid"]),
                "reason": "option_booster_long_call_after_positive_underlying_validation",
                "target_weight": (contract_notional * contracts) / max(equity, 1e-9),
                "current_weight": 0.0,
                "delta_weight": (contract_notional * contracts) / max(equity, 1e-9),
                "trade_confidence": float(r["continuation_confidence"]),
                "confidence_score": float(r["continuation_confidence"]),
                "underlying_symbol": underlying,
                "underlying_unrealized_plpc": float(pos.unrealized_plpc),
                "option_score": float(best["option_score"]),
                "option_delta": best.get("delta"),
                "option_dte": best.get("dte"),
                "option_strike": best.get("strike"),
                "option_spread_pct": best.get("spread_pct"),
            })

        if rows:
            print(f"[options] generated long-call booster orders={len(rows)}")
        return pd.DataFrame(rows)

    def _build_option_exit_orders(self, equity: float) -> pd.DataFrame:
        cfg = self.options_config
        if not cfg.enabled:
            return pd.DataFrame()

        state = self._load_state()
        high_plpc: dict[str, float] = {k: to_float(v) for k, v in state.get("option_high_plpc", {}).items()}
        rows: list[dict[str, Any]] = []

        for pos in self.broker.get_all_open_positions():
            parsed = parse_occ_option_symbol(pos.symbol)
            if not parsed or parsed.get("type") != "call" or pos.qty <= 0:
                continue

            symbol = safe_symbol(pos.symbol)
            old_high = high_plpc.get(symbol, pos.unrealized_plpc)
            new_high = max(old_high, pos.unrealized_plpc)
            high_plpc[symbol] = new_high

            today = now_utc().date()
            dte = (parsed["expiration"] - today).days

            reason = ""
            if dte <= cfg.close_before_expiration_days:
                reason = "option_booster_expiration_protection_exit"
            elif pos.unrealized_plpc <= cfg.hard_stop_pct:
                reason = "option_booster_hard_stop"
            elif pos.unrealized_plpc >= cfg.take_profit_pct:
                reason = "option_booster_take_profit"
            elif new_high >= cfg.profit_lock_trigger_pct and pos.unrealized_plpc <= new_high - cfg.profit_giveback_pct:
                reason = "option_booster_profit_giveback"

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
                    "option_plpc": float(pos.unrealized_plpc),
                    "option_high_plpc": float(new_high),
                    "option_expiration": parsed["expiration"].isoformat(),
                    "option_dte": int(dte),
                })
                high_plpc.pop(symbol, None)

        state["option_high_plpc"] = high_plpc
        state["updated_at"] = now_utc().isoformat()
        self._save_state(state)
        return pd.DataFrame(rows)

    def build_orders(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        base_symbols = self._universe()

        # Direct scan: download the configured universe plus SPY/QQQ/VIX proxy
        # for regime-aware risk/exposure controls. Regime does not gate inverse ETFs.
        initial_symbols = list(dict.fromkeys(base_symbols + self._regime_required_symbols()))
        print(f"[scanner] scanning direct universe symbols={len(initial_symbols)}")
        bars = self._download_bars(initial_symbols)
        if bars.empty:
            print("[scanner] no bars downloaded; keeping existing order files")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        regime = self._classify_market_regime(bars)
        routed_symbols = self._route_universe_for_regime(base_symbols, regime)

        if set(routed_symbols) - set(initial_symbols):
            print(f"[scanner] redownloading expanded direct universe symbols={len(routed_symbols)}")
            bars = self._download_bars(routed_symbols)
            if bars.empty:
                print("[scanner] no bars downloaded after universe expansion; exits only")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            regime = self._classify_market_regime(bars)

        features = self._build_features(bars)
        scores = self._score_latest(features)
        scores = self._apply_market_leadership(scores, bars)
        scores = self._apply_regime_to_scores(scores, regime)
        if scores.empty:
            print("[scanner] no symbols passed scanner/regime filters")
            acct = self.broker.get_account_snapshot()
            equity = max(float(acct.equity), 1.0)
            exits = self._update_high_water_and_exits(pd.DataFrame(), equity)
            option_exits = self._build_option_exit_orders(equity)
            if not option_exits.empty:
                exits = pd.concat([exits, option_exits], ignore_index=True)
            pd.DataFrame().to_csv(DEFAULT_REBALANCE_ORDERS, index=False)
            exits.to_csv(DEFAULT_EXIT_ORDERS, index=False)
            return pd.DataFrame(), exits, pd.DataFrame()

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        scores.to_csv(SCANNER_SCORES_PATH, index=False)
        print("[scanner] top setups:")
        show_cols = [
            "symbol", "market_regime", "risk_state", "close", "setup_type",
            "setup_score", "trade_confidence", "is_inverse_etf", "lead_score",
            "best_lag_corr", "leadership_confirm", "momentum_score",
            "reversal_score", "rsi14", "ret_5d", "rel_volume",
        ]
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

            is_inverse = symbol in INVERSE_ETF_SET
            # Inverse ETFs are normal long-only candidates. No bearish-regime gate.

            current_mv = positions.get(symbol).market_value if symbol in positions else 0.0
            current_weight = float(current_mv) / equity
            type_mult = self.config.reversal_target_weight_mult if r["setup_type"] == "reversal" else 1.0
            target_weight = min(
                self.config.max_target_weight,
                self.config.base_target_weight * (0.50 + 1.50 * float(r["setup_score"])) * type_mult,
                self.executor_config.max_position_weight,
                regime.max_position_weight,
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
                    "reason": f"scanner_{r['setup_type']}_{regime.state.lower()}_candidate",
                    "target_weight": target_weight,
                    "current_weight": current_weight,
                    "delta_weight": delta_weight,
                    "trade_confidence": float(r["trade_confidence"]),
                    "confidence_score": float(r["setup_score"]),
                    "market_regime": regime.state,
                    "risk_state": regime.risk_state,
                    "regime_max_gross_exposure": regime.max_gross_exposure,
                    "regime_max_position_weight": regime.max_position_weight,
                    "is_inverse_etf": bool(is_inverse),
                    "momentum_score": float(r["momentum_score"]),
                    "reversal_score": float(r["reversal_score"]),
                    "rsi14": to_float(r.get("rsi14")),
                    "ret_5d": to_float(r.get("ret_5d")),
                    "rel_volume": to_float(r.get("rel_volume")),
                    "lead_score": to_float(r.get("lead_score")),
                    "best_lag_corr": to_float(r.get("best_lag_corr")),
                    "avg_lag_corr": to_float(r.get("avg_lag_corr")),
                    "leadership_confirm": to_float(r.get("leadership_confirm")),
                    "leadership_divergence_warning": bool(r.get("leadership_divergence_warning", False)),
                }
            )

        rebalance = pd.DataFrame(rows)

        # Options booster is disabled in defensive/chop/inverse-only regimes.
        option_boosters = pd.DataFrame()
        if regime.allow_options_booster:
            option_boosters = self._build_option_booster_orders(scores, equity)
        else:
            print(f"[options] booster disabled by regime={regime.state}")

        if not option_boosters.empty:
            option_boosters["market_regime"] = regime.state
            option_boosters["risk_state"] = regime.risk_state
            rebalance = pd.concat([rebalance, option_boosters], ignore_index=True)

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

    config.account_scaled_max_gross_exposure = min(MAX_GROSS_EXPOSURE, max_gross)
    config.account_scaled_max_position_weight = min(MAX_POSITION_WEIGHT, max_pos)
    config.account_scaled_max_single_order_notional_pct = min(MAX_SINGLE_ORDER_NOTIONAL_PCT, max_order_pct)
    config.account_scaled_max_order_dollars = min(MAX_ORDER_DOLLARS, max(100.0, equity * max_order_pct))

    config.max_gross_exposure = min(config.max_gross_exposure, config.account_scaled_max_gross_exposure)
    config.max_position_weight = min(config.max_position_weight, config.account_scaled_max_position_weight)
    config.max_single_order_notional_pct = min(config.max_single_order_notional_pct, config.account_scaled_max_single_order_notional_pct)
    config.max_order_dollars = min(config.max_order_dollars, config.account_scaled_max_order_dollars)

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
        if self.config.portfolio_sentiment is None:
            self.config.portfolio_sentiment = PortfolioSentimentOverlayConfig()
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
            "market_regime": "",
            "risk_state": "",
            "is_inverse_etf": False,
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

    # -------------------------------------------------------------------------
    # PDT guard rails
    # -------------------------------------------------------------------------

    def _read_pdt_state(self) -> dict[str, Any]:
        if not PDT_STATE_PATH.exists():
            return {"day_trades": [], "same_day_buys": {}, "same_day_sells": {}}
        try:
            state = json.loads(PDT_STATE_PATH.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                raise ValueError("PDT state is not a dict")
            state.setdefault("day_trades", [])
            state.setdefault("same_day_buys", {})
            state.setdefault("same_day_sells", {})
            return state
        except Exception:
            return {"day_trades": [], "same_day_buys": {}, "same_day_sells": {}}

    def _write_pdt_state(self, state: dict[str, Any]) -> None:
        PDT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = now_utc().isoformat()
        PDT_STATE_PATH.write_text(json.dumps(state, indent=2, default=json_default) + "\n", encoding="utf-8")

    def _pdt_cutoff_date(self) -> datetime.date:
        return (now_utc() - timedelta(days=max(1, self.config.pdt_rolling_window_days + 2))).date()

    def _pdt_prune_state(self, state: dict[str, Any]) -> dict[str, Any]:
        today = now_utc().date().isoformat()
        cutoff = self._pdt_cutoff_date().isoformat()

        state["day_trades"] = [
            d for d in state.get("day_trades", [])
            if str(d.get("date", "")) >= cutoff
        ]
        state["same_day_buys"] = {
            safe_symbol(sym): lots for sym, lots in dict(state.get("same_day_buys", {})).items()
            if any(str(lot.get("date", "")) == today for lot in lots)
        }
        state["same_day_sells"] = {
            safe_symbol(sym): lots for sym, lots in dict(state.get("same_day_sells", {})).items()
            if any(str(lot.get("date", "")) == today for lot in lots)
        }
        return state

    def _pdt_recent_day_trade_count(self, state: dict[str, Any]) -> int:
        cutoff = self._pdt_cutoff_date().isoformat()
        return int(sum(1 for d in state.get("day_trades", []) if str(d.get("date", "")) >= cutoff))

    def _pdt_opened_today_qty(self, state: dict[str, Any], symbol: str) -> float:
        today = now_utc().date().isoformat()
        lots = state.get("same_day_buys", {}).get(safe_symbol(symbol), [])
        return float(sum(to_float(lot.get("remaining_qty", lot.get("qty", 0.0))) for lot in lots if str(lot.get("date", "")) == today))

    def _pdt_is_emergency_sell(self, reason: str) -> bool:
        text = str(reason or "").lower()
        emergency_terms = [
            "stop", "loss", "kill", "risk", "drawdown", "cash", "margin",
            "expiration", "hard_stop", "profit_lock", "giveback",
        ]
        return any(term in text for term in emergency_terms)

    def _pdt_guard_allows_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        equity: float,
        reason: str,
    ) -> tuple[bool, str, dict[str, Any]]:
        if not self.config.enable_pdt_guard:
            return True, "pdt_guard_disabled", {}

        state = self._pdt_prune_state(self._read_pdt_state())
        broker_flags = self.broker.get_account_flags()
        local_count = self._pdt_recent_day_trade_count(state)
        broker_count = to_int(broker_flags.get("daytrade_count", 0), 0)
        day_trade_count = max(local_count, broker_count)
        remaining = max(0, self.config.pdt_max_day_trades - day_trade_count)
        under_threshold = float(equity or 0.0) < float(self.config.pdt_equity_threshold)

        meta = {
            "equity": float(equity or 0.0),
            "under_pdt_equity_threshold": under_threshold,
            "pdt_equity_threshold": self.config.pdt_equity_threshold,
            "local_day_trade_count": local_count,
            "broker_daytrade_count": broker_count,
            "day_trade_count_used": day_trade_count,
            "day_trades_remaining": remaining,
            "broker_flags": broker_flags,
        }

        if not under_threshold:
            return True, "pdt_equity_above_threshold", meta

        if broker_flags.get("pattern_day_trader"):
            return False, "pdt_broker_account_marked_pattern_day_trader_under_25k", meta
        if broker_flags.get("trading_blocked") or broker_flags.get("account_blocked"):
            return False, "pdt_broker_account_or_trading_blocked", meta

        side = str(side).lower()
        symbol = safe_symbol(symbol)

        # Strict mode is useful for small accounts that want to swing trade only.
        # It allows risk-reducing sells but blocks new opening buys under $25K.
        if side == "buy" and self.config.pdt_strict_under_25k:
            return False, "pdt_strict_under_25k_blocks_new_buy", meta

        if side == "buy" and self.config.pdt_block_new_buys_when_near_limit:
            if day_trade_count >= max(0, self.config.pdt_max_day_trades - self.config.pdt_reserve_day_trades):
                return False, "pdt_near_day_trade_limit_blocks_new_buy", meta

        if side == "sell" and self.config.pdt_block_same_day_exits_under_25k:
            opened_today_qty = self._pdt_opened_today_qty(state, symbol)
            meta["opened_today_qty"] = opened_today_qty
            if opened_today_qty > 0:
                emergency_sell = self.config.pdt_allow_emergency_sells and self._pdt_is_emergency_sell(reason)
                if remaining <= self.config.pdt_reserve_day_trades and not emergency_sell:
                    return False, "pdt_same_day_exit_would_use_reserved_day_trade", meta
                if not emergency_sell:
                    return False, "pdt_same_day_exit_blocked_under_25k", meta

        return True, "pdt_ok", meta

    def _pdt_record_fill(self, symbol: str, side: str, qty: float, order_id: str, fill_info: dict[str, Any]) -> None:
        if not self.config.enable_pdt_guard:
            return

        filled_qty = to_float(fill_info.get("filled_qty", qty), 0.0)
        if filled_qty <= 0:
            return

        symbol = safe_symbol(symbol)
        side = str(side).lower()
        today = now_utc().date().isoformat()
        state = self._pdt_prune_state(self._read_pdt_state())
        state.setdefault("same_day_buys", {})
        state.setdefault("same_day_sells", {})
        state.setdefault("day_trades", [])

        if side == "buy":
            state["same_day_buys"].setdefault(symbol, []).append({
                "date": today,
                "qty": float(filled_qty),
                "remaining_qty": float(filled_qty),
                "order_id": order_id,
                "timestamp_utc": now_utc().isoformat(),
            })

        elif side == "sell":
            remaining_sell = float(filled_qty)
            lots = state["same_day_buys"].get(symbol, [])
            matched_today_qty = 0.0
            for lot in lots:
                if str(lot.get("date", "")) != today:
                    continue
                lot_remaining = to_float(lot.get("remaining_qty", lot.get("qty", 0.0)), 0.0)
                if lot_remaining <= 0 or remaining_sell <= 0:
                    continue
                matched = min(lot_remaining, remaining_sell)
                lot["remaining_qty"] = lot_remaining - matched
                remaining_sell -= matched
                matched_today_qty += matched

            state["same_day_sells"].setdefault(symbol, []).append({
                "date": today,
                "qty": float(filled_qty),
                "matched_same_day_buy_qty": float(matched_today_qty),
                "order_id": order_id,
                "timestamp_utc": now_utc().isoformat(),
            })

            if matched_today_qty > 0:
                state["day_trades"].append({
                    "date": today,
                    "symbol": symbol,
                    "qty": float(matched_today_qty),
                    "sell_order_id": order_id,
                    "timestamp_utc": now_utc().isoformat(),
                    "source": "local_fill_match",
                })

        self._write_pdt_state(state)

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

            pdt_allowed, pdt_message, pdt_meta = self._pdt_guard_allows_order(
                symbol=symbol,
                side=side,
                qty=qty,
                equity=equity,
                reason=reason,
            )
            if not pdt_allowed:
                print(f"[pdt] skip {side.upper()} {qty} {symbol}: {pdt_message}")
                result = {
                    "timestamp_utc": now_utc().isoformat(),
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "status": "skipped_pdt_guard",
                    "message": pdt_message,
                    **{f"pdt_{k}": v for k, v in pdt_meta.items() if k != "broker_flags"},
                }
                results.append(result)
                acct_after = self._record_account_snapshot("after_skip", cycle_id)
                acct_after["gross_exposure"] = self._gross_exposure(float(acct_after["equity"]))
                self._journal_order(cycle_id, "SKIP", row, acct_before, acct_after, pdt_message)
                write_event("order_skipped_pdt_guard", {**result, "broker_flags": pdt_meta.get("broker_flags", {})})
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
                self._pdt_record_fill(symbol, side, qty, order_id, fill_info)
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
    # OpenAI portfolio sentiment overlay
    # -------------------------------------------------------------------------

    def _sentiment_monitor_config(self):
        cfg = self.config.portfolio_sentiment or PortfolioSentimentOverlayConfig(enabled=False)
        if SentimentMonitorConfig is None:
            return None
        return SentimentMonitorConfig(
            model=cfg.model,
            max_headlines_per_symbol=cfg.max_headlines_per_symbol,
            reduce_score_threshold=cfg.reduce_score_threshold,
            exit_score_threshold=cfg.exit_score_threshold,
            min_confidence_for_action=cfg.min_confidence_for_action,
            bullish_hold_threshold=cfg.bullish_hold_threshold,
            watch_multiplier=cfg.watch_multiplier,
            reduce_multiplier=cfg.reduce_multiplier,
            exit_multiplier=cfg.exit_multiplier,
            bullish_hold_multiplier=cfg.bullish_hold_multiplier,
            max_bullish_multiplier=cfg.max_bullish_multiplier,
        )

    def _run_portfolio_sentiment_monitor(self, positions_before: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.portfolio_sentiment or PortfolioSentimentOverlayConfig(enabled=False)
        if not cfg.enabled:
            return pd.DataFrame()

        if positions_before is None or positions_before.empty:
            empty = pd.DataFrame()
            save_df(empty, LATEST_POSITION_SENTIMENT_PATH)
            print("[sentiment] no open positions; portfolio sentiment monitor skipped")
            return empty

        # Monitor stock underlyings and existing positions. Do not monitor the full
        # trading universe. Option positions are retained in the broker state, but
        # the monitor itself works best on normal equity symbols. The option exit
        # overlay below maps option contracts back to their underlyings.
        pos = positions_before.copy()
        pos["symbol"] = pos["symbol"].map(safe_symbol)
        pos["qty"] = pd.to_numeric(pos.get("qty", 0.0), errors="coerce").fillna(0.0)
        pos = pos[pos["qty"] > 0].copy()
        if pos.empty:
            empty = pd.DataFrame()
            save_df(empty, LATEST_POSITION_SENTIMENT_PATH)
            print("[sentiment] no long positions; portfolio sentiment monitor skipped")
            return empty

        if run_portfolio_sentiment_monitor is None or SentimentMonitorConfig is None:
            print("[sentiment] portfolio_sentiment_monitor.py not available; skipping LLM overlay")
            write_event("portfolio_sentiment_unavailable", {"reason": "module_import_failed"})
            return pd.DataFrame()

        try:
            monitor_cfg = self._sentiment_monitor_config()
            sentiment = run_portfolio_sentiment_monitor(
                positions_df=pos,
                paper=self.config.paper,
                config=monitor_cfg,
                use_openai=bool(cfg.use_openai),
            )
            if sentiment is None:
                sentiment = pd.DataFrame()
            if not sentiment.empty:
                sentiment.to_csv(LATEST_POSITION_SENTIMENT_PATH, index=False)
            else:
                save_df(sentiment, LATEST_POSITION_SENTIMENT_PATH)
            print(f"[sentiment] monitor complete rows={len(sentiment)} openai={cfg.use_openai}")
            write_event(
                "portfolio_sentiment_monitor_complete",
                {"rows": len(sentiment), "openai": bool(cfg.use_openai)},
            )
            return sentiment
        except Exception as exc:
            print(f"[sentiment] monitor failed; continuing without LLM overlay: {exc}")
            write_event("portfolio_sentiment_monitor_error", {"error": str(exc)})
            return pd.DataFrame()

    def _sentiment_lookup(self, sentiment_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        if sentiment_df is None or sentiment_df.empty or "symbol" not in sentiment_df.columns:
            return {}
        s = sentiment_df.copy()
        s["symbol"] = s["symbol"].map(safe_symbol)
        return {
            str(row["symbol"]): row.to_dict()
            for _, row in s.drop_duplicates("symbol", keep="last").iterrows()
            if str(row.get("symbol", ""))
        }

    def _sentiment_multiplier(self, row: dict[str, Any]) -> float:
        cfg = self.config.portfolio_sentiment or PortfolioSentimentOverlayConfig(enabled=False)
        pressure = str(row.get("exit_pressure", "hold")).lower().strip()
        score = to_float(row.get("sentiment_score", 0.0))
        conf = to_float(row.get("sentiment_confidence", 0.0))
        source_quality = to_float(row.get("source_quality", 0.0))

        if pressure == "exit" and conf >= cfg.min_confidence_for_action:
            return cfg.exit_multiplier
        if pressure == "reduce" and conf >= cfg.min_confidence_for_action:
            return cfg.reduce_multiplier
        if pressure == "watch":
            return cfg.watch_multiplier
        if (
            pressure == "hold"
            and score >= cfg.bullish_hold_threshold
            and conf >= cfg.min_confidence_for_action
            and source_quality >= 0.55
        ):
            return min(cfg.max_bullish_multiplier, cfg.bullish_hold_multiplier)
        return 1.0

    def _apply_sentiment_to_rebalance_orders(
        self,
        rebalance: pd.DataFrame,
        sentiment_df: pd.DataFrame,
    ) -> pd.DataFrame:
        cfg = self.config.portfolio_sentiment or PortfolioSentimentOverlayConfig(enabled=False)
        if not cfg.enabled or rebalance is None or rebalance.empty or sentiment_df is None or sentiment_df.empty:
            return rebalance.copy() if rebalance is not None else pd.DataFrame()

        lookup = self._sentiment_lookup(sentiment_df)
        if not lookup:
            return rebalance.copy()

        out = rebalance.copy()
        if "symbol" not in out.columns:
            return out
        out["symbol"] = out["symbol"].map(safe_symbol)
        if "underlying_symbol" in out.columns:
            out["underlying_symbol"] = out["underlying_symbol"].map(safe_symbol)
        else:
            out["underlying_symbol"] = ""

        keep_rows: list[bool] = []
        multipliers: list[float] = []
        pressures: list[str] = []
        scores: list[float] = []
        confidences: list[float] = []
        summaries: list[str] = []

        for _, r in out.iterrows():
            symbol = safe_symbol(r.get("symbol"))
            underlying = safe_symbol(r.get("underlying_symbol"))
            reason = str(r.get("reason", ""))
            is_option_booster = "option_booster" in reason.lower() or bool(underlying)
            lookup_symbol = underlying if is_option_booster and underlying else symbol
            srow = lookup.get(lookup_symbol)

            if not srow:
                keep_rows.append(True)
                multipliers.append(1.0)
                pressures.append("")
                scores.append(0.0)
                confidences.append(0.0)
                summaries.append("")
                continue

            mult = self._sentiment_multiplier(srow)
            pressure = str(srow.get("exit_pressure", "hold")).lower()
            options_allowed = bool(srow.get("options_boost_allowed", False))

            if is_option_booster and cfg.gate_options_booster and not options_allowed:
                keep_rows.append(False)
                multipliers.append(0.0)
            else:
                keep_rows.append(True)
                multipliers.append(mult)

            pressures.append(pressure)
            scores.append(to_float(srow.get("sentiment_score")))
            confidences.append(to_float(srow.get("sentiment_confidence")))
            summaries.append(str(srow.get("summary", "")))

        out["sentiment_overlay_mult"] = multipliers
        out["sentiment_exit_pressure"] = pressures
        out["sentiment_score"] = scores
        out["sentiment_confidence"] = confidences
        out["sentiment_summary"] = summaries

        numeric_cols = [
            "delta_shares", "delta_dollars", "target_weight", "current_weight",
            "delta_weight", "trade_confidence", "confidence_score",
        ]
        # Only scale BUY/add orders. Sells are risk-reduction orders and should not
        # be weakened by sentiment overlay.
        side = out.get("side", "").astype(str).str.lower() if "side" in out.columns else pd.Series("", index=out.index)
        buy_mask = side.eq("buy")
        for col in numeric_cols:
            if col in out.columns:
                vals = pd.to_numeric(out[col], errors="coerce")
                out.loc[buy_mask, col] = vals.loc[buy_mask].fillna(0.0) * out.loc[buy_mask, "sentiment_overlay_mult"]

        if "reason" in out.columns:
            out["reason"] = out["reason"].fillna("").astype(str) + out.apply(
                lambda r: (
                    f" | llm_sentiment={r.get('sentiment_exit_pressure', '')}"
                    f" score={to_float(r.get('sentiment_score')):+.2f}"
                    f" mult={to_float(r.get('sentiment_overlay_mult'), 1.0):.2f}"
                ) if str(r.get("sentiment_exit_pressure", "")) else "",
                axis=1,
            )

        before = len(out)
        out = out[pd.Series(keep_rows, index=out.index)].copy()
        dropped = before - len(out)
        if dropped:
            print(f"[sentiment] blocked option/entry rows={dropped} due to portfolio sentiment gate")
            write_event("portfolio_sentiment_blocked_orders", {"blocked_rows": dropped})

        save_df(out, SENTIMENT_ADJUSTED_REBALANCE_ORDERS_PATH)
        return out.reset_index(drop=True)

    def _build_sentiment_exit_orders(
        self,
        positions_before: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        equity: float,
    ) -> pd.DataFrame:
        cfg = self.config.portfolio_sentiment or PortfolioSentimentOverlayConfig(enabled=False)
        if not cfg.enabled or positions_before is None or positions_before.empty or sentiment_df is None or sentiment_df.empty:
            return pd.DataFrame()

        lookup = self._sentiment_lookup(sentiment_df)
        if not lookup:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        pos = positions_before.copy()
        pos["symbol"] = pos["symbol"].map(safe_symbol)
        for col in ["qty", "market_value", "current_price", "unrealized_plpc"]:
            if col not in pos.columns:
                pos[col] = 0.0
            pos[col] = pd.to_numeric(pos[col], errors="coerce").fillna(0.0)

        for _, p in pos.iterrows():
            symbol = safe_symbol(p.get("symbol"))
            if not symbol or p.get("qty", 0.0) <= 0:
                continue
            parsed = parse_occ_option_symbol(symbol)
            underlying = safe_symbol(parsed.get("underlying")) if parsed else symbol
            srow = lookup.get(underlying)
            if not srow:
                continue

            pressure = str(srow.get("exit_pressure", "hold")).lower().strip()
            conf = to_float(srow.get("sentiment_confidence", 0.0))
            score = to_float(srow.get("sentiment_score", 0.0))
            if conf < cfg.min_confidence_for_action:
                continue

            qty = int(to_float(p.get("qty", 0.0)))
            if qty <= 0:
                continue

            if pressure == "exit":
                sell_qty = qty
                reason = "llm_portfolio_sentiment_exit"
            elif pressure == "reduce" and not parsed:
                sell_qty = max(1, int(np.ceil(qty * cfg.reduce_sell_fraction)))
                reason = "llm_portfolio_sentiment_reduce"
            else:
                continue

            price = to_float(p.get("current_price", 0.0))
            mv = to_float(p.get("market_value", 0.0))
            est_dollars = abs(price * sell_qty) if price > 0 else abs(mv) * min(1.0, sell_qty / max(qty, 1))
            rows.append({
                "symbol": symbol,
                "side": "sell",
                "delta_shares": sell_qty,
                "delta_dollars": est_dollars,
                "price": price,
                "reason": f"{reason} score={score:+.2f} conf={conf:.2f} summary={str(srow.get('summary', ''))[:160]}",
                "target_weight": 0.0,
                "current_weight": mv / max(equity, 1e-9),
                "delta_weight": -min(mv, est_dollars) / max(equity, 1e-9),
                "trade_confidence": 1.0,
                "sentiment_score": score,
                "sentiment_confidence": conf,
                "sentiment_exit_pressure": pressure,
                "underlying_symbol": underlying if parsed else "",
            })

        exits = pd.DataFrame(rows)
        if not exits.empty:
            print(f"[sentiment] generated risk-reduction exits={len(exits)}")
            write_event("portfolio_sentiment_exit_orders", {"rows": len(exits)})
        return exits

    def _apply_portfolio_sentiment_overlay(
        self,
        rebalance_orders: pd.DataFrame,
        exit_orders: pd.DataFrame | None,
        positions_before: pd.DataFrame,
        equity: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cfg = self.config.portfolio_sentiment or PortfolioSentimentOverlayConfig(enabled=False)
        exits = exit_orders.copy() if exit_orders is not None else pd.DataFrame()
        if not cfg.enabled:
            return rebalance_orders, exits, pd.DataFrame()

        sentiment = self._run_portfolio_sentiment_monitor(positions_before)
        if sentiment.empty:
            return rebalance_orders, exits, sentiment

        adjusted_rebalance = self._apply_sentiment_to_rebalance_orders(rebalance_orders, sentiment)
        sentiment_exits = self._build_sentiment_exit_orders(positions_before, sentiment, equity)
        if not sentiment_exits.empty:
            exits = pd.concat([exits, sentiment_exits], ignore_index=True)

        save_df(exits if exits is not None else pd.DataFrame(), SENTIMENT_ADJUSTED_EXIT_ORDERS_PATH)
        return adjusted_rebalance, exits, sentiment

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

        rebalance_orders, exit_orders, sentiment_df = self._apply_portfolio_sentiment_overlay(
            rebalance_orders=rebalance_orders,
            exit_orders=exit_orders if exit_orders is not None else pd.DataFrame(),
            positions_before=positions_before,
            equity=equity,
        )

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
        print(f"[executor] options_booster={bool(self.config.options_booster and self.config.options_booster.enabled)}")
        print(f"[executor] portfolio_sentiment={bool(self.config.portfolio_sentiment and self.config.portfolio_sentiment.enabled)} openai={bool(self.config.portfolio_sentiment and self.config.portfolio_sentiment.use_openai)}")
        leadership_enabled = bool(self.config.scanner and self.config.scanner.leadership and self.config.scanner.leadership.enabled)
        leadership_benchmark = self.config.scanner.leadership.benchmark if leadership_enabled else "disabled"
        print(f"[executor] market_leadership={leadership_enabled} benchmark={leadership_benchmark}")
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

    parser.add_argument("--disable-pdt-guard", action="store_true", help="Disable local PDT guard rails. Broker-side PDT restrictions can still reject orders.")
    parser.add_argument("--pdt-equity-threshold", type=float, default=PDT_EQUITY_THRESHOLD)
    parser.add_argument("--pdt-max-day-trades", type=int, default=PDT_MAX_DAY_TRADES)
    parser.add_argument("--pdt-reserve-day-trades", type=int, default=PDT_RESERVE_DAY_TRADES, help="Reserve this many day trades for emergency exits.")
    parser.add_argument("--pdt-strict-under-25k", action="store_true", help="Under the PDT threshold, block new opening buys and only allow risk-reducing sells.")
    parser.add_argument("--allow-pdt-same-day-exits", action="store_true", help="Allow same-day exits under $25K unless the max day-trade count is reached.")
    parser.add_argument("--allow-pdt-new-buys-near-limit", action="store_true", help="Do not block new buys when near the day-trade limit.")
    parser.add_argument("--disable-pdt-emergency-sells", action="store_true", help="Do not allow emergency/risk-reducing same-day sells to bypass PDT same-day-exit blocking.")

    parser.add_argument("--loop", action="store_true", help="Run continuously. This is already the default; kept for compatibility.")
    parser.add_argument("--once", action="store_true", help="Run one cycle then exit. Default is to run forever.")

    parser.add_argument("--disable-scanner", action="store_true", help="Disable internal universe scanner and only read order CSVs.")
    parser.add_argument("--universe", default="", help="Comma-separated universe. Default scans DEFAULT_SCAN_UNIVERSE directly from Alpaca daily bars.")
    parser.add_argument("--use-sp500-fallback", action="store_true", help="Scan S&P 500 symbols from Wikipedia, capped by --max-universe-symbols.")
    parser.add_argument("--max-universe-symbols", type=int, default=80)
    parser.add_argument("--scanner-top-n", type=int, default=8)
    parser.add_argument("--min-setup-score", type=float, default=0.62)
    parser.add_argument("--min-buy-confidence", type=float, default=0.58)
    parser.add_argument("--min-avg-dollar-volume", type=float, default=5_000_000.0)
    parser.add_argument("--scanner-max-target-weight", type=float, default=0.14)
    parser.add_argument("--scanner-stop-loss", type=float, default=-0.035)
    parser.add_argument("--profit-lock-trigger", type=float, default=0.06)
    parser.add_argument("--profit-giveback", type=float, default=0.025)

    parser.add_argument("--disable-market-leadership", action="store_true", help="Disable lagged leadership confirmation inside the scanner.")
    parser.add_argument("--leadership-benchmark", default="QQQ", help="Benchmark used for lagged leadership scoring, usually QQQ or SPY.")
    parser.add_argument("--leadership-lookback", type=int, default=120)
    parser.add_argument("--leadership-min-obs", type=int, default=60)
    parser.add_argument("--leadership-min-score", type=float, default=0.05)
    parser.add_argument("--leadership-weight", type=float, default=0.18)
    parser.add_argument("--require-positive-leadership", action="store_true", help="Only buy symbols with positive leadership scores.")

    parser.add_argument("--disable-regime-engine", action="store_true", help="Disable SPY/QQQ/VIX regime classification and regime routing.")
    parser.add_argument("--regime-spy-symbol", default="SPY")
    parser.add_argument("--regime-qqq-symbol", default="QQQ")
    parser.add_argument("--regime-vix-symbol", default="VIXY", help="Volatility proxy symbol. VIXY is used by default because Alpaca stock bars may not expose cash ^VIX.")
    parser.add_argument("--risk-on-gross", type=float, default=0.92)
    parser.add_argument("--volatile-risk-on-gross", type=float, default=0.72)
    parser.add_argument("--chop-gross", type=float, default=0.50)
    parser.add_argument("--defensive-gross", type=float, default=0.25)
    parser.add_argument("--bear-inverse-gross", type=float, default=0.35)
    parser.add_argument("--disable-bear-inverse-routing", action="store_true", help="Compatibility flag only; inverse ETFs are always scored as normal symbols.")
    parser.add_argument("--allow-inverse-outside-bear", action="store_true", default=True, help="Compatibility flag only; inverse ETFs are allowed in every regime by default.")

    parser.add_argument("--enable-options-booster", dest="enable_options_booster", action="store_true", default=True, help="Enable long-call-only booster after positive underlying validation. Enabled by default.")
    parser.add_argument("--disable-options-booster", dest="enable_options_booster", action="store_false", help="Disable the long-call-only options booster.")
    parser.add_argument("--option-min-confidence", type=float, default=0.68)
    parser.add_argument("--option-min-underlying-plpc", type=float, default=0.003)
    parser.add_argument("--option-min-dte", type=int, default=14)
    parser.add_argument("--option-max-dte", type=int, default=45)
    parser.add_argument("--option-max-risk-pct", type=float, default=0.010)
    parser.add_argument("--option-max-risk-dollars", type=float, default=250.0)
    parser.add_argument("--option-max-contracts", type=int, default=2)
    parser.add_argument("--option-max-open-positions", type=int, default=4)
    parser.add_argument("--option-max-spread-pct", type=float, default=0.15)
    parser.add_argument(
        "--option-close-before-expiration-days",
        type=int,
        default=3,
        help="Force-close long call positions when DTE is at or below this value.",
    )
    parser.add_argument(
        "--option-block-new-entries-within-expiration-days",
        type=int,
        default=5,
        help="Do not open new calls this close to expiration.",
    )

    parser.add_argument("--disable-portfolio-sentiment", action="store_true", help="Disable OpenAI portfolio constituent sentiment monitoring.")
    parser.add_argument("--disable-openai-sentiment", action="store_true", help="Run the sentiment overlay in neutral/no-OpenAI mode.")
    parser.add_argument("--sentiment-model", default=os.getenv("OPENAI_SENTIMENT_MODEL", "gpt-5.4-mini"), help="OpenAI model used by portfolio_sentiment_monitor.py.")
    parser.add_argument("--sentiment-max-headlines", type=int, default=8)
    parser.add_argument("--sentiment-reduce-threshold", type=float, default=-0.55)
    parser.add_argument("--sentiment-exit-threshold", type=float, default=-0.75)
    parser.add_argument("--sentiment-min-confidence", type=float, default=0.70)
    parser.add_argument("--sentiment-watch-multiplier", type=float, default=0.85)
    parser.add_argument("--sentiment-reduce-multiplier", type=float, default=0.50)
    parser.add_argument("--sentiment-reduce-sell-fraction", type=float, default=0.50)
    parser.add_argument("--disable-sentiment-option-gate", action="store_true", help="Do not use portfolio sentiment to gate long-call booster entries.")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExecutorConfig:
    scanner_universe = [normalize_symbol(x) for x in str(args.universe).split(",") if normalize_symbol(x)] or None
    leadership_config = LeadershipConfig(
        enabled=not args.disable_market_leadership,
        benchmark=normalize_symbol(args.leadership_benchmark or "QQQ"),
        lookback=args.leadership_lookback,
        min_obs=args.leadership_min_obs,
        min_lead_score=args.leadership_min_score,
        leadership_weight=args.leadership_weight,
        require_positive_leadership=args.require_positive_leadership,
    )

    regime_config = RegimeConfig(
        enabled=not args.disable_regime_engine,
        spy_symbol=normalize_symbol(args.regime_spy_symbol or "SPY"),
        qqq_symbol=normalize_symbol(args.regime_qqq_symbol or "QQQ"),
        vix_proxy_symbol=normalize_symbol(args.regime_vix_symbol or "VIXY"),
        risk_on_gross=args.risk_on_gross,
        volatile_risk_on_gross=args.volatile_risk_on_gross,
        chop_gross=args.chop_gross,
        defensive_gross=args.defensive_gross,
        bear_inverse_gross=args.bear_inverse_gross,
        allow_inverse_in_bear=not args.disable_bear_inverse_routing,
        block_inverse_when_not_bear=False,
        bear_market_only_inverse_routing=False,
    )

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
        leadership=leadership_config,
        regime=regime_config,
    )
    options_config = OptionsBoosterConfig(
        enabled=args.enable_options_booster,
        min_continuation_confidence=args.option_min_confidence,
        min_underlying_unrealized_plpc=args.option_min_underlying_plpc,
        min_dte=args.option_min_dte,
        max_dte=args.option_max_dte,
        max_risk_per_trade_pct=args.option_max_risk_pct,
        max_risk_per_trade_dollars=args.option_max_risk_dollars,
        max_contracts_per_underlying=args.option_max_contracts,
        max_open_option_positions=args.option_max_open_positions,
        max_spread_pct=args.option_max_spread_pct,
        close_before_expiration_days=args.option_close_before_expiration_days,
        block_new_entries_within_expiration_days=args.option_block_new_entries_within_expiration_days,
    )

    sentiment_config = PortfolioSentimentOverlayConfig(
        enabled=not args.disable_portfolio_sentiment,
        use_openai=not args.disable_openai_sentiment,
        model=args.sentiment_model,
        max_headlines_per_symbol=args.sentiment_max_headlines,
        reduce_score_threshold=args.sentiment_reduce_threshold,
        exit_score_threshold=args.sentiment_exit_threshold,
        min_confidence_for_action=args.sentiment_min_confidence,
        watch_multiplier=args.sentiment_watch_multiplier,
        reduce_multiplier=args.sentiment_reduce_multiplier,
        reduce_sell_fraction=args.sentiment_reduce_sell_fraction,
        gate_options_booster=not args.disable_sentiment_option_gate,
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
        enable_pdt_guard=not args.disable_pdt_guard,
        pdt_equity_threshold=args.pdt_equity_threshold,
        pdt_max_day_trades=args.pdt_max_day_trades,
        pdt_reserve_day_trades=args.pdt_reserve_day_trades,
        pdt_strict_under_25k=args.pdt_strict_under_25k,
        pdt_block_same_day_exits_under_25k=not args.allow_pdt_same_day_exits,
        pdt_block_new_buys_when_near_limit=not args.allow_pdt_new_buys_near_limit,
        pdt_allow_emergency_sells=not args.disable_pdt_emergency_sells,
        scanner=scanner_config,
        options_booster=options_config,
        portfolio_sentiment=sentiment_config,
    )


def main() -> None:
    args = parse_args()

    # =========================================================================
    # Build premarket LLM sentiment universe
    # =========================================================================

    if build_sentiment_universe is not None:
        try:
            print("\n" + "=" * 80)
            print("PREMARKET LLM SENTIMENT UNIVERSE")
            print("=" * 80)

            llm_payload = build_sentiment_universe()

            market_tone = llm_payload.get("market_tone", "unknown")
            summary = llm_payload.get("summary", "")
            symbols = llm_payload.get("symbols", [])

            print(f"[llm-universe] market_tone={market_tone}")
            print(f"[llm-universe] summary={summary}")
            print(f"[llm-universe] selected_symbols={len(symbols)}")

            if symbols:
                print(
                    "[llm-universe] universe="
                    + ",".join(symbols)
                )

            selections = llm_payload.get("selections", [])

            if selections:
                print("\n[llm-universe] reasoning:")

                for i, row in enumerate(selections[:10], start=1):
                    symbol = row.get("symbol", "")
                    reason = row.get("reason", "")
                    print(f"  {i:02d}. {symbol:<8} {reason}")

            print("=" * 80 + "\n")

            write_event(
                "premarket_llm_universe_generated",
                {
                    "market_tone": market_tone,
                    "summary": summary,
                    "symbols": symbols,
                    "symbol_count": len(symbols),
                },
            )

        except Exception as exc:
            print(f"[llm-universe] generation failed: {exc}")

            write_event(
                "premarket_llm_universe_failed",
                {
                    "error": str(exc),
                },
            )

    else:
        print("[llm-universe] selector module unavailable; using existing universe")

    # =========================================================================
    # Build executor config
    # =========================================================================

    config = build_config(args)

    # Force scanner to use the LLM-created universe JSON.
    if config.scanner is not None:
        config.scanner.universe = load_llm_sentiment_universe(
            json_path=LLM_SENTIMENT_UNIVERSE_JSON_PATH,
            locked_path=LOCKED_UNIVERSE_PATH,
            max_symbols=config.scanner.max_universe_symbols,
        )

        print(
            "[executor] loaded active LLM monitoring universe "
            f"symbols={len(config.scanner.universe)}"
        )

    # =========================================================================
    # Launch executor
    # =========================================================================

    print("\n" + "=" * 80)
    print("STARTING PRODUCTION ORDER EXECUTOR")
    print("=" * 80)

    trader = ProductionOrderExecutor(config)
    trader.run()


if __name__ == "__main__":
    main()