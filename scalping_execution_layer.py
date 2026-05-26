#!/usr/bin/env python3
"""
scalping_execution_layer.py

1-minute liquidity-sweep scalping layer.

Pattern (long only):
    HTF context gate (5m + 15m EMA bullish) →
    9:30 ET 5m opening-range wick established →
    1m candle closes below OR low  (sweep) →
    1m candle closes above OR low  (reclaim) →
    bearish retest bar forms (close < open, stays above OR low) →
    bullish engulf confirmed (close > prior open, volume expansion) →
    enter at market close of engulf bar →
    bracket order: stop one tick below sweep-wick low, target 3R

MAX_BARS_TO_WORK limit: if retest or engulf not found within N bars of
state entry, reset to RANGE_DEFINED and wait for the next sweep cycle.

Standalone:
    python scalping_execution_layer.py [--symbols AAPL,NVDA]

Called from run.py as a daemon thread when SCALP_LAYER_ENABLED=1.
Paper-safe by default. Never enables live trading.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from config.env_loader import load_instance_env

load_instance_env()

ET = ZoneInfo("America/New_York")
TODAY_TAG = datetime.now(ET).strftime("%Y%m%d")
DATA_TAG = os.getenv("RUN_DATA_DATE", os.getenv("RL_DATA_DATE", TODAY_TAG))

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent

# Log directory mirrors run.py pattern
STRATEGY_ID = os.getenv("STRATEGY_ID", "").strip()
if STRATEGY_ID:
    LOG_DIR = Path(os.getenv("LOG_DIR", str(PROJECT_ROOT / "logs" / "execution" / STRATEGY_ID)))
else:
    LOG_DIR = PROJECT_ROOT / "logs" / "execution" / DATA_TAG
LOG_DIR.mkdir(parents=True, exist_ok=True)

EVENTS_JSONL = LOG_DIR / f"{DATA_TAG}_execution_events.jsonl"
SCALP_STATE_PATH = LOG_DIR / f"{DATA_TAG}_scalp_layer_state.json"
SCALP_ORDERS_PATH = LOG_DIR / f"{DATA_TAG}_scalp_layer_orders.csv"

PREMARKET_TRADE_PLAN_PATH = PROJECT_ROOT / "system-data" / DATA_TAG / f"{DATA_TAG}_premarket_trade_plan.json"

# ── Alpaca imports ─────────────────────────────────────────────────────────────
try:
    from alpaca.common.exceptions import APIError
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

try:
    from alpaca.trading.enums import OrderClass
    from alpaca.trading.requests import StopLossRequest, TakeProfitRequest
    _BRACKET_AVAILABLE = True
except ImportError:
    _BRACKET_AVAILABLE = False
    OrderClass = None
    StopLossRequest = None
    TakeProfitRequest = None

# ── Environment config ─────────────────────────────────────────────────────────
_bool = lambda key, default: os.getenv(key, default).lower() in {"1", "true", "yes"}
_float = lambda key, default: float(os.getenv(key, str(default)))
_int = lambda key, default: int(os.getenv(key, str(default)))

EXECUTE_ORDERS: bool = _bool("SCALP_EXECUTE_ORDERS", "1")
PAPER: bool = _bool("APCA_PAPER", "true") or not _bool("ALLOW_LIVE_TRADING", "0")

APCA_KEY = os.getenv("APCA_API_KEY_ID", "")
APCA_SECRET = os.getenv("APCA_API_SECRET_KEY", "")
APCA_DATA_FEED = os.getenv("SCALP_DATA_FEED", "iex")


# ======================================================================
# Config
# ======================================================================
@dataclass
class ScalpConfig:
    # Time windows (ET)
    morning_start_hhmm: int = _int("SCALP_MORNING_START", 930)
    morning_end_hhmm: int = _int("SCALP_MORNING_END", 1030)
    power_hour_start_hhmm: int = _int("SCALP_POWER_HOUR_START", 1500)
    power_hour_end_hhmm: int = _int("SCALP_POWER_HOUR_END", 1600)

    # Opening range: which 5m bar defines the range (default: 9:30 bar = index 0)
    or_bar_index: int = 0
    # Reclaim: close must exceed OR low by at least this margin
    reclaim_buffer_pct: float = _float("SCALP_RECLAIM_BUFFER_PCT", "0.0001")
    # Retest: bearish bar whose low must stay above OR low minus tolerance
    retest_or_buffer_pct: float = _float("SCALP_RETEST_OR_BUFFER_PCT", "0.0015")
    # Engulf: bullish bar's close must exceed prior open by at least this
    engulf_body_buffer_pct: float = _float("SCALP_ENGULF_BODY_BUFFER_PCT", "0.0001")

    # Volume expansion for engulf bar
    min_volume_rvol: float = _float("SCALP_MIN_VOLUME_RVOL", "1.2")
    rvol_lookback_bars: int = _int("SCALP_RVOL_LOOKBACK", "20")

    # HTF context
    htf_ema_period: int = _int("SCALP_HTF_EMA_PERIOD", "9")
    require_5m_ema_bullish: bool = _bool("SCALP_REQUIRE_5M_EMA", "true")
    require_15m_ema_bullish: bool = _bool("SCALP_REQUIRE_15M_EMA", "true")

    # Risk / sizing
    target_rr: float = _float("SCALP_TARGET_RR", "3.0")
    max_order_dollars: float = _float("SCALP_MAX_ORDER_DOLLARS", "2000.0")
    min_order_dollars: float = _float("SCALP_MIN_ORDER_DOLLARS", "100.0")
    min_stop_distance_pct: float = _float("SCALP_MIN_STOP_DISTANCE_PCT", "0.0020")

    # State machine timeouts
    max_bars_to_work: int = _int("SCALP_MAX_BARS_TO_WORK", "3")
    cooldown_minutes: int = _int("SCALP_COOLDOWN_MINUTES", "5")

    # Data lookback for bar fetch
    bar_lookback_days: int = _int("SCALP_BAR_LOOKBACK_DAYS", "2")

    # Per-day limits
    max_trades_per_symbol_per_day: int = _int("SCALP_MAX_TRADES_PER_SYMBOL_PER_DAY", "2")
    max_trades_per_day_total: int = _int("SCALP_MAX_TRADES_PER_DAY_TOTAL", "6")


CFG = ScalpConfig()


# ======================================================================
# State machine
# ======================================================================
class ScalpState(str, Enum):
    IDLE = "IDLE"
    RANGE_DEFINED = "RANGE_DEFINED"
    SWEEP_DETECTED = "SWEEP_DETECTED"
    RECLAIM_DETECTED = "RECLAIM_DETECTED"
    RETEST_FORMED = "RETEST_FORMED"
    ENTRY_SUBMITTED = "ENTRY_SUBMITTED"
    COOLDOWN = "COOLDOWN"


@dataclass
class SymbolState:
    state: ScalpState = ScalpState.IDLE
    # Opening range
    or_high: float = 0.0
    or_low: float = 0.0
    or_bar_time: datetime | None = None
    # Sweep info
    sweep_wick_low: float = 0.0
    sweep_bar_time: datetime | None = None
    # Reclaim
    reclaim_bar_time: datetime | None = None
    bars_since_state_entry: int = 0
    # Retest bar for engulf reference
    retest_open: float = 0.0
    retest_close: float = 0.0
    retest_high: float = 0.0
    retest_low: float = 0.0
    retest_bar_time: datetime | None = None
    # Entry
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    # Cooldown
    cooldown_until: datetime | None = None
    # Daily trade count
    trades_today: int = 0
    last_trade_date: str = ""
    # Last processed bar time (avoid re-processing)
    last_processed_bar_time: datetime | None = None


# ======================================================================
# Helpers
# ======================================================================
def _now_et() -> datetime:
    return datetime.now(ET)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _hhmm(dt: datetime) -> int:
    return dt.hour * 100 + dt.minute


def get_tick_size(price: float) -> float:
    if price >= 1.0:
        return 0.01
    if price >= 0.10:
        return 0.001
    return 0.0001


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _compute_rvol(bars: pd.DataFrame, current_idx: int, lookback: int = 20) -> float:
    """Relative volume: current bar volume vs mean of prior N bars."""
    if current_idx < 1 or "volume" not in bars.columns:
        return 1.0
    start = max(0, current_idx - lookback)
    prior_vol = bars["volume"].iloc[start:current_idx]
    avg = prior_vol.mean()
    if avg <= 0:
        return 1.0
    return float(bars["volume"].iloc[current_idx]) / avg


def _write_event(event: str, payload: dict[str, Any]) -> None:
    try:
        row = {"event": event, "timestamp_utc": _now_utc_iso(), **payload}
        with EVENTS_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:
        pass


def _append_order_row(row: dict[str, Any]) -> None:
    try:
        df_new = pd.DataFrame([row])
        if SCALP_ORDERS_PATH.exists():
            df_new.to_csv(SCALP_ORDERS_PATH, mode="a", header=False, index=False)
        else:
            df_new.to_csv(SCALP_ORDERS_PATH, index=False)
    except Exception:
        pass


def _safe_symbol(s: str) -> str:
    return s.strip().upper().replace(".", "/")


def _load_premarket_plan_symbols() -> list[str]:
    """Load tradeable symbols from today's premarket trade plan."""
    try:
        if not PREMARKET_TRADE_PLAN_PATH.exists():
            return []
        data = json.loads(PREMARKET_TRADE_PLAN_PATH.read_text(encoding="utf-8"))
        symbols = []
        for entry in data.get("trade_plans", []):
            sym = entry.get("symbol", "")
            if sym:
                symbols.append(_safe_symbol(sym))
        return list(dict.fromkeys(symbols))
    except Exception:
        return []


def _resolve_symbols(override: list[str] | None = None) -> list[str]:
    """Resolve symbol list from CLI override, env var, or premarket plan."""
    if override:
        return [_safe_symbol(s) for s in override]
    env_syms = os.getenv("SCALP_SYMBOLS", "").strip()
    if env_syms:
        return [_safe_symbol(s) for s in env_syms.split(",") if s.strip()]
    return _load_premarket_plan_symbols()


# ======================================================================
# ScalpingEngine
# ======================================================================
class ScalpingEngine:
    def __init__(self) -> None:
        self._states: dict[str, SymbolState] = {}
        self._daily_total_trades: int = 0
        self._last_total_date: str = ""
        self._trading_client: Any = None
        self._data_client: Any = None
        self._init_clients()

    def _init_clients(self) -> None:
        if not _ALPACA_AVAILABLE:
            print("[scalp-layer] alpaca-py not available — orders disabled")
            return
        if not APCA_KEY or not APCA_SECRET:
            print("[scalp-layer] Alpaca credentials not found — orders disabled")
            return
        try:
            self._trading_client = TradingClient(APCA_KEY, APCA_SECRET, paper=PAPER)
            self._data_client = StockHistoricalDataClient(APCA_KEY, APCA_SECRET)
            print(f"[scalp-layer] clients ready paper={PAPER}")
        except Exception as exc:
            print(f"[scalp-layer] client init failed: {exc}")

    def _get_state(self, symbol: str) -> SymbolState:
        if symbol not in self._states:
            self._states[symbol] = SymbolState()
        return self._states[symbol]

    # ── Data fetching ──────────────────────────────────────────────────────────
    def _fetch_bars(
        self, symbol: str, timeframe: Any, lookback_days: int
    ) -> pd.DataFrame:
        if self._data_client is None:
            return pd.DataFrame()
        try:
            feed = DataFeed.IEX if APCA_DATA_FEED == "iex" else DataFeed.SIP
            start = datetime.now(ET) - timedelta(days=lookback_days)
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                feed=feed,
            )
            df = self._data_client.get_stock_bars(req).df
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level=0) if symbol in df.index.get_level_values(0) else pd.DataFrame()
            if df.empty:
                return pd.DataFrame()
            df = df.copy()
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(ET)
            df = df.sort_index()
            return df
        except Exception as exc:
            print(f"[scalp-layer] bar fetch failed {symbol}: {exc}")
            _write_event("scalp_bar_fetch_failed", {"symbol": symbol, "error": str(exc)[:200]})
            return pd.DataFrame()

    def _today_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        today_date = _now_et().date()
        mask = df.index.date == today_date
        return df[mask].copy()

    # ── HTF context ───────────────────────────────────────────────────────────
    def _htf_bullish(self, symbol: str) -> bool:
        """Return True if both 5m and 15m show EMA-bullish context."""
        results = []
        checks = []
        if CFG.require_5m_ema_bullish:
            checks.append((TimeFrame.Minute * 5, "5m"))
        if CFG.require_15m_ema_bullish:
            checks.append((TimeFrame.Minute * 15, "15m"))
        for tf, label in checks:
            df = self._fetch_bars(symbol, tf, CFG.bar_lookback_days)
            if df.empty or len(df) < CFG.htf_ema_period + 2:
                print(f"[scalp-layer] {symbol} {label} bars insufficient — HTF check skipped")
                results.append(True)  # degraded: allow if data unavailable
                continue
            ema = _ema(df["close"], CFG.htf_ema_period)
            last_close = float(df["close"].iloc[-1])
            last_ema = float(ema.iloc[-1])
            bullish = last_close > last_ema
            results.append(bullish)
        return all(results)

    # ── Opening range ─────────────────────────────────────────────────────────
    def _detect_opening_range(
        self, bars_5m: pd.DataFrame, st: SymbolState
    ) -> bool:
        """
        Set opening range from the 9:30 ET 5m bar.
        Returns True if range was established (or already set).
        """
        if st.or_bar_time is not None:
            return True
        today = self._today_bars(bars_5m)
        if today.empty:
            return False
        # First bar of the session is 9:30 ET
        open_bar = today.iloc[0]
        bar_time = today.index[0]
        bar_hhmm = _hhmm(bar_time)
        # Accept the 9:30 bar only (930-934)
        if bar_hhmm < 930 or bar_hhmm > 934:
            return False
        st.or_high = float(open_bar["high"])
        st.or_low = float(open_bar["low"])
        st.or_bar_time = bar_time.to_pydatetime()
        print(
            f"[scalp-layer] {' '.join(['range'])} established "
            f"or_low={st.or_low:.2f} or_high={st.or_high:.2f}"
        )
        return True

    # ── Trading window check ──────────────────────────────────────────────────
    def _in_trading_window(self) -> bool:
        hhmm = _hhmm(_now_et())
        morning = CFG.morning_start_hhmm <= hhmm <= CFG.morning_end_hhmm
        power = CFG.power_hour_start_hhmm <= hhmm <= CFG.power_hour_end_hhmm
        return morning or power

    # ── Daily counter ─────────────────────────────────────────────────────────
    def _reset_daily_if_needed(self, st: SymbolState) -> None:
        today_str = _now_et().strftime("%Y%m%d")
        if st.last_trade_date != today_str:
            st.trades_today = 0
            st.last_trade_date = today_str

    def _reset_total_if_needed(self) -> None:
        today_str = _now_et().strftime("%Y%m%d")
        if self._last_total_date != today_str:
            self._daily_total_trades = 0
            self._last_total_date = today_str

    # ── Main per-bar state machine ────────────────────────────────────────────
    def _process_bar(
        self,
        symbol: str,
        bar: pd.Series,
        bar_time: datetime,
        bar_idx: int,
        bars_1m_today: pd.DataFrame,
        st: SymbolState,
    ) -> bool:
        """
        Process one 1m bar through the state machine.
        Returns True if an order was submitted this bar.
        """
        o = float(bar["open"])
        h = float(bar["high"])
        lo = float(bar["low"])
        c = float(bar["close"])

        if st.state == ScalpState.RANGE_DEFINED:
            # Check for sweep: close below OR low
            if c < st.or_low:
                st.state = ScalpState.SWEEP_DETECTED
                st.sweep_wick_low = lo  # full wick low
                st.sweep_bar_time = bar_time
                st.bars_since_state_entry = 0
                print(f"[scalp-layer] {symbol} sweep detected c={c:.2f} or_low={st.or_low:.2f} wick={lo:.2f}")
                _write_event("scalp_sweep_detected", {"symbol": symbol, "close": c, "or_low": st.or_low, "sweep_wick_low": lo})

        elif st.state == ScalpState.SWEEP_DETECTED:
            # Check for reclaim: close back above OR low
            reclaim_threshold = st.or_low * (1 + CFG.reclaim_buffer_pct)
            if c >= reclaim_threshold:
                st.state = ScalpState.RECLAIM_DETECTED
                st.reclaim_bar_time = bar_time
                st.bars_since_state_entry = 0
                print(f"[scalp-layer] {symbol} reclaim c={c:.2f} threshold={reclaim_threshold:.2f}")
                _write_event("scalp_reclaim_detected", {"symbol": symbol, "close": c, "or_low": st.or_low})

        elif st.state == ScalpState.RECLAIM_DETECTED:
            st.bars_since_state_entry += 1
            # Timeout: no retest found
            if st.bars_since_state_entry > CFG.max_bars_to_work:
                print(f"[scalp-layer] {symbol} reclaim timeout after {st.bars_since_state_entry} bars — reset to RANGE_DEFINED")
                st.state = ScalpState.RANGE_DEFINED
                st.bars_since_state_entry = 0
                return False
            # Retest: bearish candle (close < open), stays above OR low minus tolerance
            floor = st.or_low * (1 - CFG.retest_or_buffer_pct)
            is_bearish = c < o
            stays_above_floor = lo >= floor
            if is_bearish and stays_above_floor:
                st.state = ScalpState.RETEST_FORMED
                st.retest_open = o
                st.retest_close = c
                st.retest_high = h
                st.retest_low = lo
                st.retest_bar_time = bar_time
                st.bars_since_state_entry = 0
                print(f"[scalp-layer] {symbol} retest formed o={o:.2f} c={c:.2f}")
                _write_event("scalp_retest_formed", {"symbol": symbol, "retest_open": o, "retest_close": c})

        elif st.state == ScalpState.RETEST_FORMED:
            st.bars_since_state_entry += 1
            # Timeout
            if st.bars_since_state_entry > CFG.max_bars_to_work:
                print(f"[scalp-layer] {symbol} retest timeout after {st.bars_since_state_entry} bars — reset to RANGE_DEFINED")
                st.state = ScalpState.RANGE_DEFINED
                st.bars_since_state_entry = 0
                return False
            # Bullish engulf: close > retest open and current close > current open
            engulf_threshold = st.retest_open * (1 + CFG.engulf_body_buffer_pct)
            is_bullish = c > o
            engulfs = c >= engulf_threshold
            # Volume expansion
            rvol = _compute_rvol(bars_1m_today, bar_idx, CFG.rvol_lookback_bars)
            vol_ok = rvol >= CFG.min_volume_rvol
            if is_bullish and engulfs and vol_ok:
                print(
                    f"[scalp-layer] {symbol} engulf confirmed c={c:.2f} "
                    f"retest_open={st.retest_open:.2f} rvol={rvol:.2f}"
                )
                _write_event("scalp_engulf_confirmed", {
                    "symbol": symbol,
                    "close": c,
                    "retest_open": st.retest_open,
                    "rvol": rvol,
                })
                return self._submit_bracket(symbol, c, st)

        return False

    # ── Bracket order submission ──────────────────────────────────────────────
    def _submit_bracket(self, symbol: str, entry_price: float, st: SymbolState) -> bool:
        tick = get_tick_size(entry_price)
        stop_price = round(st.sweep_wick_low - tick, 6)
        risk = entry_price - stop_price

        if risk <= 0 or (risk / entry_price) < CFG.min_stop_distance_pct:
            print(f"[scalp-layer] {symbol} stop distance too small risk={risk:.4f} — skip")
            _write_event("scalp_entry_skipped", {"symbol": symbol, "reason": "stop_too_small", "risk": risk})
            st.state = ScalpState.RANGE_DEFINED
            return False

        target_price = round(entry_price + CFG.target_rr * risk, 6)
        notional = min(CFG.max_order_dollars, max(CFG.min_order_dollars, CFG.max_order_dollars))
        qty = max(1, int(notional / entry_price))

        st.entry_price = entry_price
        st.stop_price = stop_price
        st.target_price = target_price

        order_tag = f"scalp_{symbol}_{DATA_TAG}"
        payload = {
            "symbol": symbol,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "risk_per_share": risk,
            "rr": CFG.target_rr,
            "qty": qty,
            "notional": qty * entry_price,
            "sweep_wick_low": st.sweep_wick_low,
            "or_low": st.or_low,
            "or_high": st.or_high,
            "submitted_at_utc": _now_utc_iso(),
            "paper": PAPER,
        }

        if not EXECUTE_ORDERS:
            print(
                f"[scalp-layer] signal_only {symbol} qty={qty} "
                f"entry={entry_price:.2f} stop={stop_price:.2f} target={target_price:.2f}"
            )
            _write_event("scalp_signal_only", payload)
            _append_order_row(payload)
            st.state = ScalpState.ENTRY_SUBMITTED
            st.trades_today += 1
            self._daily_total_trades += 1
            self._enter_cooldown(st)
            return True

        if self._trading_client is None:
            print(f"[scalp-layer] {symbol} no trading client — skip")
            st.state = ScalpState.RANGE_DEFINED
            return False

        if not _BRACKET_AVAILABLE:
            print(f"[scalp-layer] bracket orders not available in alpaca-py version — submitting plain market order")
            try:
                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY,
                    client_order_id=order_tag[:48],
                )
                order = self._trading_client.submit_order(order_data=req)
                payload["order_id"] = getattr(order, "id", "")
                payload["bracket_attached"] = False
                _write_event("scalp_order_submitted", payload)
                _append_order_row(payload)
                print(f"[scalp-layer] submitted plain market {symbol} qty={qty}")
            except Exception as exc:
                print(f"[scalp-layer] order submit failed {symbol}: {exc}")
                _write_event("scalp_order_failed", {"symbol": symbol, "error": str(exc)[:300]})
                st.state = ScalpState.RANGE_DEFINED
                return False
        else:
            try:
                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY,
                    client_order_id=order_tag[:48],
                    order_class=OrderClass.BRACKET,
                    stop_loss=StopLossRequest(stop_price=round(stop_price, 2)),
                    take_profit=TakeProfitRequest(limit_price=round(target_price, 2)),
                )
                order = self._trading_client.submit_order(order_data=req)
                payload["order_id"] = getattr(order, "id", "")
                payload["bracket_attached"] = True
                _write_event("scalp_order_submitted", payload)
                _append_order_row(payload)
                print(
                    f"[scalp-layer] submitted bracket {symbol} qty={qty} "
                    f"stop={stop_price:.2f} target={target_price:.2f}"
                )
            except Exception as exc:
                print(f"[scalp-layer] bracket order failed {symbol}: {exc}")
                _write_event("scalp_order_failed", {"symbol": symbol, "error": str(exc)[:300]})
                st.state = ScalpState.RANGE_DEFINED
                return False

        st.state = ScalpState.ENTRY_SUBMITTED
        st.trades_today += 1
        self._daily_total_trades += 1
        self._enter_cooldown(st)
        return True

    def _enter_cooldown(self, st: SymbolState) -> None:
        st.cooldown_until = _now_et() + timedelta(minutes=CFG.cooldown_minutes)
        st.state = ScalpState.COOLDOWN

    # ── Per-symbol cycle ──────────────────────────────────────────────────────
    def cycle_symbol(self, symbol: str) -> None:
        st = self._get_state(symbol)

        # Daily reset
        self._reset_daily_if_needed(st)
        self._reset_total_if_needed()

        # Cooldown check
        if st.state == ScalpState.COOLDOWN:
            if st.cooldown_until and _now_et() < st.cooldown_until:
                return
            # Cooldown expired — allow new sweep cycle on same opening range
            st.state = ScalpState.RANGE_DEFINED
            st.bars_since_state_entry = 0

        # Per-symbol trade limit
        if st.trades_today >= CFG.max_trades_per_symbol_per_day:
            return

        # Total daily trade limit
        if self._daily_total_trades >= CFG.max_trades_per_day_total:
            return

        # Trading window check
        if not self._in_trading_window():
            return

        # Fetch bars
        bars_1m = self._fetch_bars(symbol, TimeFrame.Minute, CFG.bar_lookback_days)
        bars_5m = self._fetch_bars(symbol, TimeFrame.Minute * 5, CFG.bar_lookback_days)

        bars_1m_today = self._today_bars(bars_1m)
        bars_5m_today = self._today_bars(bars_5m)

        if bars_1m_today.empty or bars_5m_today.empty:
            return

        # Establish opening range (IDLE → RANGE_DEFINED)
        if st.state == ScalpState.IDLE:
            if not self._detect_opening_range(bars_5m_today, st):
                return
            # HTF filter on first setup
            if not self._htf_bullish(symbol):
                print(f"[scalp-layer] {symbol} HTF not bullish — skip today's setup")
                _write_event("scalp_htf_filter_rejected", {"symbol": symbol})
                return
        elif st.state == ScalpState.RANGE_DEFINED and st.or_bar_time is None:
            if not self._detect_opening_range(bars_5m_today, st):
                return

        if st.state == ScalpState.IDLE:
            return  # range not yet available

        # Process unprocessed 1m bars
        last_processed = st.last_processed_bar_time
        for i, (bar_time, bar) in enumerate(bars_1m_today.iterrows()):
            bar_dt = bar_time.to_pydatetime() if hasattr(bar_time, "to_pydatetime") else bar_time
            if last_processed is not None and bar_dt <= last_processed:
                continue
            st.last_processed_bar_time = bar_dt
            if self._process_bar(symbol, bar, bar_dt, i, bars_1m_today, st):
                break  # one order per cycle

    # ── Full engine cycle ──────────────────────────────────────────────────────
    def run_once(self, symbols: list[str]) -> None:
        for sym in symbols:
            try:
                self.cycle_symbol(sym)
            except Exception as exc:
                print(f"[scalp-layer] {sym} cycle error: {exc}")
                _write_event("scalp_cycle_error", {"symbol": sym, "error": str(exc)[:300]})


# ── Module-level engine (singleton per process) ────────────────────────────────
_ENGINE: ScalpingEngine | None = None


def _get_engine() -> ScalpingEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ScalpingEngine()
    return _ENGINE


def run_once(symbols: list[str] | None = None) -> None:
    """
    Entry point called from run.py background thread or standalone.
    If symbols is None, resolves from env / premarket plan.
    """
    resolved = _resolve_symbols(symbols)
    if not resolved:
        return
    _get_engine().run_once(resolved)


# ======================================================================
# Standalone entry point
# ======================================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="1m scalping execution layer")
    p.add_argument("--symbols", default="", help="Comma-separated symbol list (overrides env/plan)")
    p.add_argument("--interval", type=int, default=10, help="Poll interval seconds (default 10)")
    p.add_argument("--once", action="store_true", help="Run one cycle and exit")
    return p.parse_args()


def _main() -> None:
    args = _parse_args()
    sym_list = [_safe_symbol(s) for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    print(
        f"[scalp-layer] starting standalone "
        f"execute={EXECUTE_ORDERS} paper={PAPER} "
        f"interval={args.interval}s"
    )
    if args.once:
        run_once(sym_list)
        return
    while True:
        try:
            run_once(sym_list)
        except KeyboardInterrupt:
            print("[scalp-layer] stopped by user")
            break
        except Exception as exc:
            print(f"[scalp-layer] unhandled error: {exc}")
            _write_event("scalp_unhandled_error", {"error": str(exc)[:500]})
        time.sleep(args.interval)


if __name__ == "__main__":
    _main()
