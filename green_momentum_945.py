#!/usr/bin/env python3
"""
green_momentum_945.py -- GREEN_MOMENTUM_945 strategy: pure, testable core.

Baseline (V0): at/after 09:45 ET, go long symbols that are GREEN vs the previous regular-
session close AND trading above the regular-session VWAP; protect with a fixed $0.50 initial
stop + $0.50 trailing stop; one entry per symbol per session; flat by 15:55.

Optional enhancements (OFF by default, independently configurable):
  A  5-minute momentum confirmation  (latest completed 5m close > previous completed 5m close)
  B  opening-range breakout          (price > max high of 09:30..09:45, * (1+buffer))
  C  ATR trailing stop               (1x ATR(14, 5m) instead of fixed $0.50)

This module holds NO I/O and NO broker calls -- only the signal math, the monotonic trailing
stop, the exit decision, and the per-session diagnostics. trade.py (live) and
backtest_green_momentum.py (research) both import these so there is ONE source of truth.

Paper-only, disabled by default. It never submits orders itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, time
from typing import Any, Optional, Sequence

import pandas as pd

STRATEGY_ID = "GREEN_MOMENTUM_945"

# Exit reasons
INITIAL_PROTECTIVE_STOP = "INITIAL_PROTECTIVE_STOP"
FIXED_TRAILING_STOP = "FIXED_TRAILING_STOP"
ATR_TRAILING_STOP = "ATR_TRAILING_STOP"
END_OF_DAY_LIQUIDATION = "END_OF_DAY_LIQUIDATION"
RISK_LIMIT_EXIT = "RISK_LIMIT_EXIT"

# Target configuration. The DEFAULT reproduces the original Green Momentum 9:45 behavior (V0):
# green + VWAP + fixed $0.50 stops, enhancements OFF, disabled + paper-only.
GREEN_MOMENTUM_CONFIG: dict[str, Any] = {
    "enabled": False,
    "strategy_id": STRATEGY_ID,
    "timezone": "America/New_York",
    "entry_start": "09:45",
    "entry_end": "15:00",
    "force_exit": "15:55",
    "require_green_on_day": True,
    "min_daily_change_pct": 0.0,
    "require_above_vwap": True,
    "require_momentum_confirmation": False,
    "momentum_timeframe_minutes": 5,
    "require_opening_range_breakout": False,
    "opening_range_start": "09:30",
    "opening_range_end": "09:45",
    "opening_range_buffer_pct": 0.0,
    "stop_mode": "fixed",                 # "fixed" | "atr"
    "fixed_trailing_stop_dollars": 0.50,
    "atr_period": 14,
    "atr_timeframe_minutes": 5,
    "atr_multiplier": 1.0,
    "initial_stop_mode": "fixed",
    "initial_stop_dollars": 0.50,
    "allow_same_day_reentry": False,
    "max_positions": 5,
    "paper_only": True,
}


@dataclass
class GreenMomentumConfig:
    enabled: bool = False
    strategy_id: str = STRATEGY_ID
    timezone: str = "America/New_York"
    entry_start: str = "09:45"
    entry_end: str = "15:00"
    force_exit: str = "15:55"
    require_green_on_day: bool = True
    min_daily_change_pct: float = 0.0
    require_above_vwap: bool = True
    require_momentum_confirmation: bool = False
    momentum_timeframe_minutes: int = 5
    require_opening_range_breakout: bool = False
    opening_range_start: str = "09:30"
    opening_range_end: str = "09:45"
    opening_range_buffer_pct: float = 0.0
    stop_mode: str = "fixed"
    fixed_trailing_stop_dollars: float = 0.50
    atr_period: int = 14
    atr_timeframe_minutes: int = 5
    atr_multiplier: float = 1.0
    initial_stop_mode: str = "fixed"
    initial_stop_dollars: float = 0.50
    allow_same_day_reentry: bool = False
    max_positions: int = 5
    paper_only: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GreenMomentumConfig":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_time(s: str) -> time:
    hh, mm = str(s).split(":")[:2]
    return time(int(hh), int(mm))


# ======================================================================
# Pure signal functions
# ======================================================================
def daily_change_pct(current_price: float, previous_close: float) -> float:
    if previous_close is None or previous_close <= 0 or current_price is None or current_price <= 0:
        return float("nan")
    return (current_price / previous_close - 1.0) * 100.0


def green_on_day(current_price: float, previous_close: float, min_change_pct: float = 0.0) -> bool:
    chg = daily_change_pct(current_price, previous_close)
    if chg != chg:  # NaN -> invalid previous-close reference
        return False
    return chg > 0.0 and chg >= min_change_pct


def within_entry_window(now_et: datetime, entry_start: str, entry_end: str) -> bool:
    t = now_et.time()
    return _parse_time(entry_start) <= t < _parse_time(entry_end)


def past_force_exit(now_et: datetime, force_exit: str) -> bool:
    return now_et.time() >= _parse_time(force_exit)


def above_vwap(price: float, vwap: float) -> bool:
    if vwap is None or vwap != vwap or vwap <= 0 or price is None or price <= 0:
        return False
    return price > vwap


def momentum_confirmed(latest_completed_5m_close: float, previous_completed_5m_close: float) -> bool:
    """Strengthening momentum: last completed 5m candle closed above the preceding one.
    Requires both completed closes; never uses an incomplete/future candle."""
    if latest_completed_5m_close is None or previous_completed_5m_close is None:
        return False
    if latest_completed_5m_close != latest_completed_5m_close or previous_completed_5m_close != previous_completed_5m_close:
        return False
    return latest_completed_5m_close > previous_completed_5m_close


def opening_range_high(bars: pd.DataFrame, or_start: str, or_end: str) -> float:
    """Max high over [or_start, or_end) of the session. Excludes the 09:45 bar. NaN if no
    valid bars in the range (reject incomplete/invalid opening ranges)."""
    if bars is None or bars.empty or "et_time" not in bars.columns:
        return float("nan")
    lo, hi = _parse_time(or_start), _parse_time(or_end)
    rng = bars[(bars["et_time"] >= lo) & (bars["et_time"] < hi)]
    if rng.empty:
        return float("nan")
    return float(rng["high"].astype(float).max())


def opening_range_breakout(price: float, or_high: float, buffer_pct: float = 0.0) -> bool:
    if or_high is None or or_high != or_high or or_high <= 0 or price is None or price <= 0:
        return False
    return price > or_high * (1.0 + buffer_pct / 100.0)


def atr_from_bars(bars: pd.DataFrame, period: int = 14) -> float:
    """Wilder ATR on completed bars; NaN if too few bars (handle missing ATR explicitly)."""
    if bars is None or len(bars) < period + 1:
        return float("nan")
    h = bars["high"].astype(float); l = bars["low"].astype(float); c = bars["close"].astype(float)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    v = float(atr.iloc[-1])
    return v if v == v and v > 0 else float("nan")


def trailing_distance(cfg: GreenMomentumConfig, atr_value: Optional[float]) -> Optional[float]:
    """Trailing distance in $: fixed, or atr_multiplier*ATR. None if ATR mode but ATR invalid
    (never loosen/guess a stop on missing volatility)."""
    if cfg.stop_mode == "atr":
        if atr_value is None or atr_value != atr_value or atr_value <= 0:
            return None
        return atr_value * cfg.atr_multiplier
    return cfg.fixed_trailing_stop_dollars


def initial_stop_price(entry_fill_price: float, initial_stop_dollars: float) -> float:
    return entry_fill_price - initial_stop_dollars


def update_trailing_stop(highest_price: float, current_price: float, trail_dist: Optional[float],
                         previous_effective_stop: float, initial_stop: float) -> tuple[float, float]:
    """Monotonic trailing stop. Returns (new_highest, new_effective_stop). The stop NEVER moves
    down: effective = max(previous_effective, initial, high - trail). A None/invalid trail
    distance leaves the stop unchanged (missing ATR must not loosen the stop)."""
    new_high = max(highest_price, current_price)
    if trail_dist is None or trail_dist != trail_dist or trail_dist <= 0:
        return new_high, max(previous_effective_stop, initial_stop)
    candidate = new_high - trail_dist
    return new_high, max(previous_effective_stop, initial_stop, candidate)


# ======================================================================
# Position state
# ======================================================================
@dataclass
class GMPosition:
    strategy_id: str
    symbol: str
    entry_price: float
    entry_time: str
    filled_quantity: float
    highest_price: float
    initial_stop: float
    effective_stop: float
    stop_mode: str
    last_atr: Optional[float] = None
    exit_pending: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_position(symbol: str, entry_fill_price: float, entry_time: str, qty: float,
                 cfg: GreenMomentumConfig, atr_value: Optional[float]) -> GMPosition:
    init_stop = initial_stop_price(entry_fill_price, cfg.initial_stop_dollars)
    return GMPosition(strategy_id=STRATEGY_ID, symbol=symbol, entry_price=entry_fill_price,
                      entry_time=entry_time, filled_quantity=qty,
                      highest_price=entry_fill_price, initial_stop=init_stop,
                      effective_stop=init_stop, stop_mode=cfg.stop_mode, last_atr=atr_value)


def exit_decision(pos: GMPosition, current_price: float, now_et: datetime,
                  cfg: GreenMomentumConfig, atr_value: Optional[float]) -> tuple[bool, str, GMPosition]:
    """Advance the trailing stop and decide whether to exit. Returns (should_exit, reason, pos).
    Priority: end-of-day liquidation > stop breach. Stops never move down."""
    # Advance high-water + trailing stop on a valid observation.
    if current_price is not None and current_price > 0:
        trail = trailing_distance(cfg, atr_value if atr_value is not None else pos.last_atr)
        pos.highest_price, pos.effective_stop = update_trailing_stop(
            pos.highest_price, current_price, trail, pos.effective_stop, pos.initial_stop)
        if atr_value is not None and atr_value == atr_value:
            pos.last_atr = atr_value

    if past_force_exit(now_et, cfg.force_exit):
        return True, END_OF_DAY_LIQUIDATION, pos

    if current_price is not None and current_price > 0 and current_price <= pos.effective_stop:
        # Which stop fired? Initial if the trail never advanced above the initial level.
        if pos.effective_stop <= pos.initial_stop + 1e-9:
            reason = INITIAL_PROTECTIVE_STOP
        else:
            reason = ATR_TRAILING_STOP if cfg.stop_mode == "atr" else FIXED_TRAILING_STOP
        return True, reason, pos
    return False, "", pos


# ======================================================================
# Entry evaluation (one symbol)
# ======================================================================
def evaluate_entry(cfg: GreenMomentumConfig, now_et: datetime, price: float, previous_close: float,
                   vwap: float, latest_5m_close: Optional[float], prev_5m_close: Optional[float],
                   or_high: Optional[float], already_positioned: bool, pending_entry: bool,
                   entered_today: bool, risk_ok: bool, data_valid: bool) -> dict[str, Any]:
    """Return a diagnostics dict incl. 'eligible' bool and each condition's result. Missing/stale
    data or an invalid previous-close reference -> not eligible (never a default pass)."""
    diag: dict[str, Any] = {"symbol": None}
    win = within_entry_window(now_et, cfg.entry_start, cfg.entry_end)
    green = (green_on_day(price, previous_close, cfg.min_daily_change_pct)
             if cfg.require_green_on_day else True)
    vwap_ok = above_vwap(price, vwap) if cfg.require_above_vwap else True
    mom_ok = True
    if cfg.require_momentum_confirmation:
        mom_ok = momentum_confirmed(latest_5m_close, prev_5m_close)
    orb_ok = True
    if cfg.require_opening_range_breakout:
        orb_ok = opening_range_breakout(price, or_high, cfg.opening_range_buffer_pct)
    base = (win and green and vwap_ok and data_valid and not already_positioned
            and not pending_entry and not entered_today and risk_ok)
    eligible = bool(base and mom_ok and orb_ok)
    diag.update({
        "within_window": win, "green_on_day": green, "above_vwap": vwap_ok,
        "momentum_confirmed": mom_ok, "opening_range_breakout": orb_ok,
        "data_valid": data_valid, "already_positioned": already_positioned,
        "pending_entry": pending_entry, "entered_today": entered_today, "risk_ok": risk_ok,
        "daily_change_pct": daily_change_pct(price, previous_close),
        "eligible": eligible,
    })
    return diag
