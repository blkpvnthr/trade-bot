#!/usr/bin/env python3
"""
amps_v2.py -- AMPS V2 (Adaptive Micro-Profit Scalping), unified strategy engine.

ONE engine, ONE config, no paper/live split. It consumes normalized market features (computed
from completed candles + order-flow snapshots) and emits execution intents; the execution
destination (paper/live/backtest) is an infrastructure concern handled elsewhere.

This module is PURE and testable (no I/O, no broker). It provides:
  - entry-quality scoring (8 weighted components, [0,100]),
  - Bollinger-middle APPROACH detection (approaching-from-below / above / combined),
  - four independent entry setups,
  - mandatory entry gates + an economic (expected-net-return) filter,
  - an adaptive exit state machine (initial stop -> breakeven -> min-profit-lock -> trailing ->
    target -> momentum-deterioration -> max-hold -> emergency), monotonic (stops never widen),
  - a state-aware inverse-momentum exit,
  - execution-aware cost-adjusted breakeven / net-profit math.

HONESTY: two of the eight entry-score components (CVD acceleration, aggressive-buying pressure)
and several exit signals are ORDER-FLOW features that require genuine ticks. They are passed in
as inputs (real ticks live; neutral + flagged in bar-only backtests). This module never
fabricates order flow from OHLC. L1 quote imbalance is never represented as full-depth.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, time
from typing import Any, Optional

try:
    from costs import BASELINE_COSTS, TransactionCostConfig
except Exception:  # pragma: no cover
    BASELINE_COSTS = None

STRATEGY_ID = "AMPS_V2"

# exit reasons / states
INITIAL_STOP = "INITIAL_PROTECTIVE_STOP"
BREAKEVEN_STOP = "BREAKEVEN_STOP"
MIN_PROFIT_LOCK = "MIN_PROFIT_LOCK"
TRAILING_STOP = "TRAILING_STOP"
PROFIT_TARGET = "PROFIT_TARGET"
MOMENTUM_EXIT = "MOMENTUM_DETERIORATION_EXIT"
MAX_HOLD_EXIT = "MAX_HOLDING_PERIOD_EXIT"
EMERGENCY_EXIT = "EMERGENCY_RISK_EXIT"
EOD_EXIT = "END_OF_SESSION_LIQUIDATION"

# BB entry states
BB_APPROACHING = "APPROACHING_FROM_BELOW"
BB_ABOVE = "CONFIRMED_ABOVE"
BB_FAILED = "FAILED_CROSSOVER"
BB_NONE = "NONE"

# entry setups
SETUP_IGNITION = "MOMENTUM_IGNITION"
SETUP_PULLBACK = "TREND_PULLBACK"
SETUP_COMPRESSION = "COMPRESSION_BREAKOUT"
SETUP_OF_DIVERGENCE = "ORDER_FLOW_DIVERGENCE"
SETUP_NONE = "NONE"


@dataclass
class AmpsV2Config:
    enabled: bool = False
    strategy_id: str = STRATEGY_ID
    # mandatory gates
    min_adx: float = 20.0
    require_adx_rising: bool = True
    adx_over_di_filter: bool = False        # optional ADX > DI+ filter (5.1)
    max_spread_bps: float = 5.0
    # BB approach (5.2)
    bb_entry_policy: str = "combined"       # "confirmed" | "approaching" | "combined"
    bb_approach_distance_mode: str = "bps"  # "bps" | "atr"
    bb_approach_distance_bps: float = 10.0
    bb_approach_distance_atr_mult: float = 0.25
    # entry-quality score (5.3)
    entry_quality_weights: dict = field(default_factory=lambda: {
        "di_sep_expansion": 0.20, "adx_accel": 0.10, "bb_proximity": 0.10, "vwap_align": 0.10,
        "cvd_accel": 0.15, "buying_pressure": 0.15, "rel_volume": 0.10, "cost_efficiency": 0.10})
    min_entry_quality: float = 70.0
    # economic filter (11.3)
    min_expected_net_bps: float = 2.0
    # exits (9.1) -- all in bps unless noted
    initial_stop_bps: float = 10.0
    breakeven_activation_bps: float = 10.0
    min_profit_lock_activation_bps: float = 15.0
    min_locked_profit_bps: float = 5.0
    trailing_activation_bps: float = 20.0
    trailing_giveback_bps: float = 7.0
    final_target_bps: float = 30.0
    max_hold_minutes: float = 5.0
    bearish_confirmation_count: int = 2
    # costs
    est_round_trip_bps: float = 6.0         # entry+exit modeled cost (spread+slip+fee), bps
    max_positions: int = 5

    @classmethod
    def from_dict(cls, d: dict) -> "AmpsV2Config":
        f = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in f})

    def as_dict(self) -> dict:
        return asdict(self)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# ======================================================================
# Bollinger-middle approach detection (5.2)
# ======================================================================
def bb_distance(price: float, bb_mid: float) -> float:
    """Normalized distance (price - mid)/mid. >0 above, <0 below."""
    if bb_mid is None or bb_mid <= 0 or price is None or price <= 0:
        return float("nan")
    return (price - bb_mid) / bb_mid


def bb_approach_state(price: float, bb_mid: float, dist_now: float, dist_prev: float,
                      short_momentum: float, cfg: AmpsV2Config, atr: Optional[float] = None) -> str:
    """Classify the Bollinger-middle entry state. Approaching requires: below-but-within the
    configured distance, distance DECREASING, and positive short momentum (a stationary or
    declining price below the band is NOT 'approaching')."""
    if dist_now != dist_now:
        return BB_NONE
    if price > bb_mid:
        return BB_ABOVE
    # below the band:
    if cfg.bb_approach_distance_mode == "atr" and atr and atr > 0 and bb_mid > 0:
        thresh = (cfg.bb_approach_distance_atr_mult * atr) / bb_mid
    else:
        thresh = cfg.bb_approach_distance_bps / 1e4
    within = abs(dist_now) <= thresh
    decreasing = (dist_prev == dist_prev) and (abs(dist_now) < abs(dist_prev))  # closing the gap
    if within and decreasing and short_momentum > 0:
        return BB_APPROACHING
    return BB_NONE


def bb_policy_ok(state: str, cfg: AmpsV2Config) -> bool:
    if cfg.bb_entry_policy == "confirmed":
        return state == BB_ABOVE
    if cfg.bb_entry_policy == "approaching":
        return state == BB_APPROACHING
    return state in (BB_ABOVE, BB_APPROACHING)  # combined


# ======================================================================
# Entry-quality score (5.3). Order-flow components (cvd_accel, buying_pressure) may be absent
# in bar-only research -> pass None and the score renormalizes over available components.
# ======================================================================
def entry_quality_score(cfg: AmpsV2Config, di_plus: float, di_minus: float, di_sep_prev: float,
                        adx: float, adx_prev: float, adx_prev2: float, bb_dist: float, bb_slope: float,
                        vwap_dist: float, cvd_accel: Optional[float], buying_pressure: Optional[float],
                        rel_volume: float, expected_move_bps: float, cost_bps: float) -> tuple[float, dict]:
    comp: dict[str, Optional[float]] = {}
    di_sep = di_plus - di_minus
    comp["di_sep_expansion"] = _clamp(0.5 * _clamp(di_sep / 25.0) + 0.5 * (1.0 if di_sep > di_sep_prev else 0.0))
    adx_accel = (adx - adx_prev) - (adx_prev - adx_prev2)
    comp["adx_accel"] = _clamp(0.5 * (1.0 if adx > adx_prev else 0.0) + 0.3 * (1.0 if adx_accel > 0 else 0.0) + 0.2 * _clamp((adx - 20.0) / 20.0))
    # BB proximity: best when just above or closely approaching; slope up helps
    prox = 1.0 - _clamp(abs(bb_dist) / 0.002) if bb_dist == bb_dist else 0.0
    comp["bb_proximity"] = _clamp(0.6 * prox + 0.2 * (1.0 if bb_dist >= 0 else 0.0) + 0.2 * (1.0 if bb_slope > 0 else 0.0))
    comp["vwap_align"] = _clamp(0.6 * (1.0 if vwap_dist > 0 else 0.0) + 0.4 * _clamp(vwap_dist / 0.002))
    comp["cvd_accel"] = None if cvd_accel is None else _clamp(cvd_accel)
    comp["buying_pressure"] = None if buying_pressure is None else _clamp(buying_pressure)
    comp["rel_volume"] = _clamp((rel_volume - 1.0) / 1.5)  # 1.0x->0, 2.5x->1
    net_edge = expected_move_bps - cost_bps
    comp["cost_efficiency"] = _clamp(net_edge / 20.0)  # +20bps net edge -> 1.0
    # weighted sum over AVAILABLE components (renormalize if order-flow absent)
    w = cfg.entry_quality_weights
    num = 0.0; den = 0.0
    for k, v in comp.items():
        if v is None:
            continue
        num += w.get(k, 0.0) * v
        den += w.get(k, 0.0)
    score = 100.0 * (num / den) if den > 0 else 0.0
    diag = {k: (round(v, 4) if v is not None else None) for k, v in comp.items()}
    diag["order_flow_available"] = comp["cvd_accel"] is not None and comp["buying_pressure"] is not None
    return round(score, 2), diag


# ======================================================================
# Four entry setups (6.x). Each returns bool; order-flow-dependent ones degrade gracefully.
# ======================================================================
def setup_momentum_ignition(di_cross_up: bool, di_sep: float, di_sep_prev: float, adx: float,
                            adx_prev: float, bb_state: str, rel_volume: float,
                            cvd_accel: Optional[float]) -> bool:
    core = ((di_cross_up or di_sep > 0) and di_sep > di_sep_prev and adx > adx_prev
            and bb_state in (BB_ABOVE, BB_APPROACHING) and rel_volume > 1.1)
    of_ok = True if cvd_accel is None else cvd_accel > 0.5
    return bool(core and of_ok)


def setup_trend_pullback(trend_up: bool, near_ref: bool, stabilized: bool, buying_resumed: bool,
                         bb_state: str) -> bool:
    return bool(trend_up and near_ref and stabilized and buying_resumed
                and bb_state in (BB_ABOVE, BB_APPROACHING))


def setup_compression_breakout(bb_width_contracting: bool, realized_vol_declining: bool,
                               breakout_above_range: bool, rel_volume: float,
                               spread_bps: float, max_spread_bps: float) -> bool:
    return bool(bb_width_contracting and realized_vol_declining and breakout_above_range
                and rel_volume > 1.2 and spread_bps <= max_spread_bps)


def setup_order_flow_divergence(cvd_divergence: Optional[float], buy_sell_ratio: Optional[float],
                                quote_imbalance: Optional[float], stabilized: bool,
                                bb_state: str) -> bool:
    if cvd_divergence is None:   # order flow unavailable -> setup not identifiable
        return False
    return bool(cvd_divergence > 0.05 and (buy_sell_ratio or 0) > 1.0
                and (quote_imbalance or 0) > 0.0 and stabilized
                and bb_state in (BB_ABOVE, BB_APPROACHING))


# ======================================================================
# Mandatory gates + economic filter (5.1, 11.3)
# ======================================================================
def mandatory_gates(cfg: AmpsV2Config, di_plus: float, di_minus: float, adx: float, adx_prev: float,
                    bb_state: str, above_vwap: bool, spread_bps: float, data_fresh: bool,
                    liquidity_ok: bool, cost_feasible: bool, risk_ok: bool) -> tuple[bool, list]:
    fails = []
    if not (di_plus > di_minus): fails.append("di_plus_le_di_minus")
    if not (adx > cfg.min_adx): fails.append("adx_below_min")
    if cfg.require_adx_rising and not (adx > adx_prev): fails.append("adx_not_rising")
    if cfg.adx_over_di_filter and not (adx > di_plus): fails.append("adx_not_over_di")
    if not bb_policy_ok(bb_state, cfg): fails.append("bb_policy")
    if not above_vwap: fails.append("below_vwap")
    if spread_bps is None or spread_bps > cfg.max_spread_bps: fails.append("spread_too_wide")
    if not data_fresh: fails.append("stale_data")
    if not liquidity_ok: fails.append("insufficient_liquidity")
    if not cost_feasible: fails.append("cost_infeasible")
    if not risk_ok: fails.append("risk_block")
    return (len(fails) == 0), fails


def expected_net_return_bps(p_win: float, avg_gross_win_bps: float, avg_gross_loss_bps: float,
                            cost_bps: float) -> float:
    """Simplified two-outcome expected net return in bps (11.3)."""
    return p_win * avg_gross_win_bps - (1.0 - p_win) * abs(avg_gross_loss_bps) - cost_bps


# ======================================================================
# Adaptive exit state machine (9). bps-based, monotonic (stops never widen). Returns the
# updated position dict + (should_exit, reason).
# ======================================================================
@dataclass
class AmpsPosition:
    symbol: str
    entry_price: float
    entry_time: str
    qty: float
    highest_price: float
    effective_stop: float          # absolute price; only moves UP
    stage: str = "INITIAL"
    peak_gross_bps: float = 0.0
    last_atr: Optional[float] = None
    bb_state_at_entry: str = BB_NONE
    bearish_count: int = 0
    exit_pending: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def cost_adjusted_breakeven(entry_price: float, cost_bps: float) -> float:
    return entry_price * (1.0 + cost_bps / 1e4)


def new_amps_position(symbol, entry_price, entry_time, qty, cfg: AmpsV2Config,
                      atr: Optional[float], bb_state: str) -> AmpsPosition:
    init_stop = entry_price * (1.0 - cfg.initial_stop_bps / 1e4)
    return AmpsPosition(symbol=symbol, entry_price=entry_price, entry_time=entry_time, qty=qty,
                        highest_price=entry_price, effective_stop=init_stop, stage="INITIAL",
                        last_atr=atr, bb_state_at_entry=bb_state)


def amps_exit_decision(pos: AmpsPosition, price: float, now_et: datetime, cfg: AmpsV2Config,
                       force_exit_time: Optional[time], hold_minutes: float,
                       bearish_signals: int = 0) -> tuple[bool, str]:
    """Advance the adaptive stop and decide exit. Monotonic: effective_stop only rises.
    Priority: emergency/EOD > target > max-hold > stop breach > momentum deterioration."""
    if price and price > 0:
        pos.highest_price = max(pos.highest_price, price)
    gross_bps = (price / pos.entry_price - 1.0) * 1e4 if pos.entry_price > 0 else 0.0
    peak_bps = (pos.highest_price / pos.entry_price - 1.0) * 1e4 if pos.entry_price > 0 else 0.0
    pos.peak_gross_bps = max(pos.peak_gross_bps, peak_bps)
    e = pos.entry_price

    # --- advance protective stop through the stages (never widen) ---
    candidate = pos.effective_stop
    # breakeven (cost-adjusted) once peak >= activation
    if pos.peak_gross_bps >= cfg.breakeven_activation_bps:
        candidate = max(candidate, cost_adjusted_breakeven(e, cfg.est_round_trip_bps))
        pos.stage = "BREAKEVEN"
    # minimum locked net profit
    if pos.peak_gross_bps >= cfg.min_profit_lock_activation_bps:
        lock_price = e * (1.0 + (cfg.min_locked_profit_bps + cfg.est_round_trip_bps) / 1e4)
        candidate = max(candidate, lock_price)
        pos.stage = "MIN_PROFIT_LOCK"
    # adaptive trailing once peak >= trailing activation
    if pos.peak_gross_bps >= cfg.trailing_activation_bps:
        trail_price = pos.highest_price * (1.0 - cfg.trailing_giveback_bps / 1e4)
        candidate = max(candidate, trail_price)
        pos.stage = "TRAILING"
    pos.effective_stop = max(pos.effective_stop, candidate)  # MONOTONIC

    # --- exit triggers (priority order: EOD > target > max-hold > stop > momentum) ---
    if force_exit_time is not None and now_et.time() >= force_exit_time:
        return True, EOD_EXIT
    if gross_bps >= cfg.final_target_bps:
        return True, PROFIT_TARGET
    if hold_minutes >= cfg.max_hold_minutes:
        return True, MAX_HOLD_EXIT
    if price and price <= pos.effective_stop:
        if pos.stage == "TRAILING":
            return True, TRAILING_STOP
        if pos.stage == "MIN_PROFIT_LOCK":
            return True, MIN_PROFIT_LOCK
        if pos.stage == "BREAKEVEN":
            return True, BREAKEVEN_STOP
        return True, INITIAL_STOP
    # momentum deterioration (requires >= bearish_confirmation_count independent signals)
    pos.bearish_count = bearish_signals
    if bearish_signals >= cfg.bearish_confirmation_count:
        return True, MOMENTUM_EXIT
    return False, ""
