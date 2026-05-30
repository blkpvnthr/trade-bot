#!/usr/bin/env python3
"""
alpha_call_engine.py

Primary profit engine: long calls only on A/A+ tactical setups.

Supported setups:
    ORB          — Opening Range Breakout above premarket/first-candle high
    VWAP_RECLAIM — Price reclaims VWAP after being below with confirming volume
    PIVOT_BREAK  — Price breaks through validated resistance with conviction
    ACCUM        — Institutional accumulation pattern (high RVOL + bullish structure)

Contract selection: 21–60 DTE, delta 0.45–0.70, liquid, tight spread.

Position management:
    Stop:    –20% to –35% from entry
    Partial: +30% to +50% (take half)
    Trail:   remaining contracts

Outputs:
    daily_system_outputs/options_alpha_signals.json
    system-data/{DATE}/{DATE}_alpha_engine_state.json
"""
from __future__ import annotations

import json
import os
import math
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import alpha_probability_model as apm

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
DATA_TAG = os.getenv("RUN_DATA_DATE", datetime.now(ET).strftime("%Y%m%d"))
DAILY_OUTPUTS = PROJECT_ROOT / "daily_system_outputs"
SYSTEM_DATA = PROJECT_ROOT / "system-data" / DATA_TAG
DAILY_OUTPUTS.mkdir(parents=True, exist_ok=True)
SYSTEM_DATA.mkdir(parents=True, exist_ok=True)

ALPHA_SIGNALS_PATH = DAILY_OUTPUTS / "options_alpha_signals.json"
ALPHA_OPTIONS_UNIVERSE_PATH = DAILY_OUTPUTS / "alpha_options_universe.json"
ALPHA_VOL_DIAGNOSTIC_PATH = DAILY_OUTPUTS / "alpha_volatility_diagnostic.json"
ALPHA_STATE_PATH = SYSTEM_DATA / f"{DATA_TAG}_alpha_engine_state.json"
ALPHA_EVENTS_PATH = PROJECT_ROOT / "logs" / "execution" / DATA_TAG / f"{DATA_TAG}_alpha_engine_events.jsonl"
ALPHA_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── configuration ─────────────────────────────────────────────────────────────
@dataclass
class AlphaCallConfig:
    enabled: bool = os.getenv("ENABLE_ALPHA_CALL_ENGINE", "true").lower() in {"1", "true"}
    # Regime requirement
    require_bullish_regime: bool = True
    allowed_regimes: tuple = ("bullish", "risk_on", "uptrend", "constructive", "trend_risk_on", "neutral")
    # Setup thresholds
    min_setup_score: float = float(os.getenv("ALPHA_MIN_SETUP_SCORE", "70"))   # A = 70-84, A+ = 85+
    # Risk per trade
    risk_per_trade_pct: float = float(os.getenv("OPTIONS_RISK_PER_TRADE_PCT", "0.015"))
    max_options_exposure_pct: float = float(os.getenv("MAX_OPTIONS_EXPOSURE_PCT", "0.15"))
    # Contract selection
    min_dte: int = int(os.getenv("ALPHA_MIN_DTE", "21"))
    max_dte: int = int(os.getenv("ALPHA_MAX_DTE", "60"))
    min_delta: float = float(os.getenv("ALPHA_MIN_DELTA", "0.45"))
    max_delta: float = float(os.getenv("ALPHA_MAX_DELTA", "0.70"))
    min_open_interest: int = int(os.getenv("ALPHA_MIN_OI", "200"))
    min_volume: int = int(os.getenv("ALPHA_MIN_VOLUME", "10"))
    max_spread_pct: float = float(os.getenv("ALPHA_MAX_SPREAD_PCT", "0.05"))
    # Position management
    # Risk/reward structure: 3R target with trailing stop that activates at 1.7R.
    # R = initial risk = entry premium (max loss on long call).
    # When the position reaches +1.7R profit, the stop trails up to lock in gains.
    # At +3R the remaining position is fully exited.
    # OPTIONS_TP_R_MULTIPLE / OPTIONS_SL_PCT are honored as aliases of the existing
    # ALPHA_* env names so management config can use either; defaults are unchanged.
    r_target: float = float(os.getenv("OPTIONS_TP_R_MULTIPLE", os.getenv("ALPHA_R_TARGET", "3.0")))  # exit all at 3R
    r_trail_activate: float = float(os.getenv("ALPHA_R_TRAIL_ACTIVATE", "1.7"))  # activate trailing stop at 1.7R
    r_trail_offset: float = float(os.getenv("ALPHA_R_TRAIL_OFFSET", "0.8"))      # trail 0.8R behind peak
    stop_loss_pct: float = float(os.getenv("OPTIONS_SL_PCT", os.getenv("ALPHA_STOP_PCT", "0.28")))  # hard stop: -28% of premium
    partial_profit_pct: float = float(os.getenv("ALPHA_PARTIAL_PROFIT_PCT", "0.5"))   # +50% = take partial (sell partial_size)
    partial_size: float = float(os.getenv("ALPHA_PARTIAL_SIZE", "0.50"))   # sell 50% at +1R
    # Universe
    max_concurrent_positions: int = int(os.getenv("ALPHA_MAX_POSITIONS", "5"))
    scan_interval_seconds: int = int(os.getenv("ALPHA_SCAN_SECONDS", "60"))
    # ORB window
    orb_minutes: int = int(os.getenv("ALPHA_ORB_MINUTES", "15"))
    # Accumulation detection
    min_rvol_for_accum: float = float(os.getenv("ALPHA_MIN_RVOL_ACCUM", "1.8"))
    # ── Black-Scholes valuation layer (applied AFTER hard liquidity gates) ──────
    bs_enabled: bool = os.getenv("ENABLE_BLACK_SCHOLES", "true").lower() in {"1", "true"}
    risk_free_rate: float = float(os.getenv("OPTIONS_RISK_FREE_RATE", "0.043"))
    dividend_yield: float = float(os.getenv("OPTIONS_DIVIDEND_YIELD", "0.0"))
    bs_max_overprice_pct: float = float(os.getenv("OPTIONS_BS_MAX_OVERPRICE_PCT", "0.20"))  # reject if mkt > theo by >20%
    bs_max_theta_pct_per_day: float = float(os.getenv("OPTIONS_BS_MAX_THETA_PCT", "0.04"))  # reject if |theta|/mid > 4%/day
    bs_min_gamma_theta: float = float(os.getenv("OPTIONS_BS_MIN_GAMMA_THETA", "1.0"))       # reject if 1R convexity pays < 1 day of theta
    bs_max_iv: float = float(os.getenv("OPTIONS_BS_MAX_IV", "3.0"))                          # 300% IV ceiling
    bs_iv_momentum_rvol: float = float(os.getenv("OPTIONS_BS_IV_MOMENTUM_RVOL", "2.0"))      # RVOL that justifies high IV
    underlying_r_move_pct: float = float(os.getenv("OPTIONS_UNDERLYING_R_PCT", "0.03"))      # 1R underlying move for EV modeling
    # ── Alpha Options Screener (universe selection: momentum + options liquidity) ───
    screener_enabled: bool = os.getenv("ENABLE_ALPHA_OPTIONS_SCREENER", "true").lower() in {"1", "true"}
    screener_universe_size: int = int(os.getenv("ALPHA_OPTIONS_UNIVERSE_SIZE", "150"))
    screener_min_score: float = float(os.getenv("ALPHA_MIN_SCREENER_SCORE", "70"))
    screener_min_delta: float = float(os.getenv("ALPHA_SCREENER_MIN_DELTA", "0.35"))   # wider band for liquidity probe
    screener_max_delta: float = float(os.getenv("ALPHA_SCREENER_MAX_DELTA", "0.80"))
    screener_momentum_floor: float = float(os.getenv("ALPHA_SCREENER_MOMENTUM_FLOOR", "45"))  # watch threshold (chain-fetch budget)
    bs_exit_optimizer_enabled: bool = os.getenv("ENABLE_BS_EXIT_OPTIMIZER", "true").lower() in {"1", "true"}
    bs_exit_min_dte: int = int(os.getenv("OPTIONS_BS_EXIT_MIN_DTE", "10"))   # theta-decay exit window
    bs_exit_min_convexity: float = float(os.getenv("OPTIONS_BS_EXIT_MIN_CONVEXITY", "0.5"))
    # ── Volatility diagnostics (refine contract quality + exit; never directional) ──
    vol_diagnostics_enabled: bool = os.getenv("ENABLE_VOL_DIAGNOSTICS", "true").lower() in {"1", "true"}
    vol_iv_crush_exit_threshold: float = float(os.getenv("OPTIONS_IV_CRUSH_EXIT", "70"))  # crush risk that discourages overnight hold
    vol_gt_efficiency_bonus: float = float(os.getenv("OPTIONS_GT_EFFICIENCY_BONUS", "6.0"))  # max selection bonus from gamma/theta efficiency


# ── data structures ────────────────────────────────────────────────────────────
@dataclass
class SetupEvaluation:
    symbol: str
    setup_type: str          # ORB | VWAP_RECLAIM | PIVOT_BREAK | ACCUM
    setup_score: float       # 0–100
    grade: str               # A+ (85+) | A (70-84) | B (55-69) | C (<55)
    price: float
    vwap: float
    rvol: float
    orb_high: float
    orb_low: float
    pivot_level: float
    sector_strong: bool
    above_vwap: bool
    institutional_flow: bool
    regime_ok: bool
    reject_reason: str = ""
    notes: list[str] = field(default_factory=list)
    evaluated_at_utc: str = ""


@dataclass
class CallSignal:
    symbol: str
    setup_type: str
    setup_score: float
    grade: str
    # Contract details
    contract_symbol: str = ""
    expiry: str = ""
    strike: float = 0.0
    delta: float = 0.0
    dte: int = 0
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0
    open_interest: int = 0
    volume: int = 0
    # ── Black-Scholes valuation (filled by the options valuation layer) ─────────
    theoretical_value: float = 0.0
    market_mid: float = 0.0
    implied_volatility: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    gamma_theta_ratio: float = 0.0
    mispricing_pct: float = 0.0        # (market_mid - theoretical) / theoretical
    moneyness: float = 0.0             # S / K
    extrinsic_value: float = 0.0
    black_scholes_score: float = 0.0   # 0-100 valuation/convexity score
    expected_value_at_1R: float = 0.0  # BS option value (per share) at +1R underlying move
    expected_value_at_2R: float = 0.0
    expected_value_at_3R: float = 0.0
    expected_loss_at_stop: float = 0.0 # BS option value change (per share) at -1R underlying move
    # ── Volatility diagnostics (contract quality / exit refinement; not directional) ──
    expected_move: float = 0.0          # S · ATM_IV · sqrt(T)  (1σ move over the near expiry)
    iv_rank: float = 50.0               # 0-100 (realized-vol-band proxy when no IV history)
    iv_percentile: float = 50.0         # 0-100
    iv_term_structure: float = 1.0      # near-IV / far-IV  (>1 backwardation = event premium)
    gamma_exposure: float = 0.0         # aggregate dealer gamma proxy
    gamma_theta_efficiency: float = 0.0 # convexity-per-1R / theta-per-day (= gamma_theta_ratio)
    iv_crush_risk: float = 0.0          # 0-100
    volatility_regime_score: float = 0.0  # 0-100 (higher = friendlier for long convexity)
    # ── Probability-weighted management (adapts size + management; NEVER blocks) ──
    probability_of_profit: float = 0.0
    expected_R: float = 0.0
    probability_confidence: float = 0.0
    probability_source: str = ""
    probability_band: str = ""
    probability_size_multiplier: float = 1.0
    management_style: str = "standard"
    partial_profit_trigger: float = 0.0
    trailing_tightness: str = "normal"
    overnight_allowed_by_probability: bool = False
    # internal effective management applied at fill (probability tightens these only)
    stop_loss_pct_eff: float = 0.0
    partial_profit_pct_eff: float = 0.0
    trail_tightness_mult: float = 1.0
    # Sizing
    contracts: int = 0
    premium_cost: float = 0.0
    max_risk: float = 0.0
    # Management levels
    stop_value: float = 0.0      # cost × (1 - stop_loss_pct)
    partial_value: float = 0.0   # cost × (1 + partial_profit_pct)
    # Metadata
    signal_at_utc: str = ""
    reject_reason: str = ""
    status: str = "pending"      # pending | submitted | active | closed


@dataclass
class ActiveCall:
    """Tracks an open call position with 3R target / 1.7R trailing stop."""
    symbol: str
    contract_symbol: str
    expiry: str
    strike: float
    contracts_held: int
    contracts_original: int
    entry_mid: float
    entry_cost: float           # entry_mid × 100 × contracts (= 1R risk)
    # Hard stop (unchanged): entry_cost × (1 – stop_loss_pct)
    hard_stop_value: float = 0.0
    # 3R target: entry_cost × (1 + r_target)
    r_target_value: float = 0.0
    # 1R partial: entry_cost × (1 + partial_profit_pct)  — take 50% off at +1R
    partial_value: float = 0.0
    partial_taken: bool = False
    # Trailing stop state (activates at 1.7R)
    trail_active: bool = False
    trail_stop_value: float = 0.0   # trails up as peak rises
    peak_value: float = 0.0         # highest marked value seen
    opened_at_utc: str = ""
    setup_type: str = ""
    setup_score: float = 0.0
    # Probability-weighted management (protective-direction only).
    management_style: str = "standard"
    trail_tightness_mult: float = 1.0   # <=1.0 → trails closer to peak (more protective)
    overnight_allowed: bool = False
    probability_of_profit: float = 0.0
    probability_band: str = ""


# ── event logging ──────────────────────────────────────────────────────────────
def _alpha_event(event: str, payload: dict) -> None:
    try:
        line = json.dumps({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
        })
        with open(ALPHA_EVENTS_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ── setup detection ────────────────────────────────────────────────────────────

def _compute_orb(bars_1m: pd.DataFrame, orb_minutes: int) -> tuple[float, float, str]:
    """
    Opening-range high/low over the REAL 09:30 .. (09:30+orb_minutes) ET window of the
    most recent session present in the bars.

    Returns (orb_high, orb_low, status). status == "ok" when the window contained bars;
    otherwise a diagnostic code and (0.0, 0.0):
        "no_bars"               — empty/None input
        "no_session_open_bars"  — no bars fell inside 09:30-09:45 ET (e.g. premarket-only
                                  window, or a mid-session/completed window that doesn't
                                  include the open) — so ORB/PIVOT must NOT be scored, and
                                  the caller can report WHY instead of treating 0.0 as a level
        "error"                 — unexpected index/parse failure

    Fixes the prior bug where session-open was derived from bars.index[0] and only an
    upper bound was applied: on any window not starting at 09:30 the ORB window was empty
    and orb_high silently became 0.0, disabling ORB and PIVOT scoring without trace.
    Anchors to a true 09:30 ET wall-clock window, bounded on BOTH sides, and handles
    tz-aware and tz-naive indexes (naive assumed UTC, matching Alpaca/yfinance feeds).
    """
    try:
        if bars_1m is None or bars_1m.empty:
            return 0.0, 0.0, "no_bars"
        idx = bars_1m.index
        if not isinstance(idx, pd.DatetimeIndex):
            return 0.0, 0.0, "error"
        idx_et = idx.tz_convert(ET) if idx.tz is not None else idx.tz_localize("UTC").tz_convert(ET)
        # Anchor to the active (most recent) session date in the data.
        session_date = idx_et[-1].date()
        session_open = pd.Timestamp(
            year=session_date.year, month=session_date.month, day=session_date.day,
            hour=9, minute=30, tz=ET,
        )
        orb_end = session_open + timedelta(minutes=orb_minutes)
        in_window = (idx_et >= session_open) & (idx_et < orb_end)
        orb_bars = bars_1m[in_window]
        if orb_bars.empty:
            return 0.0, 0.0, "no_session_open_bars"
        return float(orb_bars["high"].max()), float(orb_bars["low"].min()), "ok"
    except Exception:
        return 0.0, 0.0, "error"


def _compute_vwap(bars: pd.DataFrame) -> float:
    try:
        tp = (bars["high"] + bars["low"] + bars["close"]) / 3
        vol = pd.to_numeric(bars["volume"], errors="coerce").fillna(0)
        if vol.sum() == 0:
            return float(bars["close"].iloc[-1])
        return float((tp * vol).sum() / vol.sum())
    except Exception:
        return 0.0


def _compute_rvol(bars_1m: pd.DataFrame, lookback_bars: int = 20) -> float:
    """Recent volume vs trailing average."""
    try:
        if len(bars_1m) < 2:
            return 1.0
        recent = float(bars_1m["volume"].iloc[-1])
        avg = float(bars_1m["volume"].tail(lookback_bars + 1).iloc[:-1].mean())
        return recent / avg if avg > 0 else 1.0
    except Exception:
        return 1.0


def detect_setup(
    symbol: str,
    bars_1m: pd.DataFrame,
    config: AlphaCallConfig,
    market_context: dict,
) -> SetupEvaluation:
    """Score each symbol for A/A+ call setups. Returns a SetupEvaluation."""
    now_utc = datetime.now(timezone.utc).isoformat()

    ev = SetupEvaluation(
        symbol=symbol,
        setup_type="NONE",
        setup_score=0.0,
        grade="C",
        price=0.0,
        vwap=0.0,
        rvol=0.0,
        orb_high=0.0,
        orb_low=0.0,
        pivot_level=0.0,
        sector_strong=False,
        above_vwap=False,
        institutional_flow=False,
        regime_ok=False,
        evaluated_at_utc=now_utc,
    )

    if bars_1m is None or bars_1m.empty:
        ev.reject_reason = "rejected_by_setup_quality:no_bars"
        return ev

    # ── regime gate ───────────────────────────────────────────────────────────
    bias = str(
        market_context.get("overall_market_bias")
        or market_context.get("market_bias")
        or market_context.get("regime")
        or "unknown"
    ).lower()
    regime_ok = any(r in bias for r in config.allowed_regimes)
    ev.regime_ok = regime_ok
    if not regime_ok and config.require_bullish_regime:
        ev.reject_reason = f"rejected_by_market_regime:{bias}"
        return ev

    # ── base metrics ─────────────────────────────────────────────────────────
    price = float(bars_1m["close"].iloc[-1])
    vwap = _compute_vwap(bars_1m)
    rvol = _compute_rvol(bars_1m)
    orb_high, orb_low, orb_status = _compute_orb(bars_1m, config.orb_minutes)
    above_vwap = price >= vwap

    ev.price = price
    ev.vwap = vwap
    ev.rvol = rvol
    ev.orb_high = orb_high
    ev.orb_low = orb_low
    ev.above_vwap = above_vwap
    ev.institutional_flow = rvol >= config.min_rvol_for_accum

    # ── setup scoring (A+ = 85+, A = 70–84) ─────────────────────────────────
    score = 0.0
    best_setup = "NONE"
    notes: list[str] = []

    # Surface WHY the opening range is unavailable instead of silently scoring orb_high=0.
    if orb_status != "ok":
        notes.append(f"orb_unavailable:{orb_status}")

    # Regime contribution (all setups benefit from a strong regime)
    if "bullish" in bias or "risk_on" in bias or "uptrend" in bias:
        score += 15
        notes.append(f"regime={bias}(+15)")
    elif "neutral" in bias or "constructive" in bias:
        score += 8
        notes.append(f"regime={bias}(+8)")

    # ── ORB breakout ─────────────────────────────────────────────────────────
    if orb_high > 0:
        pct_above_orb = (price - orb_high) / orb_high if orb_high > 0 else -1
        if 0 < pct_above_orb <= 0.025:           # fresh breakout, not chasing
            orb_score = 30 + min(15, rvol * 5)  # up to +45 from ORB + volume
            if orb_score > score:
                score = orb_score
                best_setup = "ORB"
            notes.append(f"ORB_break={pct_above_orb:.2%} RVOL={rvol:.1f}x")

    # ── VWAP reclaim ─────────────────────────────────────────────────────────
    if above_vwap and len(bars_1m) >= 3:
        # Check if price was below VWAP recently and just crossed
        prev_close = float(bars_1m["close"].iloc[-3])
        was_below = prev_close < vwap
        if was_below:
            vwap_score = 25 + min(15, rvol * 6)
            if above_vwap and rvol >= 1.3:
                vwap_score += 10
            if vwap_score > score:
                score = vwap_score
                best_setup = "VWAP_RECLAIM"
            notes.append(f"VWAP_reclaim price={price:.2f} vwap={vwap:.2f}")

    # ── Institutional accumulation ────────────────────────────────────────────
    if rvol >= config.min_rvol_for_accum and above_vwap:
        # Look for consecutive bullish bars on high volume
        last_3 = bars_1m.tail(3)
        bullish_bars = int((last_3["close"] > last_3["open"]).sum())
        accum_score = 20 + bullish_bars * 8 + min(15, (rvol - 1.5) * 10)
        if accum_score > score:
            score = accum_score
            best_setup = "ACCUM"
        notes.append(f"accum RVOL={rvol:.1f}x bullish_bars={bullish_bars}/3")

    # ── Pivot breakout ────────────────────────────────────────────────────────
    # Use ORB high as a rough pivot proxy; real pivot levels come from structure map
    if orb_high > 0 and price > orb_high:
        pivot_score = 22 + min(12, rvol * 4)
        if pivot_score > score:
            score = pivot_score
            best_setup = "PIVOT_BREAK"
            ev.pivot_level = orb_high
        notes.append(f"pivot_break level={orb_high:.2f} price={price:.2f}")

    # ── Sector leadership bonus ───────────────────────────────────────────────
    # Apply +5 when sector context supports risk-on momentum. Supports the legacy
    # keys (sector_leadership / sector_bias with a 'positive' flag) AND the key the
    # context builder actually emits: sector_strength.sector_rotation_bias
    # ("risk_on_growth" | "cyclical_value" | "defensive" | "mixed").
    sector_positive = False
    rotation_bias = ""
    legacy = market_context.get("sector_leadership") or market_context.get("sector_bias") or {}
    if isinstance(legacy, dict) and legacy.get("positive"):
        sector_positive = True
    sector_strength = market_context.get("sector_strength")
    if isinstance(sector_strength, dict):
        rotation_bias = str(sector_strength.get("sector_rotation_bias") or "").lower()
    if not rotation_bias:
        rotation_bias = str(market_context.get("sector_rotation_bias") or "").lower()
    if rotation_bias in ("risk_on_growth", "cyclical_value"):
        sector_positive = True
    if sector_positive:
        score = min(100, score + 5)
        ev.sector_strong = True
        notes.append(f"sector_leader(+5){':' + rotation_bias if rotation_bias else ''}")

    # ── RSI check (avoid overbought entries) ─────────────────────────────────
    try:
        closes = pd.to_numeric(bars_1m["close"], errors="coerce").dropna()
        if len(closes) >= 14:
            delta = closes.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = float((100 - 100 / (1 + rs)).iloc[-1])
            if rsi > 80:
                score = max(0, score - 15)
                notes.append(f"rsi={rsi:.0f} overbought(-15)")
            elif rsi > 70:
                score = max(0, score - 8)
                notes.append(f"rsi={rsi:.0f} extended(-8)")
            elif 45 <= rsi <= 65:
                score = min(100, score + 5)
                notes.append(f"rsi={rsi:.0f} healthy(+5)")
    except Exception:
        pass

    # ── Volume surge bonus ───────────────────────────────────────────────────
    if rvol >= 2.5:
        score = min(100, score + 10)
        notes.append(f"volume_surge RVOL={rvol:.1f}x(+10)")
    elif rvol >= 1.8:
        score = min(100, score + 5)

    ev.setup_type = best_setup
    ev.setup_score = round(score, 1)
    ev.notes = notes

    if score >= 85:
        ev.grade = "A+"
    elif score >= 70:
        ev.grade = "A"
    elif score >= 55:
        ev.grade = "B"
    else:
        ev.grade = "C"
        ev.reject_reason = f"rejected_by_setup_quality:score={score:.0f}<{config.min_setup_score}"

    return ev


# ── Black-Scholes valuation layer ────────────────────────────────────────────
# Applied AFTER the Alpha setup is approved and AFTER the hard liquidity gates.
# It refines selection by theoretical value, convexity (gamma/theta) and expected
# 3R payoff — it NEVER overrides regime / setup-quality / liquidity / risk gates.

try:                                            # use scipy when available, else math.erf
    from scipy.stats import norm as _scipy_norm

    def _ncdf(x: float) -> float:
        return float(_scipy_norm.cdf(x))

    def _npdf(x: float) -> float:
        return float(_scipy_norm.pdf(x))
except Exception:                               # pragma: no cover - fallback path
    def _ncdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _npdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def black_scholes_call(S: float, K: float, T: float, r: float,
                       sigma: float, q: float = 0.0) -> dict:
    """Black-Scholes call price + greeks (per share; theta per calendar day, vega per 1 vol pt)."""
    if S <= 0 or K <= 0:
        return {"price": 0.0, "delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    if T <= 0 or sigma <= 0:
        intrinsic = max(0.0, S - K)
        return {"price": intrinsic, "delta": 1.0 if S > K else 0.0,
                "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    eqT, erT = math.exp(-q * T), math.exp(-r * T)
    Nd1, Nd2, pdf = _ncdf(d1), _ncdf(d2), _npdf(d1)
    price = S * eqT * Nd1 - K * erT * Nd2
    delta = eqT * Nd1
    gamma = eqT * pdf / (S * sigma * sqrtT)
    theta_year = (-(S * eqT * pdf * sigma) / (2 * sqrtT)
                  - r * K * erT * Nd2 + q * S * eqT * Nd1)
    theta = theta_year / 365.0
    vega = S * eqT * pdf * sqrtT / 100.0
    return {"price": price, "delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


def implied_vol_call(market: float, S: float, K: float, T: float,
                     r: float, q: float = 0.0) -> Optional[float]:
    """Solve IV from a market call price via bisection (price is monotone in sigma)."""
    if market <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    intrinsic = max(0.0, S * math.exp(-q * T) - K * math.exp(-r * T))
    upper = S * math.exp(-q * T)
    if market < intrinsic - 1e-4 or market >= upper:   # outside no-arbitrage bounds
        return None
    lo, hi = 1e-4, 5.0
    if not (black_scholes_call(S, K, T, r, lo, q)["price"] <= market
            <= black_scholes_call(S, K, T, r, hi, q)["price"]):
        return None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        p = black_scholes_call(S, K, T, r, mid, q)["price"]
        if abs(p - market) < 1e-4:
            return mid
        if p < market:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def resolve_risk_free_rate(config: AlphaCallConfig, market_context: dict | None) -> float:
    """Risk-free rate: prefer a Treasury proxy already in macro context, else config default."""
    mc = market_context or {}
    for k in ("risk_free_rate", "ten_year_yield", "treasury_10y", "us10y", "two_year_yield"):
        v = mc.get(k)
        if v is None:
            continue
        try:
            v = float(v)
            return v / 100.0 if v > 1.0 else v   # accept percent (4.3) or decimal (0.043)
        except Exception:
            continue
    return float(config.risk_free_rate)


def price_call_contract(c: dict, S: float, config: AlphaCallConfig,
                        risk_free_rate: float, momentum_rvol: float = 1.0) -> dict:
    """
    Add Black-Scholes valuation, greeks, mispricing, score, expected-R profile and a
    `bs_reject_reason` to a hard-gate-passing contract dict. Returns the enriched dict.
    `bs_reject_reason` is "" when the contract passes all valuation gates.
    """
    r, q = risk_free_rate, config.dividend_yield
    K = float(c["strike"]); mid = float(c["mid"]); dte = int(c["dte"])
    T = max(dte, 0) / 365.0
    intrinsic = max(0.0, S - K)

    # IV: prefer feed IV, else solve numerically from the market mid.
    iv = float(c.get("implied_volatility") or 0.0)
    iv_source = "feed"
    if iv <= 0.0:
        solved = implied_vol_call(mid, S, K, T, r, q)
        if solved is None:
            c.update({
                "theoretical_value": 0.0, "market_mid": round(mid, 4), "implied_volatility": 0.0,
                "gamma": 0.0, "theta": 0.0, "vega": 0.0, "gamma_theta_ratio": 0.0,
                "mispricing_pct": 0.0, "moneyness": round(S / K, 4) if K else 0.0,
                "extrinsic_value": round(mid - intrinsic, 4), "black_scholes_score": 0.0,
                "expected_value_at_1R": 0.0, "expected_value_at_2R": 0.0,
                "expected_value_at_3R": 0.0, "expected_loss_at_stop": 0.0,
                "bs_reject_reason": "rejected_by_black_scholes_iv",
            })
            return c
        iv, iv_source = solved, "solved"

    Rm = config.underlying_r_move_pct
    bs = black_scholes_call(S, K, T, r, iv, q)
    theo, gamma, theta, vega = bs["price"], bs["gamma"], bs["theta"], bs["vega"]
    theta_abs = abs(theta)
    # Scale-invariant convexity-vs-decay. Raw gamma/theta is NOT comparable across
    # underlyings (gamma ∝ 1/S), so it would reject every contract on high-priced names.
    # Instead: convexity P&L from a +1R underlying move (½·γ·(S·R)²) per dollar of one
    # day's theta decay — i.e. "how many days of decay does a 1R convexity pop pay for".
    convexity_1R = 0.5 * gamma * (S * Rm) ** 2
    gamma_theta_ratio = convexity_1R / theta_abs if theta_abs > 1e-9 else 999.0
    theta_pct_day = theta_abs / mid if mid > 0 else 1.0
    mispricing_pct = (mid - theo) / theo if theo > 0.01 else (1.0 if mid > 0 else 0.0)
    moneyness = S / K if K > 0 else 0.0
    extrinsic = mid - intrinsic

    # Expected option value (per share) under instantaneous underlying R-moves.
    ev1 = black_scholes_call(S * (1 + 1 * Rm), K, T, r, iv, q)["price"]
    ev2 = black_scholes_call(S * (1 + 2 * Rm), K, T, r, iv, q)["price"]
    ev3 = black_scholes_call(S * (1 + 3 * Rm), K, T, r, iv, q)["price"]
    ev_stop = black_scholes_call(S * (1 - 1 * Rm), K, T, r, iv, q)["price"]
    expected_loss_at_stop = ev_stop - mid
    risk_per_share = mid - ev_stop
    expected_r = ((ev3 - mid) / risk_per_share) if risk_per_share > 1e-6 else ((ev3 - mid) / max(mid, 1e-6))

    # ── valuation gates (never override hard risk gates; only refine selection) ──
    bs_reject = ""
    if iv > config.bs_max_iv and momentum_rvol < config.bs_iv_momentum_rvol:
        bs_reject = f"rejected_by_black_scholes_iv_inflated:iv={iv:.2f}>{config.bs_max_iv:.2f}@rvol={momentum_rvol:.1f}"
    elif mispricing_pct > config.bs_max_overprice_pct:
        bs_reject = f"rejected_by_black_scholes_overpriced:mkt_over_theo={mispricing_pct:.1%}"
    elif theta_pct_day > config.bs_max_theta_pct_per_day:
        bs_reject = f"rejected_by_black_scholes_theta_decay:{theta_pct_day:.1%}/day"
    elif gamma_theta_ratio < config.bs_min_gamma_theta:
        bs_reject = f"rejected_by_black_scholes_gamma_theta:ratio={gamma_theta_ratio:.2f}"

    # ── black_scholes_score (0-100): weighted valuation + convexity + quality ────
    edge_score = _clip(0.5 - mispricing_pct * 2.0, 0.0, 1.0)             # cheaper-than-theo is better
    delta_score = _clip(1.0 - abs(c["delta"] - 0.575) / 0.25, 0.0, 1.0)  # ideal mid of 0.45-0.70
    gt_score = _clip(gamma_theta_ratio / 8.0, 0.0, 1.0)
    spread_score = _clip(1.0 - c["spread_pct"] / max(config.max_spread_pct, 1e-6), 0.0, 1.0)
    oi_score = _clip(int(c["open_interest"]) / 1000.0, 0.0, 1.0)
    vol_score = _clip(int(c["volume"]) / 100.0, 0.0, 1.0)
    liquidity_score = 0.6 * oi_score + 0.4 * vol_score
    dte_score = _clip(1.0 - abs(dte - 37.5) / 22.5, 0.0, 1.0)            # peak ~37 DTE
    moneyness_score = _clip(1.0 - abs(moneyness - 1.0) / 0.10, 0.0, 1.0) # ATM / slightly ITM
    iv_score = 1.0 if 0.20 <= iv <= 0.80 else _clip(1.0 - (max(0.0, iv - 0.80) + max(0.0, 0.20 - iv)) / 1.0, 0.0, 1.0)
    bs_score = 100.0 * (
        0.22 * edge_score + 0.16 * delta_score + 0.16 * gt_score +
        0.12 * spread_score + 0.12 * liquidity_score + 0.08 * dte_score +
        0.08 * moneyness_score + 0.06 * iv_score
    )
    # Expected-R profile refines the final selection (step 5): up to +15 for strong convexity.
    composite = bs_score + _clip(expected_r, 0.0, 5.0) * 3.0

    c.update({
        "theoretical_value": round(theo, 4),
        "market_mid": round(mid, 4),
        "implied_volatility": round(iv, 4),
        "iv_source": iv_source,
        "gamma": round(gamma, 6),
        "theta": round(theta, 6),
        "vega": round(vega, 6),
        "gamma_theta_ratio": round(gamma_theta_ratio, 3),
        "theta_pct_per_day": round(theta_pct_day, 4),
        "mispricing_pct": round(mispricing_pct, 4),
        "moneyness": round(moneyness, 4),
        "extrinsic_value": round(extrinsic, 4),
        "black_scholes_score": round(bs_score, 2),
        "composite_score": round(composite, 2),
        "expected_value_at_1R": round(ev1, 4),
        "expected_value_at_2R": round(ev2, 4),
        "expected_value_at_3R": round(ev3, 4),
        "expected_loss_at_stop": round(expected_loss_at_stop, 4),
        "expected_r_multiple": round(expected_r, 3),
        "bs_reject_reason": bs_reject,
    })
    return c


def _apply_bs_fields(sig: "CallSignal", c: dict) -> None:
    """Copy Black-Scholes valuation fields from a (possibly rejected) contract onto a signal."""
    sig.theoretical_value   = float(c.get("theoretical_value", 0.0) or 0.0)
    sig.market_mid          = float(c.get("market_mid", c.get("mid", 0.0)) or 0.0)
    sig.implied_volatility  = float(c.get("implied_volatility", 0.0) or 0.0)
    sig.gamma               = float(c.get("gamma", 0.0) or 0.0)
    sig.theta               = float(c.get("theta", 0.0) or 0.0)
    sig.vega                = float(c.get("vega", 0.0) or 0.0)
    sig.gamma_theta_ratio   = float(c.get("gamma_theta_ratio", 0.0) or 0.0)
    sig.mispricing_pct      = float(c.get("mispricing_pct", 0.0) or 0.0)
    sig.moneyness           = float(c.get("moneyness", 0.0) or 0.0)
    sig.extrinsic_value     = float(c.get("extrinsic_value", 0.0) or 0.0)
    sig.black_scholes_score = float(c.get("black_scholes_score", 0.0) or 0.0)
    sig.expected_value_at_1R = float(c.get("expected_value_at_1R", 0.0) or 0.0)
    sig.expected_value_at_2R = float(c.get("expected_value_at_2R", 0.0) or 0.0)
    sig.expected_value_at_3R = float(c.get("expected_value_at_3R", 0.0) or 0.0)
    sig.expected_loss_at_stop = float(c.get("expected_loss_at_stop", 0.0) or 0.0)


# ── volatility diagnostics ───────────────────────────────────────────────────────

def _realized_vol_annualized(bars: "pd.DataFrame") -> float:
    """Annualized realized volatility from close-to-close returns (0.0 if too few bars)."""
    try:
        if bars is None or "close" not in getattr(bars, "columns", []) or len(bars) < 20:
            return 0.0
        closes = bars["close"].astype(float)
        rets = np.log(closes / closes.shift(1)).dropna()
        if len(rets) < 15:
            return 0.0
        per_bar = float(rets.std())
        # Annualize from the bar cadence: ~252 trading days, ~390 1-min bars/day.
        # Detect cadence from index spacing; default to 1-min bars.
        bars_per_year = 252 * 390
        try:
            idx = bars.index
            if isinstance(idx, pd.DatetimeIndex) and len(idx) >= 2:
                dt_sec = abs((idx[-1] - idx[-2]).total_seconds())
                if dt_sec >= 23 * 3600:      # daily bars
                    bars_per_year = 252
                elif dt_sec >= 3000:          # hourly-ish
                    bars_per_year = 252 * 7
        except Exception:
            pass
        return per_bar * math.sqrt(bars_per_year)
    except Exception:
        return 0.0


def _chain_iv_term_points(chain: list[dict], S: float) -> list[tuple]:
    """Return [(dte, strike, iv)] for chain contracts with a usable feed IV."""
    pts = []
    today = datetime.now(ET).date()
    for c in chain or []:
        iv = float(c.get("implied_volatility") or 0.0)
        K = float(c.get("strike") or c.get("strike_price") or 0.0)
        dte = c.get("dte")
        if dte is None:
            exp = str(c.get("expiration_date") or c.get("expiry") or "")[:10]
            try:
                dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
            except Exception:
                dte = None
        if iv > 0 and K > 0 and dte is not None and dte >= 0:
            pts.append((int(dte), K, iv))
    return pts


def _atm_iv_for_group(pts: list[tuple], target_dte: int, S: float) -> tuple[float, int]:
    """ATM IV (strike nearest S) within the expiry group nearest target_dte."""
    if not pts:
        return 0.0, 0
    dtes = sorted({p[0] for p in pts})
    grp_dte = min(dtes, key=lambda d: abs(d - target_dte))
    grp = [p for p in pts if p[0] == grp_dte]
    atm = min(grp, key=lambda p: abs(p[1] - S))
    return atm[2], grp_dte


def compute_volatility_diagnostics(symbol: str, S: float, chain: list[dict],
                                   bars: "pd.DataFrame", config: AlphaCallConfig,
                                   market_context: dict | None = None,
                                   selected_contract: dict | None = None) -> dict:
    """
    Volatility/contract-quality diagnostics that REFINE selection and exit timing.
    They are never a directional driver (momentum/setup owns direction) and never
    override liquidity, Black-Scholes fair value, or RiskManager.

    Gracefully degrades: with no IV feed / no history, IV rank/percentile fall back to
    a realized-vol band (flagged), and unavailable measures return neutral defaults.
    """
    mc = market_context or {}
    out = {
        "symbol": symbol, "expected_move": 0.0, "iv_rank": 50.0, "iv_percentile": 50.0,
        "iv_term_structure": 1.0, "gamma_exposure": 0.0, "gamma_theta_efficiency": 0.0,
        "iv_crush_risk": 0.0, "volatility_regime_score": 50.0,
        "atm_iv": 0.0, "iv_history_available": False, "iv_rank_source": "none",
    }
    if S <= 0:
        return out

    pts = _chain_iv_term_points(chain, S)
    near_iv, near_dte = _atm_iv_for_group(pts, config.min_dte, S) if pts else (0.0, 0)
    far_iv, far_dte = _atm_iv_for_group(pts, config.max_dte, S) if pts else (0.0, 0)
    atm_iv = near_iv or far_iv
    out["atm_iv"] = round(atm_iv, 4)

    # Expected move (1σ) over the near expiry.
    if atm_iv > 0 and near_dte > 0:
        out["expected_move"] = round(S * atm_iv * math.sqrt(near_dte / 365.0), 4)
    elif atm_iv > 0:
        out["expected_move"] = round(S * atm_iv * math.sqrt(config.min_dte / 365.0), 4)

    # IV term structure (near / far). >1 = backwardation (event/stress premium).
    if near_iv > 0 and far_iv > 0 and far_dte != near_dte:
        out["iv_term_structure"] = round(near_iv / far_iv, 4)

    # IV rank / percentile — fall back to a realized-vol band when no IV time history.
    rv = _realized_vol_annualized(bars)
    if atm_iv > 0 and rv > 0:
        lo, hi = 0.5 * rv, 2.0 * rv
        rank = _clip((atm_iv - lo) / (hi - lo) if hi > lo else 0.5, 0.0, 1.0) * 100.0
        out["iv_rank"] = round(rank, 1)
        out["iv_percentile"] = round(rank, 1)   # single-window proxy
        out["iv_rank_source"] = "realized_vol_band"
        out["iv_history_available"] = False
    elif atm_iv > 0 and len(pts) >= 3:
        ivs = sorted(p[2] for p in pts)
        lo, hi = ivs[0], ivs[-1]
        rank = _clip((atm_iv - lo) / (hi - lo) if hi > lo else 0.5, 0.0, 1.0) * 100.0
        out["iv_rank"] = round(rank, 1)
        out["iv_percentile"] = round(rank, 1)
        out["iv_rank_source"] = "chain_cross_section"
    # else: neutral 50/50 already set (graceful fallback)

    # Gamma exposure proxy = ATM gamma · total open interest · 100 · S.
    try:
        r = resolve_risk_free_rate(config, mc)
        if atm_iv > 0 and near_dte > 0:
            atm_K = min((p[1] for p in pts), key=lambda k: abs(k - S)) if pts else S
            g = black_scholes_call(S, atm_K, near_dte / 365.0, r, atm_iv, config.dividend_yield)["gamma"]
            total_oi = sum(int(c.get("open_interest") or 0) for c in (chain or []))
            out["gamma_exposure"] = round(g * total_oi * 100.0 * S, 2)
    except Exception:
        pass

    # Gamma/theta efficiency — reuse the selected (or best-priced) contract's ratio.
    if selected_contract is not None:
        out["gamma_theta_efficiency"] = round(float(selected_contract.get("gamma_theta_ratio", 0.0) or 0.0), 3)

    # IV crush risk (0-100): backwardation + elevated rank + earnings inside the near expiry.
    crush = 0.0
    ts = out["iv_term_structure"]
    if ts > 1.15:
        crush += 40.0
    elif ts > 1.05:
        crush += 20.0
    crush += 0.4 * out["iv_rank"]
    earnings_days = None
    ec = mc.get("earnings_calendar") or mc.get("earnings_in_days")
    if isinstance(ec, dict):
        earnings_days = ec.get(symbol)
    elif isinstance(ec, (int, float)):
        earnings_days = ec
    if earnings_days is not None:
        try:
            ed = int(earnings_days)
            if 0 <= ed <= max(near_dte, config.min_dte):
                crush += 30.0
        except Exception:
            pass
    out["iv_crush_risk"] = round(_clip(crush, 0.0, 100.0), 1)

    # Volatility regime score (0-100): friendlier when IV is moderate and the macro
    # vol regime is calm; harsher at crush-prone extremes.
    reg = 50.0
    if out["iv_rank"] <= 60:
        reg += 20.0
    elif out["iv_rank"] >= 85:
        reg -= 25.0
    vr = str(mc.get("volatility_regime") or mc.get("vix_regime") or "").lower()
    if any(k in vr for k in ("low", "calm", "stable", "complacent")):
        reg += 15.0
    elif any(k in vr for k in ("high", "panic", "elevated", "stress", "fear")):
        reg -= 20.0
    vix = mc.get("vix") or mc.get("vix_level")
    if isinstance(vix, (int, float)):
        if vix < 16:
            reg += 10.0
        elif vix > 28:
            reg -= 15.0
    out["volatility_regime_score"] = round(_clip(reg, 0.0, 100.0), 1)
    return out


def overnight_hold_score(base_score: float, vol_diag: dict, config: AlphaCallConfig | None = None) -> float:
    """
    Overnight-hold quality (0-100). High IV-crush risk and backwardation reduce the
    score (don't carry a convex long position into a vol crush). Advisory only — the
    hard 3R / stop / trailing exit ladder remains authoritative.
    """
    score = float(base_score or 0.0)
    crush = float(vol_diag.get("iv_crush_risk", 0.0) or 0.0)
    score -= 0.5 * crush
    if float(vol_diag.get("iv_term_structure", 1.0) or 1.0) > 1.15:
        score -= 10.0
    reg = float(vol_diag.get("volatility_regime_score", 50.0) or 50.0)
    score += 0.15 * (reg - 50.0)
    return round(_clip(score, 0.0, 100.0), 2)


def _apply_vol_fields(sig: "CallSignal", vol_diag: dict) -> None:
    """Copy volatility-diagnostic fields onto a signal (no-op for an empty dict)."""
    if not vol_diag:
        return
    sig.expected_move = float(vol_diag.get("expected_move", 0.0) or 0.0)
    sig.iv_rank = float(vol_diag.get("iv_rank", 50.0) or 50.0)
    sig.iv_percentile = float(vol_diag.get("iv_percentile", 50.0) or 50.0)
    sig.iv_term_structure = float(vol_diag.get("iv_term_structure", 1.0) or 1.0)
    sig.gamma_exposure = float(vol_diag.get("gamma_exposure", 0.0) or 0.0)
    sig.gamma_theta_efficiency = float(vol_diag.get("gamma_theta_efficiency", 0.0) or 0.0)
    sig.iv_crush_risk = float(vol_diag.get("iv_crush_risk", 0.0) or 0.0)
    sig.volatility_regime_score = float(vol_diag.get("volatility_regime_score", 0.0) or 0.0)


# ── contract selection ─────────────────────────────────────────────────────────

def evaluate_call_contracts(
    symbol: str,
    price: float,
    chain: list[dict],
    config: AlphaCallConfig,
    risk_free_rate: float | None = None,
    momentum_rvol: float = 1.0,
) -> dict:
    """
    Full contract evaluation. Applies hard liquidity gates (DTE / delta / OI / volume /
    mid / spread), then the Black-Scholes valuation layer to the survivors.

    Returns a dict:
        best           — best qualified contract dict (passes hard + BS gates) or None
        best_rejected  — highest-scoring hard-gate survivor that FAILED a BS gate (for logging)
        qualified      — all qualified contract dicts (sorted, best first)
        counts         — diagnostic counts for the chain-health report
        best_black_scholes_score — max bs_score among priced contracts
    """
    today = datetime.now(ET).date()
    r = risk_free_rate if risk_free_rate is not None else float(config.risk_free_rate)

    hard_survivors: list[dict] = []
    for contract in chain:
        try:
            if str(contract.get("type", "")).lower() != "call":
                continue
            expiry_str = str(contract.get("expiration_date") or contract.get("expiry") or "")
            if not expiry_str:
                continue
            dte = (datetime.strptime(expiry_str[:10], "%Y-%m-%d").date() - today).days
            if not (config.min_dte <= dte <= config.max_dte):
                continue
            delta = abs(float(contract.get("delta") or contract.get("greeks", {}).get("delta", 0) or 0))
            if not (config.min_delta <= delta <= config.max_delta):
                continue
            oi = int(contract.get("open_interest") or 0)
            vol = int(contract.get("volume") or 0)
            bid = float(contract.get("bid_price") or contract.get("bid") or 0)
            ask = float(contract.get("ask_price") or contract.get("ask") or 0)
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
            if oi < config.min_open_interest or vol < config.min_volume or mid <= 0:
                continue
            spread_pct = (ask - bid) / mid if mid > 0 else 1.0
            if spread_pct > config.max_spread_pct:
                continue
            hard_survivors.append({
                "contract_symbol": str(contract.get("symbol") or ""),
                "expiry": expiry_str[:10],
                "strike": float(contract.get("strike_price") or contract.get("strike") or 0),
                "delta": delta, "dte": dte, "bid": bid, "ask": ask, "mid": mid,
                "open_interest": oi, "volume": vol, "spread_pct": spread_pct,
                "implied_volatility": float(contract.get("implied_volatility") or 0.0),
            })
        except Exception:
            continue

    counts = {
        "hard_gate_survivors": len(hard_survivors),
        "contracts_priced_by_black_scholes": 0,
        "contracts_rejected_overpriced": 0,
        "contracts_rejected_theta_decay": 0,
        "contracts_rejected_gamma_theta": 0,
        "contracts_rejected_iv": 0,
        "contracts_rejected_iv_inflated": 0,
        "qualified": 0,
    }

    # If BS layer disabled, fall back to the legacy delta-nearest / tightest-spread pick.
    if not config.bs_enabled:
        if not hard_survivors:
            return {"best": None, "best_rejected": None, "qualified": [],
                    "counts": counts, "best_black_scholes_score": 0.0}
        hard_survivors.sort(key=lambda c: (abs(c["delta"] - 0.55), c["spread_pct"]))
        return {"best": hard_survivors[0], "best_rejected": None, "qualified": hard_survivors,
                "counts": counts, "best_black_scholes_score": 0.0}

    priced: list[dict] = []
    for c in hard_survivors:
        pc = price_call_contract(c, price, config, r, momentum_rvol)
        counts["contracts_priced_by_black_scholes"] += 1
        reason = pc.get("bs_reject_reason", "")
        if reason.startswith("rejected_by_black_scholes_overpriced"):
            counts["contracts_rejected_overpriced"] += 1
        elif reason.startswith("rejected_by_black_scholes_theta_decay"):
            counts["contracts_rejected_theta_decay"] += 1
        elif reason.startswith("rejected_by_black_scholes_gamma_theta"):
            counts["contracts_rejected_gamma_theta"] += 1
        elif reason == "rejected_by_black_scholes_iv":
            counts["contracts_rejected_iv"] += 1
        elif reason.startswith("rejected_by_black_scholes_iv_inflated"):
            counts["contracts_rejected_iv_inflated"] += 1
        priced.append(pc)

    qualified = [c for c in priced if not c.get("bs_reject_reason")]
    counts["qualified"] = len(qualified)
    best_score = max((c.get("black_scholes_score", 0.0) for c in priced), default=0.0)

    qualified.sort(key=lambda c: c.get("composite_score", 0.0), reverse=True)
    # Best rejected = highest BS score among priced contracts that failed a valuation gate.
    rejected = [c for c in priced if c.get("bs_reject_reason")]
    rejected.sort(key=lambda c: c.get("black_scholes_score", 0.0), reverse=True)

    return {
        "best": qualified[0] if qualified else None,
        "best_rejected": rejected[0] if rejected else None,
        "qualified": qualified,
        "counts": counts,
        "best_black_scholes_score": round(best_score, 2),
    }


def select_call_contract(
    symbol: str,
    price: float,
    chain: list[dict],
    config: AlphaCallConfig,
    risk_free_rate: float | None = None,
    momentum_rvol: float = 1.0,
) -> Optional[dict]:
    """
    Select the best call contract: DTE 21–60, delta 0.45–0.70, liquid, tight spread,
    then refined by Black-Scholes valuation, convexity and expected-R payoff.
    Returns the contract dict (enriched with BS fields) or None.
    """
    return evaluate_call_contracts(
        symbol, price, chain, config, risk_free_rate, momentum_rvol
    )["best"]


# ── position sizing ───────────────────────────────────────────────────────────

def size_position(
    equity: float,
    mid: float,
    config: AlphaCallConfig,
) -> tuple[int, float, float]:
    """Return (contracts, premium_cost, max_risk) for a call trade."""
    max_risk_dollars = equity * config.risk_per_trade_pct
    cost_per_contract = mid * 100  # options multiplier
    if cost_per_contract <= 0:
        return 0, 0.0, 0.0
    contracts = max(1, math.floor(max_risk_dollars / cost_per_contract))
    premium_cost = contracts * cost_per_contract
    # True max risk = full premium paid (long call, no further liability)
    return contracts, round(premium_cost, 2), round(premium_cost, 2)


# ── main engine ───────────────────────────────────────────────────────────────

# ── Alpha Options Screener (universe selection) ──────────────────────────────────
# Liquid, optionable, high-momentum core. Dynamic discoveries (e.g. TradingView
# institutional candidates) are appended by the caller. This is the universe the
# Alpha engine scans for long calls — independent of the Buffett ownership universe.

ALPHA_STATIC_CORE = [
    "SPY", "QQQ", "IWM",
    "NVDA", "TSLA", "AMD", "AAPL", "MSFT", "META", "AMZN", "GOOGL",
    "AVGO", "NFLX", "COIN", "MSTR", "PLTR", "SMCI",
]


def build_alpha_universe(dynamic_symbols: list[str] | None = None,
                         size: int | None = None) -> list[str]:
    """Static liquid-optionable core + dynamically discovered high-volume names (deduped)."""
    size = size or int(os.getenv("ALPHA_OPTIONS_UNIVERSE_SIZE", "150"))
    syms = list(dict.fromkeys(
        [s.strip().upper() for s in ALPHA_STATIC_CORE] +
        [s.strip().upper() for s in (dynamic_symbols or []) if s and s.strip()]
    ))
    return syms[:size]


def _regime_risk_on(market_context: dict, config: AlphaCallConfig) -> bool:
    if str(market_context.get("trade_permission") or "normal").lower() == "halt_new_buys":
        return False
    bias = str(market_context.get("overall_market_bias") or market_context.get("market_bias")
               or market_context.get("regime") or "").lower()
    return any(r in bias for r in config.allowed_regimes)


def _alpha_momentum_row(symbol: str, ev, market_context: dict) -> dict:
    return {
        "symbol": symbol,
        "price": round(ev.price, 4),
        "rvol": round(ev.rvol, 3),
        "vwap_state": "above" if ev.above_vwap else "below",
        "orb_state": ("breakout" if ev.orb_high > 0 and ev.price > ev.orb_high
                      else "inside" if ev.orb_high > 0 else "no_orb"),
        "sector_strength": str((market_context.get("sector_strength") or {}).get("sector_rotation_bias")
                               or market_context.get("sector_rotation_bias") or "unknown"),
        "market_regime": str(market_context.get("overall_market_bias") or "unknown"),
        "setup_type": ev.setup_type,
        "grade": ev.grade,
        "momentum_score": ev.setup_score,
    }


def _chain_pass_counts(chain: list[dict], config: AlphaCallConfig,
                       min_delta: float, max_delta: float) -> dict:
    """Independent per-gate pass counts over a chain (for the universe view)."""
    today = datetime.now(ET).date()

    def _dte(c):
        try:
            return (datetime.strptime(str(c.get("expiration_date") or c.get("expiry"))[:10], "%Y-%m-%d").date() - today).days
        except Exception:
            return -999

    def _delta(c):
        return abs(float(c.get("delta") or 0))

    def _mid(c):
        b = float(c.get("bid_price") or c.get("bid") or 0); a = float(c.get("ask_price") or c.get("ask") or 0)
        return (a + b) / 2 if a > 0 and b > 0 else 0.0

    def _spread(c):
        m = _mid(c)
        b = float(c.get("bid_price") or c.get("bid") or 0); a = float(c.get("ask_price") or c.get("ask") or 0)
        return (a - b) / m if m > 0 else 1.0

    calls = [c for c in chain if str(c.get("type", "")).lower() == "call"]
    return {
        "options_chain_count": len(calls),
        "contracts_passing_dte": sum(1 for c in calls if config.min_dte <= _dte(c) <= config.max_dte),
        "contracts_passing_delta": sum(1 for c in calls if min_delta <= _delta(c) <= max_delta),
        "contracts_passing_oi": sum(1 for c in calls if int(c.get("open_interest") or 0) >= config.min_open_interest),
        "contracts_passing_volume": sum(1 for c in calls if int(c.get("volume") or 0) >= config.min_volume),
        "contracts_passing_spread": sum(1 for c in calls if _spread(c) <= config.max_spread_pct),
    }


def _liquidity_score(urow: dict, evalc: dict, config: AlphaCallConfig) -> float:
    """0-100 options-liquidity score from chain depth + best-contract quality."""
    if int(urow.get("options_chain_count") or 0) == 0:
        return 0.0
    qualified = int((evalc.get("counts") or {}).get("qualified", 0))
    best = evalc.get("best") or evalc.get("best_rejected") or {}
    oi = int(best.get("open_interest") or 0)
    spread = float(best.get("spread_pct") or 1.0)
    s = 0.0
    s += 30.0 if qualified > 0 else 0.0                                   # a tradable contract exists
    s += min(25.0, oi / 1000.0 * 25.0)                                    # depth of the best contract
    if spread <= 1.0:
        s += max(0.0, 1.0 - spread / max(config.max_spread_pct, 1e-6)) * 20.0
    s += min(15.0, int(urow.get("contracts_passing_delta") or 0) / 5.0 * 15.0)  # breadth in delta band
    s += min(10.0, int(urow.get("contracts_passing_oi") or 0) / 5.0 * 10.0)     # breadth passing OI
    return _clip(s, 0.0, 100.0)


def screen_alpha_options_universe(
    symbols: list[str],
    bars_by_symbol: dict,
    market_context: dict,
    config: AlphaCallConfig | None = None,
) -> dict:
    """
    I/O-LIGHT momentum + regime pre-screen (bars + context only; NO option chains).
    Returns the 'watch' symbols worth fetching option chains for, plus per-symbol
    momentum rows and a funnel. run.py fetches chains only for the watch list and
    passes them to evaluate_universe() — keeping this engine I/O-light.
    """
    cfg = config or AlphaCallConfig()
    regime_ok = _regime_risk_on(market_context, cfg)
    rows, watch = [], []
    funnel = {"scanned": len(symbols), "regime_ok": regime_ok,
              "passed_momentum": 0, "rejected_by_regime": 0, "rejected_by_momentum": 0}
    for symbol in symbols:
        bars = bars_by_symbol.get(symbol)
        ev = detect_setup(symbol, bars if bars is not None else pd.DataFrame(), cfg, market_context)
        row = _alpha_momentum_row(symbol, ev, market_context)
        if not regime_ok:
            row["watch"] = False
            row["reject_reason"] = f"rejected_by_regime:{row['market_regime']}"
            funnel["rejected_by_regime"] += 1
        elif ev.setup_score >= cfg.screener_momentum_floor and not ev.reject_reason:
            row["watch"] = True
            row["reject_reason"] = ""
            watch.append(symbol)
            funnel["passed_momentum"] += 1
        else:
            row["watch"] = False
            row["reject_reason"] = ev.reject_reason or \
                f"rejected_by_momentum:score={ev.setup_score:.0f}<{cfg.screener_momentum_floor:.0f}"
            funnel["rejected_by_momentum"] += 1
        rows.append(row)
    return {"watch": watch, "rows": rows, "funnel": funnel}


class AlphaCallEngine:
    """
    Primary tactical profit engine using long calls on A/A+ setups.
    Runs as a background thread. Submits through the Alpaca broker passed at init.
    """

    def __init__(self, broker: Any, config: AlphaCallConfig | None = None):
        self.broker = broker
        self.config = config or AlphaCallConfig()
        self._active: dict[str, ActiveCall] = self._load_state()
        self._signals_today: list[dict] = []
        self._prob_cfg = apm.ProbabilityConfig()   # probability-weighted management layer

    # ── state persistence ─────────────────────────────────────────────────────
    def _load_state(self) -> dict[str, ActiveCall]:
        if ALPHA_STATE_PATH.exists():
            try:
                raw = json.loads(ALPHA_STATE_PATH.read_text())
                return {
                    k: ActiveCall(**v) for k, v in raw.items()
                    if all(f in v for f in ActiveCall.__dataclass_fields__)
                }
            except Exception:
                pass
        return {}

    def _save_state(self) -> None:
        try:
            ALPHA_STATE_PATH.write_text(
                json.dumps({k: asdict(v) for k, v in self._active.items()}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _save_signals(self) -> None:
        try:
            ALPHA_SIGNALS_PATH.write_text(
                json.dumps({
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "date_tag": DATA_TAG,
                    "count": len(self._signals_today),
                    "signals": self._signals_today,
                }, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ── position management ───────────────────────────────────────────────────
    def _check_active_positions(
        self,
        options_positions: list[dict],
        waterfall_callback: Any,
        market_context: dict | None = None,
    ) -> list[dict]:
        """
        Manage all open calls using a 3R target / 1.7R trailing stop structure:

          Hard stop:   exit ALL contracts if value drops below entry × (1 – stop_pct)
          +1R partial: sell 50% at +1R profit; remainder runs for 3R
          1.7R trail:  when position exceeds +1.7R, activate trailing stop that
                       sits 0.8R behind the peak (so at 1.7R peak, stop = 0.9R)
          3R target:   exit ALL remaining contracts at full 3R

        Returns exit orders to submit.
        """
        exit_orders = []
        pos_by_contract = {p.get("symbol", ""): p for p in options_positions}

        for key, call in list(self._active.items()):
            pos = pos_by_contract.get(call.contract_symbol)
            if pos is None:
                _alpha_event("call_position_gone", {"contract": call.contract_symbol})
                del self._active[key]
                self._save_state()
                continue

            qty_held = int(pos.get("qty") or call.contracts_held)
            pnl_pct = float(pos.get("unrealized_plpc") or 0)
            current_value = call.entry_cost * (1 + pnl_pct)

            # Update peak
            if current_value > call.peak_value:
                call.peak_value = current_value

            # ── Hard stop ─────────────────────────────────────────────────────
            if current_value <= call.hard_stop_value:
                _pnl = current_value - call.entry_cost
                exit_orders.append({
                    "symbol": call.contract_symbol,
                    "side": "sell",
                    "qty": qty_held,
                    "source": "alpha_call_hard_stop",
                    "reason": f"hard_stop:value={current_value:.0f}<=stop={call.hard_stop_value:.0f}",
                    "underlying": call.symbol, "setup_type": call.setup_type,
                    "realized_pnl": round(_pnl, 2),
                    "r_multiple": round(_pnl / call.entry_cost, 3) if call.entry_cost else 0.0,
                    "probability_band": call.probability_band,
                    "probability_of_profit": call.probability_of_profit,
                    "management_style": call.management_style,
                })
                _alpha_event("call_hard_stop", {
                    "symbol": call.symbol, "contract": call.contract_symbol,
                    "pnl_pct": pnl_pct, "entry_cost": call.entry_cost,
                })
                del self._active[key]; self._save_state(); continue

            # ── 3R full target ────────────────────────────────────────────────
            if current_value >= call.r_target_value:
                _pnl = current_value - call.entry_cost
                exit_orders.append({
                    "symbol": call.contract_symbol,
                    "side": "sell",
                    "qty": qty_held,
                    "source": "alpha_call_3R_target",
                    "reason": f"3R_target:value={current_value:.0f}>=target={call.r_target_value:.0f}",
                    "underlying": call.symbol, "setup_type": call.setup_type,
                    "realized_pnl": round(_pnl, 2),
                    "r_multiple": round(_pnl / call.entry_cost, 3) if call.entry_cost else 0.0,
                    "probability_band": call.probability_band,
                    "probability_of_profit": call.probability_of_profit,
                    "management_style": call.management_style,
                })
                realized = current_value
                _alpha_event("call_3R_exit", {
                    "symbol": call.symbol, "contract": call.contract_symbol,
                    "pnl_pct": pnl_pct, "realized": realized,
                })
                if waterfall_callback:
                    waterfall_callback(realized - call.entry_cost, source="alpha_3R")
                del self._active[key]; self._save_state(); continue

            # ── +1R partial (sell half at first R) ───────────────────────────
            if not call.partial_taken and current_value >= call.partial_value:
                partial_qty = max(1, math.floor(qty_held * self.config.partial_size))
                exit_orders.append({
                    "symbol": call.contract_symbol,
                    "side": "sell",
                    "qty": partial_qty,
                    "source": "alpha_call_1R_partial",
                    "reason": f"+1R_partial:value={current_value:.0f}>=partial={call.partial_value:.0f}",
                })
                call.partial_taken = True
                call.contracts_held = qty_held - partial_qty
                partial_profit = (current_value / call.entry_cost - 1) * (call.entry_cost * self.config.partial_size)
                _alpha_event("call_1R_partial", {
                    "symbol": call.symbol, "contract": call.contract_symbol,
                    "pnl_pct": pnl_pct, "qty_sold": partial_qty,
                    "qty_remaining": call.contracts_held,
                })
                if waterfall_callback:
                    waterfall_callback(partial_profit, source="alpha_partial_1R")
                self._save_state()

            # ── 1.7R trailing stop ────────────────────────────────────────────
            r_amount = call.entry_cost   # 1R = full premium paid
            trail_activate_value = call.entry_cost * (1 + self.config.r_trail_activate)
            if current_value >= trail_activate_value:
                call.trail_active = True
                # Trail stop sits (r_trail_offset × trail_tightness) × R below the peak.
                # trail_tightness_mult <= 1.0 ⇒ stop sits CLOSER to the peak (more
                # protective). Probability can only tighten this, never loosen it.
                offset_mult = min(getattr(call, "trail_tightness_mult", 1.0) or 1.0, 1.0)
                trail_stop = call.peak_value - (self.config.r_trail_offset * offset_mult * r_amount)
                # Never lower the trail stop once set
                if trail_stop > call.trail_stop_value:
                    call.trail_stop_value = trail_stop
                self._save_state()

            if call.trail_active and call.trail_stop_value > 0 and current_value <= call.trail_stop_value:
                remaining_qty = call.contracts_held
                _pnl = current_value - call.entry_cost
                exit_orders.append({
                    "symbol": call.contract_symbol,
                    "side": "sell",
                    "qty": remaining_qty,
                    "source": "alpha_call_trail_stop",
                    "reason": (
                        f"trail_stop:value={current_value:.0f}"
                        f"<=trail={call.trail_stop_value:.0f} peak={call.peak_value:.0f}"
                    ),
                    "underlying": call.symbol, "setup_type": call.setup_type,
                    "realized_pnl": round(_pnl, 2),
                    "r_multiple": round(_pnl / call.entry_cost, 3) if call.entry_cost else 0.0,
                    "probability_band": call.probability_band,
                    "probability_of_profit": call.probability_of_profit,
                    "management_style": call.management_style,
                })
                realized_gain = current_value - call.entry_cost * (1 - self.config.partial_size if call.partial_taken else 1)
                _alpha_event("call_trail_stop_hit", {
                    "symbol": call.symbol, "contract": call.contract_symbol,
                    "pnl_pct": pnl_pct, "peak_value": call.peak_value,
                    "trail_stop": call.trail_stop_value,
                })
                if waterfall_callback and realized_gain > 0:
                    waterfall_callback(realized_gain, source="alpha_trail")
                del self._active[key]; self._save_state()

            # ── Black-Scholes theta-decay exit (advisory; hard rules above take priority) ──
            if key in self._active and self.config.bs_exit_optimizer_enabled:
                try:
                    dte_rem = (datetime.strptime(str(call.expiry)[:10], "%Y-%m-%d").date()
                               - datetime.now(ET).date()).days
                except Exception:
                    dte_rem = 999
                guide = self.black_scholes_exit_optimizer(call, current_value, dte_rem)
                if guide.get("action") in ("exit_theta_decay", "exit_iv_crush"):
                    _pnl = current_value - call.entry_cost
                    exit_orders.append({
                        "symbol": call.contract_symbol, "side": "sell", "qty": call.contracts_held,
                        "source": f"alpha_call_{guide.get('action', 'theta_decay')}",
                        "reason": guide.get("rationale", "theta_decay"),
                        "underlying": call.symbol, "setup_type": call.setup_type,
                        "realized_pnl": round(_pnl, 2),
                        "r_multiple": round(_pnl / call.entry_cost, 3) if call.entry_cost else 0.0,
                    })
                    _alpha_event("call_theta_decay_exit", {
                        "symbol": call.symbol, "contract": call.contract_symbol,
                        "dte_remaining": dte_rem, "value": current_value,
                        "rationale": guide.get("rationale"),
                    })
                    realized = current_value - call.entry_cost
                    if waterfall_callback and realized > 0:
                        waterfall_callback(realized, source="alpha_theta_exit")
                    del self._active[key]; self._save_state()

        return exit_orders

    # ── scan and signal ───────────────────────────────────────────────────────
    def evaluate_universe(
        self,
        symbols: list[str],
        bars_by_symbol: dict[str, pd.DataFrame],
        market_context: dict,
        equity: float,
        options_exposure: float,
        waterfall_budget: float,
        chains_by_symbol: dict[str, list] | None = None,
        positions: dict | None = None,
        account_state: dict | None = None,
    ) -> list[CallSignal]:
        """
        Full Alpha pipeline on PRE-FETCHED data (I/O-light):
            1. alpha options universe screening (momentum + regime)
            2. setup scoring
            3. options-chain liquidity filtering
            4. Black-Scholes contract scoring
            5. entry decision -> order intents
        Writes daily_system_outputs/alpha_options_universe.json. Returns ordered
        call-signal intents; the caller (run.py) routes them through RiskManager/
        AlpacaGateway. Signals are NOT submitted here.

        chains_by_symbol: pre-fetched {symbol: [contract dicts]} supplied by run.py/
        AlpacaGateway. If omitted, falls back to self.broker.get_options_chain (kept
        only for back-compat); the I/O-light path always passes chains in.
        """
        rfr = resolve_risk_free_rate(self.config, market_context)
        scfg = replace(self.config, min_delta=self.config.screener_min_delta,
                       max_delta=self.config.screener_max_delta)
        chains_by_symbol = chains_by_symbol or {}
        universe_rows: list[dict] = []
        signals: list[CallSignal] = []
        vol_rows: list[dict] = []
        # Empirical per-setup history for the probability-weighted management layer.
        setup_history = {}
        if self._prob_cfg.enabled:
            try:
                setup_history = apm.load_setup_history()
            except Exception:
                setup_history = {}

        # Engine-level pre-gates. A universe file is still written (so the screener
        # output is always produced) but no entries are emitted when blocked.
        exposure_blocked = options_exposure >= equity * self.config.max_options_exposure_pct
        positions_full = len(self._active) >= self.config.max_concurrent_positions
        entries_enabled = self.config.enabled and not exposure_blocked and not positions_full
        if exposure_blocked:
            _alpha_event("options_exposure_limit", {"exposure": options_exposure,
                "limit": equity * self.config.max_options_exposure_pct, "reason": "rejected_by_exposure_limit"})
        if positions_full:
            _alpha_event("max_positions_reached", {"count": len(self._active)})

        for symbol in symbols:
            bars = bars_by_symbol.get(symbol)
            if bars is None:
                bars = pd.DataFrame()
            ev = detect_setup(symbol, bars, self.config, market_context)
            urow = _alpha_momentum_row(symbol, ev, market_context)

            # Resolve the PRE-FETCHED chain (fall back to broker only if not supplied).
            chain = chains_by_symbol.get(symbol)
            if chain is None:
                try:
                    chain = self.broker.get_options_chain(symbol) if hasattr(self.broker, "get_options_chain") else []
                except Exception as exc:
                    _alpha_event("chain_fetch_error", {"symbol": symbol, "error": str(exc)[:200]})
                    chain = []
            chain = chain or []

            # Universe-view liquidity (screener-wide delta band 0.35-0.80).
            urow.update(_chain_pass_counts(chain, self.config,
                                           self.config.screener_min_delta, self.config.screener_max_delta))
            view = evaluate_call_contracts(symbol, ev.price, chain, scfg, rfr, ev.rvol) if chain else \
                {"best": None, "best_rejected": None, "counts": {"qualified": 0}, "best_black_scholes_score": 0.0}
            bview = view.get("best") or view.get("best_rejected")
            urow["best_contract"] = ({
                "contract": bview.get("contract_symbol"), "dte": bview.get("dte"), "delta": bview.get("delta"),
                "open_interest": bview.get("open_interest"), "spread_pct": bview.get("spread_pct"),
                "black_scholes_score": bview.get("black_scholes_score"), "mispricing_pct": bview.get("mispricing_pct"),
            } if bview else None)
            liq = _liquidity_score(urow, view, self.config)
            urow["liquidity_score"] = round(liq, 1)
            urow["alpha_screener_score"] = round(0.6 * liq + 0.4 * float(ev.setup_score), 1)
            urow["reject_reason"] = ""

            # ── volatility diagnostics (refine contract quality + exit; not directional) ──
            vol_diag = (compute_volatility_diagnostics(
                symbol, ev.price, chain, bars, self.config, market_context, selected_contract=bview)
                if self.config.vol_diagnostics_enabled else {})
            urow["volatility"] = {k: vol_diag.get(k) for k in (
                "expected_move", "iv_rank", "iv_percentile", "iv_term_structure",
                "gamma_exposure", "gamma_theta_efficiency", "iv_crush_risk",
                "volatility_regime_score")} if vol_diag else {}
            if vol_diag:
                vol_rows.append({"symbol": symbol, "setup_score": round(float(ev.setup_score), 1), **vol_diag})

            # ── entry decision (engine authority) ───────────────────────────────
            if not entries_enabled:
                urow["reject_reason"] = "entries_disabled"
                universe_rows.append(urow); continue
            if ev.setup_score < self.config.min_setup_score or ev.reject_reason:
                urow["reject_reason"] = ev.reject_reason or \
                    f"rejected_by_setup_quality:score={ev.setup_score:.0f}<{self.config.min_setup_score:.0f}"
                _alpha_event("setup_rejected", {"symbol": symbol, "score": ev.setup_score, "reason": urow["reject_reason"]})
                universe_rows.append(urow); continue
            if symbol in self._active:
                urow["reject_reason"] = "already_in_position"
                universe_rows.append(urow); continue

            # Authoritative contract selection with the ENGINE's tight band (0.45-0.70).
            engine_eval = evaluate_call_contracts(symbol, ev.price, chain, self.config, rfr, ev.rvol)
            contract = engine_eval["best"]
            sig = CallSignal(symbol=symbol, setup_type=ev.setup_type, setup_score=ev.setup_score,
                             grade=ev.grade, signal_at_utc=datetime.now(timezone.utc).isoformat())
            _apply_vol_fields(sig, vol_diag)

            if contract is None:
                br = engine_eval.get("best_rejected")
                if br is not None:
                    sig.reject_reason = "rejected_by_options_liquidity:no_qualifying_contract:" \
                        f"{br.get('bs_reject_reason', 'bs')}"
                    _apply_bs_fields(sig, br); sig.contract_symbol = br.get("contract_symbol", "")
                    _alpha_event("contract_rejected_by_valuation", {"symbol": symbol,
                        "best_rejected_contract": br.get("contract_symbol"), "bs_reject_reason": br.get("bs_reject_reason"),
                        "black_scholes_score": br.get("black_scholes_score"), "counts": engine_eval.get("counts")})
                else:
                    sig.reject_reason = "rejected_by_options_liquidity:no_qualifying_contract"
                    _alpha_event("contract_rejected", {"symbol": symbol, "reason": sig.reject_reason,
                        "counts": engine_eval.get("counts")})
                urow["reject_reason"] = sig.reject_reason
                universe_rows.append(urow); signals.append(sig); continue

            sig.contract_symbol = contract["contract_symbol"]; sig.expiry = contract["expiry"]
            sig.strike = contract["strike"]; sig.delta = contract["delta"]; sig.dte = contract["dte"]
            sig.bid = contract["bid"]; sig.ask = contract["ask"]; sig.mid = contract["mid"]
            sig.open_interest = contract["open_interest"]; sig.volume = contract["volume"]
            _apply_bs_fields(sig, contract)

            available = min(equity * self.config.risk_per_trade_pct, waterfall_budget) \
                if waterfall_budget > 0 else equity * self.config.risk_per_trade_pct
            contracts, premium, max_risk = size_position(equity, sig.mid, self.config)
            if contracts < 1 or premium > available:
                sig.reject_reason = f"rejected_by_cash_buffer:premium={premium:.0f}>available={available:.0f}"
                urow["reject_reason"] = sig.reject_reason
                universe_rows.append(urow); signals.append(sig); continue

            # ── Probability-Weighted Management Layer ───────────────────────────
            # Adapts SIZE + MANAGEMENT to estimated probability of profit. It NEVER
            # rejects the trade and only moves in the protective direction (reduce size,
            # tighten stop/trail, earlier partial). RiskManager still makes final approval.
            stop_pct_eff = self.config.stop_loss_pct
            partial_pct_eff = self.config.partial_profit_pct
            if self._prob_cfg.enabled:
                try:
                    prob = apm.evaluate(
                        ev.setup_type, ev.setup_score, ev.grade,
                        expected_r_bs=float(contract.get("expected_r_multiple", 1.0) or 1.0),
                        bs_score=float(sig.black_scholes_score or 0.0),
                        iv_crush_risk=float(getattr(sig, "iv_crush_risk", 0.0) or 0.0),
                        setup_history=setup_history, config=self._prob_cfg)
                    # Reduce size only — clamp to ≥1 contract so probability NEVER blocks.
                    contracts = max(1, min(contracts, int(round(contracts * prob["size_multiplier"]))))
                    premium = contracts * sig.mid * 100.0
                    max_risk = premium
                    stop_pct_eff = self.config.stop_loss_pct * prob["stop_tightness"]          # ≤ base (tighter)
                    partial_pct_eff = min(self.config.partial_profit_pct,
                                          prob["partial_profit_trigger"])                      # ≤ base (earlier)
                    sig.probability_of_profit = prob["probability_of_profit"]
                    sig.expected_R = prob["expected_R"]
                    sig.probability_confidence = prob["confidence"]
                    sig.probability_source = prob["source"]
                    sig.probability_band = prob["probability_band"]
                    sig.probability_size_multiplier = prob["size_multiplier"]
                    sig.management_style = prob["management_style"]
                    sig.partial_profit_trigger = prob["partial_profit_trigger"]
                    sig.trailing_tightness = prob["trailing_tightness"]
                    sig.overnight_allowed_by_probability = bool(prob["overnight_allowed"])
                    sig.trail_tightness_mult = prob["trail_tightness_mult"]
                except Exception as _pe:
                    _alpha_event("probability_layer_error", {"symbol": symbol, "error": str(_pe)[:160]})
            sig.stop_loss_pct_eff = stop_pct_eff
            sig.partial_profit_pct_eff = partial_pct_eff

            sig.contracts = contracts; sig.premium_cost = premium; sig.max_risk = max_risk
            sig.stop_value = premium * (1 - stop_pct_eff)
            sig.partial_value = premium * (1 + partial_pct_eff)
            sig.status = "ready"
            universe_rows.append(urow); signals.append(sig)
            _alpha_event("call_signal_generated", {"symbol": symbol, "setup": ev.setup_type, "grade": ev.grade,
                "score": ev.setup_score, "contract": sig.contract_symbol, "delta": sig.delta, "dte": sig.dte,
                "mid": sig.mid, "contracts": sig.contracts, "premium": sig.premium_cost})

        signals.sort(key=lambda s: s.setup_score, reverse=True)
        self._signals_today.extend([asdict(s) for s in signals if s.status == "ready"])
        self._save_signals()
        self._save_alpha_universe(universe_rows, market_context)
        self._save_volatility_diagnostic(vol_rows, market_context)
        return signals

    def _save_volatility_diagnostic(self, rows: list[dict], market_context: dict) -> None:
        """Write daily_system_outputs/alpha_volatility_diagnostic.json (per-symbol vol view)."""
        try:
            ALPHA_VOL_DIAGNOSTIC_PATH.write_text(json.dumps({
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "date_tag": DATA_TAG,
                "market_regime": str(market_context.get("overall_market_bias")),
                "volatility_regime": str(market_context.get("volatility_regime")
                                         or market_context.get("vix_regime") or ""),
                "count": len(rows),
                "diagnostics": rows,
            }, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass

    def _save_alpha_universe(self, rows: list[dict], market_context: dict) -> None:
        """Write daily_system_outputs/alpha_options_universe.json (the screener view)."""
        quality_prefixes = ("rejected_by_regime", "rejected_by_momentum",
                            "rejected_by_setup", "rejected_by_options_liquidity")
        for r in rows:
            rr = str(r.get("reject_reason") or "")
            r["is_candidate"] = (float(r.get("alpha_screener_score") or 0) >= self.config.screener_min_score
                                 and not any(rr.startswith(p) for p in quality_prefixes))
        candidates = [r for r in rows if r.get("is_candidate")]
        candidates.sort(key=lambda r: r.get("alpha_screener_score", 0), reverse=True)
        try:
            ALPHA_OPTIONS_UNIVERSE_PATH.write_text(json.dumps({
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "date_tag": DATA_TAG,
                "market_regime": str(market_context.get("overall_market_bias")),
                "screener": {
                    "min_score": self.config.screener_min_score,
                    "min_delta": self.config.screener_min_delta,
                    "max_delta": self.config.screener_max_delta,
                    "engine_delta_band": [self.config.min_delta, self.config.max_delta],
                },
                "scanned": len(rows),
                "candidate_count": len(candidates),
                "candidates": candidates,
                "all_scanned": rows,
            }, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass

    # ── public API (run.py calls only these high-level methods) ────────────────
    def screen_alpha_options_universe(self, symbols: list[str], bars_by_symbol: dict,
                                      market_context: dict) -> dict:
        """I/O-light momentum/regime pre-screen. Returns watch symbols + rows + funnel."""
        return screen_alpha_options_universe(symbols, bars_by_symbol, market_context, self.config)

    def check_active_positions(self, options_positions: list[dict],
                               waterfall_callback: Any = None,
                               market_context: dict | None = None) -> list[dict]:
        """
        Public EXIT pass. Manages all open calls (hard stop / +partial / 1.7R trail /
        3R target) and an optional Black-Scholes theta-decay exit. Returns exit orders.
        Must be called BEFORE evaluate_universe() each cycle (exit-before-buy).
        """
        return self._check_active_positions(options_positions, waterfall_callback,
                                            market_context=market_context)

    def select_best_contract(self, symbol: str, price: float, chain: list[dict],
                             market_context: dict | None = None) -> Optional[dict]:
        """Public wrapper around the Black-Scholes contract selector (engine delta band)."""
        rfr = resolve_risk_free_rate(self.config, market_context or {})
        return select_call_contract(symbol, price, chain, self.config, risk_free_rate=rfr)

    def black_scholes_exit_optimizer(self, call: "ActiveCall", current_value: float,
                                     dte_remaining: int, vol_diag: dict | None = None) -> dict:
        """
        Black-Scholes exit guidance for an open call (advisory; the hard 3R / stop /
        trail rules remain authoritative and are checked first). Recommends an early
        theta-decay exit when the contract is close to expiry and NOT yet profitable
        enough to justify holding through accelerating decay, or an IV-crush exit when
        volatility diagnostics flag a high overnight crush risk on a non-profitable hold.

        Returns {"action": "hold"|"exit_theta_decay"|"exit_iv_crush", "rationale", ...}.
        """
        if not self.config.bs_exit_optimizer_enabled:
            return {"action": "hold", "rationale": "bs_exit_optimizer_disabled"}
        # Convexity remaining vs decay: once near expiry and below the +partial level,
        # remaining extrinsic value bleeds fast — exit rather than donate theta.
        below_partial = current_value < call.partial_value
        near_expiry = dte_remaining <= self.config.bs_exit_min_dte
        if near_expiry and below_partial:
            return {
                "action": "exit_theta_decay",
                "rationale": f"dte={dte_remaining}<= {self.config.bs_exit_min_dte} and "
                             f"value={current_value:.0f}<partial={call.partial_value:.0f}",
                "dte_remaining": dte_remaining,
            }
        # IV-crush guard (advisory): high crush risk on a not-yet-profitable position
        # reduces the overnight-hold score below the exit threshold.
        if vol_diag and self.config.vol_diagnostics_enabled and below_partial:
            crush = float(vol_diag.get("iv_crush_risk", 0.0) or 0.0)
            if crush >= self.config.vol_iv_crush_exit_threshold:
                hold_score = overnight_hold_score(60.0, vol_diag, self.config)
                if hold_score < 40.0:
                    return {
                        "action": "exit_iv_crush",
                        "rationale": f"iv_crush_risk={crush:.0f}>= {self.config.vol_iv_crush_exit_threshold:.0f}, "
                                     f"overnight_hold_score={hold_score:.0f}",
                        "iv_crush_risk": crush, "overnight_hold_score": hold_score,
                    }
        return {"action": "hold", "rationale": "convexity_intact" if not near_expiry else "in_profit_zone"}

    def register_filled(self, sig: CallSignal) -> None:
        """Call after a call order is confirmed filled to track it."""
        cost = sig.premium_cost
        r = cost  # 1R = full premium paid
        key = sig.symbol
        self._active[key] = ActiveCall(
            symbol=sig.symbol,
            contract_symbol=sig.contract_symbol,
            expiry=sig.expiry,
            strike=sig.strike,
            contracts_held=sig.contracts,
            contracts_original=sig.contracts,
            entry_mid=sig.mid,
            entry_cost=cost,
            # Probability layer may only TIGHTEN these (protective). Falls back to config.
            hard_stop_value=cost * (1 - (sig.stop_loss_pct_eff or self.config.stop_loss_pct)),
            r_target_value=cost * (1 + self.config.r_target),      # 3R exit (never weakened)
            partial_value=cost * (1 + (sig.partial_profit_pct_eff or self.config.partial_profit_pct)),
            peak_value=cost,
            trail_stop_value=0.0,
            opened_at_utc=datetime.now(timezone.utc).isoformat(),
            setup_type=sig.setup_type,
            setup_score=sig.setup_score,
            management_style=getattr(sig, "management_style", "standard"),
            trail_tightness_mult=getattr(sig, "trail_tightness_mult", 1.0) or 1.0,
            overnight_allowed=bool(getattr(sig, "overnight_allowed_by_probability", False)),
            probability_of_profit=float(getattr(sig, "probability_of_profit", 0.0) or 0.0),
            probability_band=str(getattr(sig, "probability_band", "") or ""),
        )
        self._save_state()
        _alpha_event("call_position_opened", {
            "symbol": sig.symbol,
            "contract": sig.contract_symbol,
            "contracts": sig.contracts,
            "premium": cost,
            "hard_stop": cost * (1 - self.config.stop_loss_pct),
            "trail_activates_at": cost * (1 + self.config.r_trail_activate),
            "target_3R": cost * (1 + self.config.r_target),
            "setup": sig.setup_type,
            "grade": sig.grade,
        })
