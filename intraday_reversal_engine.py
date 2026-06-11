#!/usr/bin/env python3
"""Intraday Reversal Engine — isolated, optional candidate source.

Thesis
------
Buy fundamentally-liked names (analyst + oscillator "strong buy") that are WEAK
over the last week and RED today, but are showing genuine intraday
reversal/reclaim behavior right now: elevated RVOL, recovering off the lows,
reclaiming VWAP, RSI turning up / crossing 40, a liquidity sweep that gets
reclaimed, an opening-range reclaim, and price back above the 5m EMA20 — while
NOT being over-extended from VWAP.

Design constraints (by request)
-------------------------------
* This module does NOT modify ``run.py``.  It is a self-contained candidate
  source: ``scan()`` is a pure scanner+scorer with NO Alpaca side effects, and
  ``propose_entries()`` returns intended orders that the caller routes through
  the EXISTING ``run.AlpacaGateway.submit_market_order`` chokepoint — which is
  where every existing safety gate (market-regime, PDT, exposure, cash-buffer,
  max-positions, duplicate-symbol, account-mode) is enforced.  This engine only
  ADDS its own gates (min-score, engine max-open, no-averaging, paper-only); it
  never bypasses the substrate.
* Exits remain the responsibility of the existing exit engine.  This module only
  proposes BUY entries.
* Paper-only by default; live orders require the system to be safely live AND
  ``INTRADAY_REVERSAL_PAPER_ONLY=false``.

Env flags
---------
  INTRADAY_REVERSAL_ENABLED=false
  INTRADAY_REVERSAL_PAPER_ONLY=true
  INTRADAY_REVERSAL_MIN_SCORE=75
  INTRADAY_REVERSAL_MAX_NEW_ENTRIES_PER_DAY=2
  INTRADAY_REVERSAL_MAX_OPEN_POSITIONS=2

CLI
---
  python intraday_reversal_engine.py --demo      # dry-run: print ranked candidates, NO orders
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("intraday_reversal_engine")

ET = ZoneInfo("America/New_York")
STRONG_BUY = "strong buy"


# ---------------------------------------------------------------------------
# env helpers
# ---------------------------------------------------------------------------
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v in (None, ""):
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass
class EngineConfig:
    enabled: bool = field(default_factory=lambda: _env_bool("INTRADAY_REVERSAL_ENABLED", False))
    paper_only: bool = field(default_factory=lambda: _env_bool("INTRADAY_REVERSAL_PAPER_ONLY", True))
    min_score: float = field(default_factory=lambda: _env_float("INTRADAY_REVERSAL_MIN_SCORE", 75.0))
    max_new_entries_per_day: int = field(
        default_factory=lambda: _env_int("INTRADAY_REVERSAL_MAX_NEW_ENTRIES_PER_DAY", 2))
    max_open_positions: int = field(
        default_factory=lambda: _env_int("INTRADAY_REVERSAL_MAX_OPEN_POSITIONS", 2))
    # universe thresholds (scanner). Gates: price>5, vol>1M, rel_vol_5m>1 (+ chg>0,
    # perf_1w<0, analyst/os strong buy). 3M volume is a SCORE bonus, not a gate.
    min_price: float = 5.0
    min_volume: float = field(default_factory=lambda: _env_float("INTRADAY_REVERSAL_MIN_VOLUME", 1_000_000.0))
    min_rel_vol_5m: float = field(default_factory=lambda: _env_float("INTRADAY_REVERSAL_MIN_REL_VOL_5M", 1.0))
    volume_bonus_threshold: float = 3_000_000.0   # +10 score above this
    pref_avg_volume: float = 1_000_000.0          # +5 score above this
    # sizing fallback
    risk_per_trade_pct: float = 0.005   # <= 0.5% equity at risk
    max_notional_pct: float = 0.10      # <= 10% equity per position
    atr_stop_mult: float = 1.0          # stop = entry - atr_stop_mult * ATR
    marketable_buffer: float = 0.001    # marketable BUY limit = price * (1 + buffer)
    extended_atr_limit: float = 2.0     # penalize beyond this many ATR from VWAP
    # ── 1-7 DTE call-option overlay (target = underlying Previous Day High) ──────────
    options_enabled: bool = field(
        default_factory=lambda: _env_bool("INTRADAY_REVERSAL_OPTIONS_ENABLED", False))
    option_min_dte: int = field(default_factory=lambda: _env_int("INTRADAY_REVERSAL_OPTION_MIN_DTE", 1))
    option_max_dte: int = field(default_factory=lambda: _env_int("INTRADAY_REVERSAL_OPTION_MAX_DTE", 7))
    option_strike_offset_pct: float = field(            # 0 = ATM, >0 = OTM strike target
        default_factory=lambda: _env_float("INTRADAY_REVERSAL_OPTION_STRIKE_OFFSET_PCT", 0.0))
    option_max_spread_pct: float = field(               # reject wide bid/ask
        default_factory=lambda: _env_float("INTRADAY_REVERSAL_OPTION_MAX_SPREAD_PCT", 0.20))
    option_min_open_interest: int = field(
        default_factory=lambda: _env_int("INTRADAY_REVERSAL_OPTION_MIN_OI", 100))
    option_risk_pct: float = field(                     # premium-at-risk per trade (long call max loss = premium)
        default_factory=lambda: _env_float("INTRADAY_REVERSAL_OPTION_RISK_PCT", 0.005))
    option_marketable_buffer: float = 0.02              # buy this far over mid when no ask
    # Trailing stop on the UNDERLYING (high-water-mark trail). Recorded on every proposed
    # entry; the existing exit engine is responsible for executing it. Target = PDH (above);
    # trailing stop = this % below the post-entry peak (below).
    trailing_stop_pct: float = field(
        default_factory=lambda: _env_float("INTRADAY_REVERSAL_TRAILING_STOP_PCT", 0.0075))
    # ── Scheduling window + PDH-approach + overnight management ──────────────────────
    start_et: str = field(default_factory=lambda: os.getenv("INTRADAY_REVERSAL_START_ET", "13:30"))
    entry_cutoff_et: str = field(
        default_factory=lambda: os.getenv("INTRADAY_REVERSAL_ENTRY_CUTOFF_ET", "15:55"))
    toward_pdh_max_atr: float = field(            # PDH "reachable" within this many ATR of price
        default_factory=lambda: _env_float("INTRADAY_REVERSAL_TOWARD_PDH_MAX_ATR", 6.0))
    carry_overnight_if_up: bool = field(
        default_factory=lambda: _env_bool("INTRADAY_REVERSAL_CARRY_OVERNIGHT", True))
    overnight_trailing_stop_pct: float = field(  # TIGHTENED trail for an overnight carry
        default_factory=lambda: _env_float("INTRADAY_REVERSAL_OVERNIGHT_TRAIL_PCT", 0.004))
    next_day_high_trail_pct: float = field(      # day-2 trail to sell into the next day's high
        default_factory=lambda: _env_float("INTRADAY_REVERSAL_NEXT_DAY_TRAIL_PCT", 0.004))


def live_orders_allowed(cfg: EngineConfig, account_mode: str, system_paper_only: bool) -> bool:
    """Live orders ONLY when the engine is not paper-locked AND the surrounding
    system is genuinely live (mode==live and the system is not itself paper-only)."""
    if cfg.paper_only:
        return False
    return str(account_mode or "").strip().lower() == "live" and not bool(system_paper_only)


def _parse_hhmm(value: str, default_h: int, default_m: int):
    from datetime import time as _t
    try:
        hh, mm = (int(p) for p in str(value).split(":")[:2])
        return _t(hh, mm)
    except Exception:
        return _t(default_h, default_m)


def is_active_window(now_et, cfg: EngineConfig) -> bool:
    """True only inside the afternoon entry window: now_et in [start_et, entry_cutoff_et)
    on a weekday. This is the scheduling gate — the engine selects fresh entries only
    after ~1:30 PM ET (when the day's range and the PDH-approach are well formed)."""
    if now_et is None:
        from datetime import datetime as _dt
        now_et = _dt.now(ET)
    if now_et.weekday() >= 5:                      # Sat/Sun
        return False
    t = now_et.timetz().replace(tzinfo=None)
    return _parse_hhmm(cfg.start_et, 13, 30) <= t < _parse_hhmm(cfg.entry_cutoff_et, 15, 55)


# ---------------------------------------------------------------------------
# input / output models
# ---------------------------------------------------------------------------
@dataclass
class ReversalInput:
    """One symbol's snapshot. ``bars``/``daily_history`` are optional OHLCV
    DataFrames (time-ordered). Any explicit metric below that is left None is
    derived from the bars; setting it overrides the bar-derived value (handy for
    upstream feeds that pre-compute, and for deterministic tests)."""
    symbol: str
    price: float
    volume: float
    avg_volume: float
    chg_pct: float            # today's % change (negative == red)
    perf_1w: float            # 1-week performance %
    analyst_rating: str
    os_rating: str            # oscillator / technical "strong buy"
    tradable: bool = True
    halted: bool = False
    bars: Any = None          # intraday OHLCV (open/high/low/close/volume)
    daily_history: Any = None # daily OHLCV (for ATR)
    # optional precomputed / override metrics
    rvol: Optional[float] = None
    vol_5m: Optional[float] = None       # last-5-minute volume (universe sort key); else last 5 bars
    rel_vol_5m: Optional[float] = None   # 5-min relative volume (gate > 1); else derived
    vwap_reclaim: Optional[bool] = None
    rsi_rising: Optional[bool] = None
    rsi_cross_40: Optional[bool] = None
    close_gt_open: Optional[bool] = None
    sweep_reclaim: Optional[bool] = None
    orb_reclaim: Optional[bool] = None
    above_ema20_5m: Optional[bool] = None
    recovering: Optional[bool] = None
    extended_atr: Optional[float] = None   # distance of price above VWAP, in ATR
    near_low_atr: Optional[float] = None   # distance of price above day-low, in ATR
    atr: Optional[float] = None
    # Options overlay inputs
    prev_day_high: Optional[float] = None  # PDH (target). If None, derived from daily_history.
    option_chain: Optional[list] = None    # list of dicts: {symbol?, expiry, strike, type,
    #   bid, ask, mid?, dte?, open_interest, volume} — calls used for the 1-7 DTE overlay.


@dataclass
class ScoredCandidate:
    symbol: str
    score: float
    volume: float
    price: float
    passed_universe: bool
    passed_setup: bool
    eligible: bool                       # passed_setup AND score >= min_score
    blocked_by: Optional[str]
    reasons: list[str] = field(default_factory=list)
    score_components: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    prev_day_high: Optional[float] = None         # PDH target (carried from input)
    option_chain: Optional[list] = None           # carried for the options overlay
    price_now: float = 0.0
    vol_5m: Optional[float] = None                # last-5-minute volume (universe sort key)


@dataclass
class EntryOrder:
    symbol: str
    side: str
    qty: int
    limit_price: float
    order_type: str
    score: float
    reason: str
    target_underlying: Optional[float] = None   # PDH price target
    trailing_stop_pct: Optional[float] = None    # trailing stop on the underlying (high-water trail)


@dataclass
class OptionEntryOrder:
    occ_symbol: str
    underlying: str
    side: str                 # "buy"
    qty: int                  # number of contracts
    limit_price: float        # marketable per-contract premium
    order_type: str           # "limit"
    strike: float
    expiry: str
    dte: int
    est_premium: float        # mid at proposal
    target_underlying: float  # PDH (the price target on the underlying)
    est_target_premium: float # rough first-order premium when underlying reaches PDH
    est_target_multiple: float
    trailing_stop_pct: float  # trailing stop on the UNDERLYING (high-water trail)
    score: float
    reason: str


# ---------------------------------------------------------------------------
# indicators (pure; tolerant of short / missing series)
# ---------------------------------------------------------------------------
def _col(bars, name):
    try:
        return bars[name]
    except Exception:
        return None


def vwap_series(bars):
    try:
        tp = (bars["high"] + bars["low"] + bars["close"]) / 3.0
        vol = bars["volume"].astype(float)
        cum_v = vol.cumsum()
        return (tp * vol).cumsum() / cum_v.replace(0, float("nan"))
    except Exception:
        return None


def rsi_series(close, period: int = 14):
    try:
        import pandas as pd  # local import keeps module import light
        c = pd.Series(list(close), dtype="float64")
        if len(c) < 3:
            return None
        p = min(period, max(2, len(c) - 1))
        delta = c.diff()
        gain = delta.clip(lower=0.0).rolling(p).mean()
        loss = (-delta.clip(upper=0.0)).rolling(p).mean()
        rs = gain / loss.replace(0, float("nan"))
        return 100.0 - (100.0 / (1.0 + rs))
    except Exception:
        return None


def ema_series(values, span: int):
    try:
        import pandas as pd
        return pd.Series(list(values), dtype="float64").ewm(span=span, adjust=False).mean()
    except Exception:
        return None


def atr_from_daily(daily_history, period: int = 14) -> Optional[float]:
    try:
        h, l, c = daily_history["high"], daily_history["low"], daily_history["close"]
        pc = c.shift(1)
        tr = (h - l).combine((h - pc).abs(), max).combine((l - pc).abs(), max)
        if len(tr.dropna()) < 2:
            return None
        return float(tr.tail(period).mean())
    except Exception:
        return None


def _finite(x, d=0.0):
    try:
        x = float(x)
        return d if (math.isnan(x) or math.isinf(x)) else x
    except Exception:
        return d


def _vol_5m_of(inp: "ReversalInput") -> Optional[float]:
    if inp.vol_5m is not None:
        return _finite(inp.vol_5m)
    if inp.bars is not None and _safe_len(inp.bars) >= 1:
        try:
            return float(inp.bars["volume"].tail(5).sum())
        except Exception:
            return None
    return None


def rel_vol_5m_of(inp: "ReversalInput") -> float:
    """5-minute relative volume = last-5-min volume / the average 5-min volume
    (avg daily volume / 78 five-min RTH bars). Prefers an explicit rel_vol_5m;
    falls back to the day RVOL when 5-min volume is unavailable."""
    if inp.rel_vol_5m is not None:
        return _finite(inp.rel_vol_5m)
    v5 = _vol_5m_of(inp)
    if v5 is not None and inp.avg_volume > 0:
        per_5m = inp.avg_volume / 78.0
        return v5 / per_5m if per_5m > 0 else 0.0
    return _finite(inp.rvol)


def previous_day_high(inp: "ReversalInput") -> Optional[float]:
    """The underlying's Previous Day High (the call-overlay price target).

    Prefers the explicit ``prev_day_high``; otherwise the high of the last row of
    ``daily_history`` (which is expected to end on the PRIOR completed session — a
    live intraday feed excludes today's forming bar)."""
    if inp.prev_day_high is not None:
        return _finite(inp.prev_day_high)
    dh = inp.daily_history
    try:
        if dh is not None and _safe_len(dh) >= 1:
            return float(dh["high"].iloc[-1])
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# setup-metric derivation from bars (only for fields not explicitly provided)
# ---------------------------------------------------------------------------
def derive_metrics(inp: ReversalInput, cfg: EngineConfig) -> dict[str, Any]:
    bars = inp.bars
    m: dict[str, Any] = {}

    # RVOL: explicit, else volume / avg_volume.
    m["rvol"] = (inp.rvol if inp.rvol is not None
                 else (inp.volume / inp.avg_volume if inp.avg_volume > 0 else 0.0))

    # 5-minute volume (universe sort key) + 5-min relative volume (universe gate >1).
    m["vol_5m"] = _vol_5m_of(inp)
    m["rel_vol_5m"] = round(rel_vol_5m_of(inp), 3)

    # ATR: explicit, else daily ATR, else intraday mean range, else price proxy.
    atr = inp.atr
    if atr is None:
        atr = atr_from_daily(inp.daily_history) if inp.daily_history is not None else None
    if (atr is None or atr <= 0) and bars is not None:
        try:
            atr = float((bars["high"] - bars["low"]).tail(14).mean())
        except Exception:
            atr = None
    if atr is None or atr <= 0:
        atr = max(0.01, inp.price * 0.01)
    m["atr"] = float(atr)

    have_bars = bars is not None and _safe_len(bars) >= 3

    # VWAP reclaim: last close above VWAP after having been below it recently.
    if inp.vwap_reclaim is not None:
        m["vwap_reclaim"] = bool(inp.vwap_reclaim)
        m["vwap"] = None
    elif have_bars:
        vw = vwap_series(bars)
        try:
            last_c = float(bars["close"].iloc[-1]); last_vw = float(vw.iloc[-1])
            below_recently = bool((bars["close"].tail(6) < vw.tail(6)).any())
            m["vwap"] = last_vw
            m["vwap_reclaim"] = bool(last_c > last_vw and below_recently)
        except Exception:
            m["vwap"] = None; m["vwap_reclaim"] = False
    else:
        m["vwap"] = None; m["vwap_reclaim"] = False

    # RSI rising over last 5 bars / cross above 40.
    if inp.rsi_rising is not None and inp.rsi_cross_40 is not None:
        m["rsi_rising"] = bool(inp.rsi_rising); m["rsi_cross_40"] = bool(inp.rsi_cross_40)
    elif have_bars:
        rsi = rsi_series(bars["close"])
        try:
            r = rsi.dropna()
            m["rsi_rising"] = (bool(float(r.iloc[-1]) > float(r.iloc[-5]))
                               if inp.rsi_rising is None and len(r) >= 5 else bool(inp.rsi_rising))
            m["rsi_cross_40"] = (bool(float(r.iloc[-2]) < 40.0 <= float(r.iloc[-1]))
                                 if inp.rsi_cross_40 is None and len(r) >= 2 else bool(inp.rsi_cross_40))
        except Exception:
            m["rsi_rising"] = bool(inp.rsi_rising or False)
            m["rsi_cross_40"] = bool(inp.rsi_cross_40 or False)
    else:
        m["rsi_rising"] = bool(inp.rsi_rising or False)
        m["rsi_cross_40"] = bool(inp.rsi_cross_40 or False)

    # close > open on the confirmation (last) bar.
    if inp.close_gt_open is not None:
        m["close_gt_open"] = bool(inp.close_gt_open)
    elif have_bars:
        try:
            m["close_gt_open"] = bool(float(bars["close"].iloc[-1]) > float(bars["open"].iloc[-1]))
        except Exception:
            m["close_gt_open"] = False
    else:
        m["close_gt_open"] = False

    # Liquidity sweep then reclaim: a recent bar took out a prior swing low then
    # price closed back above that swept level.
    if inp.sweep_reclaim is not None:
        m["sweep_reclaim"] = bool(inp.sweep_reclaim)
    elif have_bars and _safe_len(bars) >= 8:
        try:
            lows = bars["low"]
            prior_low = float(lows.iloc[:-3].min())
            swept = bool((lows.tail(3) < prior_low).any())
            reclaimed = bool(float(bars["close"].iloc[-1]) > prior_low)
            m["sweep_reclaim"] = swept and reclaimed
        except Exception:
            m["sweep_reclaim"] = False
    else:
        m["sweep_reclaim"] = False

    # Opening-range reclaim: opening-range low broken then reclaimed (long).
    if inp.orb_reclaim is not None:
        m["orb_reclaim"] = bool(inp.orb_reclaim)
    elif have_bars and _safe_len(bars) >= 10:
        try:
            n_or = max(3, min(30, _safe_len(bars) // 3))
            or_low = float(bars["low"].iloc[:n_or].min())
            after = bars.iloc[n_or:]
            broke = bool((after["low"] < or_low).any())
            reclaimed = bool(float(bars["close"].iloc[-1]) > or_low)
            m["orb_reclaim"] = broke and reclaimed
        except Exception:
            m["orb_reclaim"] = False
    else:
        m["orb_reclaim"] = False

    # Price above 5m EMA20 (resampled if intraday is finer; else EMA on bars).
    if inp.above_ema20_5m is not None:
        m["above_ema20_5m"] = bool(inp.above_ema20_5m)
    elif have_bars:
        try:
            ema = ema_series(bars["close"], 20)
            m["above_ema20_5m"] = bool(float(bars["close"].iloc[-1]) >= float(ema.iloc[-1]))
        except Exception:
            m["above_ema20_5m"] = False
    else:
        m["above_ema20_5m"] = False

    # Recovering: red on the day but pushing up off the lows.
    if inp.recovering is not None:
        m["recovering"] = bool(inp.recovering)
    elif have_bars:
        try:
            day_low = float(bars["low"].min()); last_c = float(bars["close"].iloc[-1])
            off_lows = (last_c - day_low) > 0.5 * m["atr"]
            rising = last_c > float(bars["close"].iloc[max(0, _safe_len(bars) - 4)])
            m["recovering"] = bool(off_lows and rising)   # pushing up off the lows (red or green)
        except Exception:
            m["recovering"] = False
    else:
        m["recovering"] = bool(inp.chg_pct > 0)

    # Distance above day-low and above VWAP, in ATR units.
    if inp.near_low_atr is not None:
        m["near_low_atr"] = float(inp.near_low_atr)
    elif have_bars:
        try:
            m["near_low_atr"] = (float(bars["close"].iloc[-1]) - float(bars["low"].min())) / m["atr"]
        except Exception:
            m["near_low_atr"] = None
    else:
        m["near_low_atr"] = None

    if inp.extended_atr is not None:
        m["extended_atr"] = float(inp.extended_atr)
    elif have_bars and m.get("vwap"):
        try:
            m["extended_atr"] = (float(bars["close"].iloc[-1]) - float(m["vwap"])) / m["atr"]
        except Exception:
            m["extended_atr"] = 0.0
    else:
        m["extended_atr"] = 0.0

    # Moving TOWARD the Previous Day High: price below PDH (room to run), recovering /
    # pushing up, and PDH reachable within toward_pdh_max_atr. This is the late-day
    # selection signal — names climbing into their PDH.
    pdh = previous_day_high(inp)
    if pdh and pdh > 0 and m["atr"] > 0:
        dist_atr = (pdh - inp.price) / m["atr"]
        m["pdh_distance_atr"] = round(dist_atr, 3)
        m["toward_pdh"] = bool(inp.price < pdh and m.get("recovering")
                               and 0 < dist_atr <= cfg.toward_pdh_max_atr)
    else:
        m["pdh_distance_atr"] = None
        m["toward_pdh"] = False
    return m


def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# universe gate + scoring (PURE — no Alpaca, no I/O)
# ---------------------------------------------------------------------------
def universe_block(inp: ReversalInput, cfg: EngineConfig) -> Optional[str]:
    """Return the canonical block reason, or None if the symbol clears the
    universe filter. (avg_volume is PREFERRED, not required — it is a score
    bonus, never a gate.)"""
    if inp.halted or not inp.tradable:
        return "untradable_or_halted"
    if _finite(inp.price) <= cfg.min_price:
        return "price_below_5"
    if _finite(inp.volume) <= cfg.min_volume:    # vol > 1M
        return "volume_below_1m"
    if _finite(inp.chg_pct) <= 0:                # require GREEN today (reversal already underway)
        return "chg_not_positive"
    if _finite(inp.perf_1w) >= 0:                # but WEAK over the last week
        return "perf_1w_not_negative"
    if rel_vol_5m_of(inp) <= cfg.min_rel_vol_5m:  # 5-min relative volume > 1 (active right now)
        return "rel_vol_5m_below_1"
    if str(inp.analyst_rating or "").strip().lower() != STRONG_BUY:
        return "analyst_not_strong_buy"
    if str(inp.os_rating or "").strip().lower() != STRONG_BUY:
        return "os_not_strong_buy"
    return None


def score_candidate(inp: ReversalInput, cfg: Optional[EngineConfig] = None) -> ScoredCandidate:
    """Pure scorer. Universe pass is REQUIRED (not points). Returns a fully
    populated ScoredCandidate with score, reasons, score_components, blocked_by."""
    cfg = cfg or EngineConfig()
    blocked = universe_block(inp, cfg)
    if blocked:
        return ScoredCandidate(
            symbol=inp.symbol, score=0.0, volume=_finite(inp.volume), price=_finite(inp.price),
            passed_universe=False, passed_setup=False, eligible=False, blocked_by=blocked,
            reasons=[blocked], score_components={}, metrics={})

    m = derive_metrics(inp, cfg)
    comp: dict[str, float] = {}
    reasons: list[str] = []

    def add(key: str, pts: float, label: str):
        comp[key] = pts
        reasons.append(label)

    # weakness / context
    if _finite(inp.perf_1w) < 0:
        add("perf_1w_negative", 15, "1w weak")
    if _finite(inp.perf_1w) < -5:
        add("perf_1w_lt_-5", 10, "1w very weak (<-5%)")
    if _finite(inp.chg_pct) > 0:
        add("green_today", 10, "green today (reversing weekly weakness)")
    # liquidity
    if _finite(inp.volume) > cfg.volume_bonus_threshold:
        add("volume_gt_3m", 10, "volume > 3M")
    if _finite(inp.avg_volume) > cfg.pref_avg_volume:
        add("avg_volume_gt_1m", 5, "avg volume > 1M")
    rvol = _finite(m["rvol"])
    if rvol > 1.5:
        add("rvol_gt_1_5", 10, f"RVOL {rvol:.2f} > 1.5")
    if rvol > 2.0:
        add("rvol_gt_2_0", 10, f"RVOL {rvol:.2f} > 2.0")
    # reversal / reclaim structure
    if m["vwap_reclaim"]:
        add("vwap_reclaim", 20, "VWAP reclaim")
    if m["rsi_rising"]:
        add("rsi_rising", 10, "RSI rising")
    if m["rsi_cross_40"]:
        add("rsi_cross_40", 15, "RSI crossed > 40")
    if m["close_gt_open"]:
        add("close_gt_open", 10, "confirm bar close>open")
    if m["sweep_reclaim"]:
        add("sweep_reclaim", 25, "liquidity sweep + reclaim")
    if m["orb_reclaim"]:
        add("orb_reclaim", 20, "ORB reclaim")
    if m["above_ema20_5m"]:
        add("above_5m_ema20", 10, "above 5m EMA20")
    # moving toward the PDH target (late-day selection signal)
    if m.get("toward_pdh"):
        add("toward_pdh", 15, f"moving toward PDH ({m.get('pdh_distance_atr')} ATR away)")
    # over-extension penalty
    ext = _finite(m.get("extended_atr"), 0.0)
    if ext > cfg.extended_atr_limit:
        pen = -min(25.0, 5.0 * (ext - cfg.extended_atr_limit))
        add("extended_penalty", pen, f"extended {ext:.1f} ATR from VWAP")

    score = round(sum(comp.values()), 2)

    # A genuine setup requires at least one reclaim/reversal structure to be
    # present (otherwise it is just a weak red name with no trigger).
    has_trigger = any(m[k] for k in ("vwap_reclaim", "sweep_reclaim", "orb_reclaim",
                                     "rsi_cross_40"))
    passed_setup = bool(has_trigger)
    eligible = bool(passed_setup and score >= cfg.min_score)
    blocked_by = None
    if not passed_setup:
        blocked_by = "no_reversal_trigger"
    elif not eligible:
        blocked_by = "below_min_score"

    return ScoredCandidate(
        symbol=inp.symbol, score=score, volume=_finite(inp.volume), price=_finite(inp.price),
        passed_universe=True, passed_setup=passed_setup, eligible=eligible, blocked_by=blocked_by,
        reasons=reasons, score_components=comp,
        metrics={k: m[k] for k in ("rvol", "vol_5m", "rel_vol_5m", "atr", "vwap_reclaim",
                                   "rsi_rising", "rsi_cross_40", "close_gt_open", "sweep_reclaim",
                                   "orb_reclaim", "above_ema20_5m", "recovering", "extended_atr",
                                   "near_low_atr", "toward_pdh", "pdh_distance_atr")},
        prev_day_high=previous_day_high(inp), option_chain=inp.option_chain,
        price_now=_finite(inp.price), vol_5m=m.get("vol_5m"))


# ---------------------------------------------------------------------------
# scan (universe + score + funnel)
# ---------------------------------------------------------------------------
@dataclass
class ScanResult:
    scanned: int
    candidates: list[ScoredCandidate]      # passed universe, sorted volume desc then score desc
    rejected_by: dict[str, int]
    cfg: EngineConfig

    @property
    def universe_passed(self) -> int:
        return len(self.candidates)

    @property
    def setup_passed(self) -> int:
        return sum(1 for c in self.candidates if c.passed_setup)

    @property
    def eligible(self) -> list[ScoredCandidate]:
        return [c for c in self.candidates if c.eligible]

    def funnel(self, top: int = 10) -> dict[str, Any]:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scanned": self.scanned,
            "passed_universe": self.universe_passed,
            "passed_setup": self.setup_passed,
            "eligible_for_entry": len(self.eligible),
            "min_score": self.cfg.min_score,
            "rejected_by": dict(sorted(self.rejected_by.items(), key=lambda kv: -kv[1])),
            "sorted_by": "vol_5m_desc_then_score",
            "top_candidates": [
                {"symbol": c.symbol, "vol_5m": c.vol_5m, "volume": c.volume, "score": c.score,
                 "eligible": c.eligible, "blocked_by": c.blocked_by,
                 "reasons": c.reasons, "score_components": c.score_components}
                for c in self.candidates[:top]
            ],
        }


def _sort_key(c: ScoredCandidate) -> tuple[float, float]:
    """Universe sort: 5-MINUTE VOLUME desc, then SCORE desc. Falls back to day
    volume when a candidate has no 5m volume (so ordering stays deterministic)."""
    v5 = c.vol_5m if c.vol_5m is not None else c.volume
    return (-_finite(v5), -c.score)


def scan(rows: list[ReversalInput], cfg: Optional[EngineConfig] = None) -> ScanResult:
    """Score a universe. Pure: no Alpaca, no orders. Candidates that clear the
    universe filter are sorted by 5-MINUTE VOLUME DESC, then SCORE DESC."""
    cfg = cfg or EngineConfig()
    scored = [score_candidate(r, cfg) for r in rows]
    rejected: dict[str, int] = {}
    passed: list[ScoredCandidate] = []
    for c in scored:
        if not c.passed_universe:
            rejected[c.blocked_by] = rejected.get(c.blocked_by, 0) + 1
        else:
            passed.append(c)
            if c.blocked_by:   # passed universe but not eligible (no trigger / below score)
                rejected[c.blocked_by] = rejected.get(c.blocked_by, 0) + 1
    passed.sort(key=_sort_key)
    return ScanResult(scanned=len(rows), candidates=passed, rejected_by=rejected, cfg=cfg)


# ---------------------------------------------------------------------------
# sizing + entry proposal (still no Alpaca — returns intended orders)
# ---------------------------------------------------------------------------
def fallback_size(equity: float, price: float, atr: float, cfg: EngineConfig) -> int:
    """Risk <= 0.5% equity (1-ATR stop) AND notional <= 10% equity."""
    if equity <= 0 or price <= 0:
        return 0
    stop_dist = max(_finite(atr), price * 0.005)
    qty_risk = (cfg.risk_per_trade_pct * equity) / stop_dist if stop_dist > 0 else 0
    qty_notional = (cfg.max_notional_pct * equity) / price
    return int(max(0.0, min(qty_risk, qty_notional)))


def propose_entries(scan_result: ScanResult, equity: float, open_symbols: set[str],
                    cfg: Optional[EngineConfig] = None,
                    sizer: Optional[Callable[[ScoredCandidate], int]] = None,
                    opened_today: int = 0) -> list[EntryOrder]:
    """Build marketable-limit BUY orders for eligible candidates, honoring the
    ENGINE's own caps. NEVER averages into an existing symbol, never exceeds
    engine max-open or per-day caps, never emits below min_score. Returns intended
    orders only — the caller routes them through run.AlpacaGateway.submit_market_order
    so every existing safety gate still applies."""
    cfg = cfg or scan_result.cfg
    open_symbols = {str(s).upper() for s in (open_symbols or set())}
    orders: list[EntryOrder] = []
    opened = int(opened_today)
    for c in scan_result.eligible:                       # already volume-desc, score-desc
        if opened >= cfg.max_new_entries_per_day:
            break
        if (len(open_symbols) + len(orders)) >= cfg.max_open_positions:
            break
        if c.symbol.upper() in open_symbols:             # never average into a held name
            continue
        if c.score < cfg.min_score:                      # belt-and-suspenders
            continue
        atr = _finite(c.metrics.get("atr"), c.price * 0.01)
        qty = sizer(c) if sizer else fallback_size(equity, c.price, atr, cfg)
        if qty <= 0:
            continue
        limit_price = round(c.price * (1.0 + cfg.marketable_buffer), 2)
        orders.append(EntryOrder(
            symbol=c.symbol, side="buy", qty=int(qty), limit_price=limit_price,
            order_type="limit", score=c.score,
            reason=f"intraday_reversal:score={c.score:.0f}:{','.join(c.reasons[:3])}",
            target_underlying=c.prev_day_high, trailing_stop_pct=cfg.trailing_stop_pct))
        opened += 1
    return orders


# ---------------------------------------------------------------------------
# 1-7 DTE call-option overlay (target = underlying PDH; trailing stop on underlying)
# ---------------------------------------------------------------------------
def _occ_symbol(underlying: str, expiry: str, cp: str, strike: float) -> str:
    """Build an OCC-21 option symbol, e.g. AAPL + 260116 + C + 00150000."""
    root = str(underlying or "").upper()
    e = str(expiry or "").replace("-", "")
    yymmdd = e[2:8] if len(e) >= 8 else e[-6:]
    return f"{root}{yymmdd}{cp.upper()}{int(round(float(strike) * 1000)):08d}"


def _contract_dte(c: dict) -> Optional[int]:
    if c.get("dte") is not None:
        try:
            return int(c["dte"])
        except Exception:
            return None
    return None


def _contract_mid(c: dict) -> float:
    if c.get("mid") not in (None, ""):
        return _finite(c["mid"])
    b, a = _finite(c.get("bid")), _finite(c.get("ask"))
    return (b + a) / 2.0 if (b > 0 and a > 0) else 0.0


def select_call(price: float, chain: Optional[list], cfg: EngineConfig) -> tuple[Optional[dict], Optional[str]]:
    """Pick the best CALL within [option_min_dte, option_max_dte]: liquid (two-sided,
    spread <= max_spread_pct, OI >= floor), strike nearest the ATM*(1+offset) target,
    preferring the SHORTEST DTE in the window (cheapest / most leverage for an intraday
    move). Returns (contract, None) or (None, reason). Pure — no I/O."""
    calls = [c for c in (chain or []) if str(c.get("type", "call")).lower() == "call"]
    if not calls:
        return None, "no_call_chain"
    in_window = [c for c in calls if (_contract_dte(c) is not None
                 and cfg.option_min_dte <= _contract_dte(c) <= cfg.option_max_dte)]
    if not in_window:
        return None, "no_call_in_dte_window"

    def liquid(c: dict) -> bool:
        b, a, mid = _finite(c.get("bid")), _finite(c.get("ask")), _contract_mid(c)
        if b <= 0 or a <= 0 or mid <= 0:
            return False
        if (a - b) / mid > cfg.option_max_spread_pct:
            return False
        if _finite(c.get("open_interest")) < cfg.option_min_open_interest:
            return False
        return True

    liq = [c for c in in_window if liquid(c)]
    if not liq:
        return None, "no_liquid_call"
    target_strike = price * (1.0 + cfg.option_strike_offset_pct)
    best = min(liq, key=lambda c: (_contract_dte(c), abs(_finite(c["strike"]) - target_strike)))
    return best, None


def estimate_option_target_premium(strike: float, pdh: float, price: float, mid: float) -> float:
    """Rough FIRST-ORDER estimate of the call's value when the underlying reaches PDH:
    intrinsic at the target plus the current extrinsic held flat (ignores theta/vega/
    second-order delta). A planning number, NOT a pricing model."""
    intrinsic_now = max(0.0, price - strike)
    intrinsic_tgt = max(0.0, pdh - strike)
    extrinsic_now = max(0.0, mid - intrinsic_now)
    return round(intrinsic_tgt + extrinsic_now, 2)


def size_option(equity: float, premium: float, cfg: EngineConfig) -> int:
    """Contracts for a long call. Premium IS the defined max loss, so size by the
    smaller of the premium-risk budget and the 10% notional cap."""
    if equity <= 0 or premium <= 0:
        return 0
    per_contract = premium * 100.0
    by_risk = (cfg.option_risk_pct * equity) / per_contract
    by_notional = (cfg.max_notional_pct * equity) / per_contract
    return int(max(0.0, min(by_risk, by_notional)))


def propose_option_entries(scan_result: ScanResult, equity: float, open_symbols: set[str],
                           cfg: Optional[EngineConfig] = None,
                           opened_today: int = 0) -> list[OptionEntryOrder]:
    """For each eligible candidate, buy a 1-7 DTE CALL with a price target at the
    underlying's Previous Day High and a 0.75% trailing stop on the underlying.
    Honors the same engine gates as the share path (min score, engine max-open / per-day,
    never average into a held underlying). Returns intended orders only — the caller
    routes the OCC symbol through run.AlpacaGateway.submit_market_order (limit), where all
    existing safety gates still apply."""
    cfg = cfg or scan_result.cfg
    open_symbols = {str(s).upper() for s in (open_symbols or set())}
    orders: list[OptionEntryOrder] = []
    opened = int(opened_today)
    for c in scan_result.eligible:
        if opened >= cfg.max_new_entries_per_day:
            break
        if (len(open_symbols) + len(orders)) >= cfg.max_open_positions:
            break
        if c.symbol.upper() in open_symbols:                 # never average into a held name
            continue
        if c.score < cfg.min_score:
            continue
        pdh = c.prev_day_high
        if pdh is None or pdh <= c.price_now:                # target must be above current price
            continue
        contract, why = select_call(c.price_now, c.option_chain, cfg)
        if contract is None:
            continue
        mid = _contract_mid(contract)
        ask = _finite(contract.get("ask"))
        limit_price = round(ask if ask > 0 else mid * (1.0 + cfg.option_marketable_buffer), 2)
        qty = size_option(equity, limit_price, cfg)
        if qty <= 0:
            continue
        strike = _finite(contract["strike"])
        dte = _contract_dte(contract) or 0
        occ = contract.get("symbol") or _occ_symbol(c.symbol, contract.get("expiry", ""), "C", strike)
        tgt_prem = estimate_option_target_premium(strike, pdh, c.price_now, mid)
        mult = round(tgt_prem / mid, 2) if mid > 0 else 0.0
        orders.append(OptionEntryOrder(
            occ_symbol=occ, underlying=c.symbol, side="buy", qty=int(qty), limit_price=limit_price,
            order_type="limit", strike=strike, expiry=str(contract.get("expiry", "")), dte=int(dte),
            est_premium=round(mid, 2), target_underlying=round(pdh, 2), est_target_premium=tgt_prem,
            est_target_multiple=mult, trailing_stop_pct=cfg.trailing_stop_pct, score=c.score,
            reason=(f"intraday_reversal_call:score={c.score:.0f}:dte={dte}:"
                    f"target=PDH({pdh:.2f}):trail={cfg.trailing_stop_pct*100:.2f}%")))
        opened += 1
    return orders


def propose_for_engine(scan_result: ScanResult, equity: float, open_symbols: set[str],
                       cfg: Optional[EngineConfig] = None,
                       sizer: Optional[Callable[[ScoredCandidate], int]] = None,
                       opened_today: int = 0):
    """Route to the configured entry instrument: option calls when options_enabled,
    else shares."""
    cfg = cfg or scan_result.cfg
    if cfg.options_enabled:
        return propose_option_entries(scan_result, equity, open_symbols, cfg, opened_today)
    return propose_entries(scan_result, equity, open_symbols, cfg, sizer, opened_today)


# ---------------------------------------------------------------------------
# end-of-day / overnight / next-day management (PURE — no Alpaca, no orders)
# ---------------------------------------------------------------------------
@dataclass
class ManagedPosition:
    symbol: str
    qty: float
    underlying_price: float            # current underlying price
    prev_day_high: float               # PDH (the day-1 price target)
    unrealized_pl: float               # current P/L on the position (>0 == "up")
    trailing_stop_pct: float           # current trailing stop on the underlying
    pdh_hit: bool = False              # did the underlying reach PDH during the session?
    is_option: bool = False
    occ_symbol: Optional[str] = None


@dataclass
class EodAction:
    symbol: str
    action: str                        # target_hit | carry_overnight_tighten | exit_eod | hold
    new_trailing_stop_pct: Optional[float]
    exit_plan: Optional[str]           # sell_at_next_day_high | exit_at_close | None
    reason: str


def manage_eod(pos: ManagedPosition, cfg: Optional[EngineConfig] = None) -> EodAction:
    """End-of-day decision for a reversal position:
      * PDH already hit  -> target reached; the exit engine takes the target (no carry here).
      * PDH NOT hit AND position UP -> TIGHTEN the trailing stop and carry overnight to
        sell into the next day's high (the requested behavior).
      * PDH NOT hit AND not up -> close at EOD (no overnight risk on a losing reversal)."""
    cfg = cfg or EngineConfig()
    if pos.pdh_hit:
        return EodAction(pos.symbol, "target_hit", None, None,
                         "PDH reached intraday — exit engine takes the target")
    up = _finite(pos.unrealized_pl) > 0
    if up and cfg.carry_overnight_if_up:
        tightened = min(_finite(pos.trailing_stop_pct, cfg.trailing_stop_pct),
                        cfg.overnight_trailing_stop_pct)
        return EodAction(pos.symbol, "carry_overnight_tighten", round(tightened, 5),
                         "sell_at_next_day_high",
                         (f"PDH not hit but position +{pos.unrealized_pl:.2f} — tighten trail to "
                          f"{tightened*100:.2f}% and carry overnight; sell into next day's high"))
    return EodAction(pos.symbol, "exit_eod", None, "exit_at_close",
                     "PDH not hit and position not up — close at EOD (no overnight risk)")


@dataclass
class NextDayExit:
    symbol: str
    sell: bool
    reason: str


def next_day_exit_decision(price: float, running_high: float, prev_day_high: float,
                           cfg: Optional[EngineConfig] = None) -> NextDayExit:
    """Day-2 exit for an overnight carry — "sell at the next day's high":
      * if the underlying finally reaches the PDH target, sell (target);
      * otherwise track the running intraday high and sell when price retraces
        next_day_high_trail_pct below it, locking in NEAR the day's high."""
    cfg = cfg or EngineConfig()
    p, hi = _finite(price), _finite(running_high)
    if prev_day_high and p >= _finite(prev_day_high) > 0:
        return NextDayExit("", True, "pdh_target_reached")
    if hi > 0 and p <= hi * (1.0 - cfg.next_day_high_trail_pct):
        return NextDayExit("", True,
                           f"next_day_high_trail_hit (retraced {cfg.next_day_high_trail_pct*100:.2f}% "
                           f"from high {hi:.2f})")
    return NextDayExit("", False, "holding for next-day high / PDH target")


# ---------------------------------------------------------------------------
# funnel logging
# ---------------------------------------------------------------------------
def write_funnel(scan_result: ScanResult, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scan_result.funnel(), indent=2, default=str), encoding="utf-8")
    return out


def log_funnel(scan_result: ScanResult) -> None:
    f = scan_result.funnel()
    logger.info("INTRADAY_REVERSAL funnel: scanned=%d universe=%d setup=%d eligible=%d",
                f["scanned"], f["passed_universe"], f["passed_setup"], f["eligible_for_entry"])
    if f["rejected_by"]:
        logger.info("INTRADAY_REVERSAL rejected_by=%s", f["rejected_by"])


# ---------------------------------------------------------------------------
# demo / dry-run
# ---------------------------------------------------------------------------
def _demo_rows() -> list[ReversalInput]:
    """Synthetic universe spanning eligible, sub-score, and rejected cases."""
    return [
        # Strong reversal: weak 1w, GREEN today off the lows, high 5m RVOL, full reclaim stack.
        ReversalInput(symbol="RVRS", price=42.10, volume=8_200_000, avg_volume=3_100_000,
                      chg_pct=+1.8, perf_1w=-7.4, analyst_rating="strong buy", os_rating="strong buy",
                      rvol=2.6, vwap_reclaim=True, rsi_rising=True, rsi_cross_40=True,
                      close_gt_open=True, sweep_reclaim=True, orb_reclaim=True, above_ema20_5m=True,
                      recovering=True, extended_atr=0.6, atr=0.9, prev_day_high=44.50,
                      vol_5m=420_000,
                      option_chain=[
                          {"expiry": "2026-06-12", "strike": 42.0, "type": "call", "bid": 1.10,
                           "ask": 1.20, "dte": 1, "open_interest": 2500, "volume": 800},
                          {"expiry": "2026-06-12", "strike": 43.0, "type": "call", "bid": 0.65,
                           "ask": 0.72, "dte": 1, "open_interest": 1800, "volume": 600},
                          {"expiry": "2026-06-19", "strike": 42.0, "type": "call", "bid": 1.80,
                           "ask": 1.92, "dte": 7, "open_interest": 4200, "volume": 300}]),
        # Decent but milder: moderate RVOL, VWAP reclaim only.
        ReversalInput(symbol="MILD", price=18.55, volume=5_400_000, avg_volume=1_500_000,
                      chg_pct=+0.9, perf_1w=-3.1, analyst_rating="strong buy", os_rating="strong buy",
                      rvol=1.7, vwap_reclaim=True, rsi_rising=True, rsi_cross_40=False,
                      close_gt_open=True, sweep_reclaim=False, orb_reclaim=False, above_ema20_5m=True,
                      recovering=True, extended_atr=0.4, atr=0.4, prev_day_high=19.40,
                      vol_5m=610_000,
                      option_chain=[
                          {"expiry": "2026-06-13", "strike": 18.5, "type": "call", "bid": 0.55,
                           "ask": 0.62, "dte": 2, "open_interest": 1200, "volume": 150},
                          {"expiry": "2026-06-19", "strike": 19.0, "type": "call", "bid": 0.40,
                           "ask": 0.46, "dte": 7, "open_interest": 900, "volume": 100}]),
        # Over-extended: triggers but penalized hard, likely below min score.
        ReversalInput(symbol="EXTD", price=120.0, volume=4_100_000, avg_volume=900_000,
                      chg_pct=+0.4, perf_1w=-1.2, analyst_rating="strong buy", os_rating="strong buy",
                      rvol=1.6, vwap_reclaim=True, rsi_rising=True, rsi_cross_40=False,
                      close_gt_open=True, sweep_reclaim=False, orb_reclaim=False, above_ema20_5m=False,
                      recovering=False, extended_atr=4.5, atr=2.0, vol_5m=180_000),
        # Rejected universe: RED today (must be green).
        ReversalInput(symbol="REDT", price=33.0, volume=6_000_000, avg_volume=2_000_000,
                      chg_pct=-0.8, perf_1w=-2.0, analyst_rating="strong buy", os_rating="strong buy",
                      vol_5m=200_000),
        # Rejected universe: 5-min relative volume <= 1 (not active right now).
        ReversalInput(symbol="QUIT", price=27.0, volume=2_000_000, avg_volume=5_000_000,
                      chg_pct=+0.3, perf_1w=-2.0, analyst_rating="strong buy", os_rating="strong buy",
                      vol_5m=30_000),
        # Rejected universe: not strong buy.
        ReversalInput(symbol="HOLD", price=9.0, volume=9_000_000, avg_volume=2_000_000,
                      chg_pct=+0.5, perf_1w=-4.0, analyst_rating="buy", os_rating="strong buy",
                      vol_5m=300_000),
    ]


def run_demo(cfg: Optional[EngineConfig] = None) -> ScanResult:
    cfg = cfg or EngineConfig()
    res = scan(_demo_rows(), cfg)
    f = res.funnel()
    print("=" * 78)
    print(f"INTRADAY REVERSAL ENGINE — DRY RUN (no orders)   min_score={cfg.min_score}")
    print(f"enabled={cfg.enabled}  paper_only={cfg.paper_only}  "
          f"max_new/day={cfg.max_new_entries_per_day}  max_open={cfg.max_open_positions}")
    print("=" * 78)
    print(f"scanned={f['scanned']}  passed_universe={f['passed_universe']}  "
          f"passed_setup={f['passed_setup']}  eligible={f['eligible_for_entry']}")
    print(f"rejected_by={f['rejected_by']}")
    print("-" * 78)
    print(f"(universe sorted by 5-min volume desc, then score)")
    print(f"{'SYMBOL':<8}{'5MIN VOL':>11}{'DAY VOL':>12}{'SCORE':>7}{'ELIG':>6}  REASONS")
    for c in res.candidates:
        v5 = int(c.vol_5m) if c.vol_5m is not None else 0
        print(f"{c.symbol:<8}{v5:>11,}{int(c.volume):>12,}{c.score:>7.0f}"
              f"{('Y' if c.eligible else '-'):>6}  {', '.join(c.reasons[:4])}")
    # proposed share entries (dry-run sizing on a demo account; still NO orders placed)
    equity = 100_000.0
    orders = propose_entries(res, equity=equity, open_symbols=set(), cfg=cfg)
    print("-" * 78)
    print(f"PROPOSED SHARE ENTRIES (dry-run, NOT submitted): {len(orders)}")
    for o in orders:
        print(f"  BUY {o.symbol} x{o.qty} @ limit {o.limit_price}  "
              f"target=PDH({o.target_underlying})  trailSL={o.trailing_stop_pct*100:.2f}%")
    # 1-7 DTE CALL overlay: target = underlying PDH, 0.75% trailing stop on underlying
    opt = propose_option_entries(res, equity=equity, open_symbols=set(), cfg=cfg)
    print("-" * 78)
    print(f"PROPOSED 1-7DTE CALL ENTRIES (dry-run, NOT submitted): {len(opt)}")
    for o in opt:
        print(f"  BUY {o.occ_symbol} x{o.qty}c @ limit {o.limit_price}  "
              f"(strike {o.strike}, {o.dte}DTE, prem~{o.est_premium})")
        print(f"      target=PDH({o.target_underlying}) -> est prem {o.est_target_premium} "
              f"(~{o.est_target_multiple}x)   trailSL {o.trailing_stop_pct*100:.2f}% on underlying")
    print("=" * 78)
    return res


def _rows_from_json(path: str) -> list[ReversalInput]:
    import dataclasses as _dc
    fields = {f.name for f in _dc.fields(ReversalInput)}
    data = json.loads(Path(path).read_text())
    rows = data.get("rows", data) if isinstance(data, dict) else data
    return [ReversalInput(**{k: v for k, v in r.items() if k in fields}) for r in rows]


def _positions_from_json(path: str) -> list[ManagedPosition]:
    import dataclasses as _dc
    fields = {f.name for f in _dc.fields(ManagedPosition)}
    data = json.loads(Path(path).read_text())
    rows = data.get("positions", data) if isinstance(data, dict) else data
    return [ManagedPosition(**{k: v for k, v in r.items() if k in fields}) for r in rows]


def _marker_path(now_et) -> Path:
    base = os.getenv("INTRADAY_REVERSAL_LOG_DIR", "logs/intraday_reversal")
    return Path(base) / f"scan_{now_et.strftime('%Y%m%d')}.done"


def run_scan(cfg: EngineConfig, universe: Optional[str], funnel_out: str,
             force: bool, equity: float) -> int:
    """Scheduled scan: only acts inside the after-1:30 PM ET window (unless --force), at
    most once per day. Dry-run output; live routing is the caller's job via run.py."""
    now_et = datetime.now(ET)
    if not force and not is_active_window(now_et, cfg):
        print(f"[intraday-reversal] {now_et:%H:%M ET} outside active window "
              f"[{cfg.start_et}, {cfg.entry_cutoff_et}) — no-op")
        return 0
    marker = _marker_path(now_et)
    if not force and marker.exists():
        print(f"[intraday-reversal] already scanned today ({marker}) — no-op")
        return 0
    rows = _rows_from_json(universe) if universe else _demo_rows()
    res = scan(rows, cfg)
    f = res.funnel()
    print(f"[intraday-reversal] {now_et:%Y-%m-%d %H:%M ET}  scanned={f['scanned']} "
          f"universe={f['passed_universe']} setup={f['passed_setup']} eligible={f['eligible_for_entry']}")
    for c in res.candidates:
        tag = "*toward_PDH" if c.metrics.get("toward_pdh") else ""
        print(f"  {c.symbol:<8} 5mVol={int(c.vol_5m or 0):>9,} score={c.score:>4.0f} "
              f"PDH={c.prev_day_high} {('ELIGIBLE' if c.eligible else c.blocked_by or '')} {tag}")
    orders = propose_for_engine(res, equity=equity, open_symbols=set(), cfg=cfg)
    print(f"[intraday-reversal] proposed {len(orders)} entries "
          f"({'CALLS' if cfg.options_enabled else 'shares'}, dry-run, NOT submitted)")
    out = funnel_out or str(_marker_path(now_et).with_name(f"funnel_{now_et:%Y%m%d}.json"))
    write_funnel(res, out)
    print(f"[intraday-reversal] funnel -> {out}")
    if not force:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(now_et.isoformat())
    return 0


def run_manage_eod(cfg: EngineConfig, positions: Optional[str]) -> int:
    """Near-close pass: for each open reversal position decide hold/tighten+carry/exit.
    Dry-run — prints intended actions; the existing exit engine executes them."""
    if positions:
        poss = _positions_from_json(positions)
    else:   # demo: one up-but-PDH-not-hit carry, one loser to close, one target hit
        poss = [
            ManagedPosition("CARRY", qty=200, underlying_price=43.9, prev_day_high=44.5,
                            unrealized_pl=185.0, trailing_stop_pct=0.0075, pdh_hit=False),
            ManagedPosition("LOSER", qty=150, underlying_price=17.8, prev_day_high=19.4,
                            unrealized_pl=-90.0, trailing_stop_pct=0.0075, pdh_hit=False),
            ManagedPosition("WONIT", qty=100, underlying_price=44.6, prev_day_high=44.5,
                            unrealized_pl=240.0, trailing_stop_pct=0.0075, pdh_hit=True),
        ]
    print(f"[intraday-reversal] EOD management of {len(poss)} position(s):")
    for pos in poss:
        a = manage_eod(pos, cfg)
        extra = (f" -> trail {a.new_trailing_stop_pct*100:.2f}% / {a.exit_plan}"
                 if a.new_trailing_stop_pct is not None else
                 (f" -> {a.exit_plan}" if a.exit_plan else ""))
        print(f"  {pos.symbol:<8} {a.action}{extra}")
        print(f"           {a.reason}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Intraday Reversal Engine (isolated candidate source)")
    p.add_argument("--demo", action="store_true", help="dry-run: print ranked candidates, NO orders")
    p.add_argument("--scan", action="store_true",
                   help="scheduled scan (gated to the after-1:30 PM ET window, once/day)")
    p.add_argument("--manage-eod", action="store_true",
                   help="near-close pass: tighten+carry up positions that didn't hit PDH")
    p.add_argument("--universe", default="", help="JSON of universe rows for --scan (else demo)")
    p.add_argument("--positions", default="", help="JSON of positions for --manage-eod (else demo)")
    p.add_argument("--force", action="store_true", help="ignore the active-window/once-a-day gate")
    p.add_argument("--equity", type=float, default=100_000.0, help="account equity for dry-run sizing")
    p.add_argument("--funnel-out", default="", help="path to write the funnel JSON")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    cfg = EngineConfig()
    if args.manage_eod:
        return run_manage_eod(cfg, args.positions or None)
    if args.scan:
        return run_scan(cfg, args.universe or None, args.funnel_out, args.force, args.equity)
    res = run_demo(cfg)
    if args.funnel_out:
        print("funnel ->", write_funnel(res, args.funnel_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
