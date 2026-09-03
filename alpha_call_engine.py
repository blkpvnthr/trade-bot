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
import logging
import os
import math
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

_ace_log = logging.getLogger(__name__)

import alpha_probability_model as apm
import oops_option_optimizer as oops_opt

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

# Guard: suppress event file writes when running under pytest so tests don't
# contaminate production execution journals.
# NOTE: evaluated at call-time (not import-time) so that PYTEST_CURRENT_TEST is seen correctly.
def _in_test() -> bool:
    import sys as _sys
    return (
        "PYTEST_CURRENT_TEST" in os.environ
        or "pytest" in os.path.basename(os.environ.get("_", ""))
        or "unittest" in _sys.modules
        or any("unittest" in a or "pytest" in a for a in _sys.argv[:2] if a)
    )


# ── configuration ─────────────────────────────────────────────────────────────
@dataclass
class AlphaCallConfig:
    enabled: bool = os.getenv("ENABLE_ALPHA_CALL_ENGINE", "true").lower() in {"1", "true"}
    # Regime requirement
    require_bullish_regime: bool = True
    # "mixed" and "cautious" are allowed but trigger reduced sizing + tighter score gate.
    # Override via ALPHA_ALLOWED_REGIMES (comma-separated) if desired.
    allowed_regimes: tuple = tuple(
        r.strip() for r in os.getenv(
            "ALPHA_ALLOWED_REGIMES",
            "bullish,risk_on,uptrend,constructive,trend_risk_on,neutral,mixed,cautious"
        ).split(",") if r.strip()
    )
    # On mixed/cautious days use a lower score floor (B quality acceptable) but still
    # require a real technical trigger (ORB/VWAP/ACCUM). Override via ALPHA_MIXED_MIN_SCORE.
    mixed_regime_min_score: float = float(os.getenv("ALPHA_MIXED_MIN_SCORE", "40"))
    # Risk per trade is halved on mixed/cautious days to preserve capital under uncertainty.
    # Override via ALPHA_MIXED_RISK_MULT (0.0–1.0).
    mixed_regime_risk_multiplier: float = float(os.getenv("ALPHA_MIXED_RISK_MULT", "0.50"))
    # Setup thresholds
    min_setup_score: float = float(os.getenv("ALPHA_MIN_SETUP_SCORE", "70"))   # A = 70-84, A+ = 85+
    # Risk per trade
    risk_per_trade_pct: float = float(os.getenv("OPTIONS_RISK_PER_TRADE_PCT", "0.015"))
    max_options_exposure_pct: float = float(os.getenv("MAX_OPTIONS_EXPOSURE_PCT", "0.15"))
    # Kelly sizing (options_quant) — fractional Kelly on the live ALPACA ACCOUNT EQUITY,
    # capped by risk_per_trade_pct (Kelly can only reduce size below the cap, never exceed
    # it). full Kelly is shrunk for dispersed option payoffs (Perez-Marco); ¼-Kelly default.
    kelly_sizing_enabled: bool = os.getenv("OPTIONS_KELLY_SIZING", "true").lower() in {"1", "true"}
    kelly_full: float = float(os.getenv("OPTIONS_KELLY_FULL", "0.10"))     # full-Kelly fraction f*
    kelly_mult: float = float(os.getenv("OPTIONS_KELLY_MULT", "0.25"))     # fractional Kelly (¼)
    # Embedded-leverage contract selection (options_quant / Frazzini-Pedersen): tilt the
    # composite score toward the moderate-leverage (deep-ITM) band and penalise high-leverage
    # short-dated OTM contracts, which carry a negative expected premium for the buyer.
    # DISABLED by default (real-chain verdict): the embedded-leverage/deep-ITM tilt was invalidated
    # on real liquid chains (it was spread-avoidance, not edge). Reverts bs_score to the ATM-moneyness
    # preference. Re-enable only for research. See options_quant_engine for the (research-branch) EV path.
    lev_select_enabled: bool = os.getenv("OPTIONS_LEV_SELECT", "false").lower() in {"1", "true"}
    lev_band_lo: float = float(os.getenv("OPTIONS_LEV_BAND_LO", "2.0"))
    lev_band_hi: float = float(os.getenv("OPTIONS_LEV_BAND_HI", "6.0"))
    lev_select_weight: float = float(os.getenv("OPTIONS_LEV_SELECT_WEIGHT", "0.70"))  # MAX tilt (applied at wide spreads)
    # SPREAD-AWARE tilt (backtest-grounded): deep-ITM tilt strength scales with the contract's
    # bid-ask spread. LIQUID names (tight spread) → tilt≈0, ATM convexity wins (momentum backtest:
    # ATM nets more once the big move is held). WIDE-spread (illiquid discovery) names → full tilt,
    # ATM's all-extrinsic premium is taxed away → deep-ITM selected. spread_tilt =
    # clip(spread_pct / lev_spread_ref, 0, 1). Set lev_spread_aware=0 for a flat tilt at lev_select_weight.
    lev_spread_aware: bool = os.getenv("OPTIONS_LEV_SPREAD_AWARE", "true").lower() in {"1", "true"}
    # spread at which the tilt is full — set to the hard spread gate (max_spread_pct) so the tilt
    # spans the FULL admissible band: tight end (≈1%) → minimal tilt (ATM), gate-limit (≈3.5%)
    # → full tilt (deep-ITM). Contracts above max_spread_pct are hard-rejected upstream.
    lev_spread_ref: float = float(os.getenv("OPTIONS_LEV_SPREAD_REF", "0.035"))
    # Contract selection
    min_dte: int = int(os.getenv("ALPHA_MIN_DTE", "21"))
    max_dte: int = int(os.getenv("ALPHA_MAX_DTE", "60"))
    # Delta band 0.45-0.65 (REAL-CHAIN VERDICT 2026-06-07): on liquid chains a simple nearest-ATM
    # call was the ONLY net-positive real result (+9.5%); the EV selector's deep-ITM/OTM picks LOST
    # (its apparent edge was just spread-avoidance, which vanishes at liquid 7% spreads). So: stay
    # ATM-centric, no OTM lottery (floor 0.45), no deep-ITM (cap 0.65). The widened 0.85 / leverage
    # tilt are reverted — the EV machinery is a research branch (options_quant_engine), not live.
    min_delta: float = float(os.getenv("ALPHA_MIN_DELTA", "0.45"))
    max_delta: float = float(os.getenv("ALPHA_MAX_DELTA", "0.65"))
    min_open_interest: int = int(os.getenv("ALPHA_MIN_OI", "200"))
    min_volume: int = int(os.getenv("ALPHA_MIN_VOLUME", "10"))
    # Spread filter ≤10% (REAL-CHAIN VERDICT): liquid mega-cap option spreads (~7%) were profitable;
    # small-cap spreads (30-56%) made the strategy a guaranteed loser. 10% cleanly separates the
    # tradeable (liquid) from the untradeable. Min floor 0 (don't reject ultra-liquid tight quotes —
    # the earlier 1% floor could have excluded good liquid contracts). estimated-mid path EXEMPT.
    min_spread_pct: float = float(os.getenv("ALPHA_MIN_SPREAD_PCT", "0.0"))
    max_spread_pct: float = float(os.getenv("ALPHA_MAX_SPREAD_PCT", "0.10"))
    # Position management
    # Risk/reward structure: 3R target with trailing stop that activates at 1.7R.
    # R = initial risk = entry premium (max loss on long call).
    # When the position reaches +1.7R profit, the stop trails up to lock in gains.
    # At +3R the remaining position is fully exited.
    # OPTIONS_TP_R_MULTIPLE / OPTIONS_SL_PCT are honored as aliases of the existing
    # ALPHA_* env names so management config can use either; defaults are unchanged.
    # r_target raised 3.0→4.0 (2026-06-07): backtest_alpha_rtarget.py shows net return rises
    # monotonically 3R→4R→6R for both ATM and deep-ITM across realistic spreads (e.g. +12.6%→
    # +17.1%→+19.6% at 3% spread). 4R captures most of the gain with less right-tail dependence
    # than 6R; set OPTIONS_TP_R_MULTIPLE=6 to maximise the backtested mean. The 1.7R trailing
    # stop protects gains while a winner runs toward the higher target.
    r_target: float = float(os.getenv("OPTIONS_TP_R_MULTIPLE", os.getenv("ALPHA_R_TARGET", "4.0")))  # exit all at 4R
    # r_trail_activate raised 1.7→3.0 (2026-06-07): backtest_alpha_rtarget.py (trailing-stop ladder)
    # showed the 4R/6R target benefit is FULLY suppressed at 1.7R — the trail exits winners before
    # the higher target. Activating at +300% (3.0R) lets a winner run toward the 4R target before
    # the trail (0.8R behind peak) engages. Lifts deep-ITM ladder mean +11.0→+11.7%, PF 1.53→1.57;
    # gain plateaus past ~3.5R. Tradeoff: slightly deeper give-back (MaxDD −70→−72%).
    r_trail_activate: float = float(os.getenv("ALPHA_R_TRAIL_ACTIVATE", "3.0"))  # activate trailing stop at 3.0R
    r_trail_offset: float = float(os.getenv("ALPHA_R_TRAIL_OFFSET", "0.8"))      # trail 0.8R behind peak
    stop_loss_pct: float = float(os.getenv("OPTIONS_SL_PCT", os.getenv("ALPHA_STOP_PCT", "0.28")))  # hard stop: -28% of premium
    partial_profit_pct: float = float(os.getenv("ALPHA_PARTIAL_PROFIT_PCT", "0.5"))   # +50% = take partial (sell partial_size)
    partial_size: float = float(os.getenv("ALPHA_PARTIAL_SIZE", "0.50"))   # sell 50% at +1R
    # Universe
    max_concurrent_positions: int = int(os.getenv("ALPHA_MAX_POSITIONS", "10"))  # raised from 5→10 (2026-06-03); exposure cap (15%) is the binding risk limit
    scan_interval_seconds: int = int(os.getenv("ALPHA_SCAN_SECONDS", "60"))
    # ORB window
    orb_minutes: int = int(os.getenv("ALPHA_ORB_MINUTES", "15"))
    # ORB HARD GATE (2026-06-07): a long call may NOT be entered from the signal alone — the
    # UNDERLYING must first break above the opening-range high (high of 09:30-09:45 ET). No entry
    # before 09:45, while price is inside the range, or below it. This is a HARD gate BEFORE
    # contract selection (the signal only identifies a candidate; ORB confirms execution).
    # ORB requirement DISABLED by default (real-chain verdict: naive + sweep-reclaim ORB both failed
    # to improve the ATM-call distribution). Available for research via ALPHA_ORB_GATE=true.
    orb_gate_enabled: bool = os.getenv("ALPHA_ORB_GATE", "false").lower() in {"1", "true"}
    orb_require_close: bool = os.getenv("ALPHA_ORB_REQUIRE_CLOSE", "false").lower() in {"1", "true"}  # close>ORB_high, not just wick
    # ORB mode: "breakout" (simple price>OR_high) | "sweep_reclaim" (liquidity-sweep reversal:
    # sweep OR_high → pull back into range → bullish engulfing of the pullback → enter at its close).
    orb_mode: str = os.getenv("ALPHA_ORB_MODE", "sweep_reclaim").strip().lower()
    orb_candle_minutes: int = int(os.getenv("ALPHA_ORB_CANDLE_MINUTES", "5"))  # pattern candle timeframe
    orb_require_reclaim: bool = os.getenv("ALPHA_ORB_REQUIRE_RECLAIM", "true").lower() in {"1", "true"}  # engulf close > OR_high
    # LIQUIDITY-SWEEP TREND+DELTA signal (on the UNDERLYING) — gates option entry / triggers exit.
    # buySignal = bullish trend-crossover + bullish volume-delta → allow ATM-call scan;
    # sellSignal = bearish trend-crossover + bearish delta → close long calls for that underlying.
    liq_sweep_enabled: bool = os.getenv("ALPHA_LIQ_SWEEP_SIGNAL", "true").lower() in {"1", "true"}
    liq_sweep_len: int = int(os.getenv("ALPHA_LIQ_SWEEP_LEN", "12"))
    liq_sweep_mult: float = float(os.getenv("ALPHA_LIQ_SWEEP_MULT", "2.0"))
    # Real-chain verdict (2026-06-07): the buySignal entry gate tested NEUTRAL (no edge) → DISABLED
    # by default. The sellSignal is kept as an OPTIONAL emergency trend-reversal exit (default on).
    liq_sweep_buy_gate: bool = os.getenv("ALPHA_LIQ_SWEEP_BUY_GATE", "false").lower() in {"1", "true"}
    liq_sweep_sell_exit: bool = os.getenv("ALPHA_LIQ_SWEEP_SELL_EXIT", "true").lower() in {"1", "true"}
    # Analyst-target ENTRY filter (HIGHER-VALUE use): skip the BUY when analyst upside is exhausted —
    # avg-target upside < 5% (≥3 estimates). "Don't buy calls where Wall St already thinks it's fully
    # valued." Fail-open if targets missing. EXIT filter (analyst_target_exit) is separate.
    analyst_entry_filter: bool = os.getenv("ALPHA_ANALYST_ENTRY_FILTER", "true").lower() in {"1", "true"}
    analyst_entry_min_upside: float = float(os.getenv("ALPHA_ANALYST_MIN_UPSIDE", "0.05"))
    # IV-REGIME GATE — feature-flagged, DEFAULT OFF (real-chain Study 2: ATM calls with contract
    # IV in [0.30,0.50] had a POSITIVE median +28% / 57% win vs the unfiltered negative median;
    # below 0.30 the underlying doesn't move enough, above 0.50 you overpay for vol). Kept OFF
    # because an IV band is more overfittable than the (structural) ORB gate — needs the
    # ORB-interaction ablation before production. Reject the contract if its IV is outside the band.
    iv_gate_enabled: bool = os.getenv("ALPHA_IV_GATE", "false").lower() in {"1", "true"}
    iv_gate_min: float = float(os.getenv("ALPHA_IV_MIN", "0.30"))
    iv_gate_max: float = float(os.getenv("ALPHA_IV_MAX", "0.50"))
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
    # ── OptionLab payoff/risk profile (advisory; validates the chosen call) ───────
    optionlab_enabled: bool = os.getenv("ENABLE_OPTIONLAB", "true").lower() in {"1", "true"}
    # Final combined option score weights (configurable; renormalised at use):
    #   combined = w_bs*BS + w_oops*OOPS + w_optionlab*OptionLab + w_liq*liquidity
    combined_bs_weight: float = float(os.getenv("COMBINED_BS_WEIGHT", "0.25"))
    combined_oops_weight: float = float(os.getenv("COMBINED_OOPS_WEIGHT", "0.30"))
    combined_optionlab_weight: float = float(os.getenv("COMBINED_OPTIONLAB_WEIGHT", "0.25"))
    combined_liq_weight: float = float(os.getenv("COMBINED_LIQ_WEIGHT", "0.20"))


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
    orb_status: str = "ok"          # "ok" when the 09:30-09:45 ET window had bars (else the reason)
    orb_last_close: float = 0.0     # last completed 1m candle close (for ORB close-confirmation)
    orb_signal: bool = False        # sweep-reclaim pattern fired (sweep_reclaim mode)
    orb_pattern_status: str = ""    # detector status for the sweep-reclaim pattern
    orb_entry_price: float = 0.0    # engulfing-bar close (intended entry; sweep_reclaim mode)
    liq_sweep_buy: bool = False     # liquidity-sweep buySignal on the underlying (latest bar)
    liq_sweep_sell: bool = False    # liquidity-sweep sellSignal on the underlying (latest bar)
    liq_sweep_trend: int = 0        # current liquidity-sweep trend (+1/-1/0)
    notes: list[str] = field(default_factory=list)
    evaluated_at_utc: str = ""
    # ── shared streaming-indicator provenance (timing diagnostics; never directional) ──
    indicator_source: str = "fallback_static"   # "streaming" | "fallback_static"
    indicator_age_seconds: float | None = None
    streaming_indicator_available: bool = False
    fallback_used: bool = True
    # ── premarket live-feed awareness (improves scoring only; never creates a trade) ──
    premarket_source: str = ""                  # "yliveticker" | "yfinance" | ""
    premarket_gap_pct: float | None = None
    premarket_volume: float | None = None
    premarket_high: float | None = None
    premarket_low: float | None = None
    premarket_relative_strength: float | None = None
    premarket_stale: bool = False
    premarket_rank_score: float | None = None   # advisory ranking input (gap+rvol+sector+risk)


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
    # ── OOPS utility optimization (advisory; chooses the best option expression) ──
    oops_score: float = 0.0
    expected_utility: float = 0.0
    certainty_equivalent: float = 0.0
    probability_profit: float = 0.0
    probability_total_loss: float = 0.0
    expected_return: float = 0.0
    median_return: float = 0.0
    downside_tail_risk: float = 0.0
    skew: float = 0.0
    kurtosis: float = 0.0
    combined_option_score: float = 0.0
    selected_by_oops: bool = False
    oops_reject_reason: str = ""
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
    # ── shared streaming-indicator provenance (recorded by the Research Lab) ──────
    indicator_source: str = "fallback_static"   # "streaming" | "fallback_static"
    indicator_age_seconds: float | None = None
    streaming_indicator_available: bool = False
    fallback_used: bool = True
    # ── premarket live-feed awareness (recorded by the Research Lab) ──────────────
    premarket_source: str = ""
    premarket_gap_pct: float | None = None
    premarket_volume: float | None = None
    premarket_high: float | None = None
    premarket_low: float | None = None
    premarket_relative_strength: float | None = None
    premarket_stale: bool = False
    premarket_rank_score: float | None = None
    # ── OptionLab payoff/risk profile (advisory; validates the chosen call) ───────
    optionlab_score: float = 0.0
    break_even_price: float | None = None
    break_even_distance_pct: float | None = None
    profitable_price_range: list | None = None
    max_loss: float | None = None
    max_gain_estimate: float | None = None
    payoff_asymmetry: float | None = None
    payoff_quality: str = ""
    optionlab_greeks: dict | None = None
    optionlab_reject_reason: str = ""


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
    if _in_test():
        return  # never write to production journals during test runs
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


def detect_orb_sweep_reclaim(bars_1m: pd.DataFrame, orb_minutes: int = 15,
                             candle_minutes: int = 5, require_reclaim: bool = True
                             ) -> tuple[bool, float, str]:
    """3-tuple wrapper (signal, entry_price, status) for the live gate — drops the timestamp.
    See _orb_sweep_reclaim_full for the pattern definition and the engulf timestamp."""
    sig, px, status, _ts = _orb_sweep_reclaim_full(bars_1m, orb_minutes, candle_minutes, require_reclaim)
    return sig, px, status


def _orb_sweep_reclaim_full(bars_1m: pd.DataFrame, orb_minutes: int = 15,
                            candle_minutes: int = 5, require_reclaim: bool = True):
    """Liquidity-sweep ORB reversal (long): the underlying SWEEPS above the opening-range high,
    PULLS BACK into the range, then a BULLISH candle ENGULFS the pullback candle → entry at the
    CLOSE of that engulfing bar. Higher-quality than a naive breakout (filters the stop-hunt).

    Sequence on `candle_minutes`-resampled bars AFTER the 09:30..09:30+orb_minutes opening range:
      1. SWEEP    — a candle's HIGH > OR_high (took out the high / grabbed liquidity)
      2. PULLBACK — a later candle CLOSES back below OR_high (failed breakout, back in range)
      3. ENGULF   — a BULLISH candle (close>open) whose body engulfs the pullback candle's body
                    (+ reclaim back above OR_high if require_reclaim)
    Returns (signal, entry_price, status, engulf_timestamp). entry_price = engulfing bar close.
    status: "ok" | "no_bars" | "no_session_open_bars" | "no_sweep" | "swept_no_pullback" |
    "pullback_no_engulf" | "error".
    """
    try:
        if bars_1m is None or bars_1m.empty:
            return False, 0.0, "no_bars", None
        idx = bars_1m.index
        if not isinstance(idx, pd.DatetimeIndex):
            return False, 0.0, "error", None
        idx_et = idx.tz_convert(ET) if idx.tz is not None else idx.tz_localize("UTC").tz_convert(ET)
        df = bars_1m.copy(); df.index = idx_et
        session_date = idx_et[-1].date()
        sopen = pd.Timestamp(session_date.year, session_date.month, session_date.day, 9, 30, tz=ET)
        oend = sopen + timedelta(minutes=orb_minutes)
        ow = df[(df.index >= sopen) & (df.index < oend)]
        if ow.empty:
            return False, 0.0, "no_session_open_bars", None
        or_high = float(ow["high"].max())
        post = df[df.index >= oend]
        if post.empty:
            return False, 0.0, "no_sweep", None
        # resample post-open bars to the pattern timeframe
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        candles = post.resample(f"{candle_minutes}min").agg(agg).dropna()
        if candles.empty:
            return False, 0.0, "no_sweep"
        state = "seek_sweep"; pullback = None
        for ts, c in candles.iterrows():
            o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
            if state == "seek_sweep":
                if h > or_high:
                    state = "seek_pullback"
            elif state == "seek_pullback":
                if cl < or_high:                                   # pulled back into the range
                    pullback = (o, cl); state = "seek_engulf"
            elif state == "seek_engulf":
                p_open, p_close = pullback
                bullish = cl > o
                engulfs = o <= max(p_open, p_close) and cl >= max(p_open, p_close) and cl >= p_open
                reclaim = (cl > or_high) if require_reclaim else True
                if bullish and engulfs and reclaim:
                    return True, cl, "ok", ts          # ts = engulf-candle (bar-open) timestamp
                # if a fresh lower pullback prints, track the most recent down bar
                if cl < o and cl < p_close:
                    pullback = (o, cl)
        return False, 0.0, {"seek_sweep": "no_sweep", "seek_pullback": "swept_no_pullback",
                            "seek_engulf": "pullback_no_engulf"}[state], None
    except Exception:
        return False, 0.0, "error", None


def liq_sweep_latest(bars: pd.DataFrame, config: "AlphaCallConfig") -> Optional[dict]:
    """Latest Liquidity-Sweep Trend+Delta signal on the UNDERLYING, on DAILY bars. Resamples
    intraday bars to daily first (the signal is a multi-day trend). None if data insufficient."""
    if not getattr(config, "liq_sweep_enabled", False) or bars is None or getattr(bars, "empty", True):
        return None
    try:
        df = bars
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex) and len(idx) >= 2:
            if abs((idx[-1] - idx[-2]).total_seconds()) < 23 * 3600:     # intraday → resample to daily
                df = df.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                            "close": "last", "volume": "sum"}).dropna()
        import liquidity_sweep_signal as _lss
        return _lss.latest_signal(df, config.liq_sweep_len, config.liq_sweep_mult)
    except Exception:
        return None


_EXIT_CTX_CACHE: dict = {}    # symbol -> (ts, underlying_price, targets, liq_sell)


def position_exit_context(symbol: str, config: "AlphaCallConfig", ttl: float = 300.0):
    """Fail-safe (underlying_price, analyst_targets, liq_sweep_sell) for an OPEN position's exit
    filters. Cached per symbol (ttl s). Returns (None, None, False) on any failure — the existing
    exit engine then governs (per spec: missing data → do nothing)."""
    try:
        now = time.time()
    except Exception:
        import time as _t; now = _t.time()
    c = _EXIT_CTX_CACHE.get(symbol)
    if c and (now - c[0]) < ttl:
        return c[1], c[2], c[3]
    price = None; targets = None; liq_sell = False
    try:
        import analyst_target_exit as _ate
        targets = _ate.get_analyst_targets(symbol)
    except Exception:
        targets = None
    try:
        import yfinance as yf
        h = yf.Ticker(symbol).history(period="8mo", interval="1d")
        if h is not None and not h.empty:
            h = h.rename(columns=str.lower)
            price = float(h["close"].iloc[-1])
            if getattr(config, "liq_sweep_enabled", False):
                ls = liq_sweep_latest(h, config)
                liq_sell = bool(ls and ls["sellSignal"])
    except Exception:
        pass
    _EXIT_CTX_CACHE[symbol] = (now, price, targets, liq_sell)
    return price, targets, liq_sell


def orb_gate(ev: "SetupEvaluation", config: "AlphaCallConfig",
             now_et: Optional[datetime] = None) -> tuple[bool, str]:
    """HARD Opening-Range-Breakout gate, evaluated on the UNDERLYING before contract selection.
    A long call is allowed ONLY when, after 09:30+orb_minutes ET, the underlying has broken
    above the opening-range high. Returns (allowed, reason):
        waiting_for_opening_range  — before 09:45 ET (range still forming)
        opening_range_not_ready    — no bars in the 09:30-09:45 window (orb_status != ok)
        waiting_for_ORB_breakout   — price (or last close) is at/below the opening-range high
    No entry below the range, inside the range, or before the range is complete.
    """
    if not config.orb_gate_enabled:
        return True, ""
    now = now_et or datetime.now(ET)
    open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    range_end = open_t + timedelta(minutes=config.orb_minutes)   # 09:45 by default
    if now < range_end:
        return False, "waiting_for_opening_range"
    if ev.orb_status != "ok" or ev.orb_high <= 0:
        return False, "opening_range_not_ready"
    # SWEEP-RECLAIM mode: require the full sweep→pullback→bullish-engulf pattern on the underlying.
    if config.orb_mode == "sweep_reclaim":
        if ev.orb_pattern_status in ("no_bars", "no_session_open_bars", "error", ""):
            return False, "opening_range_not_ready"
        if not ev.orb_signal:
            return False, "waiting_for_ORB_sweep_reclaim"   # no sweep/pullback/engulf yet
        return True, ""
    # BREAKOUT mode: underlying last (or last completed close if required) must EXCEED ORB high.
    px = ev.orb_last_close if (config.orb_require_close and ev.orb_last_close > 0) else ev.price
    if px <= ev.orb_high:
        return False, "waiting_for_ORB_breakout"
    return True, ""


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


def _resolve_indicator_state(symbol: str, indicator_snapshot: dict | None) -> tuple[dict | None, bool]:
    """
    Return (usable_state, is_fresh) for a symbol from a StreamingIndicatorEngine
    snapshot. State is usable only when present and NOT stale; otherwise callers
    fall back to bar-based calculations (indicator_source='fallback_static').
    """
    if not indicator_snapshot:
        return None, False
    state = indicator_snapshot.get(symbol)
    if not state:
        return None, False
    if state.get("stale"):
        return state, False
    return state, True


def _resolve_premarket_state(symbol: str, premarket_snapshot: dict | None):
    """Return (symbol_quote, all_quotes) from a PremarketLiveFeed snapshot, or (None, {})."""
    if not premarket_snapshot:
        return None, {}
    quotes = premarket_snapshot.get("quotes") or {}
    return quotes.get(symbol), quotes


def _premarket_benchmark_gap(quotes: dict) -> float | None:
    """Fresh premarket gap% of a broad benchmark (SPY/QQQ/IWM) for relative strength."""
    for b in ("SPY", "QQQ", "IWM"):
        q = quotes.get(b)
        if q and not q.get("stale") and q.get("premarket_change_pct") is not None:
            try:
                return float(q["premarket_change_pct"])
            except Exception:
                return None
    return None


def detect_setup(
    symbol: str,
    bars_1m: pd.DataFrame,
    config: AlphaCallConfig,
    market_context: dict,
    indicator_snapshot: dict | None = None,
    premarket_snapshot: dict | None = None,
) -> SetupEvaluation:
    """Score each symbol for A/A+ call setups. Returns a SetupEvaluation.

    When a fresh StreamingIndicatorEngine snapshot is supplied, its VWAP is used
    for the VWAP-based gates (shared real-time indicator state). If the snapshot is
    missing or stale, the engine falls back to the existing bar-based VWAP and the
    evaluation is marked indicator_source='fallback_static'. Streaming indicators
    only inform timing — they never create or block an order on their own.
    """
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

    # ── shared real-time indicator state (timing only) ─────────────────────────
    # Resolved UP-FRONT so provenance is recorded even when the setup is later
    # rejected (e.g. by the regime gate). A fresh streaming VWAP is applied below.
    _ind_state, _ind_fresh = _resolve_indicator_state(symbol, indicator_snapshot)
    _stream_vwap: float | None = None
    if _ind_state is not None:
        ev.streaming_indicator_available = bool(_ind_state.get("streaming_indicator_available"))
        ev.indicator_age_seconds = _ind_state.get("indicator_age_seconds")
    if _ind_fresh:
        _sv = (_ind_state.get("indicators") or {}).get("vwap")
        if isinstance(_sv, (int, float)) and _sv > 0:
            _stream_vwap = float(_sv)
            ev.indicator_source = "streaming"
            ev.fallback_used = False

    # ── premarket live-feed awareness (recorded even on later rejection) ────────
    _pm_q, _pm_quotes = _resolve_premarket_state(symbol, premarket_snapshot)
    if _pm_q is not None:
        ev.premarket_source = _pm_q.get("source") or ""
        ev.premarket_stale = bool(_pm_q.get("stale"))
        ev.premarket_gap_pct = _pm_q.get("premarket_change_pct")
        ev.premarket_volume = _pm_q.get("premarket_volume")
        ev.premarket_high = _pm_q.get("premarket_high")
        ev.premarket_low = _pm_q.get("premarket_low")
        _bench = _premarket_benchmark_gap(_pm_quotes)
        if ev.premarket_gap_pct is not None:
            try:
                ev.premarket_relative_strength = round(float(ev.premarket_gap_pct) - (_bench or 0.0), 4)
            except Exception:
                ev.premarket_relative_strength = None

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
    # Prefer the fresh streaming VWAP (resolved above); else the bar-based VWAP.
    vwap = _stream_vwap if _stream_vwap is not None else _compute_vwap(bars_1m)
    rvol = _compute_rvol(bars_1m)
    orb_high, orb_low, orb_status = _compute_orb(bars_1m, config.orb_minutes)
    above_vwap = price >= vwap

    ev.price = price
    ev.vwap = vwap
    ev.rvol = rvol
    ev.orb_high = orb_high
    ev.orb_low = orb_low
    ev.orb_status = orb_status
    try:
        ev.orb_last_close = float(bars_1m["close"].iloc[-1]) if bars_1m is not None and not bars_1m.empty else 0.0
    except Exception:
        ev.orb_last_close = 0.0
    # Sweep-reclaim ORB pattern (only when configured) — needs intraday OHLC.
    if config.orb_mode == "sweep_reclaim" and bars_1m is not None and not bars_1m.empty \
            and "open" in getattr(bars_1m, "columns", []):
        sig, entry_px, pat_status = detect_orb_sweep_reclaim(
            bars_1m, config.orb_minutes, config.orb_candle_minutes, config.orb_require_reclaim)
        ev.orb_signal = sig; ev.orb_pattern_status = pat_status; ev.orb_entry_price = entry_px  # noqa
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
    elif "mixed" in bias or "cautious" in bias:
        score += 3
        notes.append(f"regime={bias}(+3,mixed_caution)")

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

    # ── Premarket awareness bonus (improves scoring only; never creates a trade) ──
    # Only strengthens an ALREADY-detected setup (best_setup != NONE) and only from a
    # FRESH premarket quote. price > premarket_high + premarket relative strength +
    # risk-on regime ⇒ stronger Alpha setup. Bounded to +8 so premarket data can never
    # lift a non-setup over the entry threshold on its own.
    if best_setup != "NONE" and ev.premarket_high and not ev.premarket_stale:
        pm_bonus = 0.0
        if price > float(ev.premarket_high):
            pm_bonus += 5.0
            notes.append(f"premarket_breakout px={price:.2f}>pmH={float(ev.premarket_high):.2f}(+5)")
        if (ev.premarket_relative_strength or 0.0) > 0 and (
                "bull" in bias or "risk_on" in bias or "uptrend" in bias):
            pm_bonus += 3.0
            notes.append(f"premarket_rel_strength={ev.premarket_relative_strength:.2f}(+3)")
        score = min(100, score + min(8.0, pm_bonus))

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

    # ── Embedded leverage (options_quant / Frazzini-Pedersen) ────────────────────
    # Ω = (S/premium)·δ. High embedded leverage (short-dated OTM/ATM) carries a NEGATIVE
    # expected premium for the buyer; leverage-efficiency peaks in the moderate (deep-ITM)
    # band. This research-grounded score REPLACES the old ad-hoc ATM-moneyness preference
    # (which had no basis and fought the leverage finding) inside bs_score, and additionally
    # tilts the composite. Fully fail-safe.
    omega = 0.0
    lev_eff = 1.0
    if config.lev_select_enabled and mid > 0:
        try:
            import options_quant as _oq
            omega = _oq.embedded_leverage(S, mid, float(c["delta"]))
            lev_eff = _oq.leverage_efficiency_score(omega, config.lev_band_lo, config.lev_band_hi)
        except Exception:
            omega, lev_eff = 0.0, 1.0

    # ── black_scholes_score (0-100): weighted valuation + convexity + quality ────
    edge_score = _clip(0.5 - mispricing_pct * 2.0, 0.0, 1.0)             # cheaper-than-theo is better
    delta_score = _clip(1.0 - abs(c["delta"] - 0.575) / 0.25, 0.0, 1.0)  # ideal mid of 0.45-0.70
    gt_score = _clip(gamma_theta_ratio / 8.0, 0.0, 1.0)
    spread_score = _clip(1.0 - c["spread_pct"] / max(config.max_spread_pct, 1e-6), 0.0, 1.0)
    oi_score = _clip(int(c["open_interest"]) / 1000.0, 0.0, 1.0)
    vol_score = _clip(int(c["volume"]) / 100.0, 0.0, 1.0)
    liquidity_score = 0.6 * oi_score + 0.4 * vol_score
    dte_score = _clip(1.0 - abs(dte - 37.5) / 22.5, 0.0, 1.0)            # peak ~37 DTE
    # SPREAD-AWARE leverage component: spread_tilt ∈ [0,1] grows with the contract's spread.
    # Liquid (tight spread) → ATM-moneyness preference (convexity wins); wide spread → FP
    # leverage-efficiency (deep-ITM, since ATM's extrinsic is taxed away). Drives both the
    # bs_score leverage term AND the multiplicative tilt below, so the two stay coherent.
    legacy_moneyness_score = _clip(1.0 - abs(moneyness - 1.0) / 0.10, 0.0, 1.0)
    if config.lev_select_enabled and config.lev_spread_aware:
        spread_tilt = _clip(float(c["spread_pct"]) / max(config.lev_spread_ref, 1e-6), 0.0, 1.0)
    else:
        spread_tilt = 1.0 if config.lev_select_enabled else 0.0
    leverage_score = (1.0 - spread_tilt) * legacy_moneyness_score + spread_tilt * lev_eff
    iv_score = 1.0 if 0.20 <= iv <= 0.80 else _clip(1.0 - (max(0.0, iv - 0.80) + max(0.0, 0.20 - iv)) / 1.0, 0.0, 1.0)
    bs_score = 100.0 * (
        0.22 * edge_score + 0.16 * delta_score + 0.16 * gt_score +
        0.12 * spread_score + 0.12 * liquidity_score + 0.08 * dte_score +
        0.08 * leverage_score + 0.06 * iv_score
    )
    # Expected-R profile refines the final selection (step 5): up to +15 for strong convexity.
    composite = bs_score + _clip(expected_r, 0.0, 5.0) * 3.0
    # Multiplicative leverage tilt — weight scaled by spread_tilt so it only bites on wide-spread
    # (illiquid) names. Refines, never overrides, the BS valuation.
    if config.lev_select_enabled:
        w = _clip(config.lev_select_weight, 0.0, 1.0) * spread_tilt
        composite = composite * (1.0 - w + w * lev_eff)

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
        "embedded_leverage": round(omega, 2),
        "leverage_efficiency": round(lev_eff, 3),
        "spread_tilt": round(spread_tilt, 3),
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


def _apply_oops_fields(sig: "CallSignal", contract: dict, selected_by_oops: bool) -> None:
    """Copy OOPS utility-optimization fields onto a signal (advisory diagnostics)."""
    sig.selected_by_oops = bool(selected_by_oops)
    if "oops_score" not in contract:
        sig.oops_reject_reason = "oops_unavailable_bs_fallback"
        return
    sig.oops_score = float(contract.get("oops_score", 0.0) or 0.0)
    sig.expected_utility = float(contract.get("expected_utility", 0.0) or 0.0)
    sig.certainty_equivalent = float(contract.get("certainty_equivalent", 0.0) or 0.0)
    sig.probability_profit = float(contract.get("probability_profit", 0.0) or 0.0)
    sig.probability_total_loss = float(contract.get("probability_total_loss", 0.0) or 0.0)
    sig.expected_return = float(contract.get("expected_return", 0.0) or 0.0)
    sig.median_return = float(contract.get("median_return", 0.0) or 0.0)
    sig.downside_tail_risk = float(contract.get("downside_tail_risk", 0.0) or 0.0)
    sig.skew = float(contract.get("skew", 0.0) or 0.0)
    sig.kurtosis = float(contract.get("kurtosis", 0.0) or 0.0)
    sig.combined_option_score = float(contract.get("combined_option_score", 0.0) or 0.0)


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

    # ── hard gate — robust to missing Alpaca snapshot data ───────────────────
    # Alpaca contract metadata initialises volume, delta, bid, and ask to 0 / 0.0
    # when the options-chain snapshot fetch fails or times out.  The previous
    # hard-reject on vol==0 or mid==0 silently dropped every contract for those
    # symbols.  Strategy when feed data is absent:
    #   • vol  == 0  →  data not available (not literally zero volume); skip gate
    #   • delta == 0 →  estimate from Black-Scholes using feed-IV or fallback IV
    #   • mid  == 0  →  estimate from Black-Scholes; mark estimated_mid=True
    # Estimated values still go through the full BS valuation layer which will
    # reject them if the theoretical price / gamma-theta don't make sense.
    _DEFAULT_IV_FALLBACK = 0.40   # conservative 40% when feed IV is absent

    hard_survivors: list[dict] = []
    # Per-gate rejection counters for CHAIN_DIAG diagnostics
    _hg_n_put = 0; _hg_n_dte = 0; _hg_n_strike = 0; _hg_n_delta = 0
    _hg_n_oi = 0; _hg_n_vol = 0; _hg_n_mid = 0; _hg_n_spread = 0
    for contract in chain:
        try:
            if str(contract.get("type", "")).lower() != "call":
                _hg_n_put += 1; continue
            expiry_str = str(contract.get("expiration_date") or contract.get("expiry") or "")
            if not expiry_str:
                _hg_n_dte += 1; continue
            dte = (datetime.strptime(expiry_str[:10], "%Y-%m-%d").date() - today).days
            if not (config.min_dte <= dte <= config.max_dte):
                _hg_n_dte += 1; continue
            K = float(contract.get("strike_price") or contract.get("strike") or 0)
            if K <= 0 or price <= 0:
                _hg_n_strike += 1; continue
            T = max(dte, 1) / 365.0
            feed_iv = float(contract.get("implied_volatility") or 0.0)

            # ── delta ─────────────────────────────────────────────────────────
            feed_delta = abs(float(
                contract.get("delta") or contract.get("greeks", {}).get("delta", 0) or 0))
            estimated_delta = False
            if feed_delta == 0.0:
                # Snapshot failed: estimate from BS with feed-IV or conservative fallback.
                iv_for_est = feed_iv if feed_iv > 0.01 else _DEFAULT_IV_FALLBACK
                _bs_est = black_scholes_call(price, K, T, r, iv_for_est, config.dividend_yield)
                feed_delta = abs(_bs_est["delta"])
                estimated_delta = True
            delta = feed_delta
            if not (config.min_delta <= delta <= config.max_delta):
                _hg_n_delta += 1; continue

            # ── OI ────────────────────────────────────────────────────────────
            # oi==0 means data not available (same pattern as vol==0); do NOT
            # hard-reject — Alpaca snapshot timeouts return 0 for all contracts.
            oi = int(contract.get("open_interest") or 0)
            if oi > 0 and oi < config.min_open_interest:   # only reject confirmed low OI
                _hg_n_oi += 1; continue

            # ── volume: vol==0 means data not available, not literally zero ───
            vol = int(contract.get("volume") or 0)
            if vol > 0 and vol < config.min_volume:   # only reject on confirmed low vol
                _hg_n_vol += 1; continue

            # ── mid price ─────────────────────────────────────────────────────
            bid = float(contract.get("bid_price") or contract.get("bid") or 0)
            ask = float(contract.get("ask_price") or contract.get("ask") or 0)
            estimated_mid = False
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
            else:
                # Snapshot failed: estimate from BS; BS layer will verify / reject.
                iv_for_est = feed_iv if feed_iv > 0.01 else _DEFAULT_IV_FALLBACK
                _bs_est = black_scholes_call(price, K, T, r, iv_for_est, config.dividend_yield)
                mid = _bs_est["price"]
                bid = ask = mid    # symmetric; spread_pct will be 0
                estimated_mid = True

            if mid <= 0:
                _hg_n_mid += 1; continue

            spread_pct = 0.0 if estimated_mid else ((ask - bid) / mid if mid > 0 else 1.0)
            # Spread band [min, max] — applied only to REAL spreads; estimated-mid contracts
            # (snapshot failed, spread=0) are exempt and go to the BS valuation layer instead.
            if not estimated_mid and (spread_pct > config.max_spread_pct
                                      or spread_pct < config.min_spread_pct):
                _hg_n_spread += 1; continue

            # IV-regime gate (feature-flagged, default OFF) — only on REAL feed IV (>0).
            if config.iv_gate_enabled and feed_iv > 0.0 and not (
                    config.iv_gate_min <= feed_iv <= config.iv_gate_max):
                _hg_n_spread += 1; continue   # counted with spread rejections (both are quality gates)

            hard_survivors.append({
                "contract_symbol": str(contract.get("symbol") or ""),
                "expiry": expiry_str[:10],
                "strike": K, "delta": delta, "dte": dte,
                "bid": bid, "ask": ask, "mid": mid,
                "open_interest": oi, "volume": vol, "spread_pct": spread_pct,
                "implied_volatility": feed_iv,
                "estimated_mid": estimated_mid,
                "estimated_delta": estimated_delta,
            })
        except Exception:
            continue

    counts = {
        "hard_gate_survivors": len(hard_survivors),
        # Per-gate rejection counters for diagnosing hard_survivors=0
        "hg_rejected_puts": _hg_n_put,
        "hg_rejected_dte": _hg_n_dte,
        "hg_rejected_strike": _hg_n_strike,
        "hg_rejected_delta": _hg_n_delta,
        "hg_rejected_oi": _hg_n_oi,
        "hg_rejected_vol": _hg_n_vol,
        "hg_rejected_mid": _hg_n_mid,
        "hg_rejected_spread": _hg_n_spread,
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
    """Return (contracts, premium_cost, max_risk) for a call trade.

    NOTE: callers that apply a mixed-regime risk multiplier should pass an
    overridden available-budget cap separately (see _eff_risk_pct in AlphaCallEngine.run).
    This function uses config.risk_per_trade_pct as the base position ceiling.
    """
    cost_per_contract = mid * 100  # options multiplier
    if cost_per_contract <= 0:
        return 0, 0.0, 0.0
    # Fractional Kelly on the account equity, capped by risk_per_trade_pct (Kelly can only
    # shrink size below the cap). Matches the live evaluate_universe sizing path.
    risk_cap_dollars = equity * config.risk_per_trade_pct
    if config.kelly_sizing_enabled:
        kelly_dollars = equity * config.kelly_full * config.kelly_mult
        max_risk_dollars = min(kelly_dollars, risk_cap_dollars)
    else:
        max_risk_dollars = risk_cap_dollars
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


def _alpha_entry_window_open() -> bool:
    """Regular-session buy window for NEW alpha entries only."""
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    entry_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    entry_cutoff = now.replace(hour=15, minute=45, second=0, microsecond=0)
    return entry_open <= now < entry_cutoff


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
        else:
            # Clear a setup-quality reject when score meets the screener watch
            # threshold.  detect_setup() gates at config.min_setup_score (55) and
            # pre-sets reject_reason for any score below that.  Without this clear
            # the screener's lower floor (default 45) never fires because
            # `not ev.reject_reason` is False for scores 45-54 — causing 60/62
            # symbols to be rejected by "momentum" even when they have enough
            # momentum to warrant chain fetching.
            if (ev.reject_reason
                    and ev.reject_reason.startswith("rejected_by_setup_quality")
                    and ev.setup_score >= cfg.screener_momentum_floor):
                ev.reject_reason = None
            if ev.setup_score >= cfg.screener_momentum_floor and not ev.reject_reason:
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
        self._oops_cfg = oops_opt.OOPSConfig()      # OOPS options utility optimizer (advisory)

    # ── state persistence ─────────────────────────────────────────────────────
    def _load_state(self) -> dict[str, ActiveCall]:
        # Try today's state file first
        if ALPHA_STATE_PATH.exists():
            try:
                raw = json.loads(ALPHA_STATE_PATH.read_text())
                loaded = {
                    k: ActiveCall(**v) for k, v in raw.items()
                    if all(f in v for f in ActiveCall.__dataclass_fields__)
                }
                if loaded:
                    return loaded
            except Exception:
                pass

        # Fall back to yesterday's state file (carry-over overnight positions)
        # These will be immediately flagged for closure by stale_overnight logic
        try:
            from datetime import timedelta
            yesterday_tag = (datetime.now(ET) - timedelta(days=1)).strftime("%Y%m%d")
            yesterday_path = (
                PROJECT_ROOT / "system-data" / yesterday_tag
                / f"{yesterday_tag}_alpha_engine_state.json"
            )
            if yesterday_path.exists():
                raw = json.loads(yesterday_path.read_text())
                carried = {
                    k: ActiveCall(**v) for k, v in raw.items()
                    if all(f in v for f in ActiveCall.__dataclass_fields__)
                }
                if carried:
                    _alpha_event("state_carried_from_yesterday", {
                        "yesterday_tag": yesterday_tag,
                        "positions": list(carried.keys()),
                    })
                    return carried
        except Exception as _e:
            _alpha_event("state_carryover_error", {"error": str(_e)})

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
        indicator_snapshot: dict | None = None,
        market_state: dict | None = None,
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

        # Dynamic-exit profile (rl_engine_levers) — gated, default OFF, paper-only, and only
        # active when the RL deployment gate rates alpha_exit ≥ paper_mode_only. When active it
        # ONLY re-times the partial/trail triggers (partial at +1R, trail the runner 1R below its
        # peak). The hard stop, 3R cap, kill switch, daily-loss halt, and RiskManager below are
        # untouched and always take priority. None → the engine's existing triggers run unchanged.
        _dyn = None
        try:
            import rl_engine_levers as _levers
            if _levers.dynamic_exit_active():
                _dyn = _levers.exit_profile()
                _alpha_event("dynamic_exit_active", {"profile": _dyn})
        except Exception:
            _dyn = None

        # Diagnostic: confirm exit pass is running each cycle
        _alpha_event("exit_pass_start", {"active_positions": len(self._active),
                                          "option_positions_from_broker": len(options_positions)})

        for key, call in list(self._active.items()):
            # FIX 2026-06-10: Purge stale/test positions whose OCC contract has already expired.
            # These can appear as phantom broker positions (e.g. XYZ250620C00100000 from a prior
            # test session) that keep triggering the hard-stop loop each cycle.  Parse the YYMMDD
            # block from the OCC symbol; if the expiry is ≥ 2 days in the past, remove from
            # _active, log a warning, and skip — no exit order is generated (Alpaca would reject
            # an order on an expired series anyway).
            try:
                import re as _re
                _m = _re.match(r"^[A-Z]+(\d{2})(\d{2})(\d{2})[CP]\d+$",
                               (call.contract_symbol or "").strip().upper())
                if _m:
                    _yy, _mm, _dd = (int(g) for g in _m.groups())
                    from datetime import date as _date_cls, datetime as _dt_cls, timezone as _tz_cls
                    _exp = _date_cls(2000 + _yy, _mm, _dd)
                    _dte = (_exp - _dt_cls.now(_tz_cls.utc).date()).days
                    if _dte < -1:
                        _alpha_event("stale_contract_purged", {
                            "contract": call.contract_symbol,
                            "underlying": call.symbol,
                            "expiry": str(_exp),
                            "dte": _dte,
                        })
                        logger.warning(
                            "ALPHA purging expired contract from active state: %s "
                            "(underlying=%s expiry=%s dte=%d) — phantom broker artifact",
                            call.contract_symbol, call.symbol, _exp, _dte,
                        )
                        del self._active[key]
                        self._save_state()
                        continue
            except Exception:
                pass

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

            # ── EOD forced close (overnight_allowed=False) ─────────────────────
            # Positions opened with overnight_allowed=False must be closed before
            # the options market closes. Force-close at/after 15:45 ET so the order
            # reaches the exchange before 16:00 ET.
            # NEW: Also force-close immediately on a NEW TRADING DAY if the position
            # was opened on a prior calendar date — catches cases where the main loop
            # was terminated before 15:45 ET and the position was never closed EOD.
            if not call.overnight_allowed:
                try:
                    _now_et = datetime.now(ET)
                    _eod_cutoff = _now_et.replace(hour=15, minute=45, second=0, microsecond=0)
                    # Check if this is a carry-over from a previous trading day
                    _opened_date = None
                    try:
                        from datetime import date as _date
                        _opened_date = datetime.fromisoformat(
                            call.opened_at_utc.replace("Z", "+00:00")
                        ).astimezone(ET).date()
                    except Exception:
                        pass
                    _is_stale_overnight = (
                        _opened_date is not None
                        and _opened_date < _now_et.date()
                    )
                    if _is_stale_overnight or _now_et >= _eod_cutoff:
                        _reason_tag = "stale_overnight_carry_over" if _is_stale_overnight else "eod_forced_close"
                        _pnl = current_value - call.entry_cost
                        exit_orders.append({
                            "symbol": call.contract_symbol,
                            "side": "sell",
                            "qty": qty_held,
                            "source": f"alpha_call_{_reason_tag}",
                            "reason": f"{_reason_tag}:overnight_not_allowed:time={_now_et.strftime('%H:%M')}ET:opened={_opened_date}",
                            "underlying": call.symbol,
                            "setup_type": call.setup_type,
                            "realized_pnl": round(_pnl, 2),
                            "r_multiple": round(_pnl / call.entry_cost, 3) if call.entry_cost else 0.0,
                            "probability_band": call.probability_band,
                            "probability_of_profit": call.probability_of_profit,
                            "management_style": call.management_style,
                        })
                        _alpha_event("call_eod_forced_close", {
                            "symbol": call.symbol, "contract": call.contract_symbol,
                            "reason": _reason_tag,
                            "time_et": _now_et.strftime("%H:%M"),
                            "opened_date": str(_opened_date),
                            "pnl": round(_pnl, 2),
                            "pnl_pct": round(pnl_pct * 100, 2),
                        })
                        if waterfall_callback and _pnl > 0:
                            waterfall_callback(_pnl, source="alpha_eod_close")
                        del self._active[key]; self._save_state(); continue
                except Exception as _eod_err:
                    _alpha_event("eod_close_check_error", {"error": str(_eod_err)})

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

            # ── Liquidity-sweep trend reversal + Analyst price-target exits ──────
            # Run AFTER emergency exits (hard/EOD stops above) but BEFORE profit-trailing.
            # Fail-safe: missing underlying price / analyst targets → skip (existing engine governs).
            _u_price, _targets, _liq_sell = position_exit_context(call.symbol, self.config)
            if self.config.liq_sweep_enabled and self.config.liq_sweep_sell_exit and _liq_sell:
                _pnl = current_value - call.entry_cost
                exit_orders.append({
                    "symbol": call.contract_symbol, "side": "sell", "qty": qty_held,
                    "source": "alpha_call_liq_sweep_sell", "reason": "liq_sweep_sell_signal",
                    "underlying": call.symbol, "setup_type": call.setup_type,
                    "realized_pnl": round(_pnl, 2),
                    "r_multiple": round(_pnl / call.entry_cost, 3) if call.entry_cost else 0.0,
                    "probability_band": call.probability_band,
                    "probability_of_profit": call.probability_of_profit,
                    "management_style": call.management_style,
                })
                _alpha_event("call_liq_sweep_sell", {"symbol": call.symbol, "contract": call.contract_symbol})
                del self._active[key]; self._save_state(); continue
            if _u_price and _targets:
                try:
                    import analyst_target_exit as _ate
                    _ae = _ate.analyst_target_exit(_u_price, _targets.get("high"), _targets.get("avg"),
                                                   _targets.get("low"), _targets.get("n"))
                except Exception:
                    _ae = {"exit": False, "tighten_trail": False, "reason": "analyst_error"}
                if _ae.get("exit"):
                    _pnl = current_value - call.entry_cost
                    exit_orders.append({
                        "symbol": call.contract_symbol, "side": "sell", "qty": qty_held,
                        "source": "alpha_call_analyst_target", "reason": _ae["reason"],
                        "underlying": call.symbol, "setup_type": call.setup_type,
                        "realized_pnl": round(_pnl, 2),
                        "r_multiple": round(_pnl / call.entry_cost, 3) if call.entry_cost else 0.0,
                        "probability_band": call.probability_band,
                        "probability_of_profit": call.probability_of_profit,
                        "management_style": call.management_style,
                    })
                    _alpha_event("call_analyst_target_exit", {
                        "symbol": call.symbol, "contract": call.contract_symbol,
                        "reason": _ae["reason"], "price": _u_price,
                        "upside_to_avg": _ae.get("upside_to_avg")})
                    del self._active[key]; self._save_state(); continue
                elif _ae.get("tighten_trail"):
                    # price ≥ avg analyst target → ENGAGE a tight protective trail NOW (give the
                    # winner room toward the high target, but lock gains since consensus upside is
                    # mostly used). tighten_mult 0.5 ⇒ stop sits 0.4R (=0.8R×0.5) behind the peak.
                    call.trail_tightness_mult = min(getattr(call, "trail_tightness_mult", 1.0) or 1.0, 0.5)
                    call.trail_active = True
                    _r = call.entry_cost or 1.0
                    _astop = call.peak_value - (self.config.r_trail_offset * call.trail_tightness_mult * _r)
                    if _astop > call.trail_stop_value:
                        call.trail_stop_value = _astop
                    self._save_state()
                    _alpha_event("call_analyst_tighten_trail", {
                        "symbol": call.symbol, "contract": call.contract_symbol,
                        "upside_to_avg": _ae.get("upside_to_avg"),
                        "trail_stop": round(call.trail_stop_value, 2)})

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
            _partial_trigger = (call.entry_cost * (1 + _dyn["partial_r"])) if _dyn else call.partial_value
            if not call.partial_taken and current_value >= _partial_trigger:
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
            _trail_activate_r = _dyn["trail_activate_r"] if _dyn else self.config.r_trail_activate
            _trail_offset_r = _dyn["trail_offset_r"] if _dyn else self.config.r_trail_offset
            trail_activate_value = call.entry_cost * (1 + _trail_activate_r)
            if current_value >= trail_activate_value:
                call.trail_active = True
                # Trail stop sits (offset × trail_tightness) × R below the peak.
                # trail_tightness_mult <= 1.0 ⇒ stop sits CLOSER to the peak (more
                # protective). Probability can only tighten this, never loosen it.
                offset_mult = min(getattr(call, "trail_tightness_mult", 1.0) or 1.0, 1.0)
                trail_stop = call.peak_value - (_trail_offset_r * offset_mult * r_amount)
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

            # ── OOPS exit assessment (ADVISORY; runs ONLY after all hard rules) ──
            # The hard stop / 3R / trailing / kill switch / RiskManager are all checked
            # above and `continue`/`del` before this point, so OOPS can NEVER override a
            # protective exit. It may only cut a deteriorating position early or tighten
            # the trail (both reduce risk). A "runner"/"hold" recommendation is advisory.
            if key in self._active and self._oops_cfg.enabled:
                try:
                    rec = oops_opt.oops_exit_recommendation(
                        current_value, call.entry_cost, dte_rem,
                        partial_taken=call.partial_taken, cfg=self._oops_cfg)
                except Exception:
                    rec = {"action": "hold"}
                if rec.get("action") == "exit":
                    _pnl = current_value - call.entry_cost
                    exit_orders.append({
                        "symbol": call.contract_symbol, "side": "sell", "qty": call.contracts_held,
                        "source": "alpha_call_oops_exit", "reason": rec.get("rationale", "oops_eu_decay"),
                        "underlying": call.symbol, "setup_type": call.setup_type,
                        "realized_pnl": round(_pnl, 2),
                        "r_multiple": round(_pnl / call.entry_cost, 3) if call.entry_cost else 0.0,
                        "probability_band": call.probability_band,
                        "probability_of_profit": call.probability_of_profit,
                        "management_style": call.management_style,
                    })
                    _alpha_event("call_oops_exit", {"symbol": call.symbol, "contract": call.contract_symbol,
                                                    "rationale": rec.get("rationale"), "value": current_value})
                    if waterfall_callback and _pnl > 0:
                        waterfall_callback(_pnl, source="alpha_oops_exit")
                    del self._active[key]; self._save_state()
                elif rec.get("action") == "tighten":
                    # Protective only: tighten the trail (never loosen).
                    new_mult = min(getattr(call, "trail_tightness_mult", 1.0) or 1.0,
                                   float(rec.get("trail_tightness_mult", 1.0)))
                    if new_mult < (call.trail_tightness_mult or 1.0):
                        call.trail_tightness_mult = new_mult
                        self._save_state()

        # Stamp exit-side diagnostics (market state + indicator provenance) onto every
        # exit order so the Research Lab records the context the exit fired under. This is
        # diagnostics only — it never changes whether an exit fires.
        _ms = market_state or {}
        _isnap = indicator_snapshot or {}
        for _ex in exit_orders:
            _ex.setdefault("market_state", _ms.get("regime"))
            _ex.setdefault("market_state_confidence", _ms.get("confidence"))
            _u = _ex.get("underlying") or _ex.get("symbol") or ""
            _snap = _isnap.get(_u) if isinstance(_isnap, dict) else None
            if isinstance(_snap, dict):
                _ex.setdefault("indicator_source", _snap.get("source"))
                _ex.setdefault("indicator_age_seconds", _snap.get("indicator_age_seconds"))
                _ex.setdefault("streaming_indicator_available", _snap.get("streaming_indicator_available"))

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
        indicator_snapshot: dict | None = None,
        premarket_snapshot: dict | None = None,
        market_state: dict | None = None,
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
        oops_all_rows: list[dict] = []     # OOPS-scored contracts across all symbols (diagnostic)
        oops_by_symbol: dict[str, dict] = {}
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
        entry_gate_reason = "entries_disabled"
        if not _alpha_entry_window_open():
            entries_enabled = False
            entry_gate_reason = "rejected_by_session:outside_entry_window"
        elif str(market_context.get("trade_permission") or "normal").lower() == "halt_new_buys":
            entries_enabled = False
            entry_gate_reason = "rejected_by_market_regime:halt_new_buys"
        # Regime gate (rl_engine_levers) — gated, default OFF, paper-only, active only when the
        # deployment gate rates alpha_entry ≥ paper_mode_only. Suppresses NEW entries when SPY is
        # not confirmed above its 200-DMA. Suppressive only: it can disable entries this cycle but
        # never forces a trade and never touches exits/sells/RiskManager (existing crisis-regime
        # halt in run.py is separate and still applies).
        regime_suppressed = False
        try:
            import rl_engine_levers as _levers
            if _levers.suppress_entry("alpha_entry", market_state, indicator_snapshot):
                regime_suppressed = True
                entries_enabled = False
                entry_gate_reason = "regime_gate:SPY_not_confirmed_above_200DMA"
                _alpha_event("regime_gate_suppressed_entries",
                             {"reason": "SPY_not_confirmed_above_200DMA", "lever": "ENABLE_RL_REGIME_GATE"})
        except Exception:
            regime_suppressed = False
        if exposure_blocked:
            _alpha_event("options_exposure_limit", {"exposure": options_exposure,
                "limit": equity * self.config.max_options_exposure_pct, "reason": "rejected_by_exposure_limit"})
        if positions_full:
            _alpha_event("max_positions_reached", {"count": len(self._active)})

        # Detect mixed/cautious regime so per-symbol logic can apply cautious sizing.
        # "mixed" and "cautious" are now in allowed_regimes but trigger reduced thresholds
        # and halved risk_per_trade to preserve capital under directional uncertainty.
        _cycle_bias = str(
            market_context.get("overall_market_bias") or market_context.get("market_bias") or ""
        ).lower()
        _is_mixed_regime = "mixed" in _cycle_bias or "cautious" in _cycle_bias
        # Effective minimum score: use mixed_regime_min_score on mixed/cautious days.
        _eff_min_score = (
            self.config.mixed_regime_min_score if _is_mixed_regime else self.config.min_setup_score
        )
        # Effective risk per trade: halved on mixed/cautious days.
        _eff_risk_pct = (
            self.config.risk_per_trade_pct * self.config.mixed_regime_risk_multiplier
            if _is_mixed_regime else self.config.risk_per_trade_pct
        )
        # Market-open confirmation guard: on mixed/cautious days block NEW entries during
        # the first 30 minutes after open (09:30-10:00 ET). This prevents entering options
        # contracts when the market's direction has not yet been confirmed — the most common
        # cause of immediate hard stops when regime flips bearish shortly after open.
        # Configurable via ALPHA_CAUTIOUS_OPEN_GUARD_MINUTES (default 30). Set to 0 to disable.
        _open_guard_blocked = False
        if _is_mixed_regime and entries_enabled:
            try:
                import pytz as _ptz
                _et_now = datetime.now(_ptz.timezone("America/New_York"))
                _market_open_et = _et_now.replace(hour=9, minute=30, second=0, microsecond=0)
                _guard_mins = int(os.getenv("ALPHA_CAUTIOUS_OPEN_GUARD_MINUTES", "30"))
                _guard_cutoff = _market_open_et + __import__("datetime").timedelta(minutes=_guard_mins)
                if _market_open_et <= _et_now < _guard_cutoff:
                    _open_guard_blocked = True
                    entries_enabled = False
                    _alpha_event("open_guard_blocked_cautious_entries", {
                        "reason": f"mixed/cautious regime within first {_guard_mins}min of open",
                        "et_now": _et_now.strftime("%H:%M:%S"),
                        "guard_until": _guard_cutoff.strftime("%H:%M:%S"),
                    })
            except Exception:
                pass  # fail open — guard is advisory

        for symbol in symbols:
            bars = bars_by_symbol.get(symbol)
            if bars is None:
                bars = pd.DataFrame()
            ev = detect_setup(symbol, bars, self.config, market_context,
                              indicator_snapshot, premarket_snapshot)
            urow = _alpha_momentum_row(symbol, ev, market_context)
            urow["indicator_source"] = ev.indicator_source
            urow["fallback_used"] = ev.fallback_used
            urow["premarket_source"] = ev.premarket_source
            urow["premarket_gap_pct"] = ev.premarket_gap_pct
            # Premarket rank score (advisory): gap + RVOL + sector + risk-on. It ONLY
            # influences candidate ordering below — never eligibility or order creation.
            _pmq = (premarket_snapshot or {}).get("quotes", {}).get(symbol) if premarket_snapshot else None
            if _pmq:
                try:
                    import premarket_live_feed as _plf
                    ev.premarket_rank_score = _plf.compute_premarket_rank_score(_pmq, market_state)
                except Exception:
                    ev.premarket_rank_score = None
            urow["premarket_rank_score"] = ev.premarket_rank_score

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
                urow["reject_reason"] = "regime_gate:SPY_not_confirmed_above_200DMA" if regime_suppressed else entry_gate_reason
                universe_rows.append(urow); continue
            # Clear a setup-quality reject_reason if the score now meets the
            # effective (regime-adjusted) threshold.  detect_setup() sets the
            # reject_reason against the static config.min_setup_score (55), but
            # on mixed/cautious days _eff_min_score is lowered to 40.  Without
            # this clear the lower threshold never fires for scores 40-54.
            if (ev.reject_reason
                    and ev.reject_reason.startswith("rejected_by_setup_quality")
                    and ev.setup_score >= _eff_min_score):
                ev.reject_reason = None
            if ev.setup_score < _eff_min_score or ev.reject_reason:
                urow["reject_reason"] = ev.reject_reason or \
                    f"rejected_by_setup_quality:score={ev.setup_score:.0f}<{_eff_min_score:.0f}"
                _alpha_event("setup_rejected", {"symbol": symbol, "score": ev.setup_score, "reason": urow["reject_reason"]})
                universe_rows.append(urow); continue
            if symbol in self._active:
                urow["reject_reason"] = "already_in_position"
                universe_rows.append(urow); continue

            # ── ORB HARD GATE — underlying must break above the opening-range high ─────
            # The signal only identifies a CANDIDATE; execution waits for the breakout on the
            # underlying. Hard gate BEFORE contract selection (no chain work if not broken out).
            _orb_ok, _orb_reason = orb_gate(ev, self.config)
            if not _orb_ok:
                urow["reject_reason"] = _orb_reason
                _alpha_event("orb_gate_block", {"symbol": symbol, "reason": _orb_reason,
                                                "price": ev.price, "orb_high": ev.orb_high})
                universe_rows.append(urow); continue

            # ── LIQUIDITY-SWEEP TREND+DELTA gate (on the UNDERLYING) ─────────────────────
            # Require a bullish trend-crossover + bullish volume-delta (buySignal) before scanning
            # options. sellSignal is handled in the EXIT path (closes open calls). Fail-open if the
            # signal can't be computed (insufficient history) — never block all trades on missing data.
            if self.config.liq_sweep_enabled:
                _ls = liq_sweep_latest(bars, self.config)
                ev.liq_sweep_buy = bool(_ls and _ls["buySignal"])
                ev.liq_sweep_sell = bool(_ls and _ls["sellSignal"])
                ev.liq_sweep_trend = int(_ls["trend"]) if _ls else 0
                # buy-gate DISABLED by default (tested neutral) — only blocks when explicitly enabled
                if self.config.liq_sweep_buy_gate and _ls is not None and not _ls["buySignal"]:
                    urow["reject_reason"] = "rejected_by_liq_sweep_no_buy"
                    _alpha_event("liq_sweep_no_buy", {"symbol": symbol, "trend": ev.liq_sweep_trend})
                    universe_rows.append(urow); continue

            # ── ANALYST-DOWNSIDE ENTRY SKIP — don't buy calls into predicted downside ─────
            # Skip if analysts (≥3) put the AVERAGE target at/below the current price. Fail-open
            # when targets are missing (per spec). Cheap: cached daily per symbol.
            if self.config.analyst_entry_filter and ev.price > 0:
                try:
                    import analyst_target_exit as _ate
                    _tg = _ate.get_analyst_targets(symbol)
                    if _tg and _ate.predicts_downside(ev.price, _tg.get("avg"), _tg.get("low"),
                                                      _tg.get("n"), self.config.analyst_entry_min_upside):
                        urow["reject_reason"] = "rejected_by_analyst_downside"
                        _alpha_event("analyst_downside_skip", {
                            "symbol": symbol, "price": ev.price, "avg_target": _tg.get("avg")})
                        universe_rows.append(urow); continue
                except Exception:
                    pass   # fail-open

            # Authoritative contract selection with the ENGINE's tight band (0.45-0.70).
            engine_eval = evaluate_call_contracts(symbol, ev.price, chain, self.config, rfr, ev.rvol)
            contract = engine_eval["best"]
            # ── OOPS utility optimization (ADVISORY) ────────────────────────────
            # Re-rank the Black-Scholes-qualified contracts by expected utility of the
            # simulated payoff distribution and select by a combined score. Falls back
            # to the Black-Scholes pick when OOPS is unavailable/empty. Never places
            # orders; RiskManager remains the only execution authority.
            oops_selected = False
            if self._oops_cfg.enabled and engine_eval.get("qualified") and contract is not None:
                try:
                    oops_rows = oops_opt.optimize_option_candidates(
                        engine_eval["qualified"], ev.price, vol_annual_default=0.30,
                        hist_daily_returns=None, cfg=self._oops_cfg,
                        iv_crush_risk=float(vol_diag.get("iv_crush_risk", 0.0) or 0.0))
                    for c in oops_rows:
                        liq = oops_opt._liquidity_score(c, self._oops_cfg)
                        c["combined_option_score"] = round(
                            self._oops_cfg.combined_bs_weight * float(c.get("black_scholes_score", 0.0))
                            + self._oops_cfg.combined_oops_weight * float(c.get("oops_score", 0.0))
                            + self._oops_cfg.combined_liq_weight * liq, 3)
                    if oops_rows:
                        best_combined = max(oops_rows, key=lambda c: c.get("combined_option_score", 0.0))
                        contract = best_combined
                        oops_selected = True
                        ces = [float(c.get("certainty_equivalent_return", 0.0) or 0.0) for c in oops_rows]
                        mean_ce = sum(ces) / len(ces) if ces else 0.0
                        oops_all_rows.extend(oops_rows)
                        oops_by_symbol[symbol] = {
                            "oops_utility_edge": round(float(best_combined.get("certainty_equivalent_return", 0.0)) - mean_ce, 4),
                            "oops_best_contract": best_combined.get("contract_symbol"),
                            "oops_contract_rankings": [
                                {"contract": r.get("contract_symbol"), "oops_score": r.get("oops_score"),
                                 "combined_option_score": r.get("combined_option_score")}
                                for r in oops_rows[:5]],
                        }
                except Exception as _oe:
                    _alpha_event("oops_optimizer_error", {"symbol": symbol, "error": str(_oe)[:160]})
            sig = CallSignal(symbol=symbol, setup_type=ev.setup_type, setup_score=ev.setup_score,
                             grade=ev.grade, signal_at_utc=datetime.now(timezone.utc).isoformat(),
                             indicator_source=ev.indicator_source,
                             indicator_age_seconds=ev.indicator_age_seconds,
                             streaming_indicator_available=ev.streaming_indicator_available,
                             fallback_used=ev.fallback_used,
                             premarket_source=ev.premarket_source,
                             premarket_gap_pct=ev.premarket_gap_pct,
                             premarket_volume=ev.premarket_volume,
                             premarket_high=ev.premarket_high,
                             premarket_low=ev.premarket_low,
                             premarket_relative_strength=ev.premarket_relative_strength,
                             premarket_stale=ev.premarket_stale,
                             premarket_rank_score=ev.premarket_rank_score)
            _apply_vol_fields(sig, vol_diag)
            if contract is not None:
                _apply_oops_fields(sig, contract, oops_selected)

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
                    _cts = engine_eval.get("counts") or {}
                    _ace_log.info(
                        "CHAIN_DIAG %s: hard_survivors=%s bs_priced=%s qualified=%s "
                        "chain_len=%s price=%.2f "
                        "[puts=%s dte=%s delta=%s oi=%s vol=%s mid=%s spread=%s]",
                        symbol, _cts.get("hard_gate_survivors", "?"),
                        _cts.get("contracts_priced_by_black_scholes", "?"),
                        _cts.get("qualified", "?"),
                        len(chain), ev.price,
                        _cts.get("hg_rejected_puts", "?"),
                        _cts.get("hg_rejected_dte", "?"),
                        _cts.get("hg_rejected_delta", "?"),
                        _cts.get("hg_rejected_oi", "?"),
                        _cts.get("hg_rejected_vol", "?"),
                        _cts.get("hg_rejected_mid", "?"),
                        _cts.get("hg_rejected_spread", "?"))
                    _alpha_event("contract_rejected", {"symbol": symbol, "reason": sig.reject_reason,
                        "counts": _cts})
                urow["reject_reason"] = sig.reject_reason
                universe_rows.append(urow); signals.append(sig); continue

            sig.contract_symbol = contract["contract_symbol"]; sig.expiry = contract["expiry"]
            sig.strike = contract["strike"]; sig.delta = contract["delta"]; sig.dte = contract["dte"]
            sig.bid = contract["bid"]; sig.ask = contract["ask"]; sig.mid = contract["mid"]
            sig.open_interest = contract["open_interest"]; sig.volume = contract["volume"]
            _apply_bs_fields(sig, contract)

            # ── OptionLab payoff/risk profile (advisory) ────────────────────────
            # Validates the CHOSEN call's payoff geometry. MAY raise/lower the combined
            # score; NEVER bypasses liquidity, RiskManager, hard stops, kill switch, or
            # the daily-loss halt, and never places an order. Fully fail-safe.
            _optionlab_ran = False
            if self.config.optionlab_enabled:
                try:
                    import optionlab_evaluator as _ole
                    _iv = float(sig.implied_volatility or 0.0)
                    _iv = _iv if 0.0 < _iv < 3.0 else None
                    _olr = _ole.get_evaluator().evaluate_long_call(
                        underlying_price=float(ev.price),
                        strike=float(contract.get("strike", 0.0)),
                        premium=float(sig.mid or contract.get("mid", 0.0)),
                        dte=float(contract.get("dte", 30) or 30),
                        iv=_iv, contracts=1,
                        delta=sig.delta, gamma=sig.gamma, theta=sig.theta, vega=sig.vega,
                        expected_move=float(vol_diag.get("expected_move", 0.0) or 0.0) or None,
                        symbol=symbol, market_state=market_state)
                    sig.optionlab_score = float(_olr.get("optionlab_score", 0.0) or 0.0)
                    sig.break_even_price = _olr.get("break_even_price")
                    sig.break_even_distance_pct = _olr.get("break_even_distance_pct")
                    sig.profitable_price_range = _olr.get("profitable_price_range")
                    sig.max_loss = _olr.get("max_loss")
                    sig.max_gain_estimate = _olr.get("max_gain_estimate")
                    sig.payoff_asymmetry = _olr.get("payoff_asymmetry")
                    sig.payoff_quality = _olr.get("payoff_quality", "")
                    sig.optionlab_greeks = _olr.get("optionlab_greeks")
                    sig.optionlab_reject_reason = _olr.get("optionlab_reject_reason", "")
                    _optionlab_ran = True   # a real OptionLab evaluation produced a score
                except Exception as _ole_e:
                    _alpha_event("optionlab_error", {"symbol": symbol, "error": str(_ole_e)[:160]})
                    sig.optionlab_reject_reason = "optionlab_failed"

            # ── Final ensemble combined option score (configurable, renormalised) ──
            # combined = w_bs·BS + w_oops·OOPS + w_optionlab·OptionLab + w_liq·liquidity.
            # ONLY recompute when OptionLab actually executed successfully; when OptionLab
            # is disabled or failed, PRESERVE the OOPS-computed combined_option_score so a
            # default optionlab_score=0 can never deflate every signal.
            if _optionlab_ran:
                try:
                    _liq = float(urow.get("liquidity_score", 0.0) or 0.0)
                    _wb = self.config.combined_bs_weight; _wo = self.config.combined_oops_weight
                    _wol = self.config.combined_optionlab_weight; _wl = self.config.combined_liq_weight
                    _wsum = _wb + _wo + _wol + _wl
                    if _wsum > 0:
                        sig.combined_option_score = round(
                            (_wb * float(sig.black_scholes_score or 0.0)
                             + _wo * float(sig.oops_score or 0.0)
                             + _wol * float(sig.optionlab_score or 0.0)
                             + _wl * _liq) / _wsum, 3)
                except Exception:
                    pass

            # SIZING — fractional Kelly on the live ALPACA ACCOUNT EQUITY (options_quant),
            # capped by the regime-adjusted risk budget and the waterfall. `equity` here is
            # the broker-seeded account equity (run.py: seed_risk_from_broker → risk.equity).
            # Kelly can only REDUCE size below the risk cap, never exceed it (min()), so all
            # existing risk controls are preserved. _eff_risk_pct is halved on mixed/cautious days.
            _risk_cap_dollars = equity * _eff_risk_pct
            if self.config.kelly_sizing_enabled:
                _kelly_dollars = equity * self.config.kelly_full * self.config.kelly_mult
                available = min(_kelly_dollars, _risk_cap_dollars)
            else:
                available = _risk_cap_dollars
            if waterfall_budget > 0:
                available = min(available, waterfall_budget)
            # Size contracts against the EFFECTIVE budget so the premium correctly
            # reflects the regime cap.  On mixed days the base size_position would
            # return 1 contract at full cost; passing available as the risk ceiling
            # instead will return 0 contracts when even 1 contract is unaffordable,
            # surfacing a clear "no viable contract" skip rather than a budget reject.
            _eff_max_risk = available  # regime-adjusted spend limit
            cost_per_contract = max(0.01, sig.mid * 100)
            _eff_contracts = max(1, int(_eff_max_risk // cost_per_contract)) if cost_per_contract > 0 else 0
            if _eff_contracts < 1 or cost_per_contract > _eff_max_risk:
                # Best contract exceeds budget. Try cheaper qualified alternatives before giving up.
                # Sort qualified list by mid price ascending to find the cheapest affordable option.
                _all_qualified = sorted(
                    [c for c in (engine_eval.get("qualified") or []) if float(c.get("mid", 0) or 0) > 0],
                    key=lambda c: float(c.get("mid", 0) or 0)
                )
                _affordable_alt = None
                for _alt in _all_qualified:
                    _alt_cost = max(0.01, float(_alt.get("mid", 0) or 0) * 100)
                    if _alt_cost <= _eff_max_risk:
                        _affordable_alt = _alt
                        break
                if _affordable_alt is not None:
                    # Switch to the cheapest qualified contract that fits within budget.
                    _alt_cost = max(0.01, float(_affordable_alt.get("mid", 0) or 0) * 100)
                    sig.contract_symbol = _affordable_alt.get("contract_symbol", sig.contract_symbol)
                    sig.expiry = _affordable_alt.get("expiry", sig.expiry)
                    sig.strike = _affordable_alt.get("strike", sig.strike)
                    sig.delta = _affordable_alt.get("delta", sig.delta)
                    sig.dte = _affordable_alt.get("dte", sig.dte)
                    sig.bid = _affordable_alt.get("bid", sig.bid)
                    sig.ask = _affordable_alt.get("ask", sig.ask)
                    sig.mid = _affordable_alt.get("mid", sig.mid)
                    sig.open_interest = _affordable_alt.get("open_interest", sig.open_interest)
                    sig.volume = _affordable_alt.get("volume", sig.volume)
                    _apply_bs_fields(sig, _affordable_alt)
                    cost_per_contract = _alt_cost
                    _eff_contracts = max(1, int(_eff_max_risk // cost_per_contract))
                    _ace_log.info(
                        "BUDGET_FALLBACK %s: best contract too expensive (%.0f>%.0f budget), "
                        "using cheaper alt %s mid=%.2f delta=%.2f dte=%s",
                        symbol, max(0.01, contract.get("mid", 0) * 100) if isinstance(contract, dict) else 0,
                        _eff_max_risk, _affordable_alt.get("contract_symbol", "?"),
                        _affordable_alt.get("mid", 0), _affordable_alt.get("delta", 0),
                        _affordable_alt.get("dte", 0))
                else:
                    # Cannot afford even the cheapest qualified contract within regime budget.
                    sig.reject_reason = (f"rejected_by_cash_buffer:premium="
                                         f"{int(cost_per_contract)}>available={int(available)}"
                                         f"(regime:{int(_eff_risk_pct*100)}%risk)")
                    urow["reject_reason"] = sig.reject_reason
                    universe_rows.append(urow); signals.append(sig); continue
            contracts, premium, max_risk = _eff_contracts, round(_eff_contracts * cost_per_contract, 2), round(_eff_contracts * cost_per_contract, 2)
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

        # Rank by setup score, with a BOUNDED premarket-rank tiebreaker/nudge (advisory).
        # Eligibility was already decided by the gates above (status=='ready'); this only
        # reorders candidates and can never make a non-setup tradeable.
        signals.sort(
            key=lambda s: s.setup_score + min(10.0, 0.1 * float(s.premarket_rank_score or 0.0)),
            reverse=True)
        self._signals_today.extend([asdict(s) for s in signals if s.status == "ready"])
        # Merge OOPS summary into the volatility diagnostic rows.
        for vr in vol_rows:
            osum = oops_by_symbol.get(vr.get("symbol"))
            if osum:
                vr.update(osum)
        self._save_signals()
        self._save_alpha_universe(universe_rows, market_context)
        self._save_volatility_diagnostic(vol_rows, market_context)
        if self._oops_cfg.enabled:
            try:
                oops_opt.write_oops_diagnostic(oops_all_rows, market_context)
            except Exception as _oe:
                _alpha_event("oops_report_error", {"error": str(_oe)[:160]})
        if self.config.optionlab_enabled:
            try:
                import optionlab_evaluator as _ole
                _ole.get_evaluator().write_report()
            except Exception as _oe:
                _alpha_event("optionlab_report_error", {"error": str(_oe)[:160]})
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
                               market_context: dict | None = None,
                               indicator_snapshot: dict | None = None,
                               market_state: dict | None = None) -> list[dict]:
        """
        Public EXIT pass. Manages all open calls (hard stop / +partial / 1.7R trail /
        3R target) and an optional Black-Scholes theta-decay exit. Returns exit orders.
        Must be called BEFORE evaluate_universe() each cycle (exit-before-buy).

        indicator_snapshot is accepted for shared real-time indicator state (volatility
        diagnostics). Exits are governed by hard 3R/stop/trail rules and are never
        blocked or created by streaming indicators; the args are forwarded for diagnostics.
        """
        return self._check_active_positions(options_positions, waterfall_callback,
                                            market_context=market_context,
                                            indicator_snapshot=indicator_snapshot,
                                            market_state=market_state)

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
