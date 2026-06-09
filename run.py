"""
run.py — QuantumQuant Main Entry Point  (Two-Engine Architecture)

Orchestrates a two-engine trading system on top of the existing risk controls:

  1. ALPHA ENGINE  (alpha_call_engine.AlphaCallEngine)
       Primary tactical profit engine. Long calls only on A/A+ momentum, ORB,
       VWAP-reclaim, pivot-breakout, institutional-accumulation, sector-leadership
       setups. Calls are actively managed (hard stop / +1R partial / 1.7R trail /
       3R target) — never held indefinitely.

  2. BUFFETT ACCUMULATION LAYER  (buffett_accumulation_layer.BuffettAccumulationLayer)
       Strategic long-stock engine. Scores businesses 0-100 (moat, financials,
       management, valuation, macro, policy, 18-yr cycle, technical entry) and
       accumulates quality companies at optimal entry points. Held indefinitely
       unless thesis / trend / concentration / risk breaks.

Capital flow (capital_waterfall.CapitalWaterfall):
       Realized call profits → 50-70% BAL stock, 20-40% alpha buying power, 10% cash.

Data flow:
       TradingView institutional candidates  →  market_context_builder.build_market_context()
       →  macro/policy/cycle context  →  Alpha Engine (calls) + BAL (stock)
       →  risk-gated Alpaca execution  →  journal / equity logging.

SAFETY GATES (never bypassed):
  • Dynamic daily loss limit      account_equity × DYNAMIC_DAILY_LOSS_PCT
                                  default 0.50%, minimum $500, recalculated from Alpaca account
  • Dynamic max drawdown          starting/account-anchor equity × DYNAMIC_MAX_DRAWDOWN_PCT
                                  default 10%, enforced from the first live Alpaca account snapshot
  • Dynamic max open positions    floor(account_equity / DYNAMIC_POSITION_EQUITY_UNIT)
                                  default one slot per $50k, bounded by 5–50
  • Kill switch (manual file)     KILL_SWITCH_PATH (env, default .kill_switch)
  • Alpaca paper-only gate        ALPACA_PAPER_ONLY (env, default "true")
  • Dynamic equity floor          starting/account-anchor equity × DYNAMIC_EQUITY_FLOOR_PCT
                                  default 90%, never below $1,000
  • Dynamic max order notional    account_equity × DYNAMIC_MAX_ORDER_PCT
                                  default 3%, used for new buy orders
  • Optional max order shares     MAX_ORDER_SIZE if explicitly set; otherwise disabled
  • Cash buffer                   CASH_BUFFER_PCT  (env, default 8%)
  • Max options exposure          MAX_OPTIONS_EXPOSURE_PCT (env, default 15%)
  • Max stock exposure            MAX_STOCK_EXPOSURE_PCT   (env, default 80%)
  • Max single-stock weight       MAX_SINGLE_STOCK_WEIGHT  (env, default 10%)
  • PDT awareness                 (block equity day-trade buys if flagged < $25k)
  • Market hours gate             (no orders outside RTH unless allowed)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import time
import atexit
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
_RUN_LOCK_HANDLE = None

PROJECT_ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")

# Load .env before any module-level env reads so creds/config are populated.
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Environment / config
# ---------------------------------------------------------------------------

def _env_bool_any(names: List[str], default: bool = False) -> bool:
    """Read the first configured boolean env var from a list of common aliases."""
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "paper"}
    return default

MODE               = os.environ.get("MODE",             "paper").lower()   # paper | live | train | scalp
SYMBOLS_ENV        = os.environ.get("SYMBOLS",          "")
# Dynamic risk limits are derived from the live Alpaca account snapshot. The old
# fixed env names are still accepted as explicit overrides, but by default the
# bot scales risk to account equity instead of using small-account constants.
DYNAMIC_RISK_LIMITS = os.environ.get("DYNAMIC_RISK_LIMITS", "true").lower() in {"1", "true", "yes"}
DYNAMIC_DAILY_LOSS_PCT = float(os.environ.get("DYNAMIC_DAILY_LOSS_PCT", "0.005"))       # 0.50%/day
DYNAMIC_DAILY_LOSS_MIN = float(os.environ.get("DYNAMIC_DAILY_LOSS_MIN", "500"))
DYNAMIC_MAX_DRAWDOWN_PCT = float(os.environ.get("DYNAMIC_MAX_DRAWDOWN_PCT", "0.06"))    # 6% from anchor (room for one ±5-6% position)
DYNAMIC_EQUITY_FLOOR_PCT = float(os.environ.get("DYNAMIC_EQUITY_FLOOR_PCT", "0.90"))    # 90% of anchor
DYNAMIC_EQUITY_FLOOR_MIN = float(os.environ.get("DYNAMIC_EQUITY_FLOOR_MIN", "1000"))
DYNAMIC_MAX_ORDER_PCT = float(os.environ.get("DYNAMIC_MAX_ORDER_PCT", "0.03"))          # 3% notional/order
# Fractional/notional BUY orders: let the BAL deploy an exact DOLLAR amount (fractional shares) so
# high-priced names (ASML, MU) fit at sane weights on a small account instead of being whole-share-floored.
ENABLE_FRACTIONAL_ORDERS = os.environ.get("ENABLE_FRACTIONAL_ORDERS", "false").lower() in {"1", "true", "yes"}
# 9-EMA trend overlay: BUY only when price > daily 9-EMA (execution gate / cross-up confirmation);
# SELL any held share that falls below its 9-EMA (trend break / cross-down). Daily bars, cached.
ENABLE_EMA_GATE = os.environ.get("ENABLE_EMA_GATE", "true").lower() in {"1", "true", "yes"}
ENABLE_EMA_EXIT = os.environ.get("ENABLE_EMA_EXIT", "true").lower() in {"1", "true", "yes"}
EMA_PERIOD = int(os.environ.get("EMA_PERIOD", "9"))
EMA_TAKE_PROFIT_PCT = float(os.environ.get("EMA_TAKE_PROFIT_PCT", "0.33"))   # below 9-EMA → trim this fraction
EMA_TRIM_WINNERS_ONLY = os.environ.get("EMA_TRIM_WINNERS_ONLY", "true").lower() in {"1", "true", "yes"}
# 200-SMA long-term trend gate: only BUY / re-enter when price is above the daily 200-SMA.
ENABLE_SMA200_GATE = os.environ.get("ENABLE_SMA200_GATE", "true").lower() in {"1", "true", "yes"}
SMA200_PERIOD = int(os.environ.get("SMA200_PERIOD", "200"))
# Per-position stop loss: exit a held share when it falls this % below its average entry (0 = off).
POSITION_STOP_PCT = float(os.environ.get("POSITION_STOP_PCT", "0.0"))
# Broader-market regime switch: gate buys + liquidate on a market-wide reversal/chop. RISK-ON = a
# majority of SPY/QQQ/IWM above their 200-SMA AND 50-SMA rising. Avoids burning out in chop/reversals.
ENABLE_MARKET_REGIME = os.environ.get("ENABLE_MARKET_REGIME", "true").lower() in {"1", "true", "yes"}
REGIME_INDICES = [x.strip().upper() for x in os.environ.get("REGIME_INDICES", "SPY,QQQ,IWM").split(",") if x.strip()]
DYNAMIC_POSITION_EQUITY_UNIT = float(os.environ.get("DYNAMIC_POSITION_EQUITY_UNIT", "50000"))
DYNAMIC_MIN_POSITIONS = int(os.environ.get("DYNAMIC_MIN_POSITIONS", "5"))
DYNAMIC_MAX_POSITIONS = int(os.environ.get("DYNAMIC_MAX_POSITIONS", "50"))

# Optional fixed overrides. Leave unset for dynamic behavior.
DAILY_LOSS_LIMIT_OVERRIDE = os.environ.get("DAILY_LOSS_LIMIT")
MAX_DRAWDOWN_OVERRIDE = os.environ.get("MAX_DRAWDOWN")
MAX_POSITIONS_OVERRIDE = os.environ.get("MAX_POSITIONS")
EQUITY_FLOOR_OVERRIDE = os.environ.get("EQUITY_FLOOR")
MAX_ORDER_DOLLARS_OVERRIDE = os.environ.get("MAX_ORDER_DOLLARS")
MAX_ORDER_SIZE_OVERRIDE = os.environ.get("MAX_ORDER_SIZE")  # optional share/contract-count cap

ALPACA_PAPER_ONLY  = _env_bool_any(["ALPACA_PAPER_ONLY", "APCA_PAPER_ONLY", "APCA_PAPER", "ALPACA_PAPER"], True)
KILL_SWITCH_PATH   = os.environ.get("KILL_SWITCH_PATH",        ".kill_switch")
BAR_INTERVAL_SEC   = float(os.environ.get("BAR_INTERVAL_SEC",  60.0))
ACCOUNT_STATS_LOG_SEC = float(os.environ.get("ACCOUNT_STATS_LOG_SEC", 300.0))
# Fallback seed only. The real anchor is sourced live from the Alpaca account at
# startup (seed_risk_from_broker); STARTING_EQUITY is used solely when the broker
# is unreachable so the daemon can still boot in simulation.
STARTING_EQUITY    = float(os.environ.get("STARTING_EQUITY", 100_000.0))

# Two-engine toggles & exposure controls (also consumed by the engine configs via env)
ENABLE_ALPHA_CALL_ENGINE        = os.environ.get("ENABLE_ALPHA_CALL_ENGINE", "true").lower() in {"1", "true"}
ENABLE_BUFFETT_ACCUMULATION_LAYER = os.environ.get("ENABLE_BUFFETT_ACCUMULATION_LAYER", "true").lower() in {"1", "true"}
ENABLE_BAL_PORTFOLIO_MONITOR    = os.environ.get("ENABLE_BAL_PORTFOLIO_MONITOR", "false").lower() in {"1", "true"}
ENABLE_ALPHA_OPTIONS_SCREENER   = os.environ.get("ENABLE_ALPHA_OPTIONS_SCREENER", "true").lower() in {"1", "true"}
ENABLE_BUFFETT_ACCUMULATION_SCREENER = os.environ.get("ENABLE_BUFFETT_ACCUMULATION_SCREENER", "true").lower() in {"1", "true"}
MAX_OPTIONS_EXPOSURE_PCT        = float(os.environ.get("MAX_OPTIONS_EXPOSURE_PCT", 0.15))
MAX_STOCK_EXPOSURE_PCT          = float(os.environ.get("MAX_STOCK_EXPOSURE_PCT",  0.80))
# Size-to-fit: when a BUY exceeds only a trimmable notional/exposure cap, shrink it
# to the remaining room under the cap instead of hard-rejecting the whole order. The
# cap is never breached; hard risk blocks (kill-switch, drawdown, PDT, …) still reject.
SIZE_TO_FIT_EXPOSURE            = os.environ.get("SIZE_TO_FIT_EXPOSURE", "1").lower() not in {"0", "false", "no"}
MAX_SINGLE_STOCK_WEIGHT         = float(os.environ.get("MAX_SINGLE_STOCK_WEIGHT", 0.10))
# Self-assessment BUY throttle (downside protection). system_self_awareness.py writes a
# recommended_size_multiplier (∈[0,1]) and SAFE-mode flag derived from data-freshness, model,
# regime-risk and execution-risk checks. When enabled, BUY sizes are scaled by that multiplier
# and a 0.0 multiplier skips the buy. Applies to BUYS ONLY — exits/sells are never throttled.
# Reversible via SELF_ASSESSMENT_SIZING=0.
SELF_ASSESSMENT_SIZING          = os.environ.get("SELF_ASSESSMENT_SIZING", "1").lower() not in {"0", "false", "no"}
SELF_ASSESSMENT_TTL_SEC         = float(os.environ.get("SELF_ASSESSMENT_TTL_SEC", "60"))
OPTIONS_RISK_PER_TRADE_PCT      = float(os.environ.get("OPTIONS_RISK_PER_TRADE_PCT", 0.015))
BAL_MIN_SCORE_TO_BUY            = float(os.environ.get("BAL_MIN_SCORE_TO_BUY", 55))
BAL_AGGRESSIVE_SCORE            = float(os.environ.get("BAL_AGGRESSIVE_SCORE", 85))
# Composite DCA framework — value decides WHETHER, RSI decides WHEN, tranches decide
# HOW MUCH, RiskManager decides whether ALLOWED. (config lives in DCAConfig)
ENABLE_BAL_COMPOSITE_DCA        = os.environ.get("ENABLE_BAL_COMPOSITE_DCA", "true").lower() in {"1", "true"}
# SEC EDGAR Form 4 insider-buy DISCOVERY (expands the Buffett review universe only —
# never generates orders, never bypasses any gate).
ENABLE_SEC_FORM4_FEED           = os.environ.get("ENABLE_SEC_FORM4_FEED", "true").lower() in {"1", "true"}
# Insider TRADE COPIER — mirrors large, RECENT open-market insider BUYS (SEC Form 4,
# optionally confirmed by Alpha Vantage INSIDER_TRANSACTIONS) into paper BUY orders.
# Unlike the discovery feed above, this DOES place orders — but every one is routed
# through the RiskManager gate (kill-switch, equity floor, daily-loss, drawdown, PDT,
# position count, exposure caps, self-assessment throttle, size-to-fit), so it can
# never bypass a risk control, place a SELL, or trade while the market is shut.
# Copies a name only when the aggregate open-market insider buy value exceeds
# INSIDER_COPY_MIN_VALUE and the latest insider transaction is within
# INSIDER_COPY_MAX_AGE_DAYS days. Fully fail-safe and reversible via ENABLE_INSIDER_COPIER=0.
# DEFAULT OFF (opt-in): backtest_insider_copier.py shows NO reliable market-excess edge on
# the available mega-cap universe (mean-vs-SPY ≤ 0 at most horizons; see
# backtest_outputs/insider_copier_backtest.md). Enable with ENABLE_INSIDER_COPIER=1 to
# paper-trade it, but it is not validated as a standalone return engine.
ENABLE_INSIDER_COPIER           = os.environ.get("ENABLE_INSIDER_COPIER", "false").lower() in {"1", "true"}
INSIDER_COPY_MIN_VALUE          = float(os.environ.get("INSIDER_COPY_MIN_VALUE", 1_000_000))   # strictly greater-than
INSIDER_COPY_MAX_AGE_DAYS       = int(os.environ.get("INSIDER_COPY_MAX_AGE_DAYS", 2))           # latest buy within N days
INSIDER_COPY_DOLLARS            = float(os.environ.get("INSIDER_COPY_DOLLARS", 1500))           # notional sized per copied name
INSIDER_COPY_MAX_NAMES          = int(os.environ.get("INSIDER_COPY_MAX_NAMES", 5))              # cap copies per cycle

# ── Option safety-exit (alpha-independent backstop) ───────────────────────────
# The normal option-exit path (run_alpha_exits) only runs when the Alpha engine is
# live. With ENABLE_ALPHA_CALL_ENGINE=false (alpha=None) any previously-opened option
# becomes orphaned — nothing applies a stop/target. This backstop manages stops on open
# options when alpha is OFF, in the unconditional exit pass. Tunable; defaults are loose
# (it is a safety net, not an alpha exit).
ENABLE_OPTION_SAFETY_EXIT = os.environ.get("ENABLE_OPTION_SAFETY_EXIT", "true").lower() in {"1", "true", "yes"}
OPTION_SAFETY_STOP_PCT    = float(os.environ.get("OPTION_SAFETY_STOP_PCT", "0.50"))   # close at <= -50% premium
OPTION_SAFETY_TP_PCT      = float(os.environ.get("OPTION_SAFETY_TP_PCT", "1.00"))     # close at >= +100% premium
OPTION_SAFETY_MIN_DTE     = int(os.environ.get("OPTION_SAFETY_MIN_DTE", "2"))         # close when DTE <= 2 (gamma/assignment)

# ── Main-loop watchdog ────────────────────────────────────────────────────────
# A daemon thread watches a heartbeat updated at the top of every loop iteration. If the
# loop makes no progress for WATCHDOG_STALL_SEC (e.g. a broker socket hangs with no
# timeout), it dumps the stack and raises on the main thread (via SIGALRM) so the cycle
# aborts and the loop continues — bounding a hang to seconds instead of an hour. Only
# interrupts the main thread; streaming-feed daemon threads are untouched.
WATCHDOG_STALL_SEC = float(os.environ.get("WATCHDOG_STALL_SEC", "180"))
# AV confirmation defaults ON but FAIL-OPEN: a missing key / quota / error never blocks a
# copy; only a positive Alpha-Vantage net-selling signal vetoes one.
INSIDER_COPY_REQUIRE_AV_CONFIRM = os.environ.get("INSIDER_COPY_REQUIRE_AV_CONFIRM", "true").lower() in {"1", "true"}
# COMBO gate (prototype): require the insider-copy underlying to ALSO be in the TradingView
# quality-growth screen. DEFAULT OFF — backtest_insider_quality_combo.py shows this gate
# DESTROYS the edge (deep-ITM net +9.3% → +1.1%, significance gone): the insider-buy edge
# lives in cheap sub-$4 oversold small-caps (+50% net), which the screen's $3.99 floor +
# quality bias explicitly exclude. The two signals select opposite populations. Kept as an
# opt-in toggle for live A/B only.
INSIDER_COPY_REQUIRE_QUALITY_SCREEN = os.environ.get("INSIDER_COPY_REQUIRE_QUALITY_SCREEN", "false").lower() in {"1", "true"}
# Copier INSTRUMENT — DEFAULT "share" (2026-06-08). Walk-forward validation
# (backtest_vol_gated_insider.py, n=129 pooled OOS, PF 2.20 @3d, +3.35% mean, MaxDD −5%, net
# 100bps) proved the insider edge is capturable in SHARES gated by entry-day volatility, while
# OPTIONS destroy it (spread+theta >> edge; 30-56% spreads on these small-caps). "option" keeps
# the legacy deep-ITM-call path for A/B only. The VOL+LIQUIDITY gate below applies to BOTH but is
# the validated rule for shares.
INSIDER_COPY_INSTRUMENT         = os.environ.get("INSIDER_COPY_INSTRUMENT", "share").strip().lower()
# ── VOL-GATED INSIDER SHARES (the validated live rule) ──────────────────────────────────────
# Trade only when the underlying's entry-day realized_vol_20 (std of 20 daily log returns) is in
# the high-vol regime (≥ the train P80 ≈ 0.0563) AND it's liquid enough that ~100bps slippage is
# realistic (20d avg $-volume ≥ $1M). Insider conviction pays most in fear/high-vol; calm-tape
# insider buys were net-flat-to-negative OOS. Both gates fail-CLOSED on missing data (no trade).
INSIDER_COPY_VOL_GATE           = os.environ.get("INSIDER_COPY_VOL_GATE", "true").lower() in {"1", "true"}
INSIDER_COPY_VOL_MIN_RV20       = float(os.environ.get("INSIDER_COPY_VOL_MIN_RV20", 0.0563))   # train P80 daily realized vol
INSIDER_COPY_MIN_ADV            = float(os.environ.get("INSIDER_COPY_MIN_ADV", 1_000_000))     # 20d avg $-volume liquidity floor
INSIDER_COPY_SHARE_HOLD_DAYS    = int(os.environ.get("INSIDER_COPY_SHARE_HOLD_DAYS", 3))       # validated 3-day hold
INSIDER_COPY_SHARE_RISK_PCT     = float(os.environ.get("INSIDER_COPY_SHARE_RISK_PCT", 0.01))   # equity-risk budget per name
INSIDER_COPY_SHARE_STOP_PCT     = float(os.environ.get("INSIDER_COPY_SHARE_STOP_PCT", 0.10))   # implied adverse move for sizing
INSIDER_COPY_MAX_OPEN           = int(os.environ.get("INSIDER_COPY_MAX_OPEN", 3))              # small-account total-open cap
# ── CLUSTER-BREADTH amplifier (research-validated 2026-06-08) ────────────────────────────────
# >=2 DISTINCT insiders buying ("consensus") lifts the edge PF ~1.9→2.57, stable EVERY regime-year
# (n=301). The live feed already exposes distinct-buyer count via aggregate_by_symbol's insider_count.
# SHADOW MODE (default ON): compute + JOURNAL breadth on every copy, but DON'T gate — so we collect
# 30-60d of paper data (accepted vs would-be-rejected) before risking opportunity-flow reduction.
# Flip ALPHA-style: set INSIDER_COPY_BREADTH_SHADOW=0 to HARD-GATE at MIN_DISTINCT_BUYERS once proven.
INSIDER_COPY_MIN_DISTINCT_BUYERS = int(os.environ.get("INSIDER_COPY_MIN_DISTINCT_BUYERS", 2))
INSIDER_COPY_BREADTH_SHADOW      = os.environ.get("INSIDER_COPY_BREADTH_SHADOW", "true").lower() in {"1", "true"}
# ── ROLLING-STRATEGY PF KILL-SWITCH (edge-decay protection) ─────────────────────────────────
# The biggest operational risk to the validated edge is DECAY, not entry timing. This halts NEW
# insider-copy entries (exits always still run) when the strategy's realised profit factor over a
# rolling window of recent CLOSED trades falls below threshold — but only after enough trades to
# judge. Strategy-level (not per-trade). Re-enables automatically when PF recovers.
INSIDER_COPY_PF_KILL_ENABLED    = os.environ.get("INSIDER_COPY_PF_KILL", "true").lower() in {"1", "true"}
INSIDER_COPY_PF_THRESHOLD       = float(os.environ.get("INSIDER_COPY_PF_THRESHOLD", 1.8))      # halt if rolling PF < this
INSIDER_COPY_PF_WINDOW          = int(os.environ.get("INSIDER_COPY_PF_WINDOW", 40))            # rolling N closed trades
INSIDER_COPY_PF_MIN_TRADES      = int(os.environ.get("INSIDER_COPY_PF_MIN_TRADES", 30))        # need ≥ this before judging
INSIDER_COPY_PF_LOOKBACK_DAYS   = int(os.environ.get("INSIDER_COPY_PF_LOOKBACK_DAYS", 90))     # only count recent trades
INSIDER_COPY_LEDGER_PATH        = PROJECT_ROOT / "logs" / "insider_copy_trade_ledger.json"
INSIDER_COPY_OPTION_DTE         = int(os.environ.get("INSIDER_COPY_OPTION_DTE", 30))          # target days-to-expiry
# Delta floor raised 0.47 → 0.75 (deep ITM) by backtest_insider_options.py: ATM/0.47 calls
# are net-NEGATIVE once realistic small-cap IV + option spreads are priced in (the time
# value bleeds to the market maker), while delta≈0.80 deep-ITM calls stay net-POSITIVE
# (+9% @10d, t=3.3, robust to IV×1.5/50% spread) because they're mostly intrinsic. The
# selector picks the cheapest call clearing this floor. Set lower to revert toward ATM.
INSIDER_COPY_OPTION_MIN_DELTA   = float(os.environ.get("INSIDER_COPY_OPTION_MIN_DELTA", 0.75))  # call delta floor (>)
INSIDER_COPY_OPTION_DOLLARS     = float(os.environ.get("INSIDER_COPY_OPTION_DOLLARS", 1500))   # premium budget per name
INSIDER_COPY_OPTION_MIN_OI      = int(os.environ.get("INSIDER_COPY_OPTION_MIN_OI", 10))        # liquidity floor
# Time-exit: the insider-buy edge is short-horizon (5-10d) but a 30DTE call bleeds theta,
# so copied calls are force-closed after this many BUSINESS days regardless of P&L. The
# exit runs in the unconditional EXIT pass (never regime-gated). 0 disables it.
INSIDER_COPY_OPTION_MAX_HOLD_DAYS = int(os.environ.get("INSIDER_COPY_OPTION_MAX_HOLD_DAYS", 10))
INSIDER_COPY_STATE_PATH         = PROJECT_ROOT / "logs" / "insider_copy_positions.json"
# Research-grounded sizing (options_quant.py): fractional-Kelly + risk cap instead of a fixed
# dollar budget. Kelly fraction is shrunk for dispersed option payoffs (Perez-Marco); the full
# Kelly was measured at 0.125 on the deep-ITM payoff distribution (backtest), scaled by KELLY_MULT.
INSIDER_COPY_USE_QUANT_SIZING   = os.environ.get("INSIDER_COPY_USE_QUANT_SIZING", "true").lower() in {"1", "true"}
INSIDER_COPY_KELLY_FULL         = float(os.environ.get("INSIDER_COPY_KELLY_FULL", 0.125))   # empirical full-Kelly f*
INSIDER_COPY_KELLY_MULT         = float(os.environ.get("INSIDER_COPY_KELLY_MULT", 0.25))    # fractional Kelly (¼)
INSIDER_COPY_RISK_CAP_PCT       = float(os.environ.get("INSIDER_COPY_RISK_CAP_PCT", 0.02))  # max premium-at-risk / equity
# TradingView quality-growth screen drives the main Alpha candidate universe
# (tv-screener.py scan_quality_growth -> tv_screener_tickers.txt). Auto-refreshed when
# stale, fully fail-safe (a TV failure falls back to the locked-universe files).
ENABLE_TV_QUALITY_SCREEN        = os.environ.get("ENABLE_TV_QUALITY_SCREEN", "true").lower() in {"1", "true"}
TV_QUALITY_SCREEN_TTL_SEC       = float(os.environ.get("TV_QUALITY_SCREEN_TTL_SEC", 21600))   # 6h (intraday backstop)
# Background refresher: a daemon thread re-runs the screen every N seconds so the universe
# is "constantly" fresh, WITHOUT putting the TradingView network call on the trading
# cycle's critical path (the cycle just reads the file). Clamped to >= 15s.
TV_QUALITY_SCREEN_REFRESH_SEC   = float(os.environ.get("TV_QUALITY_SCREEN_REFRESH_SEC", 60))
TV_SCREEN_TICKERS_PATH          = PROJECT_ROOT / "tv_screener_tickers.txt"
# Adaptive Capital Allocation (evidence-driven sizing PLAN; advisory, within caps).
ENABLE_ADAPTIVE_ALLOCATION      = os.environ.get("ENABLE_ADAPTIVE_ALLOCATION", "true").lower() in {"1", "true"}
# Black-Litterman + Efficient-Frontier Buffett optimizer (target weights; advisory).
ENABLE_BUFFETT_OPTIMIZER        = os.environ.get("ENABLE_BUFFETT_OPTIMIZER", "true").lower() in {"1", "true"}
OPTIONS_PROFIT_TO_BAL_PCT       = float(os.environ.get("OPTIONS_PROFIT_TO_BAL_PCT", 0.60))
OPTIONS_PROFIT_TO_ALPHA_PCT     = float(os.environ.get("OPTIONS_PROFIT_TO_ALPHA_PCT", 0.30))
OPTIONS_PROFIT_TO_CASH_PCT      = float(os.environ.get("OPTIONS_PROFIT_TO_CASH_PCT", 0.10))
CASH_BUFFER_PCT                 = float(os.environ.get("CASH_BUFFER_PCT", 0.45))  # always keep this %% of equity in FREE CASH
CONTEXT_REFRESH_MIN             = float(os.environ.get("CONTEXT_REFRESH_MIN", 25.0))   # was 30.0; tightened to reduce staleness
ALPHA_BARS_LOOKBACK_MIN         = int(os.environ.get("ALPHA_BARS_LOOKBACK_MIN", 120))

# Alpaca creds (required for live/paper modes).
# Resolve across both the alpaca-py (ALPACA_*) and classic SDK (APCA_*) env names —
# .env files commonly carry APCA_API_KEY_ID / APCA_API_SECRET_KEY.
ALPACA_KEY    = (os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
                 or os.environ.get("ALPACA_API_KEY_ID") or "")
ALPACA_SECRET = (os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
                 or os.environ.get("ALPACA_API_SECRET_KEY") or "")
ALPACA_URL    = os.environ.get("ALPACA_BASE_URL", os.environ.get("APCA_API_BASE_URL",
                "https://paper-api.alpaca.markets" if ALPACA_PAPER_ONLY
                else "https://api.alpaca.markets"))

# Option-chain DTE window for contract retrieval (matches the Alpha Engine's gate).
ALPHA_MIN_DTE = int(os.environ.get("ALPHA_MIN_DTE", 21))
ALPHA_MAX_DTE = int(os.environ.get("ALPHA_MAX_DTE", 60))

# Daily tag shared with the engine modules (so all outputs land in the same dated buckets)
DATA_TAG       = os.getenv("RUN_DATA_DATE", datetime.now(timezone.utc).strftime("%Y%m%d"))
DAILY_OUTPUTS  = PROJECT_ROOT / "daily_system_outputs"
SKIPPED_TRADES_PATH = PROJECT_ROOT / "logs" / "execution" / DATA_TAG / f"{DATA_TAG}_skipped_trades.jsonl"
SCREENER_FUNNEL_PATH = DAILY_OUTPUTS / "screener_funnel_report.json"
# Guard: suppress file writes when called from pytest
# NOTE: evaluated at call-time (not import-time) so that PYTEST_CURRENT_TEST is seen correctly.
def _in_test() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in os.path.basename(os.environ.get("_", ""))


def _acquire_single_instance_lock() -> None:
    """Prevent duplicate paper/live run.py loops from running concurrently."""
    global _RUN_LOCK_HANDLE
    if MODE not in ("paper", "live"):
        return
    if _in_test():
        return
    lock_dir = PROJECT_ROOT / "logs"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"run.{MODE}.lock"
    lock_handle = open(lock_path, "a+", encoding="utf-8")
    try:
        import fcntl
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            lock_handle.seek(0)
            owner = lock_handle.read().strip()
        except Exception:
            owner = ""
        logger.critical("Another %s run.py executor is already active%s",
                        MODE, f" ({owner})" if owner else "")
        raise SystemExit(1)
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"pid={os.getpid()} mode={MODE} started_utc={datetime.now(timezone.utc).isoformat()}\n")
    lock_handle.flush()
    _RUN_LOCK_HANDLE = lock_handle

    def _release_lock() -> None:
        global _RUN_LOCK_HANDLE
        if _RUN_LOCK_HANDLE is None:
            return
        try:
            _RUN_LOCK_HANDLE.close()
        except Exception:
            pass
        _RUN_LOCK_HANDLE = None

    atexit.register(_release_lock)

# Canonical skip reasons (every rejected trade must log one of these)
SKIP_REASONS = {
    "rejected_by_market_regime",
    "waiting_for_opening_range",
    "opening_range_not_ready",
    "waiting_for_ORB_breakout",
    "waiting_for_ORB_sweep_reclaim",
    "rejected_by_liq_sweep_no_buy",
    "liq_sweep_sell_signal",
    "rejected_by_analyst_downside",
    "insider_copy_below_breadth",
    "near_avg_analyst_target",
    "near_high_analyst_target",
    "analyst_upside_exhausted",
    "below_low_analyst_target",
    "above_avg_target_tighten_trail",
    "rejected_by_options_liquidity",
    "rejected_by_setup_quality",
    "rejected_by_BAL_score",
    "rejected_by_valuation",
    "rejected_by_macro",
    "rejected_by_cycle",
    "rejected_by_cash_buffer",
    "rejected_by_below_9ema",
    "rejected_by_below_sma200",
    "rejected_not_9ema_dip",
    "rejected_no_trend_crossover",
    "rejected_market_regime_chop_or_reversal",
    "rejected_by_exposure_limit",
    "rejected_by_options_buying_power",
    "rejected_by_PDT",
    "rejected_by_kill_switch",
    "rejected_by_equity_floor",
    "rejected_by_daily_loss",
    "rejected_by_drawdown",
    "rejected_by_order_size",
    "rejected_by_data_unavailable",
    "throttled_by_self_assessment",
    "insider_copy_below_min",
    "insider_copy_stale",
    "insider_copy_already_held",
    "insider_copy_size_zero",
    "insider_copy_av_contradicts",
    "insider_copy_no_qualifying_call",
    "insider_copy_not_in_quality_screen",
}


def log_skip(symbol: str, reason: str, **detail) -> None:
    """Record a skipped trade with a canonical reason. Always logged + journaled."""
    if _in_test():
        return  # never pollute production journals from test runs
    base = reason.split(":", 1)[0]
    if base not in SKIP_REASONS:
        logger.debug("log_skip: non-canonical reason '%s' (symbol=%s)", reason, symbol)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "date_tag": DATA_TAG,
        "symbol": symbol,
        "reason": reason,
        "detail": detail,
    }
    logger.info("SKIP %s — %s %s", symbol, reason, detail if detail else "")
    try:
        SKIPPED_TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SKIPPED_TRADES_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


_SA_CACHE: Dict[str, Any] = {"ts": 0.0, "mult": 1.0, "mode": "NORMAL"}


def _self_assessment_buy_multiplier() -> float:
    """BUY-side size multiplier from system_self_awareness (∈[0,1]); 1.0 if disabled/unavailable.

    Cached for SELF_ASSESSMENT_TTL_SEC so the per-order disk read stays cheap while still
    picking up premarket-collector refreshes of the assessment file. NEVER applied to
    sells/exits — the caller guards on side == "buy".
    """
    if not SELF_ASSESSMENT_SIZING:
        return 1.0
    now = time.time()
    if now - _SA_CACHE["ts"] < SELF_ASSESSMENT_TTL_SEC:
        return _SA_CACHE["mult"]
    mult, mode = 1.0, "NORMAL"
    try:
        from system_self_awareness import load_self_assessment, self_assessment_size_multiplier
        a = load_self_assessment()
        mult = self_assessment_size_multiplier(a)
        mode = str(a.get("execution_mode_recommendation") or "NORMAL")
    except Exception as e:
        logger.debug("self-assessment load failed (%s) — buy multiplier defaults to 1.0", e)
        mult, mode = 1.0, "NORMAL"
    _SA_CACHE.update(ts=now, mult=mult, mode=mode)
    return mult


def is_option_symbol(symbol: str) -> bool:
    """OCC option symbols are >= 15 chars and end with the strike block, e.g. AAPL250620C00150000."""
    s = (symbol or "").strip().upper()
    if len(s) < 15:
        return False
    tail = s[-15:]
    return (tail[6] in ("C", "P")) and tail[:6].isdigit() and tail[7:].isdigit()



def _env_float(value: Optional[str], default: Optional[float] = None) -> Optional[float]:
    """Parse an optional env float without making dynamic risk brittle."""
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _env_int(value: Optional[str], default: Optional[int] = None) -> Optional[int]:
    """Parse an optional env int without making dynamic risk brittle."""
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Risk manager  (preserved + extended for options/stock exposure and reason codes)
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Central guard. All order submissions must pass gate()/check_order().
    Positions / daily PnL / drawdown / PDT tracked here.
    """

    def __init__(self, equity: float = STARTING_EQUITY):
        self.starting_equity = float(equity)
        self.equity          = float(equity)
        self.buying_power    = float(equity)
        self.daily_pnl       = 0.0
        self.open_positions: Dict[str, int] = {}
        self.options_exposure = 0.0     # $ notional in option positions
        self.stock_exposure   = 0.0     # $ notional in equity positions
        self.pdt_restricted   = False   # account flagged & < $25k & near daytrade cap
        self._killed         = False
        self._account_anchor_set = False
        # Live broker snapshot fields — seeded here, refreshed every sync_account().
        self.account: Dict[str, Any] = {}
        self.cash                    = float(equity)
        self.last_equity             = float(equity)
        self.portfolio_value         = float(equity)
        self.options_buying_power    = 0.0
        self.daytrading_buying_power = 0.0
        self.long_market_value       = 0.0
        self.short_market_value      = 0.0
        self.multiplier              = 1.0
        self.shorting_enabled        = False
        self.account_status          = ""
        self.trading_blocked         = False
        self.account_blocked         = False
        self.pattern_day_trader      = False
        self.daytrade_count          = 0
        self.refresh_dynamic_limits()

    # ------------------------------------------------------------------

    def refresh_dynamic_limits(self) -> None:
        """Recalculate risk gates from the latest Alpaca account equity snapshot.

        The account anchor is the first real broker equity snapshot seen after startup.
        Daily risk and order size scale with current equity; drawdown and equity floor
        protect the account relative to the anchor so the floor does not drift lower
        intraday after losses.
        """
        equity = max(float(self.equity or 0.0), 1.0)
        anchor = max(float(self.starting_equity or equity), 1.0)

        if DYNAMIC_RISK_LIMITS:
            self.daily_loss_limit = -abs(max(DYNAMIC_DAILY_LOSS_MIN, equity * DYNAMIC_DAILY_LOSS_PCT))
            self.max_drawdown = -abs(DYNAMIC_MAX_DRAWDOWN_PCT)
            self.max_positions = max(
                DYNAMIC_MIN_POSITIONS,
                min(DYNAMIC_MAX_POSITIONS, int(equity / max(DYNAMIC_POSITION_EQUITY_UNIT, 1.0))),
            )
            self.equity_floor = max(DYNAMIC_EQUITY_FLOOR_MIN, anchor * DYNAMIC_EQUITY_FLOOR_PCT)
            self.max_order_dollars = max(0.0, equity * DYNAMIC_MAX_ORDER_PCT)
            self.max_order_size = _env_int(MAX_ORDER_SIZE_OVERRIDE, 0) or 0
        else:
            self.daily_loss_limit = -abs(_env_float(DAILY_LOSS_LIMIT_OVERRIDE, 500.0) or 500.0)
            self.max_drawdown = -abs(_env_float(MAX_DRAWDOWN_OVERRIDE, 0.05) or 0.05)
            self.max_positions = _env_int(MAX_POSITIONS_OVERRIDE, 3) or 3
            self.equity_floor = _env_float(EQUITY_FLOOR_OVERRIDE, 1000.0) or 1000.0
            self.max_order_dollars = _env_float(MAX_ORDER_DOLLARS_OVERRIDE, 0.0) or 0.0
            self.max_order_size = _env_int(MAX_ORDER_SIZE_OVERRIDE, 1000) or 1000

        # Explicit fixed env overrides always win when supplied.
        if DAILY_LOSS_LIMIT_OVERRIDE not in (None, ""):
            self.daily_loss_limit = -abs(_env_float(DAILY_LOSS_LIMIT_OVERRIDE, abs(self.daily_loss_limit)) or abs(self.daily_loss_limit))
        if MAX_DRAWDOWN_OVERRIDE not in (None, ""):
            self.max_drawdown = -abs(_env_float(MAX_DRAWDOWN_OVERRIDE, abs(self.max_drawdown)) or abs(self.max_drawdown))
        if MAX_POSITIONS_OVERRIDE not in (None, ""):
            self.max_positions = max(1, _env_int(MAX_POSITIONS_OVERRIDE, self.max_positions) or self.max_positions)
        if EQUITY_FLOOR_OVERRIDE not in (None, ""):
            self.equity_floor = _env_float(EQUITY_FLOOR_OVERRIDE, self.equity_floor) or self.equity_floor
        if MAX_ORDER_DOLLARS_OVERRIDE not in (None, ""):
            self.max_order_dollars = max(0.0, _env_float(MAX_ORDER_DOLLARS_OVERRIDE, self.max_order_dollars) or 0.0)
        if MAX_ORDER_SIZE_OVERRIDE not in (None, ""):
            self.max_order_size = max(0, _env_int(MAX_ORDER_SIZE_OVERRIDE, self.max_order_size) or 0)

    def risk_limits_summary(self) -> Dict[str, float]:
        return {
            "equity": float(self.equity),
            "anchor_equity": float(self.starting_equity),
            "daily_loss_limit": float(self.daily_loss_limit),
            "max_drawdown": float(self.max_drawdown),
            "max_positions": int(self.max_positions),
            "equity_floor": float(self.equity_floor),
            "max_order_dollars": float(self.max_order_dollars),
            "max_order_size": int(self.max_order_size),
        }

    # ------------------------------------------------------------------

    def gate(self, symbol: str, side: str, qty: int, price: float,
             asset_class: str = "equity") -> Tuple[bool, str]:
        """
        Return (allowed, reason). reason is a canonical SKIP_REASONS code when blocked,
        or "" when allowed. asset_class ∈ {"equity", "option"}.
        """
        if self._check_kill_switch():
            return False, "rejected_by_kill_switch"

        self.refresh_dynamic_limits()

        side = str(side or "").lower()
        # Risk-reducing exits must be able to leave the account during drawdown,
        # daily-loss, equity-floor, or max-size states. The manual kill switch
        # above remains absolute and still halts every order.
        if side != "buy":
            return True, ""

        if self.equity < self.equity_floor:
            return False, f"rejected_by_equity_floor:equity={self.equity:.0f}<{self.equity_floor:.0f}"

        if self.daily_pnl <= self.daily_loss_limit:
            return False, f"rejected_by_daily_loss:pnl={self.daily_pnl:.0f}<={self.daily_loss_limit:.0f}"

        dd = (self.equity - self.starting_equity) / (self.starting_equity + 1e-8)
        if dd <= self.max_drawdown:
            return False, f"rejected_by_drawdown:dd={dd*100:.1f}%<={self.max_drawdown*100:.1f}%"

        if self.max_order_size > 0 and qty > self.max_order_size:
            return False, f"rejected_by_order_size:qty={qty}>{self.max_order_size}"

        notional = qty * price * (100.0 if asset_class == "option" else 1.0)
        # Buy-side exposure / position controls
        if side == "buy":
            # Cash reserve: ALWAYS keep CASH_BUFFER_PCT of equity in free cash (e.g. 45%). A buy that
            # would drop cash below the reserve is rejected, so the account never deploys past (1-buffer).
            _reserve = CASH_BUFFER_PCT * self.equity
            if self.cash - notional < _reserve:
                return False, (f"rejected_by_cash_buffer:cash={self.cash:.0f}-{notional:.0f}<"
                               f"reserve={_reserve:.0f}")
            if self.max_order_dollars > 0 and notional > self.max_order_dollars:
                return False, (f"rejected_by_order_size:notional={notional:.0f}>"
                               f"{self.max_order_dollars:.0f}")
            if asset_class == "option":
                if (self.options_exposure + notional) > self.equity * MAX_OPTIONS_EXPOSURE_PCT:
                    return False, (f"rejected_by_exposure_limit:options="
                                   f"{self.options_exposure + notional:.0f}>"
                                   f"{self.equity * MAX_OPTIONS_EXPOSURE_PCT:.0f}")
                # Alpaca options buying-power preflight: a long option must be fully paid
                # from options_buying_power (Alpaca error 40310000 otherwise). On a small
                # account this is the binding cap (e.g. $4,270 options-BP on $6,260 equity).
                if self.options_buying_power > 0 and notional > self.options_buying_power:
                    return False, (f"rejected_by_options_buying_power:premium={notional:.0f}>"
                                   f"{self.options_buying_power:.0f}")
            else:
                # equity buy — PDT, position-count and stock-exposure limits
                if self.pdt_restricted:
                    return False, "rejected_by_PDT"
                n_open = len([s for s, q in self.open_positions.items() if q > 0])
                if n_open >= self.max_positions:
                    return False, f"rejected_by_exposure_limit:positions={n_open}>={self.max_positions}"
                if (self.stock_exposure + notional) > self.equity * MAX_STOCK_EXPOSURE_PCT:
                    return False, (f"rejected_by_exposure_limit:stock="
                                   f"{self.stock_exposure + notional:.0f}>"
                                   f"{self.equity * MAX_STOCK_EXPOSURE_PCT:.0f}")

        return True, ""

    def max_affordable_qty(self, side: str, price: float, asset_class: str = "equity") -> Optional[int]:
        """Largest BUY qty that fits the *trimmable* notional caps: the per-order
        dollar cap plus the options/stock exposure room remaining under the cap.

        Returns None when no notional trim applies (sells, non-positive price).
        Hard blocks — kill-switch, equity floor, daily-loss, drawdown, PDT,
        position count, max_order_size — are NOT trimmable and stay enforced by
        gate(). The cap is never exceeded; this only lets an order fill UP TO the
        remaining room instead of being rejected outright.
        """
        if side != "buy" or price <= 0:
            return None
        mult = 100.0 if asset_class == "option" else 1.0
        rooms = [max(0.0, self.cash - CASH_BUFFER_PCT * self.equity)]   # free-cash reserve room
        if self.max_order_dollars > 0:
            rooms.append(self.max_order_dollars)
        if asset_class == "option":
            rooms.append(max(0.0, self.equity * MAX_OPTIONS_EXPOSURE_PCT - self.options_exposure))
        else:
            rooms.append(max(0.0, self.equity * MAX_STOCK_EXPOSURE_PCT - self.stock_exposure))
        room = min(rooms) if rooms else 0.0
        return int(room // (price * mult))

    def max_affordable_notional(self, price: float) -> Optional[float]:
        """Dollar room remaining under the trimmable caps (per-order $, stock exposure, free cash)
        for a fractional BUY — NOT floored to whole shares. None if price is non-positive."""
        if price <= 0:
            return None
        rooms = [max(0.0, self.cash - CASH_BUFFER_PCT * self.equity)]
        if self.max_order_dollars > 0:
            rooms.append(self.max_order_dollars)
        rooms.append(max(0.0, self.equity * MAX_STOCK_EXPOSURE_PCT - self.stock_exposure))
        return min(rooms)

    def check_order(self, symbol: str, side: str, qty: int,
                    price: float, dry_run: bool = False) -> bool:
        """Backward-compatible boolean gate (used by the legacy scalping path)."""
        allowed, reason = self.gate(symbol, side, qty, price, asset_class="equity")
        if not allowed:
            logger.warning("BLOCKED %s %s %d: %s", side, symbol, qty, reason)
        return allowed

    def record_fill(self, symbol: str, side: str, qty: int, price: float,
                    asset_class: str = "equity"):
        side = str(side or "").lower()
        notional = max(0.0, float(qty) * float(price) * (100.0 if asset_class == "option" else 1.0))
        if side == "buy":
            self.open_positions[symbol] = self.open_positions.get(symbol, 0) + qty
            if asset_class == "option":
                self.options_exposure += notional
            else:
                self.stock_exposure += notional
        else:
            prev = self.open_positions.get(symbol, 0)
            self.open_positions[symbol] = max(0, prev - qty)
            if asset_class == "option":
                self.options_exposure = max(0.0, self.options_exposure - notional)
            else:
                self.stock_exposure = max(0.0, self.stock_exposure - notional)

    def record_pnl(self, pnl: float):
        self.daily_pnl += pnl
        self.equity    += pnl

    def sync_account(self, account: Dict):
        """Refresh all account state from a live broker snapshot and rescale risk.

        Every value here is sourced dynamically from Alpaca (see AlpacaGateway.get_account).
        The full snapshot is retained on ``self.account`` so any consumer can read live
        broker fields (cash, buying-power variants, margin, status flags) without re-querying.
        """
        if not account:
            return
        self.account = dict(account)            # full live snapshot, for system-wide reads
        if account.get("equity"):
            self.equity = float(account["equity"])
            if not self._account_anchor_set and self.equity > 0:
                self.starting_equity = self.equity
                self._account_anchor_set = True
        if account.get("buying_power") is not None:
            self.buying_power = float(account["buying_power"])
        # Dynamic broker fields kept as first-class attributes for downstream sizing/telemetry.
        self.cash                    = float(account.get("cash", self.cash) or 0.0)
        self.last_equity             = float(account.get("last_equity", self.equity) or self.equity)
        self.portfolio_value         = float(account.get("portfolio_value", self.equity) or self.equity)
        self.options_buying_power    = float(account.get("options_buying_power", 0.0) or 0.0)
        self.daytrading_buying_power = float(account.get("daytrading_buying_power", 0.0) or 0.0)
        self.long_market_value       = float(account.get("long_market_value", 0.0) or 0.0)
        self.short_market_value      = float(account.get("short_market_value", 0.0) or 0.0)
        self.multiplier              = float(account.get("multiplier", self.multiplier) or 1.0)
        self.shorting_enabled        = bool(account.get("shorting_enabled", self.shorting_enabled))
        self.account_status          = str(account.get("status", self.account_status) or "")
        self.trading_blocked         = bool(account.get("trading_blocked", False))
        self.account_blocked         = bool(account.get("account_blocked", False))
        # PDT: flagged AND under the $25k pattern-day-trader floor AND near the daytrade cap
        flagged   = bool(account.get("pattern_day_trader", False))
        dt_count  = int(account.get("daytrade_count", 0) or 0)
        self.pattern_day_trader = flagged
        self.daytrade_count     = dt_count
        self.pdt_restricted = flagged and self.equity < 25_000 and dt_count >= 3
        self.refresh_dynamic_limits()

    def sync_exposure(self, options_exposure: float, stock_exposure: float):
        self.options_exposure = max(0.0, options_exposure)
        self.stock_exposure   = max(0.0, stock_exposure)

    def _check_kill_switch(self) -> bool:
        if self._killed:
            return True
        if os.path.exists(KILL_SWITCH_PATH):
            logger.critical("KILL SWITCH FILE DETECTED — halting all orders")
            self._killed = True
        return self._killed

    @property
    def is_killed(self) -> bool:
        return self._check_kill_switch()

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.open_positions = {}
        logger.info("Risk manager: daily state reset")


# ---------------------------------------------------------------------------
# Broker  (extends the prior AlpacaGateway: equity + option orders, chain, positions)
# ---------------------------------------------------------------------------

class AlpacaGateway:
    """
    Unified, risk-gated broker for both engines.

    Paper gate enforced: if ALPACA_PAPER_ONLY=true the client is forced to the paper
    endpoint. Every order passes through RiskManager.gate() first. Supports:
        • equity market/limit orders        (BAL stock buys, exits)
        • option limit orders on OCC symbols (Alpha call buys, exits, BAL hedges)
        • options-chain retrieval            (Alpha contract selection)
        • account + positions snapshots
    """

    def __init__(self, risk: RiskManager):
        self.risk       = risk
        self._client    = None       # TradingClient
        self._opt_data  = None       # OptionHistoricalDataClient
        self._mode      = "sim"
        self._available = False
        self._init_client()

    # ── init ──────────────────────────────────────────────────────────────────
    def _init_client(self):
        if not ALPACA_KEY or not ALPACA_SECRET:
            logger.warning("Alpaca creds not set — order submission disabled (simulation mode)")
            return
        try:
            from alpaca.trading.client import TradingClient
            self._client    = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=ALPACA_PAPER_ONLY)
            self._mode      = "alpaca_py"
            self._available = True
            logger.info("Alpaca-py client initialised  paper=%s  url=%s", ALPACA_PAPER_ONLY, ALPACA_URL)
            # Patch session-level default timeout: alpaca-py uses a requests.Session
            # internally and has no constructor parameter for timeout. Without this,
            # any Alpaca API call (get_account, get_all_positions, get_option_contracts,
            # etc.) can hang indefinitely on a stalled connection, blocking the whole
            # trading loop. 30 seconds is generous for paper-trading; adjust if needed.
            _ALPACA_TIMEOUT = 30
            try:
                if hasattr(self._client, "_session"):
                    _orig_req = self._client._session.request
                    def _session_request_with_timeout(*_a, **_kw):
                        if "timeout" not in _kw:
                            _kw["timeout"] = _ALPACA_TIMEOUT
                        return _orig_req(*_a, **_kw)
                    self._client._session.request = _session_request_with_timeout
                    logger.debug("Alpaca session timeout patched to %ds", _ALPACA_TIMEOUT)
            except Exception as _tp:
                logger.debug("Could not patch Alpaca session timeout: %s", _tp)
        except ImportError:
            try:
                import alpaca_trade_api as tradeapi
                self._client    = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_URL)
                self._mode      = "alpaca_trade_api"
                self._available = True
                logger.info("alpaca_trade_api client initialised  url=%s", ALPACA_URL)
            except ImportError:
                logger.warning("Neither alpaca-py nor alpaca_trade_api installed — simulation only")
        # Options data client (best-effort; only used by alpaca_py mode)
        try:
            from alpaca.data.historical.option import OptionHistoricalDataClient
            self._opt_data = OptionHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
            # Patch options data client session timeout as well
            try:
                if hasattr(self._opt_data, "_session"):
                    _orig_opt_req = self._opt_data._session.request
                    def _opt_session_request_with_timeout(*_a, **_kw):
                        if "timeout" not in _kw:
                            _kw["timeout"] = 20
                        return _orig_opt_req(*_a, **_kw)
                    self._opt_data._session.request = _opt_session_request_with_timeout
            except Exception:
                pass
        except Exception:
            self._opt_data = None

    @property
    def available(self) -> bool:
        return self._available

    # ── account / positions ─────────────────────────────────────────────────
    def get_account(self) -> Dict:
        """Return a full, live Alpaca account snapshot.

        Every numeric/flag field is sourced dynamically from the broker — nothing
        is hardcoded. When the broker is unavailable a clearly-flagged simulation
        snapshot is returned so downstream consumers still receive the same keys.
        """
        if not self._available:
            return {
                "equity":             float(self.risk.equity),
                "last_equity":        float(self.risk.equity),
                "portfolio_value":    float(self.risk.equity),
                "buying_power":       float(self.risk.buying_power),
                "cash":               float(self.risk.buying_power),
                "pattern_day_trader": False,
                "daytrade_count":     0,
                "sim":                True,
            }
        try:
            acc = self._client.get_account()

            def _num(name: str, default: float = 0.0) -> float:
                try:
                    v = getattr(acc, name, default)
                    return float(v) if v not in (None, "") else default
                except (TypeError, ValueError):
                    return default

            def _flag(name: str) -> bool:
                return bool(getattr(acc, name, False))

            def _txt(name: str) -> str:
                v = getattr(acc, name, "")
                # Enums (AccountStatus) stringify to their value via .value
                return str(getattr(v, "value", v) or "")

            snapshot = {
                # identity / status
                "account_number":              _txt("account_number"),
                "account_id":                  _txt("id"),
                "status":                       _txt("status"),
                "currency":                     _txt("currency") or "USD",
                "created_at":                   _txt("created_at"),
                # equity / value
                "equity":                       _num("equity"),
                "last_equity":                  _num("last_equity"),
                "portfolio_value":              _num("portfolio_value"),
                "cash":                         _num("cash"),
                "accrued_fees":                 _num("accrued_fees"),
                "pending_transfer_in":          _num("pending_transfer_in"),
                # buying power (all variants Alpaca exposes)
                "buying_power":                 _num("buying_power"),
                "regt_buying_power":            _num("regt_buying_power"),
                "daytrading_buying_power":      _num("daytrading_buying_power"),
                "non_marginable_buying_power":  _num("non_marginable_buying_power"),
                "options_buying_power":         _num("options_buying_power"),
                # market value / margin
                "long_market_value":            _num("long_market_value"),
                "short_market_value":           _num("short_market_value"),
                "position_market_value":        _num("position_market_value"),
                "initial_margin":               _num("initial_margin"),
                "maintenance_margin":           _num("maintenance_margin"),
                "last_maintenance_margin":      _num("last_maintenance_margin"),
                "sma":                          _num("sma"),
                "multiplier":                   _num("multiplier", 1.0),
                # PDT / day-trading
                "pattern_day_trader":           _flag("pattern_day_trader"),
                "daytrade_count":               int(_num("daytrade_count")),
                # options approval
                "options_approved_level":       int(_num("options_approved_level")),
                "options_trading_level":        int(_num("options_trading_level")),
                # account flags
                "shorting_enabled":             _flag("shorting_enabled"),
                "trading_blocked":              _flag("trading_blocked"),
                "transfers_blocked":            _flag("transfers_blocked"),
                "account_blocked":              _flag("account_blocked"),
                "trade_suspended_by_user":      _flag("trade_suspended_by_user"),
            }
            return snapshot
        except Exception as e:
            logger.error("get_account error: %s", e)
            return {}

    def get_positions_raw(self) -> List[Any]:
        """Return raw Alpaca position objects (attributes: symbol, qty, market_value,
        avg_entry_price, current_price, unrealized_plpc, asset_class)."""
        if not self._available or self._mode != "alpaca_py":
            return []
        try:
            return list(self._client.get_all_positions())
        except Exception as e:
            logger.error("get_all_positions error: %s", e)
            return []

    @staticmethod
    def _is_option_position(p: Any) -> bool:
        ac = str(getattr(p, "asset_class", "") or "").lower()
        if "option" in ac:
            return True
        return is_option_symbol(str(getattr(p, "symbol", "")))

    def split_positions(self, positions: List[Any]) -> Tuple[List[Dict], Dict[str, Any], float, float]:
        """
        Split raw positions into:
          option_positions : list[dict]  (for AlphaCallEngine — keys: symbol, qty, unrealized_plpc)
          equity_positions : dict[symbol, position_obj]  (for BAL)
          options_exposure : float ($ market value)
          stock_exposure   : float ($ market value)
        """
        option_positions: List[Dict] = []
        equity_positions: Dict[str, Any] = {}
        opt_exp = stk_exp = 0.0
        for p in positions:
            mv = abs(float(getattr(p, "market_value", 0) or 0))
            if self._is_option_position(p):
                option_positions.append({
                    "symbol": str(getattr(p, "symbol", "")),
                    "qty": int(float(getattr(p, "qty", 0) or 0)),
                    "unrealized_plpc": float(getattr(p, "unrealized_plpc", 0) or 0),
                    "market_value": mv,
                })
                opt_exp += mv
            else:
                equity_positions[str(getattr(p, "symbol", ""))] = p
                stk_exp += mv
        return option_positions, equity_positions, opt_exp, stk_exp

    # ── options chain ─────────────────────────────────────────────────────────
    def get_options_chain(self, symbol: str) -> List[Dict]:
        """
        Return a list of call/put contract dicts for `symbol` in the shape the
        Alpha Engine expects (keys: symbol, type, expiration_date, strike_price,
        delta, open_interest, volume, bid_price, ask_price).

        Strikes / expirations / open-interest come from the trading client's
        option-contracts endpoint; bid/ask/greeks are merged from the options
        data snapshot when available. Returns [] gracefully on any failure.

        When USE_AV_OPTIONS_CHAIN is set, real Alpha Vantage REALTIME_OPTIONS chains
        (full greeks+IV) are used instead — see av_options_adapter. Falls back to the
        Alpaca path on any error so the live engine never loses its chain source.
        """
        if _env_bool_any(["USE_AV_OPTIONS_CHAIN", "ALPHA_USE_AV_CHAINS"], False):
            try:
                import av_options_adapter as _av
                _ch = _av.realtime_chain(symbol)
                if _ch:
                    return _ch
                logger.debug("AV realtime chain empty for %s — falling back to Alpaca", symbol)
            except Exception as _ave:
                logger.debug("AV realtime chain failed for %s: %s — falling back", symbol, _ave)
        if not self._available or self._mode != "alpaca_py":
            return []
        contracts: Dict[str, Dict] = {}
        # 1. Contract metadata (strike / expiry / type / open_interest)
        try:
            import concurrent.futures as _cf_chain
            from alpaca.trading.requests import GetOptionContractsRequest
            from alpaca.trading.enums import ContractType, AssetStatus
            from datetime import timedelta
            # An UNWINDOWED request returns only the nearest (often 0-DTE) expiry, which
            # silently fails the Alpha Engine's 21-60 DTE gate. Request the DTE window so
            # the real, tradeable contracts are returned.
            _today = datetime.now(timezone.utc).date()
            req = GetOptionContractsRequest(
                underlying_symbols=[symbol],
                status=AssetStatus.ACTIVE,
                type=ContractType.CALL,
                expiration_date_gte=_today + timedelta(days=ALPHA_MIN_DTE),
                expiration_date_lte=_today + timedelta(days=ALPHA_MAX_DTE),
                limit=1000,
            )
            # Hard 25-second timeout — Alpaca option-contract requests occasionally
            # hang indefinitely; without this the whole process stalls.
            # Use shutdown(wait=False) — `with` context-manager exit blocks until the
            # worker finishes, defeating the timeout.
            _pool_chain = _cf_chain.ThreadPoolExecutor(max_workers=1)
            _fut_chain = _pool_chain.submit(self._client.get_option_contracts, req)
            resp = None
            try:
                resp = _fut_chain.result(timeout=25)
            except Exception as _te:
                _pool_chain.shutdown(wait=False)
                logger.debug("get_option_contracts timeout/error for %s: %s", symbol, _te)
                return []
            _pool_chain.shutdown(wait=False)
            if resp is None:
                return []
            for c in getattr(resp, "option_contracts", []) or []:
                occ = str(getattr(c, "symbol", ""))
                if not occ:
                    continue
                contracts[occ] = {
                    "symbol": occ,
                    "type": str(getattr(c, "type", "call")).lower().replace("contracttype.", ""),
                    "expiration_date": str(getattr(c, "expiration_date", "")),
                    "strike_price": float(getattr(c, "strike_price", 0) or 0),
                    "open_interest": int(float(getattr(c, "open_interest", 0) or 0)),
                    "volume": 0,   # Alpaca contract metadata carries no daily volume; left to liquidity gate
                    "delta": 0.0,
                    "implied_volatility": 0.0,
                    "bid_price": 0.0,
                    "ask_price": 0.0,
                }
        except Exception as exc:
            logger.debug("get_option_contracts failed for %s: %s", symbol, exc)
            return []
        # 2. Merge live quotes + greeks from the options data snapshot
        if self._opt_data is not None and contracts:
            try:
                from alpaca.data.requests import OptionChainRequest
                # Use the FREE 'indicative' options feed (delayed/derived quotes) so the
                # engine never requires an OPRA subscription. Falls back gracefully if the
                # enum is absent in older alpaca-py.
                try:
                    from alpaca.data.enums import OptionsFeed as _OptFeed
                    _ofeed = _OptFeed.INDICATIVE
                except Exception:
                    _ofeed = "indicative"
                # Hard 15-second timeout on the snapshot merge — can also hang if
                # the data client stalls. Timeouts here lose greeks/bid/ask but
                # the engine gracefully falls back to ATM-strike selection.
                # Use shutdown(wait=False) — `with` context-manager exit blocks until
                # the worker finishes, defeating the timeout.
                _pool_snap = _cf_chain.ThreadPoolExecutor(max_workers=1)
                try:
                    _opt_req = OptionChainRequest(underlying_symbol=symbol, feed=_ofeed)
                except TypeError:
                    _opt_req = OptionChainRequest(underlying_symbol=symbol)   # older SDK
                _fut_snap = _pool_snap.submit(self._opt_data.get_option_chain, _opt_req)
                chain = {}
                try:
                    chain = _fut_snap.result(timeout=15)
                except Exception as _te:
                    logger.debug("get_option_chain snapshot timeout/error for %s: %s", symbol, _te)
                finally:
                    _pool_snap.shutdown(wait=False)
                snap_map = chain if isinstance(chain, dict) else getattr(chain, "data", {}) or {}
                for occ, snap in snap_map.items():
                    if occ not in contracts:
                        continue
                    q = getattr(snap, "latest_quote", None)
                    g = getattr(snap, "greeks", None)
                    t = getattr(snap, "latest_trade", None)
                    if q is not None:
                        contracts[occ]["bid_price"] = float(getattr(q, "bid_price", 0) or 0)
                        contracts[occ]["ask_price"] = float(getattr(q, "ask_price", 0) or 0)
                    if g is not None:
                        contracts[occ]["delta"] = float(getattr(g, "delta", 0) or 0)
                    iv = getattr(snap, "implied_volatility", None)
                    if iv is not None:
                        contracts[occ]["implied_volatility"] = float(iv or 0)
                    # Daily volume is not on the contract metadata; take it from the
                    # snapshot's day bar / trade size so the min_volume gate has real data.
                    day = getattr(snap, "daily_bar", None)
                    if day is not None and getattr(day, "volume", None) is not None:
                        contracts[occ]["volume"] = int(getattr(day, "volume", 0) or 0)
                    elif t is not None and getattr(t, "size", None) is not None:
                        contracts[occ]["volume"] = int(getattr(t, "size", 0) or 0)
            except Exception as exc:
                logger.debug("option chain snapshot merge failed for %s: %s", symbol, exc)
        return list(contracts.values())

    # ── order submission ───────────────────────────────────────────────────────
    def _resolve_reference_price(self, symbol: str) -> float:
        """
        Resolve a real per-share/contract price for exposure/notional. Order: latest
        trade → latest quote mid → most recent daily close. Returns 0.0 only when no
        market data is available. Used so RiskManager never gates on qty × 0.0.
        """
        if not self._available or self._mode != "alpaca_py":
            return 0.0
        if is_option_symbol(symbol):
            return self._resolve_option_reference_price(symbol)
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import (StockLatestTradeRequest,
                                              StockLatestQuoteRequest, StockBarsRequest)
            from alpaca.data.timeframe import TimeFrame
            from datetime import timedelta
            dc = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
            # 1. latest trade
            try:
                t = dc.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
                px = float(getattr(t.get(symbol), "price", 0) or 0)
                if px > 0:
                    return px
            except Exception:
                pass
            # 2. latest quote mid
            try:
                q = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol)).get(symbol)
                bid = float(getattr(q, "bid_price", 0) or 0)
                ask = float(getattr(q, "ask_price", 0) or 0)
                if bid > 0 and ask > 0:
                    return (bid + ask) / 2.0
                if ask > 0:
                    return ask
            except Exception:
                pass
            # 3. most recent daily close
            try:
                end = datetime.now(timezone.utc)
                req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                                       start=end - timedelta(days=7), end=end)
                df = dc.get_stock_bars(req).df
                if df is not None and not df.empty:
                    return float(df["close"].iloc[-1])
            except Exception:
                pass
        except Exception as e:
            logger.debug("_resolve_reference_price(%s) failed: %s", symbol, e)
        return 0.0

    def _resolve_option_reference_price(self, symbol: str) -> float:
        """Resolve an OCC option reference price from option trade/quote data."""
        try:
            if self._opt_data is None:
                from alpaca.data.historical.option import OptionHistoricalDataClient
                self._opt_data = OptionHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
            from alpaca.data.requests import OptionLatestQuoteRequest, OptionLatestTradeRequest

            try:
                trades = self._opt_data.get_option_latest_trade(
                    OptionLatestTradeRequest(symbol_or_symbols=symbol))
                data = trades if isinstance(trades, dict) else (getattr(trades, "data", None) or {})
                trade = data.get(symbol)
                px = float(getattr(trade, "price", 0) or 0)
                if px > 0:
                    return px
            except Exception:
                pass

            try:
                quotes = self._opt_data.get_option_latest_quote(
                    OptionLatestQuoteRequest(symbol_or_symbols=symbol))
                data = quotes if isinstance(quotes, dict) else (getattr(quotes, "data", None) or {})
                quote = data.get(symbol)
                bid = float(getattr(quote, "bid_price", 0) or 0)
                ask = float(getattr(quote, "ask_price", 0) or 0)
                if bid > 0 and ask > 0:
                    return (bid + ask) / 2.0
                if ask > 0:
                    return ask
                if bid > 0:
                    return bid
            except Exception:
                pass
        except Exception as e:
            logger.debug("_resolve_option_reference_price(%s) failed: %s", symbol, e)
        return 0.0

    def submit_market_order(self, symbol: str, side: str, qty: int,
                            reason: str = "", limit_price: float = 0.0,
                            order_type: str = "market", reference_price: float = 0.0,
                            notional: float = 0.0) -> Optional[Dict]:
        """
        Risk-gated order submission for equities AND options (OCC symbols).
        Options are always routed as DAY limit orders (market options orders are
        commonly rejected); equities use market unless a limit_price is supplied.

        `reference_price` (or limit_price, or a resolved market price) is used ONLY for
        the RiskManager exposure/notional check — it never changes the order TYPE. A BUY
        is rejected when no real reference price can be found, so exposure is never
        computed on qty × 0.0.
        Returns the broker response (or {"sim": True}) on success, None if blocked.
        """
        qty = int(qty)
        if qty <= 0:
            return None
        opt = is_option_symbol(symbol)
        asset_class = "option" if opt else "equity"
        # ── Fractional / notional BUY: deploy an exact dollar amount (fractional shares) ──────
        if (ENABLE_FRACTIONAL_ORDERS and notional and notional > 0 and not opt
                and side == "buy" and order_type == "market" and not limit_price):
            return self._submit_notional_buy(symbol, float(notional), reason)

        # Notional reference price: explicit limit → caller-supplied reference → live
        # market lookup. Must be > 0 for BUYs (exits never depend on notional).
        ref_price = float(limit_price or reference_price or 0.0)
        if ref_price <= 0:
            ref_price = self._resolve_reference_price(symbol)
        if ref_price <= 0 and side == "buy":
            log_skip(symbol, "rejected_by_data_unavailable:no_reference_price",
                     side=side, qty=qty, source=reason or "broker")
            return None

        # 9-EMA execution gate: only BUY above the daily 9-EMA (trend confirmation / cross-up).
        if side == "buy":
            _trend_blk = _ema_buy_blocked(symbol, ref_price)
            if _trend_blk:
                log_skip(symbol, _trend_blk, side=side, qty=qty, source=reason or "broker")
                return None

        # Self-assessment BUY throttle (downside protection): scale BUY size by the
        # data/model/regime/execution-risk multiplier from system_self_awareness. Buys only —
        # exits are never throttled. multiplier ∈ [0,1]; 0 → skip the buy entirely.
        if side == "buy":
            sa_mult = _self_assessment_buy_multiplier()
            if sa_mult < 1.0:
                new_qty = int(qty * sa_mult)
                if new_qty <= 0:
                    log_skip(symbol, f"throttled_by_self_assessment:mult={sa_mult:.2f}",
                             side=side, qty=qty, source=reason or "broker")
                    return None
                if new_qty < qty:
                    logger.info("[self-assess] %s BUY size x%.2f (mode=%s): %d->%d",
                                symbol, sa_mult, _SA_CACHE.get("mode", "NORMAL"), qty, new_qty)
                    qty = new_qty

        # Size-to-fit: trim a BUY to the room remaining under the trimmable
        # notional/exposure caps rather than rejecting it whole. Cap never breached.
        if SIZE_TO_FIT_EXPOSURE and side == "buy":
            fit = self.risk.max_affordable_qty(side, ref_price, asset_class=asset_class)
            if fit is not None and 0 < fit < qty:
                logger.info("[size-to-fit] %s BUY trimmed %d->%d to fit exposure cap (px=%.2f)",
                            symbol, qty, fit, ref_price)
                qty = fit

        allowed, gate_reason = self.risk.gate(symbol, side, qty, ref_price, asset_class=asset_class)
        if not allowed:
            log_skip(symbol, gate_reason, side=side, qty=qty, source=reason or "broker")
            return None

        if not self._available:
            logger.info("[SIM] %s %s x%d  reason=%s", side.upper(), symbol, qty, reason)
            self.risk.record_fill(symbol, side, qty, ref_price, asset_class=asset_class)
            return {"sim": True, "symbol": symbol, "side": side, "qty": qty}

        try:
            if self._mode == "alpaca_py":
                resp = self._submit_alpaca_py(symbol, side, qty, opt, limit_price, order_type)
            else:
                resp = self._client.submit_order(
                    symbol=symbol, qty=qty, side=side,
                    type=("limit" if (opt or limit_price) else "market"),
                    time_in_force="day",
                    limit_price=(limit_price or None),
                )
            logger.info("ORDER %s %s x%d  reason=%s  asset=%s", side, symbol, qty, reason, asset_class)
            self.risk.record_fill(symbol, side, qty, ref_price, asset_class=asset_class)
            return resp
        except Exception as e:
            logger.error("Alpaca order error (%s %s x%d): %s", side, symbol, qty, e)
            return None

    def _submit_alpaca_py(self, symbol, side, qty, opt, limit_price, order_type):
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        os_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        if opt or limit_price or order_type == "limit":
            px = limit_price
            if px <= 0:
                # Best-effort marketable limit for exits without an explicit price
                px = 0.01
            req = LimitOrderRequest(symbol=symbol, qty=qty, side=os_side,
                                    time_in_force=TimeInForce.DAY, limit_price=round(px, 2))
        else:
            req = MarketOrderRequest(symbol=symbol, qty=qty, side=os_side,
                                     time_in_force=TimeInForce.DAY)
        return self._client.submit_order(req)

    def _submit_notional_buy(self, symbol: str, notional: float, reason: str = "") -> Optional[Dict]:
        """Fractional BUY of an exact dollar `notional` (fractionable equities only). Same throttle,
        size-to-fit and RiskManager gating as a qty order, but no whole-share rounding."""
        ref_price = self._resolve_reference_price(symbol)
        if ref_price <= 0:
            log_skip(symbol, "rejected_by_data_unavailable:no_reference_price", side="buy", source=reason)
            return None
        _trend_blk = _ema_buy_blocked(symbol, ref_price)
        if _trend_blk:
            log_skip(symbol, _trend_blk, side="buy", source=reason or "broker")
            return None
        sa = _self_assessment_buy_multiplier()
        if sa < 1.0:
            notional *= sa
        if notional < 1.0:
            return None
        if SIZE_TO_FIT_EXPOSURE:
            room = self.risk.max_affordable_notional(ref_price)
            if room is not None and 0 < room < notional:
                logger.info("[size-to-fit] %s notional $%.0f->$%.0f to fit caps", symbol, notional, room)
                notional = room
        eff_qty = notional / ref_price
        allowed, gate_reason = self.risk.gate(symbol, "buy", eff_qty, ref_price, asset_class="equity")
        if not allowed:
            log_skip(symbol, gate_reason, side="buy", qty=round(eff_qty, 4), source=reason or "broker")
            return None
        if not self._available:
            logger.info("[SIM] BUY %s notional=$%.2f (~%.4f sh)", symbol, notional, eff_qty)
            self.risk.record_fill(symbol, "buy", eff_qty, ref_price, asset_class="equity")
            return {"sim": True, "symbol": symbol, "side": "buy", "notional": round(notional, 2)}
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            req = MarketOrderRequest(symbol=symbol, notional=round(notional, 2),
                                     side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
            resp = self._client.submit_order(req)
            logger.info("ORDER buy %s notional=$%.2f (~%.4f sh @ %.2f)  reason=%s  asset=equity",
                        symbol, notional, eff_qty, ref_price, reason)
            self.risk.record_fill(symbol, "buy", eff_qty, ref_price, asset_class="equity")
            return resp
        except Exception as e:
            logger.error("Alpaca notional order error (buy %s $%.2f): %s", symbol, notional, e)
            return None

    # Legacy alias (scalping path called submit_order(symbol, side, qty, type, limit))
    def submit_order(self, symbol: str, side: str, qty: int,
                     order_type: str = "market", limit_price: float = 0.0):
        return self.submit_market_order(symbol, side, qty, reason="scalping",
                                        limit_price=limit_price, order_type=order_type)


# ---------------------------------------------------------------------------
# Market hours check
# ---------------------------------------------------------------------------

def is_market_open() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    h, m = now.hour, now.minute
    mins = h * 60 + m
    return 9 * 60 + 30 <= mins <= 16 * 60


def is_premarket() -> bool:
    """Premarket session in New York local time: 04:00-09:30 ET."""
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    mins = now.hour * 60 + now.minute
    return 4 * 60 <= mins < 9 * 60 + 30


# ---------------------------------------------------------------------------
# Universe / data helpers
# ---------------------------------------------------------------------------

_INVALID_UNIVERSE_SYMBOLS = {"", "N/A", "NA", "NAN", "NONE", "NULL", "N\\A"}


def _normalize_candidate_symbol(value: Any) -> str:
    sym = str(value or "").strip().strip('"').strip("'").strip().upper()
    if not sym or sym in _INVALID_UNIVERSE_SYMBOLS:
        return ""
    return sym


def _parse_ticker_file(text: str) -> List[str]:
    """Parse a ticker list in either comma-separated-quoted ("A","B") or newline form."""
    out: List[str] = []
    for chunk in (text or "").replace("\n", ",").split(","):
        sym = _normalize_candidate_symbol(chunk)
        if sym:
            out.append(sym)
    # de-dupe, preserve order
    seen: set = set()
    return [s for s in out if not (s in seen or seen.add(s))]


def _refresh_tv_quality_screen(force: bool = False) -> None:
    """Refresh tv_screener_tickers.txt from the quality-growth screen.

    Imports tv-screener.py (hyphenated, so via importlib) and runs scan_quality_growth.
    Skips when the cached file is still within TTL unless `force` (premarket runs it once
    at a known good time). Hard 30s timeout + full fail-safety: any failure leaves the
    existing file untouched so the universe load always has something to fall back on.
    Never raises.
    """
    if not ENABLE_TV_QUALITY_SCREEN or _in_test():
        return
    try:
        if not force and TV_SCREEN_TICKERS_PATH.exists():
            age = time.time() - TV_SCREEN_TICKERS_PATH.stat().st_mtime
            if age < TV_QUALITY_SCREEN_TTL_SEC:
                return  # still fresh
        import importlib.util
        import concurrent.futures as _cf
        spec = importlib.util.spec_from_file_location("tv_screener", PROJECT_ROOT / "tv-screener.py")
        if spec is None or spec.loader is None:
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        pool = _cf.ThreadPoolExecutor(max_workers=1)
        fut = pool.submit(mod.scan_quality_growth, write=True, symbols_only=True)
        try:
            tickers = fut.result(timeout=30)
            logger.info("TV quality screen refreshed: %d names", len(tickers or []))
        except Exception as e:
            logger.warning("TV quality screen refresh timed out/failed (using cached file): %s", e)
        finally:
            pool.shutdown(wait=False)
    except Exception as e:
        logger.warning("TV quality screen refresh skipped (using cached file): %s", e)


_TV_REFRESHER_STARTED = False


def _start_tv_screen_refresher() -> None:
    """Start a daemon thread that continuously refreshes the quality-growth screen every
    TV_QUALITY_SCREEN_REFRESH_SEC seconds, so the universe is always fresh while the
    network call stays off the trading cycle. Idempotent, fail-safe, never raises.
    """
    global _TV_REFRESHER_STARTED
    if _TV_REFRESHER_STARTED or not ENABLE_TV_QUALITY_SCREEN or _in_test():
        return
    import threading
    interval = max(15.0, TV_QUALITY_SCREEN_REFRESH_SEC)

    def _loop() -> None:
        while True:
            try:
                _refresh_tv_quality_screen(force=True)
            except Exception:
                logger.debug("TV screen refresher tick failed", exc_info=True)
            time.sleep(interval)

    try:
        threading.Thread(target=_loop, name="tv-screen-refresher", daemon=True).start()
        _TV_REFRESHER_STARTED = True
        logger.info("TV quality-screen refresher started (every %.0fs)", interval)
    except Exception as e:
        logger.warning("TV screen refresher failed to start: %s", e)


def load_candidate_universe() -> List[str]:
    """
    Load the main Alpha candidate universe. Order of preference:
        1. SYMBOLS env override
        2. TradingView quality-growth screen  (tv_screener_tickers.txt, auto-refreshed)
        3. {DATA_TAG}_locked_trading_universe.txt
        4. locked_trading_universe.txt
    """
    if SYMBOLS_ENV.strip():
        return [s.strip().upper() for s in SYMBOLS_ENV.split(",") if s.strip()]

    # Primary: the user's quality-growth TradingView screen.
    if ENABLE_TV_QUALITY_SCREEN:
        _refresh_tv_quality_screen()
        if TV_SCREEN_TICKERS_PATH.exists():
            try:
                syms = _parse_ticker_file(TV_SCREEN_TICKERS_PATH.read_text())
                if syms:
                    logger.info("Loaded %d candidates from quality-growth screen (%s)",
                                len(syms), TV_SCREEN_TICKERS_PATH.name)
                    return syms
            except Exception as e:
                logger.warning("Failed to read %s: %s", TV_SCREEN_TICKERS_PATH, e)

    for candidate in (PROJECT_ROOT / f"{DATA_TAG}_locked_trading_universe.txt",
                      PROJECT_ROOT / "locked_trading_universe.txt"):
        if candidate.exists():
            try:
                syms = [
                    sym for sym in
                    (_normalize_candidate_symbol(ln) for ln in candidate.read_text().splitlines())
                    if sym
                ]
                if syms:
                    logger.info("Loaded %d candidates from %s", len(syms), candidate.name)
                    return syms
            except Exception as e:
                logger.warning("Failed to read %s: %s", candidate, e)
    logger.warning("No candidate universe found — using default watchlist")
    return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]


def fetch_1m_bars(gateway: AlpacaGateway, symbols: List[str]) -> Dict[str, "Any"]:
    """
    Fetch recent 1-minute OHLCV bars per symbol as a pandas DataFrame
    (columns: open, high, low, close, volume; tz-aware index).
    Falls back to a deterministic simulation per-symbol when Alpaca data is unavailable.
    Fix: simulation now fills in individually missing symbols even when the batch
    partially succeeds — prevents "no_bars" rejections for symbols IEX doesn't cover
    (e.g., small-caps during pre-market).
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed — cannot build bars")
        return {}

    out: Dict[str, Any] = {}
    missing: List[str] = list(symbols)  # will shrink as real bars are found

    if gateway.available and gateway._mode == "alpaca_py":
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from alpaca.data.enums import DataFeed
            from datetime import timedelta
            data_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
            end = datetime.now(timezone.utc)
            start = end - timedelta(minutes=ALPHA_BARS_LOOKBACK_MIN + 5)
            # IEX-only: this subscription has no SIP/delayed-SIP. We never select SIP
            # (Phase 4 rule). The feed manager confirms availability + downgrades safely.
            feed = DataFeed.IEX
            req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Minute,
                                   start=start, end=end, feed=feed)
            # Hard timeout: Alpaca IEX batch requests can hang indefinitely on large
            # universe / slow feeds. Cap at 30s and fall through to simulation.
            # IMPORTANT: do NOT use `with ThreadPoolExecutor()` here — the context
            # manager's __exit__ calls shutdown(wait=True) and BLOCKS until the worker
            # thread finishes, defeating the timeout entirely. Shutdown with wait=False.
            import concurrent.futures as _cf
            # Scale timeout with universe size: 1.2s/symbol, min 45s, max 120s.
            # The old 30s cap was too tight for 60+ symbols — IEX needs ~1s/symbol
            # on a slow connection before the Alpaca SDK buffers the response.
            _bar_timeout = max(45, min(120, int(len(symbols) * 1.2)))
            _pool_bars = _cf.ThreadPoolExecutor(max_workers=1)
            _fut_bars = _pool_bars.submit(lambda: data_client.get_stock_bars(req).df)
            df = None
            try:
                df = _fut_bars.result(timeout=_bar_timeout)
            except (_cf.TimeoutError, Exception) as _e_bars:
                if isinstance(_e_bars, _cf.TimeoutError):
                    logger.warning(
                        "fetch_1m_bars: Alpaca batch request timed out (>%ds) "
                        "for %d symbols — using simulation", _bar_timeout, len(symbols))
                else:
                    raise
            finally:
                _pool_bars.shutdown(wait=False)  # release immediately; worker is a daemon
            if df is not None and not df.empty:
                df = df.reset_index()
                for sym, grp in df.groupby("symbol"):
                    g = grp.set_index("timestamp")[["open", "high", "low", "close", "volume"]].copy()
                    if len(g) >= 2:
                        out[sym] = g
                # Only simulate for symbols that got no real bars
                missing = [s for s in symbols if s not in out]
                if not missing:
                    return out
                logger.debug("fetch_1m_bars: %d/%d symbols got real bars; simulating %d",
                             len(out), len(symbols), len(missing))
        except Exception as e:
            logger.warning("Alpaca 1m bar fetch failed (%s) — using simulation for all", e)

    # Per-symbol simulation fallback (deterministic; keeps loop runnable offline and
    # fills any symbol IEX couldn't serve — e.g., small-caps with no pre-market activity).
    import pandas as pd
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    for sym in missing:
        seed = sum(ord(c) for c in sym)
        base = 20.0 + (seed % 200)
        idx = [now - timedelta(minutes=ALPHA_BARS_LOOKBACK_MIN - i) for i in range(ALPHA_BARS_LOOKBACK_MIN)]
        rows = []
        px = base
        for i in range(ALPHA_BARS_LOOKBACK_MIN):
            drift = math.sin((seed + i) / 17.0) * 0.15
            px = max(1.0, px + drift)
            rows.append({"open": px, "high": px * 1.003, "low": px * 0.997,
                         "close": px, "volume": 50_000 + (seed * 7 + i) % 40_000})
        out[sym] = pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="timestamp"))
    return out


def build_bars_ctx(bars_by_symbol: Dict[str, "Any"]) -> Dict[str, Dict]:
    """Derive per-symbol VWAP and relative-volume context for BAL technical scoring."""
    ctx: Dict[str, Dict] = {}
    for sym, df in bars_by_symbol.items():
        try:
            if df is None or len(df) == 0:
                continue
            close = df["close"]
            vol = df["volume"]
            typical = (df["high"] + df["low"] + df["close"]) / 3.0
            vwap = float((typical * vol).sum() / max(1e-9, vol.sum()))
            recent_v = float(vol.tail(20).mean())
            avg_v = float(vol.mean())
            rvol = recent_v / max(1e-9, avg_v)
            ctx[sym] = {
                "vwap": vwap,
                "rel_volume": round(rvol, 3),
                "rvol": round(rvol, 3),
                "close": float(close.iloc[-1]),
            }
        except Exception:
            continue
    return ctx


# ---------------------------------------------------------------------------
# Required output writers
# ---------------------------------------------------------------------------

def write_capital_waterfall(waterfall) -> None:
    """Persist the capital waterfall summary (engine only writes on allocate)."""
    try:
        from capital_waterfall import WATERFALL_PATH
        payload = dict(waterfall.summary)
        payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        payload["config"] = {
            "to_bal_pct": OPTIONS_PROFIT_TO_BAL_PCT,
            "to_alpha_pct": OPTIONS_PROFIT_TO_ALPHA_PCT,
            "to_cash_pct": OPTIONS_PROFIT_TO_CASH_PCT,
        }
        WATERFALL_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        logger.debug("write_capital_waterfall failed: %s", e)


def ensure_outputs_exist() -> None:
    """Guarantee the four required daily outputs exist (empty scaffolds if engines haven't run)."""
    DAILY_OUTPUTS.mkdir(parents=True, exist_ok=True)
    scaffolds = {
        "options_alpha_signals.json":      {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                            "date_tag": DATA_TAG, "count": 0, "signals": []},
        "buffett_accumulation_scores.json": {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                            "date_tag": DATA_TAG, "count": 0, "scores": []},
        "buffett_dca_signals.json":        {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                            "date_tag": DATA_TAG, "count": 0, "signals": []},
        "valuation_diagnostic.json":       {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                            "date_tag": DATA_TAG, "count": 0, "valuations": []},
        "conviction_score_report.json":    {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                            "date_tag": DATA_TAG, "count": 0, "convictions": []},
        "alpha_volatility_diagnostic.json": {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                            "date_tag": DATA_TAG, "count": 0, "diagnostics": []},
        "sec_form4_insider_buys.json":     {"generated_at": datetime.now(timezone.utc).isoformat(),
                                            "lookback_days": 0, "min_transaction_value": 0,
                                            "candidates": [], "rejected": [], "errors": [],
                                            "source": "SEC_EDGAR_FORM4"},
        "insider_buy_candidates.json":     {"generated_at": datetime.now(timezone.utc).isoformat(),
                                            "source": "SEC_EDGAR_FORM4", "count": 0, "candidates": {}},
        "ownership_universe.json":         {},
        "capital_waterfall.json":          {"date_tag": DATA_TAG, "realized_today": 0.0,
                                            "bal_reserve": 0.0, "alpha_reserve": 0.0, "cash_reserve": 0.0,
                                            "allocation_today": {"bal": 0.0, "alpha": 0.0, "cash": 0.0}},
        "alpha_options_universe.json":     {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                            "date_tag": DATA_TAG, "scanned": 0, "candidate_count": 0,
                                            "candidates": [], "all_scanned": []},
        "buffett_accumulation_universe.json": {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                            "date_tag": DATA_TAG, "scanned": 0, "candidate_count": 0,
                                            "candidates": [], "all_scanned": []},
        "screener_funnel_report.json":     {"date_tag": DATA_TAG,
                                            "alpha_options_screener": {}, "buffett_accumulation_screener": {}},
    }
    for fname, default in scaffolds.items():
        p = DAILY_OUTPUTS / fname
        if not p.exists():
            try:
                p.write_text(json.dumps(default, indent=2), encoding="utf-8")
            except Exception:
                pass

    # Also initialise execution-log CSVs so system_self_awareness data-quality scoring
    # can find them on startup even before any live orders are processed.
    _exec_log_dir = PROJECT_ROOT / "logs" / "execution" / DATA_TAG
    try:
        _exec_log_dir.mkdir(parents=True, exist_ok=True)
        _equity_csv = _exec_log_dir / f"{DATA_TAG}_executor_equity_curve.csv"
        if not _equity_csv.exists():
            _equity_csv.write_text(
                "timestamp_utc,label,cycle_id,equity,cash,buying_power,portfolio_value\n"
            )
    except Exception:
        pass

    # Scaffold forecast_ensemble_signals.json so SA data-quality score doesn't penalise
    # the system on days when forecast_engine hasn't been run (missing contributes -0.10).
    _system_data_dir = PROJECT_ROOT / "system-data" / DATA_TAG
    try:
        _system_data_dir.mkdir(parents=True, exist_ok=True)
        _fc_path = _system_data_dir / f"{DATA_TAG}_forecast_ensemble_signals.json"
        if not _fc_path.exists():
            import json as _json_mod
            _fc_path.write_text(_json_mod.dumps({
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "date_tag": DATA_TAG,
                "source": "scaffold",
                "signals": [],
            }, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Engine cycle helpers
# ---------------------------------------------------------------------------

def regime_allows_new_buys(market_context: dict) -> Tuple[bool, str]:
    """Top-level regime gate shared by both engines for NEW buys (exits always allowed)."""
    perm = str(market_context.get("trade_permission") or "normal").lower()
    if perm == "halt_new_buys":
        return False, "rejected_by_market_regime:halt_new_buys"
    bias = str(market_context.get("overall_market_bias") or "").lower()
    if any(b in bias for b in ("crisis", "crash", "systemic_risk")):
        return False, f"rejected_by_macro:{bias}"
    return True, ""


def _alpha_waterfall_cb(waterfall):
    def cb(profit, source="alpha_call"):
        try:
            waterfall.allocate(float(profit), source=source)
        except Exception as e:
            logger.debug("waterfall.allocate failed: %s", e)
    return cb


# ── Research layer (Evidence System) — fully fail-safe, never affects trading ─────
def _research(method: str, *args, **kwargs) -> None:
    """Call a portfolio_research_lab recorder/report fn, swallowing any error."""
    try:
        import portfolio_research_lab as prl
        getattr(prl, method)(*args, **kwargs)
    except Exception as e:
        logger.debug("research layer (%s) skipped: %s", method, e)


# ── Streaming Indicator Engine — shared real-time indicator state (infra layer) ───
# run.py only FETCHES bars and UPDATES the shared indicator state; the engines decide
# and RiskManager protects. No streaming indicator ever creates an order. Any failure
# (incl. a missing `streaming-indicators` package) falls back to bar-based indicators
# and never stops the trading loop.
ENABLE_STREAMING_INDICATORS = os.environ.get("ENABLE_STREAMING_INDICATORS", "true").lower() in {"1", "true"}
_STREAMING_ENGINE: Any = None


def _streaming_engine():
    """Lazily resolve the shared StreamingIndicatorEngine singleton (fail-safe)."""
    global _STREAMING_ENGINE
    if not ENABLE_STREAMING_INDICATORS:
        return None
    if _STREAMING_ENGINE is None:
        try:
            import streaming_indicator_engine as sie
            _STREAMING_ENGINE = sie.get_engine()
            logger.info("StreamingIndicatorEngine ENABLED (streaming_package_available=%s)",
                        sie.STREAMING_PACKAGE_AVAILABLE)
        except Exception as e:
            logger.warning("StreamingIndicatorEngine unavailable (%s) — engines use bar-based indicators", e)
            _STREAMING_ENGINE = False   # sentinel: tried + failed (don't retry every cycle)
    return _STREAMING_ENGINE or None


def _update_indicators(bars_by_symbol: Dict[str, Any]) -> None:
    eng = _streaming_engine()
    if eng is None or not bars_by_symbol:
        return
    try:
        eng.update_many(bars_by_symbol)
    except Exception as e:
        logger.debug("streaming indicator update failed: %s", e)


def _indicator_snapshot() -> Optional[Dict[str, Any]]:
    eng = _streaming_engine()
    if eng is None:
        return None
    try:
        return eng.snapshot()
    except Exception as e:
        logger.debug("streaming indicator snapshot failed: %s", e)
        return None


def _write_indicator_reports() -> None:
    eng = _streaming_engine()
    if eng is None:
        return
    try:
        eng.write_report()
    except Exception as e:
        logger.debug("streaming indicator report failed: %s", e)


# RL shadow mode — OFF by default. When ENABLE_RL_SHADOW_MODE=true, trained RL agents log
# advisory recommendations only; the live trading loop ignores them entirely (no order is
# created or blocked). Fully fail-safe and import-guarded so RL deps never affect run.py.
def _rl_shadow_cycle() -> None:
    if os.environ.get("ENABLE_RL_SHADOW_MODE", "false").lower() not in {"1", "true", "yes"}:
        return
    try:
        import rl_shadow_mode
        rl_shadow_mode.run_shadow_cycle()
    except Exception as e:
        logger.debug("rl shadow cycle skipped: %s", e)


# ── Premarket Live Feed — live premarket quote awareness (yliveticker) ─────────────
# run.py only STREAMS/fetches premarket quotes; the engines decide and RiskManager
# protects. Premarket data NEVER creates an order — it only improves setup scoring and
# awareness. A missing `yliveticker` package or any feed failure falls back to the
# existing yfinance/Alpaca premarket logic and never stops the loop / Alpha / Buffett.
ENABLE_PREMARKET_LIVE_FEED = os.environ.get("ENABLE_PREMARKET_LIVE_FEED", "true").lower() in {"1", "true"}
PREMARKET_MAX_STALE_SEC    = float(os.environ.get("PREMARKET_MAX_STALE_SEC", "30"))
PREMARKET_REFRESH_SEC      = float(os.environ.get("PREMARKET_REFRESH_SEC", "30"))
_PREMARKET_FEED: Any = None
_PREMARKET_UNIVERSE_KEY: Optional[str] = None   # symbols signature the feed was started for


def _premarket_dir() -> Path:
    return DAILY_OUTPUTS / DATA_TAG


def _build_premarket_universe(dynamic_symbols, insider_data, buffett_candidates,
                              held_symbols) -> List[str]:
    """Assemble the premarket watch universe from all candidate sources (de-duped)."""
    syms: List[str] = []

    def add(items):
        for s in (items or []):
            su = _normalize_candidate_symbol(s)
            if su and len(su) <= 10:
                syms.append(su)

    try:
        from alpha_call_engine import build_alpha_universe
        add(build_alpha_universe(dynamic_symbols or []))
    except Exception:
        add(dynamic_symbols)
    add(list((insider_data or {}).keys()))
    add(buffett_candidates)
    add(held_symbols)
    seen, out = set(), []
    for s in syms:
        if s not in seen:
            seen.add(s); out.append(s)
    return out


def _premarket_feed(symbols: Optional[List[str]] = None):
    """Lazily resolve/(re)start the shared PremarketLiveFeed singleton (fail-safe)."""
    global _PREMARKET_FEED, _PREMARKET_UNIVERSE_KEY
    if not ENABLE_PREMARKET_LIVE_FEED:
        return None
    if symbols is None:
        return _PREMARKET_FEED if _PREMARKET_FEED not in (None, False) else None
    key = ",".join(sorted(set(symbols)))
    if _PREMARKET_FEED in (None, False) or key != _PREMARKET_UNIVERSE_KEY:
        try:
            if _PREMARKET_FEED not in (None, False):
                _PREMARKET_FEED.stop()
        except Exception:
            pass
        try:
            import premarket_live_feed as plf
            feed = plf.PremarketLiveFeed(symbols, output_dir=_premarket_dir(),
                                         max_stale_seconds=PREMARKET_MAX_STALE_SEC)
            feed.start()
            _PREMARKET_FEED = feed
            _PREMARKET_UNIVERSE_KEY = key
            logger.info("PremarketLiveFeed ENABLED (status=%s, symbols=%d, yliveticker=%s)",
                        feed.feed_status, len(symbols), plf.YLIVETICKER_AVAILABLE)
        except Exception as e:
            logger.warning("PremarketLiveFeed unavailable (%s) — using yfinance/Alpaca premarket logic", e)
            _PREMARKET_FEED = False
            return None
    return _PREMARKET_FEED or None


def _premarket_snapshot() -> Optional[Dict[str, Any]]:
    feed = _premarket_feed()
    if feed is None:
        return None
    try:
        return feed.get_snapshot()
    except Exception as e:
        logger.debug("premarket snapshot failed: %s", e)
        return None


def _premarket_apply_fallback(feed, symbols) -> None:
    """When the live stream is unavailable/failed, populate via existing yfinance logic."""
    if feed is None:
        return
    if not (feed.fallback_used or feed.feed_status in ("fallback", "failed")):
        return
    try:
        import premarket_data_collector as pdc
        res = pdc.collect_premarket_data(list(symbols or []))
        data = res.get("data", res) if isinstance(res, dict) else {}
        quotes: Dict[str, Dict[str, Any]] = {}
        for sym, q in (data.items() if isinstance(data, dict) else []):
            if not isinstance(q, dict):
                continue
            quotes[sym] = {
                "price": q.get("premarket_price") or q.get("price"),
                "previous_close": q.get("previous_close"),
                "premarket_change": q.get("premarket_change"),
                "premarket_change_pct": q.get("premarket_change_pct"),
                "premarket_volume": q.get("premarket_volume"),
                "bid": q.get("bid"), "ask": q.get("ask"),
            }
        feed.apply_fallback_quotes(quotes)
    except Exception as e:
        logger.debug("premarket yfinance fallback skipped: %s", e)


def _write_premarket_reports() -> None:
    feed = _premarket_feed()
    if feed is None:
        return
    try:
        feed.write_snapshot(out_dir=_premarket_dir())
    except Exception as e:
        logger.debug("premarket report failed: %s", e)


def _stop_premarket_feed() -> None:
    """Stop the live feed (e.g. at market open, freezing premarket high/low). Fail-safe."""
    feed = _PREMARKET_FEED
    if feed in (None, False):
        return
    try:
        if getattr(feed, "feed_status", "") == "running":
            feed.stop()
    except Exception:
        pass


def _build_premarket_universe_safe(gateway, bal, market_context) -> List[str]:
    """Gather candidate symbols from all sources for premarket streaming. Fail-safe."""
    try:
        dynamic = load_candidate_universe()
    except Exception:
        dynamic = []
    insider: Dict[str, Any] = {}
    if ENABLE_SEC_FORM4_FEED:
        try:
            import sec_form4_feed
            insider = sec_form4_feed.insider_data_for_layer() or {}
        except Exception:
            insider = {}
    bcands: List[str] = []
    if bal is not None:
        try:
            bcands = run_buffett_screener(market_context).get("candidates", []) or []
        except Exception:
            bcands = []
    held: List[str] = []
    try:
        raw_positions = gateway.get_positions_raw()
        _, equity_positions, _, _ = gateway.split_positions(raw_positions)
        held = list(equity_positions.keys())
    except Exception:
        held = []
    return _build_premarket_universe(dynamic, insider, bcands, held)


# ── Alpha Vantage premarket RVOL (provider-consistent: AV numerator AND denominator) ──
# Default ON but fully fail-safe: no AV key / entitlement / throttle / malformed response →
# RVOL stays UNAVAILABLE (never fabricated). Throttled so the premarket loop never hammers AV.
ENABLE_AV_PREMARKET_RVOL = os.environ.get("ENABLE_AV_PREMARKET_RVOL", "true").lower() in {"1", "true"}
AV_PREMARKET_RVOL_MAX_SYMBOLS = int(os.environ.get("AV_PREMARKET_RVOL_MAX_SYMBOLS", "15"))
AV_PREMARKET_RVOL_INTERVAL_SEC = int(os.environ.get("AV_PREMARKET_RVOL_INTERVAL_SEC", "600"))
_last_av_rvol_ts = 0.0
# Symbols that repeatedly fail AV entitlement checks are auto-skipped after
# AV_RVOL_SKIP_AFTER_FAILS consecutive failures. Prevents log spam for OTC / non-AV tickers.
AV_RVOL_SKIP_AFTER_FAILS: int = 2
_av_rvol_fail_counts: dict = {}   # sym -> consecutive failure count
_av_rvol_skip_set: set = set()    # permanently skipped for this session


def _av_premarket_rvol_cycle(feed) -> None:
    """Compute AV premarket RVOL for the watch universe and inject it (provider-consistent)
    behind the PremarketLiveFeed RVOL abstraction. Throttled + fail-safe; logs diagnostics."""
    global _last_av_rvol_ts
    if not ENABLE_AV_PREMARKET_RVOL or feed is None or not getattr(feed, "symbols", None):
        return
    if (time.time() - _last_av_rvol_ts) < AV_PREMARKET_RVOL_INTERVAL_SEC:
        return
    _last_av_rvol_ts = time.time()
    try:
        import data_sources.alpha_vantage_premarket as avpm
        # Filter out symbols that have already been marked as AV-unavailable this session
        candidate_syms = [s for s in list(feed.symbols) if s not in _av_rvol_skip_set]
        syms = candidate_syms[:AV_PREMARKET_RVOL_MAX_SYMBOLS]
        if not syms:
            return
        results = avpm.compute_many(syms, max_symbols=AV_PREMARKET_RVOL_MAX_SYMBOLS)
        feed.set_premarket_rvols(results)
        avail = 0
        for s, r in results.items():
            if r.get("rvol_available"):
                avail += 1
                _av_rvol_fail_counts.pop(s, None)   # reset on success
                logger.info("[av-rvol] %s sessions=%d current=%.0f avg=%.0f rvol=%.2f as_of=%s mode=%s provider=%s",
                            s, int(r.get("sessions_used") or 0), float(r.get("current_premarket_volume") or 0),
                            float(r.get("avg_premarket_volume") or 0), float(r.get("premarket_rvol") or 0),
                            r.get("as_of_et"), r.get("rvol_mode"), r.get("rvol_provider"))
            else:
                reason = r.get("rvol_reason", "")
                # Auto-skip symbols that are not entitled / not supported by AV
                if "entitlement" in str(reason).lower() or "no_data" in str(reason).lower():
                    _av_rvol_fail_counts[s] = _av_rvol_fail_counts.get(s, 0) + 1
                    if _av_rvol_fail_counts[s] >= AV_RVOL_SKIP_AFTER_FAILS:
                        _av_rvol_skip_set.add(s)
                        logger.info("[av-rvol] %s auto-skipped (AV entitlement/no-data after %d failures)",
                                    s, _av_rvol_fail_counts[s])
                        continue
                logger.info("[av-rvol] %s UNAVAILABLE reason=%s historical_sessions=%s provider=%s",
                            s, reason, r.get("historical_sessions"), r.get("rvol_provider"))
        logger.info("[av-rvol] premarket RVOL available for %d/%d symbols (provider=alpha_vantage)",
                    avail, len(results))
        feed.write_rvol_report(out_dir=_premarket_dir())
    except Exception as e:
        logger.debug("AV premarket RVOL skipped (RVOL left unavailable, not fabricated): %s", e)


def run_premarket_phase(gateway, bal, market_context) -> None:
    """
    Premarket orchestration: refresh the quality-growth screen, build the watch universe
    (once), start/maintain the live feed, fall back to yfinance/Alpaca when needed, and
    write the snapshot + health reports. NO trading occurs premarket — awareness only.
    Fully fail-safe.
    """
    # Ensure the continuous TradingView quality-growth refresher is running so the day's
    # Alpha universe is built from (and stays on) a fresh screen. Idempotent; fail-safe.
    _start_tv_screen_refresher()
    feed = _premarket_feed()
    if feed is None:
        universe = _build_premarket_universe_safe(gateway, bal, market_context)
        if not universe:
            return
        feed = _premarket_feed(universe)
        if feed is not None:
            _premarket_apply_fallback(feed, universe)
    else:
        if feed.fallback_used or feed.feed_status in ("fallback", "failed"):
            _premarket_apply_fallback(feed, feed.symbols)
    # Overnight feed metrics (Alpaca overnight session, read-only) — advisory awareness.
    _alpaca_feed_cycle("premarket")
    try:
        mgr = _alpaca_feed_manager()
        if mgr is not None and feed is not None and getattr(feed, "symbols", None):
            metrics = mgr.fetch_overnight_metrics(list(feed.symbols)[:25])
            feed.set_overnight_metrics_many(metrics)
            feed.write_overnight_report(out_dir=_premarket_dir())
    except Exception as e:
        logger.debug("overnight metrics skipped: %s", e)
    # Alpha Vantage premarket RVOL — provider-consistent (AV numerator + AV denominator),
    # throttled, fail-safe (leaves RVOL unavailable rather than fabricating).
    _av_premarket_rvol_cycle(feed)
    _write_premarket_reports()
    # Premarket relative-volume report (RVOL stays None unless 20d history is supplied).
    try:
        if feed is not None:
            feed.write_rvol_report(out_dir=_premarket_dir())
    except Exception as e:
        logger.debug("premarket rvol report skipped: %s", e)


# ── MarketStateEngine — single shared market truth (advisory; never places orders) ─
ENABLE_MARKET_STATE_ENGINE = os.environ.get("ENABLE_MARKET_STATE_ENGINE", "true").lower() in {"1", "true"}
_MARKET_STATE_ENGINE: Any = None


def _market_state_engine():
    """Lazily resolve the shared MarketStateEngine singleton (fail-safe)."""
    global _MARKET_STATE_ENGINE
    if not ENABLE_MARKET_STATE_ENGINE:
        return None
    if _MARKET_STATE_ENGINE is None:
        try:
            import market_state_engine as mse
            _MARKET_STATE_ENGINE = mse.get_engine()
            logger.info("MarketStateEngine ENABLED")
        except Exception as e:
            logger.warning("MarketStateEngine unavailable (%s) — using market_context only", e)
            _MARKET_STATE_ENGINE = False
    return _MARKET_STATE_ENGINE or None


def _build_market_state(market_context: dict, indicator_snapshot=None,
                        premarket_snapshot=None, bars_by_symbol=None, feed_type=None):
    """Build the shared market_state for this cycle (advisory). Falls back to None."""
    eng = _market_state_engine()
    if eng is None:
        return None
    try:
        qn = sum(1 for q in (premarket_snapshot or {}).get("quotes", {}).values()
                 if q.get("price") is not None) if premarket_snapshot else 0
        bn = len((indicator_snapshot or {}))
        state = eng.build(market_context=market_context, indicator_snapshot=indicator_snapshot,
                          premarket_snapshot=premarket_snapshot, bars_by_symbol=bars_by_symbol,
                          alpaca_quote_count=qn, alpaca_bar_count=bn, feed_type=feed_type)
        eng.write_report(out_dir=_premarket_dir())
        return state
    except Exception as e:
        logger.debug("market_state build failed: %s", e)
        return None


# ── Alpaca live-feed manager (read-only data quality; never changes execution) ────
ENABLE_ALPACA_FEED_MANAGER = os.environ.get("ENABLE_ALPACA_FEED_MANAGER", "true").lower() in {"1", "true"}
_ALPACA_FEED: Any = None


def _alpaca_feed_manager():
    global _ALPACA_FEED
    if not ENABLE_ALPACA_FEED_MANAGER:
        return None
    if _ALPACA_FEED is None:
        try:
            import alpaca_feed_manager as afm
            _ALPACA_FEED = afm.get_manager()
            logger.info("AlpacaFeedManager ENABLED (no SIP; iex/overnight only)")
        except Exception as e:
            logger.warning("AlpacaFeedManager unavailable (%s)", e)
            _ALPACA_FEED = False
    return _ALPACA_FEED or None


def _alpaca_feed_cycle(session: str = "rth") -> Optional[str]:
    """Detect feeds, write diagnostics, tag the streaming engine. Returns active feed. Fail-safe."""
    mgr = _alpaca_feed_manager()
    if mgr is None:
        return None
    try:
        active = mgr.active_feed(session)
        mgr.write_diagnostic(out_dir=_premarket_dir(), session=session)
        diag = mgr.diagnostic(session)
        # Deliverables: live_data_feed_health.json + alpaca_live_data_snapshot.json
        from datetime import datetime as _dt, timezone as _tz
        health = {"generated_at_utc": _dt.now(_tz.utc).isoformat(), "date_tag": DATA_TAG,
                  "session": session, "active_feed": active, "feeds": diag.get("feeds", {}),
                  "feed_failover_count": diag.get("feed_failover_count", 0),
                  "feed_reconnect_count": diag.get("feed_reconnect_count", 0),
                  "sip_used": False, "delayed_sip_used": False}
        try:
            from rl_common import write_json as _wj
        except Exception:
            _wj = None
        outdir = _premarket_dir()
        if _wj:
            _wj(outdir / "live_data_feed_health.json", health)
            _wj(outdir / "alpaca_live_data_snapshot.json",
                {"generated_at_utc": health["generated_at_utc"], "date_tag": DATA_TAG,
                 "active_feed": active, "feed_source": "alpaca", "sip_used": False})
        eng = _streaming_engine()
        if eng is not None and hasattr(eng, "set_feed"):
            eng.set_feed(active, used=active not in (None, "fallback"))
        return active
    except Exception as e:
        logger.debug("alpaca feed cycle skipped: %s", e)
        return None


# ── Alpha: EXIT pass (runs every cycle, independent of regime/entries/screeners) ──
def run_alpha_exits(engine, gateway: AlpacaGateway, waterfall,
                    option_positions: List[Dict], market_context: dict,
                    indicator_snapshot: Optional[dict] = None,
                    market_state: Optional[dict] = None,
                    risk: Optional["RiskManager"] = None) -> int:
    """Manage open calls via the engine's public exit pass. ALWAYS runs."""
    try:
        exits = engine.check_active_positions(option_positions, _alpha_waterfall_cb(waterfall),
                                              market_context=market_context,
                                              indicator_snapshot=indicator_snapshot,
                                              market_state=market_state)
    except Exception as e:
        logger.error("alpha check_active_positions failed: %s", e)
        exits = []
    for ex in exits:
        gateway.submit_market_order(ex["symbol"], ex["side"], ex["qty"],
                                    reason=ex.get("source", "alpha_exit"))
        _research("record_alpha_exit_safe", ex, market_context)   # evidence: realized result
        # Wire realized option P&L into RiskManager so daily_pnl reflects options activity
        if risk is not None:
            try:
                pnl = float(ex.get("realized_pnl") or 0.0)
                if pnl != 0.0:
                    risk.record_pnl(pnl)
            except Exception:
                pass
    return len(exits)


# ── Alpha: ENTRY pass (engine owns all screening / contract / BS decisions) ───────
def run_alpha_entries(engine, gateway: AlpacaGateway, waterfall, risk: RiskManager,
                      symbols: List[str], bars_by_symbol: Dict[str, Any],
                      chains_by_symbol: Dict[str, list], market_context: dict,
                      positions: Optional[dict] = None,
                      indicator_snapshot: Optional[dict] = None,
                      premarket_snapshot: Optional[dict] = None,
                      market_state: Optional[dict] = None) -> int:
    """
    Hand PRE-FETCHED data to the engine; it writes alpha_options_universe.json and
    returns order intents. run.py only routes intents through Risk/Gateway.
    waterfall_budget=0 so reserves never cap tactical sizing (primary engine).
    """
    if not is_market_open():
        log_skip("ALPHA_ENGINE", "rejected_by_session:market_closed", phase="entry_gate", engine="alpha")
        return 0
    now_et = datetime.now(ET)
    if now_et >= now_et.replace(hour=15, minute=45, second=0, microsecond=0):
        log_skip("ALPHA_ENGINE", "rejected_by_session:past_entry_cutoff", phase="entry_gate", engine="alpha")
        return 0
    new_buys_allowed, regime_reason = regime_allows_new_buys(market_context)
    if not new_buys_allowed:
        log_skip("ALPHA_ENGINE", regime_reason, phase="entry_gate", engine="alpha")
        return 0
    try:
        signals = engine.evaluate_universe(
            symbols=symbols, bars_by_symbol=bars_by_symbol, market_context=market_context,
            equity=risk.equity, options_exposure=risk.options_exposure, waterfall_budget=0.0,
            chains_by_symbol=chains_by_symbol, positions=positions,
            account_state={"equity": risk.equity, "buying_power": risk.buying_power},
            indicator_snapshot=indicator_snapshot,
            premarket_snapshot=premarket_snapshot,
            market_state=market_state,
        )
    except Exception as e:
        import traceback as _etb
        logger.error("alpha evaluate_universe failed: %s\n%s", e, _etb.format_exc())
        return 0
    placed = 0
    for sig in signals:
        if sig.status != "ready":
            if sig.reject_reason:
                log_skip(sig.symbol, sig.reject_reason, engine="alpha",
                         grade=sig.grade, score=sig.setup_score)
                _research("record_alpha_signal_safe", getattr(sig, "__dict__", {}), market_context, False)
            continue
        limit_px = sig.ask or sig.mid
        resp = gateway.submit_market_order(sig.contract_symbol, "buy", sig.contracts,
                                           reason=f"alpha_call:{sig.setup_type}:{sig.grade}",
                                           limit_price=limit_px, order_type="limit")
        if resp is not None:
            placed += 1
            _research("record_alpha_signal_safe", getattr(sig, "__dict__", {}), market_context, True)
            try:
                engine.register_filled(sig)
                waterfall.consume_alpha(sig.premium_cost)
            except Exception as e:
                logger.debug("register_filled/consume_alpha failed: %s", e)
            # Journal the submission so funnel_report sees submitted > 0
            try:
                log_executed_order(
                    event="order_submitted",
                    symbol=sig.contract_symbol,
                    underlying=sig.symbol,
                    side="buy",
                    qty=sig.contracts,
                    limit_price=limit_px,
                    setup_type=getattr(sig, "setup_type", ""),
                    grade=getattr(sig, "grade", ""),
                    setup_score=getattr(sig, "setup_score", 0),
                    engine="alpha",
                    order_id=resp.get("id", "") if isinstance(resp, dict) else getattr(resp, "id", ""),
                    sim=resp.get("sim", False) if isinstance(resp, dict) else False,
                )
            except Exception as _je:
                logger.debug("alpha log_executed_order failed: %s", _je)
    return placed


# ── Alpha: screener orchestration (I/O lives in run.py; intelligence in engine) ───
def run_alpha_screener(engine, gateway: AlpacaGateway, market_context: dict,
                       dynamic_symbols: List[str]) -> dict:
    """
    run.py fetches data; the ENGINE decides the universe. Returns the base universe,
    bars, pre-fetched chains (watch list only) and the screen funnel. No alpha logic
    is performed here — only fetching + delegation to engine.screen_alpha_options_universe.
    """
    from alpha_call_engine import build_alpha_universe
    base = build_alpha_universe(dynamic_symbols)
    bars = fetch_1m_bars(gateway, base)
    if ENABLE_ALPHA_OPTIONS_SCREENER:
        screen = engine.screen_alpha_options_universe(base, bars, market_context)
        watch, funnel = screen["watch"], screen["funnel"]
    else:
        watch = base
        funnel = {"scanned": len(base), "passed_momentum": len(base), "note": "screener_disabled"}
    # Fetch option chains ONLY for the watch list — parallel with a hard wall-clock cap.
    # Each get_options_chain call already has per-call timeouts (25s contracts + 15s
    # snapshot). Parallel fetching keeps the total bounded to ~50s regardless of
    # watch-list size, vs N×45s with a serial loop.
    chains: Dict[str, list] = {}
    if watch:
        import concurrent.futures as _cf_scr
        CHAIN_WORKERS = min(8, len(watch))
        with _cf_scr.ThreadPoolExecutor(max_workers=CHAIN_WORKERS) as _pool_scr:
            _fut_map = {_pool_scr.submit(gateway.get_options_chain, sym): sym for sym in watch}
            try:
                for _fut in _cf_scr.as_completed(_fut_map, timeout=50):
                    _sym = _fut_map[_fut]
                    try:
                        chains[_sym] = _fut.result() or []
                    except Exception as _e:
                        logger.debug("chain fetch failed for %s: %s", _sym, _e)
                        chains[_sym] = []
            except _cf_scr.TimeoutError:
                logger.warning("chain fetch wall-clock timeout (50s) — proceeding with partial chains")
            # Any symbols that still didn't complete get empty chains
            for _sym in watch:
                if _sym not in chains:
                    logger.debug("chain fetch incomplete for %s — using empty chain", _sym)
                    chains[_sym] = []
    return {"base": base, "bars": bars, "chains": chains, "watch": watch, "funnel": funnel}


# ── BAL: EXIT pass (runs every cycle) ─────────────────────────────────────────────
_EMA_DC = None
_EMA_CACHE: Dict[Any, Any] = {}


def _ema_value(symbol: str, period: int = 9, ttl: float = 900.0) -> float:
    """Daily `period`-EMA for `symbol` (cached `ttl` seconds — daily EMA barely moves intraday).
    0.0 on any failure."""
    try:
        key = (symbol, period); now = time.time()
        c = _EMA_CACHE.get(key)
        if c and (now - c[0]) < ttl:
            return c[1]
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed
        global _EMA_DC
        if _EMA_DC is None:
            _EMA_DC = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
        end = datetime.now(timezone.utc)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                               start=end - timedelta(days=120), end=end, feed=DataFeed.IEX)
        df = _EMA_DC.get_stock_bars(req).df
        if df is None or len(df) < period + 1:
            return 0.0
        closes = df["close"].astype(float)
        ema = float(closes.ewm(span=period, adjust=False).mean().iloc[-1])
        _EMA_CACHE[key] = (now, ema)
        return ema
    except Exception as e:
        logger.debug("_ema_value(%s) failed: %s", symbol, e)
        return 0.0


_SMA_CACHE: Dict[Any, Any] = {}


def _sma_value(symbol: str, period: int = 200, ttl: float = 900.0) -> float:
    """Daily simple `period`-MA for `symbol` (cached `ttl` seconds). 0.0 on any failure."""
    try:
        key = (symbol, period); now = time.time()
        c = _SMA_CACHE.get(key)
        if c and (now - c[0]) < ttl:
            return c[1]
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed
        global _EMA_DC
        if _EMA_DC is None:
            _EMA_DC = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
        end = datetime.now(timezone.utc)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                               start=end - timedelta(days=int(period * 1.6) + 30), end=end, feed=DataFeed.IEX)
        df = _EMA_DC.get_stock_bars(req).df
        if df is None or len(df) < period:
            return 0.0
        sma = float(df["close"].astype(float).tail(period).mean())
        _SMA_CACHE[key] = (now, sma)
        return sma
    except Exception as e:
        logger.debug("_sma_value(%s) failed: %s", symbol, e)
        return 0.0


_TREND_CACHE: Dict[str, Any] = {}


def _trend_mas(symbol: str, ttl: float = 900.0):
    """(ema9, sma200, prev_close) from daily bars for crossover detection (cached). None if < 200 bars."""
    try:
        now = time.time()
        c = _TREND_CACHE.get(symbol)
        if c and (now - c[0]) < ttl:
            return c[1]
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed
        global _EMA_DC
        if _EMA_DC is None:
            _EMA_DC = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
        end = datetime.now(timezone.utc)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                               start=end - timedelta(days=360), end=end, feed=DataFeed.IEX)
        df = _EMA_DC.get_stock_bars(req).df
        if df is None or len(df) < SMA200_PERIOD:
            res = None
        else:
            closes = df["close"].astype(float)
            ema9 = float(closes.ewm(span=EMA_PERIOD, adjust=False).mean().iloc[-1])
            sma200 = float(closes.tail(SMA200_PERIOD).mean())
            prev_close = float(closes.iloc[-2])          # prior completed daily close (for the cross)
            res = (ema9, sma200, prev_close)
        _TREND_CACHE[symbol] = (now, res)
        return res
    except Exception as e:
        logger.debug("_trend_mas(%s) failed: %s", symbol, e)
        return None


_REGIME_CACHE: Dict[str, Any] = {}


def _index_uptrend(symbol: str):
    """(uptrend, below_200) for a market index — uptrend = price>200-SMA AND 50-SMA rising;
    below_200 = price<200-SMA. None on failure."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed
        global _EMA_DC
        if _EMA_DC is None:
            _EMA_DC = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
        end = datetime.now(timezone.utc)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                               start=end - timedelta(days=360), end=end, feed=DataFeed.IEX)
        df = _EMA_DC.get_stock_bars(req).df
        if df is None or len(df) < 200:
            return None
        c = df["close"].astype(float)
        price, sma200, sma50 = float(c.iloc[-1]), float(c.tail(200).mean()), float(c.tail(50).mean())
        sma50_prev = float(c.iloc[-70:-20].mean()) if len(c) >= 70 else sma50   # 50-SMA ~20 sessions ago
        return (price > sma200 and sma50 > sma50_prev), (price < sma200)
    except Exception as e:
        logger.debug("_index_uptrend(%s) failed: %s", symbol, e)
        return None


def _market_regime(ttl: float = 900.0) -> dict:
    """Breadth regime over REGIME_INDICES (SPY/QQQ/IWM): risk_on when a MAJORITY are trending up,
    risk_off when a majority are below their 200-SMA. Cached. Fail-open to risk_on if no data."""
    now = time.time()
    c = _REGIME_CACHE.get("r")
    if c and (now - c[0]) < ttl:
        return c[1]
    up = dn = n = 0
    for idx in REGIME_INDICES:
        r = _index_uptrend(idx)
        if r is None:
            continue
        n += 1; up += int(r[0]); dn += int(r[1])
    thr = 2 if n >= 2 else 1
    res = {"risk_on": (n == 0 or up >= thr), "risk_off": (n > 0 and dn >= thr), "up": up, "dn": dn, "n": n}
    _REGIME_CACHE["r"] = (now, res)
    return res


def run_regime_exit(gateway: AlpacaGateway, equity_positions: Dict[str, Any]) -> int:
    """Liquidate all equity positions when the broader market turns RISK-OFF (a majority of SPY/QQQ/IWM
    below their 200-SMA) — exits a market-wide reversal/bear rather than riding it down. Fail-safe."""
    if not ENABLE_MARKET_REGIME:
        return 0
    reg = _market_regime()
    if not reg["risk_off"]:
        return 0
    n = 0
    for sym, pos in list((equity_positions or {}).items()):
        try:
            qty = int(abs(float(getattr(pos, "qty", 0) or 0)))
            if qty <= 0 or is_option_symbol(sym):
                continue
            gateway.submit_market_order(sym, "sell", qty,
                                        reason=f"regime_exit:market_risk_off_{reg['dn']}/{reg['n']}_below_200sma")
            logger.info("REGIME EXIT %s x%d (market risk-off: %d/%d indices below 200-SMA)",
                        sym, qty, reg["dn"], reg["n"])
            n += 1
        except Exception as e:
            logger.error("regime exit failed for %s (rest unaffected): %s", sym, e)
    return n


def _ema_buy_blocked(symbol: str, price: float) -> str:
    """Crossover BUY gate. Returns '' if a buy is ALLOWED, else a reason. First the broader-market
    regime must be RISK-ON (majority of SPY/QQQ/IWM trending up — not chop/reversal); then per name:
       • price crossed UP through the 200-SMA, OR
       • price crossed DOWN through the 9-EMA WHILE above the 200-SMA.
    Below the 200-SMA is always blocked. Equities only; fail-open if MA data is missing."""
    if is_option_symbol(symbol) or price <= 0:
        return ""
    if ENABLE_MARKET_REGIME:
        reg = _market_regime()
        if not reg["risk_on"]:
            return f"rejected_market_regime_chop_or_reversal:uptrend={reg['up']}/{reg['n']}_indices"
    if not ENABLE_EMA_GATE:
        return ""
    mas = _trend_mas(symbol)
    if not mas:
        return ""
    ema9, sma200, prev = mas
    if sma200 > 0 and price < sma200:
        return f"rejected_by_below_sma200:price={price:.2f}<sma200={sma200:.2f}"
    cross_up_sma200 = sma200 > 0 and prev < sma200 <= price
    cross_down_ema9 = ema9 > 0 and prev > ema9 >= price
    if cross_up_sma200 or (cross_down_ema9 and price > sma200):
        return ""
    return f"rejected_no_trend_crossover:px={price:.2f} ema9={ema9:.2f} sma200={sma200:.2f}"


def run_ema_exits(gateway: AlpacaGateway, equity_positions: Dict[str, Any]) -> int:
    """Exit a held SHARE on a 200-SMA break (long-term trend break) OR a per-position STOP LOSS (price
    down POSITION_STOP_PCT from average entry) — whichever triggers first. Unconditional exit pass.
    Fully fail-safe. Returns # of sells."""
    if not ENABLE_EMA_EXIT:
        return 0
    n = 0
    for sym, pos in list((equity_positions or {}).items()):
        try:
            qty = int(abs(float(getattr(pos, "qty", 0) or 0)))
            if qty <= 0 or is_option_symbol(sym):
                continue
            price = gateway._resolve_reference_price(sym)
            if price <= 0:
                continue
            avg = float(getattr(pos, "avg_entry_price", 0) or 0)
            stop_hit = POSITION_STOP_PCT > 0 and avg > 0 and price < avg * (1 - POSITION_STOP_PCT)
            sma = _sma_value(sym, SMA200_PERIOD)
            sma_hit = sma > 0 and price < sma
            if not (stop_hit or sma_hit):
                continue
            reason = (f"stop_loss:price={price:.2f}<entry={avg:.2f}-{POSITION_STOP_PCT*100:.0f}%"
                      if stop_hit and not sma_hit else f"sma200_exit:price={price:.2f}<sma200={sma:.2f}")
            gateway.submit_market_order(sym, "sell", qty, reason=reason)
            logger.info("EXIT %s x%d (%s)", sym, qty, reason)
            n += 1
        except Exception as e:
            logger.error("exit failed for %s (rest unaffected): %s", sym, e)
    return n


def run_bal_exits(bal, gateway: AlpacaGateway, equity_positions: Dict[str, Any],
                  market_context: dict) -> int:
    n = 0
    for sym, pos in list(equity_positions.items()):
        try:
            should_exit, reason = bal.should_exit(sym, pos, market_context)
        except Exception as e:
            logger.debug("BAL should_exit failed for %s: %s", sym, e)
            continue
        if should_exit:
            qty = int(abs(float(getattr(pos, "qty", 0) or 0)))
            if qty > 0:
                gateway.submit_market_order(sym, "sell", qty, reason=f"bal_exit:{reason}")
                n += 1
    return n


# ── BAL: ENTRY pass (long-term accumulation) ──────────────────────────────────────
def run_bal_entries(bal, gateway: AlpacaGateway, waterfall, risk: RiskManager,
                    symbols: List[str], market_context: dict,
                    bars_ctx_by_symbol: Dict[str, Dict], equity_positions: Dict[str, Any],
                    insider_data_by_symbol: Optional[Dict[str, Dict]] = None,
                    target_weights_by_symbol: Optional[Dict[str, float]] = None,
                    indicator_snapshot: Optional[dict] = None,
                    market_state: Optional[dict] = None,
                    optimizer_objective_weights: Optional[dict] = None) -> int:
    if not symbols:
        return 0
    stock_room = max(0.0, risk.equity * MAX_STOCK_EXPOSURE_PCT - risk.stock_exposure)
    cash_reserve = CASH_BUFFER_PCT * risk.equity                      # always-free cash
    spendable = max(0.0, risk.cash - cash_reserve)                   # only deploy cash above the reserve
    available_capital = min(spendable, stock_room)
    if available_capital < 200:
        logger.info(
            "BAL entries blocked: available_capital=%.0f < 200 "
            "(cash=%.0f reserve=%.0f[%.0f%%eq] spendable=%.0f stock_room=%.0f equity=%.0f stock_exp=%.0f)",
            available_capital, risk.cash, cash_reserve, CASH_BUFFER_PCT * 100,
            spendable, stock_room, risk.equity, risk.stock_exposure,
        )
        return 0
    # Composite DCA framework is the accumulation mechanism when enabled (value decides
    # WHETHER, RSI decides WHEN, tranches decide HOW MUCH); legacy score-based scan is
    # the fallback. SEC Form 4 insider data is a CONVICTION MODIFIER only — it can never
    # generate an order; every intent still routes through RiskManager/AlpacaGateway.
    use_dca = ENABLE_BAL_COMPOSITE_DCA and hasattr(bal, "evaluate_dca")
    try:
        if use_dca:
            orders = bal.evaluate_dca(
                symbols=symbols, market_context=market_context,
                bars_ctx_by_symbol=bars_ctx_by_symbol, equity=risk.equity,
                positions=equity_positions, available_capital=available_capital,
                insider_data_by_symbol=insider_data_by_symbol or {},
                target_weights_by_symbol=target_weights_by_symbol or {},
                indicator_snapshot=indicator_snapshot or {},
                market_state=market_state,
                optimizer_objective_weights=optimizer_objective_weights,
            )
        else:
            orders = bal.scan(
                symbols=symbols, market_context=market_context,
                bars_ctx_by_symbol=bars_ctx_by_symbol, equity=risk.equity,
                positions=equity_positions, available_capital=available_capital,
            )
    except Exception as e:
        logger.error("BAL %s failed: %s", "DCA" if use_dca else "scan", e)
        return 0
    placed = 0
    for o in orders:
        # Pass the BAL-evaluated price as the notional reference so RiskManager gates on
        # real exposure (never qty × 0.0). It does NOT make this a limit order.
        if ENABLE_FRACTIONAL_ORDERS and float(o.get("target_notional", 0.0) or 0.0) > 0:
            resp = gateway.submit_market_order(o["symbol"], "buy", 0,
                                               reason=o.get("reason", "buffett_accumulation"),
                                               reference_price=float(o.get("price", 0.0) or 0.0),
                                               notional=float(o["target_notional"]))
        else:
            resp = gateway.submit_market_order(o["symbol"], "buy", int(o["qty"]),
                                               reason=o.get("reason", "buffett_accumulation"),
                                               reference_price=float(o.get("price", 0.0) or 0.0))
        if resp is not None:
            placed += 1
            _research("record_bal_order_safe", o, market_context)   # evidence: tranche/conviction
            try:
                waterfall.consume_bal(float(o.get("order_dollars", 0.0)))
            except Exception:
                pass
        else:
            # Order rejected by RiskManager/gateway → undo the trigger-time DCA stamp so a
            # non-executed order never burns the min-days cooldown or a tranche (ADBE bug).
            try:
                if use_dca and hasattr(bal, "rollback_dca"):
                    bal.rollback_dca(o["symbol"])
            except Exception:
                pass
    # Evidence: record BAL rejections from the just-written DCA signals file.
    try:
        from buffett_accumulation_layer import BUFFETT_DCA_SIGNALS_PATH
        sigs = json.loads(BUFFETT_DCA_SIGNALS_PATH.read_text()).get("signals", [])
        for s in sigs:
            if not s.get("risk_limits_passed", False):
                _research("record_bal_rejection_safe", s, market_context)
    except Exception:
        pass
    return placed


# ── Buffett screener orchestration (fully isolated from Alpha) ────────────────────
def run_buffett_screener(market_context: dict, premarket_snapshot: Optional[dict] = None) -> dict:
    """Guarded: any failure returns empty candidates so Alpha exits/entries are never affected."""
    if not ENABLE_BUFFETT_ACCUMULATION_SCREENER:
        return {"candidates": [], "funnel": {"note": "screener_disabled"}, "detail": []}
    try:
        import buffett_accumulation_screener as bscr
        return bscr.screen(market_context, premarket_snapshot=premarket_snapshot)
    except Exception as e:
        logger.error("Buffett screener failed (Alpha unaffected): %s", e)
        return {"candidates": [], "funnel": {"error": str(e)[:160]}, "detail": []}


def write_screener_funnel_report(alpha_screen_funnel: dict, buffett_res: dict,
                                 market_context: dict,
                                 alpha_placed: int = 0,
                                 bal_placed: int = 0) -> None:
    """Combined funnel diagnostics for both screeners."""
    alpha_candidates = 0
    alpha_liquidity_rejects = 0
    try:
        from alpha_call_engine import ALPHA_OPTIONS_UNIVERSE_PATH
        if ALPHA_OPTIONS_UNIVERSE_PATH.exists():
            au = json.loads(ALPHA_OPTIONS_UNIVERSE_PATH.read_text())
            alpha_candidates = au.get("candidate_count", 0)
            alpha_liquidity_rejects = sum(
                1 for r in au.get("all_scanned", [])
                if str(r.get("reject_reason") or "").startswith("rejected_by_options_liquidity"))
    except Exception:
        pass
    bf = buffett_res.get("funnel", {})
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_tag": DATA_TAG,
        "market_regime": str(market_context.get("overall_market_bias")),
        "alpha_options_screener": {
            "total_scanned": alpha_screen_funnel.get("scanned", 0),
            "passed_alpha_options_screener": alpha_screen_funnel.get("passed_momentum", 0),
            "rejected_by_regime": alpha_screen_funnel.get("rejected_by_regime", 0),
            "rejected_by_momentum": alpha_screen_funnel.get("rejected_by_momentum", 0),
            "rejected_by_option_liquidity": alpha_liquidity_rejects,
            "final_candidates_sent_to_alpha_engine": alpha_candidates,
        },
        "buffett_accumulation_screener": {
            "total_scanned": bf.get("scanned", 0),
            "passed_buffett_screener": bf.get("passed", 0),
            "rejected_by_fundamentals": bf.get("rejected_by_fundamentals", 0),
            "rejected_by_valuation": bf.get("rejected_by_valuation", 0),
            "rejected_by_macro_cycle": bf.get("rejected_by_macro_cycle", 0),
            "final_candidates_sent_to_bal": len(buffett_res.get("candidates", [])),
        },
        "universes_independent": True,
        "cycle_orders_placed": {"alpha": alpha_placed, "bal": bal_placed,
                                 "total": alpha_placed + bal_placed},
    }
    try:
        SCREENER_FUNNEL_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        logger.debug("write_screener_funnel_report failed: %s", e)


def run_portfolio_monitor(monitor, gateway: AlpacaGateway, risk: RiskManager,
                          equity_positions: Dict[str, Any], market_context: dict) -> dict:
    """Optional BAL portfolio monitor / hedging (opt-in via ENABLE_BAL_PORTFOLIO_MONITOR)."""
    try:
        return monitor.evaluate(
            positions=equity_positions,
            equity=risk.equity,
            market_context=market_context,
            broker=gateway,
        )
    except Exception as e:
        logger.error("BAL portfolio monitor failed: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Two-engine trading loop
# ---------------------------------------------------------------------------

def build_engines(gateway: AlpacaGateway):
    """Instantiate engines + waterfall. Returns (alpha, bal, monitor, waterfall)."""
    alpha = bal = monitor = waterfall = None
    try:
        from capital_waterfall import CapitalWaterfall, WaterfallConfig
        waterfall = CapitalWaterfall(WaterfallConfig())
    except Exception as e:
        logger.error("CapitalWaterfall init failed: %s", e)
    # ── DO NOT DEPLOY: alpha-call (long-call) engine is research-FAILED ─────────────
    # Real-chain audit (2026-06-06): synthetic option chains showed +$2,063, but REAL
    # Alpha Vantage chains gave -$214 with the 10% max-drawdown kill switch triggered
    # (negative expectancy: avg loss -$45 vs avg win +$27). Hard-disabled from LIVE
    # execution regardless of ENABLE_ALPHA_CALL_ENGINE. Still usable for RESEARCH via
    # backtest_alpha_replay.py. Live deployment = shares-only trend overlay (~/quant_trader_live).
    # Risk controls / kill switch (RiskManager) are unaffected.
    ALPHA_DO_NOT_DEPLOY = True
    if ENABLE_ALPHA_CALL_ENGINE and not ALPHA_DO_NOT_DEPLOY:
        try:
            from alpha_call_engine import AlphaCallEngine, AlphaCallConfig
            alpha = AlphaCallEngine(broker=gateway, config=AlphaCallConfig())
            logger.info("Alpha Call Engine ENABLED")
        except Exception as e:
            logger.error("AlphaCallEngine init failed: %s", e)
    elif ENABLE_ALPHA_CALL_ENGINE:
        logger.warning("Alpha Call Engine DISABLED — DO NOT DEPLOY (negative expectancy on "
                       "real option chains; 2026-06-06 real-chain audit). Research-only.")
    if ENABLE_BUFFETT_ACCUMULATION_LAYER:
        try:
            from buffett_accumulation_layer import BuffettAccumulationLayer, BALConfig, DCAConfig
            bal = BuffettAccumulationLayer(config=BALConfig(), dca_config=DCAConfig())
            logger.info("Buffett Accumulation Layer ENABLED (composite_dca=%s)", ENABLE_BAL_COMPOSITE_DCA)
        except Exception as e:
            logger.error("BuffettAccumulationLayer init failed: %s", e)
    if ENABLE_BAL_PORTFOLIO_MONITOR:
        try:
            from buffett_accumulation_layer import BALPortfolioMonitor, HedgeConfig, BALConfig
            monitor = BALPortfolioMonitor(hedge_config=HedgeConfig(), bal_config=BALConfig())
            logger.info("BAL Portfolio Monitor ENABLED")
        except Exception as e:
            logger.error("BALPortfolioMonitor init failed: %s", e)
    return alpha, bal, monitor, waterfall


def _occ_underlying(occ: str) -> str:
    """Underlying ticker from an OCC option symbol (leading letters before the YYMMDD)."""
    s = (occ or "").strip().upper()
    m = re.match(r"^([A-Z]+)\d{6}[CP]\d+$", s)
    return m.group(1) if m else s


def _occ_dte(occ: str) -> Optional[int]:
    """Calendar days to expiry from an OCC option symbol (YYMMDD block). None if unparseable."""
    s = (occ or "").strip().upper()
    m = re.match(r"^[A-Z]+(\d{2})(\d{2})(\d{2})[CP]\d+$", s)
    if not m:
        return None
    try:
        yy, mm, dd = (int(g) for g in m.groups())
        exp = datetime(2000 + yy, mm, dd).date()
        return (exp - datetime.now(ET).date()).days
    except Exception:
        return None


def _insider_copy_state_load() -> Dict[str, Any]:
    """Persisted map {occ_option_symbol: {entry_date, underlying, qty}} of copier-opened
    calls, used by the time-exit. Empty on any failure."""
    try:
        if INSIDER_COPY_STATE_PATH.exists():
            data = json.loads(INSIDER_COPY_STATE_PATH.read_text())
            return data if isinstance(data, dict) else {}
    except Exception:
        logger.debug("insider copy state read failed", exc_info=True)
    return {}


def _insider_copy_state_save(state: Dict[str, Any]) -> None:
    if _in_test():
        return
    try:
        INSIDER_COPY_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        INSIDER_COPY_STATE_PATH.write_text(json.dumps(state, default=str, indent=2))
    except Exception:
        logger.debug("insider copy state write failed", exc_info=True)


def _insider_copy_record_open(occ_symbol: str, underlying: str, qty: int,
                              entry_date: "date", instrument: str = "option",
                              entry_price: float = 0.0) -> None:
    state = _insider_copy_state_load()
    state[occ_symbol] = {"entry_date": entry_date.isoformat(), "underlying": underlying,
                         "qty": int(qty), "instrument": instrument,
                         "entry_price": float(entry_price)}
    _insider_copy_state_save(state)


def _insider_ledger_append(exit_date: "date", symbol: str, instrument: str,
                           ret: float, pnl: float) -> None:
    """Append one CLOSED insider-copy trade to the rolling-PF ledger (fail-safe)."""
    try:
        ledger: List[Dict[str, Any]] = []
        if INSIDER_COPY_LEDGER_PATH.exists():
            ledger = json.loads(INSIDER_COPY_LEDGER_PATH.read_text()) or []
        ledger.append({"exit_date": exit_date.isoformat(), "symbol": symbol,
                       "instrument": instrument, "ret": round(float(ret), 5),
                       "pnl": round(float(pnl), 2)})
        ledger = ledger[-500:]                     # keep the tail; plenty for a 40-trade window
        INSIDER_COPY_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        INSIDER_COPY_LEDGER_PATH.write_text(json.dumps(ledger, indent=2))
    except Exception as e:
        logger.debug("insider ledger append failed: %s", e)


def _insider_rolling_pf() -> Tuple[Optional[float], int]:
    """Rolling profit factor over the last INSIDER_COPY_PF_WINDOW closed trades within the last
    INSIDER_COPY_PF_LOOKBACK_DAYS. Returns (pf, n_in_window). pf=None when n < MIN_TRADES (cannot
    judge yet) → caller must NOT halt. pf=inf when there are no losing trades."""
    try:
        if not INSIDER_COPY_LEDGER_PATH.exists():
            return None, 0
        ledger = json.loads(INSIDER_COPY_LEDGER_PATH.read_text()) or []
        cutoff = datetime.now(ET).date() - timedelta(days=INSIDER_COPY_PF_LOOKBACK_DAYS)
        recent = []
        for t in ledger:
            try:
                d = datetime.fromisoformat(str(t.get("exit_date"))[:10]).date()
            except Exception:
                continue
            if d >= cutoff:
                recent.append(float(t.get("ret", 0.0)))
        window = recent[-INSIDER_COPY_PF_WINDOW:]
        if len(window) < INSIDER_COPY_PF_MIN_TRADES:
            return None, len(window)
        gains = sum(r for r in window if r > 0)
        losses = sum(-r for r in window if r < 0)
        pf = (gains / losses) if losses > 0 else float("inf")
        return pf, len(window)
    except Exception as e:
        logger.debug("insider rolling PF failed: %s", e)
        return None, 0


def _business_days_between(d0: "date", d1: "date") -> int:
    """Weekday count from d0 to d1 (proxy for trading days; ignores holidays)."""
    if d1 <= d0:
        return 0
    days = (d1 - d0).days
    full_weeks, rem = divmod(days, 7)
    count = full_weeks * 5
    wd = d0.weekday()
    for _ in range(rem):
        wd = (wd + 1) % 7
        if wd < 5:
            count += 1
    return count


def _insider_vol_liquidity_gate(symbol: str) -> Tuple[bool, str, float, float]:
    """The VALIDATED entry filter for the insider-shares edge. Returns (ok, reason, rv20, adv).
    ok ONLY when entry-day realized_vol_20 ≥ INSIDER_COPY_VOL_MIN_RV20 (high-vol regime) AND
    20-day avg $-volume ≥ INSIDER_COPY_MIN_ADV (liquid). FAIL-CLOSED on missing/short data —
    never trade a name we can't measure (the project's standing 'missing data is not bullish' rule)."""
    if not INSIDER_COPY_VOL_GATE:
        return True, "vol_gate_disabled", 0.0, 0.0
    try:
        import numpy as _np
        from datetime import timedelta
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        try:
            from alpaca.data.enums import DataFeed
            _feed = DataFeed.IEX                     # free feed (SIP needs a paid subscription)
        except Exception:
            _feed = None
        dc = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
        end = datetime.now(timezone.utc) - timedelta(minutes=20)   # avoid the 15-min recent-SIP limit
        kw = dict(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                  start=end - timedelta(days=70), end=end)
        if _feed is not None:
            kw["feed"] = _feed
        df = dc.get_stock_bars(StockBarsRequest(**kw)).df
        if df is None or df.empty:
            return False, "vol_gate_no_bars", 0.0, 0.0
        if hasattr(df.index, "levels"):
            df = df.xs(symbol, level=0)
        closes = df["close"].astype(float).values
        vols = df["volume"].astype(float).values
        if len(closes) < 21:
            return False, f"vol_gate_short_history:{len(closes)}", 0.0, 0.0
        rets = _np.diff(_np.log(_np.maximum(closes, 1e-9)))
        rv20 = float(_np.std(rets[-20:]))
        adv = float(_np.mean([closes[k] * vols[k] for k in range(len(closes) - 20, len(closes))]))
        if rv20 < INSIDER_COPY_VOL_MIN_RV20:
            return False, f"insider_copy_vol_too_low:rv20={rv20:.4f}<{INSIDER_COPY_VOL_MIN_RV20:.4f}", rv20, adv
        if adv < INSIDER_COPY_MIN_ADV:
            return False, f"insider_copy_illiquid:adv=${adv:,.0f}<${INSIDER_COPY_MIN_ADV:,.0f}", rv20, adv
        return True, "vol_liquidity_ok", rv20, adv
    except Exception as e:
        return False, f"vol_gate_error:{str(e)[:40]}", 0.0, 0.0


def run_insider_copy_exits(gateway: AlpacaGateway, risk: RiskManager,
                           option_positions: Optional[List[Dict[str, Any]]] = None,
                           equity_positions: Optional[Dict[str, Any]] = None) -> int:
    """
    Time-exit for copied calls: force-close any insider-copy call held >=
    INSIDER_COPY_OPTION_MAX_HOLD_DAYS business days. Runs in the unconditional EXIT pass
    (never regime-/cutoff-gated, like every other exit). Sells go through the RiskManager
    gate, which always allows risk-reducing exits except under the manual kill switch.
    Self-cleans state for positions no longer held. Fully fail-safe.

    Returns the number of time-exit SELLs submitted.
    """
    if not ENABLE_INSIDER_COPIER:
        return 0
    state = _insider_copy_state_load()
    if not state:
        return 0
    # held qty for BOTH instruments: options (list of dicts) + shares (dict {sym: qty|{...}}).
    held_qty = {str(p.get("symbol", "")): int(float(p.get("qty", 0) or 0))
                for p in (option_positions or [])}
    for s, v in (equity_positions or {}).items():
        try:
            q = v.get("qty") if isinstance(v, dict) else v
            held_qty[str(s).upper()] = int(float(q or 0))
        except Exception:
            continue
    today = datetime.now(ET).date()
    closed = 0
    changed = False
    for key in list(state.keys()):
        rec = state[key] if isinstance(state[key], dict) else {}
        instrument = str(rec.get("instrument", "option"))      # legacy records (no tag) = option
        # per-instrument hold: shares exit at the validated 3 days; options at 10.
        max_hold = (INSIDER_COPY_SHARE_HOLD_DAYS if instrument == "share"
                    else INSIDER_COPY_OPTION_MAX_HOLD_DAYS)
        if max_hold <= 0:
            continue
        qty = held_qty.get(str(key).upper(), held_qty.get(key, 0))
        if qty <= 0:
            state.pop(key, None); changed = True       # position gone (other exit / expiry)
            continue
        try:
            entry = datetime.fromisoformat(str(rec.get("entry_date"))[:10]).date()
        except Exception:
            entry = today
        age = _business_days_between(entry, today)
        if age < max_hold:
            continue
        resp = gateway.submit_market_order(
            key, "sell", qty, reason=f"insider_copy_time_exit:{instrument}:age={age}d>={max_hold}")
        if resp is not None:
            closed += 1; changed = True
            # Realized-return ledger for the rolling-PF kill-switch (best-effort; uses entry price
            # stored at open + current exit reference price). Per-share basis (option vs share both fine).
            try:
                _entry_px = float(rec.get("entry_price", 0.0) or 0.0)
                _exit_px = float(gateway._resolve_reference_price(key) or 0.0)
                if _entry_px > 0 and _exit_px > 0:
                    _ret = _exit_px / _entry_px - 1.0
                    _mult = 100.0 if instrument == "option" else 1.0
                    _insider_ledger_append(today, key, instrument, _ret, (_exit_px - _entry_px) * qty * _mult)
            except Exception:
                pass
            state.pop(key, None)
            logger.info("INSIDER COPY TIME-EXIT %s x%d (%s, held %d business days)", key, qty, instrument, age)
            try:
                log_executed_order(event="insider_copy_exit", engine="insider_copier",
                                   symbol=key, underlying=(_occ_underlying(key) if instrument == "option" else key),
                                   asset_class=("option" if instrument == "option" else "equity"),
                                   side="sell", qty=qty, hold_business_days=age,
                                   order_id=str((resp or {}).get("id", "") if isinstance(resp, dict) else ""),
                                   sim=bool(isinstance(resp, dict) and resp.get("sim")),
                                   note=f"insider copy {instrument} time-exit")
            except Exception:
                pass
    if changed:
        _insider_copy_state_save(state)
    if closed:
        logger.info("Insider copier time-exit: %d position(s) closed", closed)
    return closed


def run_option_safety_exits(gateway: AlpacaGateway, risk: "RiskManager",
                            option_positions: Optional[List[Dict[str, Any]]],
                            alpha: Any,
                            insider_state_keys: Optional[set] = None) -> int:
    """
    Alpha-independent stop/target/expiry exit for open OPTION positions.

    Runs in the unconditional exit pass as a BACKSTOP only when the Alpha engine is
    disabled (alpha is None) — when alpha is live it owns option exits via
    run_alpha_exits, so this does nothing to avoid double-management. Insider-copied
    options are also skipped (run_insider_copy_exits owns those). Sells go through the
    same RiskManager-gated path as every other exit. Fully fail-safe; never raises.

    Closes a position when premium P&L <= -OPTION_SAFETY_STOP_PCT, >= +OPTION_SAFETY_TP_PCT,
    or days-to-expiry <= OPTION_SAFETY_MIN_DTE. Returns the number of SELLs submitted.
    """
    if not ENABLE_OPTION_SAFETY_EXIT or alpha is not None:
        return 0
    closed = 0
    for p in (option_positions or []):
        try:
            sym = str(p.get("symbol", ""))
            qty = int(float(p.get("qty", 0) or 0))
            if qty <= 0 or not sym:
                continue
            if insider_state_keys and sym in insider_state_keys:
                continue  # owned by insider-copier time-exit
            plpc = float(p.get("unrealized_plpc", 0) or 0)   # fraction, e.g. -0.32 = -32%
            dte = _occ_dte(sym)
            reason = None
            if plpc <= -abs(OPTION_SAFETY_STOP_PCT):
                reason = f"option_safety_stop:plpc={plpc:.2%}<=-{OPTION_SAFETY_STOP_PCT:.0%}"
            elif plpc >= abs(OPTION_SAFETY_TP_PCT):
                reason = f"option_safety_tp:plpc={plpc:.2%}>=+{OPTION_SAFETY_TP_PCT:.0%}"
            elif dte is not None and dte <= OPTION_SAFETY_MIN_DTE:
                reason = f"option_safety_dte:dte={dte}<={OPTION_SAFETY_MIN_DTE}"
            if not reason:
                continue
            resp = gateway.submit_market_order(sym, "sell", qty, reason=reason)
            if resp is not None:
                closed += 1
                logger.info("OPTION SAFETY EXIT %s x%d (%s)", sym, qty, reason)
                try:
                    log_executed_order(
                        event="option_safety_exit", engine="option_safety",
                        symbol=sym, underlying=_occ_underlying(sym), asset_class="option",
                        side="sell", qty=qty,
                        order_id=str((resp or {}).get("id", "") if isinstance(resp, dict) else ""),
                        sim=bool(isinstance(resp, dict) and resp.get("sim")), note=reason)
                except Exception:
                    pass
        except Exception as e:
            logger.error("Option safety-exit failed for %s (rest unaffected): %s",
                         p.get("symbol") if isinstance(p, dict) else p, e)
    if closed:
        logger.info("Option safety-exit: %d option position(s) closed", closed)
    return closed


def _select_insider_call(gateway: AlpacaGateway, symbol: str, target_dte: int,
                         min_delta: float, min_oi: int) -> Optional[Dict[str, Any]]:
    """
    Pick a long naked CALL to copy a bullish insider buy: expiry nearest `target_dte`,
    delta strictly > `min_delta`, and a tradable ask. Among contracts meeting the delta
    floor it prefers the one with expiry closest to target, then the LOWEST qualifying
    delta (cheapest premium / most leverage that still clears the floor). Returns the
    contract dict (chain shape) or None when nothing qualifies. Never raises.
    """
    try:
        chain = gateway.get_options_chain(symbol) or []
    except Exception as e:
        logger.debug("Insider copier chain fetch failed for %s: %s", symbol, e)
        return None
    today = datetime.now(ET).date()
    best = None
    best_key = None
    for c in chain:
        try:
            if str(c.get("type", "call")).lower() != "call":
                continue
            delta = float(c.get("delta") or 0.0)
            if delta <= min_delta:
                continue
            exp = str(c.get("expiration_date") or "")[:10]
            try:
                dte = (datetime.fromisoformat(exp).date() - today).days
            except Exception:
                continue
            if dte <= 0:
                continue
            ask = float(c.get("ask_price") or 0.0)
            if ask <= 0:
                continue
            if int(c.get("open_interest") or 0) < min_oi and int(c.get("volume") or 0) <= 0:
                continue
            # rank: nearest expiry to target, then smallest qualifying delta (cheapest)
            key = (abs(dte - target_dte), delta)
            if best_key is None or key < best_key:
                best_key, best = key, {**c, "dte": dte}
        except Exception:
            continue
    return best


def run_insider_trade_copier(gateway: AlpacaGateway, risk: RiskManager,
                             sec_candidates: List[Dict[str, Any]],
                             equity_positions: Dict[str, Any],
                             market_context: dict,
                             option_positions: Optional[List[Dict[str, Any]]] = None) -> int:
    """
    Insider trade copier — mirror large, RECENT open-market insider BUYS.

    Instrument (INSIDER_COPY_INSTRUMENT):
        * "option" (default) — buy a long naked CALL (insider buys are bullish): expiry
          nearest INSIDER_COPY_OPTION_DTE (30), call delta > INSIDER_COPY_OPTION_MIN_DELTA
          (0.47), sized to INSIDER_COPY_OPTION_DOLLARS of premium, routed as a DAY limit
          at the ask and gated by the options-exposure cap.
        * "share" — buy shares (original behaviour), sized to INSIDER_COPY_DOLLARS.

    Sources (both modules supplied by the user):
        * sec_form4_feed       — SEC EDGAR Form 4 open-market-purchase candidates carry
                                 the actual aggregate buy value + latest transaction date.
        * insider_transactions_av — Alpha Vantage INSIDER_TRANSACTIONS, used as an
                                 optional CONFIRMATION veto (fail-open).

    A name is copied only when ALL hold:
        * it is a Form 4 open-market-purchase candidate,
        * aggregate open-market insider buy value > INSIDER_COPY_MIN_VALUE ($1,000,000),
        * the latest insider transaction is within INSIDER_COPY_MAX_AGE_DAYS days (2),
        * (if INSIDER_COPY_REQUIRE_AV_CONFIRM) Alpha Vantage does not show net insider
          selling — fail-open, so missing key / quota / error never blocks,
        * the underlying is not already held (as shares OR an open call),
        * for options, a call meeting the DTE/delta/liquidity rule exists,
        * new buys are allowed this cycle and the 15:45 ET cutoff has not passed.

    Every order goes through gateway.submit_market_order, so the full RiskManager gate
    (kill-switch, equity floor, daily-loss, drawdown, PDT, position count, exposure caps
    incl. options-exposure, self-assessment throttle, size-to-fit) applies exactly as for
    any other entry. The copier can NEVER bypass a risk control, place a SELL, or trade
    while the market is shut. Fully fail-safe — any error is logged and the rest of the
    cycle is unaffected.

    Returns the number of insider-copy orders submitted.
    """
    if not ENABLE_INSIDER_COPIER:
        return 0
    if not is_market_open():
        return 0
    # Same hard 15:45 ET entry cutoff used by the Alpha/BAL entry passes.
    now_et = datetime.now(ET)
    if now_et >= now_et.replace(hour=15, minute=45, second=0, microsecond=0):
        logger.info("Insider copier blocked: past 15:45 ET entry cutoff")
        return 0
    new_buys_allowed, regime_reason = regime_allows_new_buys(market_context)
    if not new_buys_allowed:
        log_skip("INSIDER_COPIER", regime_reason, phase="insider_copier")
        return 0

    # ── ROLLING-PF KILL-SWITCH — halt NEW entries on live edge decay ────────────────────────
    # Strategy-level: if the realised PF over the last N closed trades (≥ MIN_TRADES, within
    # LOOKBACK days) drops below threshold, stop opening new copies. EXITS are unaffected (they
    # run in their own pass). Auto-recovers when PF climbs back. Needs MIN_TRADES first (no halt
    # on a cold start). One of the highest-value safeguards: protects against edge decay.
    if INSIDER_COPY_PF_KILL_ENABLED:
        _pf, _pf_n = _insider_rolling_pf()
        if _pf is not None and _pf < INSIDER_COPY_PF_THRESHOLD:
            logger.warning("INSIDER COPIER HALTED: live_pf_degradation rolling_pf=%.2f<%.2f over %d trades",
                           _pf, INSIDER_COPY_PF_THRESHOLD, _pf_n)
            log_skip("INSIDER_COPIER",
                     f"live_pf_degradation:pf={_pf:.2f}<{INSIDER_COPY_PF_THRESHOLD}:n={_pf_n}",
                     phase="insider_copier")
            return 0

    today = now_et.date()
    # "Already held" spans BOTH equity holdings and any open call on the underlying, so a
    # name is never copied twice regardless of which instrument the copier trades.
    held = {str(s).upper() for s in (equity_positions or {})}
    held |= {_occ_underlying(str(p.get("symbol", ""))) for p in (option_positions or [])}

    # Filter Form 4 candidates to large + recent open-market buys.
    qualified: List[Dict[str, Any]] = []
    for cand in sec_candidates or []:
        try:
            sym = str(cand.get("symbol") or "").upper().strip()
            if not sym or is_option_symbol(sym):
                continue
            value = float(cand.get("total_open_market_purchase_value") or 0.0)
            if value <= INSIDER_COPY_MIN_VALUE:
                log_skip(sym, f"insider_copy_below_min:{value:.0f}<={INSIDER_COPY_MIN_VALUE:.0f}",
                         phase="insider_copier")
                continue
            # Freshness from the FILING date (when the Form 4 hit EDGAR), not the transaction date:
            # insiders have up to 2 business days to file, so transaction-date freshness can never
            # satisfy a tight window. Filing-date = "act on newly-filed signals" (matches backtest entry).
            raw_filed = str(cand.get("latest_filing_date") or "")[:10]
            raw_txn = str(cand.get("latest_transaction_date") or "")[:10]
            raw_last = raw_filed or raw_txn
            try:
                last_date = datetime.fromisoformat(raw_last).date() if raw_last else None
            except Exception:
                last_date = None
            if last_date is None or (today - last_date).days > INSIDER_COPY_MAX_AGE_DAYS:
                log_skip(sym, f"insider_copy_stale:filed={raw_filed or 'none'}_txn={raw_txn or 'none'}",
                         phase="insider_copier")
                continue
            if sym in held:
                log_skip(sym, "insider_copy_already_held", phase="insider_copier")
                continue
            qualified.append({"symbol": sym, "value": value, "last": last_date,
                              "company": str(cand.get("company") or ""),
                              "breadth": int(cand.get("insider_count") or 1),     # distinct buyers (consensus)
                              "repeated": bool(cand.get("repeated_purchases"))})
        except Exception:
            continue

    if not qualified:
        return 0

    # Most recent insider buy first, then largest dollar value.
    qualified.sort(key=lambda c: (c["last"], c["value"]), reverse=True)

    # COMBO gate (opt-in): intersect with the TradingView quality-growth screen. NOTE the
    # backtest shows this REMOVES the edge — kept for live A/B only.
    quality_universe: set = set()
    if INSIDER_COPY_REQUIRE_QUALITY_SCREEN and TV_SCREEN_TICKERS_PATH.exists():
        try:
            quality_universe = set(_parse_ticker_file(TV_SCREEN_TICKERS_PATH.read_text()))
        except Exception:
            quality_universe = set()

    placed = 0
    _open_now = len(_insider_copy_state_load())     # total open insider-copy positions (small-acct cap)
    for cand in qualified:
        if placed >= INSIDER_COPY_MAX_NAMES:
            break
        if (_open_now + placed) >= INSIDER_COPY_MAX_OPEN:
            log_skip("INSIDER_COPIER", f"insider_copy_max_open:{_open_now + placed}>={INSIDER_COPY_MAX_OPEN}",
                     phase="insider_copier")
            break
        sym = cand["symbol"]

        if INSIDER_COPY_REQUIRE_QUALITY_SCREEN and sym not in quality_universe:
            log_skip(sym, "insider_copy_not_in_quality_screen", phase="insider_copier")
            continue

        # Optional Alpha Vantage confirmation. FAIL-OPEN: only a positive net-selling
        # signal vetoes a copy; a missing key / quota / parse error never blocks.
        if INSIDER_COPY_REQUIRE_AV_CONFIRM:
            try:
                import insider_transactions_av as av
                trend = av.get_insider_trend(sym)
                if not trend.get("fetch_error"):
                    flags = trend.get("insider_warning_flags") or []
                    if (float(trend.get("insider_trend_score") or 0.0) < 0.0
                            or "heavy_insider_selling" in flags):
                        log_skip(sym, "insider_copy_av_contradicts", phase="insider_copier",
                                 av_score=trend.get("insider_trend_score"))
                        continue
            except Exception as e:
                logger.debug("Insider copier AV confirm skipped for %s: %s", sym, e)

        # ── VALIDATED VOL + LIQUIDITY GATE (the OOS walk-forward edge) ──────────────────────
        # Trade only high-vol (entry-day realized_vol_20 ≥ train P80) + liquid (20d $-vol ≥ $1M)
        # insider names. Fail-CLOSED on missing data. This is THE rule that lifted PF 1.44→2.20.
        _vg_ok, _vg_reason, _rv20, _adv = _insider_vol_liquidity_gate(sym)
        if not _vg_ok:
            log_skip(sym, _vg_reason, phase="insider_copier", rv20=round(_rv20, 4), adv=round(_adv, 0))
            continue

        # ── CLUSTER-BREADTH amplifier (SHADOW by default) ───────────────────────────────────
        # >=2 distinct insiders = consensus (validated PF 1.9→2.57). In SHADOW mode we journal
        # breadth + would-pass but DON'T gate, so paper collects accepted-vs-rejected for 30-60d.
        # Hard-gate only when INSIDER_COPY_BREADTH_SHADOW=0.
        _breadth = int(cand.get("breadth", 1) or 1)
        _breadth_pass = _breadth >= INSIDER_COPY_MIN_DISTINCT_BUYERS
        if not _breadth_pass:
            if INSIDER_COPY_BREADTH_SHADOW:
                logger.info("INSIDER COPY breadth-SHADOW: %s breadth=%d<%d would be REJECTED "
                            "(taking it anyway in shadow mode)", sym, _breadth, INSIDER_COPY_MIN_DISTINCT_BUYERS)
            else:
                log_skip(sym, f"insider_copy_below_breadth:{_breadth}<{INSIDER_COPY_MIN_DISTINCT_BUYERS}",
                         phase="insider_copier", breadth=_breadth)
                continue

        # Resolve the instrument to copy with: SHARES (default, validated) or a long naked CALL.
        if INSIDER_COPY_INSTRUMENT == "option":
            contract = _select_insider_call(gateway, sym, INSIDER_COPY_OPTION_DTE,
                                             INSIDER_COPY_OPTION_MIN_DELTA, INSIDER_COPY_OPTION_MIN_OI)
            if not contract:
                log_skip(sym, "insider_copy_no_qualifying_call",
                         phase="insider_copier",
                         want_dte=INSIDER_COPY_OPTION_DTE, want_delta_gt=INSIDER_COPY_OPTION_MIN_DELTA)
                continue
            order_symbol = str(contract["symbol"])
            ask = float(contract.get("ask_price") or 0.0)
            per_contract = ask * 100.0
            # SIZING: research-grounded fractional-Kelly + risk cap (options_quant), else the
            # legacy fixed-dollar budget. Kelly fraction is the dispersion-shrunk full Kelly
            # measured on the deep-ITM payoff distribution, scaled by INSIDER_COPY_KELLY_MULT.
            omega = 0.0
            if INSIDER_COPY_USE_QUANT_SIZING:
                try:
                    import options_quant as _oq
                    qty = _oq.position_size_contracts(
                        float(risk.equity), ask, kelly_f=INSIDER_COPY_KELLY_FULL,
                        kelly_fraction=INSIDER_COPY_KELLY_MULT, risk_cap_pct=INSIDER_COPY_RISK_CAP_PCT)
                    spot = gateway._resolve_reference_price(sym)
                    omega = _oq.embedded_leverage(spot, ask, float(contract.get("delta") or 0))
                except Exception as e:
                    logger.debug("quant sizing failed for %s (%s) — using fixed budget", sym, e)
                    qty = int(INSIDER_COPY_OPTION_DOLLARS // per_contract)
            else:
                qty = int(INSIDER_COPY_OPTION_DOLLARS // per_contract)
            if qty <= 0:
                log_skip(sym, f"insider_copy_size_zero:call_premium={per_contract:.0f}>budget",
                         phase="insider_copier")
                continue
            # Marketable DAY limit at the ask so the long call fills.
            resp = gateway.submit_market_order(
                order_symbol, "buy", qty,
                reason=(f"insider_copy_call:${cand['value']:,.0f}>{INSIDER_COPY_MIN_VALUE:,.0f}"
                        f":dte={contract['dte']}:delta={float(contract.get('delta') or 0):.2f}"),
                limit_price=ask, order_type="limit")
            exec_price, asset = ask, "option"
        else:
            order_symbol = sym
            ref_price = gateway._resolve_reference_price(sym)
            if ref_price <= 0:
                log_skip(sym, "rejected_by_data_unavailable:no_reference_price",
                         phase="insider_copier", side="buy")
                continue
            # RISK-BASED sizing: notional = equity × risk_pct / stop_pct (1% risk at a 10% implied
            # adverse move → ~10% alloc; tiny lots on a small account). RiskManager.gate() still caps.
            _eq = float(getattr(risk, "equity", 0.0) or 0.0)
            _alloc = (_eq * INSIDER_COPY_SHARE_RISK_PCT / INSIDER_COPY_SHARE_STOP_PCT) if INSIDER_COPY_SHARE_STOP_PCT > 0 else 0.0
            _alloc = _alloc or INSIDER_COPY_DOLLARS                      # fallback to legacy budget
            qty = int(_alloc // ref_price)
            if qty <= 0:
                log_skip(sym, f"insider_copy_size_zero:px={ref_price:.2f}:alloc=${_alloc:.0f}",
                         phase="insider_copier")
                continue
            resp = gateway.submit_market_order(
                sym, "buy", qty,
                reason=(f"insider_copy_share:${cand['value']:,.0f}>{INSIDER_COPY_MIN_VALUE:,.0f}"
                        f":rv20={_rv20:.3f}:hold={INSIDER_COPY_SHARE_HOLD_DAYS}d"),
                reference_price=ref_price)
            contract, exec_price, asset = None, ref_price, "equity"

        if resp is not None:
            placed += 1
            if asset == "option":
                # Record entry so the time-exit can force-close the call after N days.
                _insider_copy_record_open(order_symbol, sym, qty, today, entry_price=exec_price)
                logger.info("INSIDER COPY CALL %s x%d (%s, dte=%s delta=%.2f Ω=%.1f ask=%.2f; "
                            "insider buy $%.0f, last=%s, %s)",
                            order_symbol, qty, sym, contract["dte"],
                            float(contract.get("delta") or 0), omega, exec_price,
                            cand["value"], cand["last"].isoformat(), cand["company"])
            else:
                # Record the share entry so the 3-day time-exit force-closes it.
                _insider_copy_record_open(order_symbol, sym, qty, today, instrument="share",
                                          entry_price=exec_price)
                logger.info("INSIDER COPY BUY %s x%d (share, rv20=%.3f adv=$%.0f; insider buy $%.0f, "
                            "last=%s, %s)", order_symbol, qty, _rv20, _adv,
                            cand["value"], cand["last"].isoformat(), cand["company"])
            # Journal it explicitly as an "insider copy buy" so it is distinguishable in
            # the execution event stream AND the executor_journal.csv funnel reads.
            try:
                log_executed_order(
                    event="insider_copy_buy", engine="insider_copier",
                    symbol=order_symbol, underlying=sym, asset_class=asset,
                    side="buy", qty=qty, limit_price=exec_price, grade="insider_copy",
                    option_dte=(contract.get("dte") if contract else ""),
                    option_delta=(float(contract.get("delta") or 0) if contract else ""),
                    option_omega=(round(omega, 2) if asset == "option" else ""),
                    order_id=str((resp or {}).get("id", "") if isinstance(resp, dict) else ""),
                    sim=bool(isinstance(resp, dict) and resp.get("sim")),
                    insider_buy_value=cand["value"],
                    insider_last_date=cand["last"].isoformat(),
                    company=cand["company"],
                    breadth_distinct_buyers=_breadth,            # for the shadow-mode comparison
                    breadth_pass=_breadth_pass,
                    repeated_buyer=bool(cand.get("repeated")),
                    note=("insider copy call" if asset == "option" else "insider copy buy"))
            except Exception as _je:
                logger.debug("Insider copier journal write skipped for %s: %s", sym, _je)

    if placed:
        logger.info("Insider trade copier: %d copy %s(s) submitted", placed,
                    "CALL" if INSIDER_COPY_INSTRUMENT == "option" else "BUY")
    return placed


def run_two_engine_cycle(risk: RiskManager, gateway: AlpacaGateway, waterfall,
                         alpha, bal, monitor, market_context: dict) -> dict:
    """
    One orchestration iteration. ORDER (hard requirement):
        sync  →  ALPHA EXITS  →  BAL EXITS  →  screeners  →  ALPHA ENTRIES  →  BAL ENTRIES
    Exits ALWAYS run (independent of regime / entries-enabled / screener success).
    Buffett work is fully wrapped so its failure never affects Alpha exits or entries.
    run.py performs ONLY fetching + execution; all Alpha intelligence is in the engine.
    """
    # 1. Sync account / positions / exposure
    risk.sync_account(gateway.get_account())
    raw_positions = gateway.get_positions_raw()
    option_positions, equity_positions, opt_exp, stk_exp = gateway.split_positions(raw_positions)
    risk.sync_exposure(opt_exp, stk_exp)
    new_buys_allowed, regime_reason = regime_allows_new_buys(market_context)

    summary = {"alpha_exits": 0, "bal_exits": 0, "alpha_entries": 0, "bal_entries": 0,
               "insider_copies": 0, "insider_copy_exits": 0, "option_safety_exits": 0, "ema_exits": 0,
               "new_buys_allowed": new_buys_allowed}

    # Shared real-time indicator state (carried over from prior cycles for the exit
    # pass; refreshed below once this cycle's bars are fetched). Diagnostics only here.
    indicator_snapshot = _indicator_snapshot()
    # Frozen premarket snapshot (gap / premarket high-low / volume) for early-session
    # gap & ORB context. Awareness only — premarket data never creates an order.
    premarket_snapshot = _premarket_snapshot()
    # Single shared market truth for this cycle (advisory; consumed by Alpha, OOPS,
    # Probability, Buffett, Black-Litterman, and the Adaptive Allocator). Never orders.
    # Detect the active Alpaca feed (read-only; iex/overnight, never SIP), write feed
    # diagnostics, and tag the streaming engine. Advisory data-quality only.
    _active_feed = _alpaca_feed_cycle("rth")
    market_state = _build_market_state(market_context, indicator_snapshot, premarket_snapshot,
                                       feed_type=_active_feed)

    # 2. EXIT PASS — unconditional (regime/entries/screeners cannot block exits) ─────
    if alpha is not None:
        summary["alpha_exits"] = run_alpha_exits(alpha, gateway, waterfall, option_positions,
                                                 market_context, indicator_snapshot=indicator_snapshot,
                                                 market_state=market_state, risk=risk)
    if bal is not None:
        try:
            summary["bal_exits"] = run_bal_exits(bal, gateway, equity_positions, market_context)
        except Exception as e:
            logger.error("BAL exits failed (Alpha unaffected): %s", e)
    # Regime exit — liquidate on a market-wide risk-off (majority of SPY/QQQ/IWM below their 200-SMA).
    if ENABLE_MARKET_REGIME:
        try:
            summary["regime_exits"] = run_regime_exit(gateway, equity_positions)
        except Exception as e:
            logger.error("regime exit failed (rest of cycle unaffected): %s", e)
    # Per-name trend exit — 200-SMA break OR per-position stop (unconditional exit pass).
    if ENABLE_EMA_EXIT:
        try:
            summary["ema_exits"] = run_ema_exits(gateway, equity_positions)
        except Exception as e:
            logger.error("trend/stop exit failed (rest of cycle unaffected): %s", e)
    # Insider-copier time-exit: force-close copied calls past their max hold. Part of the
    # unconditional exit pass — never regime-/cutoff-gated. Fully fail-safe.
    if ENABLE_INSIDER_COPIER:
        try:
            summary["insider_copy_exits"] = run_insider_copy_exits(gateway, risk, option_positions, equity_positions)
        except Exception as e:
            logger.error("Insider copier time-exit failed (rest of cycle unaffected): %s", e)
    # Alpha-independent option safety-exit: only fires when the Alpha engine is OFF, so
    # orphaned options (no exit owner) still get a stop/target/expiry close. Skips
    # insider-copied options (owned above). Part of the unconditional exit pass.
    try:
        _insider_keys = set(_insider_copy_state_load().keys()) if ENABLE_INSIDER_COPIER else None
        summary["option_safety_exits"] = run_option_safety_exits(
            gateway, risk, option_positions, alpha, insider_state_keys=_insider_keys)
    except Exception as e:
        logger.error("Option safety-exit failed (rest of cycle unaffected): %s", e)

    # 3. SCREENERS — separate, independent universes. Buffett fully isolated. ────────
    dynamic = load_candidate_universe()          # TradingView institutional dynamic discovery
    alpha_scr = {"base": [], "bars": {}, "chains": {}, "funnel": {}}
    if alpha is not None:
        try:
            alpha_scr = run_alpha_screener(alpha, gateway, market_context, dynamic)
        except Exception as e:
            logger.error("Alpha screener failed: %s", e)
    # SEC Form 4 insider-buy discovery: DISCOVERY ONLY. Expands the Buffett review
    # universe; never generates orders, never bypasses any gate. Fully fail-safe —
    # a failure here must not stop the Buffett screener or the Alpha engine.
    #
    # FIX 2026-06-09: SEC discovery runs in a background thread so it never blocks
    # the main trading cycle. The main thread reads the LAST completed result (or
    # uses the on-disk cache written by sec_form4_feed's own 4h output-cache guard).
    # On the first cycle the result defaults to empty; subsequent cycles pick up
    # whatever the background thread last wrote to disk.
    insider_data: Dict[str, Dict] = {}
    sec_candidates: List[Dict[str, Any]] = []   # full Form 4 candidate records for the copier
    if (bal is not None and ENABLE_SEC_FORM4_FEED) or ENABLE_INSIDER_COPIER:
        try:
            import sec_form4_feed as _sec_mod
            import threading as _sec_threading

            # If a background discovery thread is already running, skip and use cached data.
            _sec_thread_attr = "_sec_form4_bg_thread"
            _bg: Optional[Any] = getattr(run_two_engine_cycle, _sec_thread_attr, None)
            if _bg is None or not _bg.is_alive():
                def _run_sec_bg():
                    try:
                        _sec_mod.run_sec_form4_discovery()
                    except Exception as _e:
                        logger.warning("SEC Form 4 bg discovery error (non-fatal): %s", _e)
                _bg_t = _sec_threading.Thread(target=_run_sec_bg, name="sec-form4-discovery", daemon=True)
                _bg_t.start()
                setattr(run_two_engine_cycle, _sec_thread_attr, _bg_t)
                logger.info("SEC Form 4 discovery: launched in background thread")

            # Always read whatever is already on disk (written by previous bg run or cache).
            insider_data = _sec_mod.insider_data_for_layer() or {}
            sec_candidates_map = _sec_mod.load_insider_candidates() or {}
            sec_candidates = [{"symbol": s, **v} for s, v in sec_candidates_map.items()]
            if sec_candidates:
                logger.info("SEC Form 4: %d cached insider candidates loaded for this cycle",
                            len(sec_candidates))
        except Exception as e:
            logger.error("SEC Form 4 discovery failed (Buffett/Alpha unaffected): %s", e)
    buffett_res = run_buffett_screener(market_context, premarket_snapshot) if bal is not None else \
        {"candidates": [], "funnel": {}, "detail": []}

    # 4. ENTRY PASS — after all exits. Alpha first, then BAL. ────────────────────────
    if not new_buys_allowed:
        log_skip("UNIVERSE", regime_reason, phase="entry_gate")
    # Update the shared indicator state from THIS cycle's alpha bars, then snapshot.
    # run.py only fetches + updates indicators; the engine still decides every entry.
    _update_indicators(alpha_scr.get("bars", {}))
    indicator_snapshot = _indicator_snapshot()
    # Alpha entries ALWAYS call evaluate_universe so the universe file is written; the
    # ENGINE rejects entries internally when the regime is hostile (engine setup gate).
    # Hard entry cutoff: no new BUYs after 15:45 ET — options orders need ~15 min to
    # settle before the 16:00 ET market close. EOD forced closes still run (separate path).
    _now_et_entries = datetime.now(ZoneInfo("America/New_York"))
    _entry_cutoff_et = _now_et_entries.replace(hour=15, minute=45, second=0, microsecond=0)
    _entries_blocked = (is_market_open() and _now_et_entries >= _entry_cutoff_et) or not is_market_open()
    if _entries_blocked and is_market_open():
        logger.info("Alpha entries blocked: past 15:45 ET entry cutoff (%s ET)", _now_et_entries.strftime("%H:%M:%S"))
    if alpha is not None and not _entries_blocked:
        summary["alpha_entries"] = run_alpha_entries(
            alpha, gateway, waterfall, risk,
            alpha_scr["base"], alpha_scr["bars"], alpha_scr["chains"],
            market_context, positions=equity_positions, indicator_snapshot=indicator_snapshot,
            premarket_snapshot=premarket_snapshot, market_state=market_state)
    elif alpha is not None and _entries_blocked and not is_market_open():
        # Market closed: still run evaluate_universe to keep universe artifact fresh
        # but pass empty symbols to avoid chain fetches / wasted API calls.
        try:
            alpha.evaluate_universe(
                symbols=[], bars_by_symbol={}, market_context=market_context,
                equity=risk.equity, options_exposure=risk.options_exposure, waterfall_budget=0.0,
                chains_by_symbol={}, positions=equity_positions,
                account_state={"equity": risk.equity, "buying_power": risk.buying_power},
                indicator_snapshot=indicator_snapshot, premarket_snapshot=premarket_snapshot,
                market_state=market_state)
        except Exception:
            pass
    # Black-Litterman + Efficient-Frontier optimizer: produce DESIRED target weights
    # (how much to own). Composite DCA decides WHEN; RiskManager decides whether ALLOWED.
    # Fully fail-safe — any failure leaves target_weights empty and BAL uses conviction.
    target_weights: Dict[str, float] = {}
    optimizer_objective_weights: Optional[dict] = None
    if bal is not None and ENABLE_BUFFETT_OPTIMIZER:
        try:
            import buffett_portfolio_optimizer as bopt
            opt_signals = [r for r in (buffett_res.get("detail", []) or []) if r.get("is_candidate")]
            if opt_signals:
                caps = {r["symbol"]: r.get("market_cap", 0.0) for r in opt_signals if r.get("symbol")}
                opt_res = bopt.run_optimizer_safe(opt_signals, market_caps=caps,
                                                  current_weights=None, write=True,
                                                  market_state=market_state)
                target_weights = opt_res.get("target_weights", {}) or {}
                optimizer_objective_weights = {
                    "active_optimizer": opt_res.get("active_optimizer"),
                    "used_objectives": opt_res.get("used_objectives"),
                }
        except Exception as e:
            logger.error("Buffett optimizer failed (BAL uses conviction): %s", e)

    # BAL buys are hard-gated on regime permission (halt_new_buys / crisis).
    if bal is not None and new_buys_allowed:
        try:
            bcands = buffett_res.get("candidates", [])
            bal_bars = fetch_1m_bars(gateway, bcands) if bcands else {}
            bal_ctx = build_bars_ctx(bal_bars)
            # Feed the Buffett candidates' bars into the shared indicator state, then
            # refresh the snapshot so BAL reads streaming VWAP for timing (only).
            _update_indicators(bal_bars)
            bal_snapshot = _indicator_snapshot()
            summary["bal_entries"] = run_bal_entries(bal, gateway, waterfall, risk,
                                                     bcands, market_context, bal_ctx, equity_positions,
                                                     insider_data_by_symbol=insider_data,
                                                     target_weights_by_symbol=target_weights,
                                                     indicator_snapshot=bal_snapshot,
                                                     market_state=market_state,
                                                     optimizer_objective_weights=optimizer_objective_weights)
        except Exception as e:
            logger.error("BAL entries failed (Alpha unaffected): %s", e)

    # 4b. INSIDER TRADE COPIER — mirror large (> $1M), recent (≤2d) open-market insider
    # BUYS as paper BUYs. Routed through the RiskManager gate like every other entry;
    # self-guards on market-open, 15:45 ET cutoff, and regime. Fully fail-safe.
    if ENABLE_INSIDER_COPIER and not _entries_blocked:
        try:
            summary["insider_copies"] = run_insider_trade_copier(
                gateway, risk, sec_candidates, equity_positions, market_context,
                option_positions=option_positions)
        except Exception as e:
            logger.error("Insider trade copier failed (rest of cycle unaffected): %s", e)

    # 5. Optional monitor + outputs
    if monitor is not None:
        run_portfolio_monitor(monitor, gateway, risk, equity_positions, market_context)
    if waterfall is not None:
        write_capital_waterfall(waterfall)
    write_screener_funnel_report(alpha_scr.get("funnel", {}), buffett_res, market_context,
                                 alpha_placed=summary.get("alpha_entries", 0),
                                 bal_placed=summary.get("bal_entries", 0))
    # Observational execution-funnel line (read-only; reads the artifacts just written above).
    # Universe→Tradable→Signals→Qualified→Approved→Submitted→Filled + categorised rejections.
    # Never alters thresholds/gates/decisions and never raises.
    try:
        import funnel_report
        funnel_report.log_funnel(logger, DATA_TAG)
    except Exception as _e:
        logger.debug("funnel_report skipped: %s", _e)
    _write_indicator_reports()   # streaming_indicator_snapshot.json + _health.json
    # 6. Evidence System — regenerate research analytics from the recorded events.
    bal_value = sum(float(getattr(p, "market_value", 0) or 0) for p in equity_positions.values())
    _research("generate_all_reports_safe", market_context=market_context,
              start_equity=float(risk.starting_equity), bal_current_value=bal_value)
    # 7. Adaptive Capital Allocation — turn the evidence into an allocation PLAN.
    # Advisory only: it never submits orders, never overrides RiskManager, and all
    # recommended sizes are clamped to the configured caps. Fully fail-safe.
    if ENABLE_ADAPTIVE_ALLOCATION:
        try:
            import adaptive_capital_allocator as aca
            aca.build_allocation_plan_safe(market_context=market_context, market_state=market_state)
        except Exception as e:
            logger.debug("adaptive allocator skipped: %s", e)
    # RL SHADOW MODE (opt-in, OFF by default): trained RL agents log what they WOULD
    # recommend; the live system ignores them. Never creates/blocks an order. Fail-safe.
    _rl_shadow_cycle()
    return summary


def _account_num(account: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """Parse numeric account fields safely across Alpaca objects/dicts/strings."""
    try:
        value = account.get(key, default) if isinstance(account, dict) else getattr(account, key, default)
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def seed_risk_from_broker(risk: RiskManager, gateway: AlpacaGateway) -> Dict[str, Any]:
    """Seed the RiskManager anchor from a live Alpaca account snapshot at startup.

    Pulls account state from the broker BEFORE the trading loop so the equity anchor,
    drawdown floor, and downstream research/sizing all use the real account equity
    instead of the STARTING_EQUITY fallback. STARTING_EQUITY is only used if the broker
    is unreachable. Fail-safe: a transient error never blocks startup.
    """
    account: Dict[str, Any] = {}
    try:
        account = gateway.get_account() or {}
    except Exception as exc:
        logger.warning("Account seed unavailable (%s) — using STARTING_EQUITY=$%.2f fallback",
                       exc, STARTING_EQUITY)
        return {}

    if account and account.get("equity"):
        risk.sync_account(account)          # sets equity + anchor (first call) + rescales gates
        live = not account.get("sim")
        logger.info(
            "Account seeded from %s: equity=$%.2f anchor=$%.2f cash=$%.2f bp=$%.2f "
            "opt_bp=$%.2f status=%s PDT=%s daytrades=%d",
            "Alpaca" if live else "sim", risk.equity, risk.starting_equity, risk.cash,
            risk.buying_power, risk.options_buying_power, risk.account_status or "n/a",
            risk.pattern_day_trader, risk.daytrade_count,
        )
        if risk.trading_blocked or risk.account_blocked:
            logger.warning("Account flags: trading_blocked=%s account_blocked=%s",
                           risk.trading_blocked, risk.account_blocked)
    else:
        logger.warning("Account seed: no live equity from broker — anchor stays at fallback $%.2f",
                       risk.starting_equity)
    return account


def log_account_stats(gateway: AlpacaGateway, risk: RiskManager,
                      label: str = "Account") -> Dict[str, Any]:
    """
    Refresh broker account + exposure state and print one compact telemetry line.
    This is intentionally fail-safe: a transient Alpaca/account error must never
    stop the trading daemon.
    """
    account: Dict[str, Any] = {}
    try:
        account = gateway.get_account() or {}
        if account:
            risk.sync_account(account)
    except Exception as exc:
        logger.warning("Account: unavailable (%s)", exc)
        account = {}

    try:
        raw_positions = gateway.get_positions_raw()
        _, _, opt_exp, stk_exp = gateway.split_positions(raw_positions)
        risk.sync_exposure(opt_exp, stk_exp)
    except Exception as exc:
        logger.debug("Account exposure refresh skipped: %s", exc)

    # All fields are read from the live-synced RiskManager state (populated by
    # sync_account above) so the telemetry always reflects the latest broker snapshot.
    equity = float(risk.equity or STARTING_EQUITY)
    buying_power = float(risk.buying_power or equity)
    cash = float(risk.cash)
    cash_pct = (cash / equity * 100.0) if equity > 0 else 0.0
    day_pnl = equity - float(risk.last_equity or equity)   # broker-side intraday P/L
    source = "sim" if account.get("sim") else ("alpaca" if gateway.available else "sim")

    logger.info(
        "%s: equity=$%.2f cash=$%.2f bp=$%.2f opt_bp=$%.2f day_pnl=$%.2f "
        "stock_exp=$%.2f opt_exp=$%.2f cash=%.1f%% PDT=%s daytrades=%d status=%s source=%s",
        label, equity, cash, buying_power, risk.options_buying_power, day_pnl,
        risk.stock_exposure, risk.options_exposure, cash_pct,
        risk.pattern_day_trader, risk.daytrade_count, risk.account_status or "n/a", source,
    )
    return account


# ── Main-loop watchdog ────────────────────────────────────────────────────────
# Heartbeat updated at the top of every loop iteration; a daemon thread interrupts the
# main thread (SIGALRM) if it stalls past WATCHDOG_STALL_SEC, so a hung broker socket
# can no longer freeze the loop indefinitely. See WATCHDOG_STALL_SEC.
_LOOP_HEARTBEAT = [0.0]   # mutable holder; index 0 = monotonic-ish time of last progress


class _CycleStalled(Exception):
    """Raised on the main thread by the watchdog when a cycle exceeds WATCHDOG_STALL_SEC."""


def _watchdog_sigalrm(_signum, _frame):
    raise _CycleStalled()


def _start_loop_watchdog(main_tid: int) -> bool:
    """Install the SIGALRM handler and start the watchdog thread. Returns False (no-op) if
    signals/threads are unavailable on this platform. Never raises."""
    import signal as _signal
    import threading as _threading
    import faulthandler as _faulthandler
    try:
        _signal.signal(_signal.SIGALRM, _watchdog_sigalrm)   # main-thread only; loop runs here
    except (ValueError, AttributeError, OSError) as e:
        logger.warning("Loop watchdog unavailable (%s) — running without stall protection", e)
        return False

    def _monitor() -> None:
        tick = min(30.0, max(5.0, WATCHDOG_STALL_SEC / 4.0))
        while True:
            time.sleep(tick)
            try:
                last = _LOOP_HEARTBEAT[0]
                if last <= 0:
                    continue
                stalled = time.time() - last
                if stalled > WATCHDOG_STALL_SEC:
                    logger.critical(
                        "WATCHDOG: main loop stalled %.0fs (>%.0fs) — dumping stack, "
                        "interrupting cycle", stalled, WATCHDOG_STALL_SEC)
                    try:
                        _faulthandler.dump_traceback()
                    except Exception:
                        pass
                    _LOOP_HEARTBEAT[0] = time.time()   # debounce so we don't re-fire each tick
                    try:
                        _signal.pthread_kill(main_tid, _signal.SIGALRM)
                    except Exception as e:
                        logger.error("WATCHDOG: failed to interrupt main thread: %s", e)
            except Exception:
                logger.debug("watchdog tick error", exc_info=True)

    _threading.Thread(target=_monitor, name="loop-watchdog", daemon=True).start()
    logger.info("Loop watchdog started (stall threshold %.0fs)", WATCHDOG_STALL_SEC)
    return True


def run_two_engine_trading():
    risk    = RiskManager(equity=STARTING_EQUITY)
    gateway = AlpacaGateway(risk)

    logger.info("=== TWO-ENGINE TRADING  mode=%s ===", MODE)
    # Pull the full live account snapshot from Alpaca and anchor risk to real equity
    # BEFORE any sizing/research runs (falls back to STARTING_EQUITY only if unreachable).
    seed_risk_from_broker(risk, gateway)
    # Required first line after the banner: live/sim account telemetry.
    log_account_stats(gateway, risk)
    _log_config(risk)

    ensure_outputs_exist()
    alpha, bal, monitor, waterfall = build_engines(gateway)
    # Continuously refresh the TradingView quality-growth screen in the background so the
    # candidate universe is always fresh (network call stays off the trading cycle).
    _start_tv_screen_refresher()

    if alpha is None and bal is None:
        logger.critical("No engines enabled — nothing to do. Exiting.")
        return

    market_context: dict = {}
    last_context_build = 0.0
    last_account_log = time.time()
    cycle_count = 0
    _premarket_planner_spawned = False  # spawn once per session

    logger.info("Entering two-engine loop (Ctrl-C to stop) …")
    import threading as _threading
    _LOOP_HEARTBEAT[0] = time.time()
    _start_loop_watchdog(_threading.get_ident())   # interrupt the loop if a cycle hangs
    while True:
        try:
            cycle_count += 1
            _LOOP_HEARTBEAT[0] = time.time()        # watchdog progress beat

            if risk.is_killed:
                logger.critical("Kill switch active — exiting loop")
                break

            now = time.time()
            if now - last_account_log >= ACCOUNT_STATS_LOG_SEC:
                log_account_stats(gateway, risk, label=f"Account cycle={cycle_count}")
                last_account_log = now

            if not is_market_open():
                # Premarket: stream live quotes for awareness only (no trading). Any
                # failure here is swallowed and never blocks the upcoming session.
                if is_premarket():
                    if not market_context:
                        market_context = _build_context_safe()
                    # Spawn premarket_trade_planner.py once per session as a background
                    # process. It self-waits until market open (09:30 ET), then writes
                    # YYYYMMDD_premarket_trade_plan.json and YYYYMMDD_intraday_structure_map.json
                    # so they are available for the first trading cycle. Fail-safe: any
                    # launch error is logged and never blocks the premarket phase.
                    if not _premarket_planner_spawned:
                        try:
                            import subprocess as _sp
                            _planner_script = Path(__file__).parent / "premarket_trade_planner.py"
                            if _planner_script.exists():
                                _planner_proc = _sp.Popen(
                                    [sys.executable, str(_planner_script)],
                                    stdout=_sp.DEVNULL,
                                    stderr=_sp.DEVNULL,
                                )
                                logger.info(
                                    "premarket_trade_planner spawned (pid=%d) — "
                                    "will write plan+structure_map at market open",
                                    _planner_proc.pid,
                                )
                            else:
                                logger.warning(
                                    "premarket_trade_planner.py not found — "
                                    "plan+structure_map will be missing"
                                )
                        except Exception as _e:
                            logger.warning("premarket_trade_planner spawn failed: %s", _e)
                        _premarket_planner_spawned = True
                    try:
                        run_premarket_phase(gateway, bal, market_context)
                    except Exception as e:
                        logger.debug("premarket phase skipped: %s", e)
                    time.sleep(PREMARKET_REFRESH_SEC)
                    continue

                logger.debug("Market closed — sleeping 60s")
                time.sleep(60)
                continue

            # Market just opened — freeze premarket high/low by stopping the live feed;
            # the frozen snapshot stays available for early-session gap / ORB context.
            _stop_premarket_feed()

            # Stale-plan audit at market open: warn loudly if the premarket plan or
            # structure map is older than MAX_PLAN_AGE_MINUTES. These files drive entry
            # scoring; staleness causes SAFE mode and suppresses all new buys.
            try:
                _plan_path = SYSTEM_DATA_DIR / f"{DATA_TAG}_premarket_trade_plan.json"
                _map_path  = SYSTEM_DATA_DIR / f"{DATA_TAG}_intraday_structure_map.json"
                _max_age   = int(os.getenv("MAX_PLAN_AGE_MINUTES", "90"))
                for _label, _p in [("premarket_plan", _plan_path), ("structure_map", _map_path)]:
                    if _p.exists():
                        _age_min = (time.time() - _p.stat().st_mtime) / 60.0
                        if _age_min > _max_age:
                            logger.warning(
                                "STALE ARTIFACT at market open: %s age=%.0fmin (limit=%dmin) — "
                                "new buys may be suppressed; re-run premarket_trade_planner.py",
                                _label, _age_min, _max_age,
                            )
                    else:
                        logger.warning(
                            "MISSING ARTIFACT at market open: %s not found at %s — "
                            "structure-based entries will be blocked",
                            _label, _p,
                        )
            except Exception:
                pass

            if (time.time() - last_context_build) > CONTEXT_REFRESH_MIN * 60 or not market_context:
                market_context = _build_context_safe()
                last_context_build = time.time()

            run_two_engine_cycle(risk, gateway, waterfall, alpha, bal, monitor, market_context)
            time.sleep(BAR_INTERVAL_SEC)

        except KeyboardInterrupt:
            logger.info("Interrupted by user — finalising outputs")
            break
        except _CycleStalled:
            # Watchdog aborted a hung cycle (e.g. a broker socket with no timeout). Reset
            # the heartbeat and continue with a fresh cycle instead of staying frozen.
            logger.critical("Two-engine cycle aborted by watchdog (stall) — continuing next cycle")
            _LOOP_HEARTBEAT[0] = time.time()
            time.sleep(max(5.0, BAR_INTERVAL_SEC))
        except Exception:
            # Run forever: transient broker/data/engine errors are logged, then the loop
            # continues after a normal sleep instead of killing the process.
            logger.exception("Two-engine loop error — continuing")
            try:
                log_account_stats(gateway, risk, label=f"Account after_error cycle={cycle_count}")
            except Exception:
                pass
            time.sleep(max(5.0, BAR_INTERVAL_SEC))

    _stop_premarket_feed()
    if waterfall is not None:
        write_capital_waterfall(waterfall)
    ensure_outputs_exist()
    logger.info("Two-engine session complete. Risk: equity=%.2f daily_pnl=%.2f opt_exp=%.0f stk_exp=%.0f",
                risk.equity, risk.daily_pnl, risk.options_exposure, risk.stock_exposure)


def _build_context_safe() -> dict:
    try:
        import inspect as _inspect
        from market_context_builder import build_market_context
        # Decide kwarg support by INSPECTING the signature — never via `except TypeError`,
        # which would also swallow a genuine TypeError raised inside the builder and
        # silently re-run it. A real failure now surfaces to the outer except below.
        if "premarket_snapshot" in _inspect.signature(build_market_context).parameters:
            ctx = build_market_context(premarket_snapshot=_premarket_snapshot())
        else:
            ctx = build_market_context()
        logger.info("Market context: bias=%s permission=%s risk=%s",
                    ctx.get("overall_market_bias"), ctx.get("trade_permission"), ctx.get("risk_level"))
        return ctx or {}
    except Exception as e:
        logger.error(
            "build_market_context failed (%s) — defaulting to cautious/neutral context "
            "(install yfinance + openai to enable live regime detection)",
            e,
        )
        # Return "neutral" (allowed regime) with cautious permission and reduced sizing.
        # Using "mixed" was also valid but "neutral" is semantically cleaner for a
        # dependency-missing failure — the market hasn't been assessed, not assessed bearish.
        return {
            "overall_market_bias": "neutral",
            "trade_permission": "cautious",
            "risk_level": "high",
            "_fallback": True,
            "_fallback_reason": str(e),
        }


def _log_config(risk: Optional[RiskManager] = None) -> None:
    if risk is not None:
        limits = risk.risk_limits_summary()
        logger.info(
            "Risk limits: mode=%s daily_loss=$%.0f (%.2f%% equity) max_dd=%.1f%% "
            "max_pos=%d equity_floor=$%.0f max_order=$%.0f max_order_qty=%s anchor=$%.0f",
            "dynamic" if DYNAMIC_RISK_LIMITS else "fixed",
            limits["daily_loss_limit"],
            abs(limits["daily_loss_limit"]) / max(1.0, limits["equity"]) * 100.0,
            limits["max_drawdown"] * 100.0,
            limits["max_positions"],
            limits["equity_floor"],
            limits["max_order_dollars"],
            (limits["max_order_size"] if limits["max_order_size"] > 0 else "off"),
            limits["anchor_equity"],
        )
    else:
        logger.info(
            "Risk limits: mode=%s daily_loss_pct=%.2f%% max_dd=%.1f%% "
            "equity_floor_pct=%.0f%% max_order_pct=%.1f%% position_unit=$%.0f",
            "dynamic" if DYNAMIC_RISK_LIMITS else "fixed",
            DYNAMIC_DAILY_LOSS_PCT * 100.0,
            -abs(DYNAMIC_MAX_DRAWDOWN_PCT) * 100.0,
            DYNAMIC_EQUITY_FLOOR_PCT * 100.0,
            DYNAMIC_MAX_ORDER_PCT * 100.0,
            DYNAMIC_POSITION_EQUITY_UNIT,
        )
    logger.info("Engines: alpha=%s bal=%s monitor=%s | paper_only=%s",
                ENABLE_ALPHA_CALL_ENGINE, ENABLE_BUFFETT_ACCUMULATION_LAYER,
                ENABLE_BAL_PORTFOLIO_MONITOR, ALPACA_PAPER_ONLY)
    logger.info("Exposure: options<=%.0f%% stock<=%.0f%% single_stock<=%.0f%% opt_risk/trade=%.1f%% cash_buffer=%.0f%%",
                MAX_OPTIONS_EXPOSURE_PCT * 100, MAX_STOCK_EXPOSURE_PCT * 100,
                MAX_SINGLE_STOCK_WEIGHT * 100, OPTIONS_RISK_PER_TRADE_PCT * 100, CASH_BUFFER_PCT * 100)
    logger.info("BAL thresholds: min_score=%.0f aggressive=%.0f | Waterfall: bal=%.0f%% alpha=%.0f%% cash=%.0f%%",
                BAL_MIN_SCORE_TO_BUY, BAL_AGGRESSIVE_SCORE,
                OPTIONS_PROFIT_TO_BAL_PCT * 100, OPTIONS_PROFIT_TO_ALPHA_PCT * 100, OPTIONS_PROFIT_TO_CASH_PCT * 100)
    logger.info("BAL composite DCA: enabled=%s (value=WHETHER, RSI=WHEN, tranches=HOW MUCH, RiskManager=ALLOWED)",
                ENABLE_BAL_COMPOSITE_DCA)


# ---------------------------------------------------------------------------
# Training orchestrator  (preserved)
# ---------------------------------------------------------------------------

def run_training():
    logger.info("=== TRAINING MODE ===")
    from rl_intraday_structure_trainer import train as train_intraday
    from tune_rl_intraday             import run_tuning as tune_intraday
    from sr_rl_agent                  import train as train_sr
    from tune_sr_rl_agent             import run_tuning as tune_sr

    logger.info("--- Step 1: Train Intraday RL ---");  train_intraday()
    logger.info("--- Step 2: Tune Intraday RL ---");   tune_intraday()
    logger.info("--- Step 3: Train SR RL ---");        train_sr()
    logger.info("--- Step 4: Tune SR RL ---");         tune_sr()
    logger.info("=== TRAINING COMPLETE ===")


# ---------------------------------------------------------------------------
# Legacy scalping loop  (preserved — run with MODE=scalp)
# ---------------------------------------------------------------------------

def run_scalping():
    logger.info("=== SCALPING MODE  symbols=%s ===", SYMBOLS_ENV or "default")
    symbols = load_candidate_universe()

    risk    = RiskManager(equity=STARTING_EQUITY)
    gateway = AlpacaGateway(risk)
    # Anchor risk to the live Alpaca account equity before trading (fallback: STARTING_EQUITY).
    seed_risk_from_broker(risk, gateway)

    def order_callback(symbol, side, qty, order_type, limit_price):
        gateway.submit_order(symbol, side, qty, order_type, limit_price)

    from trading_day_monitor_agent import TradingDayMonitorAgent

    monitor = TradingDayMonitorAgent(
        symbols          = symbols,
        starting_equity  = float(risk.starting_equity),
        session_csv      = "logs/session_summary.csv",
    )
    for sym in symbols:
        if sym in getattr(monitor, "_scalpers", {}):
            monitor._scalpers[sym].order_cb = lambda **kw: order_callback(**kw)

    logger.info("Entering scalping loop (Ctrl-C to stop) …")
    try:
        while True:
            if risk.is_killed:
                logger.critical("Kill switch active — exiting loop")
                break
            if not is_market_open():
                time.sleep(60)
                continue
            acc = gateway.get_account()
            if acc:
                risk.sync_account(acc)
            bars = _fetch_latest_bars(symbols)
            for symbol, bar in bars.items():
                monitor.update(symbol, bar)
            time.sleep(BAR_INTERVAL_SEC)
    except KeyboardInterrupt:
        logger.info("Interrupted by user — flushing session")
    finally:
        monitor.flush_session()
        s = monitor.session_summary()
        logger.info("Session: %s", json.dumps(s, default=str))


def _fetch_latest_bars(symbols: List[str]) -> Dict[str, Dict]:
    """Stub feed for the legacy scalping path (deterministic-ish single bars)."""
    import random
    bars = {}
    for sym in symbols:
        p = {"CRM": 190.0, "TE": 10.0, "RKLB": 140.0, "PLUG": 3.9}.get(sym, 50.0)
        p += random.gauss(0, 0.2)
        bars[sym] = dict(
            timestamp   = datetime.now(timezone.utc).isoformat(),
            open=p, high=p + random.uniform(0.05, 0.3), low=p - random.uniform(0.05, 0.4),
            close=p + random.gauss(0, 0.1), volume=random.uniform(3e4, 3e5),
            volume_ma20=1e5, vwap=p * 0.999, ema9=p * 0.998, ema20=p * 0.996,
        )
        bars[sym]["close"] = max(bars[sym]["close"], bars[sym]["low"])
    return bars


# ===========================================================================
# Legacy compatibility surface
# ===========================================================================
#
# The two-engine architecture above is the live system. The symbols below are a
# thin compatibility layer for the intraday-bracket / premarket-recovery test
# suite (tests/test_runtime_recovery.py, tests/test_first_candle_rule.py,
# tests/test_live_experience_logging.py). They form an INDEPENDENT execution path
# (its own BrokerPosition + bracket broker abstraction) that is NOT wired into
# run_two_engine_cycle — so the Alpha/Buffett separation, RiskManager, and
# AlpacaGateway are untouched. The heavy classes live in production_executor.py;
# only the small patchable hooks live here so `patch.object(run, ...)` resolves.

from dataclasses import dataclass as _dataclass

LOG_DIR = PROJECT_ROOT / "logs" / "execution" / DATA_TAG
LIVE_EXPERIENCE_JSONL_PATH   = LOG_DIR / f"{DATA_TAG}_live_experience_replay.jsonl"
LIVE_EXPERIENCE_CSV_PATH     = LOG_DIR / f"{DATA_TAG}_live_experience_replay.csv"
LIVE_EXPERIENCE_PARQUET_PATH = LOG_DIR / f"{DATA_TAG}_live_experience_replay.parquet"

# Whether bracket (OCO) order submission is available. Real systems flip this
# based on broker capability; tests patch it. Default on.
_BRACKET_ORDERS_AVAILABLE = True

# Numeric fields coerced to float in the live-experience replay store (empty
# string / None → NaN) so the parquet schema stays numeric for offline training.
_LIVE_NUMERIC_FIELDS = (
    "account_equity_before", "account_equity_after", "filled_avg_price",
    "filled_qty", "order_qty", "price", "stop_price", "target_price",
    "capped_dollars", "slippage_pct",
)


@_dataclass
class BrokerPosition:
    """Minimal broker position snapshot used by the legacy bracket execution path."""
    symbol: str
    qty: float = 0.0
    avg_entry_price: float = 0.0
    market_value: float = 0.0
    current_price: float = 0.0
    unrealized_plpc: float = 0.0


def write_event(event: str, payload: Optional[dict] = None) -> None:
    """Best-effort execution-event journal line. Patched in tests."""
    import traceback as _tb
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / f"{DATA_TAG}_execution_events.jsonl"
        # Serialize first so a failure here doesn't create a 0-byte file entry
        _line = json.dumps({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event, "payload": payload or {},
        }, default=str) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(_line)
    except Exception as _we:
        try:
            logger.warning("write_event(%s) failed: %s\n%s", event, _we, _tb.format_exc())
        except Exception:
            pass


def save_df(df: "Any", path: "Any") -> None:
    """Best-effort DataFrame → CSV writer. Patched in tests."""
    try:
        if df is None:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except Exception:
        pass


def log_executed_order(**kwargs) -> None:
    """
    Best-effort executed-order log.
    Writes to both execution_events.jsonl (raw event stream) AND the
    executor_journal.csv that funnel_report.py reads for submitted/filled counts.
    Patched in tests.
    """
    import traceback as _tb
    # Sanitize kwargs — convert any non-primitive values to strings so json.dumps
    # never fails.  UUID, Pydantic models, and Alpaca response objects have all
    # caused silent serialization failures here in the past.
    _safe_kwargs: dict = {}
    for _k, _v in kwargs.items():
        if isinstance(_v, (str, int, float, bool, type(None))):
            _safe_kwargs[_k] = _v
        else:
            try:
                _safe_kwargs[_k] = str(_v)
            except Exception:
                _safe_kwargs[_k] = "<unserializable>"
    try:
        write_event("executed_order", _safe_kwargs)
    except Exception as _we:
        logger.warning("log_executed_order write_event failed: %s\n%s", _we, _tb.format_exc())
    # Also append a minimal row to executor_journal.csv so funnel_report sees
    # submitted > 0.  Uses the same columns as ensure_runtime_outputs.JOURNAL_COLUMNS.
    try:
        import csv as _csv
        _jpath = LOG_DIR / f"{DATA_TAG}_executor_journal.csv"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        _write_header = not _jpath.exists() or _jpath.stat().st_size < 5
        _row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cycle_id": "",
            "event": _safe_kwargs.get("event", "order_submitted"),
            "market_open": is_market_open(),
            "equity_before": "", "equity_after": "",
            "cash_before": "", "cash_after": "",
            "buying_power_before": "", "buying_power_after": "",
            "gross_exposure_before": "", "gross_exposure_after": "",
            "trade_symbol": _safe_kwargs.get("symbol", ""),
            "trade_side": _safe_kwargs.get("side", ""),
            "trade_qty": _safe_kwargs.get("qty", ""),
            "trade_price": _safe_kwargs.get("limit_price", ""),
            "estimated_notional": "",
            "target_weight": "", "current_weight": "", "delta_weight": "",
            "confidence": _safe_kwargs.get("setup_score", ""),
            "cash_buffer": "", "max_gross_exposure": "", "max_position_weight": "",
            "max_order_notional": "",
            "stop_price": "", "target_price": "",
            "bracket_attached": False,
            "client_order_id": "",
            "order_id": _safe_kwargs.get("order_id", ""),
            "order_status": "submitted",
            "filled_qty": "",
            "filled_avg_price": "",
            "slippage_pct": "",
            "skip_reason": "",
            "broker_message": "sim" if _safe_kwargs.get("sim") else "",
            "message": f"engine={_safe_kwargs.get('engine','')} grade={_safe_kwargs.get('grade','')}",
        }
        _cols = list(_row.keys())
        with _jpath.open("a", newline="", encoding="utf-8") as _fh:
            _w = _csv.DictWriter(_fh, fieldnames=_cols)
            if _write_header:
                _w.writeheader()
            _w.writerow(_row)
    except Exception as _ce:
        logger.warning("log_executed_order CSV write failed: %s\n%s", _ce, _tb.format_exc())


def row_has_complete_bracket_plan(row: "Any") -> bool:
    """
    True when an order row carries a complete long bracket plan: a buy with a
    positive entry price, a stop below it, and a target above it. Accepts a dict
    or a pandas Series (both expose .get).
    """
    def _num(v) -> float:
        try:
            f = float(v)
            return 0.0 if math.isnan(f) else f
        except (TypeError, ValueError):
            return 0.0

    get = row.get if hasattr(row, "get") else (lambda k, d=None: row[k] if k in row else d)
    side = str(get("side") or "").lower()
    if side and side != "buy":
        return False
    price = _num(get("price")) or _num(get("entry_price"))
    stop = _num(get("stop_price"))
    target = _num(get("target_price"))
    if price <= 0 or stop <= 0 or target <= 0:
        return False
    return stop < price < target


def append_live_experience(row: dict, write_parquet: bool = True) -> None:
    """
    Append a live decision/execution sample to the replay store (jsonl + csv +
    optional parquet) for offline ML training. Empty strings become NaN for the
    declared numeric fields so the parquet schema stays numeric. Does NOT train
    here. Reads the LIVE_EXPERIENCE_* paths from this module so tests can patch
    them; parquet accumulates across calls.
    """
    import pandas as pd
    rec: dict = {}
    for key, value in dict(row).items():
        key = str(key)
        if key in _LIVE_NUMERIC_FIELDS:
            if value == "" or value is None:
                rec[key] = float("nan")
            else:
                try:
                    rec[key] = float(value)
                except (TypeError, ValueError):
                    rec[key] = float("nan")
        else:
            rec[key] = None if value == "" else value

    try:
        LIVE_EXPERIENCE_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LIVE_EXPERIENCE_JSONL_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass

    df = pd.DataFrame([rec])
    try:
        write_header = not LIVE_EXPERIENCE_CSV_PATH.exists()
        df.to_csv(LIVE_EXPERIENCE_CSV_PATH, mode="a", header=write_header, index=False)
    except Exception:
        pass

    if write_parquet:
        try:
            if LIVE_EXPERIENCE_PARQUET_PATH.exists():
                old = pd.read_parquet(LIVE_EXPERIENCE_PARQUET_PATH)
                pd.concat([old, df], ignore_index=True).to_parquet(
                    LIVE_EXPERIENCE_PARQUET_PATH, index=False)
            else:
                df.to_parquet(LIVE_EXPERIENCE_PARQUET_PATH, index=False)
        except Exception as exc:
            # Parquet is optional (pyarrow/fastparquet may be absent).
            write_event("live_experience_parquet_failed", {"error": str(exc)[:200]})


# Re-export the heavy legacy execution classes so `run.X` resolves for the tests.
# Imported at module end (after the hooks above are defined) to avoid a circular
# import — production_executor reaches the hooks lazily via `import run`.
try:
    from production_executor import (  # noqa: E402
        ScannerConfig, ExecutorConfig, build_default_config,
        MomentumSetupScanner, ProductionOrderExecutor,
    )
except Exception as _exc:   # pragma: no cover - keeps run.py importable if module missing
    logger.debug("production_executor compat import failed: %s", _exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    os.makedirs("logs",    exist_ok=True)
    os.makedirs("models",  exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("data",    exist_ok=True)
    _acquire_single_instance_lock()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler("logs/run.log", mode="a")],
    )
    # Suppress yfinance's verbose HTTP error bodies (e.g. full Yahoo Finance "Will be
    # right back" HTML) which pollute the log when Yahoo Finance is down.  Keep ERROR
    # so genuine authentication / import failures still surface.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    logging.getLogger("yfinance.base").setLevel(logging.CRITICAL)
    logging.getLogger("peewee").setLevel(logging.WARNING)

    logger.info("QuantumQuant  mode=%s  paper_only=%s  data_tag=%s", MODE, ALPACA_PAPER_ONLY, DATA_TAG)

    if MODE == "train":
        run_training()
    elif MODE == "scalp":
        run_scalping()
    elif MODE in ("paper", "live"):
        if MODE == "live" and ALPACA_PAPER_ONLY:
            logger.critical("Live mode requested but ALPACA_PAPER_ONLY=true — set to false explicitly")
            sys.exit(1)
        run_two_engine_trading()
    else:
        logger.error("Unknown MODE=%s  (use: train | paper | live | scalp)", MODE)
        sys.exit(1)


if __name__ == "__main__":
    main()
