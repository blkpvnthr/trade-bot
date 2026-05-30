#!/usr/bin/env python3
"""
buffett_accumulation_layer.py

Strategic long-stock engine using Buffett-style fundamental analysis.

BAL score (0-100):
    1. Moat quality          0-15
    2. Financial strength    0-20
    3. Management quality    0-10
    4. Valuation             0-20
    5. Macro alignment       0-10
    6. Economic policy       0-10
    7. 18-year cycle phase   0-10
    8. Technical entry       0-5

Actions:
    85-100: aggressive_accumulate  (up to max_single_stock_weight)
    70-84:  accumulate             (~4% of equity)
    55-69:  starter                (~2% of equity)
    <55:    no_action

Stock is held indefinitely unless thesis, trend, concentration, or risk breaks.

Outputs:
    daily_system_outputs/buffett_accumulation_scores.json
    daily_system_outputs/ownership_universe.json
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
DATA_TAG = os.getenv("RUN_DATA_DATE", datetime.now(ET).strftime("%Y%m%d"))
YESTERDAY_TAG = (datetime.now(ET) - timedelta(days=1)).strftime("%Y%m%d")
DAILY_OUTPUTS = PROJECT_ROOT / "daily_system_outputs"
SYSTEM_DATA = PROJECT_ROOT / "system-data" / DATA_TAG
DAILY_OUTPUTS.mkdir(parents=True, exist_ok=True)
SYSTEM_DATA.mkdir(parents=True, exist_ok=True)

BAL_SCORES_PATH = DAILY_OUTPUTS / "buffett_accumulation_scores.json"
OWNERSHIP_UNIVERSE_PATH = DAILY_OUTPUTS / "ownership_universe.json"
BUFFETT_DCA_SIGNALS_PATH = DAILY_OUTPUTS / "buffett_dca_signals.json"
VALUATION_DIAGNOSTIC_PATH = DAILY_OUTPUTS / "valuation_diagnostic.json"
CONVICTION_REPORT_PATH = DAILY_OUTPUTS / "conviction_score_report.json"
BAL_EVENTS_PATH = PROJECT_ROOT / "logs" / "execution" / DATA_TAG / f"{DATA_TAG}_bal_events.jsonl"
BAL_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── configuration ─────────────────────────────────────────────────────────────
@dataclass
class BALConfig:
    enabled: bool = os.getenv("ENABLE_BUFFETT_ACCUMULATION_LAYER", "true").lower() in {"1", "true"}
    min_score_to_buy: float = float(os.getenv("BAL_MIN_SCORE_TO_BUY", "55"))
    aggressive_score: float = float(os.getenv("BAL_AGGRESSIVE_SCORE", "85"))
    normal_score: float = float(os.getenv("BAL_NORMAL_SCORE", "70"))
    max_single_stock_weight: float = float(os.getenv("MAX_SINGLE_STOCK_WEIGHT", "0.10"))
    max_stock_exposure_pct: float = float(os.getenv("MAX_STOCK_EXPOSURE_PCT", "0.80"))
    fundamental_cache_hours: int = int(os.getenv("BAL_FUNDAMENTAL_CACHE_HOURS", "6"))
    min_market_cap_m: float = float(os.getenv("BAL_MIN_MARKET_CAP_M", "500"))
    # Thesis break: exit if score collapses below this
    exit_score_threshold: float = float(os.getenv("BAL_EXIT_SCORE", "30"))
    # Trend break: exit if price drops this far below cost basis
    exit_drawdown_pct: float = float(os.getenv("BAL_EXIT_DRAWDOWN", "0.35"))
    scan_interval_minutes: int = int(os.getenv("BAL_SCAN_INTERVAL_MINUTES", "30"))


@dataclass
class DCAConfig:
    """
    Composite DCA framework configuration.

    Core principle:
        Value decides WHETHER to DCA  (intrinsic-value discount + business quality)
        RSI helps decide WHEN to DCA  (technical dislocation only — never alone)
        Tranches decide HOW MUCH to DCA
        RiskManager decides whether the order is ALLOWED  (gated in run.py)
    """
    enabled: bool = os.getenv("ENABLE_BAL_COMPOSITE_DCA", "true").lower() in {"1", "true"}

    # Composite score component weights (sum to 1.0; normalised in __post_init__).
    valuation_weight: float = float(os.getenv("BAL_DCA_VALUATION_WEIGHT", "0.35"))
    technical_weight: float = float(os.getenv("BAL_DCA_TECHNICAL_WEIGHT", "0.25"))
    business_quality_weight: float = float(os.getenv("BAL_DCA_BUSINESS_QUALITY_WEIGHT", "0.20"))
    macro_cycle_weight: float = float(os.getenv("BAL_DCA_MACRO_CYCLE_WEIGHT", "0.10"))
    sentiment_weight: float = float(os.getenv("BAL_DCA_SENTIMENT_WEIGHT", "0.10"))

    # Composite-score action thresholds (0-100).
    small_score: float = float(os.getenv("BAL_DCA_SMALL_SCORE", "55"))
    normal_score: float = float(os.getenv("BAL_DCA_NORMAL_SCORE", "70"))
    aggressive_score: float = float(os.getenv("BAL_DCA_AGGRESSIVE_SCORE", "85"))

    # Valuation discount tiers: price / intrinsic_value ceilings that unlock each
    # DCA aggressiveness level (smaller ratio = deeper discount).
    first_discount: float = float(os.getenv("BAL_DCA_FIRST_DISCOUNT", "0.95"))    # small
    deep_discount: float = float(os.getenv("BAL_DCA_DEEP_DISCOUNT", "0.85"))      # normal
    panic_discount: float = float(os.getenv("BAL_DCA_PANIC_DISCOUNT", "0.75"))    # aggressive
    extreme_discount: float = float(os.getenv("BAL_DCA_EXTREME_DISCOUNT", "0.65"))  # maximum

    # Tranche controls.
    tranche_pct: float = float(os.getenv("BAL_DCA_TRANCHE_PCT", "0.20"))          # 20% of target per tranche
    min_days_between_dca: int = int(os.getenv("BAL_MIN_DAYS_BETWEEN_DCA", "3"))
    max_dca_count_per_symbol: int = int(os.getenv("BAL_MAX_DCA_COUNT_PER_SYMBOL", "5"))

    # Business-quality gate (mirrors BAL thresholds).
    min_score_to_buy: float = float(os.getenv("BAL_MIN_SCORE_TO_BUY", "55"))
    bal_aggressive_score: float = float(os.getenv("BAL_AGGRESSIVE_SCORE", "85"))
    thesis_break_score: float = float(os.getenv("BAL_EXIT_SCORE", "30"))

    # Target allocation per holding (fraction of equity the 5 tranches fill toward).
    max_single_stock_weight: float = float(os.getenv("MAX_SINGLE_STOCK_WEIGHT", "0.10"))
    min_dca_order_dollars: float = float(os.getenv("BAL_DCA_MIN_ORDER_DOLLARS", "200"))

    # ── Conviction → target allocation tiers ────────────────────────────────────
    # conviction_score (0-100) determines target_position_pct as a fraction of the
    # max_single_stock_weight cap. Below the watchlist floor → no DCA (watchlist only).
    conviction_large_score: float = float(os.getenv("BAL_CONVICTION_LARGE", "90"))
    conviction_normal_score: float = float(os.getenv("BAL_CONVICTION_NORMAL", "75"))
    conviction_small_score: float = float(os.getenv("BAL_CONVICTION_SMALL", "60"))
    conviction_large_frac: float = float(os.getenv("BAL_CONVICTION_LARGE_FRAC", "1.00"))
    conviction_normal_frac: float = float(os.getenv("BAL_CONVICTION_NORMAL_FRAC", "0.70"))
    conviction_small_frac: float = float(os.getenv("BAL_CONVICTION_SMALL_FRAC", "0.45"))

    # ── Insider purchase modifier (modifier only — never an automatic trigger) ──
    insider_min_amount: float = float(os.getenv("BAL_INSIDER_MIN_AMOUNT", "100000"))     # $ to register
    insider_large_amount: float = float(os.getenv("BAL_INSIDER_LARGE_AMOUNT", "1000000"))  # $ = strong
    insider_undervalued_p2iv: float = float(os.getenv("BAL_INSIDER_MAX_P2IV", "1.00"))   # must be undervalued to count

    def __post_init__(self) -> None:
        total = (self.valuation_weight + self.technical_weight + self.business_quality_weight
                 + self.macro_cycle_weight + self.sentiment_weight)
        if total > 0 and abs(total - 1.0) > 1e-6:
            self.valuation_weight /= total
            self.technical_weight /= total
            self.business_quality_weight /= total
            self.macro_cycle_weight /= total
            self.sentiment_weight /= total


# ── data structures ────────────────────────────────────────────────────────────
@dataclass
class FundamentalData:
    symbol: str
    market_cap: float = 0.0
    revenue_ttm: float = 0.0
    revenue_growth_yoy: float = 0.0
    eps_ttm: float = 0.0
    eps_growth_yoy: float = 0.0
    gross_margin: float = 0.0
    operating_margin: float = 0.0
    net_margin: float = 0.0
    roe: float = 0.0
    roic: float = 0.0
    free_cash_flow: float = 0.0
    fcf_yield: float = 0.0
    debt_to_equity: float = 999.0
    current_ratio: float = 0.0
    shares_growth_yoy: float = 0.0
    pe_ratio: float = 0.0
    peg_ratio: float = 0.0
    ev_ebitda: float = 0.0
    price_to_book: float = 0.0
    current_price: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    fifty_two_week_low: float = 0.0
    fifty_two_week_high: float = 0.0
    beta: float = 1.0
    # Daily-timeframe technicals (computed from 1y history) — feed the DCA technical
    # dislocation score. RSI is a TIMING input only; it can never trigger DCA alone.
    rsi_14: float = 50.0
    atr_pct: float = 0.0            # ATR(14) / price
    atr_expansion: float = 1.0      # recent ATR(14) / ~3-month average ATR
    volume_capitulation: float = 1.0  # recent 5d avg volume / ~3-month avg volume
    insider_ownership_pct: float = 0.0
    dividend_yield: float = 0.0
    payout_ratio: float = 0.0
    sector: str = ""
    industry: str = ""
    fetched_at_utc: str = ""
    fetch_error: str = ""


@dataclass
class BALScorecard:
    symbol: str
    moat_score: float = 0.0         # 0-15
    financial_score: float = 0.0    # 0-20
    management_score: float = 0.0   # 0-10
    valuation_score: float = 0.0    # 0-20
    macro_score: float = 0.0        # 0-10
    policy_score: float = 0.0       # 0-10
    cycle_score: float = 0.0        # 0-10
    technical_score: float = 0.0    # 0-5
    bal_score: float = 0.0          # 0-100
    action: str = "no_action"
    position_size_pct: float = 0.0
    skip_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    scored_at_utc: str = ""


@dataclass
class DCASignal:
    """One symbol's composite DCA evaluation. Serialised to buffett_dca_signals.json."""
    symbol: str
    timestamp: str = ""
    price: float = 0.0
    intrinsic_value: float = 0.0
    price_to_intrinsic_value: float = 0.0
    BAL_score: float = 0.0
    composite_dca_score: float = 0.0
    valuation_discount_score: float = 0.0
    technical_dislocation_score: float = 0.0
    business_quality_score: float = 0.0
    macro_cycle_opportunity_score: float = 0.0
    sentiment_capitulation_score: float = 0.0
    RSI: float = 50.0
    distance_from_50DMA: float = 0.0
    distance_from_200DMA: float = 0.0
    ATR_expansion: float = 1.0
    volume_capitulation: float = 1.0
    target_position_pct: float = 0.0
    current_position_pct: float = 0.0
    remaining_dca_tranches: int = 0
    DCA_action: str = "no_dca"          # no_dca | small | normal | aggressive
    DCA_tranche_size: float = 0.0       # dollars intended for this tranche
    tranche_size_pct: float = 0.0       # tranche dollars as a fraction of equity
    reject_reason: str = ""
    risk_limits_passed: bool = False
    journal_reason: str = ""            # canonical reason code for the journal
    dca_reason: str = ""                # human-readable decision summary
    # ── multi-model valuation diagnostics ──────────────────────────────────────
    valuation_model_values: dict = field(default_factory=dict)
    valuation_model_count: int = 0
    valuation_confidence: float = 0.0
    valuation_dispersion: float = 0.0
    primary_valuation_method: str = "none"
    # ── conviction + allocation ─────────────────────────────────────────────────
    conviction_score: float = 0.0
    target_allocation_pct: float = 0.0
    max_allocation_pct: float = 0.0
    allocation_reason: str = ""
    profitability_score: float = 0.0
    earnings_quality_score: float = 0.0
    # ── insider purchase modifier ───────────────────────────────────────────────
    insider_buy_detected: bool = False
    insider_buy_score: float = 0.0
    insider_buy_amount: float = 0.0
    insider_buy_role: str = ""
    insider_buy_confidence: float = 0.0
    insider_buy_used_in_conviction: bool = False
    insider_buy_rejected_reason: str = ""


# ── event logging ──────────────────────────────────────────────────────────────
def _bal_event(event: str, payload: dict) -> None:
    try:
        line = json.dumps({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
        })
        with open(BAL_EVENTS_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ── fundamental data ──────────────────────────────────────────────────────────

def _fetch_fundamentals(symbol: str) -> FundamentalData:
    fd = FundamentalData(symbol=symbol, fetched_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        def _f(key: str, default: float = 0.0) -> float:
            try:
                return float(info.get(key) or default)
            except Exception:
                return default

        fd.market_cap = _f("marketCap")
        fd.current_price = _f("currentPrice") or _f("regularMarketPrice")
        fd.revenue_ttm = _f("totalRevenue")
        fd.gross_margin = _f("grossMargins")
        fd.operating_margin = _f("operatingMargins")
        fd.net_margin = _f("profitMargins")
        fd.roe = _f("returnOnEquity")
        fd.roic = _f("returnOnAssets") * 1.8   # rough ROIC proxy
        fd.eps_ttm = _f("trailingEps")
        fd.eps_growth_yoy = _f("earningsGrowth") or _f("earningsQuarterlyGrowth")
        fd.revenue_growth_yoy = _f("revenueGrowth")
        fd.free_cash_flow = _f("freeCashflow")
        fd.debt_to_equity = _f("debtToEquity", 999) / 100  # yfinance returns pct
        fd.current_ratio = _f("currentRatio")
        fd.pe_ratio = _f("trailingPE") or _f("forwardPE")
        fd.peg_ratio = _f("pegRatio")
        fd.ev_ebitda = _f("enterpriseToEbitda")
        fd.price_to_book = _f("priceToBook")
        fd.beta = _f("beta", 1.0)
        fd.insider_ownership_pct = _f("heldPercentInsiders")
        fd.dividend_yield = _f("dividendYield")
        fd.payout_ratio = _f("payoutRatio")
        fd.fifty_two_week_low = _f("fiftyTwoWeekLow")
        fd.fifty_two_week_high = _f("fiftyTwoWeekHigh")
        fd.sector = str(info.get("sector") or "")
        fd.industry = str(info.get("industry") or "")

        if fd.market_cap > 0 and fd.free_cash_flow != 0:
            fd.fcf_yield = fd.free_cash_flow / fd.market_cap

        try:
            hist = ticker.history(period="1y", interval="1d")
            if not hist.empty:
                closes = hist["Close"]
                fd.sma_50 = float(closes.tail(50).mean()) if len(closes) >= 50 else 0.0
                fd.sma_200 = float(closes.tail(200).mean()) if len(closes) >= 200 else \
                    (float(closes.mean()) if len(closes) >= 100 else 0.0)
                _compute_daily_technicals(fd, hist)
        except Exception:
            pass

    except Exception as exc:
        fd.fetch_error = str(exc)[:200]
    return fd


def _cache_path(symbol: str) -> Path:
    return SYSTEM_DATA / f"bal_fd_{symbol}.json"


def get_fundamentals(symbol: str, config: BALConfig) -> FundamentalData:
    path = _cache_path(symbol)
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            fetched = datetime.fromisoformat(raw.get("fetched_at_utc", "2000-01-01T00:00:00+00:00"))
            age_h = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600
            if age_h < config.fundamental_cache_hours:
                return FundamentalData(**{
                    k: raw[k] for k in FundamentalData.__dataclass_fields__ if k in raw
                })
        except Exception:
            pass
    fd = _fetch_fundamentals(symbol)
    try:
        path.write_text(json.dumps(fd.__dict__, default=str), encoding="utf-8")
    except Exception:
        pass
    return fd


def _compute_daily_technicals(fd: FundamentalData, hist: "pd.DataFrame") -> None:
    """
    Populate fd.rsi_14 / atr_pct / atr_expansion / volume_capitulation from a daily
    OHLCV history DataFrame (columns: Open/High/Low/Close/Volume). Best-effort: any
    failure leaves the neutral defaults intact. These are DCA *timing* inputs only.
    """
    try:
        closes = hist["Close"].astype(float)
        highs = hist["High"].astype(float)
        lows = hist["Low"].astype(float)
        vols = hist["Volume"].astype(float)
        price = fd.current_price or float(closes.iloc[-1])

        # RSI(14), Wilder smoothing
        if len(closes) >= 15:
            delta = closes.diff()
            gain = delta.clip(lower=0.0)
            loss = (-delta).clip(lower=0.0)
            avg_gain = gain.ewm(alpha=1 / 14.0, min_periods=14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / 14.0, min_periods=14, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0.0, np.nan)
            rsi = 100.0 - 100.0 / (1.0 + rs)
            last_rsi = rsi.iloc[-1]
            if pd.notna(last_rsi):
                fd.rsi_14 = float(last_rsi)
            elif pd.notna(avg_loss.iloc[-1]) and avg_loss.iloc[-1] == 0:
                fd.rsi_14 = 100.0  # no losses over window

        # ATR(14) + expansion (recent vs ~3-month average)
        if len(closes) >= 15:
            prev_close = closes.shift(1)
            tr = pd.concat([
                (highs - lows),
                (highs - prev_close).abs(),
                (lows - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1 / 14.0, min_periods=14, adjust=False).mean()
            last_atr = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else 0.0
            if price > 0 and last_atr > 0:
                fd.atr_pct = last_atr / price
            long_atr = float(atr.tail(60).mean()) if len(atr.dropna()) >= 30 else last_atr
            if long_atr > 0 and last_atr > 0:
                fd.atr_expansion = last_atr / long_atr

        # Volume capitulation: recent 5d avg vs ~3-month avg
        if len(vols) >= 10:
            recent_v = float(vols.tail(5).mean())
            base_v = float(vols.tail(60).mean()) if len(vols) >= 60 else float(vols.mean())
            if base_v > 0:
                fd.volume_capitulation = recent_v / base_v
    except Exception:
        pass


# Conservative sector-baseline P/E multiples for the sector-relative model. A
# market_context["sector_pe"][sector] override takes precedence when supplied.
_SECTOR_PE = {
    "technology": 24.0, "communication services": 19.0, "consumer cyclical": 20.0,
    "consumer discretionary": 20.0, "consumer defensive": 19.0, "consumer staples": 19.0,
    "healthcare": 18.0, "financial services": 13.0, "financials": 13.0,
    "industrials": 18.0, "energy": 11.0, "utilities": 16.0, "materials": 14.0,
    "real estate": 16.0, "basic materials": 14.0,
}
# Normalized "fair" multiples for the historical-multiple models (used as a proxy
# when a true historical multiple series is unavailable — conservative anchors).
_FAIR_EV_FCF_MULTIPLE = 18.0
_FAIR_EV_EBIT_MULTIPLE = 14.0
_MIN_VALUATION_MODELS = 3   # below this, confidence is reduced and we fall back


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def _discount_and_growth(fd: FundamentalData) -> tuple[float, float]:
    """Beta-scaled discount rate (8-13%) and conservative clamped growth (-5%..15%)."""
    disc = 0.09 + 0.03 * max(0.0, min(float(fd.beta or 1.0), 2.0) - 1.0)
    disc = min(max(disc, 0.08), 0.13)
    g_raw = fd.eps_growth_yoy or fd.revenue_growth_yoy or 0.0
    g = max(-0.05, min(float(g_raw), 0.15))
    return disc, g


def _net_debt_per_share(fd: FundamentalData, shares: float) -> Optional[float]:
    """
    Estimate net debt per share from D/E and book equity (book equity ≈ market_cap /
    price_to_book). Cash is ignored (conservative: overstates net debt → understates
    equity value). Returns None when book equity can't be derived (so EV-based models
    are skipped rather than fabricated).
    """
    if shares <= 0 or fd.price_to_book <= 0 or fd.market_cap <= 0:
        return None
    book_equity = fd.market_cap / fd.price_to_book
    if book_equity <= 0:
        return None
    total_debt = max(0.0, float(fd.debt_to_equity)) * book_equity
    return total_debt / shares


def multi_model_intrinsic_value(fd: FundamentalData, market_context: dict | None = None) -> dict:
    """
    Multi-model intrinsic value. Computes every model for which data is available,
    takes the MEDIAN of valid model values as the intrinsic value, and reports
    confidence + dispersion diagnostics. Never fabricates values: a model is only
    included when its inputs are present and produce a positive estimate.

    Models (per share):
      1. dcf                – 5-yr two-stage owner-earnings DCF + terminal value
      2. owner_earnings     – FCF perpetuity (Gordon, conservative denom floor)
      3. graham             – Graham earnings power, eps * (8.5 + 0.5·growth%)
      4. ev_fcf_multiple    – fair EV/FCF · FCF − net debt
      5. ev_ebit_multiple   – fair EV/EBIT · EBIT − net debt
      6. peg                – eps · clamp(growth%) at PEG≈1
      7. sector_relative    – eps · sector-baseline P/E

    Returns a dict with intrinsic_value, valuation_model_values, valuation_model_count,
    valuation_confidence (0-1), valuation_dispersion, primary_valuation_method, and the
    discount/growth used. When fewer than _MIN_VALUATION_MODELS are available it falls
    back to the owner-earnings/Graham blend and reduces confidence (does not fabricate).
    """
    mc = market_context or {}
    price = fd.current_price
    if price <= 0:
        return {"intrinsic_value": 0.0, "valuation_model_values": {}, "valuation_model_count": 0,
                "valuation_confidence": 0.0, "valuation_dispersion": 0.0,
                "primary_valuation_method": "none", "reason": "no_price"}

    shares = (fd.market_cap / price) if (fd.market_cap > 0 and price > 0) else 0.0
    disc, g = _discount_and_growth(fd)
    models: dict[str, float] = {}

    # 1. DCF — two-stage owner-earnings DCF
    if shares > 0 and fd.free_cash_flow > 0:
        fcf_ps = fd.free_cash_flow / shares
        g1 = min(g, 0.12)            # stage-1 growth (5 yrs), capped
        gt = 0.025                   # terminal growth
        pv, cf = 0.0, fcf_ps
        for yr in range(1, 6):
            cf *= (1.0 + g1)
            pv += cf / ((1.0 + disc) ** yr)
        terminal = cf * (1.0 + gt) / max(disc - gt, 0.04)
        pv += terminal / ((1.0 + disc) ** 5)
        if pv > 0:
            models["dcf"] = pv

    # 2. owner earnings (single-stage perpetuity, conservative)
    if shares > 0 and fd.free_cash_flow > 0:
        fcf_ps = fd.free_cash_flow / shares
        g_perp = min(g, 0.04)
        iv = fcf_ps * (1.0 + g_perp) / max(disc - g_perp, 0.06)
        if iv > 0:
            models["owner_earnings"] = iv

    # 3. Graham earnings power
    if fd.eps_ttm > 0:
        fair_pe = max(8.0, min(8.5 + g * 100.0 * 0.5, 20.0))
        iv = fd.eps_ttm * fair_pe
        if iv > 0:
            models["graham"] = iv

    # 4. EV/FCF historical-multiple proxy  (needs net debt)
    ndps = _net_debt_per_share(fd, shares)
    if shares > 0 and fd.free_cash_flow > 0 and ndps is not None:
        fcf_ps = fd.free_cash_flow / shares
        ev_fair = _FAIR_EV_FCF_MULTIPLE * fcf_ps
        iv = ev_fair - ndps
        if iv > 0:
            models["ev_fcf_multiple"] = iv

    # 5. EV/EBIT historical-multiple proxy  (EBIT ≈ operating_margin · revenue; needs net debt)
    if shares > 0 and ndps is not None and fd.operating_margin > 0 and fd.revenue_ttm > 0:
        ebit_ps = (fd.operating_margin * fd.revenue_ttm) / shares
        ev_fair = _FAIR_EV_EBIT_MULTIPLE * ebit_ps
        iv = ev_fair - ndps
        if iv > 0:
            models["ev_ebit_multiple"] = iv

    # 6. PEG-based growth valuation (fair P/E ≈ growth% at PEG 1, capped)
    growth_pct = (fd.eps_growth_yoy or fd.revenue_growth_yoy or 0.0) * 100.0
    if fd.eps_ttm > 0 and growth_pct > 0:
        fair_pe = max(8.0, min(growth_pct, 30.0))
        iv = fd.eps_ttm * fair_pe
        if iv > 0:
            models["peg"] = iv

    # 7. sector-relative valuation (sector-baseline P/E)
    sector_key = str(fd.sector or "").strip().lower()
    sector_pe_override = (mc.get("sector_pe") or {}).get(sector_key) if isinstance(mc.get("sector_pe"), dict) else None
    sector_pe = sector_pe_override if sector_pe_override else _SECTOR_PE.get(sector_key)
    if fd.eps_ttm > 0 and sector_pe:
        iv = fd.eps_ttm * float(sector_pe)
        if iv > 0:
            models["sector_relative"] = iv

    values = [v for v in models.values() if v > 0]
    count = len(values)
    detail: dict = {
        "valuation_model_values": {k: round(v, 2) for k, v in models.items()},
        "valuation_model_count": count,
        "discount_rate": round(disc, 4),
        "growth_used": round(g, 4),
    }

    if count == 0:
        detail.update({"intrinsic_value": 0.0, "valuation_confidence": 0.0,
                       "valuation_dispersion": 0.0, "primary_valuation_method": "none"})
        return detail

    median_iv = _median(values)

    # Dispersion = coefficient of variation (std / mean); confidence blends model
    # count with cluster tightness.
    mean_v = sum(values) / count
    var = sum((v - mean_v) ** 2 for v in values) / count
    std = var ** 0.5
    cv = (std / mean_v) if mean_v > 0 else 1.0
    count_factor = min(count / 5.0, 1.0)
    dispersion_factor = _clamp(1.0 - cv, 0.3, 1.0) if isinstance(cv, float) else 1.0
    confidence = count_factor * dispersion_factor

    fallback_used = False
    if count < _MIN_VALUATION_MODELS:
        # Too few models: fall back to the owner-earnings/Graham blend and halve confidence.
        blend = [models[k] for k in ("owner_earnings", "graham") if k in models]
        if blend:
            median_iv = sum(blend) / len(blend)
        confidence *= 0.5
        fallback_used = True

    # Primary method = the model closest to the (final) intrinsic value.
    primary = min(models.items(), key=lambda kv: abs(kv[1] - median_iv))[0] if models else "none"

    detail.update({
        "intrinsic_value": round(median_iv, 2),
        "valuation_confidence": round(confidence, 3),
        "valuation_dispersion": round(cv, 3),
        "primary_valuation_method": primary,
        "valuation_fallback_used": fallback_used,
    })
    return detail


def estimate_intrinsic_value(fd: FundamentalData, market_context: dict | None = None) -> tuple[float, dict]:
    """
    Backwards-compatible wrapper around the multi-model valuation engine.
    Returns (intrinsic_value, detail). A non-positive intrinsic value means
    "valuation unknown" (no DCA on valuation).
    """
    detail = multi_model_intrinsic_value(fd, market_context)
    return float(detail.get("intrinsic_value", 0.0) or 0.0), detail


# ── scoring components ─────────────────────────────────────────────────────────

def _score_moat(fd: FundamentalData) -> tuple[float, list[str]]:
    """0-15: quantitative proxies for competitive moat."""
    score, notes = 0.0, []

    # Gross margin → pricing power
    if fd.gross_margin >= 0.60:
        score += 4.0; notes.append(f"gross_margin={fd.gross_margin:.0%} exceptional(+4)")
    elif fd.gross_margin >= 0.40:
        score += 2.5; notes.append(f"gross_margin={fd.gross_margin:.0%} strong(+2.5)")
    elif fd.gross_margin >= 0.25:
        score += 1.0
    elif fd.gross_margin > 0:
        notes.append(f"gross_margin={fd.gross_margin:.0%} thin")

    # Revenue growth consistency → sticky demand
    if fd.revenue_growth_yoy >= 0.15:
        score += 3.0; notes.append(f"rev_growth={fd.revenue_growth_yoy:.0%}(+3)")
    elif fd.revenue_growth_yoy >= 0.08:
        score += 2.0
    elif fd.revenue_growth_yoy >= 0.02:
        score += 0.5
    elif fd.revenue_growth_yoy < 0:
        score -= 1.0; notes.append(f"rev_declining={fd.revenue_growth_yoy:.0%}(-1)")

    # ROE → sustained capital efficiency moat
    if fd.roe >= 0.25:
        score += 4.0; notes.append(f"ROE={fd.roe:.0%} exceptional(+4)")
    elif fd.roe >= 0.15:
        score += 2.5; notes.append(f"ROE={fd.roe:.0%} strong(+2.5)")
    elif fd.roe >= 0.10:
        score += 1.0
    else:
        notes.append(f"ROE={fd.roe:.0%} weak")

    # Scale = market leadership proxy
    if fd.market_cap >= 50e9:
        score += 2.0; notes.append(f"mcap=${fd.market_cap/1e9:.0f}B large-cap leader(+2)")
    elif fd.market_cap >= 10e9:
        score += 1.0
    elif fd.market_cap >= 1e9:
        score += 0.5

    # Operating margin
    if fd.operating_margin >= 0.25:
        score += 2.0; notes.append(f"op_margin={fd.operating_margin:.0%} strong(+2)")
    elif fd.operating_margin >= 0.15:
        score += 1.0

    return min(score, 15.0), notes


def _score_financial_strength(fd: FundamentalData) -> tuple[float, list[str]]:
    """0-20: financial quality."""
    score, notes = 0.0, []

    if fd.free_cash_flow > 0:
        score += 3.0; notes.append(f"FCF=${fd.free_cash_flow/1e6:.0f}M positive(+3)")
    else:
        notes.append(f"FCF=${fd.free_cash_flow/1e6:.0f}M negative")

    if fd.fcf_yield >= 0.06:
        score += 3.0; notes.append(f"FCF_yield={fd.fcf_yield:.1%}(+3)")
    elif fd.fcf_yield >= 0.03:
        score += 2.0
    elif fd.fcf_yield >= 0.01:
        score += 0.5
    elif fd.fcf_yield < 0:
        score -= 2.0

    if fd.revenue_growth_yoy >= 0.12:
        score += 2.0
    elif fd.revenue_growth_yoy >= 0.05:
        score += 1.0

    if fd.eps_growth_yoy >= 0.20:
        score += 2.0; notes.append(f"EPS_growth={fd.eps_growth_yoy:.0%}(+2)")
    elif fd.eps_growth_yoy >= 0.10:
        score += 1.0
    elif fd.eps_growth_yoy < 0:
        score -= 1.0

    if fd.roe >= 0.20:
        score += 2.0
    elif fd.roe >= 0.12:
        score += 1.0

    if fd.roic >= 0.15:
        score += 2.0; notes.append(f"ROIC={fd.roic:.0%} high(+2)")
    elif fd.roic >= 0.10:
        score += 1.0

    if fd.debt_to_equity <= 0.30:
        score += 2.0; notes.append(f"D/E={fd.debt_to_equity:.2f} fortress(+2)")
    elif fd.debt_to_equity <= 0.80:
        score += 1.0
    elif fd.debt_to_equity > 2.0:
        score -= 1.0; notes.append(f"D/E={fd.debt_to_equity:.2f} high leverage(-1)")

    if fd.operating_margin >= 0.20:
        score += 1.0
    if fd.net_margin >= 0.15:
        score += 1.0

    return min(max(score, 0.0), 20.0), notes


def _score_management(fd: FundamentalData) -> tuple[float, list[str]]:
    """0-10: management quality proxies."""
    score, notes = 0.0, []

    if fd.roic >= 0.18:
        score += 3.0; notes.append(f"ROIC={fd.roic:.0%} excellent allocation(+3)")
    elif fd.roic >= 0.12:
        score += 2.0
    elif fd.roic >= 0.08:
        score += 1.0

    if fd.shares_growth_yoy < -0.02:
        score += 3.0; notes.append(f"buybacks={fd.shares_growth_yoy:.1%} shareholder-friendly(+3)")
    elif fd.shares_growth_yoy < 0.01:
        score += 1.5
    elif fd.shares_growth_yoy > 0.05:
        score -= 1.0; notes.append(f"dilution={fd.shares_growth_yoy:.1%}(-1)")

    if fd.insider_ownership_pct >= 0.10:
        score += 2.0; notes.append(f"insider={fd.insider_ownership_pct:.0%} aligned(+2)")
    elif fd.insider_ownership_pct >= 0.03:
        score += 1.0

    if 0 < fd.dividend_yield <= 0.04 and fd.payout_ratio < 0.60:
        score += 2.0; notes.append(f"dividend={fd.dividend_yield:.1%} disciplined(+2)")
    elif fd.dividend_yield > 0.05 and fd.payout_ratio > 0.80:
        notes.append(f"payout={fd.payout_ratio:.0%} high → potential trap")

    return min(max(score, 0.0), 10.0), notes


def _score_valuation(fd: FundamentalData) -> tuple[float, list[str]]:
    """0-20: valuation attractiveness and margin of safety."""
    score, notes = 0.0, []

    if fd.fcf_yield >= 0.07:
        score += 5.0; notes.append(f"FCF_yield={fd.fcf_yield:.1%} exceptional MOS(+5)")
    elif fd.fcf_yield >= 0.04:
        score += 3.0; notes.append(f"FCF_yield={fd.fcf_yield:.1%} fair value(+3)")
    elif fd.fcf_yield >= 0.02:
        score += 1.0
    elif fd.fcf_yield < 0:
        score -= 2.0

    if 0 < fd.pe_ratio <= 15:
        score += 4.0; notes.append(f"P/E={fd.pe_ratio:.1f} value(+4)")
    elif 15 < fd.pe_ratio <= 25:
        score += 2.5
    elif 25 < fd.pe_ratio <= 40:
        score += 1.0
    elif fd.pe_ratio > 60:
        score -= 1.0; notes.append(f"P/E={fd.pe_ratio:.1f} rich(-1)")

    if 0 < fd.peg_ratio <= 1.0:
        score += 4.0; notes.append(f"PEG={fd.peg_ratio:.2f} growth-at-value(+4)")
    elif 1.0 < fd.peg_ratio <= 1.5:
        score += 2.0
    elif fd.peg_ratio > 3.0:
        score -= 1.0

    # Distance from 52W low = margin of safety proxy
    if fd.current_price > 0 and fd.fifty_two_week_low > 0 and fd.fifty_two_week_high > 0:
        rng = fd.fifty_two_week_high - fd.fifty_two_week_low
        if rng > 0:
            pct = (fd.current_price - fd.fifty_two_week_low) / rng
            if pct <= 0.20:
                score += 4.0; notes.append(f"lower 20% of 52W range = MOS zone(+4)")
            elif pct <= 0.40:
                score += 2.0
            elif pct > 0.80:
                notes.append("near 52W high → thin margin of safety")

    if 0 < fd.ev_ebitda <= 10:
        score += 3.0; notes.append(f"EV/EBITDA={fd.ev_ebitda:.1f} cheap(+3)")
    elif 10 < fd.ev_ebitda <= 20:
        score += 1.5
    elif fd.ev_ebitda > 40:
        score -= 1.0

    return min(max(score, 0.0), 20.0), notes


def _score_macro_alignment(mc: dict) -> tuple[float, list[str]]:
    """0-10: macro regime alignment for long-term equity accumulation."""
    score, notes = 0.0, []
    bias = str(mc.get("overall_market_bias") or mc.get("market_bias") or mc.get("regime") or "").lower()
    rate_dir = str(mc.get("rate_direction") or mc.get("fed_direction") or "").lower()
    infl = str(mc.get("inflation_trend") or "").lower()
    rec_risk = str(mc.get("recession_risk") or "low").lower()

    if any(b in bias for b in ("bullish", "risk_on", "uptrend", "constructive")):
        score += 4.0; notes.append(f"regime={bias}(+4)")
    elif any(b in bias for b in ("neutral", "mixed", "sideways")):
        score += 2.0
    elif any(b in bias for b in ("bearish", "risk_off", "downtrend")):
        notes.append(f"regime={bias} headwind for accumulation")

    if any(r in rate_dir for r in ("cutting", "easing", "dovish")):
        score += 2.0; notes.append(f"rates={rate_dir} tailwind(+2)")
    elif any(r in rate_dir for r in ("pausing", "holding")):
        score += 1.0
    elif any(r in rate_dir for r in ("hiking", "tightening", "hawkish")):
        notes.append(f"rates={rate_dir} multiple headwind")

    if any(i in infl for i in ("falling", "cooling", "disinflationary")):
        score += 2.0; notes.append("inflation cooling(+2)")
    elif any(i in infl for i in ("stable", "anchored")):
        score += 1.0
    elif any(i in infl for i in ("rising", "hot")):
        notes.append("inflation hot → real return risk")

    if any(r in rec_risk for r in ("low", "minimal")):
        score += 1.0
    elif any(r in rec_risk for r in ("elevated", "high")):
        score -= 1.0; notes.append(f"recession_risk={rec_risk}(-1)")

    return min(max(score, 0.0), 10.0), notes


def _score_economic_policy(mc: dict) -> tuple[float, list[str]]:
    """0-10: economic policy and calendar event alignment."""
    score, notes = 5.0, []   # start neutral

    fomc = str(mc.get("fomc_stance") or mc.get("fed_stance") or "").lower()
    if any(f in fomc for f in ("dovish", "accommodative", "easy")):
        score += 2.0; notes.append(f"FOMC={fomc}(+2)")
    elif any(f in fomc for f in ("hawkish", "restrictive", "tight")):
        score -= 2.0; notes.append(f"FOMC={fomc}(-2)")

    cpi = str(mc.get("cpi_trend") or "").lower()
    if any(c in cpi for c in ("falling", "below_target", "cooling")):
        score += 1.0
    elif any(c in cpi for c in ("rising", "above_target", "hot")):
        score -= 1.0

    tariff = str(mc.get("tariff_risk") or "low").lower()
    if any(t in tariff for t in ("high", "elevated", "escalating")):
        score -= 2.0; notes.append(f"tariff_risk={tariff}(-2)")

    fiscal = str(mc.get("fiscal_policy") or "").lower()
    if any(f in fiscal for f in ("stimulus", "expansionary")):
        score += 1.0
    elif any(f in fiscal for f in ("austerity", "contractionary")):
        score -= 1.0

    return min(max(score, 0.0), 10.0), notes


def _score_cycle_phase(mc: dict) -> tuple[float, list[str]]:
    """
    0-10: 18-year Saeculum cycle positioning.
    Early recovery/crisis reset → best opportunity.
    Speculative acceleration/distribution → caution.
    """
    phase_raw = str(
        mc.get("cycle_phase") or mc.get("economic_cycle") or mc.get("18yr_cycle_phase") or "unknown"
    ).lower()

    phase_scores = {
        "early_recovery": 10.0, "crisis_reset": 9.0, "expansion": 8.0,
        "mid_cycle": 6.0, "late_expansion": 5.0,
        "speculative_acceleration": 4.0, "distribution": 2.0,
        "late_cycle": 2.0, "crisis": 1.0, "recession": 1.0,
    }
    score = 5.0
    matched = "unknown"
    for k, v in phase_scores.items():
        if k in phase_raw:
            score = v; matched = k; break

    if matched == "unknown":
        # Infer from regime
        bias = str(mc.get("overall_market_bias") or "").lower()
        if any(b in bias for b in ("bullish", "uptrend", "risk_on")):
            score = 7.0; matched = "inferred_expansion"
        elif any(b in bias for b in ("bearish", "downtrend", "risk_off")):
            score = 2.0; matched = "inferred_late_cycle"

    return score, [f"cycle={matched} score={score:.0f}/10"]


def _score_technical_entry(fd: FundamentalData, bars_ctx: dict) -> tuple[float, list[str]]:
    """0-5: technical entry quality for long-term accumulation."""
    score, notes = 0.0, []
    price = fd.current_price
    if price <= 0:
        return 0.0, ["no_price"]

    # Long-term uptrend: above 200 SMA
    if fd.sma_200 > 0 and price >= fd.sma_200 * 0.98:
        score += 1.0; notes.append(f"above_200SMA={fd.sma_200:.2f}(+1)")

    # Medium-term: above 50 SMA
    if fd.sma_50 > 0 and price >= fd.sma_50:
        score += 0.5

    # Intraday VWAP
    vwap = float(bars_ctx.get("vwap") or 0)
    if vwap > 0 and price >= vwap:
        score += 1.0; notes.append(f"above_VWAP={vwap:.2f}(+1)")

    # Value zone: lower 30% of annual range
    if fd.fifty_two_week_low > 0 and fd.fifty_two_week_high > 0:
        rng = fd.fifty_two_week_high - fd.fifty_two_week_low
        if rng > 0:
            pct = (price - fd.fifty_two_week_low) / rng
            if pct <= 0.20:
                score += 2.0; notes.append("lower 20% of 52W = accumulation zone(+2)")
            elif pct <= 0.35:
                score += 1.0

    # Institutional participation signal
    rvol = float(bars_ctx.get("rel_volume") or bars_ctx.get("rvol") or 0)
    if rvol >= 1.5:
        score += 0.5; notes.append(f"RVOL={rvol:.1f}x institutional(+0.5)")

    return min(score, 5.0), notes


# ── main scoring ───────────────────────────────────────────────────────────────

def score_symbol(
    symbol: str,
    config: BALConfig,
    market_context: dict,
    bars_ctx: dict,
    fd: FundamentalData | None = None,
) -> BALScorecard:
    card = BALScorecard(symbol=symbol, scored_at_utc=datetime.now(timezone.utc).isoformat())

    if fd is None:
        fd = get_fundamentals(symbol, config)

    if fd.fetch_error and fd.current_price <= 0:
        card.skip_reasons.append("rejected_by_data_unavailable")
        return card

    if fd.market_cap > 0 and fd.market_cap < config.min_market_cap_m * 1e6:
        card.skip_reasons.append("rejected_by_market_cap")

    s1, n1 = _score_moat(fd)
    s2, n2 = _score_financial_strength(fd)
    s3, n3 = _score_management(fd)
    s4, n4 = _score_valuation(fd)
    s5, n5 = _score_macro_alignment(market_context)
    s6, n6 = _score_economic_policy(market_context)
    s7, n7 = _score_cycle_phase(market_context)
    s8, n8 = _score_technical_entry(fd, bars_ctx)

    card.moat_score = round(s1, 2)
    card.financial_score = round(s2, 2)
    card.management_score = round(s3, 2)
    card.valuation_score = round(s4, 2)
    card.macro_score = round(s5, 2)
    card.policy_score = round(s6, 2)
    card.cycle_score = round(s7, 2)
    card.technical_score = round(s8, 2)
    card.bal_score = round(s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8, 1)
    card.notes = (n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8)[:20]

    # Rejection gates
    if card.bal_score < config.min_score_to_buy:
        card.skip_reasons.append(f"rejected_by_BAL_score:{card.bal_score:.1f}<{config.min_score_to_buy}")
    if s4 < 4.0:
        card.skip_reasons.append("rejected_by_valuation")
    if s5 < 2.5:
        card.skip_reasons.append("rejected_by_macro")
    if s7 < 2.0:
        card.skip_reasons.append("rejected_by_cycle")
    if s1 < 3.0:
        card.skip_reasons.append("rejected_by_BAL_score:moat_too_weak")

    # Action
    if not card.skip_reasons:
        if card.bal_score >= config.aggressive_score:
            card.action = "aggressive_accumulate"
            card.position_size_pct = min(config.max_single_stock_weight, 0.06)
        elif card.bal_score >= config.normal_score:
            card.action = "accumulate"
            card.position_size_pct = min(config.max_single_stock_weight, 0.04)
        else:
            card.action = "starter"
            card.position_size_pct = min(config.max_single_stock_weight, 0.02)

    return card


# ── composite DCA scoring ────────────────────────────────────────────────────────
#
# Five weighted components (0-100 each) → composite_dca_score (0-100):
#     valuation_discount      35%   ← intrinsic-value discount decides WHETHER to DCA
#     technical_dislocation   25%   ← RSI + DMA + ATR + capitulation decide WHEN
#     business_quality        20%   ← moat/financial/management = worth accumulating
#     macro_cycle_opportunity 10%   ← fed/rates/inflation/cycle/recession-fear
#     sentiment_capitulation  10%   ← selloff/downgrades/abnormal volume
#
# RSI lives ONLY inside technical_dislocation and can never trigger DCA by itself.

_DCA_ACTION_RANK = {"no_dca": 0, "small": 1, "normal": 2, "aggressive": 3}


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _dca_valuation_discount_score(p2iv: float) -> float:
    """0-100 from price/intrinsic-value. Deeper discount → higher score. p2iv<=0 → 0."""
    if p2iv <= 0:
        return 0.0
    # Piecewise-linear anchors (ratio, score). Lower ratio = cheaper = higher score.
    pts = [(0.60, 100.0), (0.65, 95.0), (0.75, 85.0), (0.85, 70.0),
           (0.95, 55.0), (1.00, 40.0), (1.10, 20.0), (1.25, 0.0)]
    if p2iv <= pts[0][0]:
        return 100.0
    if p2iv >= pts[-1][0]:
        return 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= p2iv <= x1:
            return round(y0 + (y1 - y0) * (p2iv - x0) / (x1 - x0), 1)
    return 0.0


def _dca_technical_dislocation_score(fd: FundamentalData, bars_ctx: dict) -> tuple[float, dict]:
    """
    0-100 technical dislocation. RSI is one INPUT among several and is capped so it
    can never dominate — DCA is never triggered by RSI alone.

    Components (max): RSI 30, dist<50DMA 20, dist<200DMA 15, ATR expansion 15,
    capitulation volume 12, VWAP reclaim 8.
    """
    score, notes = 0.0, []
    price = fd.current_price or float(bars_ctx.get("close") or 0)

    rsi = float(fd.rsi_14)
    if rsi < 20:
        score += 30.0; notes.append(f"RSI={rsi:.0f}<20 extreme(+30)")
    elif rsi < 25:
        score += 24.0; notes.append(f"RSI={rsi:.0f}<25 strong(+24)")
    elif rsi < 30:
        score += 18.0; notes.append(f"RSI={rsi:.0f}<30 oversold(+18)")
    elif rsi < 40:
        score += 8.0
    elif rsi > 70:
        score -= 5.0; notes.append(f"RSI={rsi:.0f}>70 overbought(-5)")

    dist50 = (price - fd.sma_50) / fd.sma_50 if fd.sma_50 > 0 else 0.0
    if dist50 <= -0.15:
        score += 20.0; notes.append(f"-{abs(dist50):.0%} vs 50DMA(+20)")
    elif dist50 <= -0.10:
        score += 15.0
    elif dist50 <= -0.05:
        score += 8.0

    dist200 = (price - fd.sma_200) / fd.sma_200 if fd.sma_200 > 0 else 0.0
    # Healthy dip below the 200DMA is the sweet spot; a totally broken long-term
    # trend gets only a small contribution (the thesis gate handles real breakage).
    if -0.20 <= dist200 <= -0.08:
        score += 15.0; notes.append(f"-{abs(dist200):.0%} vs 200DMA dip(+15)")
    elif -0.08 < dist200 < 0.0:
        score += 8.0
    elif dist200 < -0.30:
        score += 4.0

    if fd.atr_expansion >= 1.8:
        score += 15.0; notes.append(f"ATR x{fd.atr_expansion:.1f} expansion(+15)")
    elif fd.atr_expansion >= 1.4:
        score += 10.0
    elif fd.atr_expansion >= 1.15:
        score += 5.0

    if fd.volume_capitulation >= 2.0:
        score += 12.0; notes.append(f"vol x{fd.volume_capitulation:.1f} capitulation(+12)")
    elif fd.volume_capitulation >= 1.5:
        score += 8.0
    elif fd.volume_capitulation >= 1.2:
        score += 4.0

    vwap = float(bars_ctx.get("vwap") or 0)
    close = float(bars_ctx.get("close") or price)
    if vwap > 0 and close >= vwap and dist50 < 0:
        score += 8.0; notes.append("VWAP reclaim after selloff(+8)")

    detail = {
        "distance_from_50DMA": round(dist50, 4),
        "distance_from_200DMA": round(dist200, 4),
        "notes": notes,
    }
    return _clamp(score), detail


def _dca_business_quality_score(card: BALScorecard) -> float:
    """0-100 from moat / financial strength / management (the 'worth owning' gate)."""
    moat_n = card.moat_score / 15.0 if card.moat_score else 0.0
    fin_n = card.financial_score / 20.0 if card.financial_score else 0.0
    mgmt_n = card.management_score / 10.0 if card.management_score else 0.0
    return round(100.0 * (0.40 * moat_n + 0.40 * fin_n + 0.20 * mgmt_n), 1)


def _dca_macro_cycle_opportunity_score(card: BALScorecard, market_context: dict) -> float:
    """
    0-100 macro/cycle opportunity for accumulation. Uses the BAL macro/policy/cycle
    sub-scores; recession FEAR is treated as opportunity (buy when others are fearful).
    """
    macro_n = card.macro_score / 10.0 if card.macro_score else 0.0
    policy_n = card.policy_score / 10.0 if card.policy_score else 0.0
    cycle_n = card.cycle_score / 10.0 if card.cycle_score else 0.0
    base = 100.0 * (0.40 * macro_n + 0.30 * cycle_n + 0.30 * policy_n)

    rec = str(market_context.get("recession_risk") or market_context.get("recession_fear") or "").lower()
    if any(r in rec for r in ("high", "elevated", "rising")):
        base += 10.0  # capitulation/fear = better long-term entry
    liq = str(market_context.get("liquidity_conditions") or market_context.get("liquidity") or "").lower()
    if any(l in liq for l in ("tight", "draining", "contracting")):
        base -= 5.0
    elif any(l in liq for l in ("ample", "easing", "expanding")):
        base += 5.0
    return round(_clamp(base), 1)


def _dca_sentiment_capitulation_score(fd: FundamentalData, market_context: dict) -> float:
    """
    0-100 sentiment capitulation. Combines explicit market-context signals (when
    present) with technical capitulation proxies (oversold RSI, abnormal selloff
    volume, ATR spike). Capitulation = contrarian accumulation opportunity.
    """
    score = 0.0

    # Explicit context signals (all optional / best-effort).
    sector = str(market_context.get("sector_selloff") or market_context.get("sector_stress") or "").lower()
    if any(s in sector for s in ("high", "severe", "selloff", "stress")):
        score += 20.0
    downgrades = market_context.get("analyst_downgrades")
    if isinstance(downgrades, (int, float)) and downgrades >= 2:
        score += 15.0
    elif any(d in str(downgrades or "").lower() for d in ("spike", "high", "wave")):
        score += 15.0
    news = str(market_context.get("news_sentiment") or market_context.get("negative_news") or "").lower()
    if any(n in news for n in ("very_negative", "panic", "fear", "negative_spike")):
        score += 15.0
    pc = market_context.get("put_call_ratio")
    if isinstance(pc, (int, float)) and pc >= 1.0:
        score += 15.0  # elevated put/call = fear

    # Technical capitulation proxies (always available).
    if fd.rsi_14 < 25:
        score += 15.0
    elif fd.rsi_14 < 32:
        score += 8.0
    if fd.volume_capitulation >= 2.0:
        score += 15.0
    elif fd.volume_capitulation >= 1.5:
        score += 8.0
    if fd.atr_expansion >= 1.6:
        score += 7.0

    return round(_clamp(score), 1)


def _valuation_eligible_action(p2iv: float, cfg: DCAConfig, bal_score: float) -> tuple[str, str]:
    """
    Map the intrinsic-value discount to the deepest DCA action it justifies and a
    tier label. Returns (eligible_action, tier). 'no_dca'/'not_undervalued' when the
    price is not at a meaningful discount.

      p2iv <= extreme(0.65): maximum  → aggressive (requires BAL_score >= aggressive)
      p2iv <= panic(0.75):   aggressive
      p2iv <= deep(0.85):    normal
      p2iv <= first(0.95):   small
      else:                  not undervalued
    """
    if p2iv <= 0:
        return "no_dca", "unknown_valuation"
    if p2iv <= cfg.extreme_discount:
        if bal_score >= cfg.bal_aggressive_score:
            return "aggressive", "extreme"
        return "normal", "extreme_capped"   # max tier needs aggressive-grade business
    if p2iv <= cfg.panic_discount:
        return "aggressive", "panic"
    if p2iv <= cfg.deep_discount:
        return "normal", "deep"
    if p2iv <= cfg.first_discount:
        return "small", "first"
    return "no_dca", "not_undervalued"


def _composite_action(score: float, cfg: DCAConfig) -> str:
    if score >= cfg.aggressive_score:
        return "aggressive"
    if score >= cfg.normal_score:
        return "normal"
    if score >= cfg.small_score:
        return "small"
    return "no_dca"


# ── conviction / insider / quality scoring ───────────────────────────────────────

def _profitability_score(fd: FundamentalData) -> float:
    """0-100 profitability quality from ROE / ROIC / net & operating margins."""
    roe = _clamp(float(fd.roe or 0) / 0.20, 0.0, 1.0)
    roic = _clamp(float(fd.roic or 0) / 0.15, 0.0, 1.0)
    net_m = _clamp(float(fd.net_margin or 0) / 0.20, 0.0, 1.0)
    op_m = _clamp(float(fd.operating_margin or 0) / 0.25, 0.0, 1.0)
    return round(100.0 * (0.35 * roe + 0.30 * roic + 0.20 * net_m + 0.15 * op_m), 1)


def _earnings_quality_score(fd: FundamentalData) -> float:
    """
    0-100 earnings quality. Rewards cash-backed earnings (FCF covering net income),
    positive margins, and low leverage; penalises share dilution. A high accounting
    earnings number not backed by cash flow scores poorly.
    """
    score = 0.0
    net_income = float(fd.net_margin or 0) * float(fd.revenue_ttm or 0)
    if fd.free_cash_flow > 0 and net_income > 0:
        cover = fd.free_cash_flow / net_income
        score += 45.0 * _clamp(cover, 0.0, 1.2) / 1.2   # FCF ≥ 1.2× NI → full marks
    elif fd.free_cash_flow > 0:
        score += 25.0                                   # positive FCF, NI unknown
    if fd.net_margin > 0.10:
        score += 20.0
    elif fd.net_margin > 0.0:
        score += 10.0
    if fd.debt_to_equity <= 0.5:
        score += 15.0
    elif fd.debt_to_equity <= 1.5:
        score += 8.0
    if fd.shares_growth_yoy <= 0.0:
        score += 20.0                                   # buybacks / no dilution
    elif fd.shares_growth_yoy <= 0.02:
        score += 10.0
    return round(_clamp(score), 1)


def compute_insider_buy_score(insider_data: dict | None, fd: FundamentalData) -> dict:
    """
    Score insider PURCHASES (0-100) as a CONVICTION MODIFIER. This never triggers a
    buy by itself — the caller decides whether it is *used* (only when BAL score and
    valuation already pass). Reads an optional per-symbol insider_data dict:
        {"amount": <usd net buy>, "role": "CEO"|"CFO"|"Director"|..., "cluster": bool,
         "buys": <count>, "sells": <count>}
    Falls back to neutral (no signal) when no data is supplied. When the supplied
    dict carries a precomputed "score" (e.g. the SEC Form 4 discovery-quality score),
    that authoritative score is used directly — the layer only decides USAGE.

    Returns: {detected, score, amount, role, confidence}.
    """
    d = insider_data or {}
    amount = float(d.get("amount") or d.get("net_buy_usd") or 0.0)
    role = str(d.get("role") or d.get("title") or "").strip()
    cluster = bool(d.get("cluster") or (int(d.get("buys", 0) or 0) >= 3))
    precomputed = d.get("score", d.get("insider_buy_score"))
    detected = amount > 0 or bool(d.get("detected")) or (precomputed is not None and float(precomputed) > 0)

    if not detected:
        return {"detected": False, "score": 0.0, "amount": 0.0, "role": "", "confidence": 0.0}

    # Prefer an authoritative precomputed score (SEC Form 4 discovery).
    if precomputed is not None:
        return {"detected": True, "score": round(_clamp(float(precomputed)), 1),
                "amount": round(amount, 2), "role": role,
                "confidence": round(min(float(d.get("confidence", 0.75) or 0.75), 1.0), 2)}

    # Amount tier (0-60)
    if amount >= 5_000_000:
        score = 60.0
    elif amount >= 1_000_000:
        score = 48.0
    elif amount >= 250_000:
        score = 32.0
    elif amount >= 100_000:
        score = 18.0
    else:
        score = 8.0
    # Role weight (0-25)
    role_l = role.lower()
    if any(r in role_l for r in ("ceo", "chief executive", "cfo", "chief financial", "chairman", "founder")):
        score += 25.0
    elif any(r in role_l for r in ("director", "president", "coo", "officer", "evp", "svp")):
        score += 15.0
    elif role_l:
        score += 8.0
    # Cluster buying (0-15)
    if cluster:
        score += 15.0

    confidence = 0.5
    if role_l:
        confidence += 0.25
    if amount >= 250_000:
        confidence += 0.25
    return {"detected": True, "score": round(_clamp(score), 1), "amount": round(amount, 2),
            "role": role, "confidence": round(min(confidence, 1.0), 2)}


# Conviction component weights (sum to 1.0). Each input is normalised to 0-100.
_CONVICTION_WEIGHTS = {
    "bal": 0.25, "moat": 0.10, "financial": 0.12, "profitability": 0.10,
    "management": 0.08, "valuation_discount": 0.15, "macro_cycle": 0.05,
    "insider": 0.07, "earnings_quality": 0.08,
}


def compute_conviction_score(
    card: BALScorecard,
    fd: FundamentalData,
    valuation_discount_score: float,
    macro_cycle_score: float,
    insider_buy_score: float,
    earnings_quality_score: float,
    profitability_score: float,
) -> dict:
    """
    0-100 Buffett conviction blending business quality, valuation discount, macro/cycle
    opportunity, insider buying, and earnings quality. Drives target allocation — it
    does NOT relax the quality or valuation gates (those are enforced separately).
    Returns {conviction_score, breakdown}.
    """
    inputs = {
        "bal": _clamp(float(card.bal_score)),
        "moat": _clamp(card.moat_score / 15.0 * 100.0),
        "financial": _clamp(card.financial_score / 20.0 * 100.0),
        "profitability": _clamp(profitability_score),
        "management": _clamp(card.management_score / 10.0 * 100.0),
        "valuation_discount": _clamp(valuation_discount_score),
        "macro_cycle": _clamp(macro_cycle_score),
        "insider": _clamp(insider_buy_score),
        "earnings_quality": _clamp(earnings_quality_score),
    }
    conviction = sum(_CONVICTION_WEIGHTS[k] * inputs[k] for k in _CONVICTION_WEIGHTS)
    return {"conviction_score": round(_clamp(conviction), 1),
            "breakdown": {k: round(inputs[k], 1) for k in inputs}}


def conviction_target_allocation(conviction: float, cfg: DCAConfig | None = None) -> tuple[float, float, str]:
    """
    Conviction → (target_allocation_pct, max_allocation_pct, reason). Target is a
    fraction of the max_single_stock_weight cap; below the watchlist floor the target
    is 0 (watchlist only — no DCA). Free function so the screener and the layer share
    one allocation policy.
    """
    cfg = cfg or DCAConfig()
    cap = cfg.max_single_stock_weight
    if conviction >= cfg.conviction_large_score:
        return round(cap * cfg.conviction_large_frac, 4), round(cap, 4), \
            f"conviction>={cfg.conviction_large_score:.0f}:larger_allocation"
    if conviction >= cfg.conviction_normal_score:
        return round(cap * cfg.conviction_normal_frac, 4), round(cap, 4), \
            f"conviction>={cfg.conviction_normal_score:.0f}:normal_allocation"
    if conviction >= cfg.conviction_small_score:
        return round(cap * cfg.conviction_small_frac, 4), round(cap, 4), \
            f"conviction>={cfg.conviction_small_score:.0f}:small_allocation"
    return 0.0, round(cap, 4), f"conviction<{cfg.conviction_small_score:.0f}:watchlist_only"


# ── main engine ────────────────────────────────────────────────────────────────

class BuffettAccumulationLayer:
    """
    Strategic long-stock engine. Scores candidates 0-100 and accumulates
    quality businesses at optimal entry points. Holdings are permanent unless
    thesis, trend, concentration, or risk limits break.
    """

    def __init__(self, config: BALConfig | None = None, dca_config: DCAConfig | None = None):
        self.config = config or BALConfig()
        self.dca_config = dca_config or DCAConfig()
        self._ownership: dict[str, dict] = self._load_ownership()

    def _load_ownership(self) -> dict[str, dict]:
        if OWNERSHIP_UNIVERSE_PATH.exists():
            try:
                return json.loads(OWNERSHIP_UNIVERSE_PATH.read_text())
            except Exception:
                pass
        return {}

    def _save_ownership(self) -> None:
        try:
            OWNERSHIP_UNIVERSE_PATH.write_text(
                json.dumps(self._ownership, indent=2, default=str), encoding="utf-8"
            )
        except Exception:
            pass

    def _save_scores(self, scorecards: list[BALScorecard]) -> None:
        try:
            BAL_SCORES_PATH.write_text(
                json.dumps({
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "date_tag": DATA_TAG,
                    "count": len(scorecards),
                    "scores": [asdict(c) for c in scorecards],
                }, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    def scan(
        self,
        symbols: list[str],
        market_context: dict,
        bars_ctx_by_symbol: dict[str, dict],
        equity: float,
        positions: dict[str, Any],
        available_capital: float,
    ) -> list[dict]:
        """
        Score all candidates. Return accumulation buy orders for those passing BAL threshold.
        Every rejected symbol gets a logged reason.
        """
        if not self.config.enabled:
            return []

        # Stock exposure check
        stock_value = sum(
            float(getattr(p, "market_value", 0) or 0) for p in positions.values()
        )
        stock_pct = stock_value / equity if equity > 0 else 0.0

        scorecards: list[BALScorecard] = []
        orders: list[dict] = []

        for symbol in symbols:
            bars_ctx = bars_ctx_by_symbol.get(symbol, {})
            card = score_symbol(symbol, self.config, market_context, bars_ctx)
            scorecards.append(card)

            if card.action == "no_action" or card.skip_reasons:
                for reason in card.skip_reasons:
                    _bal_event("bal_skip", {"symbol": symbol, "reason": reason, "score": card.bal_score})
                continue

            if stock_pct >= self.config.max_stock_exposure_pct:
                _bal_event("bal_skip", {"symbol": symbol, "reason": "rejected_by_exposure_limit", "score": card.bal_score})
                continue

            # Single-stock weight check
            existing = positions.get(symbol)
            existing_value = float(getattr(existing, "market_value", 0) or 0)
            existing_weight = existing_value / equity if equity > 0 else 0.0
            if existing_weight >= self.config.max_single_stock_weight:
                _bal_event("bal_skip", {"symbol": symbol, "reason": "rejected_by_exposure_limit:single_stock_full", "score": card.bal_score})
                continue

            # Size the order
            target_dollars = equity * card.position_size_pct
            room = (self.config.max_single_stock_weight - existing_weight) * equity
            order_dollars = min(target_dollars, room, available_capital)

            if order_dollars < 200:
                _bal_event("bal_skip", {"symbol": symbol, "reason": "rejected_by_cash_buffer:insufficient", "score": card.bal_score})
                continue

            fd = get_fundamentals(symbol, self.config)
            price = fd.current_price
            if price <= 0:
                _bal_event("bal_skip", {"symbol": symbol, "reason": "rejected_by_data_unavailable:no_price", "score": card.bal_score})
                continue

            qty = max(1, int(order_dollars / price))
            orders.append({
                "symbol": symbol,
                "side": "buy",
                "qty": qty,
                "order_dollars": round(qty * price, 2),
                "price": price,
                "source": "buffett_accumulation_layer",
                "action": card.action,
                "bal_score": card.bal_score,
                "reason": f"BAL:{card.action}:score={card.bal_score:.1f}",
                "hold_intent": "long_term_indefinite",
                "bracket_attached": False,   # BAL positions use separate exit logic
                "notes": card.notes[:5],
            })

            self._ownership[symbol] = {
                "symbol": symbol,
                "first_accumulated_utc": self._ownership.get(symbol, {}).get(
                    "first_accumulated_utc", datetime.now(timezone.utc).isoformat()
                ),
                "last_scored_utc": card.scored_at_utc,
                "bal_score": card.bal_score,
                "action": card.action,
                "hold_intent": "long_term_indefinite",
                "thesis": {
                    "moat": card.moat_score,
                    "financials": card.financial_score,
                    "management": card.management_score,
                    "valuation": card.valuation_score,
                    "macro": card.macro_score,
                    "policy": card.policy_score,
                    "cycle": card.cycle_score,
                    "technical": card.technical_score,
                },
            }

            _bal_event("bal_accumulate", {
                "symbol": symbol,
                "action": card.action,
                "score": card.bal_score,
                "qty": qty,
                "dollars": round(qty * price, 2),
            })

        self._save_scores(scorecards)
        self._save_ownership()
        return orders

    def should_exit(self, symbol: str, position: Any, market_context: dict) -> tuple[bool, str]:
        """
        Long-term stock exit check. Only exits on thesis break, severe trend break,
        catastrophic macro, or concentration breach. Returns (exit, reason).
        """
        rec = self._ownership.get(symbol, {})

        # Thesis break: BAL score collapsed
        bal_score = float(rec.get("bal_score", 70))
        if bal_score < self.config.exit_score_threshold:
            return True, f"thesis_break:BAL_score={bal_score:.0f}<{self.config.exit_score_threshold}"

        # Severe drawdown (35% below cost basis)
        cost = float(getattr(position, "avg_entry_price", 0) or 0)
        price = float(getattr(position, "current_price", 0) or 0)
        if cost > 0 and price > 0:
            dd = (price - cost) / cost
            if dd < -self.config.exit_drawdown_pct:
                return True, f"trend_break:drawdown={dd:.1%}<-{self.config.exit_drawdown_pct:.0%}"

        # Catastrophic macro regime
        bias = str(market_context.get("overall_market_bias") or "").lower()
        if any(b in bias for b in ("crisis", "crash", "systemic_risk")):
            return True, f"macro_crisis:{bias}"

        return False, ""

    # ── composite DCA ───────────────────────────────────────────────────────────

    def _target_allocation_from_conviction(self, conviction: float) -> tuple[float, float, str]:
        """Instance wrapper over conviction_target_allocation using this layer's DCAConfig."""
        return conviction_target_allocation(conviction, self.dca_config)

    def _save_dca_signals(self, signals: list[DCASignal]) -> None:
        try:
            BUFFETT_DCA_SIGNALS_PATH.write_text(
                json.dumps({
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "date_tag": DATA_TAG,
                    "count": len(signals),
                    "config": {
                        "weights": {
                            "valuation_discount": self.dca_config.valuation_weight,
                            "technical_dislocation": self.dca_config.technical_weight,
                            "business_quality": self.dca_config.business_quality_weight,
                            "macro_cycle_opportunity": self.dca_config.macro_cycle_weight,
                            "sentiment_capitulation": self.dca_config.sentiment_weight,
                        },
                        "action_thresholds": {
                            "small": self.dca_config.small_score,
                            "normal": self.dca_config.normal_score,
                            "aggressive": self.dca_config.aggressive_score,
                        },
                        "discount_tiers": {
                            "first": self.dca_config.first_discount,
                            "deep": self.dca_config.deep_discount,
                            "panic": self.dca_config.panic_discount,
                            "extreme": self.dca_config.extreme_discount,
                        },
                        "max_dca_count_per_symbol": self.dca_config.max_dca_count_per_symbol,
                        "min_days_between_dca": self.dca_config.min_days_between_dca,
                    },
                    "signals": [asdict(s) for s in signals],
                }, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _save_valuation_diagnostic(self, rows: list[dict]) -> None:
        try:
            VALUATION_DIAGNOSTIC_PATH.write_text(json.dumps({
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "date_tag": DATA_TAG,
                "min_models_for_full_confidence": _MIN_VALUATION_MODELS,
                "count": len(rows),
                "valuations": rows,
            }, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass

    def _save_conviction_report(self, rows: list[dict]) -> None:
        try:
            CONVICTION_REPORT_PATH.write_text(json.dumps({
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "date_tag": DATA_TAG,
                "weights": _CONVICTION_WEIGHTS,
                "allocation_tiers": {
                    "large": self.dca_config.conviction_large_score,
                    "normal": self.dca_config.conviction_normal_score,
                    "small": self.dca_config.conviction_small_score,
                },
                "count": len(rows),
                "convictions": rows,
            }, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass

    def evaluate_dca(
        self,
        symbols: list[str],
        market_context: dict,
        bars_ctx_by_symbol: dict[str, dict],
        equity: float,
        positions: dict[str, Any],
        available_capital: float,
        fd_by_symbol: dict[str, FundamentalData] | None = None,
        insider_data_by_symbol: dict[str, dict] | None = None,
    ) -> list[dict]:
        """
        Composite DCA evaluation with multi-model valuation, conviction-driven
        allocation, and an insider-purchase conviction modifier.

        Decision order (final rule):
            Quality determines ELIGIBILITY (business-quality gate),
            Valuation determines ATTRACTIVENESS (multi-model intrinsic value),
            Conviction determines ALLOCATION (target_position_pct + tranche size),
            Composite DCA determines TIMING (when to add),
            RiskManager determines PERMISSION (the hard gate, in run.py).

        Writes buffett_dca_signals.json + valuation_diagnostic.json +
        conviction_score_report.json and journals a canonical reason code for every
        symbol. Returns BUY order intents for run.py to route through the gateway.
        """
        cfg = self.dca_config
        if not cfg.enabled:
            return []

        today = datetime.now(ET).date()
        now_iso = datetime.now(timezone.utc).isoformat()
        insider_data_by_symbol = insider_data_by_symbol or {}
        signals: list[DCASignal] = []
        orders: list[dict] = []
        val_rows: list[dict] = []
        conv_rows: list[dict] = []

        for symbol in symbols:
            bars_ctx = bars_ctx_by_symbol.get(symbol, {})
            fd = (fd_by_symbol or {}).get(symbol)
            if fd is None:
                fd = get_fundamentals(symbol, self.config)
            card = score_symbol(symbol, self.config, market_context, bars_ctx, fd=fd)

            price = fd.current_price or float(bars_ctx.get("close") or 0.0)

            # ── valuation (multi-model) ─────────────────────────────────────────
            val_detail = multi_model_intrinsic_value(fd, market_context)
            iv = float(val_detail.get("intrinsic_value", 0.0) or 0.0)
            p2iv = round(price / iv, 4) if (iv > 0 and price > 0) else 0.0

            # ── composite DCA component scores ──────────────────────────────────
            val_s = _dca_valuation_discount_score(p2iv)
            tech_s, tech_detail = _dca_technical_dislocation_score(fd, bars_ctx)
            biz_s = _dca_business_quality_score(card)
            macro_s = _dca_macro_cycle_opportunity_score(card, market_context)
            sent_s = _dca_sentiment_capitulation_score(fd, market_context)
            composite = round(
                cfg.valuation_weight * val_s + cfg.technical_weight * tech_s
                + cfg.business_quality_weight * biz_s + cfg.macro_cycle_weight * macro_s
                + cfg.sentiment_weight * sent_s, 1)

            # ── quality sub-scores + insider modifier ───────────────────────────
            prof_s = _profitability_score(fd)
            eq_s = _earnings_quality_score(fd)
            insider = compute_insider_buy_score(insider_data_by_symbol.get(symbol), fd)
            bal_ok = card.bal_score >= cfg.min_score_to_buy
            thesis_intact = card.bal_score >= cfg.thesis_break_score and \
                card.moat_score >= 3.0 and card.financial_score >= 6.0
            undervalued = 0 < p2iv <= cfg.insider_undervalued_p2iv
            insider_used = bool(insider["detected"]) and thesis_intact and bal_ok and undervalued
            insider_reject = ""
            if insider["detected"] and not insider_used:
                if not thesis_intact:
                    insider_reject = "thesis_broken"
                elif not bal_ok:
                    insider_reject = "weak_fundamentals"
                elif not undervalued:
                    insider_reject = "not_undervalued"
            insider_for_conviction = insider["score"] if insider_used else 0.0

            # ── conviction → target allocation ──────────────────────────────────
            conv = compute_conviction_score(card, fd, val_s, macro_s,
                                            insider_for_conviction, eq_s, prof_s)
            conviction = conv["conviction_score"]
            target_pct, max_alloc_pct, alloc_reason = self._target_allocation_from_conviction(conviction)

            rec = self._ownership.get(symbol, {})
            dca_count = int(rec.get("dca_count", 0) or 0)
            cur_pos = positions.get(symbol)
            cur_value = abs(float(getattr(cur_pos, "market_value", 0) or 0)) if cur_pos is not None else 0.0
            cur_pct = round(cur_value / equity, 4) if equity > 0 else 0.0
            remaining_tranches = max(0, cfg.max_dca_count_per_symbol - dca_count)

            sig = DCASignal(
                symbol=symbol, timestamp=now_iso, price=round(price, 4),
                intrinsic_value=iv, price_to_intrinsic_value=p2iv,
                BAL_score=round(card.bal_score, 1), composite_dca_score=composite,
                valuation_discount_score=val_s, technical_dislocation_score=round(tech_s, 1),
                business_quality_score=biz_s, macro_cycle_opportunity_score=macro_s,
                sentiment_capitulation_score=sent_s, RSI=round(float(fd.rsi_14), 1),
                distance_from_50DMA=tech_detail["distance_from_50DMA"],
                distance_from_200DMA=tech_detail["distance_from_200DMA"],
                ATR_expansion=round(float(fd.atr_expansion), 3),
                volume_capitulation=round(float(fd.volume_capitulation), 3),
                target_position_pct=round(target_pct, 4), current_position_pct=cur_pct,
                remaining_dca_tranches=remaining_tranches,
                valuation_model_values=val_detail.get("valuation_model_values", {}),
                valuation_model_count=int(val_detail.get("valuation_model_count", 0)),
                valuation_confidence=float(val_detail.get("valuation_confidence", 0.0)),
                valuation_dispersion=float(val_detail.get("valuation_dispersion", 0.0)),
                primary_valuation_method=str(val_detail.get("primary_valuation_method", "none")),
                conviction_score=conviction, target_allocation_pct=round(target_pct, 4),
                max_allocation_pct=round(max_alloc_pct, 4), allocation_reason=alloc_reason,
                profitability_score=prof_s, earnings_quality_score=eq_s,
                insider_buy_detected=bool(insider["detected"]),
                insider_buy_score=float(insider["score"]),
                insider_buy_amount=float(insider["amount"]),
                insider_buy_role=str(insider["role"]),
                insider_buy_confidence=float(insider["confidence"]),
                insider_buy_used_in_conviction=insider_used,
                insider_buy_rejected_reason=insider_reject,
            )

            # diagnostics rows (written for EVERY scanned symbol, pass or fail)
            val_rows.append({"symbol": symbol, "price": round(price, 4), "intrinsic_value": iv,
                             "price_to_intrinsic_value": p2iv, **{k: val_detail.get(k) for k in (
                                 "valuation_model_values", "valuation_model_count",
                                 "valuation_confidence", "valuation_dispersion",
                                 "primary_valuation_method", "valuation_fallback_used")}})
            conv_rows.append({"symbol": symbol, "conviction_score": conviction,
                              "breakdown": conv["breakdown"], "target_allocation_pct": round(target_pct, 4),
                              "max_allocation_pct": round(max_alloc_pct, 4),
                              "allocation_reason": alloc_reason,
                              "insider_buy_used_in_conviction": insider_used})

            def _reject(reason: str, journal: str) -> None:
                sig.reject_reason = reason
                sig.journal_reason = journal
                sig.DCA_action = "no_dca"
                sig.dca_reason = f"no_dca:{journal}"
                sig.risk_limits_passed = False
                signals.append(sig)
                _bal_event("dca_reject", {"symbol": symbol, "reason": journal,
                                          "composite": composite, "p2iv": p2iv,
                                          "conviction": conviction, "bal_score": round(card.bal_score, 1)})

            # ── 1. Quality gate — ELIGIBILITY (worth owning?) ───────────────────
            if price <= 0 or (fd.fetch_error and price <= 0):
                _reject("data_unavailable", "rejected_dca_BAL_score"); continue
            if card.bal_score < cfg.thesis_break_score:
                _reject(f"thesis_broken:BAL={card.bal_score:.0f}", "rejected_dca_thesis_broken"); continue
            if card.moat_score < 3.0 or card.financial_score < 6.0:
                _reject("quality_deterioration", "rejected_dca_thesis_broken"); continue
            if card.bal_score < cfg.min_score_to_buy:
                _reject(f"BAL_score={card.bal_score:.0f}<{cfg.min_score_to_buy:.0f}",
                        "rejected_dca_BAL_score"); continue

            # ── 2. Conviction gate — ALLOCATION (below floor → watchlist only) ──
            if conviction < cfg.conviction_small_score or target_pct <= 0:
                _reject(f"conviction={conviction:.0f}<{cfg.conviction_small_score:.0f}:watchlist_only",
                        "rejected_dca_low_conviction"); continue

            # ── 3. Composite-score gate — TIMING (RSI alone can never reach here) ─
            comp_action = _composite_action(composite, cfg)
            if comp_action == "no_dca":
                _reject(f"composite={composite:.0f}<{cfg.small_score:.0f}",
                        "rejected_dca_score_too_low"); continue

            # ── 4. Valuation gate — ATTRACTIVENESS (value decides WHETHER) ──────
            is_starter = dca_count == 0 and cur_pct < (target_pct * cfg.tranche_pct * 0.5)
            if is_starter:
                if p2iv > 1.15:   # never open even a starter when clearly overvalued
                    _reject(f"starter_overvalued:p2iv={p2iv:.2f}", "rejected_dca_not_undervalued"); continue
                final_action, tier = comp_action, "starter"
            else:
                val_action, tier = _valuation_eligible_action(p2iv, cfg, card.bal_score)
                if val_action == "no_dca":
                    _reject(f"not_undervalued:p2iv={p2iv:.2f}", "rejected_dca_not_undervalued"); continue
                final_action = comp_action if _DCA_ACTION_RANK[comp_action] <= _DCA_ACTION_RANK[val_action] \
                    else val_action

            # ── 5. Tranche / timing / concentration controls ────────────────────
            if dca_count >= cfg.max_dca_count_per_symbol:
                _reject(f"max_tranches={dca_count}>={cfg.max_dca_count_per_symbol}",
                        "rejected_dca_max_tranches"); continue
            if cur_pct >= target_pct - 1e-6:
                _reject(f"target_reached:{cur_pct:.2%}>={target_pct:.2%}",
                        "rejected_dca_concentration"); continue
            if cur_pct >= cfg.max_single_stock_weight - 1e-6:
                _reject(f"single_stock_full:{cur_pct:.2%}", "rejected_dca_concentration"); continue
            last_dca = str(rec.get("last_dca_date") or "")
            if last_dca:
                try:
                    delta_days = (today - date.fromisoformat(last_dca)).days
                    if delta_days < cfg.min_days_between_dca:
                        _reject(f"min_days:{delta_days}<{cfg.min_days_between_dca}",
                                "rejected_dca_min_days"); continue
                except Exception:
                    pass
            if available_capital < cfg.min_dca_order_dollars:
                _reject(f"cash_buffer:{available_capital:.0f}<{cfg.min_dca_order_dollars:.0f}",
                        "rejected_dca_cash_buffer"); continue

            # ── 6. Size the tranche — conviction-scaled (HOW MUCH) ──────────────
            target_dollars = target_pct * equity
            base_tranche = target_dollars * cfg.tranche_pct
            action_mult = {"small": 0.5, "normal": 1.0, "aggressive": 1.5}.get(final_action, 1.0)
            conviction_mult = _clamp(0.8 + 0.4 * (conviction / 100.0), 0.8, 1.2)
            desired = base_tranche * action_mult * conviction_mult
            room_to_target = max(0.0, target_dollars - cur_value)
            room_to_max = max(0.0, cfg.max_single_stock_weight * equity - cur_value)
            order_dollars = min(desired, room_to_target, room_to_max, available_capital)

            if order_dollars < cfg.min_dca_order_dollars:
                limiter = "rejected_dca_concentration" if min(room_to_target, room_to_max) <= order_dollars \
                    else "rejected_dca_cash_buffer"
                _reject(f"order_too_small:${order_dollars:.0f}", limiter); continue

            qty = max(1, int(order_dollars / price))
            order_dollars = round(qty * price, 2)
            if order_dollars > available_capital and qty > 1:
                qty -= 1
                order_dollars = round(qty * price, 2)

            # ── 7. Journal reason (which driver led the decision) ───────────────
            if is_starter:
                journal = "dca_triggered_composite_score"
            else:
                drivers = {
                    "dca_triggered_intrinsic_value_discount": cfg.valuation_weight * val_s,
                    "dca_triggered_technical_dislocation": cfg.technical_weight * tech_s,
                    "dca_triggered_macro_cycle_opportunity": cfg.macro_cycle_weight * macro_s,
                }
                journal = max(drivers, key=lambda k: drivers[k])
                if drivers[journal] <= 0:
                    journal = "dca_triggered_composite_score"

            sig.DCA_action = final_action
            sig.DCA_tranche_size = order_dollars
            sig.tranche_size_pct = round(order_dollars / equity, 4) if equity > 0 else 0.0
            sig.journal_reason = journal
            sig.risk_limits_passed = True
            sig.dca_reason = (f"{final_action} DCA: conviction={conviction:.0f} ({alloc_reason}), "
                              f"composite={composite:.0f}, p2iv={p2iv:.2f}, tier={tier}, "
                              f"insider_used={insider_used}")
            signals.append(sig)

            orders.append({
                "symbol": symbol, "side": "buy", "qty": qty,
                "order_dollars": order_dollars, "price": price,
                "source": "buffett_dca", "action": final_action,
                "bal_score": round(card.bal_score, 1),
                "composite_dca_score": composite, "conviction_score": conviction,
                "tier": tier, "target_allocation_pct": round(target_pct, 4),
                "reason": f"buffett_dca:{final_action}:{journal}:conv={conviction:.0f}:score={composite:.0f}:p2iv={p2iv:.2f}",
                "journal_reason": journal, "dca_reason": sig.dca_reason,
                "hold_intent": "long_term_indefinite", "bracket_attached": False,
            })

            # ── 8. Persist tranche state on the ownership record ────────────────
            avg_cost = float(getattr(cur_pos, "avg_entry_price", 0) or 0) if cur_pos is not None else 0.0
            existing = self._ownership.get(symbol, {})
            existing.update({
                "symbol": symbol,
                "first_accumulated_utc": existing.get("first_accumulated_utc", now_iso),
                "last_scored_utc": card.scored_at_utc,
                "bal_score": round(card.bal_score, 1),
                "conviction_score": conviction,
                "action": card.action,
                "hold_intent": "long_term_indefinite",
                "target_position_pct": round(target_pct, 4),
                "current_position_pct": cur_pct,
                "dca_count": dca_count + 1,
                "remaining_dca_tranches": max(0, remaining_tranches - 1),
                "last_dca_date": today.isoformat(),
                "average_cost": round(avg_cost, 4),
                "intrinsic_value": iv,
                "price_to_intrinsic_value": p2iv,
                "deployed_tiers": list(dict.fromkeys((existing.get("deployed_tiers") or []) + [tier])),
            })
            self._ownership[symbol] = existing

            _bal_event("dca_triggered", {
                "symbol": symbol, "action": final_action, "journal": journal,
                "composite": composite, "conviction": conviction, "p2iv": p2iv, "tier": tier,
                "qty": qty, "dollars": order_dollars, "dca_count": dca_count + 1,
            })

        self._save_dca_signals(signals)
        self._save_valuation_diagnostic(val_rows)
        self._save_conviction_report(conv_rows)
        self._save_ownership()
        return orders


# =============================================================================
# BAL Portfolio Monitor & Hedging Layer
# =============================================================================

@dataclass
class HedgeConfig:
    """
    Controls how the BAL portfolio protects itself against seasonality
    and broad market drawdowns using systematic hedging.
    """
    enabled: bool = os.getenv("BAL_HEDGE_ENABLED", "true").lower() in {"1", "true"}
    # Trigger thresholds (portfolio drawdown from high-water mark)
    hedge_light_dd: float = float(os.getenv("BAL_HEDGE_LIGHT_DD", "0.05"))     # -5%  → light hedge
    hedge_medium_dd: float = float(os.getenv("BAL_HEDGE_MEDIUM_DD", "0.10"))   # -10% → medium hedge
    hedge_heavy_dd: float = float(os.getenv("BAL_HEDGE_HEAVY_DD", "0.15"))     # -15% → heavy hedge
    # Regime-based triggers (even without drawdown)
    hedge_on_bearish_regime: bool = True
    hedge_on_seasonality: bool = True        # Sep-Oct historically weak
    # Hedge sizing as % of equity
    light_hedge_pct: float = float(os.getenv("BAL_HEDGE_LIGHT_PCT", "0.03"))   # 3% equity in SPY puts
    medium_hedge_pct: float = float(os.getenv("BAL_HEDGE_MEDIUM_PCT", "0.06"))
    heavy_hedge_pct: float = float(os.getenv("BAL_HEDGE_HEAVY_PCT", "0.10"))
    # Hedge instruments
    broad_market_hedge: str = os.getenv("BAL_HEDGE_INSTRUMENT", "SPY")         # puts on SPY
    sector_hedge_map: dict = field(default_factory=lambda: {
        "Technology": "QQQ",
        "Financials": "XLF",
        "Energy": "XLE",
        "Healthcare": "XLV",
        "Consumer Discretionary": "XLY",
        "Industrials": "XLI",
        "default": "SPY",
    })
    # Put contract parameters
    put_min_dte: int = int(os.getenv("BAL_HEDGE_MIN_DTE", "30"))
    put_max_dte: int = int(os.getenv("BAL_HEDGE_MAX_DTE", "90"))
    put_delta: float = float(os.getenv("BAL_HEDGE_PUT_DELTA", "0.25"))         # ~25-delta puts
    # Seasonal weak months
    weak_months: tuple = (9, 10)    # September, October
    # Exit hedge when: drawdown recovered to X% below HWM
    remove_hedge_dd: float = float(os.getenv("BAL_HEDGE_REMOVE_DD", "0.03"))


@dataclass
class PortfolioMonitorState:
    """Tracks BAL portfolio health metrics over time."""
    high_water_mark: float = 0.0
    current_drawdown: float = 0.0
    hedge_level: str = "none"           # none | light | medium | heavy
    active_hedges: list[dict] = field(default_factory=list)
    last_rebalance_utc: str = ""
    last_hedge_check_utc: str = ""
    concentration_warnings: list[str] = field(default_factory=list)
    seasonality_active: bool = False
    updated_at_utc: str = ""


MONITOR_STATE_PATH = SYSTEM_DATA / f"{DATA_TAG}_bal_monitor_state.json"
HEDGE_EVENTS_PATH = PROJECT_ROOT / "logs" / "execution" / DATA_TAG / f"{DATA_TAG}_bal_hedge_events.jsonl"
HEDGE_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def _hedge_event(event: str, payload: dict) -> None:
    try:
        line = json.dumps({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
        })
        with open(HEDGE_EVENTS_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _is_seasonally_weak(config: HedgeConfig) -> bool:
    """Return True during historically weak months."""
    return datetime.now(ET).month in config.weak_months


def _select_hedge_instrument(sectors: list[str], config: HedgeConfig) -> str:
    """Pick the most concentrated sector ETF as hedge, or SPY for broad hedge."""
    if not sectors:
        return config.broad_market_hedge
    sector_counts: dict[str, int] = {}
    for s in sectors:
        sector_counts[s] = sector_counts.get(s, 0) + 1
    dominant = max(sector_counts, key=lambda k: sector_counts[k]) if sector_counts else ""
    return config.sector_hedge_map.get(dominant, config.sector_hedge_map["default"])


def _compute_portfolio_drawdown(
    positions: dict[str, Any],
    monitor_state: PortfolioMonitorState,
) -> tuple[float, float]:
    """
    Returns (current_portfolio_value, current_drawdown_pct).
    Drawdown is measured from the running high-water mark.
    """
    total_value = sum(
        float(getattr(p, "market_value", 0) or 0) for p in positions.values()
    )
    if total_value > monitor_state.high_water_mark:
        monitor_state.high_water_mark = total_value

    if monitor_state.high_water_mark > 0:
        dd = (total_value - monitor_state.high_water_mark) / monitor_state.high_water_mark
    else:
        dd = 0.0

    return total_value, dd


def _check_concentration(
    positions: dict[str, Any],
    equity: float,
    max_single_weight: float = 0.12,
) -> list[str]:
    """Return list of symbols exceeding concentration limits."""
    warnings = []
    for sym, pos in positions.items():
        val = float(getattr(pos, "market_value", 0) or 0)
        weight = val / equity if equity > 0 else 0
        if weight > max_single_weight:
            warnings.append(f"{sym}:{weight:.0%}>{max_single_weight:.0%}")
    return warnings


class BALPortfolioMonitor:
    """
    Monitors the BAL portfolio for:
      1. Portfolio-level drawdown (triggers systematic hedging)
      2. Seasonal weakness (Sep/Oct)
      3. Regime deterioration (bearish macro)
      4. Concentration drift
      5. Thesis re-evaluation (triggers re-scoring when positions age)

    Hedging instruments: long puts on SPY or sector ETFs.
    Hedges are sized as a percentage of equity and managed separately
    from the long-term stock positions — they are tactical, not permanent.
    """

    def __init__(self, hedge_config: HedgeConfig | None = None, bal_config: BALConfig | None = None):
        self.hcfg = hedge_config or HedgeConfig()
        self.bcfg = bal_config or BALConfig()
        self._state = self._load_state()

    def _load_state(self) -> PortfolioMonitorState:
        if MONITOR_STATE_PATH.exists():
            try:
                raw = json.loads(MONITOR_STATE_PATH.read_text())
                return PortfolioMonitorState(**{
                    k: raw[k] for k in PortfolioMonitorState.__dataclass_fields__ if k in raw
                })
            except Exception:
                pass
        return PortfolioMonitorState()

    def _save_state(self) -> None:
        self._state.updated_at_utc = datetime.now(timezone.utc).isoformat()
        try:
            MONITOR_STATE_PATH.write_text(
                json.dumps(self._state.__dict__, indent=2, default=str), encoding="utf-8"
            )
        except Exception:
            pass

    def _determine_hedge_level(
        self,
        drawdown: float,
        market_context: dict,
    ) -> str:
        """Determine required hedge level from drawdown + regime + seasonality."""
        bias = str(market_context.get("overall_market_bias") or "").lower()
        bearish_regime = any(b in bias for b in ("bearish", "risk_off", "downtrend", "crisis"))
        seasonal_weak = _is_seasonally_weak(self.hcfg)

        # Drawdown-based escalation
        if drawdown <= -self.hcfg.hedge_heavy_dd:
            return "heavy"
        elif drawdown <= -self.hcfg.hedge_medium_dd:
            return "medium"
        elif drawdown <= -self.hcfg.hedge_light_dd:
            return "light"

        # Regime-based without drawdown yet
        if self.hcfg.hedge_on_bearish_regime and bearish_regime:
            return "light"

        # Seasonality
        if self.hcfg.hedge_on_seasonality and seasonal_weak:
            return "light"

        return "none"

    def _hedge_pct_for_level(self, level: str) -> float:
        return {
            "none": 0.0,
            "light": self.hcfg.light_hedge_pct,
            "medium": self.hcfg.medium_hedge_pct,
            "heavy": self.hcfg.heavy_hedge_pct,
        }.get(level, 0.0)

    def evaluate(
        self,
        positions: dict[str, Any],
        equity: float,
        market_context: dict,
        broker: Any,
    ) -> dict:
        """
        Full portfolio evaluation cycle. Returns a monitoring report dict.
        Emits hedge/rebalance orders directly through the broker.
        """
        report: dict = {
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "portfolio_value": 0.0,
            "drawdown": 0.0,
            "hedge_level": "none",
            "hedge_orders": [],
            "concentration_warnings": [],
            "seasonality_active": False,
            "regime": "",
            "actions_taken": [],
        }

        if not self.hcfg.enabled:
            return report

        # ── 1. Drawdown tracking ──────────────────────────────────────────────
        port_value, drawdown = _compute_portfolio_drawdown(positions, self._state)
        self._state.current_drawdown = drawdown
        report["portfolio_value"] = port_value
        report["drawdown"] = drawdown

        # ── 2. Regime ─────────────────────────────────────────────────────────
        bias = str(market_context.get("overall_market_bias") or "").lower()
        report["regime"] = bias

        # ── 3. Seasonality ────────────────────────────────────────────────────
        seasonal = _is_seasonally_weak(self.hcfg)
        self._state.seasonality_active = seasonal
        report["seasonality_active"] = seasonal
        if seasonal:
            _hedge_event("seasonality_active", {
                "month": datetime.now(ET).month,
                "weak_months": list(self.hcfg.weak_months),
            })

        # ── 4. Concentration check ────────────────────────────────────────────
        conc_warnings = _check_concentration(positions, equity, self.bcfg.max_single_stock_weight * 1.2)
        self._state.concentration_warnings = conc_warnings
        report["concentration_warnings"] = conc_warnings
        if conc_warnings:
            _hedge_event("concentration_warning", {"warnings": conc_warnings})

        # ── 5. Determine hedge level ──────────────────────────────────────────
        required_level = self._determine_hedge_level(drawdown, market_context)
        prior_level = self._state.hedge_level
        report["hedge_level"] = required_level

        _hedge_event("hedge_evaluation", {
            "drawdown": drawdown,
            "prior_level": prior_level,
            "required_level": required_level,
            "regime": bias,
            "seasonal": seasonal,
        })

        # ── 6. Adjust hedges if level changed ────────────────────────────────
        if required_level != prior_level:
            sectors = [
                str(getattr(p, "sector", "") or "") for p in positions.values()
            ]
            hedge_instrument = _select_hedge_instrument(sectors, self.hcfg)

            if required_level == "none" and prior_level != "none":
                # Remove all active hedges
                for h in self._state.active_hedges:
                    try:
                        contract = h.get("contract_symbol") or hedge_instrument
                        qty = int(h.get("qty", 0))
                        if qty > 0 and broker:
                            broker.submit_market_order(
                                symbol=contract, side="sell", qty=qty,
                                reason="bal_hedge_removed:conditions_improved",
                            )
                            report["hedge_orders"].append({"action": "remove", "symbol": contract, "qty": qty})
                            report["actions_taken"].append(f"removed_hedge:{contract}")
                    except Exception as _exc:
                        _hedge_event("hedge_remove_error", {"error": str(_exc)[:200]})
                self._state.active_hedges = []

            elif required_level != "none":
                # Size the put hedge
                hedge_dollars = equity * self._hedge_pct_for_level(required_level)
                _hedge_event("hedge_order_intent", {
                    "instrument": hedge_instrument,
                    "level": required_level,
                    "dollars": hedge_dollars,
                    "drawdown": drawdown,
                    "dte_range": f"{self.hcfg.put_min_dte}-{self.hcfg.put_max_dte}",
                    "target_delta": self.hcfg.put_delta,
                })
                # Record hedge intent — actual contract selection goes through Alpha Engine's
                # chain lookup at execution time (broker.get_options_chain)
                hedge_record = {
                    "instrument": hedge_instrument,
                    "type": "put",
                    "level": required_level,
                    "target_dollars": hedge_dollars,
                    "target_delta": self.hcfg.put_delta,
                    "dte_range": [self.hcfg.put_min_dte, self.hcfg.put_max_dte],
                    "triggered_by": {
                        "drawdown": drawdown,
                        "regime": bias,
                        "seasonal": seasonal,
                    },
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                self._state.active_hedges.append(hedge_record)
                report["hedge_orders"].append(hedge_record)
                report["actions_taken"].append(
                    f"hedge_{required_level}:{hedge_instrument}:${hedge_dollars:.0f}"
                )

            self._state.hedge_level = required_level

        self._state.last_hedge_check_utc = datetime.now(timezone.utc).isoformat()
        self._save_state()
        return report

    @property
    def hedge_level(self) -> str:
        return self._state.hedge_level

    @property
    def drawdown(self) -> float:
        return self._state.current_drawdown
