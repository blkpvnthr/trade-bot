#!/usr/bin/env python3
"""
opportunity_scoring.py

Score replacement candidate symbols to produce OpportunityCandidate objects.

Opportunity score (0-100) uses 9 weighted factors:
  1. rvol_score (0.20)
  2. dollar_volume_score (0.15)
  3. ta_alignment_score (0.15)
  4. macd_alignment_score (0.12)
  5. mtf_recommendation_score (0.12)
  6. price_position_score (0.10)
  7. breakout_probability (0.08)
  8. institutional_score (0.05)
  9. liquidity_score (0.03)

Setup type classification: vwap_reclaim, breakout, momentum_continuation, etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from opportunity_filters import (
    passes_dollar_volume_filter,
    passes_mtf_alignment_filter,
    passes_price_filter,
    passes_ta_buy_filter,
    passes_rvol_filter,
)
from opportunity_utils import (
    clip01,
    normalize_symbol,
    rec_score_to_label,
    to_float,
)


@dataclass
class OpportunityCandidate:
    """Complete specification of a replacement opportunity."""

    symbol: str
    score: float  # 0-100
    setup_type: str
    rvol: float
    price: float
    premarket_volume: int
    dollar_volume: float
    trend_strength: float  # 0-100
    breakout_probability: float  # 0-100
    near_level: str  # VWAP | SUPPORT | RESISTANCE | ORB_HIGH | NONE
    ta_recommendation_1m: str
    ta_recommendation_5m: str
    ta_recommendation_15m: str
    ta_recommendation_1h: str
    ta_recommendation_1d: str
    macd_alignment: bool
    ma_alignment: bool
    likely_footprint_imbalance: bool
    replacement_for: str | None = None
    reasons: list[str] = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


# ─────────────────────────────────────────────────────────────────────────────
# Setup Type Classification
# ─────────────────────────────────────────────────────────────────────────────


def _classify_setup_type(
    symbol: str,
    tv_row: dict[str, Any],
    structure: dict[str, Any] | None,
) -> tuple[str, str]:
    """
    Classify the opportunity setup type and nearest price level.

    Returns (setup_type, near_level)
    """
    if structure is None:
        return "generic", "NONE"

    symbol_norm = normalize_symbol(symbol)
    price = to_float(tv_row.get("close"), 0.0)
    rvol = to_float(tv_row.get("relative_volume_10d_calc"), 1.0)

    # Get TA fields
    rec_1m = rec_score_to_label(to_float(tv_row.get("Recommend.All|1"), 0.0))
    rec_5m = rec_score_to_label(to_float(tv_row.get("Recommend.All|5"), 0.0))
    rec_15m = rec_score_to_label(to_float(tv_row.get("Recommend.All|15"), 0.0))
    rec_1h = rec_score_to_label(to_float(tv_row.get("Recommend.All|60"), 0.0))
    rec_1d = rec_score_to_label(to_float(tv_row.get("Recommend.All"), 0.0))

    bullish_labels = {"BUY", "STRONG_BUY"}
    mtf_bullish_count = sum(1 for r in [rec_1m, rec_5m, rec_15m, rec_1h] if r in bullish_labels)

    # Check MACD bullish across TFs
    macd_1m = to_float(tv_row.get("MACD.macd|1"), 0.0) > to_float(tv_row.get("MACD.signal|1"), 0.0)
    macd_5m = to_float(tv_row.get("MACD.macd|5"), 0.0) > to_float(tv_row.get("MACD.signal|5"), 0.0)
    macd_15m = to_float(tv_row.get("MACD.macd|15"), 0.0) > to_float(tv_row.get("MACD.signal|15"), 0.0)
    macd_bullish_count = sum([macd_1m, macd_5m, macd_15m])

    # Get VWAP
    vwap_data = structure.get("vwap", {})
    vwap = to_float(vwap_data.get("vwap"), 0.0)
    price_above_vwap = bool(vwap_data.get("price_above_vwap", False))

    # Setup classification (priority order)

    # 1. VWAP reclaim near support
    if not price_above_vwap and price > vwap * 0.99 and rvol >= 1.5:
        return "vwap_reclaim_near_pivot_support", "VWAP"

    # 2. Breakout above resistance
    orb = structure.get("orb", {})
    orb_high = to_float(orb.get("high"), 0.0)
    if orb_high > 0 and price > orb_high and macd_bullish_count >= 2 and rvol >= 2.0:
        return "breakout_above_resistance", "ORB_HIGH"

    # 3. Momentum continuation
    if macd_bullish_count >= 3 and rvol >= 2.5 and mtf_bullish_count >= 3:
        return "momentum_continuation", "VWAP" if price_above_vwap else "SUPPORT"

    # 4. Mean reversion oversold
    rsi = to_float(tv_row.get("RSI"), 50.0)
    if rsi < 35 and price < to_float(tv_row.get("close|5"), 0.0):
        return "mean_reversion_oversold", "SUPPORT"

    # 5. Institutional accumulation
    if rvol >= 3.0 and mtf_bullish_count >= 3:
        return "institutional_accumulation", "VWAP"

    # 6. Volume surge consolidation
    if rvol >= 2.0 and abs(price - to_float(tv_row.get("close|1"), 0.0)) / price < 0.005:
        return "volume_surge_consolidation", "VWAP"

    # 7. Default
    return "generic", "NONE"


def _classify_near_level(
    symbol: str,
    tv_row: dict[str, Any],
    structure: dict[str, Any] | None,
) -> str:
    """
    Classify the nearest price level (VWAP/SUPPORT/RESISTANCE/ORB_HIGH/NONE).
    """
    if structure is None:
        return "NONE"

    price = to_float(tv_row.get("close"), 0.0)
    near_level_pct = 0.01  # 1% threshold

    # Check VWAP
    vwap = to_float(structure.get("vwap", {}).get("vwap"), 0.0)
    if vwap > 0 and abs(price - vwap) / vwap < near_level_pct:
        return "VWAP"

    # Check ORB
    orb = structure.get("orb", {})
    orb_high = to_float(orb.get("high"), 0.0)
    if orb_high > 0 and abs(price - orb_high) / orb_high < near_level_pct:
        return "ORB_HIGH"

    # Check S/R levels (simplified)
    support_levels = structure.get("support_levels", [])
    if support_levels and isinstance(support_levels, list):
        for level in support_levels:
            level_price = to_float(level.get("price") if isinstance(level, dict) else level, 0.0)
            if level_price > 0 and abs(price - level_price) / level_price < near_level_pct:
                return "SUPPORT"

    return "NONE"


# ─────────────────────────────────────────────────────────────────────────────
# Opportunity Scoring
# ─────────────────────────────────────────────────────────────────────────────


def score_candidate(
    symbol: str,
    tv_row: dict[str, Any],
    plan_scores: dict[str, Any] | None,
    structure: dict[str, Any] | None,
    config: Any,  # OpportunityConfig
) -> OpportunityCandidate:
    """
    Compute the opportunity score for a single candidate symbol.

    Parameters
    ----------
    symbol : str
        Symbol to score
    tv_row : dict
        TradingView screener row with Recommend.All fields, MACD, RSI, volumes, etc.
    plan_scores : dict | None
        Symbol score dict from premarket_trade_plan.json if available
    structure : dict | None
        Structure dict from intraday_structure_map.json if available
    config : OpportunityConfig
        Configuration with score thresholds

    Returns
    -------
    OpportunityCandidate
        Scored opportunity candidate
    """
    symbol = normalize_symbol(symbol)
    reasons = []

    # Basic data
    price = to_float(tv_row.get("close"), 0.0)
    volume = to_float(tv_row.get("volume"), 0.0)
    rvol = to_float(tv_row.get("relative_volume_10d_calc"), 1.0)
    dollar_volume = price * volume if volume > 0 else 0.0
    market_cap = to_float(tv_row.get("market_cap_basic"), 0.0)

    # TA recommendations
    rec_1m = rec_score_to_label(to_float(tv_row.get("Recommend.All|1"), 0.0))
    rec_5m = rec_score_to_label(to_float(tv_row.get("Recommend.All|5"), 0.0))
    rec_15m = rec_score_to_label(to_float(tv_row.get("Recommend.All|15"), 0.0))
    rec_1h = rec_score_to_label(to_float(tv_row.get("Recommend.All|60"), 0.0))
    rec_1d = rec_score_to_label(to_float(tv_row.get("Recommend.All"), 0.0))

    # MACD alignment
    macd_bullish = [
        to_float(tv_row.get("MACD.macd|1"), 0.0) > to_float(tv_row.get("MACD.signal|1"), 0.0),
        to_float(tv_row.get("MACD.macd|5"), 0.0) > to_float(tv_row.get("MACD.signal|5"), 0.0),
        to_float(tv_row.get("MACD.macd|15"), 0.0) > to_float(tv_row.get("MACD.signal|15"), 0.0),
    ]
    macd_alignment = sum(macd_bullish) >= 2

    # MA alignment (simplified: check vs closes)
    ma_alignment = True  # Placeholder: would need SMA data

    # ─────────────────────────────────────────────────────────────────────────────
    # Composite Opportunity Score (weights sum to 1.0 → scaled to 0-100)
    # ─────────────────────────────────────────────────────────────────────────────

    # Factor 1: RVOL score (0.20)
    rvol_score = clip01((rvol - 1.0) / 4.0)

    # Factor 2: Dollar volume score (0.15)
    dv_score = clip01(math.log10(max(1, dollar_volume / config.min_dollar_volume)) / 2.0)

    # Factor 3: TA alignment score (0.15)
    # Fraction of 5 TFs (1m/5m/15m/1h/1d) that are BUY/STRONG_BUY; STRONG_BUY = 1.5x
    bullish_labels = {"BUY", "STRONG_BUY"}
    recs = [rec_1m, rec_5m, rec_15m, rec_1h, rec_1d]
    ta_bullish = sum(1.5 if rec == "STRONG_BUY" else 1.0 for rec in recs if rec in bullish_labels)
    ta_alignment_score = clip01(ta_bullish / (5.0 * 1.5))  # Max is 5 * 1.5

    # Factor 4: MACD alignment score (0.12)
    macd_score = sum(macd_bullish) / 3.0

    # Factor 5: MTF recommendation score (0.12)
    # Average Recommend.All scores across 4 TFs
    rec_scores = [
        to_float(tv_row.get("Recommend.All|1"), 0.0),
        to_float(tv_row.get("Recommend.All|5"), 0.0),
        to_float(tv_row.get("Recommend.All|15"), 0.0),
        to_float(tv_row.get("Recommend.All|60"), 0.0),
    ]
    avg_rec = sum(rec_scores) / 4.0 if rec_scores else 0.0
    mtf_rec_score = clip01((avg_rec + 1.0) / 2.0)  # Map [-1, 1] → [0, 1]

    # Factor 6: Price position score (0.10)
    price_pos_score = 0.0
    if structure:
        vwap_data = structure.get("vwap", {})
        if vwap_data.get("price_above_vwap"):
            price_pos_score += 0.6
        vwap = to_float(vwap_data.get("vwap"), 0.0)
        if vwap > 0 and price > vwap * 0.99:
            price_pos_score += 0.3

        orb = structure.get("orb", {})
        orb_high = to_float(orb.get("high"), 0.0)
        if orb_high > 0 and price > orb_high:
            price_pos_score += 0.3
    price_pos_score = clip01(price_pos_score)

    # Factor 7: Breakout probability (0.08)
    breakout_prob = 0.5  # Default
    if plan_scores and "breakout_probability" in plan_scores:
        breakout_prob = to_float(plan_scores.get("breakout_probability"), 0.5)

    # Factor 8: Institutional score (0.05)
    institutional_score = 0.5  # Default proxy
    # Would be from llm_candidate_discovery if available

    # Factor 9: Liquidity score (0.03)
    if market_cap >= 10e9:
        liquidity_score = 1.0
    elif market_cap >= 1e9:
        liquidity_score = 0.80
    elif market_cap >= 500e6:
        liquidity_score = 0.65
    elif market_cap >= 100e6:
        liquidity_score = 0.45
    elif market_cap >= 50e6:
        liquidity_score = 0.25
    else:
        liquidity_score = 0.10

    # Weighted composite (weights sum to 1.0)
    composite_0_1 = (
        0.20 * rvol_score
        + 0.15 * dv_score
        + 0.15 * ta_alignment_score
        + 0.12 * macd_score
        + 0.12 * mtf_rec_score
        + 0.10 * price_pos_score
        + 0.08 * breakout_prob
        + 0.05 * institutional_score
        + 0.03 * liquidity_score
    )
    opportunity_score = clip01(composite_0_1) * 100.0

    # Classify setup type and near level
    setup_type, near_level = _classify_setup_type(symbol, tv_row, structure)
    trend_strength = opportunity_score  # Proxy: use opportunity score as trend strength

    # Likely footprint imbalance
    likely_footprint = rvol >= 3.0 and macd_alignment and avg_rec > 0.0

    # Premarket volume (placeholder if not in data)
    premarket_volume = to_float(tv_row.get("volume|1"), 0.0) if tv_row.get("volume|1") else 0

    return OpportunityCandidate(
        symbol=symbol,
        score=opportunity_score,
        setup_type=setup_type,
        rvol=rvol,
        price=price,
        premarket_volume=int(premarket_volume),
        dollar_volume=dollar_volume,
        trend_strength=clip01(trend_strength / 100.0) * 100.0,
        breakout_probability=breakout_prob * 100.0,
        near_level=near_level,
        ta_recommendation_1m=rec_1m,
        ta_recommendation_5m=rec_5m,
        ta_recommendation_15m=rec_15m,
        ta_recommendation_1h=rec_1h,
        ta_recommendation_1d=rec_1d,
        macd_alignment=macd_alignment,
        ma_alignment=ma_alignment,
        likely_footprint_imbalance=likely_footprint,
        replacement_for=None,
        reasons=reasons,
    )


def score_candidates_batch(
    tv_rows: list[dict[str, Any]],
    plan_scores_by_symbol: dict[str, dict[str, Any]],
    structures_by_symbol: dict[str, dict[str, Any]],
    config: Any,  # OpportunityConfig
    log_fn: Callable[[str, dict], None] | None = None,
) -> list[OpportunityCandidate]:
    """
    Score a list of TV screener rows.

    Parameters
    ----------
    tv_rows : list[dict]
        TV screener rows
    plan_scores_by_symbol : dict
        Symbol → premarket plan scores
    structures_by_symbol : dict
        Symbol → intraday structure
    config : OpportunityConfig
        Configuration
    log_fn : Callable | None
        Logger function

    Returns
    -------
    list[OpportunityCandidate]
        Sorted list by score desc
    """
    candidates = []

    for tv_row in tv_rows:
        try:
            symbol = normalize_symbol(tv_row.get("name"))
            if not symbol:
                continue

            plan_score = plan_scores_by_symbol.get(symbol)
            structure = structures_by_symbol.get(symbol)

            candidate = score_candidate(symbol, tv_row, plan_score, structure, config)
            candidates.append(candidate)

            if log_fn:
                log_fn("replacement_candidate_discovered", {
                    "symbol": symbol,
                    "score": round(candidate.score, 1),
                    "setup_type": candidate.setup_type,
                })

        except Exception as e:
            if log_fn:
                log_fn("external_http_error", {
                    "module": "score_candidates_batch",
                    "symbol": tv_row.get("name", "unknown"),
                    "error": str(e)[:300],
                })

    # Sort by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates
