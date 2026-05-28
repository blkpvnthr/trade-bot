#!/usr/bin/env python3
"""
opportunity_filters.py

Stateless filter predicates for the Opportunity Finder.
Pure functions — no I/O, no side effects.
Used by opportunity_state.py, opportunity_scoring.py, and opportunity_rotation.py.
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────
# Price & Volume Filters
# ─────────────────────────────────────────────────────────────────────────────


def passes_price_filter(price: float, min_price: float, max_price: float) -> bool:
    """
    Check if price is within [min_price, max_price].

    Parameters
    ----------
    price : float
        Current price
    min_price : float
        Minimum acceptable price
    max_price : float
        Maximum acceptable price

    Returns
    -------
    bool
        True if min_price <= price <= max_price
    """
    return min_price <= price <= max_price


def passes_rvol_filter(rvol: float, min_rvol: float) -> bool:
    """
    Check if RVOL (relative volume) meets minimum threshold.

    Parameters
    ----------
    rvol : float
        Relative volume ratio (e.g., 2.5 = 2.5x avg)
    min_rvol : float
        Minimum acceptable RVOL

    Returns
    -------
    bool
        True if rvol >= min_rvol
    """
    return rvol >= min_rvol


def passes_dollar_volume_filter(dollar_volume: float, min_dollar_volume: float) -> bool:
    """
    Check if dollar volume meets minimum threshold.

    Parameters
    ----------
    dollar_volume : float
        Dollar volume (price * volume)
    min_dollar_volume : float
        Minimum acceptable dollar volume

    Returns
    -------
    bool
        True if dollar_volume >= min_dollar_volume
    """
    return dollar_volume >= min_dollar_volume


# ─────────────────────────────────────────────────────────────────────────────
# TradingView TA Filters
# ─────────────────────────────────────────────────────────────────────────────


def passes_ta_buy_filter(
    rec_1m: str,
    rec_5m: str,
    rec_15m: str,
    require_buy_or_strong_buy: bool,
) -> bool:
    """
    Check if TA recommendations meet minimum bullish threshold.

    If require_buy_or_strong_buy=True:
      At least one of (rec_1m, rec_5m, rec_15m) must be BUY or STRONG_BUY.
    Otherwise:
      Always returns True (no filter applied).

    Parameters
    ----------
    rec_1m : str
        1m recommendation label (e.g., "BUY", "STRONG_BUY", "NEUTRAL")
    rec_5m : str
        5m recommendation label
    rec_15m : str
        15m recommendation label
    require_buy_or_strong_buy : bool
        If True, enforce the bullish filter

    Returns
    -------
    bool
        True if filter passes
    """
    if not require_buy_or_strong_buy:
        return True

    bullish_labels = {"BUY", "STRONG_BUY"}
    recs = [rec_1m, rec_5m, rec_15m]
    return any(rec in bullish_labels for rec in recs)


def passes_mtf_alignment_filter(
    rec_1m: str,
    rec_5m: str,
    rec_15m: str,
    rec_1h: str,
    require_mtf_alignment: bool,
) -> bool:
    """
    Check if multi-timeframe TA recommendations are aligned.

    If require_mtf_alignment=True:
      At least 3 of 4 timeframes must be BUY or STRONG_BUY.
    Otherwise:
      Always returns True (no filter applied).

    Parameters
    ----------
    rec_1m : str
        1m recommendation label
    rec_5m : str
        5m recommendation label
    rec_15m : str
        15m recommendation label
    rec_1h : str
        1h recommendation label
    require_mtf_alignment : bool
        If True, enforce the alignment filter

    Returns
    -------
    bool
        True if filter passes
    """
    if not require_mtf_alignment:
        return True

    bullish_labels = {"BUY", "STRONG_BUY"}
    recs = [rec_1m, rec_5m, rec_15m, rec_1h]
    bullish_count = sum(1 for rec in recs if rec in bullish_labels)
    return bullish_count >= 3


# ─────────────────────────────────────────────────────────────────────────────
# Staleness & Score Filters
# ─────────────────────────────────────────────────────────────────────────────


def is_stale(staleness_score: float, threshold: float) -> bool:
    """
    Check if staleness score exceeds threshold.

    Parameters
    ----------
    staleness_score : float
        Staleness score 0-100 (higher = more stale)
    threshold : float
        Staleness threshold

    Returns
    -------
    bool
        True if staleness_score >= threshold
    """
    return staleness_score >= threshold


def is_strong_enough(opportunity_score: float, min_score: float) -> bool:
    """
    Check if opportunity score meets minimum threshold.

    Parameters
    ----------
    opportunity_score : float
        Opportunity score 0-100
    min_score : float
        Minimum acceptable score

    Returns
    -------
    bool
        True if opportunity_score >= min_score
    """
    return opportunity_score >= min_score


# ─────────────────────────────────────────────────────────────────────────────
# Exclusion Filters
# ─────────────────────────────────────────────────────────────────────────────


def is_excluded_from_rotation(symbol: str, exclusion_list: list[str]) -> bool:
    """
    Check if symbol is in the exclusion list.

    Parameters
    ----------
    symbol : str
        Symbol to check
    exclusion_list : list[str]
        List of symbols to exclude (case-insensitive comparison)

    Returns
    -------
    bool
        True if symbol is in exclusion_list
    """
    symbol_upper = symbol.upper()
    return any(exc.upper() == symbol_upper for exc in exclusion_list)
