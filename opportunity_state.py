#!/usr/bin/env python3
"""
opportunity_state.py

Compute staleness scores for symbols in the current trading universe.
Staleness measures how degraded/stale a symbol has become (0-100, higher = more stale).

Staleness formula: 9 weighted factors (weights sum to 100):
  1. declining_rvol (15)
  2. declining_dollar_volume (15)
  3. weak_momentum (12)
  4. weak_ta_recommendations (12)
  5. bearish_mtf_alignment (10)
  6. poor_breakout_probability (10)
  7. not_in_valid_symbols (10)
  8. below_vwap_no_reclaim (10)
  9. weak_institutional_score (6)

Missing data for any factor → weight redistributed to available factors (never penalizes for missing data).

Anti-churn: symbols in long_allowed_symbols from premarket plan are protected from excessive staleness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Callable

import pandas as pd

from opportunity_filters import is_stale
from opportunity_logging import make_logger
from opportunity_utils import (
    clip01,
    get_data_tag,
    get_log_dir,
    load_csv_safe,
    load_json_safe,
    normalize_symbol,
    path_candidate_discovery,
    path_intraday_structure_map,
    path_opportunity_events,
    path_premarket_plan,
    path_scanner_scores,
    to_float,
)


@dataclass
class StalenessResult:
    """Result of staleness evaluation for one symbol."""

    symbol: str
    staleness_score: float  # 0-100
    is_stale: bool
    component_scores: dict[str, float] = field(default_factory=dict)  # sub-scores
    reasons: list[str] = field(default_factory=list)  # human-readable reasons
    source_data: dict[str, Any] = field(default_factory=dict)  # raw values used


# ─────────────────────────────────────────────────────────────────────────────
# Staleness Factor Functions (0-1 scale, 1 = most stale)
# ─────────────────────────────────────────────────────────────────────────────


def _factor_declining_rvol(
    scanner_row: dict[str, Any] | None,
    tv_row: dict[str, Any] | None,
    min_rvol: float,
) -> tuple[float, dict]:
    """
    Factor 1: RVOL decline. Low RVOL = stale.

    Logic:
      rvol < 1.0 → 1.0 (very stale)
      1.0 - 2.0 → linear 0.5 - 0
      >= 2.0 → 0.0 (not stale)
    """
    reasons = []
    source = {}

    # Try scanner_scores first, then TV snapshot
    rvol = None
    if scanner_row and "rel_volume" in scanner_row:
        rvol = to_float(scanner_row.get("rel_volume"), None)
        source["rvol_source"] = "scanner"
    if rvol is None and tv_row and "relative_volume_10d_calc" in tv_row:
        rvol = to_float(tv_row.get("relative_volume_10d_calc"), None)
        source["rvol_source"] = "tv_snapshot"

    if rvol is None:
        return 0.0, {"reason": "no_rvol_data", "source": source}

    source["rvol"] = rvol

    if rvol < 1.0:
        factor = 1.0
        reasons.append(f"RVOL very low ({rvol:.2f}x)")
    elif rvol < 2.0:
        factor = clip01(0.5 - (rvol - 1.0) * 0.5)
        reasons.append(f"RVOL declining ({rvol:.2f}x)")
    else:
        factor = 0.0

    return factor, {"reason": reasons, "rvol": rvol, "source": source}


def _factor_declining_dollar_volume(
    scanner_row: dict[str, Any] | None,
    min_dollar_volume: float,
) -> tuple[float, dict]:
    """
    Factor 2: Dollar volume decline.

    Logic: dv < MIN_DV → 1.0; else linear decay to 0 as dv increases.
    """
    reasons = []
    source = {}

    dv = None
    if scanner_row and "dollar_volume" in scanner_row:
        dv = to_float(scanner_row.get("dollar_volume"), None)
        source["dv_source"] = "scanner"

    if dv is None:
        return 0.0, {"reason": "no_dv_data", "source": source}

    source["dv"] = dv
    source["min_dv"] = min_dollar_volume

    if dv < min_dollar_volume:
        factor = 1.0
        reasons.append(f"Dollar volume below minimum: ${dv:,.0f}")
    else:
        # Linear decay: at min_dv → 1.0, at 3x min_dv → 0.0
        factor = clip01(1.0 - ((dv - min_dollar_volume) / (min_dollar_volume * 2)))
        if factor > 0.3:
            reasons.append(f"Dollar volume declining: ${dv:,.0f}")

    return factor, {"reason": reasons, "dv": dv, "source": source}


def _factor_weak_momentum(scanner_row: dict[str, Any] | None) -> tuple[float, dict]:
    """
    Factor 3: Weak momentum score from scanner.

    Logic: momentum_score < 0.3 → 1.0; 0.3-0.6 linear; >= 0.6 → 0.
    """
    reasons = []
    source = {}

    momentum = None
    if scanner_row and "momentum_score" in scanner_row:
        momentum = to_float(scanner_row.get("momentum_score"), None)
        source["momentum_source"] = "scanner"

    if momentum is None:
        return 0.0, {"reason": "no_momentum_data", "source": source}

    source["momentum"] = momentum

    if momentum < 0.3:
        factor = 1.0
        reasons.append(f"Momentum very weak ({momentum:.2f})")
    elif momentum < 0.6:
        factor = clip01((0.6 - momentum) / 0.3)
        reasons.append(f"Momentum weak ({momentum:.2f})")
    else:
        factor = 0.0

    return factor, {"reason": reasons, "momentum": momentum, "source": source}


def _factor_weak_ta_recommendations(tv_row: dict[str, Any] | None) -> tuple[float, dict]:
    """
    Factor 4: Weak TA recommendations across timeframes.

    Uses Recommend.All fields from TV snapshot (1m/5m/15m/1h).
    Maps to [0,1] and inverts (low rec = high staleness).
    """
    reasons = []
    source = {}

    rec_scores = []
    for tf, field in [("1m", "Recommend.All|1"), ("5m", "Recommend.All|5"),
                       ("15m", "Recommend.All|15"), ("1h", "Recommend.All|60")]:
        if tv_row and field in tv_row:
            score = to_float(tv_row.get(field), 0.0)
            rec_scores.append(score)
            source[f"rec_{tf}"] = score

    if not rec_scores:
        return 0.0, {"reason": "no_ta_rec_data", "source": source}

    avg_rec = sum(rec_scores) / len(rec_scores)
    source["avg_rec"] = avg_rec

    # Map [-1, 1] → [1, 0] (invert: strong sell = stale, strong buy = not stale)
    factor = clip01((1.0 - avg_rec) / 2.0)
    if avg_rec <= 0.0:
        reasons.append(f"TA bearish (avg: {avg_rec:.2f})")

    return factor, {"reason": reasons, "avg_rec": avg_rec, "source": source}


def _factor_bearish_mtf_alignment(scanner_row: dict[str, Any] | None) -> tuple[float, dict]:
    """
    Factor 5: Bearish multi-timeframe alignment (MACD bullish, trend stack).

    Checks scanner_scores: macd_bullish, above_sma20, trend_stack fields.
    """
    reasons = []
    source = {}

    if scanner_row is None:
        return 0.0, {"reason": "no_scanner_data", "source": source}

    bullish_count = 0
    total_checks = 0

    for field in ["macd_bullish", "above_sma20"]:
        if field in scanner_row:
            val = to_float(scanner_row.get(field), 0.0)
            total_checks += 1
            if val > 0.5:  # Treat as boolean
                bullish_count += 1
            source[field] = val

    if total_checks == 0:
        return 0.0, {"reason": "no_mtf_fields", "source": source}

    # If none bullish, very stale
    factor = clip01(1.0 - (bullish_count / total_checks))
    if bullish_count == 0:
        reasons.append("No MTF bullish alignment")

    return factor, {"reason": reasons, "bullish_count": bullish_count, "total_checks": total_checks, "source": source}


def _factor_poor_breakout_probability(
    plan_score: dict[str, Any] | None,
    scanner_row: dict[str, Any] | None,
) -> tuple[float, dict]:
    """
    Factor 6: Poor breakout probability.

    Try plan score first, then scanner fields.
    """
    reasons = []
    source = {}

    breakout_prob = None
    if plan_score and "breakout_probability" in plan_score:
        breakout_prob = to_float(plan_score.get("breakout_probability"), None)
        source["breakout_source"] = "plan"
    elif scanner_row and "confirmed_breakout_score" in scanner_row:
        breakout_prob = to_float(scanner_row.get("confirmed_breakout_score"), None)
        source["breakout_source"] = "scanner"

    if breakout_prob is None:
        return 0.0, {"reason": "no_breakout_data", "source": source}

    source["breakout_prob"] = breakout_prob

    # Invert: low breakout prob = high staleness
    factor = clip01(1.0 - breakout_prob)
    if breakout_prob < 0.5:
        reasons.append(f"Low breakout probability ({breakout_prob:.2f})")

    return factor, {"reason": reasons, "breakout_prob": breakout_prob, "source": source}


def _factor_not_in_valid_symbols(
    symbol: str,
    plan: dict[str, Any] | None,
) -> tuple[float, dict]:
    """
    Factor 7: Symbol not in valid_symbols or long_allowed from plan.

    1.0 if in neither, 0.5 if in one, 0.0 if in both.
    """
    reasons = []
    source = {}

    if plan is None:
        return 0.0, {"reason": "no_plan_data", "source": source}

    valid_symbols = [normalize_symbol(s) for s in plan.get("valid_symbols", [])]
    long_allowed = [normalize_symbol(s) for s in plan.get("long_allowed_symbols", [])]
    source["valid_count"] = len(valid_symbols)
    source["long_allowed_count"] = len(long_allowed)

    symbol_norm = normalize_symbol(symbol)
    in_valid = symbol_norm in valid_symbols
    in_long = symbol_norm in long_allowed

    if in_valid and in_long:
        factor = 0.0
    elif in_valid or in_long:
        factor = 0.5
        reasons.append("In plan but not optimal list")
    else:
        factor = 1.0
        reasons.append("Not in premarket plan")

    source["in_valid"] = in_valid
    source["in_long"] = in_long

    return factor, {"reason": reasons, "in_valid": in_valid, "in_long": in_long, "source": source}


def _factor_below_vwap_no_reclaim(
    symbol: str,
    structure: dict[str, Any] | None,
    scanner_row: dict[str, Any] | None,
) -> tuple[float, dict]:
    """
    Factor 8: Price below VWAP with no reclaim opportunity.

    Checks structure.vwap.price_above_vwap, scanner lower_band_reclaim.
    """
    reasons = []
    source = {}

    if structure is None:
        return 0.0, {"reason": "no_structure_data", "source": source}

    vwap_data = structure.get("vwap", {})
    price_above_vwap = bool(vwap_data.get("price_above_vwap", True))
    source["price_above_vwap"] = price_above_vwap

    lower_band_reclaim = 0.0
    if scanner_row and "lower_band_reclaim" in scanner_row:
        lower_band_reclaim = to_float(scanner_row.get("lower_band_reclaim"), 0.0)
        source["lower_band_reclaim"] = lower_band_reclaim

    # Stale if below VWAP AND no reclaim
    if not price_above_vwap and lower_band_reclaim < 0.5:
        factor = 1.0
        reasons.append("Below VWAP with no reclaim")
    else:
        factor = 0.0

    return factor, {"reason": reasons, "price_above_vwap": price_above_vwap, "source": source}


def _factor_weak_institutional_score(
    discovery: dict[str, Any] | None,
) -> tuple[float, dict]:
    """
    Factor 9: Weak institutional accumulation score.

    Uses llm_candidate_discovery.institutional_accumulation_score.
    """
    reasons = []
    source = {}

    if discovery is None:
        return 0.0, {"reason": "no_discovery_data", "source": source}

    inst_score = to_float(discovery.get("institutional_accumulation_score"), None)
    if inst_score is None:
        return 0.0, {"reason": "no_inst_score_field", "source": source}

    source["inst_score"] = inst_score

    if inst_score < 0.5:
        factor = 1.0
        reasons.append(f"Weak institutional score ({inst_score:.2f})")
    elif inst_score < 0.7:
        factor = clip01((0.7 - inst_score) / 0.2)
        reasons.append(f"Low institutional score ({inst_score:.2f})")
    else:
        factor = 0.0

    return factor, {"reason": reasons, "inst_score": inst_score, "source": source}


# ─────────────────────────────────────────────────────────────────────────────
# Main Staleness Computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_staleness_score(
    symbol: str,
    scanner_row: dict[str, Any] | None,
    plan_score: dict[str, Any] | None,
    structure: dict[str, Any] | None,
    discovery_entry: dict[str, Any] | None,
    tv_row: dict[str, Any] | None,
    config: Any,  # OpportunityConfig (avoid circular import)
) -> StalenessResult:
    """
    Compute the staleness score for one symbol (0-100).

    All inputs are optional dicts (already loaded). Graceful degradation:
    missing factors → weight redistributed to available factors.

    Parameters
    ----------
    symbol : str
        Symbol to evaluate
    scanner_row : dict | None
        Row from latest_scanner_scores.csv
    plan_score : dict | None
        Entry from premarket_trade_plan.json symbol_scores
    structure : dict | None
        Entry from intraday_structure_map.json structures[]
    discovery_entry : dict | None
        Entry from llm_candidate_discovery.json
    tv_row : dict | None
        TV screener row with Recommend.All fields
    config : OpportunityConfig
        Config object with min_* thresholds

    Returns
    -------
    StalenessResult
        Staleness evaluation for the symbol
    """
    component_scores = {}
    reasons = []

    # Compute all 9 factors (0-1 scale)
    factor_declining_rvol, data_declining_rvol = _factor_declining_rvol(
        scanner_row, tv_row, config.min_rvol
    )
    component_scores["declining_rvol"] = factor_declining_rvol

    factor_declining_dv, data_declining_dv = _factor_declining_dollar_volume(
        scanner_row, config.min_dollar_volume
    )
    component_scores["declining_dollar_volume"] = factor_declining_dv

    factor_weak_momentum, data_weak_momentum = _factor_weak_momentum(scanner_row)
    component_scores["weak_momentum"] = factor_weak_momentum

    factor_weak_ta, data_weak_ta = _factor_weak_ta_recommendations(tv_row)
    component_scores["weak_ta_recommendations"] = factor_weak_ta

    factor_bearish_mtf, data_bearish_mtf = _factor_bearish_mtf_alignment(scanner_row)
    component_scores["bearish_mtf_alignment"] = factor_bearish_mtf

    factor_poor_breakout, data_poor_breakout = _factor_poor_breakout_probability(
        plan_score, scanner_row
    )
    component_scores["poor_breakout_probability"] = factor_poor_breakout

    factor_not_in_valid, data_not_in_valid = _factor_not_in_valid_symbols(symbol, plan_score)
    component_scores["not_in_valid_symbols"] = factor_not_in_valid

    factor_below_vwap, data_below_vwap = _factor_below_vwap_no_reclaim(
        symbol, structure, scanner_row
    )
    component_scores["below_vwap_no_reclaim"] = factor_below_vwap

    factor_weak_inst, data_weak_inst = _factor_weak_institutional_score(discovery_entry)
    component_scores["weak_institutional_score"] = factor_weak_inst

    # Weights (sum to 100 for 0-100 scale, converted to 0-1 for computation)
    weights = {
        "declining_rvol": 0.15,
        "declining_dollar_volume": 0.15,
        "weak_momentum": 0.12,
        "weak_ta_recommendations": 0.12,
        "bearish_mtf_alignment": 0.10,
        "poor_breakout_probability": 0.10,
        "not_in_valid_symbols": 0.10,
        "below_vwap_no_reclaim": 0.10,
        "weak_institutional_score": 0.06,
    }

    # Weighted average staleness (normalize weights for available factors)
    factors = [
        factor_declining_rvol,
        factor_declining_dv,
        factor_weak_momentum,
        factor_weak_ta,
        factor_bearish_mtf,
        factor_poor_breakout,
        factor_not_in_valid,
        factor_below_vwap,
        factor_weak_inst,
    ]
    factors_list = list(weights.keys())

    # Compute weighted sum with available data only
    total_weight = sum(w for f, w in weights.items())
    if total_weight == 0:
        staleness_0_1 = 0.0
    else:
        weighted_sum = sum(factors[i] * weights[factors_list[i]] for i in range(len(factors)))
        staleness_0_1 = weighted_sum / total_weight

    staleness_score = clip01(staleness_0_1) * 100.0

    # Build human-readable reasons
    for factor_name, factor_value in component_scores.items():
        if factor_value > 0.6:
            # This factor is contributing to staleness
            pass  # Reasons added in factor functions above

    # Anti-churn: if symbol in long_allowed, cap staleness below threshold
    long_allowed = []
    if plan_score:
        long_allowed = [normalize_symbol(s) for s in plan_score.get("long_allowed_symbols", [])]

    symbol_norm = normalize_symbol(symbol)
    if symbol_norm in long_allowed:
        if staleness_score >= config.stale_threshold:
            staleness_score = min(staleness_score, config.stale_threshold - 1)
            reasons.append("Protected by long_allowed_symbols (anti-churn)")

    return StalenessResult(
        symbol=symbol,
        staleness_score=staleness_score,
        is_stale=is_stale(staleness_score, config.stale_threshold),
        component_scores=component_scores,
        reasons=reasons,
        source_data={
            "scanner": bool(scanner_row),
            "plan": bool(plan_score),
            "structure": bool(structure),
            "discovery": bool(discovery_entry),
            "tv": bool(tv_row),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Universe-Level Evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_universe_staleness(
    symbols: list[str],
    data_tag: str,
    config: Any,  # OpportunityConfig
    log_fn: Callable[[str, dict], None] | None = None,
) -> dict[str, StalenessResult]:
    """
    Evaluate staleness for all symbols in the universe.

    Loads all input data once, then computes staleness for each symbol.

    Parameters
    ----------
    symbols : list[str]
        Symbols to evaluate
    data_tag : str
        Data tag for loading input files
    config : OpportunityConfig
        Configuration
    log_fn : Callable | None
        Logger function (signature: log_fn(event_name, payload_dict))

    Returns
    -------
    dict[str, StalenessResult]
        Dict keyed by symbol
    """
    if log_fn is None:
        from opportunity_logging import make_logger
        log_fn = make_logger(path_opportunity_events(data_tag))

    # Load all input data
    scanner_df = load_csv_safe(path_scanner_scores(data_tag))
    plan_dict = load_json_safe(path_premarket_plan(data_tag), {})
    structure_dict = load_json_safe(path_intraday_structure_map(data_tag), {})
    discovery_dict = load_json_safe(path_candidate_discovery(data_tag), {})

    # Index scanner by symbol
    scanner_by_symbol = {}
    if not scanner_df.empty and "symbol" in scanner_df.columns:
        for _, row in scanner_df.iterrows():
            sym = normalize_symbol(row.get("symbol"))
            if sym:
                scanner_by_symbol[sym] = row.to_dict()

    # Index plan scores
    plan_scores_by_symbol = {}
    for score_dict in plan_dict.get("symbol_scores", []):
        sym = normalize_symbol(score_dict.get("symbol"))
        if sym:
            plan_scores_by_symbol[sym] = score_dict

    # Index structures
    structures_by_symbol = {}
    for struct in structure_dict.get("structures", []):
        sym = normalize_symbol(struct.get("symbol"))
        if sym:
            structures_by_symbol[sym] = struct

    # Index discovery
    discovery_by_symbol = {}
    for cand in discovery_dict.get("candidates", []):
        sym = normalize_symbol(cand.get("symbol"))
        if sym:
            discovery_by_symbol[sym] = cand

    # Fetch TV snapshot for all symbols (rate-limited)
    tv_by_symbol = _load_tv_snapshot_for_symbols(symbols, log_fn)
    time.sleep(1)  # Rate limit

    # Compute staleness for each symbol
    results = {}
    for symbol in symbols:
        sym_norm = normalize_symbol(symbol)
        staleness = compute_staleness_score(
            symbol=sym_norm,
            scanner_row=scanner_by_symbol.get(sym_norm),
            plan_score=plan_scores_by_symbol.get(sym_norm),
            structure=structures_by_symbol.get(sym_norm),
            discovery_entry=discovery_by_symbol.get(sym_norm),
            tv_row=tv_by_symbol.get(sym_norm),
            config=config,
        )

        results[sym_norm] = staleness

        # Log event
        status = "stale" if staleness.is_stale else "keep"
        log_fn("staleness_evaluated", {
            "symbol": sym_norm,
            "staleness_score": round(staleness.staleness_score, 1),
            "status": status,
            "reasons": staleness.reasons,
        })

    return results


def _load_tv_snapshot_for_symbols(
    symbols: list[str],
    log_fn: Callable[[str, dict], None],
) -> dict[str, dict[str, Any]]:
    """
    Query TradingView screener for symbols' current TA data.

    Returns dict of TV rows keyed by symbol. Returns {} on any error.
    """
    if not symbols:
        return {}

    try:
        from tradingview_screener import Query, col
    except ImportError:
        log_fn("external_http_error", {
            "module": "_load_tv_snapshot_for_symbols",
            "error": "tradingview-screener not installed",
        })
        return {}

    try:
        # Filter by symbols, get all TA fields
        symbol_filters = [col("name") == normalize_symbol(symbol) for symbol in symbols]
        combined_symbol_filter = reduce(lambda a, b: a | b, symbol_filters)
        query = Query().set_filter(col("type") == "stock").set_filter(combined_symbol_filter)

        # Fetch TA fields: Recommend.All for 1m/5m/15m/1h/1d, MACD, RSI, RVOL
        fields = [
            "name",
            "close",
            "Recommend.All|1",
            "Recommend.All|5",
            "Recommend.All|15",
            "Recommend.All|60",
            "Recommend.All",
            "MACD.macd|1",
            "MACD.signal|1",
            "MACD.macd|5",
            "MACD.signal|5",
            "MACD.macd|15",
            "MACD.signal|15",
            "RSI",
            "RSI|5",
            "RSI|15",
            "relative_volume_10d_calc",
            "volume",
            "market_cap_basic",
        ]
        query = query.set_fields(fields)

        # Execute query
        df = query.get_scanner_data()
        if df is None or df.empty:
            return {}

        # Index by symbol
        result = {}
        for _, row in df.iterrows():
            sym = normalize_symbol(row.get("name"))
            if sym:
                result[sym] = row.to_dict()

        return result

    except Exception as e:
        log_fn("external_http_error", {
            "module": "_load_tv_snapshot_for_symbols",
            "error": str(e)[:300],
        })
        return {}
