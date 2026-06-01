#!/usr/bin/env python3
"""
opportunity_finder.py

Opportunity Finder Main Orchestrator

Adaptive engine that continuously reviews the active trading universe,
removes stale/low-quality symbols, and replaces them with stocks most likely
to produce bullish momentum continuation and volume footprint imbalance setups.

Does NOT place orders. Feeds refreshed universes into downstream execution systems.

Importable: from opportunity_finder import run_opportunity_finder
Standalone: python opportunity_finder.py
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from config.env_loader import load_instance_env
from opportunity_filters import (
    passes_dollar_volume_filter,
    passes_mtf_alignment_filter,
    passes_price_filter,
    passes_rvol_filter,
    passes_ta_buy_filter,
)
from opportunity_logging import make_logger
from opportunity_rotation import apply_rotation
from opportunity_scoring import score_candidates_batch, OpportunityCandidate
from opportunity_state import evaluate_universe_staleness
from opportunity_utils import (
    clip01,
    get_data_tag,
    get_daily_outputs_dir,
    get_log_dir,
    get_out_dir,
    instance_name,
    load_csv_safe,
    load_json_safe,
    normalize_symbol,
    path_opportunity_events,
    path_opportunity_report,
    path_orderflow_candidates,
    to_float,
    write_json_safe,
    write_locked_universe,
    load_locked_universe,
)

PROJECT_ROOT = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OpportunityConfig:
    """Opportunity Finder configuration, loaded from environment variables."""

    enabled: bool
    replace_stale: bool
    min_rvol: float
    min_dollar_volume: float
    min_price: float
    max_price: float
    max_universe_size: int
    min_universe_size: int
    stale_threshold: float
    min_replacement_score: float
    max_replacements: int
    require_ta_buy: bool
    require_mtf: bool
    data_tag: str
    instance_name: str

    @classmethod
    def from_env(cls) -> OpportunityConfig:
        """Load all configuration from environment variables."""
        data_tag = get_data_tag()
        return cls(
            enabled=os.getenv("OPPORTUNITY_FINDER_ENABLED", "true").lower() in {"1", "true"},
            replace_stale=os.getenv("OPPORTUNITY_FINDER_REPLACE_STALE_NAMES", "true").lower() in {"1", "true"},
            min_rvol=float(os.getenv("MIN_RVOL", "2.0")),
            min_dollar_volume=float(os.getenv("MIN_DOLLAR_VOLUME", "20000000")),
            min_price=float(os.getenv("MIN_PRICE", "2")),
            max_price=float(os.getenv("MAX_PRICE", "750")),
            max_universe_size=int(os.getenv("MAX_UNIVERSE_SIZE", "25")),
            min_universe_size=int(os.getenv("MIN_UNIVERSE_SIZE", "8")),
            stale_threshold=float(os.getenv("STALE_SYMBOL_THRESHOLD", "65")),
            min_replacement_score=float(os.getenv("MIN_REPLACEMENT_SCORE", "75")),
            max_replacements=int(os.getenv("MAX_REPLACEMENTS_PER_CYCLE", "5")),
            require_ta_buy=os.getenv("REQUIRE_TV_TA_BUY_OR_STRONG_BUY", "true").lower() in {"1", "true"},
            require_mtf=os.getenv("REQUIRE_MULTI_TIMEFRAME_ALIGNMENT", "true").lower() in {"1", "true"},
            data_tag=data_tag,
            instance_name=instance_name(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Discovery: Fetch Replacement Candidates
# ─────────────────────────────────────────────────────────────────────────────


def _fetch_replacement_candidates(
    config: OpportunityConfig,
    existing_universe: set[str],
    log_fn: Callable[[str, dict], None],
) -> list[dict[str, Any]]:
    """
    Query TradingView screener for replacement candidates.

    Filters: price, RVOL, dollar volume, market cap, TA recommendations.
    Ordered by RVOL desc, limit 250.

    Returns list of raw TV screener rows.
    """
    try:
        from tradingview_screener import Query, col
    except ImportError:
        log_fn("external_http_error", {
            "module": "_fetch_replacement_candidates",
            "error": "tradingview-screener not installed",
        })
        return []

    try:
        # Build query — tradingview_screener >= 2.x API uses .where()/.select()/.limit()
        query = (
            Query()
            .where(
                col("type") == "stock",
                col("close") >= config.min_price,
                col("close") <= config.max_price,
                col("relative_volume_10d_calc") >= config.min_rvol,
                col("market_cap_basic") >= 50000000,
                col("Recommend.All|5") > 0,
                col("Recommend.All|15") > 0,
            )
            .select(
                "name", "close", "change", "volume", "relative_volume_10d_calc", "market_cap_basic",
                "close|1", "close|5", "close|15", "close|60",
                "volume|1", "volume|5", "volume|15",
                "MACD.macd|1", "MACD.signal|1", "MACD.macd|5", "MACD.signal|5",
                "MACD.macd|15", "MACD.signal|15",
                "Recommend.All", "Recommend.All|1", "Recommend.All|5", "Recommend.All|15", "Recommend.All|60",
                "RSI", "RSI|5", "RSI|15",
            )
            .order_by("relative_volume_10d_calc", ascending=False)
            .limit(250)
        )

        # Execute — returns (count, df) in tradingview_screener >= 2.x
        result = query.get_scanner_data()
        _, df = result if isinstance(result, tuple) else (None, result)
        if df is None or df.empty:
            return []

        # Convert to list of dicts, post-filter by dollar volume
        rows = []
        for _, row in df.iterrows():
            symbol = normalize_symbol(row.get("name"))
            if symbol in existing_universe:
                continue

            price = to_float(row.get("close"), 0.0)
            volume = to_float(row.get("volume"), 0.0)
            dollar_vol = price * volume if price > 0 else 0.0

            if dollar_vol < config.min_dollar_volume:
                continue

            rows.append(row.to_dict())

        return rows

    except Exception as e:
        log_fn("external_http_error", {
            "module": "_fetch_replacement_candidates",
            "error": str(e)[:300],
        })
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Report Building
# ─────────────────────────────────────────────────────────────────────────────


def _build_report(
    config: OpportunityConfig,
    original_universe: list[str],
    staleness_by_symbol: dict[str, Any],
    candidates: list[OpportunityCandidate],
    rotation: Any,  # RotationResult
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Build the full opportunity_finder_report.json payload."""
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_tag": config.data_tag,
        "instance_name": config.instance_name,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "config": {
            "enabled": config.enabled,
            "replace_stale": config.replace_stale,
            "min_rvol": config.min_rvol,
            "min_dollar_volume": config.min_dollar_volume,
            "min_price": config.min_price,
            "max_price": config.max_price,
            "max_universe_size": config.max_universe_size,
            "min_universe_size": config.min_universe_size,
            "stale_threshold": config.stale_threshold,
            "min_replacement_score": config.min_replacement_score,
            "max_replacements": config.max_replacements,
            "require_ta_buy": config.require_ta_buy,
            "require_mtf": config.require_mtf,
        },
        "universe_summary": {
            "original_size": len(original_universe),
            "final_size": len(rotation.final_universe),
            "removed_count": len(rotation.removed_symbols),
            "added_count": len(rotation.added_symbols),
            "rotation_count": rotation.rotation_count,
            "rotation_blocked_reason": rotation.rotation_blocked_reason,
        },
        "original_universe": original_universe,
        "final_universe": rotation.final_universe,
        "removed_symbols": rotation.removed_symbols,
        "added_symbols": rotation.added_symbols,
        "staleness_results": [
            {
                "symbol": stale.symbol,
                "staleness_score": round(stale.staleness_score, 1),
                "is_stale": stale.is_stale,
                "component_scores": {k: round(v, 2) for k, v in stale.component_scores.items()},
                "reasons": stale.reasons,
            }
            for stale in staleness_by_symbol.values()
        ],
        "opportunity_candidates": [
            {
                "symbol": c.symbol,
                "score": round(c.score, 1),
                "setup_type": c.setup_type,
                "rvol": round(c.rvol, 2),
                "premarket_volume": c.premarket_volume,
                "dollar_volume": round(c.dollar_volume, 0),
                "trend_strength": round(c.trend_strength, 1),
                "breakout_probability": round(c.breakout_probability, 1),
                "near_level": c.near_level,
                "ta_recommendation_1m": c.ta_recommendation_1m,
                "ta_recommendation_5m": c.ta_recommendation_5m,
                "ta_recommendation_15m": c.ta_recommendation_15m,
                "ta_recommendation_1h": c.ta_recommendation_1h,
                "ta_recommendation_1d": c.ta_recommendation_1d,
                "macd_alignment": c.macd_alignment,
                "ma_alignment": c.ma_alignment,
                "likely_footprint_imbalance": c.likely_footprint_imbalance,
            }
            for c in candidates[:30]  # Top 30 candidates
        ],
        "rotation_report": rotation.report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output Writing
# ─────────────────────────────────────────────────────────────────────────────


def _write_orderflow_candidates(
    candidates: list[OpportunityCandidate],
    rotation: Any,  # RotationResult
    config: OpportunityConfig,
    log_fn: Callable[[str, dict], None],
) -> None:
    """Write latest_orderflow_candidates.json (added candidates only)."""
    # Only candidates that made it into the added set
    added_set = set(rotation.added_symbols)
    added_candidates = [c for c in candidates if c.symbol in added_set]

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_tag": config.data_tag,
        "instance_name": config.instance_name,
        "candidates": [
            {
                "symbol": c.symbol,
                "score": round(c.score, 1),
                "setup_type": c.setup_type,
                "rvol": round(c.rvol, 2),
                "premarket_volume": c.premarket_volume,
                "dollar_volume": round(c.dollar_volume, 0),
                "trend_strength": round(c.trend_strength, 1),
                "breakout_probability": round(c.breakout_probability, 1),
                "near_level": c.near_level,
                "ta_recommendation_1m": c.ta_recommendation_1m,
                "ta_recommendation_5m": c.ta_recommendation_5m,
                "ta_recommendation_15m": c.ta_recommendation_15m,
                "ta_recommendation_1h": c.ta_recommendation_1h,
                "ta_recommendation_1d": c.ta_recommendation_1d,
                "macd_alignment": c.macd_alignment,
                "ma_alignment": c.ma_alignment,
                "likely_footprint_imbalance": c.likely_footprint_imbalance,
            }
            for c in added_candidates
        ],
    }

    # Primary path
    primary_path = path_orderflow_candidates(config.data_tag)
    write_json_safe(primary_path, payload, log_fn)

    # Compatibility alias (daily_system_outputs/)
    compat_path = get_daily_outputs_dir() / "latest_orderflow_candidates.json"
    write_json_safe(compat_path, payload, log_fn)


def _write_compatibility_aliases(
    report: dict[str, Any],
    config: OpportunityConfig,
    log_fn: Callable[[str, dict], None],
) -> None:
    """Write compatibility aliases to daily_system_outputs/."""
    out_dir = get_daily_outputs_dir()

    # opportunity_finder_report.json
    report_path = out_dir / "opportunity_finder_report.json"
    write_json_safe(report_path, report, log_fn)


# ─────────────────────────────────────────────────────────────────────────────
# Main Run Function
# ─────────────────────────────────────────────────────────────────────────────


def run_opportunity_finder(config: OpportunityConfig | None = None) -> dict[str, Any]:
    """
    Main orchestrator for the Opportunity Finder.

    Parameters
    ----------
    config : OpportunityConfig | None
        Configuration object. If None, loads from environment.

    Returns
    -------
    dict[str, Any]
        The full report dict.
    """
    # Load instance env
    load_instance_env()

    # Build config if not provided
    if config is None:
        config = OpportunityConfig.from_env()

    # Initialize logger
    events_path = path_opportunity_events(config.data_tag)
    log_fn = make_logger(events_path)

    # Start timing
    start_time = time.time()

    # Log start
    log_fn("opportunity_finder_start", {
        "data_tag": config.data_tag,
        "instance_name": config.instance_name,
        "enabled": config.enabled,
    })

    # ─────────────────────────────────────────────────────────────────────────────
    # Guard: if not enabled, return early
    # ─────────────────────────────────────────────────────────────────────────────

    if not config.enabled:
        log_fn("opportunity_finder_disabled", {})

        # Return empty report
        return {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_tag": config.data_tag,
            "instance_name": config.instance_name,
            "status": "disabled",
            "elapsed_seconds": round(time.time() - start_time, 2),
        }

    # ─────────────────────────────────────────────────────────────────────────────
    # Load current universe
    # ─────────────────────────────────────────────────────────────────────────────

    current_universe = load_locked_universe(config.data_tag)
    if not current_universe:
        current_universe = ["AAPL", "NVDA", "MSFT"]  # Fallback
        log_fn("external_http_error", {
            "module": "run_opportunity_finder",
            "error": "no_universe_loaded_using_fallback",
        })

    current_set = set(normalize_symbol(s) for s in current_universe)

    # ─────────────────────────────────────────────────────────────────────────────
    # Load market context
    # ─────────────────────────────────────────────────────────────────────────────

    market_context = load_json_safe(
        get_out_dir(config.data_tag) / f"{config.data_tag}_market_context.json", {}
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # Fetch replacement candidates from screener
    # ─────────────────────────────────────────────────────────────────────────────

    tv_rows = _fetch_replacement_candidates(config, current_set, log_fn)
    log_fn("external_http_error", {
        "module": "discovery",
        "candidates_found": len(tv_rows),
    })

    # ─────────────────────────────────────────────────────────────────────────────
    # Evaluate staleness of current universe
    # ─────────────────────────────────────────────────────────────────────────────

    staleness_by_symbol = evaluate_universe_staleness(
        current_universe, config.data_tag, config, log_fn
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # Load premarket plan data for scoring
    # ─────────────────────────────────────────────────────────────────────────────

    plan = load_json_safe(
        get_out_dir(config.data_tag) / f"{config.data_tag}_premarket_trade_plan.json", {}
    )
    plan_scores_by_symbol = {}
    for score_dict in plan.get("symbol_scores", []):
        sym = normalize_symbol(score_dict.get("symbol"))
        if sym:
            plan_scores_by_symbol[sym] = score_dict

    # Load intraday structure map
    structure_map = load_json_safe(
        get_out_dir(config.data_tag) / f"{config.data_tag}_intraday_structure_map.json", {}
    )
    structures_by_symbol = {}
    for struct in structure_map.get("structures", []):
        sym = normalize_symbol(struct.get("symbol"))
        if sym:
            structures_by_symbol[sym] = struct

    # ─────────────────────────────────────────────────────────────────────────────
    # Score replacement candidates
    # ─────────────────────────────────────────────────────────────────────────────

    candidates = score_candidates_batch(
        tv_rows, plan_scores_by_symbol, structures_by_symbol, config, log_fn
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # Apply rotation
    # ─────────────────────────────────────────────────────────────────────────────

    rotation = apply_rotation(
        current_universe, staleness_by_symbol, candidates, config, market_context, log_fn
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # Write updated universe file
    # ─────────────────────────────────────────────────────────────────────────────

    if rotation.final_universe != current_universe:
        write_locked_universe(rotation.final_universe, config.data_tag, log_fn)

    # ─────────────────────────────────────────────────────────────────────────────
    # Write output files
    # ─────────────────────────────────────────────────────────────────────────────

    elapsed = time.time() - start_time

    report = _build_report(
        config,
        current_universe,
        staleness_by_symbol,
        candidates,
        rotation,
        elapsed,
    )

    # Write primary report
    report_path = path_opportunity_report(config.data_tag)
    write_json_safe(report_path, report, log_fn)

    # Write candidates
    _write_orderflow_candidates(candidates, rotation, config, log_fn)

    # Write compatibility aliases
    _write_compatibility_aliases(report, config, log_fn)

    # ─────────────────────────────────────────────────────────────────────────────
    # Log completion
    # ─────────────────────────────────────────────────────────────────────────────

    log_fn("opportunity_finder_complete", {
        "data_tag": config.data_tag,
        "instance_name": config.instance_name,
        "final_universe_size": len(rotation.final_universe),
        "removed_count": len(rotation.removed_symbols),
        "added_count": len(rotation.added_symbols),
        "elapsed_seconds": round(elapsed, 2),
    })

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Entry Point
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import json
    import sys

    load_instance_env()
    result = run_opportunity_finder()

    final_universe = result.get("final_universe", [])
    print(f"\n[opportunity_finder] complete!")
    print(f"  Final universe size: {len(final_universe)}")
    print(f"  Symbols: {', '.join(final_universe[:10])}")
    print(f"  Elapsed: {result.get('elapsed_seconds', 0):.2f}s")

    sys.exit(0)
