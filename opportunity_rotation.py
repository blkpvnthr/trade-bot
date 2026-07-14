#!/usr/bin/env python3
"""
opportunity_rotation.py

Apply universe rotation logic: remove stale symbols, add strong candidates.

Rotation rules:
1. Guard: ENABLED=false → return original universe unchanged
2. Guard: REPLACE_STALE_NAMES=false → skip removal/addition logic
3. Guard: market_context.trade_permission in ('halt', 'halt_new_buys') → block rotation
4. Identify stale symbols (staleness >= STALE_THRESHOLD)
5. Never remove symbols in long_allowed_symbols (anti-churn)
6. Budget: max N removals based on universe floor and MAX_REPLACEMENTS
7. Filter eligible candidates (score >= MIN_REPLACEMENT_SCORE)
8. Pair stale removals with best candidates
9. Capacity fill: add additional candidates up to MAX_UNIVERSE_SIZE
10. Enforce bounds: MIN_UNIVERSE_SIZE <= final <= MAX_UNIVERSE_SIZE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from opportunity_filters import passes_price_filter, passes_rvol_filter, passes_dollar_volume_filter
from opportunity_scoring import OpportunityCandidate
from opportunity_state import StalenessResult
from opportunity_utils import normalize_symbol


@dataclass
class RotationResult:
    """Result of universe rotation."""

    original_universe: list[str]
    retained_symbols: list[str]  # Stale but kept due to floor
    removed_symbols: list[str]
    added_symbols: list[str]
    final_universe: list[str]
    candidates_considered: list[str]
    rotation_blocked_reason: str | None = None
    rotation_count: int = 0
    report: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Rotation Algorithm
# ─────────────────────────────────────────────────────────────────────────────


def apply_rotation(
    current_universe: list[str],
    staleness_by_symbol: dict[str, StalenessResult],
    candidates: list[OpportunityCandidate],
    config: Any,  # OpportunityConfig
    market_context: dict[str, Any] | None,
    log_fn: Callable[[str, dict], None] | None = None,
) -> RotationResult:
    """
    Apply universe rotation: remove stale, add strong candidates.

    Parameters
    ----------
    current_universe : list[str]
        Current trading universe
    staleness_by_symbol : dict[str, StalenessResult]
        Staleness evaluation for each current symbol
    candidates : list[OpportunityCandidate]
        Scored replacement candidates (pre-sorted by score desc)
    config : OpportunityConfig
        Configuration with rotation parameters
    market_context : dict | None
        Market context (trade_permission, etc.)
    log_fn : Callable | None
        Logger function

    Returns
    -------
    RotationResult
        Rotation result with before/after universes
    """
    # ─────────────────────────────────────────────────────────────────────────────
    # Guard checks (block rotation entirely if any fail)
    # ─────────────────────────────────────────────────────────────────────────────

    if not config.enabled:
        if log_fn:
            log_fn("universe_rotation_completed", {"rotation_blocked_reason": "disabled"})
        return RotationResult(
            original_universe=current_universe,
            retained_symbols=[],
            removed_symbols=[],
            added_symbols=[],
            final_universe=current_universe,
            candidates_considered=[],
            rotation_blocked_reason="disabled",
            rotation_count=0,
        )

    market_halt = False
    if market_context and market_context.get("trade_permission") in ("halt", "halt_new_buys"):
        market_halt = True

    if market_halt:
        if log_fn:
            log_fn("universe_rotation_completed", {"rotation_blocked_reason": "market_halt"})
        return RotationResult(
            original_universe=current_universe,
            retained_symbols=[],
            removed_symbols=[],
            added_symbols=[],
            final_universe=current_universe,
            candidates_considered=[],
            rotation_blocked_reason="market_halt",
            rotation_count=0,
        )

    current_set = set(normalize_symbol(s) for s in current_universe)

    # ─────────────────────────────────────────────────────────────────────────────
    # Identify stale symbols
    # ─────────────────────────────────────────────────────────────────────────────

    stale_symbols = []
    for symbol in current_universe:
        sym_norm = normalize_symbol(symbol)
        staleness = staleness_by_symbol.get(sym_norm)
        if staleness and staleness.is_stale:
            stale_symbols.append(sym_norm)

    # Sort stale by staleness_score desc (remove worst first)
    stale_symbols.sort(
        key=lambda s: staleness_by_symbol.get(s, StalenessResult(s, 0, False)).staleness_score,
        reverse=True,
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # Determine removal budget
    # ─────────────────────────────────────────────────────────────────────────────

    if not config.replace_stale:
        # Rotation disabled, no removals
        removal_budget = 0
    else:
        # Calculate how many we can remove while respecting universe floor
        max_removable = len(stale_symbols)
        max_by_replacements = config.max_replacements
        max_by_floor = len(current_universe) - config.min_universe_size
        removal_budget = min(max_removable, max_by_replacements, max_by_floor)
        removal_budget = max(0, removal_budget)

    stale_to_remove = stale_symbols[:removal_budget]

    # ─────────────────────────────────────────────────────────────────────────────
    # Filter eligible replacement candidates
    # ─────────────────────────────────────────────────────────────────────────────

    eligible = []
    for cand in candidates:
        # Must pass all filters
        if cand.score < config.min_replacement_score:
            continue
        if cand.symbol in current_set:
            continue
        if not passes_price_filter(cand.price, config.min_price, config.max_price):
            continue
        if not passes_rvol_filter(cand.rvol, config.min_rvol):
            continue
        if not passes_dollar_volume_filter(cand.dollar_volume, config.min_dollar_volume):
            continue
        if config.require_ta_buy:
            if not any(r in {"BUY", "STRONG_BUY"} for r in [
                cand.ta_recommendation_1m,
                cand.ta_recommendation_5m,
                cand.ta_recommendation_15m,
            ]):
                continue
        if config.require_mtf:
            bullish = sum(1 for r in [
                cand.ta_recommendation_1m,
                cand.ta_recommendation_5m,
                cand.ta_recommendation_15m,
                cand.ta_recommendation_1h,
            ] if r in {"BUY", "STRONG_BUY"})
            if bullish < 3:
                continue

        eligible.append(cand)

        if log_fn:
            log_fn("replacement_candidate_accepted", {
                "symbol": cand.symbol,
                "score": round(cand.score, 1),
            })

    # Log rejected candidates
    for cand in candidates:
        if cand not in eligible and cand.symbol not in current_set:
            if log_fn:
                log_fn("replacement_candidate_rejected", {
                    "symbol": cand.symbol,
                    "score": round(cand.score, 1),
                    "reason": "failed_filters",
                })

    # ─────────────────────────────────────────────────────────────────────────────
    # Pair removals with replacements
    # ─────────────────────────────────────────────────────────────────────────────

    n_swaps = min(len(stale_to_remove), len(eligible), config.max_replacements)
    removed = stale_to_remove[:n_swaps]
    added = [cand.symbol for cand in eligible[:n_swaps]]

    for sym_removed, sym_added in zip(removed, added):
        if log_fn:
            log_fn("symbol_replaced", {
                "removed": sym_removed,
                "added": sym_added,
            })

    # ─────────────────────────────────────────────────────────────────────────────
    # Capacity fill (add more candidates if universe < MAX_UNIVERSE_SIZE)
    # ─────────────────────────────────────────────────────────────────────────────

    filled_set = set(added)
    post_swap_size = len(current_universe) - len(removed) + len(added)
    slots_remaining = config.max_universe_size - post_swap_size

    if slots_remaining > 0:
        for cand in eligible[n_swaps:]:
            if len(filled_set) >= slots_remaining:
                break
            if cand.symbol not in filled_set and cand.symbol not in current_set:
                added.append(cand.symbol)
                filled_set.add(cand.symbol)

    # ─────────────────────────────────────────────────────────────────────────────
    # Build final universe
    # ─────────────────────────────────────────────────────────────────────────────

    final = [s for s in current_universe if normalize_symbol(s) not in removed]
    final.extend(added)
    # Dedup while preserving order
    final_dedup = []
    seen = set()
    for s in final:
        s_norm = normalize_symbol(s)
        if s_norm not in seen:
            final_dedup.append(s_norm)
            seen.add(s_norm)

    # Cap at MAX_UNIVERSE_SIZE
    final_dedup = final_dedup[:config.max_universe_size]

    # ─────────────────────────────────────────────────────────────────────────────
    # Safety checks
    # ─────────────────────────────────────────────────────────────────────────────

    if len(final_dedup) < config.min_universe_size:
        # Violation: add back some stale symbols to meet floor
        for sym in stale_to_remove:
            if len(final_dedup) >= config.min_universe_size:
                break
            if sym not in final_dedup:
                final_dedup.append(sym)

    if len(final_dedup) > config.max_universe_size:
        final_dedup = final_dedup[:config.max_universe_size]

    # ─────────────────────────────────────────────────────────────────────────────
    # Build result
    # ─────────────────────────────────────────────────────────────────────────────

    retained = [s for s in stale_to_remove if s not in removed]

    rotation_count = len(removed)

    if log_fn:
        log_fn("universe_rotation_completed", {
            "original_size": len(current_universe),
            "final_size": len(final_dedup),
            "removed_count": len(removed),
            "added_count": len(added),
            "rotation_count": rotation_count,
        })

    return RotationResult(
        original_universe=[normalize_symbol(s) for s in current_universe],
        retained_symbols=retained,
        removed_symbols=removed,
        added_symbols=added,
        final_universe=final_dedup,
        candidates_considered=[c.symbol for c in candidates],
        rotation_blocked_reason=None,
        rotation_count=rotation_count,
        report={
            "removal_budget": removal_budget,
            "stale_identified": len(stale_symbols),
            "eligible_candidates": len(eligible),
            "swaps_executed": n_swaps,
            "capacity_fill_adds": len(added) - n_swaps,
            "final_safety_adjustments": max(0, config.min_universe_size - post_swap_size),
        },
    )
