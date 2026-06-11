#!/usr/bin/env python3
"""
trade_provenance.py — one shared classifier for trade ORIGIN and STRATEGY.

Every bot order is submitted with a client_order_id stamped by
``position_ownership.make_bot_coid(engine)`` → ``"qqbot-<engine>-<uuid>"`` (the ownership guard
is on by default). Manual trades placed in the Alpaca app/web carry their own client_order_id
(or none) and never match that prefix. So a filled order's client_order_id is a reliable,
broker-persisted provenance signal — we just have to read it.

This module centralises that read so the dashboard P&L path, the trade-journal report, and the
equity-curve builder all classify the same way. It is pure/stateless and never touches the broker.

ORIGIN
    automated_production  — placed by this bot (is_bot_coid → True)
    manual                — placed by a human (Alpaca app/web), or any non-bot fill
    automated_shadow      — simulated recommendation, never executed (rl_shadow_mode / research_platform)
    automated_backtest    — replay-only (backtest_outputs/); never a real fill
(Only production vs manual are derivable from a live Alpaca fill; shadow/backtest live in their own
already-isolated stores and are tagged at their source.)

Attribution rule for round-trips: a trip's origin = the origin of the lot that OPENED it, so the bot
closing a manually-opened position correctly stays in the *manual* bucket.
"""
from __future__ import annotations

from position_ownership import BOT_COID_PREFIX, is_bot_coid

# ── origin constants ────────────────────────────────────────────────────────────
ORIGIN_PRODUCTION = "automated_production"
ORIGIN_MANUAL = "manual"
ORIGIN_SHADOW = "automated_shadow"
ORIGIN_BACKTEST = "automated_backtest"

UNATTRIBUTED = "unattributed"

# engine token (as embedded in the coid, alnum-collapsed by make_bot_coid) → strategy_id
_ENGINE_TO_STRATEGY = {
    "insider": "insider_swing",
    "insidercopy": "insider_swing",
    "trumpcopy": "trump_copy",
    "trump": "trump_copy",
    "buffett": "buffett_accumulation",
    "buffettdca": "buffett_accumulation",
    "dca": "buffett_accumulation",
    "accumulation": "buffett_accumulation",
    "alpha": "alpha_call",
    "alphacall": "alpha_call",
    "scalp": "scalping",
    "scalping": "scalping",
    "notional": "buffett_accumulation",  # notional buys are the BAL accumulation path
}


def classify_origin(coid: str) -> str:
    """Origin of a live Alpaca fill from its client_order_id. Bot prefix → production, else manual."""
    return ORIGIN_PRODUCTION if is_bot_coid(coid) else ORIGIN_MANUAL


def engine_from_coid(coid: str) -> str:
    """Engine token embedded in a bot coid ('qqbot-<engine>-<rand>'), else UNATTRIBUTED.

    Legacy bot orders were stamped 'qqbot-<rand10>' with no engine segment → UNATTRIBUTED.
    Non-bot (manual) coids → UNATTRIBUTED (their strategy is, by definition, discretionary).
    """
    if not is_bot_coid(coid):
        return UNATTRIBUTED
    remainder = str(coid)[len(BOT_COID_PREFIX):].lstrip("-")
    parts = remainder.split("-")
    # 'engine-rand' → 2 parts (engine present); 'rand10' → 1 part (no engine)
    if len(parts) >= 2 and parts[0]:
        return parts[0]
    return UNATTRIBUTED


def strategy_from_engine(engine: str) -> str:
    """Map an engine token to a coarse strategy_id; unknown/unattributed pass through."""
    e = str(engine or "").lower()
    if not e or e == UNATTRIBUTED:
        return UNATTRIBUTED
    return _ENGINE_TO_STRATEGY.get(e, e)


def provenance_for_coid(coid: str) -> dict:
    """Convenience: {origin, engine, strategy_id} for a client_order_id."""
    origin = classify_origin(coid)
    engine = engine_from_coid(coid)
    return {"origin": origin, "engine": engine, "strategy_id": strategy_from_engine(engine)}
