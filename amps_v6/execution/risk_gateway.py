"""Independent deterministic risk gateway (V6 spec sections 17, 25-26, 29).

The LAST gate before the execution gateway. It is deterministic, fail-closed, and independent of
any analytical agent or LLM: even if every agent says "buy", the risk gateway can refuse. It does
NOT submit orders -- it approves/denies an OrderIntent and returns a sized, reasoned decision.

Reuses the repo's account-tiered risk thinking (trade.py build_account_risk_profile) conceptually,
but stays pure/importable so it can be unit-tested without a broker. Protective EXITS are never
gated by entry-alpha; this gateway governs ENTRIES and position increases.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from amps_v6.configuration.platform_state import PlatformState
from amps_v6.execution.order_intent import OrderIntent


@dataclass
class RiskLimits:
    max_open_positions: int = 5
    max_intraday_positions: int = 2
    max_swing_positions: int = 3
    max_intraday_capital_pct: float = 40.0
    max_swing_capital_pct: float = 60.0
    max_order_notional_pct: float = 20.0        # single order <= this % of equity
    cash_buffer_pct: float = 10.0               # keep this % of equity in cash
    risk_per_trade_pct: float = 0.5             # % of equity risked to the protective stop
    max_effective_leverage: float = 3.0         # per-position effective (daily) leverage ceiling
    allow_unauthorized_opposing_exposure: bool = False


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    approved_qty: float = 0.0
    sized_notional: float = 0.0


@dataclass
class PortfolioContext:
    equity: float
    buying_power: float
    open_positions: int = 0
    intraday_positions: int = 0
    swing_positions: int = 0
    intraday_capital_used: float = 0.0
    swing_capital_used: float = 0.0
    held_symbols: tuple = ()
    opposing_held: tuple = ()                    # symbols whose exposure opposes this intent


def _is_intraday(horizon: str) -> bool:
    return horizon in ("MICRO_SCALP", "INTRADAY_MOMENTUM", "EXTENDED_INTRADAY")


def size_by_risk(equity: float, limits: RiskLimits, entry_price: float,
                 protective_stop: Optional[float], effective_leverage: float = 1.0) -> float:
    """Fixed-fractional risk sizing: qty such that (entry - stop) * qty ~= risk_per_trade_pct of
    equity, then capped by the per-order notional ceiling. Never increases risk to chase returns."""
    if entry_price <= 0:
        return 0.0
    risk_budget = equity * (limits.risk_per_trade_pct / 100.0)
    if protective_stop and protective_stop > 0 and entry_price > protective_stop:
        per_share_risk = entry_price - protective_stop
        qty_risk = risk_budget / per_share_risk if per_share_risk > 0 else 0.0
    else:
        # no stop distance given -> fall back to notional cap only (conservative)
        qty_risk = float("inf")
    notional_cap = equity * (limits.max_order_notional_pct / 100.0)
    qty_notional = notional_cap / entry_price
    return max(0.0, min(qty_risk, qty_notional))


def evaluate(intent: OrderIntent, price: float, ctx: PortfolioContext, state: PlatformState,
             limits: Optional[RiskLimits] = None) -> RiskDecision:
    """Deterministic approve/deny + sizing for an ENTRY/increase intent. Fail-closed."""
    limits = limits or RiskLimits()
    if intent.side != "buy":
        return RiskDecision(False, "risk_gateway governs entries/increases only (side!=buy)")
    if price <= 0:
        return RiskDecision(False, "non-positive price")
    # kill switches / suppression
    if state.kill.get("global", False):
        return RiskDecision(False, "global kill switch TRIPPED")
    if state.kill.get("strategy", False) or state.kill.get("new_entries", False):
        return RiskDecision(False, "strategy/new_entries kill switch TRIPPED")
    if intent.symbol in state.symbol_suppressed:
        return RiskDecision(False, f"symbol {intent.symbol} suppressed")
    if intent.setup_family in state.setup_family_suppressed:
        return RiskDecision(False, f"setup_family {intent.setup_family} suppressed")
    if intent.effective_leverage and abs(intent.effective_leverage) > limits.max_effective_leverage:
        if state.kill.get("leveraged_etf", False):
            return RiskDecision(False, "leveraged_etf kill switch TRIPPED")
    # opposing-exposure conflict (spec 25) unless authorized hedge mode
    if ctx.opposing_held and not limits.allow_unauthorized_opposing_exposure:
        return RiskDecision(False, f"unauthorized opposing exposure vs {list(ctx.opposing_held)}")
    # overnight permission for swing holds
    if not _is_intraday(intent.approved_trading_horizon):
        if not intent.overnight_permission or not state.overnight_trading_eligible:
            return RiskDecision(False, "overnight permission/eligibility absent for swing entry")
        if state.kill.get("overnight_entries", False):
            return RiskDecision(False, "overnight_entries kill switch TRIPPED")
    # position-count limits
    if ctx.open_positions >= limits.max_open_positions:
        return RiskDecision(False, "max_open_positions reached")
    if _is_intraday(intent.approved_trading_horizon) and ctx.intraday_positions >= limits.max_intraday_positions:
        return RiskDecision(False, "max_intraday_positions reached")
    if not _is_intraday(intent.approved_trading_horizon) and ctx.swing_positions >= limits.max_swing_positions:
        return RiskDecision(False, "max_swing_positions reached")
    # sizing
    qty = size_by_risk(ctx.equity, limits, price, intent.protective_stop, intent.effective_leverage)
    if intent.qty > 0:
        qty = min(qty, intent.qty)        # never exceed requested
    if qty <= 0:
        return RiskDecision(False, "sized qty <= 0")
    notional = qty * price
    # capital bucket + cash buffer + buying power
    bucket_used = ctx.intraday_capital_used if _is_intraday(intent.approved_trading_horizon) else ctx.swing_capital_used
    bucket_cap = ctx.equity * ((limits.max_intraday_capital_pct if _is_intraday(intent.approved_trading_horizon)
                                else limits.max_swing_capital_pct) / 100.0)
    if bucket_used + notional > bucket_cap:
        return RiskDecision(False, "horizon capital bucket cap exceeded")
    min_cash = ctx.equity * (limits.cash_buffer_pct / 100.0)
    if ctx.buying_power - notional < min_cash:
        return RiskDecision(False, "cash buffer / buying power insufficient")
    return RiskDecision(True, "approved", approved_qty=qty, sized_notional=notional)
