"""Typed order-intent contract (V6 spec section 29 extended contract).

An OrderIntent is what the strategy/portfolio layer PRODUCES and the risk gateway + execution
gateway CONSUME. It is deliberately broker-agnostic: no broker objects, no submission here. The
execution gateway is the only thing that turns an approved intent into a broker order.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class OrderIntent:
    symbol: str
    side: str                       # "buy" | "sell"
    qty: float
    reason: str
    # extended V6 contract
    setup_family: str = "unknown"
    original_trading_horizon: str = "NONE"
    approved_trading_horizon: str = "NONE"
    management_timeframe: str = "5m"
    expected_holding_duration: str = ""         # human-readable, e.g. "5 trading days"
    expected_net_return_bps: float = 0.0
    overnight_permission: bool = False
    overnight_risk_policy_version: str = "unsigned"
    underlying_exposure: str = ""
    effective_leverage: float = 1.0
    evidence_id: Optional[str] = None
    promotion_evidence_id: Optional[str] = None
    # execution plumbing (filled by the gateway; kept here for the audit record)
    client_order_id: Optional[str] = None
    entry_price_hint: Optional[float] = None
    protective_stop: Optional[float] = None
    profit_target: Optional[float] = None

    def as_dict(self) -> dict:
        return asdict(self)

    def notional(self, price: float) -> float:
        return abs(self.qty) * float(price)
