"""Shared, configurable transaction-cost model for equities.

Before this module: walkforward.py modeled slippage only (no commission, no
market impact), backtest.py modeled an options-specific spread/slippage/
commission stack with no market-impact term, and portfolio_optimizer.py had
no cost awareness whatsoever. This is the one place equity cost assumptions
live now; walk-forward evaluation should round-trip through it so a reported
Sharpe reflects a consistent, named cost regime.

All costs are expressed in basis points (bps) of trade notional. A square-root
market-impact model (impact scales with sqrt(participation), the standard
form used in execution-cost literature) adds a size-dependent term on top of
the fixed spread/slippage/commission assumptions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransactionCostConfig:
    spread_bps: float = 10.0
    slippage_bps: float = 5.0
    commission_bps: float = 1.0
    impact_coefficient: float = 10.0  # bps of impact at 100% participation (order_notional == ADV)

    def round_trip_cost_bps(self) -> float:
        """Fixed (size-independent) one-way cost: spread + slippage + commission."""
        return self.spread_bps + self.slippage_bps + self.commission_bps


# Named scenarios for cost-sensitivity testing. "baseline" keeps the
# slippage assumption that was already in use (5 bps) and adds the
# commission/impact terms that were previously entirely absent.
BASELINE_COSTS = TransactionCostConfig(spread_bps=10.0, slippage_bps=5.0, commission_bps=1.0, impact_coefficient=10.0)
ADVERSE_COSTS = TransactionCostConfig(spread_bps=20.0, slippage_bps=15.0, commission_bps=2.0, impact_coefficient=25.0)
SEVERE_COSTS = TransactionCostConfig(spread_bps=40.0, slippage_bps=35.0, commission_bps=3.0, impact_coefficient=60.0)

COST_SCENARIOS: dict[str, TransactionCostConfig] = {
    "baseline": BASELINE_COSTS,
    "adverse": ADVERSE_COSTS,
    "severe": SEVERE_COSTS,
}


def market_impact_bps(
    order_notional: float,
    average_dollar_volume: float,
    config: TransactionCostConfig,
) -> float:
    """Square-root market-impact model: impact (bps) scales with
    sqrt(order_notional / average_dollar_volume). Returns 0.0 for
    non-positive/unknown notional or ADV rather than raising -- a missing
    liquidity figure is a data/eligibility problem to be filtered upstream,
    not a reason to blow up cost estimation.
    """
    if order_notional <= 0 or average_dollar_volume <= 0:
        return 0.0
    participation = order_notional / average_dollar_volume
    return float(config.impact_coefficient * math.sqrt(participation))


def total_cost_bps(
    order_notional: float,
    average_dollar_volume: float,
    config: TransactionCostConfig,
) -> float:
    """One-way total cost in bps: fixed round-trip assumption (spread +
    slippage + commission) plus a size-dependent market-impact term."""
    return config.round_trip_cost_bps() + market_impact_bps(order_notional, average_dollar_volume, config)


def fixed_fee_to_bps(fee_dollars: float, notional: float) -> float:
    """Convert a flat per-trade/per-contract fee into bps of notional."""
    if notional <= 0:
        return 0.0
    return float(fee_dollars / notional * 10_000.0)


def total_cost_bps_vectorized(order_notional, average_dollar_volume, config: TransactionCostConfig):
    """Vectorized (numpy/pandas-friendly) version of total_cost_bps, for
    computing a per-row cost-aware label/feature over a whole dataframe
    without a slow Python-level .apply() loop. Accepts and returns
    array-likes; non-positive notional/ADV rows get impact=0, matching the
    scalar function's behavior."""
    notional = np.asarray(order_notional, dtype=float)
    adv = np.asarray(average_dollar_volume, dtype=float)
    valid = (notional > 0) & (adv > 0)
    participation = np.where(valid, np.divide(notional, adv, out=np.zeros_like(notional), where=valid), 0.0)
    impact = np.where(valid, config.impact_coefficient * np.sqrt(np.clip(participation, 0.0, None)), 0.0)
    return config.round_trip_cost_bps() + impact
