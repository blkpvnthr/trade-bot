"""The ONE shared fill/slippage model. Champion and every challenger use the SAME instance, so
performance differences come from the strategy, never from optimistic fill assumptions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class FillModel:
    slippage_bps: float = 5.0            # per-side, in basis points of reference price
    commission_per_share: float = 0.0

    def fill(self, side: str, ref_price: float, qty: float = 1.0,
             adv: Optional[float] = None) -> Dict[str, float]:
        sign = 1.0 if side.lower() == "buy" else -1.0
        slip = ref_price * (self.slippage_bps / 1e4) * sign
        return {"price": ref_price + slip, "slippage": abs(slip),
                "commission": self.commission_per_share * qty}

    def round_trip_cost(self) -> float:
        """Fractional round-trip cost to subtract from a gross return (shared across portfolios)."""
        return 2.0 * self.slippage_bps / 1e4
