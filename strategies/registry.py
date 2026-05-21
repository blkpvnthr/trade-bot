"""strategies/registry.py — central lookup for all strategy classes."""
from __future__ import annotations

from typing import Any

from strategies.base import BaseStrategy
from strategies.tux_mtf import TuxMTF, TuxMTFLongOnly
from strategies.pivot_points import PivotPointsStrategy
from strategies.orb_8am import ORB8amStrategy
from strategies.liquidity_sweep import LiquiditySweepStrategy
from strategies.flawless_victory import FlawlessVictoryStrategy

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "tux_mtf_original": TuxMTF,
    "tux_mtf_long_only": TuxMTFLongOnly,
    "pivot_points": PivotPointsStrategy,
    "orb_8am": ORB8amStrategy,
    "liquidity_sweep": LiquiditySweepStrategy,
    "flawless_victory": FlawlessVictoryStrategy,
}


def get_strategy(strategy_id: str, params: dict[str, Any] | None = None) -> BaseStrategy:
    """Instantiate a strategy by ID, optionally overriding default params."""
    cls = STRATEGY_REGISTRY.get(strategy_id)
    if cls is None:
        raise ValueError(
            f"Unknown strategy '{strategy_id}'. "
            f"Available: {sorted(STRATEGY_REGISTRY)}"
        )
    return cls(params=params)
