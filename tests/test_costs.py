"""Tests for costs.py: shared, configurable transaction-cost model."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from costs import (
    ADVERSE_COSTS,
    BASELINE_COSTS,
    COST_SCENARIOS,
    SEVERE_COSTS,
    TransactionCostConfig,
    fixed_fee_to_bps,
    market_impact_bps,
    total_cost_bps,
    total_cost_bps_vectorized,
)


def test_round_trip_cost_is_sum_of_components():
    cfg = TransactionCostConfig(spread_bps=10, slippage_bps=5, commission_bps=1)
    assert cfg.round_trip_cost_bps() == pytest.approx(16.0)


def test_market_impact_zero_for_missing_liquidity_data():
    cfg = BASELINE_COSTS
    assert market_impact_bps(order_notional=1000, average_dollar_volume=0, config=cfg) == 0.0
    assert market_impact_bps(order_notional=0, average_dollar_volume=1_000_000, config=cfg) == 0.0
    assert market_impact_bps(order_notional=-100, average_dollar_volume=1_000_000, config=cfg) == 0.0


def test_market_impact_scales_with_sqrt_of_participation():
    cfg = TransactionCostConfig(impact_coefficient=10.0)
    # 100% participation (order == ADV) -> impact == coefficient
    assert market_impact_bps(1_000_000, 1_000_000, cfg) == pytest.approx(10.0)
    # 4% participation -> sqrt(0.04) = 0.2 -> impact = 10 * 0.2 = 2.0
    assert market_impact_bps(40_000, 1_000_000, cfg) == pytest.approx(2.0)
    # 1% participation -> sqrt(0.01) = 0.1 -> impact = 1.0
    assert market_impact_bps(10_000, 1_000_000, cfg) == pytest.approx(1.0)


def test_market_impact_increasing_in_order_size():
    cfg = BASELINE_COSTS
    small = market_impact_bps(1_000, 10_000_000, cfg)
    large = market_impact_bps(500_000, 10_000_000, cfg)
    assert 0 <= small < large


def test_total_cost_combines_fixed_and_impact():
    cfg = TransactionCostConfig(spread_bps=10, slippage_bps=5, commission_bps=1, impact_coefficient=10)
    total = total_cost_bps(order_notional=40_000, average_dollar_volume=1_000_000, config=cfg)
    assert total == pytest.approx(16.0 + 2.0)  # round-trip 16 + impact 2.0 at 4% participation


def test_adverse_and_severe_scenarios_strictly_worse_than_baseline():
    notional, adv = 50_000, 2_000_000
    baseline_cost = total_cost_bps(notional, adv, BASELINE_COSTS)
    adverse_cost = total_cost_bps(notional, adv, ADVERSE_COSTS)
    severe_cost = total_cost_bps(notional, adv, SEVERE_COSTS)
    assert baseline_cost < adverse_cost < severe_cost


def test_cost_scenarios_registry_contains_all_three():
    assert set(COST_SCENARIOS) == {"baseline", "adverse", "severe"}
    assert COST_SCENARIOS["baseline"] is BASELINE_COSTS


def test_fixed_fee_to_bps_known_value():
    # $0.15 fee on a $1,500 notional trade = 1 bps
    assert fixed_fee_to_bps(0.15, 1_500) == pytest.approx(1.0)


def test_fixed_fee_to_bps_zero_notional_is_zero_not_inf():
    assert fixed_fee_to_bps(5.0, 0) == 0.0


def test_total_cost_bps_vectorized_matches_scalar_elementwise():
    notional = np.array([0.0, -100.0, 10_000.0, 40_000.0, 1_000_000.0])
    adv = np.array([1_000_000.0, 1_000_000.0, 0.0, 1_000_000.0, 1_000_000.0])
    got = total_cost_bps_vectorized(notional, adv, BASELINE_COSTS)
    expected = np.array([total_cost_bps(n, a, BASELINE_COSTS) for n, a in zip(notional, adv)])
    np.testing.assert_allclose(got, expected)


def test_total_cost_bps_vectorized_scales_with_participation():
    notional = np.array([10_000.0, 40_000.0, 1_000_000.0])
    adv = np.full(3, 1_000_000.0)
    got = total_cost_bps_vectorized(notional, adv, BASELINE_COSTS)
    assert got[0] < got[1] < got[2]
