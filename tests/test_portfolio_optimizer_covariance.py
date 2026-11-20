"""Tests for portfolio_optimizer.py's Objective 2 covariance-aware sizing:
shrinkage covariance, exposure-group/leveraged caps, turnover penalty,
conservative deterministic fallback. All additive -- optimize_weights()/
optimize_portfolio() (the existing, live-default path) must be unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

import portfolio_optimizer as po
import universe as uni


def _candidates(symbols=None, **overrides):
    symbols = symbols or ["AAPL", "MSFT", "JPM", "XOM", "QQQ"]
    n = len(symbols)
    base = {
        "symbol": symbols,
        "price": [100.0] * n,
        "trade_confidence": [0.7] * n,
        "continuation_prob": [0.65] * n,
        "model_score": [0.6] * n,
        "expected_return_5d": [0.02] * n,
        "expected_sharpe_score": [1.0] * n,
        "trend_score": [0.6] * n,
        "momentum_score": [0.5] * n,
        "volatility20": [0.02] * n,
        "confluence_reversal_risk": [0.1] * n,
        "bear": [0] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _returns_history(symbols, n_days=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    return pd.DataFrame(rng.normal(0, 0.015, size=(n_days, len(symbols))), index=dates, columns=symbols)


# =============================================================================
# single_expected_score: de-duplication of correlated confidence channels
# =============================================================================

def test_single_expected_score_does_not_multiply_prob_channels():
    """The old pipeline computes optimizer_score * continuation_prob *
    model_score, then allocation_score * trade_confidence**power -- four
    multiplicative passes. single_expected_score must use an AVERAGE (one
    channel), not a product, so scaling all three confidence-like inputs
    down by the same factor scales the result by roughly that factor, not
    that factor cubed/quartic."""
    config = po.OptimizerConfig()
    high = _candidates(continuation_prob=[0.9] * 5, model_score=[0.9] * 5, trade_confidence=[0.9] * 5)
    low = _candidates(continuation_prob=[0.3] * 5, model_score=[0.3] * 5, trade_confidence=[0.3] * 5)

    score_high = po.single_expected_score(high, config).iloc[0]
    score_low = po.single_expected_score(low, config).iloc[0]

    # Under multiplication (0.9^3=0.729 vs 0.3^3=0.027), ratio would be ~27x.
    # Under averaging, the confidence_proxy component differs by 3x and only
    # contributes config.confidence_weight (0.25) of the total -- ratio must
    # be far below 27x.
    assert score_low > 0
    ratio = score_high / score_low
    assert ratio < 5.0


def test_single_expected_score_returns_nonnegative():
    config = po.OptimizerConfig()
    candidates = _candidates(confluence_reversal_risk=[0.9] * 5, bear=[1] * 5)
    scores = po.single_expected_score(candidates, config)
    assert (scores >= 0).all()


# =============================================================================
# build_shrinkage_covariance
# =============================================================================

def test_build_shrinkage_covariance_returns_none_on_short_history():
    hist = _returns_history(["AAPL", "MSFT"], n_days=5)
    assert po.build_shrinkage_covariance(hist, min_lookback_days=20) is None


def test_build_shrinkage_covariance_returns_none_on_single_symbol():
    hist = _returns_history(["AAPL"], n_days=100)
    assert po.build_shrinkage_covariance(hist, min_lookback_days=20) is None


def test_build_shrinkage_covariance_shape_matches_symbols():
    symbols = ["AAPL", "MSFT", "JPM"]
    hist = _returns_history(symbols, n_days=100)
    result = po.build_shrinkage_covariance(hist, min_lookback_days=20)
    assert result is not None
    cov, order = result
    assert cov.shape == (3, 3)
    assert set(order) == set(symbols)


def test_build_shrinkage_covariance_is_positive_semidefinite():
    symbols = ["AAPL", "MSFT", "JPM", "XOM"]
    hist = _returns_history(symbols, n_days=150)
    cov, _ = po.build_shrinkage_covariance(hist, min_lookback_days=20)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert (eigenvalues >= -1e-10).all()


# =============================================================================
# fallback_inverse_vol_weights
# =============================================================================

def test_fallback_inverse_vol_weights_lower_vol_gets_higher_weight():
    config = po.OptimizerConfig()
    candidates = _candidates(symbols=["LOWVOL", "HIVOL"], volatility20=[0.01, 0.05])
    candidates["expected_score"] = [1.0, 1.0]
    result = po.fallback_inverse_vol_weights(candidates, config, max_investable_weight=0.5)
    low = result[result["symbol"] == "LOWVOL"]["target_weight"].iloc[0]
    hi = result[result["symbol"] == "HIVOL"]["target_weight"].iloc[0]
    assert low > hi


def test_fallback_inverse_vol_weights_respects_gross_cap():
    config = po.OptimizerConfig(max_position_weight=0.9)
    candidates = _candidates()
    candidates["expected_score"] = 1.0
    result = po.fallback_inverse_vol_weights(candidates, config, max_investable_weight=0.4)
    assert result["target_weight"].sum() <= 0.4 + 1e-6


def test_fallback_inverse_vol_weights_handles_all_nan_volatility():
    config = po.OptimizerConfig()
    candidates = _candidates(volatility20=[np.nan] * 5)
    candidates["expected_score"] = 1.0
    result = po.fallback_inverse_vol_weights(candidates, config, max_investable_weight=0.5)
    assert not result.empty
    assert np.isfinite(result["target_weight"]).all()


# =============================================================================
# solve_covariance_aware_weights: constraints
# =============================================================================

def test_solve_covariance_aware_weights_respects_exposure_group_cap():
    symbols = ["AAPL", "MSFT", "JPM"]
    uni.register_equity_sectors({"AAPL": "Information Technology", "MSFT": "Information Technology", "JPM": "Financials"})
    try:
        cov = np.eye(3) * 0.0004
        expected_scores = pd.Series([1.0, 1.0, 1.0], index=symbols)
        prev_weights = pd.Series(0.0, index=symbols)
        exposure_groups = {s: uni.classify_symbol(s).exposure_group for s in symbols}
        config = po.OptimizerConfig(max_position_weight=0.5)
        cov_config = po.CovarianceAwareConfig(exposure_group_cap=0.20, risk_aversion=0.1, turnover_penalty=0.0)

        weights = po.solve_covariance_aware_weights(
            expected_scores, cov, symbols, prev_weights, exposure_groups, set(), config, cov_config,
            max_investable_weight=0.9,
        )
        assert weights is not None
        it_weight = weights["AAPL"] + weights["MSFT"]
        assert it_weight <= 0.20 + 1e-4
    finally:
        uni.register_equity_sectors({})


def test_solve_covariance_aware_weights_respects_leveraged_cap():
    symbols = ["TQQQ", "SQQQ", "AAPL"]
    cov = np.eye(3) * 0.0004
    expected_scores = pd.Series([1.0, 1.0, 1.0], index=symbols)
    prev_weights = pd.Series(0.0, index=symbols)
    exposure_groups = {s: uni.classify_symbol(s).exposure_group for s in symbols}
    leveraged = {"TQQQ", "SQQQ"}
    config = po.OptimizerConfig(max_position_weight=0.5)
    cov_config = po.CovarianceAwareConfig(leveraged_inverse_cap=0.15, exposure_group_cap=0.9, risk_aversion=0.1, turnover_penalty=0.0)

    weights = po.solve_covariance_aware_weights(
        expected_scores, cov, symbols, prev_weights, exposure_groups, leveraged, config, cov_config,
        max_investable_weight=0.9,
    )
    assert weights is not None
    assert weights["TQQQ"] + weights["SQQQ"] <= 0.15 + 1e-4


def test_solve_covariance_aware_weights_respects_gross_exposure():
    symbols = ["AAPL", "MSFT", "JPM", "XOM"]
    cov = np.eye(4) * 0.0004
    expected_scores = pd.Series([1.0] * 4, index=symbols)
    prev_weights = pd.Series(0.0, index=symbols)
    exposure_groups = {s: s for s in symbols}
    config = po.OptimizerConfig(max_position_weight=0.9)
    cov_config = po.CovarianceAwareConfig(exposure_group_cap=0.9, risk_aversion=0.1, turnover_penalty=0.0)

    weights = po.solve_covariance_aware_weights(
        expected_scores, cov, symbols, prev_weights, exposure_groups, set(), config, cov_config,
        max_investable_weight=0.5,
    )
    assert weights is not None
    assert weights.sum() <= 0.5 + 1e-4


def test_solve_covariance_aware_weights_penalizes_turnover():
    """Starting already-invested in AAPL should (all else equal) favor
    staying in AAPL over an equally-attractive alternative, due to the
    turnover penalty."""
    symbols = ["AAPL", "MSFT"]
    cov = np.eye(2) * 0.0004
    expected_scores = pd.Series([1.0, 1.0], index=symbols)
    exposure_groups = {s: s for s in symbols}
    config = po.OptimizerConfig(max_position_weight=0.5)

    prev_weights_none = pd.Series([0.0, 0.0], index=symbols)
    prev_weights_aapl = pd.Series([0.3, 0.0], index=symbols)

    cov_config = po.CovarianceAwareConfig(exposure_group_cap=0.9, risk_aversion=0.1, turnover_penalty=5.0)

    w_none = po.solve_covariance_aware_weights(expected_scores, cov, symbols, prev_weights_none, exposure_groups, set(), config, cov_config, 0.6)
    w_prev = po.solve_covariance_aware_weights(expected_scores, cov, symbols, prev_weights_aapl, exposure_groups, set(), config, cov_config, 0.6)

    assert w_prev["AAPL"] > w_none["AAPL"]


# =============================================================================
# optimize_weights_covariance_aware: end-to-end
# =============================================================================

def test_optimize_weights_covariance_aware_uses_fallback_when_no_history():
    candidates = _candidates()
    result = po.optimize_weights_covariance_aware(candidates, returns_history=None)
    assert not result.empty
    assert (result["sizing_method"] == "FALLBACK_INVERSE_VOL").all()


def test_optimize_weights_covariance_aware_uses_shrinkage_path_with_history():
    symbols = ["AAPL", "MSFT", "JPM", "XOM", "QQQ"]
    candidates = _candidates(symbols=symbols)
    hist = _returns_history(symbols, n_days=150)
    result = po.optimize_weights_covariance_aware(candidates, returns_history=hist)
    assert not result.empty
    assert (result["sizing_method"] == "COVARIANCE_AWARE_SHRINKAGE").all()


def test_optimize_weights_covariance_aware_empty_candidates_returns_empty():
    empty = _candidates()[0:0]
    assert po.optimize_weights_covariance_aware(empty, returns_history=None).empty


def test_optimize_weights_covariance_aware_target_dollars_consistent_with_price():
    candidates = _candidates()
    result = po.optimize_weights_covariance_aware(candidates, returns_history=None)
    for _, row in result.iterrows():
        assert row["target_dollars"] == pytest.approx(row["target_shares"] * row["price"])


# =============================================================================
# Regression: existing default path is unchanged
# =============================================================================

def test_existing_optimize_weights_unaffected_by_new_additions():
    candidates = _candidates()
    result = po.optimize_weights(candidates)
    assert "sizing_method" not in result.columns  # new field never appears in the old path
    assert not result.empty


def test_existing_optimize_portfolio_unaffected_by_new_additions():
    candidates = _candidates()
    targets, rebalance, exits, summary = po.optimize_portfolio(candidates, equity=100_000.0)
    assert "sizing_method" not in targets.columns
    assert summary["position_count"] >= 0
