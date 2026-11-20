"""Tests for metrics.py, using synthetic equity curves with known expected
values. Directly targets the Objective 1 bug: Sharpe must come from daily
excess returns (mean/std * sqrt(252)), never from CAGR / volatility or
CAGR / drawdown.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

import metrics
from metrics import (
    apply_turnover_cost,
    bootstrap_sharpe_ci,
    calmar_ratio,
    cagr,
    deflated_sharpe_ratio,
    effective_sample_size_ar1,
    equity_curve_from_daily_returns,
    max_drawdown,
    sharpe_ratio,
    sharpe_standard_error,
    sortino_ratio,
    summarize_performance,
    turnover_from_weights,
    win_rate,
)


def test_sharpe_matches_independent_reference_formula():
    returns = [0.02, 0.01, -0.01, 0.03, -0.02]
    arr = np.asarray(returns)
    expected = arr.mean() / arr.std(ddof=1) * math.sqrt(252)
    assert sharpe_ratio(returns) == pytest.approx(expected, rel=1e-9)


def test_sharpe_zero_volatility_returns_zero_not_inf_or_nan():
    # Constant daily return -> std == 0. The correct behavior is an explicit
    # 0.0, not inf/NaN silently propagating into a report.
    assert sharpe_ratio([0.001] * 30) == 0.0


def test_sharpe_single_observation_returns_zero():
    assert sharpe_ratio([0.01]) == 0.0


def test_sharpe_subtracts_risk_free_rate():
    returns = [0.001] * 60  # flat 0.1%/day -> zero vol either way, so use varying returns
    returns = [0.002, 0.0, 0.001, -0.001, 0.0015, -0.0005] * 10
    arr = np.asarray(returns)
    rf_annual = 0.05
    daily_rf = (1.0 + rf_annual) ** (1.0 / 252) - 1.0
    excess = arr - daily_rf
    expected = excess.mean() / excess.std(ddof=1) * math.sqrt(252)
    got = sharpe_ratio(returns, risk_free_rate_annual=rf_annual)
    assert got == pytest.approx(expected, rel=1e-9)
    # A positive risk-free rate must lower the Sharpe versus ignoring it.
    assert got < sharpe_ratio(returns, risk_free_rate_annual=0.0)


def test_sharpe_is_not_cagr_over_vol():
    """Regression test for the exact bug being fixed: on a return series with
    a geometric/arithmetic mean gap, CAGR/annualized-vol and the correct
    mean-excess/std formula must disagree -- proving the implementation is
    not secretly doing the wrong thing."""
    returns = [0.05, -0.04, 0.06, -0.05, 0.08, -0.07] * 20
    arr = np.asarray(returns)

    total_return = float(np.prod(1 + arr) - 1)
    n = len(arr)
    wrong_annual_return = (1 + total_return) ** (252 / n) - 1
    wrong_annual_vol = arr.std(ddof=1) * math.sqrt(252)
    wrong_sharpe = wrong_annual_return / wrong_annual_vol

    correct_sharpe = sharpe_ratio(returns)

    assert correct_sharpe != pytest.approx(wrong_sharpe, rel=1e-6)
    expected_correct = arr.mean() / arr.std(ddof=1) * math.sqrt(252)
    assert correct_sharpe == pytest.approx(expected_correct, rel=1e-9)


def test_sortino_only_penalizes_downside():
    # All-positive returns -> no downside deviation -> defined as 0.0, not inf.
    assert sortino_ratio([0.01, 0.02, 0.015, 0.03]) == 0.0

    returns = [0.02, -0.01, 0.03, -0.02, 0.01, -0.015]
    arr = np.asarray(returns)
    downside = arr[arr < 0]
    expected = arr.mean() / downside.std(ddof=1) * math.sqrt(252)
    assert sortino_ratio(returns) == pytest.approx(expected, rel=1e-9)


def test_cagr_one_year_equals_total_return():
    assert cagr(0.21, 252) == pytest.approx(0.21, rel=1e-9)


def test_cagr_two_years_annualizes_down():
    # 44% over 2 years (504 trading days) -> ~20% annualized ((1.44)^0.5 - 1)
    assert cagr(0.44, 504) == pytest.approx(1.44 ** 0.5 - 1, rel=1e-9)


def test_max_drawdown_known_curve():
    # 100 -> 110 (peak) -> 90 -> 120: trough 90 vs peak 110 => -18.1818...%
    dd = max_drawdown([100, 110, 90, 120])
    assert dd == pytest.approx(90 / 110 - 1, rel=1e-9)


def test_max_drawdown_monotonic_up_is_zero():
    assert max_drawdown([100, 101, 105, 110]) == 0.0


def test_calmar_known_values():
    assert calmar_ratio(0.20, -0.10) == pytest.approx(2.0, rel=1e-9)
    assert calmar_ratio(0.20, 0.0) == 0.0  # undefined -> explicit 0.0, not inf


def test_equity_curve_from_daily_returns():
    curve = equity_curve_from_daily_returns([0.1, -0.1, 0.1], starting_equity=100.0)
    assert curve[0] == pytest.approx(110.0)
    assert curve[1] == pytest.approx(99.0)
    assert curve[2] == pytest.approx(108.9, rel=1e-6)


def test_win_rate():
    assert win_rate([10, -5, 3, -1, 0]) == pytest.approx(0.4)  # 2 of 5 strictly positive
    assert win_rate([]) == 0.0


def test_turnover_from_weights_known_matrix():
    weights = pd.DataFrame({"A": [0.5, 0.5, 0.2], "B": [0.0, 0.3, 0.3]})
    turnover = turnover_from_weights(weights)
    # period 0: entering from flat -> |0.5|+|0.0| = 0.5
    assert turnover.iloc[0] == pytest.approx(0.5)
    # period 1: |0.5-0.5| + |0.3-0.0| = 0.3
    assert turnover.iloc[1] == pytest.approx(0.3)
    # period 2: |0.2-0.5| + |0.3-0.3| = 0.3
    assert turnover.iloc[2] == pytest.approx(0.3)


def test_apply_turnover_cost_known_values():
    gross = [0.01, 0.02, -0.01]
    turnover = [0.5, 1.0, 0.3]
    net = apply_turnover_cost(gross, turnover, cost_bps=10.0)
    expected = [0.01 - 0.5 * 10 / 10_000, 0.02 - 1.0 * 10 / 10_000, -0.01 - 0.3 * 10 / 10_000]
    np.testing.assert_allclose(net, expected)


def test_apply_turnover_cost_length_mismatch_raises():
    with pytest.raises(ValueError):
        apply_turnover_cost([0.01, 0.02], [0.5], cost_bps=5.0)


def test_sharpe_standard_error_increases_as_n_decreases():
    se_small = sharpe_standard_error(sharpe=1.0, n_obs=30)
    se_large = sharpe_standard_error(sharpe=1.0, n_obs=1000)
    assert se_small > se_large > 0


def test_sharpe_standard_error_nan_for_tiny_sample():
    assert math.isnan(sharpe_standard_error(sharpe=1.0, n_obs=1))


def test_bootstrap_ci_nan_for_short_sample():
    lo, hi = bootstrap_sharpe_ci([0.01] * 5, random_state=0)
    assert math.isnan(lo) and math.isnan(hi)


def test_bootstrap_ci_brackets_point_estimate_on_longer_sample():
    rng = np.random.default_rng(42)
    returns = rng.normal(loc=0.001, scale=0.01, size=500)
    point = sharpe_ratio(returns)
    lo, hi = bootstrap_sharpe_ci(returns, random_state=42)
    assert lo <= point <= hi


def test_deflated_sharpe_decreases_with_more_trials():
    dsr_1_trial = deflated_sharpe_ratio(observed_sharpe=1.5, n_obs=252, n_trials=1)
    dsr_100_trials = deflated_sharpe_ratio(observed_sharpe=1.5, n_obs=252, n_trials=100)
    assert 0.0 <= dsr_100_trials <= dsr_1_trial <= 1.0


def test_deflated_sharpe_single_trial_matches_plain_significance():
    from scipy.stats import norm
    sharpe = 1.2
    n_obs = 252
    dsr = deflated_sharpe_ratio(observed_sharpe=sharpe, n_obs=n_obs, n_trials=1)
    per_period_sr = sharpe / math.sqrt(252)
    expected = norm.cdf(per_period_sr * math.sqrt(n_obs - 1))
    assert dsr == pytest.approx(expected, rel=1e-6)


def test_summarize_performance_smoke_and_internal_consistency():
    rng = np.random.default_rng(7)
    returns = rng.normal(loc=0.0008, scale=0.012, size=300)
    trades = rng.normal(loc=50, scale=200, size=40)
    exposure = np.clip(rng.normal(loc=0.6, scale=0.1, size=300), 0, 1)
    weights = pd.DataFrame({"A": np.clip(rng.normal(0.3, 0.05, 300), 0, 1)})
    turnover = turnover_from_weights(weights)

    summary = summarize_performance(
        returns, trade_pnls=trades, exposure=exposure, per_period_turnover=turnover, random_state=7
    )

    assert summary.n_obs == 300
    assert summary.sharpe == pytest.approx(sharpe_ratio(returns), rel=1e-9)
    assert summary.sortino == pytest.approx(sortino_ratio(returns), rel=1e-9)
    assert summary.trade_count == 40
    assert 0.0 <= summary.win_rate <= 1.0
    assert not math.isnan(summary.sharpe_se)
    assert summary.sharpe_ci_low <= summary.sharpe <= summary.sharpe_ci_high


def test_effective_sample_size_ar1_no_autocorrelation_equals_n():
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.01, 500)  # iid, rho ~ 0
    n_eff = effective_sample_size_ar1(returns)
    assert n_eff == pytest.approx(500, rel=0.15)  # sampling noise in rho estimate


def test_effective_sample_size_ar1_positive_autocorrelation_shrinks_n():
    rng = np.random.default_rng(2)
    innovations = rng.normal(0, 0.01, 500)
    ar_returns = np.zeros(500)
    ar_returns[0] = innovations[0]
    for i in range(1, 500):
        ar_returns[i] = 0.8 * ar_returns[i - 1] + innovations[i]  # strong positive AR(1)

    n_eff = effective_sample_size_ar1(ar_returns)
    assert n_eff < 500 * 0.3  # rho=0.8 -> n_eff = n*0.2/1.8 ~ 0.111n


def test_effective_sample_size_ar1_never_exceeds_n():
    returns = [0.01, -0.01, 0.02, -0.02, 0.015, -0.015, 0.01, -0.01]
    assert effective_sample_size_ar1(returns) <= len(returns)


def test_effective_sample_size_ar1_tiny_sample_returns_n():
    assert effective_sample_size_ar1([0.01, 0.02]) == 2.0
