"""Regression tests for Objective 1's wiring of costs.py/metrics.py into
walkforward.py: entry/exit cost helpers, and calculate_window_metrics using
the correct (non-CAGR-based) Sharpe formula plus Sortino/Calmar/uncertainty.

Importing walkforward.py pulls in torch; this module is intentionally kept
separate from the lightweight universe/metrics/costs test files so a slow or
missing torch install doesn't block those from running.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

import costs
import metrics
import walkforward as wf


def test_entry_exit_slippage_sourced_from_shared_costs_module():
    assert wf.ENTRY_SLIPPAGE_BPS == costs.BASELINE_COSTS.slippage_bps
    assert wf.EXIT_SLIPPAGE_BPS == costs.BASELINE_COSTS.slippage_bps
    assert wf.ENTRY_COMMISSION_BPS == pytest.approx(costs.BASELINE_COSTS.commission_bps / 2.0)


def test_entry_cost_includes_slippage_commission_and_impact():
    notional, adv = 50_000, 1_000_000  # 5% participation
    got = wf._entry_cost_bps(notional, adv)
    expected = (
        costs.BASELINE_COSTS.slippage_bps
        + costs.BASELINE_COSTS.commission_bps / 2.0
        + costs.market_impact_bps(notional, adv, costs.BASELINE_COSTS)
    )
    assert got == pytest.approx(expected)


def test_entry_and_exit_cost_combine_to_full_round_trip_commission():
    # Each side carries half the round-trip commission; full round trip
    # (entry notional == exit notional, same ADV) should equal the shared
    # config's full commission_bps plus 2x slippage plus 2x impact.
    notional, adv = 10_000, 2_000_000
    entry = wf._entry_cost_bps(notional, adv)
    exit_ = wf._exit_cost_bps(notional, adv)
    impact = costs.market_impact_bps(notional, adv, costs.BASELINE_COSTS)
    expected_total = 2 * costs.BASELINE_COSTS.slippage_bps + costs.BASELINE_COSTS.commission_bps + 2 * impact
    assert entry + exit_ == pytest.approx(expected_total)


def _synthetic_daily(n=40, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    daily_return = rng.normal(0.001, 0.01, n)
    equity = wf.STARTING_EQUITY * np.cumprod(1 + daily_return)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "equity": equity,
            "open_positions": rng.integers(0, 4, n),
            "gross_exposure": np.clip(rng.normal(0.5, 0.1, n), 0, 1),
        }
    )


def test_calculate_window_metrics_sharpe_matches_metrics_module():
    daily = _synthetic_daily()
    events = pd.DataFrame(columns=["event", "pnl", "return", "exit_reason"])

    result = wf.calculate_window_metrics(daily, events)

    d = daily.sort_values("timestamp").copy()
    d["daily_return"] = d["equity"].pct_change().fillna(0.0)
    expected_sharpe = metrics.sharpe_ratio(d["daily_return"], wf.RISK_FREE_RATE_ANNUAL)

    assert result["sharpe"] == pytest.approx(expected_sharpe, rel=1e-9)
    assert "sortino" in result and "calmar" in result and "sharpe_se" in result
    assert not math.isnan(result["sharpe_se"])


def test_calculate_window_metrics_sharpe_is_not_old_cagr_over_vol_bug():
    daily = _synthetic_daily(n=60, seed=11)
    events = pd.DataFrame(columns=["event", "pnl", "return", "exit_reason"])
    result = wf.calculate_window_metrics(daily, events)

    d = daily.sort_values("timestamp").copy()
    d["daily_return"] = d["equity"].pct_change().fillna(0.0)
    total_return = d["equity"].iloc[-1] / d["equity"].iloc[0] - 1
    wrong_ann_ret = (1 + total_return) ** (252 / len(d)) - 1
    wrong_ann_vol = d["daily_return"].std() * np.sqrt(252)
    wrong_sharpe = wrong_ann_ret / wrong_ann_vol if wrong_ann_vol > 0 else 0.0

    assert result["sharpe"] != pytest.approx(wrong_sharpe, rel=1e-6)


def test_calculate_window_metrics_empty_daily_has_full_schema():
    result = wf.calculate_window_metrics(pd.DataFrame(), pd.DataFrame())
    for key in ("sharpe", "sharpe_se", "sortino", "calmar", "max_drawdown"):
        assert key in result
    assert math.isnan(result["sharpe_se"])
    assert result["sortino"] == 0.0 and result["calmar"] == 0.0


# =============================================================================
# Purged / embargoed walk-forward windowing
# =============================================================================

def _bdate_list(n: int, start: str = "2020-01-02") -> list[pd.Timestamp]:
    return list(pd.bdate_range(start, periods=n))


def test_default_embargo_meets_minimum_label_horizon():
    # FUTURE_RET_5D is a 5-trading-day-forward label in both model.py and
    # walkforward.py; the embargo must be at least that long.
    assert wf.EMBARGO_DAYS >= 5


def test_make_windows_leaves_exact_embargo_gap_between_train_and_test():
    dates = _bdate_list(wf.TRAIN_DAYS + 5 + wf.TEST_DAYS + 10)
    windows = wf.make_windows(dates, embargo_days=5, holdout_days=0)
    assert len(windows) >= 1

    _, train_dates, test_dates = windows[0]
    train_end_idx = dates.index(pd.Timestamp(train_dates[-1]))
    test_start_idx = dates.index(pd.Timestamp(test_dates[0]))
    # Exactly 5 dates purged between train end and test start (neither side
    # includes them) -- this is the fix for the original zero-gap bug.
    assert test_start_idx - train_end_idx - 1 == 5


def test_make_windows_zero_embargo_reproduces_old_zero_gap_behavior():
    """Confirms embargo_days is load-bearing: with embargo_days=0 the old
    (leaky) adjacency is reproduced, proving the default's gap above is the
    embargo parameter doing real work, not an artifact of window arithmetic."""
    dates = _bdate_list(wf.TRAIN_DAYS + wf.TEST_DAYS + 10)
    windows = wf.make_windows(dates, embargo_days=0, holdout_days=0)
    _, train_dates, test_dates = windows[0]
    train_end_idx = dates.index(pd.Timestamp(train_dates[-1]))
    test_start_idx = dates.index(pd.Timestamp(test_dates[0]))
    assert test_start_idx - train_end_idx - 1 == 0


def test_make_windows_never_touches_holdout_dates():
    holdout_days = 60
    dates = _bdate_list(wf.TRAIN_DAYS + 5 + wf.TEST_DAYS * 3 + holdout_days + 20)
    windows = wf.make_windows(dates, embargo_days=5, holdout_days=holdout_days)
    holdout = set(wf.get_final_holdout_dates(dates, holdout_days).tolist())

    assert len(holdout) == holdout_days
    assert len(windows) >= 1
    for _, train_dates, test_dates in windows:
        assert not (set(train_dates) & holdout)
        assert not (set(test_dates) & holdout)


def test_get_final_holdout_dates_is_the_most_recent_slice():
    dates = _bdate_list(200)
    holdout = wf.get_final_holdout_dates(dates, holdout_days=60)
    assert list(holdout) == dates[-60:]


def test_get_final_holdout_dates_empty_when_insufficient_history():
    dates = _bdate_list(50)
    holdout = wf.get_final_holdout_dates(dates, holdout_days=60)
    assert len(holdout) == 0


def test_usable_dates_excludes_holdout_tail():
    dates = _bdate_list(200)
    usable = wf.usable_dates_excluding_holdout(dates, holdout_days=60)
    assert usable == dates[:-60]
    assert set(usable).isdisjoint(set(dates[-60:]))


# =============================================================================
# Objective 2: risk-adjusted ranking/sizing (flagged, off by default)
# =============================================================================

def _eligible_day(n=4, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "symbol": [f"SYM{i}" for i in range(n)],
        "tradable_symbol": 1,
        "close": rng.uniform(20, 100, n),
        "avg_dollar_volume20": rng.uniform(5_000_000, 50_000_000, n),
        "relative_volume_20": rng.uniform(1.0, 2.0, n),
        "range_expansion": rng.uniform(0.01, 0.03, n),
        "explosive_alpha_probability": np.linspace(0.6, 0.9, n),
        "market_regime": "TREND_RISK_ON",
        "momentum_negative": 0,
        "late_reversal_risk": 0,
        "trend_alignment": 1,
        "breakout_above_prev_high": 1,
        "explosion_setup_score": 0.8,
        "momentum_score": 0.6,
        "price_indicator_alignment_score": 0.5,
        "indicator_transition_score": 0.5,
        "context_interaction_score": 0.5,
        "atr_pct": rng.uniform(0.01, 0.04, n),
        "volatility_20d": rng.uniform(0.01, 0.04, n),
        "daily_range_pct": rng.uniform(0.01, 0.04, n),
    })


def test_risk_adjusted_ranking_disabled_by_default():
    assert wf.RISK_ADJUSTED_RANKING_ENABLED is False


def test_score_entry_candidates_risk_adjusted_uses_prob_exactly_once_uncalibrated():
    day = _eligible_day()
    scored = wf.score_entry_candidates_risk_adjusted(day, equity=100_000, high_water_mark=100_000, calibration=None)
    # Uncalibrated fallback: risk_adjusted_score should equal prob exactly (one channel).
    assert np.allclose(scored["risk_adjusted_score"].values, scored["explosive_alpha_probability"].values)


def test_score_entry_candidates_risk_adjusted_applies_calibration():
    day = _eligible_day()

    class _FakeCalibration:
        def predict(self, x):
            return x * 2.0  # arbitrary monotonic transform to prove it's applied

    scored = wf.score_entry_candidates_risk_adjusted(day, equity=100_000, high_water_mark=100_000, calibration=_FakeCalibration())
    assert np.allclose(scored["risk_adjusted_score"].values, scored["explosive_alpha_probability"].values * 2.0)


def test_score_entry_candidates_risk_adjusted_respects_eligibility_gates():
    day = _eligible_day()
    day.loc[0, "avg_dollar_volume20"] = 100.0  # fails liquidity gate
    scored = wf.score_entry_candidates_risk_adjusted(day, equity=100_000, high_water_mark=100_000, calibration=None)
    assert "SYM0" not in scored["symbol"].values


def test_build_weight_plan_risk_adjusted_ranks_by_single_score_not_volume():
    day = _eligible_day(n=3)
    # Make the LOWEST-prob candidate have the HIGHEST volume, to prove
    # volume alone doesn't drive selection/weight under the new path.
    day["avg_dollar_volume20"] = [100_000_000, 10_000_000, 5_000_000]  # SYM0 highest volume, lowest prob
    scored = wf.score_entry_candidates_risk_adjusted(day, equity=100_000, high_water_mark=100_000, calibration=None)
    plan = wf.build_weight_plan_risk_adjusted(scored, equity=100_000, high_water_mark=100_000)
    assert not plan.empty
    # Highest prob (SYM2) should get the largest weight, not highest-volume SYM0.
    top = plan.sort_values("target_weight", ascending=False).iloc[0]
    assert top["symbol"] == "SYM2"


def test_build_weight_plan_risk_adjusted_weights_sum_within_gross_cap():
    day = _eligible_day(n=4)
    scored = wf.score_entry_candidates_risk_adjusted(day, equity=100_000, high_water_mark=100_000, calibration=None)
    plan = wf.build_weight_plan_risk_adjusted(scored, equity=100_000, high_water_mark=100_000)
    if not plan.empty:
        assert plan["target_weight"].sum() <= wf.BASE_MAX_GROSS_EXPOSURE + 1e-6


def test_build_weight_plan_risk_adjusted_empty_input_returns_empty():
    assert wf.build_weight_plan_risk_adjusted(pd.DataFrame(), equity=100_000, high_water_mark=100_000).empty


def _position(entry_price=100.0, bars_held=0, entry_probability=0.6, peak_probability=0.6):
    return wf.Position(
        symbol="AAPL",
        entry_timestamp=pd.Timestamp("2024-01-01"),
        entry_price=entry_price,
        shares=10,
        weight_at_entry=0.1,
        target_weight=0.1,
        entry_probability=entry_probability,
        peak_probability=peak_probability,
        bars_held=bars_held,
    )


def test_turnover_controls_disabled_by_default():
    assert wf.TURNOVER_CONTROLS_ENABLED is False


def test_turnover_aware_suppresses_soft_exit_before_min_hold():
    pos = _position(bars_held=1)  # below MIN_HOLD_BARS_SOFT_EXIT=3
    row = pd.Series({"explosive_alpha_probability": 0.1, "momentum_negative": 0, "momentum_favorable": 1,
                      "breakout_above_prev_high": 0, "explosion_setup_score": 0.0})
    decision = wf.meta_controller_decision_turnover_aware(pos, row, close_price=100.0)
    assert decision == "HOLD"
    # Confirm the underlying (ungated) decision really was a soft exit.
    assert wf.meta_controller_decision(pos, row, close_price=100.0) == "EXIT_MODEL_FAILURE"


def test_turnover_aware_allows_soft_exit_after_min_hold():
    pos = _position(bars_held=wf.MIN_HOLD_BARS_SOFT_EXIT)
    row = pd.Series({"explosive_alpha_probability": 0.1, "momentum_negative": 0, "momentum_favorable": 1,
                      "breakout_above_prev_high": 0, "explosion_setup_score": 0.0})
    assert wf.meta_controller_decision_turnover_aware(pos, row, close_price=100.0) == "EXIT_MODEL_FAILURE"


def test_turnover_aware_never_gates_hard_exits():
    """TARGET_100 is a hard, structural exit -- must fire even at bars_held=0."""
    pos = _position(entry_price=100.0, bars_held=0)
    row = pd.Series({"explosive_alpha_probability": 0.6, "momentum_negative": 0, "momentum_favorable": 1,
                      "breakout_above_prev_high": 0, "explosion_setup_score": 0.0})
    close_price = 100.0 * (1 + wf.TARGET_TRADE_RETURN + 0.01)  # comfortably clears the 20% target
    assert wf.meta_controller_decision_turnover_aware(pos, row, close_price) == "EXIT_TARGET_100"


def test_model_exit_decision_uses_turnover_aware_wrapper_when_enabled(monkeypatch):
    monkeypatch.setattr(wf, "TURNOVER_CONTROLS_ENABLED", True)
    pos = _position(bars_held=1)
    row = pd.Series({"explosive_alpha_probability": 0.1, "momentum_negative": 0, "momentum_favorable": 1,
                      "breakout_above_prev_high": 0, "explosion_setup_score": 0.0, "avg_dollar_volume20": 1e7})
    reason, price, action = wf.model_exit_decision(pos, row, close_price=100.0)
    assert reason is None and action == "HOLD"


def test_risk_adjusted_entry_gate_tightens_under_turnover_controls(monkeypatch):
    day = _eligible_day(n=1)
    day["explosive_alpha_probability"] = [0.6]
    # No calibration -> expected_edge == prob == 0.6, which clears the
    # default (0.0) gate but not a raised MIN_EDGE_OVER_COST_ATR bar.
    monkeypatch.setattr(wf, "TURNOVER_CONTROLS_ENABLED", True)
    monkeypatch.setattr(wf, "MIN_EDGE_OVER_COST_ATR", 0.8)
    scored = wf.score_entry_candidates_risk_adjusted(day, equity=100_000, high_water_mark=100_000, calibration=None)
    assert scored.empty


def test_gradual_vol_dd_scaling_disabled_by_default():
    assert wf.GRADUAL_VOL_DD_SCALING_ENABLED is False


def test_capital_preservation_multiplier_gradual_matches_discrete_at_anchors():
    """Must agree exactly with the discrete step function at each of its
    own threshold anchors -- the gradual version is an interpolation of
    the SAME anchors, not a different policy."""
    hwm = 100_000.0
    for state, ratio in [("KILL", wf.EQUITY_KILL), ("DEFENSIVE", wf.EQUITY_DEFENSIVE),
                          ("RECOVERY", wf.EQUITY_RECOVERY), ("NORMAL", 1.0)]:
        equity = hwm * ratio
        gradual = wf.capital_preservation_multiplier_gradual(equity, hwm)
        discrete = wf.CAPITAL_PRESERVATION_GROSS_MULT[state]
        assert gradual == pytest.approx(discrete, abs=1e-6)


def test_capital_preservation_multiplier_gradual_is_smooth_between_anchors():
    """No abrupt jump: multiplier must change continuously as equity ratio
    crosses a threshold, unlike the discrete version's step."""
    hwm = 100_000.0
    just_above = wf.capital_preservation_multiplier_gradual(hwm * (wf.EQUITY_RECOVERY + 0.0001), hwm)
    just_below = wf.capital_preservation_multiplier_gradual(hwm * (wf.EQUITY_RECOVERY - 0.0001), hwm)
    assert abs(just_above - just_below) < 0.01  # small input change -> small output change

    # Confirm the discrete version, by contrast, DOES jump at this boundary.
    discrete_above = wf.CAPITAL_PRESERVATION_GROSS_MULT[wf.capital_state(hwm * (wf.EQUITY_RECOVERY + 0.0001), hwm)]
    discrete_below = wf.CAPITAL_PRESERVATION_GROSS_MULT[wf.capital_state(hwm * (wf.EQUITY_RECOVERY - 0.0001), hwm)]
    assert abs(discrete_above - discrete_below) > 0.1


def test_capital_preservation_multiplier_gradual_monotonic_in_drawdown():
    hwm = 100_000.0
    ratios = np.linspace(0.85, 1.05, 20)
    mults = [wf.capital_preservation_multiplier_gradual(hwm * r, hwm) for r in ratios]
    assert all(m2 >= m1 - 1e-9 for m1, m2 in zip(mults, mults[1:]))


def test_capital_preservation_multiplier_gradual_zero_hwm_returns_full():
    assert wf.capital_preservation_multiplier_gradual(100_000.0, 0.0) == 1.0


def test_realized_vol_scaler_neutral_with_insufficient_history():
    assert wf.realized_vol_exposure_scaler(None) == 1.0
    assert wf.realized_vol_exposure_scaler(pd.Series([0.01, 0.02])) == 1.0


def test_realized_vol_scaler_never_exceeds_one():
    """Must never scale UP exposure, even when realized vol is very low --
    per Objective 2 priority 7, no leveraging up to chase a vol target."""
    quiet_returns = pd.Series(np.full(30, 0.0001))
    assert wf.realized_vol_exposure_scaler(quiet_returns, target_vol=0.20) <= 1.0


def test_realized_vol_scaler_scales_down_when_vol_elevated():
    rng = np.random.default_rng(3)
    hot_returns = pd.Series(rng.normal(0, 0.04, 30))  # ~63% annualized vol
    scaler = wf.realized_vol_exposure_scaler(hot_returns, target_vol=0.15)
    assert scaler < 1.0


def test_regime_gross_exposure_gradual_vs_discrete_toggle(monkeypatch):
    hwm = 100_000.0
    equity = hwm * (wf.EQUITY_RECOVERY - 0.001)  # just inside DEFENSIVE
    monkeypatch.setattr(wf, "GRADUAL_VOL_DD_SCALING_ENABLED", False)
    discrete_result = wf.regime_gross_exposure("TREND_RISK_ON", "DEFENSIVE", equity=equity, high_water_mark=hwm)

    monkeypatch.setattr(wf, "GRADUAL_VOL_DD_SCALING_ENABLED", True)
    gradual_result = wf.regime_gross_exposure("TREND_RISK_ON", "DEFENSIVE", equity=equity, high_water_mark=hwm)

    assert discrete_result == pytest.approx(wf.REGIME_GROSS_EXPOSURE["TREND_RISK_ON"] * wf.CAPITAL_PRESERVATION_GROSS_MULT["DEFENSIVE"])
    assert gradual_result != discrete_result  # gradual interpolates, not a step


def test_get_entry_candidates_uses_default_path_when_flag_off(monkeypatch):
    monkeypatch.setattr(wf, "RISK_ADJUSTED_RANKING_ENABLED", False)
    day = _eligible_day(n=3)
    day["risk_on_score"] = 0.6
    result = wf.get_entry_candidates(day, available_slots=2, equity=100_000, high_water_mark=100_000)
    if not result.empty:
        assert result.iloc[0]["sizing_method"] == "PROB_EV_KELLY_VOL_DRAWDOWN_CORR"


def test_get_entry_candidates_uses_risk_adjusted_path_when_flag_on(monkeypatch):
    monkeypatch.setattr(wf, "RISK_ADJUSTED_RANKING_ENABLED", True)
    day = _eligible_day(n=3)
    day["risk_on_score"] = 0.6
    result = wf.get_entry_candidates(day, available_slots=2, equity=100_000, high_water_mark=100_000)
    if not result.empty:
        assert result.iloc[0]["sizing_method"] == "RISK_ADJUSTED_SINGLE_SCORE"


def test_fit_edge_calibration_returns_none_on_insufficient_history():
    tiny_df = pd.DataFrame({
        "timestamp": pd.bdate_range("2024-01-02", periods=5),
        "symbol": ["AAPL"] * 5,
        wf.TARGET_LABEL: [0, 1, 0, 1, 0],
    })
    assert wf.fit_edge_calibration(tiny_df, feature_cols=[]) is None


# =============================================================================
# Objective 2: cost-aware, risk-normalized target (flagged, off by default)
# =============================================================================

def _target_frame():
    return pd.DataFrame({
        "future_return_1d": [0.01, -0.03, 0.005, 0.01],
        "future_return_5d": [0.05, 0.02, -0.01, 0.003],
        "atr_pct": [0.02, 0.02, 0.02, 0.001],
        "avg_dollar_volume20": [5_000_000, 5_000_000, 5_000_000, 5_000_000],
        "tradable_symbol": [1, 1, 1, 1],
    })


def test_risk_aware_target_disabled_by_default():
    assert wf.RISK_AWARE_TARGET_ENABLED is False


def test_risk_aware_target_positive_when_edge_clears_atr_multiple():
    out = wf.make_target_risk_aware(_target_frame())
    assert out[wf.TARGET_LABEL].iloc[0] == 1
    assert out["risk_adjusted_edge_5d"].iloc[0] > wf.RISK_AWARE_MIN_EDGE_ATR


def test_risk_aware_target_negative_on_immediate_failure_despite_good_edge():
    out = wf.make_target_risk_aware(_target_frame())
    # row 1 has a decent 5d edge but future_return_1d=-3% fails no_immediate_failure
    assert out["risk_adjusted_edge_5d"].iloc[1] > 0.5
    assert out[wf.TARGET_LABEL].iloc[1] == 0


def test_risk_aware_target_negative_on_negative_forward_return():
    out = wf.make_target_risk_aware(_target_frame())
    assert out[wf.TARGET_LABEL].iloc[2] == 0


def test_risk_aware_target_same_raw_return_different_label_by_volatility():
    """The entire point of risk normalization: an IDENTICAL small raw
    forward return should be judged differently depending on the symbol's
    own volatility -- a 0.3% move clears a 0.1%-ATR name's own bar but is
    noise for a 2%-ATR name. The fixed-percentage make_target() cannot make
    this distinction; make_target_risk_aware() must."""
    same_return_different_atr = pd.DataFrame({
        "future_return_1d": [0.001, 0.001],
        "future_return_5d": [0.003, 0.003],
        "atr_pct": [0.001, 0.02],
        "avg_dollar_volume20": [5_000_000, 5_000_000],
        "tradable_symbol": [1, 1],
    })
    out = wf.make_target_risk_aware(same_return_different_atr)
    assert out[wf.TARGET_LABEL].iloc[0] == 1   # low-ATR name: 0.3% clears its own bar
    assert out[wf.TARGET_LABEL].iloc[1] == 0   # high-ATR name: same 0.3% is noise
    assert out["risk_adjusted_edge_5d"].iloc[0] > out["risk_adjusted_edge_5d"].iloc[1]


def test_risk_aware_target_label_column_is_blocked_from_features():
    assert "risk_adjusted_edge_5d" in wf.LEAKAGE_COLUMNS
    assert wf.TARGET_LABEL in wf.LEAKAGE_COLUMNS


def test_prepare_data_uses_binary_target_by_default(monkeypatch):
    """Confirms the flag actually gates which target function runs, without
    needing the full prepare_data pipeline's other feature steps."""
    calls = {"binary": 0, "risk_aware": 0}
    monkeypatch.setattr(wf, "make_target", lambda df: (calls.__setitem__("binary", calls["binary"] + 1), df)[1])
    monkeypatch.setattr(wf, "make_target_risk_aware", lambda df: (calls.__setitem__("risk_aware", calls["risk_aware"] + 1), df)[1])
    monkeypatch.setattr(wf, "RISK_AWARE_TARGET_ENABLED", False)

    df = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01")], "symbol": ["AAPL"]})
    for fn_name in ["ensure_aliases", "add_stock_specific_features", "add_tradability_flags",
                    "add_market_regime", "add_market_correlation_features",
                    "add_price_indicator_interaction_features", "add_momentum_state"]:
        monkeypatch.setattr(wf, fn_name, lambda d: d)

    wf.prepare_data(df)
    assert calls == {"binary": 1, "risk_aware": 0}
