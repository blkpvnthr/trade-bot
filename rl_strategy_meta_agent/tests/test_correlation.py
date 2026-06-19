from datetime import date, timedelta

import numpy as np

from rl_meta.correlation import (parse_family_override, family_for, returns_series,
                                 pairwise, correlation_matrices, effective_breadth,
                                 independence_report, FILL_SAME_FAMILY,
                                 FILL_DIFF_FAMILY)
from rl_meta.types import TradeRecord


def tr(exit_date, ret):
    return TradeRecord("2020-01-01", exit_date, 1.0, 1.0, ret * 10000, ret, 1)


# -- family tagging --------------------------------------------------------
def test_family_override_comment_formats():
    assert parse_family_override("// family: trend") == "trend"
    assert parse_family_override("//strategy_family = mean_reversion") == "mean_reversion"
    assert parse_family_override("// family = breakout") == "breakout"
    assert parse_family_override("x = ta.sma(close, 10)") is None


def test_family_inference_from_adapter_and_name():
    assert family_for("Whatever", "tjl_trend_join") == "trend"
    assert family_for("Whatever", "pdh_pdl_break_retest") == "breakout"
    assert family_for("Whatever", "rsi_mean_reversion") == "mean_reversion"
    # name fallback when no adapter
    assert family_for("Options Flow Swing", None) == "flow"
    assert family_for("ATR Squeeze Expansion", None) == "volatility"
    assert family_for("Opening Range Breakout", None) == "breakout"
    assert family_for("Some Random Thing", None) == "unknown"


def test_family_override_wins_over_adapter():
    assert family_for("X", "tjl_trend_join", override="mean-reversion") == "mean_reversion"


# -- return series ---------------------------------------------------------
def test_daily_returns_aggregate_by_exit_date():
    s = returns_series([tr("2023-03-01", 0.10), tr("2023-03-01", 0.20),
                        tr("2023-03-05", -0.05)], freq="daily")
    assert set(s) == {"2023-03-01", "2023-03-05"}
    assert abs(s["2023-03-01"] - (1.1 * 1.2 - 1)) < 1e-9   # compounded
    assert abs(s["2023-03-05"] - (-0.05)) < 1e-9


def test_monthly_returns_aggregate_by_month():
    s = returns_series([tr("2023-03-01", 0.10), tr("2023-03-20", 0.10),
                        tr("2023-04-01", 0.05)], freq="monthly")
    assert set(s) == {"2023-03", "2023-04"}
    assert abs(s["2023-03"] - (1.1 * 1.1 - 1)) < 1e-9


# -- pairwise / overlap ----------------------------------------------------
def _series(keys, vals):
    return dict(zip(keys, vals))


def test_pairwise_uses_only_overlapping_dates():
    a = _series([f"p{i}" for i in range(10)], np.sin(np.arange(10)).tolist())
    b = _series([f"p{i}" for i in range(5, 15)], np.sin(np.arange(5, 15)).tolist())
    corr, overlap, reliable = pairwise(a, b, min_overlap=6)
    assert overlap == 5                  # p5..p9 only
    assert not reliable                  # 5 < 6


def test_low_overlap_marked_unreliable():
    a = _series(["m1", "m2", "m3"], [0.1, -0.2, 0.3])
    b = _series(["m1", "m2", "m3"], [0.2, -0.1, 0.25])
    _, overlap, reliable = pairwise(a, b, min_overlap=6)
    assert overlap == 3 and not reliable


# -- effective breadth -----------------------------------------------------
def test_effective_breadth_one_for_clones():
    keys = [f"p{i:02d}" for i in range(60)]
    base = np.sin(np.linspace(0, 6, 60))
    series = {f"c{k}": _series(keys, (base + 1e-3 * k).tolist()) for k in range(5)}
    fams = {k: "trend" for k in series}
    _, _, _, _, cleaned, _ = correlation_matrices(series, fams, "monthly", 6)
    assert effective_breadth(np.asarray(cleaned, float)) < 1.5


def test_effective_breadth_n_for_independent():
    keys = [f"p{i:02d}" for i in range(60)]
    series = {f"i{k}": _series(keys, np.sin(np.linspace(0, (k + 1) * 6.283, 60)
                                            * (k + 1)).tolist()) for k in range(5)}
    fams = {k: "unknown" for k in series}
    _, _, _, _, cleaned, _ = correlation_matrices(series, fams, "monthly", 6)
    assert effective_breadth(np.asarray(cleaned, float)) > 3.5


# -- conservative fill by family ------------------------------------------
def test_unreliable_filled_by_family():
    keys = ["m1", "m2", "m3"]                       # 3 points -> unreliable (<6)
    a = _series(keys, [0.1, -0.2, 0.3])
    b = _series(keys, [-0.3, 0.1, 0.05])
    # same-family fill
    rep_same = correlation_matrices({"a": a, "b": b},
                                    {"a": "trend", "b": "trend"}, "monthly", 6)[4]
    assert rep_same[0][1] == FILL_SAME_FAMILY
    # different-family fill
    rep_diff = correlation_matrices({"a": a, "b": b},
                                    {"a": "trend", "b": "flow"}, "monthly", 6)[4]
    assert rep_diff[0][1] == FILL_DIFF_FAMILY


# -- independence score ----------------------------------------------------
def test_independence_score_decreases_with_correlation():
    days = [(date(2020, 1, 1) + timedelta(days=i)).isoformat() for i in range(40)]
    base = np.sin(np.linspace(0, 6, 40))
    orth = np.sin(np.linspace(0, 25, 40))
    trades = {
        # A and B near-identical (high mutual correlation) -> low independence
        "A": [tr(days[i], float(base[i])) for i in range(40)],
        "B": [tr(days[i], float(base[i] + 1e-3)) for i in range(40)],
        # C orthogonal -> high independence
        "C": [tr(days[i], float(orth[i])) for i in range(40)],
    }
    rep = independence_report(trades, {"A": "trend", "B": "trend", "C": "trend"},
                              freq="daily", min_overlap=30)
    sA = rep.per_strategy["A"]["independence_score"]
    sC = rep.per_strategy["C"]["independence_score"]
    assert sA < sC                       # the clone is less independent
    assert 1 <= rep.effective_breadth <= 3


def test_report_handles_empty_and_singleton():
    assert independence_report({}, {}).n_strategies == 0
    one = independence_report({"x": [tr("2023-01-05", 0.1), tr("2023-02-05", 0.2)]},
                              {"x": "trend"})
    assert one.effective_breadth == 1.0
    assert one.per_strategy["x"]["independence_score"] == 1.0
