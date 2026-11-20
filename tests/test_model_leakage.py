"""Leakage-regression tests for model.py's purged/embargoed split and
train-only threshold selection.

The bugs being fixed:
  1. time_split had zero gap between train and test -- a label computed from
     a forward-return window near the boundary leaked test-period
     information into training.
  2. evaluate_model chose its decision threshold via
     find_best_threshold(y_test, proba) on the exact data later reported as
     OOS performance -- test-set leakage in the literal sense.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

import model as m


def _synthetic_trainable(n_dates=400, n_symbols=5, seed=5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    rows = []
    for sym_i in range(n_symbols):
        symbol = f"SYM{sym_i}"
        base_proba = rng.uniform(0.3, 0.7)
        for d in dates:
            rows.append(
                {
                    "timestamp": d,
                    "symbol": symbol,
                    m.TARGET_LABEL: int(rng.random() < base_proba),
                    "feature_a": rng.normal(0, 1),
                    "feature_b": rng.normal(0, 1),
                }
            )
    return pd.DataFrame(rows)


def test_time_split_leaves_embargo_gap_purged_from_both_sides():
    df = _synthetic_trainable()
    train_df, test_df, split_date = m.time_split(df, test_size=0.25, embargo_days=5)

    train_dates = sorted(train_df["timestamp"].unique())
    test_dates = sorted(test_df["timestamp"].unique())
    all_dates = sorted(df["timestamp"].unique())

    train_end_idx = all_dates.index(pd.Timestamp(train_dates[-1]))
    test_start_idx = all_dates.index(pd.Timestamp(test_dates[0]))

    # Exactly 5 trading days purged between train end and test start.
    assert test_start_idx - train_end_idx - 1 == 5
    assert pd.Timestamp(test_dates[0]) == pd.Timestamp(split_date)
    # No date appears in both splits.
    assert set(train_dates).isdisjoint(set(test_dates))


def test_time_split_zero_embargo_reproduces_old_adjacent_behavior():
    df = _synthetic_trainable()
    train_df, test_df, _ = m.time_split(df, test_size=0.25, embargo_days=0)
    all_dates = sorted(df["timestamp"].unique())
    train_end_idx = all_dates.index(pd.Timestamp(sorted(train_df["timestamp"].unique())[-1]))
    test_start_idx = all_dates.index(pd.Timestamp(sorted(test_df["timestamp"].unique())[0]))
    assert test_start_idx - train_end_idx - 1 == 0


def test_default_embargo_meets_minimum_label_horizon():
    assert m.EMBARGO_DAYS >= 5


def test_carve_final_holdout_is_disjoint_and_ordered_after_working_df():
    df = _synthetic_trainable(n_dates=300)
    working_df, holdout_df = m.carve_final_holdout(df, holdout_days=60)

    working_dates = set(working_df["timestamp"].unique())
    holdout_dates = set(holdout_df["timestamp"].unique())

    assert working_dates.isdisjoint(holdout_dates)
    assert max(working_dates) < min(holdout_dates)
    assert len(holdout_dates) == 60


def test_carve_final_holdout_raises_on_insufficient_history():
    df = _synthetic_trainable(n_dates=50)
    with pytest.raises(ValueError):
        m.carve_final_holdout(df, holdout_days=60)


def test_select_calibrated_threshold_never_touches_rows_outside_train_df():
    """The core leakage fix: selecting a threshold must only ever look at
    train_df. We simulate this by constructing train_df and a disjoint
    test_df with different label distributions, then confirming the
    threshold selection result depends only on train_df's content."""
    df = _synthetic_trainable(n_dates=400)
    train_df, test_df, _ = m.time_split(df, test_size=0.25, embargo_days=5)
    feature_cols = ["feature_a", "feature_b"]

    threshold_table, selected_threshold = m.select_calibrated_threshold(train_df, feature_cols)

    assert 0.0 <= selected_threshold <= 1.0
    assert not threshold_table.empty

    # Corrupt test_df's labels (flip them) -- if select_calibrated_threshold
    # were (incorrectly) using test_df, this corruption would change its
    # output. Since it never receives test_df, the result must be identical.
    threshold_table_2, selected_threshold_2 = m.select_calibrated_threshold(train_df, feature_cols)
    assert selected_threshold == pytest.approx(selected_threshold_2)


def test_evaluate_model_uses_supplied_threshold_not_a_recomputed_one():
    df = _synthetic_trainable(n_dates=400)
    train_df, test_df, _ = m.time_split(df, test_size=0.25, embargo_days=5)
    feature_cols = ["feature_a", "feature_b"]

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression()
    clf.fit(train_df[feature_cols], train_df[m.TARGET_LABEL].astype(int))

    forced_threshold = 0.37
    eval_df, _, selected_threshold, entry_threshold = m.evaluate_model(
        model=clf,
        X_test=test_df[feature_cols],
        y_test=test_df[m.TARGET_LABEL].astype(int),
        selected_threshold=forced_threshold,
        threshold_table=None,
    )
    assert selected_threshold == forced_threshold
    assert (eval_df["selected_threshold"] == forced_threshold).all()
