"""Tests for nested_cv.py: inner-fold splitting and the trial ledger used
for Objective 2's deflated-Sharpe trial counting.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from nested_cv import TrialLedger, select_best_config, split_inner_fold


def _dates(n):
    return list(pd.bdate_range("2023-01-02", periods=n))


def test_split_inner_fold_leaves_embargo_gap():
    dates = _dates(252)
    inner_train, inner_val = split_inner_fold(dates, val_fraction=0.2, embargo_days=5)
    train_end_idx = dates.index(inner_train[-1])
    val_start_idx = dates.index(inner_val[0])
    assert val_start_idx - train_end_idx - 1 == 5
    assert set(inner_train).isdisjoint(set(inner_val))


def test_split_inner_fold_val_fraction_approx():
    dates = _dates(300)
    inner_train, inner_val = split_inner_fold(dates, val_fraction=0.2, embargo_days=5)
    assert len(inner_val) == pytest.approx(60, abs=1)


def test_split_inner_fold_raises_on_too_few_dates():
    with pytest.raises(ValueError):
        split_inner_fold(_dates(5), val_fraction=0.2, embargo_days=5)


def test_split_inner_fold_never_touches_outer_test_dates():
    """inner fold only ever sees the dates passed in -- caller is
    responsible for passing only outer-train dates, never outer-test."""
    outer_train_dates = _dates(252)
    outer_test_dates = pd.bdate_range(outer_train_dates[-1] + pd.Timedelta(days=10), periods=42)
    inner_train, inner_val = split_inner_fold(outer_train_dates, val_fraction=0.2, embargo_days=5)
    assert set(inner_train + inner_val).isdisjoint(set(outer_test_dates))


def test_select_best_config_picks_highest_score():
    configs = [{"threshold": 0.5}, {"threshold": 0.6}, {"threshold": 0.7}]
    scores = {0.5: 1.0, 0.6: 2.5, 0.7: 0.3}
    best, best_score, table = select_best_config(configs, lambda c: scores[c["threshold"]])
    assert best == {"threshold": 0.6}
    assert best_score == 2.5
    assert len(table) == 3


def test_trial_ledger_records_and_counts():
    ledger = TrialLedger()
    ledger.record("baseline", "binary target, existing ranking", {}, [1.0, -0.5, 2.0])
    ledger.record("target_only", "cost-aware risk-normalized target", {"target": "risk_adj"}, [0.5, 0.8])
    assert ledger.n_trials == 2
    assert ledger.trials[0].n_outer_windows == 3


def test_trial_ledger_rejects_duplicate_trial_id():
    ledger = TrialLedger()
    ledger.record("baseline", "desc", {}, [1.0])
    with pytest.raises(ValueError):
        ledger.record("baseline", "desc2", {}, [2.0])


def test_trial_ledger_save_and_load_roundtrip(tmp_path):
    ledger = TrialLedger()
    ledger.record("baseline", "desc", {"a": 1}, [1.0, -0.5], holdout_net_sharpe=-3.72, pooled_net_sharpe=0.82)
    path = tmp_path / "ledger.json"
    ledger.save(path)

    loaded = TrialLedger.load(path)
    assert loaded.n_trials == 1
    assert loaded.trials[0].trial_id == "baseline"
    assert loaded.trials[0].holdout_net_sharpe == -3.72


def test_trial_ledger_load_missing_file_returns_empty(tmp_path):
    ledger = TrialLedger.load(tmp_path / "does_not_exist.json")
    assert ledger.n_trials == 0


def test_trial_ledger_accumulates_across_save_load_cycles(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = TrialLedger()
    ledger.record("baseline", "desc", {}, [1.0])
    ledger.save(path)

    ledger2 = TrialLedger.load(path)
    ledger2.record("target_only", "desc", {}, [0.5])
    ledger2.save(path)

    final = TrialLedger.load(path)
    assert final.n_trials == 2
    assert {t.trial_id for t in final.trials} == {"baseline", "target_only"}
