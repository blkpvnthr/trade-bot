"""Leakage-prevention tests for the exit-quality decision-point model.

The whole point of this module is that mid-trade decisions may only use what is
known at bar k. These tests fail loudly if future information ever leaks into a
decision-time feature, or if a trade is split across train/test.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta import exit_quality as eq
from rl_meta.trade_features import (DURING_TRADE_FEATURES, EXIT_FEATURES,
                                    PRE_ENTRY_FEATURES)


R = np.array([0.0, 0.01, 0.03, 0.02, 0.05, 0.01, -0.01])  # rise, peak, give back


def test_running_features_ignore_the_future():
    # corrupting every bar AFTER k must not change the features computed at k
    for k in range(1, len(R) - 1):
        f = eq.running_features(R, k)
        r2 = R.copy()
        r2[k + 1:] = -9.0
        assert eq.running_features(r2, k) == f, f"future leaked into k={k}"


def test_targets_do_depend_on_the_future():
    # targets are labels: they MUST move when the post-k path changes
    base = eq.decision_targets(R, 3)
    r2 = R.copy()
    r2[4:] = 0.50  # huge rally after k -> holding becomes good, no giveback
    changed = eq.decision_targets(r2, 3)
    assert changed["hold_longer_good"] == 1
    assert changed["future_giveback"] <= base["future_giveback"]


def test_decision_inputs_exclude_whole_trade_during_and_exit_features():
    # RUN_FEATURES are point-in-time; none may be a whole-trade during/exit metric
    leaky = set(DURING_TRADE_FEATURES) | set(EXIT_FEATURES)
    assert not (set(eq.RUN_FEATURES) & leaky), "a whole-trade metric is used as input"
    # the only model inputs are pre-entry + running features
    allowed = set(PRE_ENTRY_FEATURES) | set(eq.RUN_FEATURES)
    rows = eq.build_decision_rows(R, {p: 1.0 for p in PRE_ENTRY_FEATURES}, {"trade_id": "t"})
    feat_keys = allowed  # what the trainer feeds the model
    assert not (feat_keys & (set(eq.DECISION_FLAGS) | {"future_giveback", "final_ret"})), \
        "a target name collides with a feature name"
    assert rows and all("ret_now" in r for r in rows)


def test_return_path_slices_only_within_the_trade_and_signs_direction():
    ts = np.arange(10, dtype="int64")
    close = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], float)
    # trade spans bars 2..6 -> 5 closes; long return relative to entry_price 102
    r = eq.return_path(ts, close, 2, 6, 102.0, is_long=True)
    assert len(r) == 5 and r[0] == 0.0 and r[-1] > 0
    # short of the same path is the negated return
    rs = eq.return_path(ts, close, 2, 6, 102.0, is_long=False)
    assert np.allclose(rs, -r)
    # too few bars -> no decision
    assert eq.return_path(ts, close, 2, 3, 102.0, True) is None


def test_trailing_stop_locks_before_the_dump():
    assert eq.trailing_stop_return(R, 0.01) < R.max()


def test_group_split_keeps_each_trade_whole():
    # a trade's decision points must never straddle train and test
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit
    dp = pd.DataFrame({"trade_id": np.repeat([f"t{i}" for i in range(20)], 3),
                       "symbol": np.repeat([f"S{i%4}" for i in range(20)], 3),
                       "x": np.arange(60)})
    gi, ge = next(GroupShuffleSplit(1, test_size=0.3, random_state=0)
                  .split(dp, groups=dp["symbol"]))
    train_tids = set(dp.iloc[gi]["trade_id"])
    test_tids = set(dp.iloc[ge]["trade_id"])
    assert not (train_tids & test_tids), "a trade leaked across the split"


def test_no_broker_or_order_path_in_module():
    src = Path(eq.__file__).read_text().lower()
    for banned in ("tradingclient", "submit_order", "alpaca", "base_url", "orders"):
        assert banned not in src, f"exit_quality must stay research-only ({banned})"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
