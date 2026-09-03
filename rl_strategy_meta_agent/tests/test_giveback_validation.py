"""Unit tests for the giveback-overlay validation math (metrics / gates / equity)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import giveback_policy_validation as V


def test_metrics_basic():
    x = np.array([0.02, 0.03, -0.01, 0.04])
    base = np.array([0.01, 0.05, -0.02, 0.03])
    severe = np.array([False, True, True, False])
    et = np.arange(4)
    m = V._metrics(x, base, severe, et)
    assert m["total_return"] == round(float(x.sum()), 4)
    assert m["win_rate"] == 0.75
    assert m["max_drawdown"] >= 0.0


def test_gates_reward_lower_dd():
    base = {"max_drawdown": 0.10, "profit_factor": 2.0, "total_return": 1.0}
    better = {"max_drawdown": 0.07, "profit_factor": 2.1, "total_return": 0.95}
    g = V._gates(better, base)
    assert g["dd_better_20pct"] and g["pf_within_5pct"] and g["total_within_10pct"]
    worse = {"max_drawdown": 0.099, "profit_factor": 1.0, "total_return": 0.5}
    g2 = V._gates(worse, base)
    assert not g2["dd_better_20pct"] and not g2["pf_within_5pct"]


def test_equity_curve_runs():
    x = np.array([0.1, -0.05, 0.2]); et = np.arange(3)
    e = V._equity(x, et, frac=0.1, e0=100_000)
    assert e["final_equity"] > 0 and "max_drawdown_pct" in e and e["max_drawdown_pct"] >= 0


def test_label_fn_selects_target():
    import giveback_policy_backtest as B
    r = np.array([0.0, 0.01, 0.03, 0.04, 0.05, 0.02, -0.01])
    sg = B._label_fn("severe_giveback")
    ts = B._label_fn("tighten_stop_better")
    assert sg(r, 2) in (0, 1) and ts(r, 4) == 1     # tightening at the peak helps


def test_recovery_cost_splits_gains_and_misses():
    x = np.array([0.05, -0.02, 0.00])     # vs base: +0.05, -0.05, -0.03
    base = np.array([0.00, 0.03, 0.03])
    rc = V._recovery_cost(x, base)
    assert rc["avoided_giveback_return"] > 0 and rc["missed_recovery_return"] > 0
    assert rc["n_avoided"] == 1 and rc["n_missed"] == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
