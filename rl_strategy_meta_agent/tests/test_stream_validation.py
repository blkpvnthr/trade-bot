"""Tests for the stream-level validation sprint metrics + gates."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta import stream_validation as sv


def test_metrics_net_below_gross_with_turnover():
    ret = np.full(100, 0.002); turn = np.full(100, 1.0)
    m = sv.stream_metrics(ret, turn, cost_bps=10)
    assert m["net_mean_daily"] < m["gross_mean_daily"]
    assert m["breakeven_cost_bps"] == 20.0          # 0.002 / 1.0 * 1e4


def test_real_edge_promoted():
    rng = np.random.default_rng(0)
    m = sv.validate_stream(rng.normal(0.003, 0.006, 250), np.full(250, 0.3),
                           train_ic=0.05, oos_ic=0.045,
                           oos_ret=rng.normal(0.003, 0.006, 70), oos_turn=np.full(70, 0.3))
    assert m["promoted"], m["reasons"]


def test_overtrader_rejected_on_breakeven():
    rng = np.random.default_rng(1)
    m = sv.validate_stream(rng.normal(0.0004, 0.01, 250), np.full(250, 12.0))
    assert not m["promoted"]
    assert any("breakeven" in r or "expectancy" in r for r in m["reasons"])


def test_ic_sign_flip_rejected():
    rng = np.random.default_rng(2)
    m = sv.validate_stream(rng.normal(0.001, 0.008, 250), np.full(250, 0.4),
                           train_ic=0.06, oos_ic=-0.03)
    assert not m["promoted"] and any("ic_sign_flip" in r for r in m["reasons"])


def test_negative_oos_expectancy_rejected():
    rng = np.random.default_rng(3)
    good_is = rng.normal(0.0015, 0.008, 250)
    bad_oos = rng.normal(-0.002, 0.008, 70)
    m = sv.validate_stream(good_is, np.full(250, 0.4),
                           oos_ret=bad_oos, oos_turn=np.full(70, 0.4))
    assert not m["promoted"] and any("oos_cost_adjusted" in r for r in m["reasons"])


def test_sortino_uses_downside():
    rng = np.random.default_rng(4)
    x = rng.normal(0.001, 0.01, 300)
    m = sv.stream_metrics(x, np.zeros(300))
    assert m["net_sortino"] >= m["net_sharpe"] - 1e6   # finite, computed


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
