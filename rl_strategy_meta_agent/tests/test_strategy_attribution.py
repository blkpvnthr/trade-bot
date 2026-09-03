"""Unit tests for the strategy-attribution selection math."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import strategy_attribution as S


def test_trade_metrics_basic():
    ret = np.array([0.02, -0.01, 0.03, -0.02])
    et = np.arange(4)
    m = S._trade_metrics(ret, et)
    assert m["trades"] == 4
    assert m["total_return"] == round(float(ret.sum()), 4)
    assert m["win_rate"] == 0.5
    assert m["max_drawdown"] >= 0.0


def test_trade_metrics_empty():
    assert S._trade_metrics(np.array([]), np.array([]))["trades"] == 0


def test_pf_and_auc_guards():
    assert S._pf(np.array([0.1, 0.2, -0.1])) > 0
    assert S._auc([1, 1, 1], [0.5, 0.6, 0.7]) is None     # one class -> None
    assert S._auc([0, 1] * 30, list(np.linspace(0, 1, 60))) is not None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
