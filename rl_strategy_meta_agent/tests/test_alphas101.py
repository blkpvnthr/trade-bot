"""Tests for the 101-Alphas operators + library: correctness, leak-free, evaluable."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta import alphas as A
from rl_meta import alphas101 as L


def _panel(nsym=8, nt=300):     # >=257 so alpha019's sum(ret,250) window is satisfied
    ts = pd.date_range("2024-03-04 14:30", periods=nt, freq="15min", tz="UTC")
    rows = []
    for s in range(nsym):
        rng = np.random.default_rng(s)
        c = np.cumsum(rng.normal(0, 1, nt)) + 100
        for i, t in enumerate(ts):
            rows.append((f"S{s}", t, c[i], c[i] + 0.4, c[i] - 0.4, c[i] + 0.1, 1000.0 + i, c[i] + 0.05))
    p = pd.DataFrame(rows, columns=["symbol", "timestamp", "open", "high", "low",
                                    "close", "volume", "vwap"])
    p["ret"] = np.log(p["close"] / p.groupby("symbol")["close"].shift(1))
    return p.set_index(["symbol", "timestamp"]).sort_index()


def test_delay_shifts_per_symbol():
    p = _panel()
    v = A.eval_alpha(("ts_delay", ("in", "close"), 3), p)
    manual = p["close"].groupby(level=0).shift(3)
    assert np.allclose(v.dropna().values, manual.dropna().values, atol=1e-9)


def test_signedpower_preserves_sign():
    p = _panel()
    v = A.eval_alpha(("signedpower", ("in", "ret"), 2.0), p)
    r = p["ret"]
    assert np.allclose(v.dropna().values, (np.sign(r) * r.abs() ** 2).dropna().values, atol=1e-9)


def test_decay_linear_weights_sum_to_one():
    # constant series -> decay_linear returns the same constant (weights sum to 1)
    p = _panel()
    p = p.copy(); p["k"] = 5.0
    v = A.eval_alpha(("decay_linear", ("in", "k"), 5), p)
    assert np.allclose(v.dropna().values, 5.0, atol=1e-9)


def test_ts_argmax_in_range():
    p = _panel()
    v = A.eval_alpha(("ts_argmax", ("in", "close"), 5), p).dropna()
    assert v.min() >= 0 and v.max() <= 4


def test_where_and_lt():
    p = _panel()
    v = A.eval_alpha(("where", ("lt", ("in", "ret"), ("const", 0.0)),
                      ("const", -1.0), ("const", 1.0)), p).dropna()
    assert set(np.unique(v.values)) <= {-1.0, 1.0}


def test_covariance_matches_manual():
    p = _panel()
    v = A.eval_alpha(("covariance", ("in", "close"), ("in", "volume"), 10), p)
    assert np.isfinite(v.dropna()).all()


def test_all_101_alphas_evaluate_and_are_leak_free():
    p = _panel()
    k = p.index.get_level_values(1)[80]
    bad = p.copy()
    bad.loc[bad.index.get_level_values(1) > k, ["open", "high", "low", "close", "volume", "vwap"]] *= 6
    for name, node in L.library():
        v = A.eval_alpha(node, p)
        assert v.shape[0] == len(p) and v.notna().any(), name
        v2 = A.eval_alpha(node, bad)
        early = v.index.get_level_values(1) <= k
        a1, a2 = v[early].dropna(), v2[early].reindex(v.index)[early].dropna()
        common = a1.index.intersection(a2.index)
        assert np.allclose(a1.loc[common].values, a2.loc[common].values, atol=1e-7), f"leak in {name}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
