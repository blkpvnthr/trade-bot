"""Tests for FDR / multiple-testing control and ic_series."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta import multiple_testing as mt
from rl_meta import alphas as A


def test_benjamini_hochberg_known_vector():
    d, cutoff, n = mt.benjamini_hochberg([0.001, 0.01, 0.2, 0.7, 0.9], q=0.1)
    assert n == 2 and d.tolist() == [True, True, False, False, False]


def test_bh_burden_makes_it_stricter():
    p = [0.001, 0.02]
    _, _, n_small = mt.benjamini_hochberg(p, q=0.1, n_tests=2)
    _, _, n_big = mt.benjamini_hochberg(p, q=0.1, n_tests=1000)
    assert n_big <= n_small                         # larger search burden -> fewer discoveries


def test_wild_bootstrap_low_p_for_strong_signal():
    rng = np.random.default_rng(0)
    strong = rng.normal(0.1, 0.03, 250)            # clear positive IC
    null = rng.normal(0.0, 0.05, 250)              # no signal
    assert mt.wild_bootstrap_pvalue(strong, 1000) < 0.05
    assert mt.wild_bootstrap_pvalue(null, 1000) > 0.10


def test_wild_bootstrap_degenerate_returns_one():
    assert mt.wild_bootstrap_pvalue(np.ones(5)) == 1.0
    assert mt.wild_bootstrap_pvalue(np.full(200, 0.03)) == 1.0   # zero variance


def test_fdr_control_recovers_signal_and_bounds_false():
    rng = np.random.default_rng(1)
    pvals = [mt.wild_bootstrap_pvalue(rng.normal(0.12 if i < 4 else 0.0, 0.04, 200), 500, seed=i)
             for i in range(80)]
    res = mt.fdr_control(pvals, q=0.1, n_tests=80)
    disc = set(np.where(res["discoveries"])[0])
    assert len(disc & set(range(4))) >= 2          # finds the real ones
    assert res["n_discoveries"] <= 10              # few false positives


def test_ic_series_matches_spearman_mean():
    ts = pd.date_range("2024-03-04", periods=60, freq="15min", tz="UTC")
    rows = [(f"S{s}", t, np.sin(i / 5 + s)) for s in range(6) for i, t in enumerate(ts)]
    df = pd.DataFrame(rows, columns=["symbol", "timestamp", "x"]).set_index(["symbol", "timestamp"]).sort_index()
    x = df["x"]; y = df["x"].groupby(level=0).shift(1)
    series = A.ic_series(x, y)
    mean_ic, _, _ = A.spearman_ic(x, y)
    assert abs(series.mean() - mean_ic) < 1e-4       # spearman_ic rounds to 5dp


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
