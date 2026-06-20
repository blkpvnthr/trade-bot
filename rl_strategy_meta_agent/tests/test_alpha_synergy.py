"""Tests for synergy-based selection (AlphaGen combined-IC objective, RL-free)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta import alpha_synergy as syn


def test_independent_alphas_diversify():
    # k independent alphas with equal IC -> combined IC = sqrt(k) * ic
    for k in (2, 4, 8):
        g = np.full(k, 0.1)
        assert abs(syn.combination_ic(g, np.eye(k), ridge=0) - np.sqrt(k) * 0.1) < 1e-6


def test_duplicate_adds_no_synergy():
    g = np.array([0.1, 0.1])
    M = np.array([[1.0, 0.999], [0.999, 1.0]])
    gain = syn.combination_ic(g, M) - syn.combination_ic(g[:1], M[:1, :1])
    assert gain < 0.02


def test_anticorrelated_alphas_super_diversify():
    # negatively-correlated equal-IC alphas combine to MORE than sqrt(2)*ic
    g = np.array([0.1, 0.1]); M = np.array([[1.0, -0.5], [-0.5, 1.0]])
    assert syn.combination_ic(g, M, ridge=0) > np.sqrt(2) * 0.1


def test_forward_select_picks_across_clusters():
    g = np.array([0.12, 0.11, 0.10, 0.10])
    M = np.array([[1, .95, .1, .1], [.95, 1, .1, .1],
                  [.1, .1, 1, .9], [.1, .1, .9, 1]], float)
    sel, trace = syn.forward_select(g, M, k=4)
    assert sel[0] == 0                                   # highest standalone IC first
    assert (2 in sel or 3 in sel)                        # then jump to the other cluster
    assert trace == sorted(trace)                        # combined IC is non-decreasing


def test_forward_select_stops_on_diminishing_returns():
    g = np.array([0.1, 0.1, 0.1])
    M = np.full((3, 3), 0.999); np.fill_diagonal(M, 1.0)  # all near-identical
    sel, _ = syn.forward_select(g, M, k=3, min_gain=0.01)
    assert len(sel) == 1                                 # extras add ~nothing -> stop


def test_optimal_weights_solve():
    g = np.array([0.1, 0.05]); M = np.eye(2)
    w = syn.optimal_weights(g, M, ridge=0)
    assert np.allclose(w, g)                             # M=I -> w = g


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
