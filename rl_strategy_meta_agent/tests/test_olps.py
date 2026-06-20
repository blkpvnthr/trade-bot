"""Tests for OLPS allocators: simplex weights, no lookahead, cost accounting."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta import olps


def _data(T=200, N=5, seed=0):
    rng = np.random.default_rng(seed)
    P = 100 * np.cumprod(1 + rng.normal(0.0003, 0.02, (T, N)), axis=0)
    return P, P[1:] / P[:-1]


def test_simplex_projection():
    v = olps.simplex_project(np.array([0.5, 0.2, -0.3, 1.1]))
    assert abs(v.sum() - 1) < 1e-9 and (v >= -1e-12).all()


def test_all_relative_algos_on_simplex():
    _, X = _data()
    for name, fn in olps.ALGOS_RELATIVE.items():
        B = fn(X)
        assert B.shape == X.shape
        assert np.allclose(B.sum(1), 1.0, atol=1e-6), f"{name} off simplex"
        assert (B >= -1e-9).all(), f"{name} negative weight"


def test_price_algos_and_meta_on_simplex():
    P, X = _data()
    for B in (olps.olmar(P), olps.rmr(P)):
        assert np.allclose(B.sum(1), 1.0, atol=1e-6) and (B >= -1e-9).all()
    Bm = olps.aggregating(X, [olps.eg(X), olps.pamr(X), olps.crp(X)])
    assert np.allclose(Bm.sum(1), 1.0, atol=1e-6)


def test_no_lookahead_weights_depend_only_on_past():
    # an online algo's weight at time t must not change if the FUTURE relatives change
    _, X = _data(seed=1)
    t = 120
    B1 = olps.eg(X)
    Xb = X.copy(); Xb[t + 1:] *= 1.5
    B2 = olps.eg(Xb)
    assert np.allclose(B1[:t + 1], B2[:t + 1], atol=1e-9)
    Bp1, Bp2 = olps.pamr(X), olps.pamr(Xb)
    assert np.allclose(Bp1[:t + 1], Bp2[:t + 1], atol=1e-9)


def test_cost_reduces_wealth_and_turnover_nonneg():
    _, X = _data()
    B = olps.pamr(X)                                    # high-turnover mean-reversion
    gross, turn = olps.wealth(X, B, cost=0.0)
    net, _ = olps.wealth(X, B, cost=0.001)
    assert (turn >= -1e-12).all()
    assert net[-1] < gross[-1]                          # costs strictly reduce wealth
    assert turn.mean() > 0                              # pamr actually trades


def test_crp_wealth_matches_manual():
    _, X = _data()
    B = olps.crp(X)
    curve, _ = olps.wealth(X, B)
    manual = np.cumprod((B * X).sum(1))                 # no cost, no drift adjustment in return
    assert abs(curve[-1] - manual[-1]) < 1e-6


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
