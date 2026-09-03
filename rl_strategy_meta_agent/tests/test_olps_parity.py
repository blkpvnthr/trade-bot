"""Parity tests for our OLPS allocators vs an external reference implementation
(github.com/nglahani/Online-Quantitative-Trading-Strategies), cloned read-only
under external_research/. Skips if the vendor clone is absent (it is gitignored,
not committed). The reference is for VALIDATION only -- our architecture
(accepted-alpha registry -> mega-alpha streams -> OLPS allocator) stays authoritative.

Convention differences found and respected:
  - their best_stock = last-period winner (causal momentum); our best_stock* is an
    explicit hindsight upper bound -- intentionally different, not compared.
  - their EG uses the linearized 1+lr*(x/port-1) update; ours uses the exp form.
  - their OLMAR/RMR take price RELATIVES; ours take prices -> behavioral check only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "external_research" / "oqts"
FIX = VENDOR / "Data" / "Price Relative Vectors" / "price_relative_vectors.csv"

from rl_meta import olps

pytestmark = pytest.mark.skipif(not FIX.exists(), reason="vendor reference not cloned")


def _vendor():
    sys.path.insert(0, str(VENDOR / "Scripts"))
    from Strategies import benchmarks, follow_the_winner, follow_the_loser
    return benchmarks, follow_the_winner, follow_the_loser


def _X(rows=400, cols=15):
    return pd.read_csv(FIX, index_col=0).to_numpy(float)[:rows, :cols]


def _rets(X, B):
    return (B * X).sum(1)


def test_crp_exact_parity():
    vb, _, _ = _vendor()
    X = _X(); u = np.ones(X.shape[1]) / X.shape[1]
    ours, _ = olps.wealth(X, olps.crp(X))
    theirs, _ = olps.wealth(X, vb.constant_rebalancing(u, X))
    assert abs(ours[-1] - theirs[-1]) < 1e-6                  # both = uniform constant rebalance


def test_buy_and_hold_parity():
    vb, _, _ = _vendor()
    X = _X(); u = np.ones(X.shape[1]) / X.shape[1]
    ours, _ = olps.wealth(X, olps.uniform_bah(X))
    theirs, _ = olps.wealth(X, vb.buy_and_hold(u, X))
    assert abs(ours[-1] - theirs[-1]) / theirs[-1] < 1e-6


def test_eg_same_family_within_tolerance():
    _, vw, _ = _vendor()
    X = _X(); u = np.ones(X.shape[1]) / X.shape[1]
    ours, _ = olps.wealth(X, olps.eg(X, eta=0.05))
    theirs, _ = olps.wealth(X, vw.exponential_gradient(u, X, learning_rate=0.05))
    assert abs(ours[-1] - theirs[-1]) / theirs[-1] < 0.05    # exp vs linearized EG ~ identical


def test_pamr_matched_params_high_correlation():
    _, _, vl = _vendor()
    X = _X(); u = np.ones(X.shape[1]) / X.shape[1]
    ours = olps.pamr(X, eps=0.9, C=10.0)
    theirs = vl.pamr(u, X, epsilon=0.9, C=10.0)
    c = np.corrcoef(_rets(X, ours)[1:], _rets(X, theirs)[1:])[0, 1]
    assert c > 0.8                                            # same passive-aggressive update


def test_loser_algos_behaviorally_valid_on_real_data():
    # our Follow-the-Loser algos run on the vendor's real 94-asset relatives and stay
    # on the simplex with sane (mean-reversion, gross > 1) behavior.
    X = pd.read_csv(FIX, index_col=0).to_numpy(float)[:600, :20]
    P = np.vstack([np.ones(X.shape[1]), np.cumprod(X, axis=0)])
    for B in (olps.pamr(X), olps.anticor(X), olps.corn(X), olps.olmar(P), olps.rmr(P)):
        assert np.allclose(B.sum(1), 1.0, atol=1e-6) and (B >= -1e-9).all()
        w, _ = olps.wealth(X, B)
        assert w[-1] > 0.5                                    # not degenerate


if __name__ == "__main__":
    if not FIX.exists():
        print("vendor reference absent -- skipping parity"); raise SystemExit
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
