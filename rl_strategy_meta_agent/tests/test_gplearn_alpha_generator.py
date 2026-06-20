"""Tests for the gplearn alpha generator. Skip cleanly if gplearn is absent."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta import gplearn_alpha_generator as G
from rl_meta import alphas as A
from rl_meta import alpha_registry as R

pytestmark = pytest.mark.skipif(not G.HAVE_GPLEARN, reason="gplearn not installed")
FEATS = ["f0", "f1", "f2", "f3"]


def _xy(n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 4))
    return X, X[:, 0] * X[:, 1] - 0.5 * X[:, 2]


def _panel(nsym=6, nt=80):
    ts = pd.date_range("2024-03-04 14:30", periods=nt, freq="15min", tz="UTC")
    rows = []
    for s in range(nsym):
        rng = np.random.default_rng(s)
        for i, t in enumerate(ts):
            rows.append((f"S{s}", t, *rng.normal(0, 1, 4)))
    return pd.DataFrame(rows, columns=["symbol", "timestamp", *FEATS]).set_index(
        ["symbol", "timestamp"]).sort_index()


def test_generates_parseable_asts():
    X, y = _xy()
    progs = G.generate(X, y, FEATS, n_programs=5, generations=6, population=500, seed=1)
    assert progs and all(isinstance(n, tuple) for _, n in progs)


def test_deterministic_seed_reproducibility():
    X, y = _xy()
    a = G.generate(X, y, FEATS, 5, 6, 500, seed=7)
    b = G.generate(X, y, FEATS, 5, 6, 500, seed=7)
    assert [f for f, _ in a] == [f for f, _ in b]


def test_formulas_evaluate_on_a_panel():
    X, y = _xy()
    progs = G.generate(X, y, FEATS, 5, 6, 500, seed=2)
    p = _panel()
    for _, node in progs:
        v = A.eval_alpha(node, p)
        assert v.shape[0] == len(p)


def test_generated_programs_are_pit_safe():
    X, y = _xy()
    for _, node in G.generate(X, y, FEATS, 5, 6, 500, seed=3):
        assert R.is_pit_safe(node)              # gplearn alphas use only PIT features, no ts ops


def test_only_safe_operators_used():
    X, y = _xy()
    allowed = set(G._OP.values()) | {"in", "const"}
    for _, node in G.generate(X, y, FEATS, 5, 6, 500, seed=4):
        stack = [node]
        while stack:
            n = stack.pop()
            if isinstance(n, tuple):
                assert n[0] in allowed, n[0]
                stack.extend(a for a in n[1:] if isinstance(a, tuple))


def test_accepted_alphas_cannot_bypass_registry_gates():
    # a gplearn-derived candidate that fails a gate must be rejected, not accepted
    cand = {"formula": "mul(f0,f1)", "ic": 0.2, "oos_ic": -0.1, "p_value": 0.0,
            "fdr_significant": True, "turnover": 0.3, "leak_ok": True}
    acc, rej = R.promote([cand], None)
    assert acc == [] and any("oos_sign_flip" in r["reasons"][0] for r in rej)


if __name__ == "__main__":
    if not G.HAVE_GPLEARN:
        print("gplearn absent -- skipping"); raise SystemExit
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
