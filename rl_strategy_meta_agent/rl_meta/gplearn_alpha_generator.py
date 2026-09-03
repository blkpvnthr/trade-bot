"""gplearn symbolic-regression alpha generator -- an ADDITIONAL research-only source
that feeds the existing validation stack (IC -> block wild-bootstrap p-values -> BH
FDR -> OOS -> linear-independence/effective-breadth -> accepted-alpha registry). It
does NOT replace the AlphaEvolve engine.

The goal is ORTHOGONAL symbolic alphas, not higher-IC volatility clones: gplearn
searches the arithmetic-combination space of our leak-free per-bar features, which is
a different region than our time-series-operator evolution. Every program is parsed
into the existing AST so it is evaluated, gated, and scored identically -- and tested
for independence from the existing accepted (volatility-dominated) alpha space.

Optional dependency: if gplearn is absent, HAVE_GPLEARN is False and callers/tests
skip. Research-only: no broker/order/execution/live path.
"""
from __future__ import annotations

import numpy as np

try:
    from gplearn.genetic import SymbolicTransformer
    HAVE_GPLEARN = True
except Exception:
    HAVE_GPLEARN = False

# user-restricted safe operator set (no sin/cos/tan; all leak-free elementwise)
SAFE_FUNCTIONS = ["add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "inv", "max", "min"]
# gplearn function name -> our AST op (max/min renamed to avoid clashing with ts_max/ts_min)
_OP = {"add": "add", "sub": "sub", "mul": "mul", "div": "div", "sqrt": "sqrt",
       "log": "log", "abs": "abs", "neg": "neg", "inv": "inv", "max": "gmax", "min": "gmin"}


def program_to_ast(program, feature_names):
    """Convert a gplearn _Program (prefix list) into our AST tuple. Leak-free by
    construction: leaves are PIT feature columns or constants; no time-series ops."""
    it = iter(program.program)

    def build():
        el = next(it)
        if isinstance(el, (int, np.integer)):
            return ("in", feature_names[int(el)])
        if isinstance(el, (float, np.floating)):
            return ("const", float(el))
        args = [build() for _ in range(el.arity)]              # el is a gplearn _Function
        return (_OP[el.name], *args)

    return build()


def generate(X, y, feature_names, n_programs=12, generations=10, population=1000,
             seed=0, parsimony=0.001):
    """Fit a SymbolicTransformer (Spearman/IC metric) and return [(formula, ast)] for
    its diverse hall-of-fame programs. X: 2D array (rows=samples), y: 1D target."""
    if not HAVE_GPLEARN:
        raise RuntimeError("gplearn not installed")
    gp = SymbolicTransformer(
        generations=generations, population_size=population,
        hall_of_fame=max(40, 3 * n_programs), n_components=n_programs,
        function_set=[f for f in SAFE_FUNCTIONS], metric="spearman",
        parsimony_coefficient=parsimony, max_samples=0.9, const_range=(-2.0, 2.0),
        feature_names=list(feature_names), random_state=seed, n_jobs=1, verbose=0)
    gp.fit(np.asarray(X, float), np.asarray(y, float))
    out = []
    for p in gp._best_programs:
        try:
            out.append((str(p), program_to_ast(p, feature_names)))
        except Exception:
            continue                                            # skip unparseable programs
    return out


def _demo():
    if not HAVE_GPLEARN:
        print("gplearn_alpha_generator._demo SKIP (gplearn absent)"); return
    rng = np.random.default_rng(0)
    feats = ["f0", "f1", "f2", "f3"]
    X = rng.normal(0, 1, (800, 4))
    y = X[:, 0] * X[:, 1] - 0.5 * X[:, 2]                       # a learnable arithmetic target
    progs = generate(X, y, feats, n_programs=5, generations=6, population=500, seed=1)
    assert progs and all(isinstance(n, tuple) for _, n in progs)
    # determinism: same seed -> same programs
    progs2 = generate(X, y, feats, n_programs=5, generations=6, population=500, seed=1)
    assert [f for f, _ in progs] == [f for f, _ in progs2]
    print(f"gplearn_alpha_generator._demo OK  {len(progs)} programs | e.g. {progs[0][0][:60]}")


if __name__ == "__main__":
    _demo()
