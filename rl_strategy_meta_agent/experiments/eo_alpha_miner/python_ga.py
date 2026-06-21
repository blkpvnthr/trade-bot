"""Pure-Python fallback miner: a small EO-style GA (population, tournament
selection, subtree crossover via mutate, elitism, deterministic seed) over the
existing alphas.py expression trees. Used when the C++ EO binary is absent (CI),
so the experiment and tests run everywhere. Research only.

Reuses alphas.py for the tree representation, mutation, and evaluation -- the
only thing here is the generational loop.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root on path
from rl_meta.alphas import random_alpha, mutate, eval_alpha, formula_str, spearman_ic


def _fitness(node, panel, target, parsimony=0.002):
    ic, _, _ = spearman_ic(eval_alpha(node, panel), target)
    size = _nodes(node)
    return abs(ic) - parsimony * size if np.isfinite(ic) else -1e9


def _nodes(node) -> int:
    return 1 + sum(_nodes(c) for c in node[1:] if isinstance(c, tuple)) if isinstance(node, tuple) else 0


def mine(panel, target, feature_names, cfg: dict) -> list[tuple]:
    """Evolve and return the top-K unique expression-tree nodes (best first)."""
    rng = np.random.default_rng(cfg.get("seed", 42))
    pop_size = cfg.get("population", 120)
    gens = cfg.get("generations", 25)
    max_depth = cfg.get("max_depth", 4)
    mut_rate = cfg.get("mutation_rate", 0.3)
    top_k = cfg.get("top_k", 25)
    t_size = 3

    pop = [random_alpha(rng, inputs=feature_names, max_depth=max_depth) for _ in range(pop_size)]
    scored = [(_fitness(n, panel, target), n) for n in pop]
    for _ in range(gens):
        scored.sort(key=lambda x: -x[0])
        elite = scored[: max(2, pop_size // 10)]            # elitism
        children = []
        while len(children) < pop_size - len(elite):
            # tournament select a parent, then mutate (subtree perturbation)
            cand = [scored[rng.integers(len(scored))] for _ in range(t_size)]
            parent = max(cand, key=lambda x: x[0])[1]
            child = mutate(parent, rng, inputs=feature_names) if rng.random() < mut_rate else \
                random_alpha(rng, inputs=feature_names, max_depth=max_depth)
            children.append((_fitness(child, panel, target), child))
        scored = elite + children
    scored.sort(key=lambda x: -x[0])

    out, seen = [], set()
    for _, n in scored:
        f = formula_str(n)
        if f in seen:
            continue
        seen.add(f); out.append(n)
        if len(out) >= top_k:
            break
    return out


def _demo():
    import pandas as pd
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-02 14:30", periods=40, freq="15min", tz="UTC")
    rows = []
    for s in range(8):
        for t in ts:
            f0, f1 = rng.normal(size=2)
            rows.append((f"S{s}", t, f0, f1, f0 - 0.5 * f1 + 0.3 * rng.normal()))
    p = pd.DataFrame(rows, columns=["symbol", "timestamp", "f0", "f1", "target"]).set_index(
        ["symbol", "timestamp"]).sort_index()
    a = mine(p, p["target"], ["f0", "f1"], {"seed": 1, "population": 60, "generations": 10})
    b = mine(p, p["target"], ["f0", "f1"], {"seed": 1, "population": 60, "generations": 10})
    assert a and [formula_str(n) for n in a] == [formula_str(n) for n in b]   # deterministic
    print(f"python_ga._demo OK  {len(a)} alphas, best={formula_str(a[0])[:50]}")


if __name__ == "__main__":
    _demo()
