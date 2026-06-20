"""Synergy-based alpha selection (AlphaGen, Yu et al. KDD'23) -- RL-FREE.

AlphaGen's headline argument: selecting alphas by standalone IC + mutual-IC
(correlation) filtering is MISALIGNED with the combined model's performance. The
right objective is the COMBINED mega-alpha's IC. Their Theorem 3.1 makes this
closed-form for a linear combination of normalised alphas: the best achievable
combined IC of a set is

    combined_IC*(F) = sqrt( g' M^-1 g )          (optimal weights w* = M^-1 g)

where g = vector of standalone ICs and M = the mutual-IC (correlation) matrix.
No gradient descent, no z_t, and -- importantly for this project -- no RL: we keep
our evolutionary generator and only borrow the SELECTION OBJECTIVE, so we can test
whether it beats our IC+independence registry at building the ensemble.

Research-only.
"""
from __future__ import annotations

import numpy as np


def optimal_weights(g, M, ridge=1e-2):
    g = np.asarray(g, float); M = np.asarray(M, float)
    return np.linalg.solve(M + ridge * np.eye(len(g)), g)


def combination_ic(g, M, ridge=1e-2):
    """Best achievable combined IC of the set (Theorem 3.1, sqrt(g' M^-1 g))."""
    g = np.asarray(g, float)
    w = optimal_weights(g, M, ridge)
    return float(np.sqrt(max(g @ w, 0.0)))


def forward_select(g, M, k, ridge=1e-2, min_gain=1e-4):
    """Greedy forward selection by MARGINAL combined-IC gain (AlphaGen Algorithm 1's
    objective, RL-free). Returns (selected_indices, combined_ic_trace)."""
    g = np.asarray(g, float); M = np.asarray(M, float)
    sel, remaining, trace, best = [], list(range(len(g))), [], 0.0
    while remaining and len(sel) < k:
        scored = [(combination_ic(g[sel + [j]], M[np.ix_(sel + [j], sel + [j])], ridge), j)
                  for j in remaining]
        ic, j = max(scored)
        if sel and ic <= best + min_gain:               # diminishing returns -> stop
            break
        best = ic; sel.append(j); remaining.remove(j); trace.append(round(ic, 5))
    return sel, trace


def _demo():
    rng = np.random.default_rng(0)
    k = 6
    g = np.full(k, 0.1)                                  # equal standalone IC
    # case 1: independent alphas (M = I) -> combined IC = sqrt(k)*ic (diversification)
    M = np.eye(k)
    assert abs(combination_ic(g, M, ridge=0) - np.sqrt(k) * 0.1) < 1e-6
    # case 2: a duplicate adds ~no synergy
    g2 = np.array([0.1, 0.1]); M2 = np.array([[1.0, 0.999], [0.999, 1.0]])
    gain = combination_ic(g2, M2) - combination_ic(g2[:1], M2[:1, :1])
    assert gain < 0.02, gain                             # near-duplicate -> tiny gain
    # forward selection prefers the diverse set
    G = np.array([0.12, 0.11, 0.10, 0.10])
    Mx = np.array([[1, .95, .1, .1], [.95, 1, .1, .1], [.1, .1, 1, .9], [.1, .1, .9, 1]], float)
    sel, trace = forward_select(G, Mx, k=4)
    assert 0 in sel and (2 in sel or 3 in sel)           # picks across both clusters
    print(f"alpha_synergy._demo OK  indep combIC {combination_ic(g, np.eye(k), 0):.3f} "
          f"| dup gain {gain:.4f} | forward {sel} trace {trace}")


if __name__ == "__main__":
    _demo()
