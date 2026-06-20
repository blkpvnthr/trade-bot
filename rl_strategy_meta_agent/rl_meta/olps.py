"""Online Portfolio Selection (OLPS) algorithms -- the Li & Hoi survey taxonomy.

Each algorithm consumes a sequence of price-relative vectors x_t (x_{t,i} =
p_{t,i}/p_{t-1,i}) and produces a sequence of simplex weight vectors b_t, where
b_t is chosen BEFORE x_t is revealed (online, no lookahead). The two hindsight
benchmarks (best_stock, bcrp) deliberately use the whole path and are upper
bounds, clearly labelled as such.

  Benchmarks        : uniform_bah, best_stock*, crp, bcrp*        (* = hindsight)
  Follow-the-Winner : eg (Exponential Gradient)
  Follow-the-Loser  : pamr, olmar, anticor (mean-reversion)

Research-only: pure simulation, no broker / orders / live trading.
"""
from __future__ import annotations

import numpy as np


def simplex_project(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex (Duchi et al. 2008)."""
    v = np.asarray(v, float)
    if v.sum() == 1 and (v >= 0).all():
        return v
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1
    rho = np.nonzero(u - css / (np.arange(len(v)) + 1) > 0)[0][-1]
    theta = css[rho] / (rho + 1)
    return np.maximum(v - theta, 0)


def uniform_bah(X):
    """Buy-and-hold a uniform portfolio (no rebalancing; weights drift)."""
    T, N = X.shape
    b = np.ones(N) / N
    B = np.zeros((T, N))
    for t in range(T):
        B[t] = b
        b = b * X[t]; b = b / b.sum()
    return B


def crp(X, b0=None):
    """Constant Rebalanced Portfolio (rebalance to b0 each period; uniform default)."""
    T, N = X.shape
    b0 = np.ones(N) / N if b0 is None else simplex_project(b0)
    return np.tile(b0, (T, 1))


def best_stock(X):
    """HINDSIGHT: hold the single best-performing asset for the whole path."""
    T, N = X.shape
    i = int(np.argmax(np.prod(X, axis=0)))
    B = np.zeros((T, N)); B[:, i] = 1.0
    return B


def bcrp(X):
    """HINDSIGHT: best constant-rebalanced portfolio (max sum log<b,x_t>)."""
    from scipy.optimize import minimize
    T, N = X.shape
    cons = {"type": "eq", "fun": lambda b: b.sum() - 1}
    res = minimize(lambda b: -np.log(np.maximum(X @ b, 1e-12)).sum(),
                   np.ones(N) / N, bounds=[(0, 1)] * N, constraints=cons, method="SLSQP")
    b = simplex_project(res.x)
    return np.tile(b, (T, 1))


def eg(X, eta=0.05):
    """Exponential Gradient (Helmbold et al.) -- Follow-the-Winner."""
    T, N = X.shape
    b = np.ones(N) / N
    B = np.zeros((T, N))
    for t in range(T):
        B[t] = b
        x = X[t]
        b = b * np.exp(eta * x / (b @ x + 1e-12)); b = b / b.sum()
    return B


def pamr(X, eps=0.5, C=500.0):
    """Passive Aggressive Mean Reversion (Li et al.) -- Follow-the-Loser."""
    T, N = X.shape
    b = np.ones(N) / N
    B = np.zeros((T, N))
    for t in range(T):
        B[t] = b
        x = X[t]; xm = x.mean()
        loss = max(0.0, b @ x - eps)
        denom = np.sum((x - xm) ** 2) + 1e-12
        tau = min(C, loss / denom)
        b = simplex_project(b - tau * (x - xm))
    return B


def olmar(P, w=5, eps=10.0):
    """Online Moving Average Reversion (Li & Hoi). Takes PRICES P (T x N); predicts
    next relative as MA-reversion and does a passive-aggressive update."""
    P = np.asarray(P, float)
    T, N = P.shape
    b = np.ones(N) / N
    B = np.zeros((T - 1, N))
    for t in range(1, T):
        B[t - 1] = b
        lo = max(0, t - w)
        ma = P[lo:t].mean(axis=0)
        xpred = ma / (P[t - 1] + 1e-12)                  # predicted reversion relative
        xbar = xpred.mean()
        denom = np.sum((xpred - xbar) ** 2) + 1e-12
        tau = max(0.0, (eps - b @ xpred)) / denom
        b = simplex_project(b + tau * (xpred - xbar))
    return B


def anticor(X, w=10):
    """Anticor (Borodin et al.) -- Follow-the-Loser via cross-window correlation."""
    T, N = X.shape
    b = np.ones(N) / N
    B = np.zeros((T, N))
    L = np.log(np.maximum(X, 1e-12))
    for t in range(T):
        B[t] = b
        if t < 2 * w:
            continue
        y1 = L[t - 2 * w:t - w]; y2 = L[t - w:t]
        mu1, mu2 = y1.mean(0), y2.mean(0)
        s1, s2 = y1.std(0) + 1e-12, y2.std(0) + 1e-12
        cov = (y1 - mu1).T @ (y2 - mu2) / w
        corr = cov / np.outer(s1, s2)
        claim = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if mu2[i] >= mu2[j] and corr[i, j] > 0:
                    claim[i, j] = corr[i, j] + max(0, -corr[i, i]) + max(0, -corr[j, j])
        transfer = np.zeros((N, N))
        for i in range(N):
            s = claim[i].sum()
            if s > 0:
                transfer[i] = b[i] * claim[i] / s
        b = b + transfer.sum(0) - transfer.sum(1)
        b = simplex_project(b)
    return B


def ftl(X):
    """Follow the Leader -- online BCRP on the history so far (Follow-the-Winner)."""
    T, N = X.shape
    b = np.ones(N) / N
    B = np.zeros((T, N))
    for t in range(T):
        B[t] = b
        if t >= 2:
            b = bcrp(X[:t + 1])[0]
    return B


def ons(X, eta=0.0, beta=1.0, delta=0.125):
    """Online Newton Step (Follow-the-Regularized-Leader / 2nd-order FTW)."""
    T, N = X.shape
    b = np.ones(N) / N
    A = np.eye(N); p = np.zeros(N)
    B = np.zeros((T, N))
    for t in range(T):
        B[t] = b
        x = X[t]
        grad = x / (b @ x + 1e-12)
        A += np.outer(grad, grad)
        p += (1 + 1 / beta) * grad
        b = simplex_project(delta * np.linalg.solve(A, p))
    return B


def _geometric_median(Pw, iters=20, eps=1e-8):
    y = Pw.mean(0)
    for _ in range(iters):
        d = np.linalg.norm(Pw - y, axis=1) + eps
        w = 1 / d
        y = (w[:, None] * Pw).sum(0) / w.sum()
    return y


def rmr(P, w=5, eps=10.0):
    """Robust Median Reversion (Huang et al.) -- OLMAR with an L1-median predictor."""
    P = np.asarray(P, float)
    T, N = P.shape
    b = np.ones(N) / N
    B = np.zeros((T - 1, N))
    for t in range(1, T):
        B[t - 1] = b
        lo = max(0, t - w)
        med = _geometric_median(P[lo:t]) if t - lo >= 2 else P[t - 1]
        xpred = med / (P[t - 1] + 1e-12)
        xbar = xpred.mean()
        denom = np.sum((xpred - xbar) ** 2) + 1e-12
        tau = max(0.0, (eps - b @ xpred)) / denom
        b = simplex_project(b + tau * (xpred - xbar))
    return B


def corn(X, w=5, rho=0.2):
    """CORN: Correlation-driven Nonparametric pattern matching. BCRP over historical
    days whose preceding window correlates (> rho) with the current window."""
    T, N = X.shape
    b = np.ones(N) / N
    B = np.zeros((T, N))
    L = np.log(np.maximum(X, 1e-12))
    for t in range(T):
        B[t] = b
        if t <= w:
            continue
        cur = L[t - w:t].ravel()
        matched = [tp for tp in range(w, t)
                   if np.corrcoef(cur, L[tp - w:tp].ravel())[0, 1] > rho]
        b = bcrp(X[matched])[0] if len(matched) >= 1 else np.ones(N) / N
    return B


def aggregating(X, expert_Bs, eta=0.05):
    """Aggregating Algorithm / meta-learner: combine expert portfolios, weighting
    each by its realized log-wealth so far (capital flows to what is working)."""
    T, N = X.shape
    K = len(expert_Bs)
    logw = np.zeros(K)
    B = np.zeros((T, N))
    for t in range(T):
        wts = np.exp(logw - logw.max()); wts /= wts.sum()
        B[t] = sum(wts[k] * expert_Bs[k][t] for k in range(K))
        for k in range(K):
            logw[k] += eta * np.log(max(expert_Bs[k][t] @ X[t], 1e-12))
    return B


def wealth(X, B, cost=0.0):
    """Cumulative wealth of weight sequence B on relatives X, with proportional
    transaction cost `cost` per unit turnover. Returns (wealth_curve, turnover_t)."""
    T, N = X.shape
    w_prev = np.zeros(N)
    cur = 1.0
    curve = np.empty(T); turn = np.empty(T)
    for t in range(T):
        b = B[t]
        to = np.abs(b - w_prev).sum()                    # turnover vs drifted prev weight
        gross = b @ X[t]
        cur *= gross * (1 - cost * to)
        curve[t] = cur; turn[t] = to
        drift = b * X[t]
        w_prev = drift / (drift.sum() + 1e-12)           # weights drift with prices
    return curve, turn


# algorithms that consume price RELATIVES X (prices-based ones: olmar, rmr take P)
ALGOS_RELATIVE = {"uniform_bah": uniform_bah, "crp": crp, "best_stock*": best_stock,
                  "bcrp*": bcrp, "eg": eg, "ftl": ftl, "ons": ons,
                  "pamr": pamr, "anticor": anticor, "corn": corn}
# near-duplicates from the survey, mapped to their canonical implementation:
#   FTRL / Online-Newton-Update -> ons ; Online-Gradient-Update -> eg ;
#   nearest-neighbour / kernel log-optimal -> corn (pattern matching) ;
#   Follow-the-Leading-History -> aggregating (meta-learning)


def _demo():
    rng = np.random.default_rng(0)
    P = 100 * np.cumprod(1 + rng.normal(0.0003, 0.02, (300, 5)), axis=0)
    X = P[1:] / P[:-1]
    for name, fn in ALGOS_RELATIVE.items():
        B = fn(X)
        assert B.shape == X.shape
        assert np.allclose(B.sum(1), 1.0, atol=1e-6), f"{name} weights off-simplex"
        assert (B >= -1e-9).all(), f"{name} negative weight"
    for fn in (olmar, rmr):
        Bp = fn(P)
        assert np.allclose(Bp.sum(1), 1.0, atol=1e-6) and (Bp >= -1e-9).all()
    experts = [eg(X), pamr(X), crp(X)]
    Bm = aggregating(X, experts)
    assert np.allclose(Bm.sum(1), 1.0, atol=1e-6)
    c0, _ = wealth(X, crp(X))
    c1, t1 = wealth(X, pamr(X), cost=0.001)
    assert c0[-1] > 0 and len(t1) == len(X)
    print(f"olps._demo OK  {len(ALGOS_RELATIVE)+3} algos | crp wealth {c0[-1]:.3f}")


if __name__ == "__main__":
    _demo()
