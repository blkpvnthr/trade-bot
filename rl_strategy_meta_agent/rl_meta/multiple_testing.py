"""Multiple-testing / data-snooping control for alpha mining (Giglio-Liao-Xiu 2021,
"Thousands of Alpha Tests"): when thousands of alphas are tested and the best are
picked by IC, many "discoveries" are false. We control the FALSE DISCOVERY RATE
with Benjamini-Hochberg, using a WILD BOOTSTRAP for a finite-sample p-value on each
alpha's per-period IC series, plus a deep-null SCREEN to improve power.

Pure stats, research-only.
"""
from __future__ import annotations

import numpy as np


def wild_bootstrap_pvalue(ic_t, n_boot: int = 1000, seed: int = 0, block: int = 1) -> float:
    """Two-sided p-value for mean(IC)=0 from the per-period IC series, via a
    Rademacher wild bootstrap under the null (the paper's finite-sample method).
    `block`>1 flips signs in BLOCKS to respect autocorrelation -- needed when the
    label spans several forward bars (overlapping windows -> correlated IC_t),
    otherwise i.i.d. flips understate p-values."""
    x = np.asarray(ic_t, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20 or x.std(ddof=1) < 1e-12:
        return 1.0
    se = x.std(ddof=1) / np.sqrt(n) + 1e-12
    t_obs = x.mean() / se
    c = x - x.mean()                                   # impose the null (mean 0)
    rng = np.random.default_rng(seed)
    nblk = int(np.ceil(n / max(1, block)))
    signs = np.repeat(rng.choice([-1.0, 1.0], size=(n_boot, nblk)), max(1, block), axis=1)[:, :n]
    bs = c * signs
    bt = bs.mean(1) / (bs.std(1, ddof=1) / np.sqrt(n) + 1e-12)
    return float((np.abs(bt) >= abs(t_obs)).mean())


def asymptotic_pvalue(ic_t) -> float:
    """Normal-approx two-sided p-value (fallback / cross-check)."""
    from math import erf, sqrt
    x = np.asarray(ic_t, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 20 or x.std(ddof=1) < 1e-12:
        return 1.0
    t = abs(x.mean() / (x.std(ddof=1) / np.sqrt(n) + 1e-12))
    return float(2 * (1 - 0.5 * (1 + erf(t / sqrt(2)))))


def benjamini_hochberg(pvalues, q: float = 0.1, n_tests: int | None = None):
    """B-H step-up FDR control. `n_tests` lets the burden reflect the FULL search
    (e.g. all alphas evaluated) even if p-values are only computed for the top set.
    Returns (discoveries_bool over input order, cutoff_pvalue, n_discoveries)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    ps = p[order]
    m = n_tests or len(p)
    k = len(ps)
    thresh = 0
    for i in range(1, k + 1):
        if ps[i - 1] <= (i / m) * q:
            thresh = i
    cutoff = ps[thresh - 1] if thresh > 0 else -1.0
    disc = p <= cutoff
    return disc, float(cutoff), int(disc.sum())


def fdr_control(pvalues, q: float = 0.1, n_tests: int | None = None,
                screen_cutoff: float = 0.5):
    """Screen deep-null alphas (p > screen_cutoff) for power, then B-H on the rest.
    Returns dict with the boolean discovery mask + diagnostics."""
    p = np.asarray(pvalues, float)
    keep = p <= screen_cutoff                          # data-driven deep-null screen
    m = int(keep.sum()) if n_tests is None else min(n_tests, int(keep.sum()) if keep.sum() else 1)
    disc = np.zeros(len(p), bool)
    if keep.any():
        d, cutoff, _ = benjamini_hochberg(p[keep], q, n_tests=max(m, int(keep.sum())))
        idx = np.where(keep)[0]
        disc[idx[d]] = True
    return {"discoveries": disc, "n_discoveries": int(disc.sum()),
            "n_screened": int(keep.sum()), "q": q,
            "expected_false_discoveries": round(q * int(disc.sum()), 2)}


def _demo():
    rng = np.random.default_rng(0)
    # 5 "real" alphas (IC mean 0.1, low noise) + 95 null alphas (IC mean 0)
    pvals = []
    for i in range(100):
        mu = 0.1 if i < 5 else 0.0
        ic_t = rng.normal(mu, 0.05, 200)
        pvals.append(wild_bootstrap_pvalue(ic_t, 500, seed=i))
    res = fdr_control(pvals, q=0.1, n_tests=100)
    # the 5 real ones should dominate the discoveries; few false positives
    disc_idx = set(np.where(res["discoveries"])[0])
    assert len(disc_idx & set(range(5))) >= 3, disc_idx
    assert res["n_discoveries"] <= 12, res          # FDR keeps false positives low
    # B-H sanity on a known vector
    d, cut, n = benjamini_hochberg([0.001, 0.01, 0.2, 0.7, 0.9], q=0.1)
    assert n == 2 and d.tolist() == [True, True, False, False, False]
    print(f"multiple_testing._demo OK  discoveries {sorted(disc_idx)} (truth 0-4)")


if __name__ == "__main__":
    _demo()
