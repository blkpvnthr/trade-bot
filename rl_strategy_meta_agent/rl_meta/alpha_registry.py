"""Accepted-alpha registry: the single promotion gate into the mega-alpha ensemble.

An alpha is ACCEPTED only if it clears every gate; otherwise it is REJECTED with
the exact failing reasons. This is the "FDR replaces vibes" layer -- the final
discipline on top of generation / evolution / IC / multiple-testing.

Gates:
  fdr_significant       Benjamini-Hochberg discovery (block wild-bootstrap p-value)
  oos_sign_preserved    sign(train IC) == sign(OOS IC)
  ic_decay_acceptable   |OOS IC| >= retain * |train IC|
  turnover_below_cap    per-bar book turnover <= cap (also catches degenerate alphas)
  no_leakage            only point-in-time operators / non-negative windows
  corr_below_cap        |corr| to already-accepted alphas < cap (greedy, independence)

Research-only: pure bookkeeping, no broker / order / allocator / live path.
"""
from __future__ import annotations

import numpy as np

DEFAULT_CFG = {
    "oos_min_retain": 0.4,      # OOS IC must keep >= 40% of train IC magnitude
    "corr_cap": 0.7,            # pairwise abs-correlation cap (fast pre-filter)
    "indep_residual_min": 0.3,  # LINEAR INDEPENDENCE: >=30% of the alpha must be
                                # orthogonal to the span of the already-accepted set
    "turnover_cap": 2.0,        # fraction of book per bar; degenerate alphas exceed this
    "require_oos_sign": True,
}


def effective_breadth(corr_sub) -> float:
    """Participation ratio of the accepted-set correlation eigenvalues: the number
    of EFFECTIVE independent directions the ensemble spans (Linear-Independence paper).
    1 = all redundant, N = fully independent."""
    ev = np.linalg.eigvalsh(np.asarray(corr_sub, float))
    ev = ev[ev > 1e-9]
    return round(float(ev.sum() ** 2 / (ev ** 2).sum()), 3) if len(ev) else 0.0


def _orthogonal_residual(corr, cand, accepted):
    """Fraction of the candidate's variance NOT explained by the accepted set --
    sqrt(1 - R^2) where R^2 = r' Raa^-1 r (multiple correlation). Catches alphas
    inside the SPAN of others even when every pairwise correlation is modest."""
    A = [a for a in accepted if a in corr.index and a in corr.columns]
    if not A:
        return 1.0
    R = corr.loc[A, A].to_numpy(float) + 1e-6 * np.eye(len(A))
    r = corr.loc[cand, A].to_numpy(float)
    try:
        r2 = float(r @ np.linalg.solve(R, r))
    except np.linalg.LinAlgError:
        r2 = float(r @ np.linalg.pinv(R) @ r)
    return float(np.sqrt(max(0.0, 1.0 - min(r2, 1.0))))


def is_pit_safe(node) -> bool:
    """Point-in-time check: no operator references the future (all windows >= 1).
    Our grammar is PIT by construction; this catches any future-shift if added."""
    if not isinstance(node, tuple):
        return True
    op = node[0]
    if op in ("ts_delay", "ts_mean", "ts_std", "ts_sum", "ts_min", "ts_max", "ts_zscore",
              "ts_delta", "ts_ema", "ts_rank", "decay_linear", "ts_product",
              "ts_argmax", "ts_argmin", "ts_corr", "covariance"):
        if any(isinstance(a, (int, float)) and a < 1 for a in node[1:]):  # negative window
            return False
    return all(is_pit_safe(a) for a in node[1:] if isinstance(a, tuple))


def _gate_reasons(c, cfg) -> list:
    """Non-correlation gate failures for one candidate (exact reasons)."""
    r = []
    if not c.get("fdr_significant", False):
        r.append(f"fdr_not_significant(p={c.get('p_value')})")
    ic, oic = c.get("ic"), c.get("oos_ic")
    if oic is None or not np.isfinite(oic):
        r.append("oos_ic_missing")
    else:
        if cfg["require_oos_sign"] and np.sign(ic) != np.sign(oic):
            r.append(f"oos_sign_flip(train={ic:+.3f},oos={oic:+.3f})")
        if abs(oic) < cfg["oos_min_retain"] * abs(ic):
            r.append(f"ic_decay_excessive(oos={oic:+.3f}<{cfg['oos_min_retain']}*|{ic:+.3f}|)")
    to = c.get("turnover")
    if to is None or not np.isfinite(to) or to > cfg["turnover_cap"]:
        r.append(f"turnover_above_cap({to})")
    if not c.get("leak_ok", True):
        r.append("leakage_violation")
    return r


def promote(candidates, corr=None, cfg=None):
    """candidates: dicts with ic, oos_ic, p_value, fdr_significant, turnover, leak_ok,
    formula. corr: SIGNED correlation DataFrame (formula x formula) among candidates.
    Returns (accepted, rejected); each rejected carries 'reasons'. Gate order: scalar
    gates -> pairwise correlation cap -> LINEAR INDEPENDENCE (rank contribution)."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    acorr = corr.abs() if corr is not None else None
    accepted, rejected, survivors = [], [], []
    for c in sorted(candidates, key=lambda x: -abs(x["ic"])):
        reasons = _gate_reasons(c, cfg)
        (rejected if reasons else survivors).append(
            {**c, "reasons": reasons} if reasons else c)
    for c in survivors:
        f = c["formula"]
        cmax = 0.0
        if acorr is not None and accepted and f in acorr.index:
            vals = [float(acorr.loc[f, a["formula"]]) for a in accepted if a["formula"] in acorr.columns]
            cmax = max((v for v in vals if np.isfinite(v)), default=0.0)
        if cmax >= cfg["corr_cap"]:
            rejected.append({**c, "reasons": [f"correlated_redundant(corr={cmax:.2f})"]})
            continue
        resid = _orthogonal_residual(corr, f, [a["formula"] for a in accepted]) if corr is not None else 1.0
        if resid < cfg["indep_residual_min"]:
            rejected.append({**c, "reasons": [f"linearly_dependent(residual={resid:.2f}<"
                                              f"{cfg['indep_residual_min']}; inside span of accepted)"]})
            continue
        accepted.append({**c, "corr_to_existing": round(cmax, 3), "rank_contribution": round(resid, 3)})
    return accepted, rejected


def _demo():
    cands = [
        {"formula": "a", "ic": 0.30, "oos_ic": 0.28, "p_value": 0.0, "fdr_significant": True,
         "turnover": 0.3, "leak_ok": True},                       # clean -> accept
        {"formula": "b", "ic": 0.20, "oos_ic": -0.05, "p_value": 0.0, "fdr_significant": True,
         "turnover": 0.3, "leak_ok": True},                       # sign flip -> reject
        {"formula": "c", "ic": 0.25, "oos_ic": 0.02, "p_value": 0.0, "fdr_significant": True,
         "turnover": 0.3, "leak_ok": True},                       # decayed -> reject
        {"formula": "d", "ic": 0.22, "oos_ic": 0.20, "p_value": 0.3, "fdr_significant": False,
         "turnover": 0.3, "leak_ok": True},                       # not significant -> reject
        {"formula": "e", "ic": 0.21, "oos_ic": 0.20, "p_value": 0.0, "fdr_significant": True,
         "turnover": 99.0, "leak_ok": True},                      # turnover -> reject
    ]
    import pandas as pd
    corr = pd.DataFrame(np.eye(5), index=list("abcde"), columns=list("abcde"))
    acc, rej = promote(cands, corr)
    assert [a["formula"] for a in acc] == ["a"], [a["formula"] for a in acc]
    by = {r["formula"]: r["reasons"][0].split("(")[0] for r in rej}
    assert by["b"] == "oos_sign_flip" and by["c"] == "ic_decay_excessive"
    assert by["d"] == "fdr_not_significant" and by["e"] == "turnover_above_cap"
    assert is_pit_safe(("ts_mean", ("in", "close"), 5)) and not is_pit_safe(("ts_delay", ("in", "x"), -1))
    # LINEAR INDEPENDENCE: C sits inside the span of 5 modestly-correlated accepted
    # alphas (each pairwise 0.45 < cap) -> rejected though no single corr is high.
    idx = ["A1", "A2", "A3", "A4", "A5", "C"]
    M = np.eye(6)
    for i in range(5):
        M[i, 5] = M[5, i] = 0.45
    cz = pd.DataFrame(M, index=idx, columns=idx)
    cc = [{"formula": f, "ic": 0.3 - 0.01 * i, "oos_ic": 0.28 - 0.01 * i, "p_value": 0.0,
           "fdr_significant": True, "turnover": 0.3, "leak_ok": True} for i, f in enumerate(idx)]
    acc2, rej2 = promote(cc, cz)
    assert [a["formula"] for a in acc2] == ["A1", "A2", "A3", "A4", "A5"]
    assert any("linearly_dependent" in r["reasons"][0] for r in rej2 if r["formula"] == "C")
    assert effective_breadth(cz.loc[idx[:5], idx[:5]]) == 5.0
    print("alpha_registry._demo OK  accepted", [a["formula"] for a in acc],
          "| rejected", by, "| indep-gate drops C, breadth 5.0")


if __name__ == "__main__":
    _demo()
