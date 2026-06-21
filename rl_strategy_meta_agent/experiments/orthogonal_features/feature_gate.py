"""Orthogonal-feature gate: the falsifiable test a NEW feature source must pass
before any generator (AlphaEvolve / gplearn / EO) is allowed to mine over it.

gplearn and EO both confirmed that symbolic search over the current price/volume
panel converges to one volatility/giveback factor and creates no ensemble synergy.
So the constraint is no longer search quality -- it is whether we have independent
INPUT INFORMATION. This module answers, for each candidate feature column:

  1. Standalone PIT IC vs forward return (block wild-bootstrap p + BH FDR).
  2. Orthogonality: correlation to the volatility cluster + rank-contribution
     (residual) vs the existing accepted-alpha space.
  3. Incremental ensemble value: effective-breadth lift and combined OOS IC lift
     when the feature is added to the accepted set, net of turnover/cost.

Verdict per feature: ORTHOGONAL_INFORMATIVE / REDUNDANT_VOLATILITY / UNINFORMATIVE.

Reuses the existing stack (multiple_testing, alpha_registry, alpha_synergy, alphas)
so the bar is identical to the registry that admitted the current alphas. The caller
is responsible for passing LEAK-FREE features (backward/contemporaneous only); the
forward label is the only future-looking series. Research only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root on path
from rl_meta.alphas import eval_alpha, spearman_ic, ic_series
from rl_meta import alpha_registry as reg
from rl_meta import alpha_synergy as syn
from rl_meta import multiple_testing as mt

DEFAULTS = {
    "label": "fut_ret_1",          # directional forward-return target
    "vol_features": ["atr_pct", "rel_volume", "volume_acceleration"],
    "split_q": 0.7, "block": 8, "fdr_q": 0.10,
    "vol_corr_cap": 0.7,           # >= this to the vol cluster => volatility proxy
    "orthogonal_residual_min": 0.5,  # rank-contribution vs accepted set to count as independent
    "min_combined_ic_lift": 0.002,  # OOS combined-IC lift must clear this noise floor
    "turnover_cap": 2.0,
}


def _cs_rank(s):
    return s.groupby(s.index.get_level_values(1)).rank(pct=True)


def _cs_z(s):
    tl = s.index.get_level_values(1)
    return ((s - s.groupby(tl).transform("mean")) / (s.groupby(tl).transform("std") + 1e-9)).fillna(0.0)


def _turnover(rank_series):
    d = rank_series.groupby(level=0).diff().abs()
    return float(d.groupby(d.index.get_level_values(1)).mean().mean() * 2.0)


def study(panel, candidates: pd.DataFrame, accepted_nodes=None, cfg=None) -> dict:
    """panel: base panel with the forward label + volatility features.
    candidates: DataFrame aligned to panel.index, one column per candidate feature.
    accepted_nodes: list of AST nodes (the existing accepted-alpha space) or None.
    Returns a report dict with per-feature records + ensemble-lift summary."""
    c = {**DEFAULTS, **(cfg or {})}
    target = panel[c["label"]]
    ts = panel.index.get_level_values(1)
    cut = pd.Series(ts).quantile(c["split_q"])
    tr, oo = np.asarray(ts < cut), np.asarray(ts >= cut)

    # existing accepted-alpha space (cs-z), its breadth + combined OOS IC baseline
    acc = accepted_nodes or []
    acc_z = [_cs_z(eval_alpha(n, panel)) for n in acc]
    g_acc = np.array([spearman_ic(z[tr], target[tr])[0] for z in acc_z]) if acc_z else np.array([])

    # build one correlation matrix over accepted + candidates + vol features
    cols = {}
    for i, z in enumerate(acc_z):
        cols[f"ACC{i}"] = _cs_rank(z)
    for name in candidates.columns:
        cols[f"CAND::{name}"] = _cs_rank(candidates[name])
    vol_present = [v for v in c["vol_features"] if v in panel.columns]
    for v in vol_present:
        cols[f"VOL::{v}"] = _cs_rank(panel[v])
    V = pd.DataFrame(cols)
    samp = np.random.default_rng(0).choice(len(V), min(40000, len(V)), replace=False)
    corr = V.iloc[samp].corr().fillna(0.0)
    acc_names = [f"ACC{i}" for i in range(len(acc_z))]

    base_breadth = reg.effective_breadth(corr.loc[acc_names, acc_names].to_numpy()) if acc_names else 0.0
    base_combic = _combined_oos_ic(acc_z, g_acc, corr.loc[acc_names, acc_names].to_numpy(),
                                   target, tr, oo) if len(acc_z) >= 2 else 0.0

    pvals, recs = [], []
    for name in candidates.columns:
        feat = candidates[name]
        rank = _cs_rank(feat)
        ic_tr, _, _ = spearman_ic(rank[tr], target[tr])
        sign = 1.0 if ic_tr >= 0 else -1.0
        rank_s = rank if sign > 0 else (1.0 - rank)
        ic_tr = abs(ic_tr)
        ic_oo, _, _ = spearman_ic(rank_s[oo], target[oo])
        p = mt.wild_bootstrap_pvalue(ic_series(rank_s[tr], target[tr]), 1000, block=c["block"])
        pvals.append(p)
        vol_corr = max((abs(corr.loc[f"CAND::{name}", f"VOL::{v}"]) for v in vol_present), default=0.0)
        resid = reg._orthogonal_residual(corr, f"CAND::{name}", acc_names) if acc_names else 1.0
        recs.append({"feature": name, "sign": sign, "train_ic": round(ic_tr, 4),
                     "oos_ic": round(ic_oo, 4), "p_value": round(p, 4),
                     "vol_corr": round(float(vol_corr), 3),
                     "rank_contribution": round(float(resid), 3),
                     "turnover": round(_turnover(rank_s), 3)})

    fdr = mt.fdr_control(pvals, q=c["fdr_q"], n_tests=len(pvals))
    for r, sig in zip(recs, fdr["discoveries"]):
        r["fdr_significant"] = bool(sig)
        # incremental ensemble lift: accepted + THIS feature
        z = _cs_z(candidates[r["feature"]]) * r["sign"]
        names = acc_names + [f"CAND::{r['feature']}"]
        M = corr.loc[[n for n in names], [n for n in names]].to_numpy()
        if len(acc_z) >= 1:
            g2 = np.append(g_acc, r["train_ic"])
            r["breadth_lift"] = round(reg.effective_breadth(M) - base_breadth, 3)
            r["combined_oos_ic_lift"] = round(
                _combined_oos_ic(acc_z + [z], g2, M, target, tr, oo) - base_combic, 4)
        else:
            r["breadth_lift"] = None; r["combined_oos_ic_lift"] = None
        r["verdict"] = _verdict(r, c)

    return {"label": c["label"], "n_candidates": len(recs),
            "base_effective_breadth": round(base_breadth, 3),
            "base_combined_oos_ic": round(base_combic, 4),
            "orthogonal_informative": sum(r["verdict"] == "ORTHOGONAL_INFORMATIVE" for r in recs),
            "features": recs,
            "summary": _summary(recs)}


def _combined_oos_ic(zs, g, M, target, tr, oo):
    w = syn.optimal_weights(g, M)
    mega = sum(wi * z for wi, z in zip(w, zs))
    return spearman_ic(mega[oo], target[oo])[0]


def _verdict(r, c):
    if not r["fdr_significant"] and (r["combined_oos_ic_lift"] or 0) <= 0:
        return "UNINFORMATIVE"
    if r["turnover"] > c["turnover_cap"]:
        return "UNINFORMATIVE"            # signal churns faster than cost allows
    if r["vol_corr"] >= c["vol_corr_cap"] or r["rank_contribution"] < c["orthogonal_residual_min"]:
        return "REDUNDANT_VOLATILITY"
    lift = r["combined_oos_ic_lift"]
    if lift is None:                      # no accepted set to compare -> judge standalone
        return "ORTHOGONAL_INFORMATIVE" if r["fdr_significant"] else "WEAK"
    # the user's bar: orthogonal AND a real OOS combined-IC lift (breadth alone is not enough)
    if lift > c["min_combined_ic_lift"]:
        return "ORTHOGONAL_INFORMATIVE"
    return "WEAK"                         # orthogonal but no OOS predictive contribution


def _summary(recs):
    n_orth = sum(r["verdict"] == "ORTHOGONAL_INFORMATIVE" for r in recs)
    if n_orth:
        return (f"{n_orth} feature(s) clear the gate: independent of the volatility cluster AND "
                "add ensemble breadth/combined OOS IC. Worth feeding to the generators.")
    return ("No candidate cleared the gate -- all are volatility-redundant or uninformative for "
            "forward return. This source does not add orthogonal information; new data needed.")


def _demo():
    # synthetic panel: target driven by a HIDDEN orthogonal feature + a volatility proxy
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-02 14:30", periods=200, freq="15min", tz="UTC")
    rows = []
    for s in range(20):
        for t in ts:
            vol = abs(rng.normal())                       # volatility-like
            orth = rng.normal()                           # independent signal
            fut = 0.3 * orth + 0.02 * rng.normal()        # forward return ~ orthogonal feature
            rows.append((f"S{s}", t, vol, orth, rng.normal(), fut))
    p = pd.DataFrame(rows, columns=["symbol", "timestamp", "atr_pct", "_orth", "_noise", "fut_ret_1"]
                     ).set_index(["symbol", "timestamp"]).sort_index()
    p["rel_volume"] = p["atr_pct"]; p["volume_acceleration"] = p["atr_pct"]
    cand = pd.DataFrame({"vol_proxy": p["atr_pct"] + 0.01 * rng.normal(size=len(p)),
                         "orthogonal_signal": p["_orth"],
                         "pure_noise": p["_noise"]}, index=p.index)
    rep = study(p, cand, accepted_nodes=None)
    v = {r["feature"]: r["verdict"] for r in rep["features"]}
    assert v["orthogonal_signal"] != "UNINFORMATIVE", v          # real signal detected
    assert v["pure_noise"] == "UNINFORMATIVE", v                 # noise rejected
    print("feature_gate._demo OK", v)


if __name__ == "__main__":
    _demo()
