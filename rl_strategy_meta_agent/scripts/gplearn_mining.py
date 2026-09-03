#!/usr/bin/env python3
"""gplearn symbolic-regression alphas through the existing validation stack, with the
headline question: did gplearn discover a NEW INDEPENDENT signal family, or just
rediscover the volatility/giveback factor?

Generates candidates from 3 targets (severe_giveback, forward_return, risk-adjusted
forward_return), parses each to our AST, then scores them NOT by IC alone but by:
FDR significance, OOS IC sign/decay, turnover, ORTHOGONALITY to the existing
accepted-alpha space (rank contribution), volatility-proxy correlation, and the
incremental effective-breadth + combined-IC lift they bring to the ensemble.

Research-only: no broker/order/execution/live path; no retired strategies.

    python scripts/gplearn_mining.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta.alphas import BASE_INPUTS, eval_alpha, formula_str, alpha_id, category, spearman_ic
from rl_meta import gplearn_alpha_generator as gp, alpha_registry, alpha_synergy as syn
from rl_meta import multiple_testing as mt
from alpha_mining import _turnover                      # reuse the exact turnover metric

H_BLOCK = 8
SRC = "gplearn_symbolic_regression"
VOL_FEATURES = ["atr_pct", "realized_vol_20", "candle_range_pct", "rel_volume"]


def _to_node(x):
    return tuple(_to_node(e) for e in x) if isinstance(x, list) else x


def _cs_rank(s):
    return s.groupby(s.index.get_level_values(1)).rank()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/trade_outcomes/_alpha_panel_25_8_v2.parquet")
    ap.add_argument("--nodes", default="data/trade_outcomes/accepted_alpha_nodes.json")
    ap.add_argument("--out", default="data/trade_outcomes")
    ap.add_argument("--n-programs", type=int, default=12)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--population", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not gp.HAVE_GPLEARN:
        print("gplearn not installed -- nothing to do."); return
    out = Path(args.out)

    panel = pd.read_parquet(args.panel)
    giveback = panel["severe_giveback"]
    ts = panel.index.get_level_values(1)
    cut = pd.Series(ts).quantile(0.7)
    tr, oo = np.asarray(ts < cut), np.asarray(ts >= cut)
    targets = {"severe_giveback": giveback,
               "forward_return": panel["fut_ret_1"],
               "risk_adj_forward_return": panel["fut_ret_1"] / (panel["atr_pct"].abs() + 1e-3)}

    # --- generate from each target (gplearn fit on TRAIN rows only) ---
    Xtr = panel[BASE_INPUTS][tr].fillna(0.0).to_numpy(float)
    samp = np.random.default_rng(args.seed).choice(len(Xtr), min(15000, len(Xtr)), replace=False)
    progs, seen = [], set()
    for i, (tname, tgt) in enumerate(targets.items()):
        y = tgt[tr].fillna(0.0).to_numpy(float)
        for f, node in gp.generate(Xtr[samp], y[samp], BASE_INPUTS, args.n_programs,
                                   args.generations, args.population, args.seed + i):
            if f in seen or not alpha_registry.is_pit_safe(node):
                continue
            seen.add(f)
            progs.append({"node": node, "formula": f, "source_target": tname})
    print(f"gplearn generated {len(progs)} distinct valid candidate alphas from "
          f"{len(targets)} targets")

    # --- candidate metrics vs the giveback target (the registry's target) ---
    train, oos = panel[tr], panel[oo]
    for r in progs:
        v = eval_alpha(r["node"], train)
        r["ic"], r["ic_ir"], _ = spearman_ic(v, giveback[tr])
        r["oos_ic"], _, _ = spearman_ic(eval_alpha(r["node"], oos), giveback[oo])
        r["p_value"] = round(mt.wild_bootstrap_pvalue(_ic_series(v, giveback[tr]), 1000, block=H_BLOCK), 4)
        r["turnover"] = _turnover(eval_alpha(r["node"], panel))
        r["leak_ok"] = True
        r["category"] = "gplearn:" + category(r["node"])
    fdr = mt.fdr_control([r["p_value"] for r in progs], q=0.10, n_tests=len(progs))
    for r, d in zip(progs, fdr["discoveries"]):
        r["fdr_significant"] = bool(d)

    # --- correlation matrix: gplearn + EXISTING accepted + volatility features ---
    existing = [(formula_str(_to_node(d["node"])), _to_node(d["node"]))
                for d in json.loads(Path(args.nodes).read_text())]
    cols = {}
    for f, node in existing:
        cols["EX::" + f] = _cs_rank(eval_alpha(node, panel))
    for r in progs:
        cols[r["formula"]] = _cs_rank(eval_alpha(r["node"], panel))
    vol_feats = [vf for vf in VOL_FEATURES if vf in panel.columns]   # only those present
    for vf in vol_feats:
        cols["VOL::" + vf] = _cs_rank(panel[vf])
    V = pd.DataFrame(cols)
    rows = np.random.default_rng(0).choice(len(V), min(40000, len(V)), replace=False)
    corr = V.iloc[rows].corr()
    ex_cols = ["EX::" + f for f, _ in existing]

    for r in progs:                                     # orthogonality vs the existing space
        r["rank_contribution_vs_existing"] = round(
            alpha_registry._orthogonal_residual(corr, r["formula"], ex_cols), 3)
        r["vol_corr"] = round(float(max(abs(corr.loc[r["formula"], "VOL::" + vf]) for vf in vol_feats)), 3)

    # --- registry promotion (FDR + OOS + independence among gplearn) ---
    gcorr = corr.loc[[r["formula"] for r in progs], [r["formula"] for r in progs]]
    accepted, rejected = alpha_registry.promote(progs, gcorr, {"corr_cap": 0.7,
                         "indep_residual_min": 0.3, "turnover_cap": 2.0, "oos_min_retain": 0.4})

    # --- THE QUESTION: new family or volatility rediscovery? ---
    orthogonal = [a for a in accepted if a["rank_contribution_vs_existing"] >= 0.5
                  and a["vol_corr"] < 0.7]
    vol_proxies = [r for r in progs if r["vol_corr"] >= 0.7]
    g_ex = np.array([spearman_ic(eval_alpha(n, train), giveback[tr])[0] for _, n in existing])
    M_ex = corr.loc[ex_cols, ex_cols].to_numpy()
    breadth_before = alpha_registry.effective_breadth(M_ex)
    combic_before = syn.combination_ic(g_ex, M_ex)
    add_cols = ex_cols + [a["formula"] for a in orthogonal]
    g_all = np.concatenate([g_ex, [a["ic"] for a in orthogonal]])
    M_all = corr.loc[add_cols, add_cols].to_numpy()
    breadth_after = alpha_registry.effective_breadth(M_all)
    combic_after = syn.combination_ic(g_all, M_all)

    rep = {"generated": len(progs), "fdr_significant": int(sum(r["fdr_significant"] for r in progs)),
           "registry_accepted": len(accepted), "registry_rejected": len(rejected),
           "orthogonal_to_existing": len(orthogonal), "volatility_proxies": len(vol_proxies),
           "effective_breadth_before": breadth_before, "effective_breadth_after": breadth_after,
           "combined_ic_before": round(combic_before, 4), "combined_ic_after": round(combic_after, 4),
           "verdict": _verdict(orthogonal, breadth_before, breadth_after, combic_before, combic_after)}

    out.mkdir(parents=True, exist_ok=True)
    _csv(progs, out / "gplearn_candidates.csv",
         ["formula", "source_target", "ic", "oos_ic", "p_value", "fdr_significant",
          "turnover", "rank_contribution_vs_existing", "vol_corr"])
    _csv(accepted, out / "gplearn_accepted_alphas.csv",
         ["formula", "source_target", "ic", "oos_ic", "p_value", "turnover",
          "rank_contribution_vs_existing", "vol_corr", "corr_to_existing"])
    _csv([{**r, "reasons": "; ".join(r["reasons"])} for r in rejected],
         out / "gplearn_rejected_alphas.csv",
         ["formula", "source_target", "ic", "oos_ic", "p_value", "vol_corr", "reasons"])
    (out / "gplearn_run_report.json").write_text(json.dumps(rep, indent=2, default=str))
    _print(rep, orthogonal)
    print(f"\nartifacts -> {out}/ (gplearn_candidates.csv, _accepted, _rejected, _run_report.json)")


def _ic_series(alpha, target):
    from rl_meta.alphas import ic_series
    return ic_series(alpha, target)


def _csv(records, path, cols):
    pd.DataFrame([{c: r.get(c) for c in cols} for r in records]).to_csv(path, index=False)


def _verdict(orthogonal, b0, b1, c0, c1):
    if orthogonal and b1 > b0 + 0.3 and c1 > c0 + 0.005:
        return "NEW_SIGNAL_FAMILY: gplearn found alphas orthogonal to the existing space that lift breadth+combined IC"
    if orthogonal:
        return "WEAK_NEW_SIGNAL: some orthogonal alphas, but little breadth/combined-IC lift"
    return "VOLATILITY_REDISCOVERY: gplearn alphas are redundant with the existing volatility/giveback factor"


def _print(rep, orthogonal):
    print(f"\n=== gplearn alpha mining ===")
    print(f"generated {rep['generated']} | FDR-significant {rep['fdr_significant']} | "
          f"registry-accepted {rep['registry_accepted']} | volatility-proxies {rep['volatility_proxies']}")
    print(f"orthogonal-to-existing (rank-contrib>=0.5, vol-corr<0.7): {rep['orthogonal_to_existing']}")
    print(f"effective breadth: {rep['effective_breadth_before']} -> {rep['effective_breadth_after']}")
    print(f"combined IC:       {rep['combined_ic_before']} -> {rep['combined_ic_after']}")
    if orthogonal:
        print("orthogonal alphas found:")
        for a in sorted(orthogonal, key=lambda x: -x["rank_contribution_vs_existing"])[:5]:
            print(f"  rc {a['rank_contribution_vs_existing']} vol_corr {a['vol_corr']} "
                  f"IC {a['ic']:+.3f} (oos {a['oos_ic']:+.3f}) [{a['source_target']}]  {a['formula'][:50]}")
    print(f"\n=== VERDICT: {rep['verdict']} ===")


if __name__ == "__main__":
    main()
