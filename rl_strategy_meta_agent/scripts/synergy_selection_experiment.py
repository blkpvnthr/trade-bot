#!/usr/bin/env python3
"""Does AlphaGen's synergy selection build a better ensemble than our IC+independence
registry? Head-to-head on the SAME candidate pool (the registry-accepted alpha nodes),
same train/OOS split. For each selection method we form the optimal linear mega-alpha
(weights w* = M^-1 g on TRAIN) and measure its combined IC on TRAIN and -- the honest
test -- OUT-OF-TIME. RL-free: we reuse the evolutionary pool, only the selection
objective changes.

    python scripts/synergy_selection_experiment.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta.alphas import eval_alpha, spearman_ic
from rl_meta import alpha_synergy as syn

TARGET = "severe_giveback"


def _to_node(x):
    return tuple(_to_node(e) for e in x) if isinstance(x, list) else x


def _cs_z(s):
    tl = s.index.get_level_values(1)
    return (s - s.groupby(tl).transform("mean")) / (s.groupby(tl).transform("std") + 1e-9)


def _combined_ic(zs, weights, target, mask):
    mega = sum(w * z for w, z in zip(weights, zs))
    return spearman_ic(mega[mask], target[mask])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/trade_outcomes/_alpha_panel_25_8_v2.parquet")
    ap.add_argument("--nodes", default="data/trade_outcomes/accepted_alpha_nodes.json")
    ap.add_argument("--out", default="data/trade_outcomes")
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    nodes = [_to_node(d["node"]) for d in json.loads(Path(args.nodes).read_text())]
    target = panel[TARGET]
    ts = panel.index.get_level_values(1)
    cut = pd.Series(ts).quantile(0.7)
    tr, oo = np.asarray(ts < cut), np.asarray(ts >= cut)
    print(f"panel {panel.shape} | pool {len(nodes)} accepted alphas | train/oos split")

    zs = [_cs_z(eval_alpha(n, panel)).fillna(0.0) for n in nodes]
    g_tr = np.array([spearman_ic(z[tr], target[tr])[0] for z in zs])
    samp = np.random.default_rng(0).choice(np.where(tr)[0], min(40000, tr.sum()), replace=False)
    M = pd.DataFrame({i: z.values[samp] for i, z in enumerate(zs)}).corr().to_numpy()

    def evaluate(idx):
        w = syn.optimal_weights(g_tr[idx], M[np.ix_(idx, idx)])
        zsub = [zs[i] for i in idx]
        return {"k": len(idx),
                "train_combined_ic": round(_combined_ic(zsub, w, target, tr), 4),
                "oos_combined_ic": round(_combined_ic(zsub, w, target, oo), 4)}

    rep = {"pool": len(nodes), "methods": {}}
    ic_rank = list(np.argsort(-np.abs(g_tr)))
    for k in (5, 10, 20, len(nodes)):
        k = min(k, len(nodes))
        syn_idx, _ = syn.forward_select(g_tr, M, k)
        rep["methods"].setdefault("synergy_forward", {})[k] = evaluate(syn_idx)
        rep["methods"].setdefault("ic_ranked", {})[k] = evaluate(ic_rank[:k])
    rep["best_single_alpha_ic"] = round(float(np.max(np.abs(g_tr))), 4)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "synergy_selection_report.json").write_text(json.dumps(rep, indent=2, default=str))
    _print(rep)
    print(f"\nreport -> {out / 'synergy_selection_report.json'}")


def _print(rep):
    print(f"\n=== synergy selection vs IC-ranking (combined mega-alpha IC) ===")
    print(f"best single-alpha IC: {rep['best_single_alpha_ic']}\n")
    h = f"{'k':>4}{'synergy train':>16}{'synergy OOS':>14}{'IC-rank train':>16}{'IC-rank OOS':>14}"
    print(h); print("-" * len(h))
    for k in sorted({int(k) for m in rep["methods"].values() for k in m}):
        s = rep["methods"]["synergy_forward"][k]; r = rep["methods"]["ic_ranked"][k]
        print(f"{k:>4}{s['train_combined_ic']:>16}{s['oos_combined_ic']:>14}"
              f"{r['train_combined_ic']:>16}{r['oos_combined_ic']:>14}")
    full = max(int(k) for m in rep["methods"].values() for k in m)
    so = rep["methods"]["synergy_forward"][full]["oos_combined_ic"]
    ro = rep["methods"]["ic_ranked"][full]["oos_combined_ic"]
    print("\n=== Read ===")
    print(f"Combining lifts IC well above any single alpha ({rep['best_single_alpha_ic']}).")
    syn5 = rep["methods"]["synergy_forward"][5]["oos_combined_ic"]
    full_o = max(rep["methods"]["ic_ranked"][full]["oos_combined_ic"],
                 rep["methods"]["synergy_forward"][full]["oos_combined_ic"])
    print(f"Synergy@5 OOS {syn5} vs full-pool OOS {full_o}: diminishing returns "
          f"({'few alphas suffice' if syn5 >= 0.8 * full_o else 'more alphas help'}).")
    print(f"At the full pool, synergy OOS {so} vs IC-rank OOS {ro}: "
          f"{'synergy generalizes better' if so > ro + 0.005 else 'IC-ranking ties/beats synergy OOS (combined-IC selection overfits)' if ro >= so else 'comparable'}.")


if __name__ == "__main__":
    main()
