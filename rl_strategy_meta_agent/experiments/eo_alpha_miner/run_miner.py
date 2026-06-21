#!/usr/bin/env python3
"""EO-backed symbolic alpha miner -- CLI.

Mines symbolic alphas from an OHLCV CSV with the EO genetic-programming backbone
(falls back to a pure-Python GA when the C++ binary is absent), validates each on
held-out time splits, and runs the falsifiable synergy-selection comparison:

  A. simple IC-ranking
  B. IC minus correlation penalty (greedy)
  C. combined mega-alpha synergy objective (AlphaGen-style, RL-free)

Reports whether synergy selection (C) lifts OUT-OF-SAMPLE combined IC over plain
IC-ranking (A). Research only -- imports no trading/order/live code.

    python experiments/eo_alpha_miner/run_miner.py --csv fixtures/smoke_ohlcv.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl_meta.alphas import eval_alpha, formula_str, spearman_ic
from rl_meta import alpha_synergy as syn
import features as FEAT
import backend as BK
import python_ga as PGA


def _cs_z(s):
    tl = s.index.get_level_values(1)
    return ((s - s.groupby(tl).transform("mean")) / (s.groupby(tl).transform("std") + 1e-9)).fillna(0.0)


def _splits(panel, f_train, f_val):
    ts = panel.index.get_level_values(1)
    u = np.sort(pd.unique(ts))
    a, b = u[int(len(u) * f_train)], u[int(len(u) * (f_train + f_val))]
    arr = np.asarray(ts)
    return arr < a, (arr >= a) & (arr < b), arr >= b


def _load_ohlcv(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.set_index(["symbol", "timestamp"]).sort_index()


def _hit_rate(alpha, target, mask):
    za, zt = _cs_z(alpha)[mask], _cs_z(target)[mask]
    ok = np.sign(za.values) == np.sign(zt.values)
    return round(float(np.mean(ok)), 4)


def _greedy_ic_corr(g, M, k, lam=0.5):
    chosen, remaining = [], list(range(len(g)))
    while remaining and len(chosen) < k:
        best, bs = None, -1e9
        for j in remaining:
            pen = max((abs(M[j, c]) for c in chosen), default=0.0)
            score = abs(g[j]) - lam * pen
            if score > bs:
                bs, best = score, j
        chosen.append(best); remaining.remove(best)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="OHLCV CSV: symbol,timestamp,open,high,low,close,volume[,vwap]")
    ap.add_argument("--out", default="experiments/eo_alpha_miner/out")
    ap.add_argument("--backend", choices=["auto", "eo", "python"], default="auto")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--population", type=int, default=300)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--crossover-rate", type=float, default=0.7)
    ap.add_argument("--mutation-rate", type=float, default=0.3)
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--accepted-nodes", default="data/trade_outcomes/accepted_alpha_nodes.json")
    args = ap.parse_args()

    csv = args.csv or str(Path(__file__).resolve().parent / "fixtures" / "smoke_ohlcv.csv")
    cfg = {k: getattr(args, k) for k in
           ("population", "generations", "max_depth", "seed", "crossover_rate", "mutation_rate", "top_k")}
    use_eo = (args.backend == "eo") or (args.backend == "auto" and BK.available())

    panel, feats, tgt = FEAT.build_features(_load_ohlcv(csv))
    target = panel[tgt]
    tr, va, te = _splits(panel, args.train_frac, args.val_frac)
    print(f"panel {panel.shape} | {len(feats)} features | split tr/va/te = "
          f"{tr.sum()}/{va.sum()}/{te.sum()} | backend = {'EO-C++' if use_eo else 'python-GA'}")

    # --- mine on TRAIN only ---
    if use_eo:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        train_csv = str(Path(args.out) / "_train_features.csv")
        FEAT.export_for_eo(panel[tr], feats, tgt, train_csv)
        nodes = BK.run(train_csv, str(Path(args.out) / "_eo_formulas.txt"), cfg, feats)
    else:
        nodes = PGA.mine(panel[tr], target[tr], feats, cfg)
    print(f"mined {len(nodes)} candidate expressions")

    # --- validate each candidate across splits (sign-flip so train IC >= 0) ---
    cands, seen = [], set()
    accepted = _load_accepted(args.accepted_nodes, panel)
    for node in nodes:
        f = formula_str(node)
        if f in seen:
            continue
        seen.add(f)
        a = eval_alpha(node, panel)
        tic = spearman_ic(a[tr], target[tr])[0]
        if not np.isfinite(tic) or abs(tic) < 1e-6:
            continue
        sign = 1.0 if tic >= 0 else -1.0
        a = a * sign
        z = _cs_z(a)
        rec = {"formula": f, "sign": sign, "complexity": _nodes(node),
               "train_ic": round(abs(tic), 4),
               "val_ic": round(spearman_ic(a[va], target[va])[0], 4),
               "oos_ic": round(spearman_ic(a[te], target[te])[0], 4),
               "hit_rate": _hit_rate(a, target, te),
               "alpha_corr": _max_corr(z, accepted, tr)}
        rec["stable"] = bool(rec["val_ic"] > 0 and rec["oos_ic"] > 0)
        cands.append((rec, z))

    recs = [c[0] for c in cands]
    print(f"validated {len(recs)} alphas | stable (val+oos same sign) = {sum(r['stable'] for r in recs)}")

    # --- synergy comparison A / B / C ---
    synrep = _synergy(cands, target, tr, te)

    # --- artifacts ---
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "alphas.json").write_text(json.dumps(recs, indent=2, default=str))
    pd.DataFrame(recs).to_csv(out / "alphas.csv", index=False)
    _report(out / "report.md", csv, use_eo, panel, feats, recs, synrep, cfg)
    print(f"\nartifacts -> {out}/ (alphas.json, alphas.csv, report.md)")
    print(f"=== synergy verdict: {synrep['verdict']} ===")


def _synergy(cands, target, tr, te):
    if len(cands) < 2:
        return {"verdict": "too few alphas for synergy comparison", "table": []}
    zs = [z for _, z in cands]
    g = np.array([c["train_ic"] for c, _ in cands])
    samp = np.random.default_rng(0).choice(np.where(tr)[0], min(20000, int(tr.sum())), replace=False)
    M = pd.DataFrame({i: z.values[samp] for i, z in enumerate(zs)}).corr().fillna(0.0).to_numpy()

    def oos(idx):
        w = syn.optimal_weights(g[idx], M[np.ix_(idx, idx)])
        mega = sum(wi * zs[i] for wi, i in zip(w, idx))
        return round(spearman_ic(mega[te], target[te])[0], 4)

    ic_rank = list(np.argsort(-g))
    table, verdict_rows = [], []
    for k in sorted({3, 5, 10, len(cands)}):
        k = min(k, len(cands))
        A = oos(ic_rank[:k])
        B = oos(_greedy_ic_corr(g, M, k))
        C = oos(syn.forward_select(g, M, k)[0])
        table.append({"k": k, "A_ic_rank": A, "B_ic_corr": B, "C_synergy": C})
        verdict_rows.append((C, A))
    Cbest = max(r[0] for r in verdict_rows); Abest = max(r[1] for r in verdict_rows)
    verdict = ("synergy (C) lifts OOS IC over IC-ranking (A)" if Cbest > Abest + 0.005
               else "IC-ranking (A) ties/beats synergy (C) OOS -- no diversification gain"
               if Abest >= Cbest else "comparable")
    return {"verdict": verdict, "table": table, "best_single_ic": round(float(g.max()), 4)}


def _load_accepted(path, panel):
    p = Path(path)
    if not p.exists():
        return []
    def to_node(x):
        return tuple(to_node(e) for e in x) if isinstance(x, list) else x
    out = []
    for d in json.loads(p.read_text()):
        try:
            out.append(_cs_z(eval_alpha(to_node(d["node"]), panel)))
        except Exception:
            continue
    return out


def _max_corr(z, accepted, tr):
    if not accepted:
        return None
    cs = [abs(np.corrcoef(z[tr].values, a[tr].values)[0, 1]) for a in accepted]
    cs = [c for c in cs if np.isfinite(c)]
    return round(max(cs), 4) if cs else None


def _nodes(node):
    return 1 + sum(_nodes(c) for c in node[1:] if isinstance(c, tuple)) if isinstance(node, tuple) else 0


def _report(path, csv, use_eo, panel, feats, recs, synrep, cfg):
    top = sorted(recs, key=lambda r: -r["oos_ic"])[:10]
    L = ["# EO-backed symbolic alpha miner -- run report", "",
         f"- dataset: `{csv}`  ({panel.shape[0]} rows, {len(feats)} features)",
         f"- backend: **{'EO C++ genetic programming' if use_eo else 'Python GA fallback'}**",
         f"- config: {cfg}", "",
         "## Top alphas by OOS IC", "",
         "| formula | train IC | val IC | OOS IC | hit | nodes | stable |",
         "|---|---|---|---|---|---|---|"]
    for r in top:
        L.append(f"| `{r['formula'][:60]}` | {r['train_ic']} | {r['val_ic']} | "
                 f"{r['oos_ic']} | {r['hit_rate']} | {r['complexity']} | {r['stable']} |")
    L += ["", "## Synergy selection: A (IC-rank) vs B (IC+corr) vs C (synergy)", "",
          f"best single-alpha train IC: {synrep.get('best_single_ic')}", "",
          "| k | A OOS IC | B OOS IC | C OOS IC |", "|---|---|---|---|"]
    for t in synrep["table"]:
        L.append(f"| {t['k']} | {t['A_ic_rank']} | {t['B_ic_corr']} | {t['C_synergy']} |")
    L += ["", f"**Verdict: {synrep['verdict']}**", ""]
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
