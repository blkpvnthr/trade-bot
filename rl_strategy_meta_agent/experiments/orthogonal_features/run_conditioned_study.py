#!/usr/bin/env python3
"""Earnings/event-CONDITIONED meta-features through the gate.

The news scan showed event STRUCTURE (earnings_within_7d) carries orthogonal OOS
signal while sentiment does not. This tests the next hypothesis: does event proximity
work as a CONDITIONING variable on existing signals -- base_signal x event_proximity --
adding OOS lift BEYOND the standalone parts? (idio_reversal x earnings_within_7d,
volatility x earnings_within_14d, ORB x catalyst_window, ...)

The honest test: each interaction is scored against an accepted space that ALREADY
contains the 8 accepted alphas PLUS every standalone base signal and conditioner
(feature_gate.accepted_extra). So ORTHOGONAL_INFORMATIVE here means the interaction
adds combined OOS IC that neither parent provides alone. Research only.

    python experiments/orthogonal_features/run_conditioned_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RL_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import feature_gate as G
import news_features as NF
from run_feature_study import build_candidates

# base_signal -> conditioner pairs to test (the user's examples + a curated grid)
PAIRS = [
    ("idio_reversal", "earnings_within_7d"), ("idio_reversal", "earnings_within_14d"),
    ("idio_reversal", "event_recency_decay"), ("idio_reversal", "catalyst_score"),
    ("vol_factor", "earnings_within_7d"), ("vol_factor", "earnings_within_14d"),
    ("orb_breakout", "earnings_within_7d"), ("orb_breakout", "catalyst_score"),
    ("rel_strength_5", "earnings_within_7d"),
]


def main():
    panel = pd.read_parquet(RL_ROOT / "data/trade_outcomes/_alpha_panel_25_8_v2.parquet")
    syms = sorted(panel.index.get_level_values(0).unique())
    nodes_path = RL_ROOT / "data/trade_outcomes/accepted_alpha_nodes.json"
    nodes = None
    if nodes_path.exists():
        def to_node(x):
            return tuple(to_node(e) for e in x) if isinstance(x, list) else x
        nodes = [to_node(d["node"]) for d in json.loads(nodes_path.read_text())]

    # base signals (price) + conditioners (leak-free news/earnings)
    px = build_candidates(panel)
    base = pd.DataFrame({"idio_reversal": px["idio_reversal"], "rel_strength_5": px["rel_strength_5"],
                         "vol_factor": panel["atr_pct"], "orb_breakout": panel["or_high_dist"]},
                        index=panel.index)
    cond = NF.align(panel.index, NF.load_news(syms), NF.load_earnings(syms), {"window_h": 24.0})

    # interactions = cs_z(base) * conditioner  (leak-free: both factors are PIT)
    inter = {}
    for b, k in PAIRS:
        src = base[b] if b in base else cond[b]
        inter[f"{b}__x__{k}"] = (G._cs_z(src) * cond[k]).fillna(0.0)
    inter = pd.DataFrame(inter, index=panel.index)

    # accepted space already holds the standalone parts -> tests for ADDED value
    extra = pd.concat([base, cond[["earnings_within_7d", "earnings_within_14d",
                                   "event_recency_decay", "catalyst_score"]]], axis=1)
    print(f"panel {panel.shape} | {inter.shape[1]} conditioned interactions | "
          f"accepted set = {len(nodes) if nodes else 0} alphas + {extra.shape[1]} standalone parts")

    rep = G.study(panel, inter, accepted_nodes=nodes, accepted_extra=extra)
    out = Path(__file__).resolve().parent / "out"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rep["features"]).to_csv(out / "conditioned_feature_gate_report.csv", index=False)
    (out / "conditioned_feature_gate_report.json").write_text(json.dumps(rep, indent=2, default=str))
    acc = [f for f in rep["features"] if f["verdict"] == "ORTHOGONAL_INFORMATIVE"]
    pd.DataFrame(acc).to_csv(out / "accepted_conditioned_features.csv", index=False)
    _print(rep, acc)
    print(f"\nartifacts -> {out}/ (conditioned_feature_gate_report.csv/.json, "
          f"accepted_conditioned_features.csv)")


def _print(rep, acc):
    print(f"\n=== earnings/event-conditioned meta-features (does conditioning add beyond the parts?) ===")
    print(f"accepted-set combined OOS IC (incl. standalone parts): {rep['base_combined_oos_ic']}\n")
    for r in sorted(rep["features"], key=lambda x: -(x.get("combined_oos_ic_lift") or -9)):
        print(f"  {r['feature']:<34} IC {r['train_ic']:+.3f} (oos {r['oos_ic']:+.3f}) "
              f"FDR {str(r['fdr_significant']):<5} rankcontrib {r['rank_contribution']:.2f} "
              f"lift {r.get('combined_oos_ic_lift')} turn {r['turnover']:.2f} -> {r['verdict']}")
    verdict = ("CONDITIONING ADDS BEYOND THE PARTS: " + ", ".join(f["feature"] for f in acc)) if acc else \
              ("conditioning adds NOTHING beyond the standalone components -- the earnings effect is a "
               "standalone feature, not a useful modulator of existing signals (on this universe/horizon)")
    print(f"\n=== VERDICT: {verdict} ===")


if __name__ == "__main__":
    main()
