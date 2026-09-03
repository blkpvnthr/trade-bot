#!/usr/bin/env python3
"""Options / IV-structure features through the orthogonal-feature gate.

Raised bar: the accepted space is now the VALIDATED LEADS --
[volatility_factor, idio_reversal, earnings_within_7d] -- plus the 8 accepted alphas.
The question is no longer "is this different from volatility?" but "does this add
information beyond every validated lead we already have?"

Candidates (tiered):
  Tier 1: iv_rank, iv_percentile, term_slope (IV30-IV7), pc_oi_ratio, oi_change_1d/5d
  Tier 2: skew_25d (25d put IV - call IV)
  Composite: iv_rank_x_earnings7 -- elevated IV AND earnings imminent, tested as a
             STANDALONE candidate (info coming + market expects it to matter).

    python experiments/orthogonal_features/run_options_study.py [--backfill]

Research only: read-only options data, no broker/order/live path.
"""
from __future__ import annotations

import argparse
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
import options_features as OF
from run_feature_study import build_candidates

TIER1 = ["iv_rank", "iv_percentile", "term_slope", "pc_oi_ratio", "oi_change_1d", "oi_change_5d"]
TIER2 = ["skew_25d"]
OPT_COLS = TIER1 + TIER2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--label", default="fut_ret_1", help="forward-return horizon (fut_ret_1=15m, fut_ret_4=1h)")
    args = ap.parse_args()

    panel = pd.read_parquet(RL_ROOT / "data/trade_outcomes/_alpha_panel_25_8_v2.parquet")
    syms = sorted(panel.index.get_level_values(0).unique())
    nodes_path = RL_ROOT / "data/trade_outcomes/accepted_alpha_nodes.json"
    nodes = None
    if nodes_path.exists():
        def to_node(x):
            return tuple(to_node(e) for e in x) if isinstance(x, list) else x
        nodes = [to_node(d["node"]) for d in json.loads(nodes_path.read_text())]

    if args.backfill:
        OF.backfill(syms, "2025-07-15", "2026-06-18", workers=8, force=True)

    feats = OF.load_features(syms)
    n_sym = sum(not f.empty for f in feats.values())
    n_days = sum(len(f) for f in feats.values())
    opt = OF.align(panel.index, feats, OPT_COLS)

    # validated leads form the accepted space (the raised bar)
    px = build_candidates(panel)
    earn = NF.align(panel.index, NF.load_news(syms), NF.load_earnings(syms), {"window_h": 24.0})
    extra = pd.DataFrame({"vol_factor": panel["atr_pct"], "idio_reversal": px["idio_reversal"],
                          "earnings_within_7d": earn["earnings_within_7d"]}, index=panel.index)

    # composite candidate: elevated IV AND earnings imminent (standalone feature)
    opt["iv_rank_x_earnings7"] = (G._cs_z(opt["iv_rank"]) * earn["earnings_within_7d"]).fillna(0.0)
    cand = opt[[c for c in opt.columns if opt[c].std() > 1e-9]]
    dropped = [c for c in opt.columns if c not in cand.columns]
    print(f"panel {panel.shape} | options for {n_sym}/{len(syms)} symbols, {n_days} symbol-days | "
          f"{cand.shape[1]} candidates | accepted = {len(nodes) if nodes else 0} alphas + 3 validated leads")
    if dropped:
        print(f"  (dropped constant: {dropped})")

    rep = G.study(panel, cand, accepted_nodes=nodes, accepted_extra=extra, cfg={"label": args.label})
    rep["data"] = {"label": args.label, "symbols_with_options": n_sym, "symbol_days": n_days,
                   "accepted_leads": list(extra.columns), "tier1": TIER1, "tier2": TIER2}

    out = Path(__file__).resolve().parent / "out"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rep["features"]).to_csv(out / "options_feature_gate_report.csv", index=False)
    (out / "options_feature_gate_report.json").write_text(json.dumps(rep, indent=2, default=str))
    acc = [f for f in rep["features"] if f["verdict"] == "ORTHOGONAL_INFORMATIVE"]
    rej = [f for f in rep["features"] if f["verdict"] != "ORTHOGONAL_INFORMATIVE"]
    pd.DataFrame(acc).to_csv(out / "accepted_options_features.csv", index=False)
    pd.DataFrame(rej).to_csv(out / "rejected_options_features.csv", index=False)
    _print(rep, acc)
    print(f"\nartifacts -> {out}/ (options_feature_gate_report.csv/.json, "
          f"accepted_options_features.csv, rejected_options_features.csv)")


def _print(rep, acc):
    print(f"\n=== options/IV through the gate (bar = beats every validated lead) ===")
    print(f"accepted-set (8 alphas + vol + idio_reversal + earnings_within_7d) "
          f"combined OOS IC: {rep['base_combined_oos_ic']}\n")
    for r in sorted(rep["features"], key=lambda x: -(x.get("combined_oos_ic_lift") or -9)):
        print(f"  {r['feature']:<22} IC {r['train_ic']:+.3f} (oos {r['oos_ic']:+.3f}) "
              f"FDR {str(r['fdr_significant']):<5} volcorr {r['vol_corr']:.2f} "
              f"rankcontrib {r['rank_contribution']:.2f} lift {r.get('combined_oos_ic_lift')} "
              f"turn {r['turnover']:.2f} -> {r['verdict']}")
    verdict = ("OPTIONS/IV ADDS ORTHOGONAL EDGE beyond every validated lead: "
               + ", ".join(f["feature"] for f in acc)) if acc else \
              ("NO options/IV feature beats the validated-lead set -- options positioning is "
               "redundant with what we already have on this universe/horizon")
    print(f"\n=== VERDICT: {verdict} ===")


if __name__ == "__main__":
    main()
