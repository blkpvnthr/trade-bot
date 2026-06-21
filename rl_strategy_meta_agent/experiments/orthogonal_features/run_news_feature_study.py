#!/usr/bin/env python3
"""Wire news / catalyst / FinBERT features through the orthogonal-feature gate.

The real question: does EXTERNAL information (news) add orthogonal, OOS-useful
signal beyond the volatility/giveback cluster -- or does it fail just like
price-only symbolic search? Reuses feature_gate.study (PIT IC + block-bootstrap +
BH FDR + vol-cluster orthogonality + combined-OOS-IC lift net of turnover) so the
bar is identical to the registry that admitted today's alphas.

    python experiments/orthogonal_features/run_news_feature_study.py
        [--backfill]   # (re)fetch news before the study

Research only: no broker/order/live path.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RL_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import feature_gate as G
import news_features as NF

# analyst_revision_flag is intentionally NOT built: AV EARNINGS_ESTIMATES is
# snapshot-only (not point-in-time backtestable), so it cannot be tested leak-free.
NOT_PIT = ["analyst_revision_flag"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true", help="(re)fetch news first")
    ap.add_argument("--window-h", type=float, default=24.0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out"))
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
        t0 = pd.Timestamp(panel.index.get_level_values(1).min()) - pd.Timedelta(days=3)
        t1 = pd.Timestamp(panel.index.get_level_values(1).max()) + pd.Timedelta(days=1)
        NF.backfill(syms, start=t0.to_pydatetime().replace(tzinfo=timezone.utc),
                    end=t1.to_pydatetime().replace(tzinfo=timezone.utc), force=True)

    news = NF.load_news(syms)
    earn = NF.load_earnings(syms)
    n_articles = sum(len(v) for v in news.values())
    n_with_news = sum(len(v) > 0 for v in news.values())
    cand = NF.align(panel.index, news, earn, {"window_h": args.window_h})
    # drop all-constant columns (e.g. no earnings data) -- the gate can't score them
    nonconst = [c for c in cand.columns if cand[c].std() > 1e-9]
    dropped = [c for c in cand.columns if c not in nonconst]
    cand = cand[nonconst]
    print(f"panel {panel.shape} | {n_articles} articles across {n_with_news}/{len(syms)} symbols | "
          f"{cand.shape[1]} candidate features | accepted set {len(nodes) if nodes else 0}")
    if dropped:
        print(f"  (dropped constant features: {dropped})")

    rep = G.study(panel, cand, accepted_nodes=nodes)
    rep["data"] = {"articles": n_articles, "symbols_with_news": n_with_news,
                   "window_h": args.window_h, "not_pit_excluded": NOT_PIT,
                   "constant_dropped": dropped}

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    feats = rep["features"]
    df = pd.DataFrame(feats)
    df.to_csv(out / "news_feature_gate_report.csv", index=False)
    (out / "news_feature_gate_report.json").write_text(json.dumps(rep, indent=2, default=str))
    acc = [f for f in feats if f["verdict"] == "ORTHOGONAL_INFORMATIVE"]
    rej = [f for f in feats if f["verdict"] != "ORTHOGONAL_INFORMATIVE"]
    pd.DataFrame(acc).to_csv(out / "accepted_external_features.csv", index=False)
    pd.DataFrame(rej).to_csv(out / "rejected_external_features.csv", index=False)
    _print(rep, acc)
    print(f"\nartifacts -> {out}/ (news_feature_gate_report.csv/.json, "
          f"accepted_external_features.csv, rejected_external_features.csv)")


def _print(rep, acc):
    print(f"\n=== news/catalyst/FinBERT through the orthogonal-feature gate ===")
    print(f"base effective breadth {rep['base_effective_breadth']} | "
          f"base combined OOS IC {rep['base_combined_oos_ic']}\n")
    for r in sorted(rep["features"], key=lambda x: -(x.get("combined_oos_ic_lift") or -9)):
        print(f"  {r['feature']:<22} IC {r['train_ic']:+.3f} (oos {r['oos_ic']:+.3f}) "
              f"FDR {str(r['fdr_significant']):<5} volcorr {r['vol_corr']:.2f} "
              f"rankcontrib {r['rank_contribution']:.2f} lift {r.get('combined_oos_ic_lift')} "
              f"turn {r['turnover']:.2f} -> {r['verdict']}")
    verdict = ("EXTERNAL INFORMATION ADDS ORTHOGONAL EDGE: "
               + ", ".join(f["feature"] for f in acc)) if acc else \
              ("NO orthogonal external edge -- news features are redundant/uninformative for "
               "forward return, same failure as price-only search. The bottleneck is not the "
               "feature family tried; needs a source with genuine cross-sectional forward signal.")
    print(f"\n=== VERDICT: {verdict} ===")


if __name__ == "__main__":
    main()
