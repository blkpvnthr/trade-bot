#!/usr/bin/env python3
"""Build daily return streams for the new formulaic-alpha stack -> strategy_streams.parquet.

Produces deployable-candidate streams the OLPS allocator can route capital across:
  alpha_mined_ensemble   (registry-accepted alpha composite, long-short)
  giveback_exit_model    (long book, exposure cut by predicted giveback)
  ORB_relvol             (opening-range breakout + relative volume)
  linear_regression_baseline, mean_reversion_baseline   (validated families)

Retired strategies (martingale/pin_bar/sniper) are never produced. Requires the
cached alpha panel and accepted_alpha_nodes.json (run scripts/alpha_mining.py first).
Research-only.

    python scripts/build_strategy_streams.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta import strategy_streams as ss
from rl_meta.alphas import eval_alpha


def _to_node(x):
    return tuple(_to_node(e) for e in x) if isinstance(x, list) else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/trade_outcomes/trade_outcome_dataset.parquet")
    ap.add_argument("--panel", default="data/trade_outcomes/_alpha_panel_25_8_v2.parquet")
    ap.add_argument("--nodes", default="data/trade_outcomes/accepted_alpha_nodes.json")
    ap.add_argument("--out", default="data/trade_outcomes")
    args = ap.parse_args()
    out = Path(args.out)

    panel = pd.read_parquet(args.panel)
    nodes = [_to_node(d["node"]) for d in json.loads(Path(args.nodes).read_text())]
    print(f"panel {panel.shape} | accepted alpha nodes {len(nodes)}")

    node_values = [eval_alpha(n, panel) for n in nodes]
    giveback = ss.composite_signal(panel, node_values)

    streams = {
        "alpha_mined_ensemble": ss.long_short_stream(panel, -giveback),   # long low-risk
        "giveback_exit_model": ss.exposure_stream(panel, giveback),
        "ORB_relvol": ss.orb_relvol_stream(panel),
        "linear_regression_baseline": ss.baseline_stream(args.dataset, "linear_regression"),
        "mean_reversion_baseline": ss.baseline_stream(args.dataset, "mean_reversion"),
    }
    ss.validate_no_retired(streams.keys())                  # hard guard
    streams = {k: v for k, v in streams.items() if v is not None and len(v) > 20}

    # inner-join to the common date window so the allocator compares apples-to-apples
    df = pd.DataFrame(streams).dropna().sort_index()
    df.index.name = "date"
    df.to_parquet(out / "strategy_streams.parquet")
    print("\nstream summary (daily return):")
    for c in df.columns:
        s = df[c].dropna()
        print(f"  {c:<28} n={len(s):>4} mean {s.mean()*100:+.3f}% std {s.std()*100:.3f}% "
              f"cum {((1+s.fillna(0)).prod()-1)*100:+.1f}%")
    print(f"\nstreams -> {out / 'strategy_streams.parquet'}  ({df.shape[1]} assets x {df.shape[0]} days)")


if __name__ == "__main__":
    main()
