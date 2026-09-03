#!/usr/bin/env python3
"""Pilot run of the orthogonal-feature gate on the cached alpha panel.

First scan: candidate features computed from EXISTING price/volume data, chosen to
probe axes the current volatility-dominant panel underweights (idiosyncratic
reversal, cross-sectional relative strength, downside asymmetry, return skew), plus
a volatility control that SHOULD verdict REDUNDANT -- validating the gate on real
data. The honest expectation is that price-derived features stay redundant; the gate
exists to tell us which genuinely-new sources (news/catalyst, options flow,
execution/fill-quality) earn integration. Research only.

    python experiments/orthogonal_features/run_feature_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import feature_gate as G


def build_candidates(panel: pd.DataFrame) -> pd.DataFrame:
    """Leak-free per-name candidate features (vary cross-sectionally) on non-vol axes.
    All use backward/contemporaneous data only; the label fut_ret_1 is the only future."""
    bysym = lambda c: panel[c].groupby(level=0)
    cs = lambda s: s.groupby(s.index.get_level_values(1))
    r = panel["ret"]
    c = {}
    # market-residual (idiosyncratic) reversal: name return minus cross-sectional mean
    c["idio_reversal"] = -(r - cs(r).transform("mean"))
    # cross-sectional relative strength over 5 bars (momentum vs peers)
    r5 = bysym("close").transform(lambda s: s / s.shift(5) - 1.0)
    c["rel_strength_5"] = r5 - cs(r5).transform("median")
    # downside asymmetry: rolling downside vol / total vol (different from atr level)
    neg = panel["ret"].where(panel["ret"] < 0, 0.0)
    dn = neg.groupby(level=0).transform(lambda s: s.rolling(20).std())
    tot = bysym("ret").transform(lambda s: s.rolling(20).std())
    c["downside_ratio"] = dn / (tot + 1e-9)
    # rolling return skew (higher moment, plausibly orthogonal to vol level)
    c["ret_skew_20"] = bysym("ret").transform(lambda s: s.rolling(20).skew())
    # CONTROL: Parkinson range vol -- expected REDUNDANT_VOLATILITY
    c["parkinson_vol"] = bysym("high").transform(
        lambda s: (np.log(s / panel.loc[s.index, "low"]) ** 2).rolling(20).mean())
    return pd.DataFrame(c, index=panel.index)


def main():
    panel_path = ROOT / "data/trade_outcomes/_alpha_panel_25_8_v2.parquet"
    nodes_path = ROOT / "data/trade_outcomes/accepted_alpha_nodes.json"
    out = Path(__file__).resolve().parent / "out"
    out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(panel_path)
    cand = build_candidates(panel)
    nodes = None
    if nodes_path.exists():
        def to_node(x):
            return tuple(to_node(e) for e in x) if isinstance(x, list) else x
        nodes = [to_node(d["node"]) for d in json.loads(nodes_path.read_text())]
    print(f"panel {panel.shape} | {cand.shape[1]} candidate features | "
          f"accepted set = {len(nodes) if nodes else 0} alphas")

    rep = G.study(panel, cand, accepted_nodes=nodes)
    (out / "feature_study.json").write_text(json.dumps(rep, indent=2, default=str))
    _report_md(out / "feature_study.md", rep)
    _print(rep)
    print(f"\nartifacts -> {out}/ (feature_study.json, feature_study.md)")


def _report_md(path, rep):
    L = ["# Orthogonal-feature gate -- pilot scan", "",
         f"- target label: `{rep['label']}` (directional forward return)",
         f"- accepted-set effective breadth: {rep['base_effective_breadth']} | "
         f"combined OOS IC: {rep['base_combined_oos_ic']}", "",
         "| feature | train IC | OOS IC | p | FDR | vol corr | rank contrib | "
         "breadth lift | comb OOS IC lift | turnover | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["features"]:
        L.append(f"| {r['feature']} | {r['train_ic']} | {r['oos_ic']} | {r['p_value']} | "
                 f"{r['fdr_significant']} | {r['vol_corr']} | {r['rank_contribution']} | "
                 f"{r.get('breadth_lift')} | {r.get('combined_oos_ic_lift')} | {r['turnover']} | "
                 f"**{r['verdict']}** |")
    L += ["", f"**{rep['summary']}**", ""]
    path.write_text("\n".join(L))


def _print(rep):
    print(f"\n=== orthogonal-feature gate (label {rep['label']}) ===")
    print(f"base effective breadth {rep['base_effective_breadth']} | "
          f"base combined OOS IC {rep['base_combined_oos_ic']}\n")
    for r in rep["features"]:
        print(f"  {r['feature']:<16} IC {r['train_ic']:+.3f} (oos {r['oos_ic']:+.3f}) "
              f"volcorr {r['vol_corr']:.2f} rankcontrib {r['rank_contribution']:.2f} "
              f"lift {r.get('combined_oos_ic_lift')} -> {r['verdict']}")
    print(f"\n=== {rep['summary']} ===")


if __name__ == "__main__":
    main()
