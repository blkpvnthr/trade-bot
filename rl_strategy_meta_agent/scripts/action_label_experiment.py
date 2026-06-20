#!/usr/bin/env python3
"""Counterfactual action labels vs P(severe_giveback): is the signal ACTIONABLE?

The giveback model answers "will a severe giveback occur?" -- but the portfolio
cares about "which ACTION now maximizes trade expectancy?". These are different
learning problems. This experiment:

  1 DIAGNOSTIC (the key question): among decision bars where severe_giveback==1,
    how many recovered and made a NEW HIGH above the decision-time peak before
    the strategy's exit? And how often was HOLDING actually the better action?
    If that share is large, it explains the whole earlier result (good AUC, lower
    drawdown, falling total return, walk-forward collapse): the model is tagging
    recoverable pullbacks/volatility, not terminal deterioration.

  2 ACTION LABELS: train directly on exit_now_better / tighten_stop_better /
    take_partial_better (counterfactual: did the action beat holding to exit?)
    and compare their held-out AUC to the giveback model on the SAME rows.

Research-only: no broker/order/risk path.

    python scripts/action_label_experiment.py --sample-trades 8000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta.exit_quality import (running_features, decision_targets,
                                  sample_decision_ks, RUN_FEATURES)
from rl_meta.market_context import CTX_NUMERIC
from rl_meta.management_policies import action_labels, giveback_recovery
from rl_meta.trade_features import PRE_ENTRY_FEATURES
from rl_meta.trade_quality import make_models, classification_metrics, proba
from giveback_policy_backtest import _load_refs, _frames, _path_and_idx

ACTION_LABELS = ["exit_now_better", "tighten_stop_better", "take_partial_better"]


def build(df, pre_cols, bars_dir, refs):
    feat = pre_cols + list(RUN_FEATURES) + list(CTX_NUMERIC)
    rows, labels, sym = [], [], []
    cols = pre_cols + ["entry_time", "exit_time", "entry_price", "direction"]
    for s, g in df.groupby("symbol"):
        from rl_meta.bar_coverage import cached_bars
        bars = cached_bars(s, "15Min", bars_dir)
        if bars.empty:
            continue
        bars, ts_ns, ctx = _frames(bars, refs, s)
        close = bars["close"].to_numpy(float)
        for t in g[cols].itertuples(index=False):
            lo, r = _path_and_idx(t, ts_ns, close)
            if r is None:
                continue
            pre = [getattr(t, c) for c in pre_cols]
            for k in sample_decision_ks(len(r) - 1):
                run = running_features(r, k)
                crow = ctx.iloc[lo + k]
                rows.append(pre + [run[f] for f in RUN_FEATURES] +
                            [crow[c] for c in CTX_NUMERIC])
                lab = {**action_labels(r, k), **giveback_recovery(r, k)}
                lab["severe_giveback"] = int(decision_targets(r, k)["severe_giveback"])
                labels.append(lab); sym.append(s)
    return (pd.DataFrame(rows, columns=feat), pd.DataFrame(labels),
            np.asarray(sym))


def _group_auc(X, y, groups):
    from sklearn.model_selection import GroupShuffleSplit
    if y.nunique() < 2:
        return None
    gi, ge = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(X, y, groups))
    best = -1
    for _, m in make_models().items():
        m.fit(X.iloc[gi], y.iloc[gi])
        best = max(best, classification_metrics(y.iloc[ge], proba(m, X.iloc[ge]))["auc"])
    return round(best, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/trade_outcomes/trade_outcome_dataset.parquet")
    ap.add_argument("--bars-dir", default="data/bars")
    ap.add_argument("--sample-trades", type=int, default=8000)
    ap.add_argument("--out", default="data/trade_outcomes")
    args = ap.parse_args()
    bars_dir = Path(args.bars_dir)

    df = pd.read_parquet(args.dataset)
    df = df[df["has_pre_features"]].copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    pre_cols = [c for c in PRE_ENTRY_FEATURES if c in df.columns]
    if args.sample_trades and len(df) > args.sample_trades:
        df = df.sample(args.sample_trades, random_state=0)
    start = df["entry_time"].min().date().isoformat()
    end = (df["exit_time"].max() + pd.Timedelta(days=1)).date().isoformat()
    refs = _load_refs(start, end, bars_dir)

    print("building decision rows + action labels...")
    X, lab, sym = build(df, pre_cols, bars_dir, refs)
    print(f"decision bars: {len(X):,}\n")

    # 1) DIAGNOSTIC ----------------------------------------------------------
    sg = lab["severe_giveback"] == 1
    n_sg = int(sg.sum())
    diag = {
        "decision_bars": int(len(lab)),
        "severe_giveback_bars": n_sg,
        "severe_giveback_rate": round(float(sg.mean()), 3),
        "among_severe_giveback": {
            "made_new_high_after_pct": round(float(lab.loc[sg, "new_high_after"].mean()), 3),
            "exit_now_was_better_pct": round(float(lab.loc[sg, "exit_now_better"].mean()), 3),
            "holding_was_better_pct": round(float(1 - lab.loc[sg, "exit_now_better"].mean()), 3),
        },
        "base_rates": {c: round(float(lab[c].mean()), 3)
                       for c in ACTION_LABELS + ["new_high_after", "severe_giveback"]},
    }

    # 2) ACTION-LABEL models vs giveback, same rows, group-by-symbol ---------
    aucs = {}
    for tgt in ACTION_LABELS + ["severe_giveback"]:
        aucs[tgt] = _group_auc(X, lab[tgt].astype(int), sym)

    rep = {"diagnostic": diag, "held_out_auc": aucs}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "action_label_report.json").write_text(json.dumps(rep, indent=2, default=str))
    _print(rep)
    print(f"\nreport -> {out / 'action_label_report.json'}")


def _print(rep):
    d = rep["diagnostic"]; a = d["among_severe_giveback"]
    print("=== DIAGNOSTIC: are severe-giveback bars actually 'exit now' bars? ===")
    print(f"decision bars {d['decision_bars']:,} | severe_giveback bars "
          f"{d['severe_giveback_bars']:,} ({d['severe_giveback_rate']:.2f})")
    print(f"  among severe_giveback==1:")
    print(f"    made a NEW HIGH after the decision : {a['made_new_high_after_pct']*100:.1f}%")
    print(f"    exiting now WAS better             : {a['exit_now_was_better_pct']*100:.1f}%")
    print(f"    HOLDING was better (exit was wrong): {a['holding_was_better_pct']*100:.1f}%")
    print("\n=== held-out AUC: giveback vs ACTION labels (same rows) ===")
    for tgt, v in rep["held_out_auc"].items():
        tag = "  <- prediction target" if tgt == "severe_giveback" else "  <- ACTION target"
        print(f"  {tgt:<22} AUC {v}{tag}")
    sg = rep["held_out_auc"]["severe_giveback"]
    exn = rep["held_out_auc"]["exit_now_better"]
    print("\n=== Read ===")
    if a["holding_was_better_pct"] >= 0.4:
        print(f"-> {a['holding_was_better_pct']*100:.0f}% of severe-giveback bars were bars where "
              "HOLDING beat exiting: the giveback flag tags recoverable pullbacks/volatility, "
              "NOT terminal deterioration. That is why acting on it bleeds return.")
    if exn is not None and sg is not None:
        gap = sg - exn
        print(f"-> giveback AUC {sg} vs exit_now_better AUC {exn} "
              f"(gap {gap:+.3f}): the ACTIONABLE label is "
              f"{'harder to predict' if gap > 0.03 else 'about as predictable'} than the "
              "giveback event -- predictive signal != actionable signal.")


if __name__ == "__main__":
    main()
