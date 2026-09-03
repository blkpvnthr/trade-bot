#!/usr/bin/env python3
"""Label every trade with leak-free chart/candle patterns and report their value.

For each trade we read only the candles with timestamp <= entry_time, run the
detectors in rl_meta.patterns, and attach the pattern columns to the trade-
outcome dataset. No post-entry bar and no outcome/exit field is used to assign a
pattern. The report measures win rate / profit factor / return by pattern and,
critically, whether the pattern labels add predictive signal beyond the raw
indicator features -- for entry prediction and for exit-quality prediction.

Research-only: no allocator, broker, order, or risk-gate path is touched.

    python scripts/label_trade_patterns.py --dataset data/trade_outcomes/trade_outcome_dataset.parquet \
        --bars-dir data/bars --timeframe 15Min --out data/trade_outcomes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta import patterns as P
from rl_meta.market_context import build_context_frame
from rl_meta.bar_coverage import cached_bars
from rl_meta.trade_features import PRE_ENTRY_FEATURES
from rl_meta.exit_quality import RUN_FEATURES
from rl_meta.trade_quality import make_models, classification_metrics, proba

NUM_PATTERN_COLS = P.FLAG_COLS          # numeric columns usable as model features


def label(df, bars_dir, timeframe):
    out = {}
    syms = df["symbol"].nunique()
    for i, (sym, g) in enumerate(df.groupby("symbol"), 1):
        bars = cached_bars(sym, timeframe, bars_dir)
        if bars.empty:
            for idx in g.index:
                out[idx] = P.unknown_row()
            continue
        bars = bars.sort_values("timestamp").reset_index(drop=True)
        ts_ns = pd.to_datetime(bars["timestamp"], utc=True).astype("int64").to_numpy()
        ctx = build_context_frame(bars)            # refs not needed for price-action
        for t in g[["entry_time"]].itertuples():
            e = int(np.searchsorted(ts_ns, t.entry_time.value, "right")) - 1  # last bar <= entry
            out[t.Index] = P.detect_patterns(bars, e, ctx)
        if i % 10 == 0 or i == syms:
            print(f"  labeled {i}/{syms} symbols")
    return pd.DataFrame.from_dict(out, orient="index")


def _group_auc(X, y, groups):
    from sklearn.model_selection import GroupShuffleSplit
    gi, ge = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(X, y, groups))
    best = -1
    for _, m in make_models().items():
        m.fit(X.iloc[gi], y.iloc[gi])
        best = max(best, classification_metrics(y.iloc[ge], proba(m, X.iloc[ge]))["auc"])
    return round(best, 3)


def _profit_factor(ret):
    up = ret[ret > 0].sum()
    dn = -ret[ret < 0].sum()
    return round(float(up / dn), 3) if dn > 0 else float("inf")


def _perf_by(df, by, min_n=50):
    out = {}
    for key, sub in df.groupby(by):
        if len(sub) < min_n:
            continue
        out[str(key)] = {
            "trades": int(len(sub)),
            "win_rate": round(float(sub["outcome_win"].astype(float).mean()), 3),
            "profit_factor": _profit_factor(sub["return_pct"]),
            "avg_return_pct": round(float(sub["return_pct"].mean()), 4)}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["avg_return_pct"]))


def report(aug, pre_cols, bars_dir):
    rep = {}
    labeled = aug["primary_pattern"].notna() & (aug["primary_pattern"] != "unknown")
    rep["rows_total"] = int(len(aug))
    rep["rows_labeled"] = int(labeled.sum())
    rep["rows_unknown"] = int((aug["primary_pattern"] == "unknown").sum())
    rep["rows_no_detectable_pattern"] = int((aug["primary_pattern"] == "none").sum())

    lab = aug[labeled].copy()
    lab["outcome_win"] = lab["outcome_win"].astype(float)
    rep["primary_pattern_frequency"] = {k: int(v) for k, v in
                                        lab["primary_pattern"].value_counts().items()}
    rep["flag_frequency"] = {c: int(aug[c].fillna(0).gt(0).sum())
                             for c in NUM_PATTERN_COLS if aug[c].dropna().isin([0, 1]).all()}
    rep["performance_by_pattern"] = _perf_by(lab, "primary_pattern")

    oos = lab[lab["split"] == "oos"]
    oos_perf = _perf_by(oos, "primary_pattern", min_n=30) if len(oos) else {}
    rep["oos_performance_by_pattern"] = oos_perf
    items = list(oos_perf.items())
    rep["best_oos_patterns"] = [k for k, _ in items[:3]]
    rep["worst_oos_patterns"] = [k for k, _ in items[-3:]]

    rep["performance_by_strategy_pattern"] = {
        s: _perf_by(sub, "primary_pattern", min_n=40)
        for s, sub in lab.groupby("strategy_id")}
    top_syms = lab["symbol"].value_counts().head(8).index
    rep["performance_by_symbol"] = {
        s: _perf_by(lab[lab["symbol"] == s], "primary_pattern", min_n=40) for s in top_syms}

    # --- does the pattern label add signal beyond raw indicators? ---
    m = aug["has_pre_features"] & aug["outcome_win"].notna() & labeled
    sub = aug[m]
    y = sub["outcome_win"].astype(int)
    g = sub["symbol"]
    ent = sub[pre_cols]
    pat_feat = [c for c in NUM_PATTERN_COLS if c not in pre_cols]   # avoid dup columns
    rich = pd.concat([ent, sub[pat_feat]], axis=1)
    rep["entry_prediction"] = {"entry_only_auc": _group_auc(ent, y, g),
                               "entry_plus_patterns_auc": _group_auc(rich, y, g),
                               "new_pattern_features": len(pat_feat)}

    rep["exit_quality_prediction"] = _exit_improvement(aug, bars_dir)
    return rep


def _exit_improvement(aug, bars_dir):
    v2 = Path(bars_dir).parent / "trade_outcomes" / "decision_points_v2.parquet"
    if not v2.exists():
        return {"note": "run scripts/trade_management_research.py first (decision_points_v2 missing)"}
    dec = pd.read_parquet(v2)
    key = aug["symbol"].astype(str) + "|" + pd.to_datetime(aug["entry_time"], utc=True).map(
        lambda x: x.isoformat())
    ren = {c: f"pat_{c}" for c in NUM_PATTERN_COLS}              # prefix to avoid collisions
    pat = aug[NUM_PATTERN_COLS].rename(columns=ren).copy()
    pat["trade_id"] = key.values
    dd = dec.merge(pat.drop_duplicates("trade_id"), on="trade_id", how="left")
    base = [c for c in PRE_ENTRY_FEATURES if c in dd.columns] + list(RUN_FEATURES)
    pat_feat = list(ren.values())
    res = {}
    for tgt in ["severe_giveback", "early_exit_good"]:
        if tgt not in dd or dd[tgt].nunique() < 2:
            continue
        y = dd[tgt].astype(int)
        res[tgt] = {"pre+run_auc": _group_auc(dd[base], y, dd["symbol"]),
                    "pre+run+patterns_auc": _group_auc(dd[base + pat_feat], y, dd["symbol"])}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/trade_outcomes/trade_outcome_dataset.parquet")
    ap.add_argument("--bars-dir", default="data/bars")
    ap.add_argument("--timeframe", default="15Min")
    ap.add_argument("--out", default="data/trade_outcomes")
    ap.add_argument("--sample", type=int, default=0, help="0 = all rows")
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    if args.sample and len(df) > args.sample:
        df = df.sample(args.sample, random_state=0)
    pre_cols = [c for c in PRE_ENTRY_FEATURES if c in df.columns]
    print(f"Labeling {len(df):,} trades across {df['symbol'].nunique()} symbols...")
    pat = label(df, Path(args.bars_dir), args.timeframe)

    aug = df.drop(columns=[c for c in P.ALL_COLS if c in df.columns]).join(pat)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    aug.to_parquet(out / "trade_outcome_dataset_patterns.parquet", index=False)
    aug.to_csv(out / "trade_outcome_dataset_patterns.csv", index=False)

    print("\nPrimary pattern distribution:")
    for p, n in aug["primary_pattern"].value_counts().items():
        print(f"  {p:<26}{n:>8,}")

    rep = report(aug, pre_cols, Path(args.bars_dir))
    (out / "pattern_report.json").write_text(json.dumps(rep, indent=2, default=str))
    _print_verdict(rep)
    print(f"\nartifacts -> {out}/  (parquet, csv, pattern_report.json)")


def _print_verdict(rep):
    print(f"\nrows: {rep['rows_total']:,} | labeled {rep['rows_labeled']:,} | "
          f"unknown {rep['rows_unknown']:,} | no-pattern {rep['rows_no_detectable_pattern']:,}")
    print("\nWin rate / profit factor / avg return by primary pattern:")
    for k, v in rep["performance_by_pattern"].items():
        print(f"  {k:<24} win {v['win_rate']:.2f}  PF {v['profit_factor']:<5}  "
              f"ret {v['avg_return_pct']*100:+.2f}%  (n={v['trades']:,})")
    ep = rep["entry_prediction"]
    print(f"\nEntry prediction: entry-only {ep['entry_only_auc']}  ->  "
          f"+patterns {ep['entry_plus_patterns_auc']}")
    xq = rep["exit_quality_prediction"]
    if isinstance(xq, dict) and "note" not in xq:
        for tgt, d in xq.items():
            print(f"Exit ({tgt}): pre+run {d['pre+run_auc']}  ->  "
                  f"+patterns {d['pre+run+patterns_auc']}")
    else:
        print(f"Exit quality: {xq.get('note')}")


if __name__ == "__main__":
    main()
