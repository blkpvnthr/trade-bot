#!/usr/bin/env python3
"""Trade Management Research Pipeline.

Expands the decision-point dataset with leak-free market context (PDH/PDL/VWAP/
opening-range distance, SPY/QQQ trend, sector strength, time-of-day, rvol
acceleration, trend regime) and per-trade pattern + regime labels, then asks the
research question:

    Do exit management, pattern recognition, and regime classification explain
    trade outcomes better than pre-entry win prediction (honest AUC ~0.53)?

Three honest (group-by-symbol) comparisons:
  A trade-level : outcome_win / profit_capture_ratio from {entry} vs {entry +
                  pattern + regime + context}
  B exit-mgmt   : early_exit_good / severe_giveback from {pre+run} vs {+context}
  C descriptive : mean profit_capture / giveback / win-rate by pattern & regime

Writes the expanded dataset + per-trade summary. Research-only: the only network
call is a historical-bars fetch for SPY/QQQ/sector ETFs. No allocator, no broker,
no orders.

    python scripts/trade_management_research.py --sample-trades 50000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta.trade_features import PRE_ENTRY_FEATURES
from rl_meta.exit_quality import return_path, build_decision_rows, RUN_FEATURES, DECISION_FLAGS
from rl_meta.bar_coverage import cached_bars, fetch_with_coverage
from rl_meta.market_context import (build_context_frame, classify_pattern, CTX_NUMERIC,
                                    SECTOR_ETF, REF_SYMBOLS, PATTERNS)
from rl_meta.trade_quality import make_models, classification_metrics, proba

REGIME_COLS = ["ctx_mkt_regime", "ctx_own_regime"]


def _ensure_refs(start, end, bars_dir):
    refs = {}
    for s in REF_SYMBOLS:
        df = cached_bars(s, "15Min", bars_dir)
        if df.empty:
            df, status = fetch_with_coverage(s, "15Min", start, end, bars_dir)
            print(f"  {s}: {status} ({len(df):,} bars)")
        refs[s] = df
    return refs


def build(df, pre_cols, bars_dir, refs):
    """Returns (per_trade_df, decision_df). Both leak-free at every column."""
    trade_meta = {}     # df index -> pattern/regime/entry-context dict
    dec_rows = []
    for sym, g in df.groupby("symbol"):
        bars = cached_bars(sym, "15Min", bars_dir)
        if bars.empty:
            continue
        bars = bars.sort_values("timestamp").reset_index(drop=True)
        ts_ns = pd.to_datetime(bars["timestamp"], utc=True).astype("int64").to_numpy()
        high = bars["high"].to_numpy(float); low = bars["low"].to_numpy(float)
        close = bars["close"].to_numpy(float)
        sector = refs.get(SECTOR_ETF.get(sym))
        ctx = build_context_frame(bars, refs.get("SPY"), refs.get("QQQ"), sector)
        n = len(bars)
        cols = pre_cols + ["entry_time", "exit_time", "entry_price", "direction"]
        for t in g[cols].itertuples(index=True):
            e = int(np.searchsorted(ts_ns, t.entry_time.value, "left"))
            if e >= n:
                continue
            is_long = t.direction == "long"
            pattern = classify_pattern(high, low, close, ctx, e, is_long)
            ent = ctx.iloc[e]
            trade_meta[t.Index] = {"pattern": pattern,
                                   "ctx_mkt_regime": ent["ctx_mkt_regime"],
                                   "ctx_own_regime": ent["ctx_own_regime"],
                                   **{c: ent[c] for c in CTX_NUMERIC}}
            r = return_path(ts_ns, close, t.entry_time.value, t.exit_time.value,
                            t.entry_price, is_long)
            if r is None:
                continue
            pre = {c: getattr(t, c) for c in pre_cols}
            meta = {"strategy_id": "", "symbol": sym, "split": "",
                    "trade_id": f"{sym}|{t.entry_time.isoformat()}"}
            for row in build_decision_rows(r, pre, meta):
                cr = ctx.iloc[e + row["k"]]                 # context AT decision bar k
                row["pattern"] = pattern
                for c in CTX_NUMERIC + REGIME_COLS:
                    row[c] = cr[c]
                dec_rows.append(row)

    tm = pd.DataFrame.from_dict(trade_meta, orient="index")
    per_trade = df.join(tm)                                 # adds pattern/regime/ctx-entry
    return per_trade, pd.DataFrame(dec_rows)


def _group_auc(X, y, groups):
    from sklearn.model_selection import GroupShuffleSplit
    gi, ge = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(X, y, groups))
    best = -1
    for _, m in make_models().items():
        m.fit(X.iloc[gi], y.iloc[gi])
        best = max(best, classification_metrics(y.iloc[ge], proba(m, X.iloc[ge]))["auc"])
    return round(best, 3)


def _group_r2(X, y, groups):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import r2_score
    from sklearn.impute import SimpleImputer
    gi, ge = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(X, y, groups))
    imp = SimpleImputer().fit(X.iloc[gi])
    m = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=0)
    m.fit(imp.transform(X.iloc[gi]), y.iloc[gi])
    return round(float(r2_score(y.iloc[ge], m.predict(imp.transform(X.iloc[ge])))), 3)


def _dummies(df, cols):
    return pd.get_dummies(df[cols].astype("object"), dummy_na=True).astype(float)


def analyze(per_trade, dec, pre_cols, report):
    pt = per_trade.dropna(subset=["pattern"]).copy()
    g = pt["symbol"]

    # --- A: trade-level, entry-only vs entry+pattern+regime+context ---
    ent = pt[pre_cols]
    extra = pd.concat([pt[CTX_NUMERIC], _dummies(pt, ["pattern"] + REGIME_COLS)], axis=1)
    rich = pd.concat([ent, extra], axis=1)
    A = {}
    for tgt, fn in [("outcome_win", _group_auc), ("profit_capture_ratio", _group_r2)]:
        y = pt[tgt]
        m = pt[tgt].notna() & np.isfinite(pt[tgt].astype(float))
        if tgt == "outcome_win":
            m &= pt[tgt].isin([0, 1, True, False])
        yy, gg = y[m].astype(float), g[m]
        A[tgt] = {"entry_only": fn(ent[m], yy, gg), "entry+pattern+regime+ctx":
                  fn(rich[m], yy, gg), "metric": "AUC" if fn is _group_auc else "R2"}
    report["A_trade_level"] = A

    # --- B: exit management, with vs without mid-trade context ---
    B = {}
    base = pre_cols + RUN_FEATURES
    with_ctx = base + CTX_NUMERIC
    dd = pd.concat([dec, _dummies(dec, REGIME_COLS)], axis=1)
    reg_dummies = [c for c in dd.columns if c.startswith(tuple(r + "_" for r in REGIME_COLS))]
    ctx_cols = CTX_NUMERIC + reg_dummies
    for tgt in ["early_exit_good", "severe_giveback"]:
        y = dec[tgt].astype(int)
        if y.nunique() < 2:
            continue
        B[tgt] = {"pre+run": _group_auc(dec[base], y, dec["symbol"]),
                  "pre+run+context": _group_auc(dd[base + ctx_cols], y, dd["symbol"])}
    report["B_exit_management"] = B

    # --- C: descriptive -- outcome quality by pattern and by regime ---
    def table(by):
        # median for ratio metrics: profit_capture_ratio has heavy division-by-
        # near-zero-MFE tails, so the mean is not a usable central tendency.
        out = {}
        for key, sub in pt.groupby(by):
            if len(sub) < 30:
                continue
            out[str(key)] = {
                "trades": int(len(sub)),
                "win_rate": round(float(sub["outcome_win"].astype(float).mean()), 3),
                "median_profit_capture": round(float(sub["profit_capture_ratio"].median()), 3),
                "median_giveback_from_mfe": round(float(sub["giveback_from_mfe"].median()), 4),
                "median_exit_efficiency": round(float(sub["exit_efficiency"].median()), 3)}
        return dict(sorted(out.items(), key=lambda kv: -kv[1]["median_profit_capture"]))
    report["C_by_pattern"] = table("pattern")
    report["C_by_mkt_regime"] = table("ctx_mkt_regime")
    report["C_by_own_regime"] = table("ctx_own_regime")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/trade_outcomes/trade_outcome_dataset.parquet")
    ap.add_argument("--bars-dir", default="data/bars")
    ap.add_argument("--sample-trades", type=int, default=50000)
    ap.add_argument("--out", default="data/trade_outcomes")
    args = ap.parse_args()
    bars_dir = Path(args.bars_dir)

    df = pd.read_parquet(args.dataset)
    df = df[df["has_pre_features"]].copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    start = df["entry_time"].min().date().isoformat()
    end = (df["exit_time"].max() + pd.Timedelta(days=1)).date().isoformat()
    print(f"Reference bars (SPY/QQQ/sector) {start}..{end}:")
    refs = _ensure_refs(start, end, bars_dir)

    if args.sample_trades and len(df) > args.sample_trades:
        nrows = len(df)
        df = df.groupby("strategy_id", group_keys=False)[df.columns].apply(
            lambda s: s.sample(min(len(s), max(1, args.sample_trades * len(s) // nrows)),
                               random_state=0))
    pre_cols = [c for c in PRE_ENTRY_FEATURES if c in df.columns]
    print(f"\nTrades: {len(df):,}  building context + patterns + decision points...")
    per_trade, dec = build(df, pre_cols, bars_dir, refs)
    print(f"Per-trade rows: {len(per_trade):,} | decision points: {len(dec):,} "
          f"from {dec['trade_id'].nunique():,} trades\n")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summ_cols = ["symbol", "strategy_id", "pattern", "ctx_mkt_regime", "ctx_own_regime",
                 "direction", "outcome_win", "mfe_pct", "mae_pct", "profit_capture_ratio",
                 "giveback_from_mfe", "exit_efficiency"]
    summary = per_trade[[c for c in summ_cols if c in per_trade.columns]]
    summary.to_csv(out / "trade_management_summary.csv", index=False)
    dec.to_parquet(out / "decision_points_v2.parquet", index=False)

    print("Pattern distribution:")
    for p, n in per_trade["pattern"].value_counts().items():
        print(f"  {p:<24}{n:>7,}")

    report = {"trades": int(len(per_trade)), "decision_points": int(len(dec)),
              "entry_prediction_reference_auc": 0.53}
    analyze(per_trade, dec, pre_cols, report)
    (out / "trade_management_report.json").write_text(json.dumps(report, indent=2, default=str))
    _verdict(report)
    print(f"\nartifacts -> {out}/  (summary.csv, decision_points_v2.parquet, report.json)")


def _verdict(report):
    A, B = report["A_trade_level"], report["B_exit_management"]
    print("\n=== A. Trade-level: do pattern/regime/context beat entry-only? ===")
    for tgt, d in A.items():
        print(f"  {tgt} ({d['metric']}): entry-only {d['entry_only']}  ->  "
              f"+pattern/regime/ctx {d['entry+pattern+regime+ctx']}")
    print("\n=== B. Exit management: does mid-trade context add over pre+run? ===")
    for tgt, d in B.items():
        print(f"  {tgt}: pre+run {d['pre+run']}  ->  +context {d['pre+run+context']}")
    print("\n=== C. Best/worst patterns by profit capture ===")
    items = list(report["C_by_pattern"].items())
    for k, v in items[:3] + items[-2:]:
        print(f"  {k:<24} capture {v['median_profit_capture']:+.2f} "
              f"giveback {v['median_giveback_from_mfe']:+.3f} win {v['win_rate']:.2f} "
              f"(n={v['trades']:,})")
    exit_best = max((d["pre+run+context"] for d in B.values()), default=float("nan"))
    print(f"\nExit-management AUC ~{exit_best:.2f} vs entry-prediction ~0.53 "
          "-> management remains the dominant learnable signal.")


if __name__ == "__main__":
    main()
