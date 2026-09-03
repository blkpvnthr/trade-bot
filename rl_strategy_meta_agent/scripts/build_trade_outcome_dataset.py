#!/usr/bin/env python3
"""Build the trade-outcome learning dataset (one row per imported TradingView trade).

Pre-entry features need cached 15Min bars; without them those columns are NaN
(run fetch_alpaca_bars.py --timeframe 15Min first). During/exit-quality features,
labels, splits, and strategy context are real from the exports regardless.

    python scripts/build_trade_outcome_dataset.py --timeframe 15Min \
        --manifest data/tradingview_exports/manifest.json --bars-dir data/bars \
        --out data/trade_outcomes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta.trade_features import (read_trades_with_context, build_row,
                                   FEATURE_GROUPS, PRE_ENTRY_FEATURES)
from rl_meta.bar_coverage import cached_bars
from rl_meta.registry import read_registry
from rl_meta.performance import PerformanceDB
from rl_meta.labels import normalize_bucket


def _context_lookup():
    reg = {m.strategy_id: m for m in read_registry()}
    db = PerformanceDB()
    rl = {c["strategy_id"]: bool(c["eligible"]) for c in db.cells(split="full")
          if c["symbol"] == "AGGREGATE"}

    def ctx(sid, human_bucket):
        m = reg.get(sid)
        return {
            "strategy_family": m.strategy_family if m else "unknown",
            "strategy_confidence": m.strategy_confidence if m else np.nan,
            "verification_status": m.verification_status if m else "",
            "independence_score": m.independence_score if m else np.nan,
            "avg_correlation": m.avg_correlation if m else np.nan,
            "composite_score": np.nan,                 # not persisted; optional
            "rl_eligible": rl.get(sid, False),
            "human_bucket": human_bucket,
        }
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", default="15Min")
    ap.add_argument("--manifest", default="data/tradingview_exports/manifest.json")
    ap.add_argument("--bars-dir", default="data/bars")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--out", default="data/trade_outcomes")
    args = ap.parse_args()

    recs = [r for r in json.loads(Path(args.manifest).read_text()).get("records", [])
            if r["timeframe"] == args.timeframe]
    # dedup collision records that share a dest_path (else trades double-count)
    seen, deduped = set(), []
    for r in recs:
        if r["dest_path"] not in seen:
            seen.add(r["dest_path"]); deduped.append(r)
    recs = deduped
    ctx = _context_lookup()
    bars_cache: dict = {}
    rows, total_trades, missing_bars = [], 0, set()

    for r in recs:
        csv = Path(r["dest_path"])
        if not csv.exists():
            continue
        trades = read_trades_with_context(csv)
        total_trades += len(trades)
        sym = r.get("symbol") or "UNKNOWN"
        if sym not in bars_cache:
            cb = cached_bars(sym, args.timeframe, Path(args.bars_dir))
            bars_cache[sym] = cb if not cb.empty else None
            if cb.empty:
                missing_bars.add(sym)
        meta = {"strategy_id": r["strategy_id"], "symbol": sym, "timeframe": args.timeframe}
        c = ctx(r["strategy_id"], normalize_bucket(r.get("source_folder_label")))
        for t in trades:
            rows.append(build_row(t, bars_cache[sym], meta, c))

    if not rows:
        print("No trades. Run import_tradingview_trades.py first.")
        return
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "trade_outcome_dataset.csv", index=False)
    try:
        df.to_parquet(out / "trade_outcome_dataset.parquet", index=False)
        parquet = True
    except Exception as e:                              # pyarrow missing etc.
        parquet = False
        print(f"[parquet skipped: {e}]")
    (out / "feature_groups.json").write_text(json.dumps(FEATURE_GROUPS, indent=2))
    report = _report(df, total_trades, missing_bars, parquet)
    (out / "build_report.json").write_text(json.dumps(report, indent=2, default=str))
    _print_report(df, report, out)


def _report(df, total_trades, missing_bars, parquet):
    win = df["outcome_win"].dropna()
    return {
        "total_trades_processed": total_trades, "rows_written": len(df),
        "rows_with_pre_features": int(df["has_pre_features"].sum()),
        "rows_missing_bars": int((~df["has_pre_features"]).sum()),
        "symbols_missing_bars": sorted(missing_bars),
        "rows_by_strategy": df["strategy_id"].value_counts().to_dict(),
        "rows_by_symbol": df["symbol"].value_counts().to_dict(),
        "rows_by_split": df["split"].value_counts().to_dict(),
        "rows_by_human_bucket": df["human_bucket"].value_counts().to_dict(),
        "win_loss_flat": {"win": int(df["outcome_win"].sum()),
                          "loss": int(df["outcome_loss"].sum()),
                          "flat": int(df["outcome_flat"].sum())},
        "overall_win_rate": round(float(win.mean()), 4) if len(win) else None,
        "parquet_written": parquet,
        "missing_data_warnings": (
            (["pre-entry + regime features are NaN: no 15Min OHLCV bars cached"]
             if int(df["has_pre_features"].sum()) == 0 else [])
            + ["regime context (SPY/QQQ/VIX) not available -> NaN"]),
    }


def _print_report(df, rep, out):
    print("=== Trade-outcome dataset build report ===")
    for k in ("total_trades_processed", "rows_written", "rows_with_pre_features",
              "rows_missing_bars", "overall_win_rate"):
        print(f"{k:<24} {rep[k]}")
    print(f"win/loss/flat            {rep['win_loss_flat']}")
    print(f"rows by split            {rep['rows_by_split']}")
    print(f"rows by human_bucket     {rep['rows_by_human_bucket']}")
    print("rows by strategy         " + ", ".join(f"{k}:{v}" for k, v in rep["rows_by_strategy"].items()))

    # Exit-quality findings (REAL -- from CSV excursions, no bars needed)
    print("\n=== Exit-quality findings (winners vs losers, by strategy) ===")
    print(f"{'strategy':<18}{'capture(win)':>13}{'giveback(win)':>14}{'exit_eff':>9}{'n':>7}")
    for sid, g in df.groupby("strategy_id"):
        w = g[g["outcome_win"] == 1]
        cap = w["profit_capture_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        gb = w["giveback_from_mfe"].dropna()
        eff = g["exit_efficiency"].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"{sid[:18]:<18}{(cap.mean() if len(cap) else float('nan')):>13.2f}"
              f"{(gb.mean() if len(gb) else float('nan')):>14.4f}"
              f"{(eff.mean() if len(eff) else float('nan')):>9.2f}{len(g):>7}")

    # Top pre-entry features correlated with winners (only if bars present)
    print("\n=== Top pre-entry features correlated with winners ===")
    if rep["rows_with_pre_features"] == 0:
        print("  PENDING -- no 15Min bars cached; pre-entry features are NaN.")
    else:
        sub = df[df["has_pre_features"]].copy()
        corrs = {f: sub[f].corr(sub["outcome_win"]) for f in PRE_ENTRY_FEATURES
                 if sub[f].notna().any()}
        for f, cval in sorted(corrs.items(), key=lambda x: -abs(x[1] or 0))[:8]:
            print(f"  {f:<28} corr={cval:+.3f}")

    if rep["missing_data_warnings"]:
        print("\n=== Missing-data warnings ===")
        for w in rep["missing_data_warnings"]:
            print(f"  - {w}")
    print(f"\ndataset -> {out}/trade_outcome_dataset.csv"
          f"{' (+ .parquet)' if rep['parquet_written'] else ''}")
    print(f"groups  -> {out}/feature_groups.json")
    print(f"report  -> {out}/build_report.json")


if __name__ == "__main__":
    main()
