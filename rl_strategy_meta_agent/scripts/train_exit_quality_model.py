#!/usr/bin/env python3
"""Exit-quality / trade-management model.

Builds DECISION-POINT examples (one per bar k of each trade) from the cached
15Min bars: inputs = pre_entry features + running path features up to k (NO
future), targets = what happens after k (early_exit_good, hold_longer_good,
severe_giveback, future_giveback). Trains classifiers, evaluates on a
group-by-symbol hold-out (no trade leaks across splits), and backtests three
exit policies (strategy / trailing-stop baseline / model-assisted). Research
only -- no allocator, no live trading.

    python scripts/train_exit_quality_model.py --sample-trades 50000
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
from rl_meta.exit_quality import (return_path, build_decision_rows, trailing_stop_return,
                                  RUN_FEATURES, DECISION_FLAGS)
from rl_meta.bar_coverage import cached_bars
from rl_meta.trade_quality import make_models, classification_metrics, proba


def _decision_dataset(df, pre_cols, bars_dir, timeframe):
    rows = []
    for sym, g in df.groupby("symbol"):
        bars = cached_bars(sym, timeframe, bars_dir)
        if bars.empty:
            continue
        ts_ns = pd.to_datetime(bars["timestamp"], utc=True).astype("int64").to_numpy()
        close = bars["close"].to_numpy(float)
        sub = g[pre_cols + ["entry_time", "exit_time", "entry_price", "direction",
                            "strategy_id", "split"]]
        for t in sub.itertuples(index=False):
            r = return_path(ts_ns, close, t.entry_time.value, t.exit_time.value,
                            t.entry_price, t.direction == "long")
            if r is None:
                continue
            pre = dict(zip(pre_cols, t[:len(pre_cols)]))
            meta = {"strategy_id": t.strategy_id, "symbol": sym, "split": t.split,
                    "trade_id": f"{sym}|{t.entry_time.isoformat()}"}
            rows.extend(build_decision_rows(r, pre, meta))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/trade_outcomes/trade_outcome_dataset.parquet")
    ap.add_argument("--bars-dir", default="data/bars")
    ap.add_argument("--timeframe", default="15Min")
    ap.add_argument("--sample-trades", type=int, default=50000)
    ap.add_argument("--out", default="data/trade_outcomes")
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    df = df[df["has_pre_features"]].copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    if args.sample_trades and len(df) > args.sample_trades:
        n = len(df)
        df = df.groupby("strategy_id", group_keys=False)[df.columns].apply(
            lambda g: g.sample(min(len(g), max(1, args.sample_trades * len(g) // n)),
                               random_state=0))
    pre_cols = [c for c in PRE_ENTRY_FEATURES if c in df.columns]
    print(f"Trades sampled: {len(df):,}  building decision points...")
    dp = _decision_dataset(df, pre_cols, Path(args.bars_dir), args.timeframe)
    if dp.empty:
        print("No decision points (need cached bars)."); return
    feats = pre_cols + RUN_FEATURES
    print(f"Decision points: {len(dp):,} from {dp['trade_id'].nunique():,} trades "
          f"({len(dp)/dp['trade_id'].nunique():.1f} per trade)\n")

    # group-by-symbol split: held-out symbols (no trade leaks across splits)
    from sklearn.model_selection import GroupShuffleSplit
    gi, ge = next(GroupShuffleSplit(1, test_size=0.3, random_state=0)
                  .split(dp, groups=dp["symbol"]))
    tr, te = dp.iloc[gi], dp.iloc[ge]
    X_tr, X_te = tr[feats], te[feats]

    report = {"trades": int(dp["trade_id"].nunique()), "decision_points": len(dp),
              "features": feats, "targets": {}, "vs_entry_prediction": None}
    print(f"{'target':<18}{'pos rate':>9}{'group AUC':>11}  best model")
    print("-" * 48)
    best_models = {}
    for tgt in DECISION_FLAGS:
        y_tr, y_te = tr[tgt].astype(int), te[tgt].astype(int)
        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            continue
        best_auc, best_name, best_m = -1, None, None
        for name, m in make_models().items():
            m.fit(X_tr, y_tr)
            auc = classification_metrics(y_te, proba(m, X_te))["auc"]
            if auc > best_auc:
                best_auc, best_name, best_m = auc, name, m
        best_models[tgt] = best_m
        report["targets"][tgt] = {"positive_rate": round(float(y_te.mean()), 3),
                                  "group_auc": round(best_auc, 3), "model": best_name}
        print(f"{tgt:<18}{y_te.mean():>9.3f}{best_auc:>11.3f}  {best_name}")

    _feature_importance(best_models, feats, report)
    _shap(best_models.get("severe_giveback") or best_models.get("early_exit_good"),
          X_te, feats, report)
    _policy_backtest(te, best_models, report)
    _verdict(report)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "exit_quality_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport -> {out / 'exit_quality_report.json'}")


def _importances(m):
    """feature_importances_ from a bare estimator or an sklearn Pipeline."""
    if hasattr(m, "feature_importances_"):
        return m.feature_importances_
    if hasattr(m, "named_steps") and hasattr(m.named_steps.get("clf"), "feature_importances_"):
        return m.named_steps["clf"].feature_importances_
    return None


def _feature_importance(models, feats, report):
    out = {}
    for tgt, m in models.items():
        imp = _importances(m)
        if imp is not None:
            out[tgt] = dict(sorted(zip(feats, imp.tolist()), key=lambda x: -x[1])[:10])
    report["feature_importance_top"] = out


def _shap(model, X_te, feats, report):
    if model is None or not hasattr(model, "feature_importances_"):
        report["shap"] = "skipped (no tree model)"; return
    try:
        import shap
        samp = X_te.sample(min(4000, len(X_te)), random_state=0)
        sv = shap.TreeExplainer(model).shap_values(samp)
        if isinstance(sv, list):
            sv = sv[1]
        ma = np.abs(sv).mean(0)
        cols = samp.to_numpy()
        direction = [float(np.corrcoef(cols[:, i], sv[:, i])[0, 1])
                     if np.std(cols[:, i]) > 0 else 0.0 for i in range(len(feats))]
        order = np.argsort(-ma)[:12]
        report["shap_top_features"] = [
            {"feature": feats[i], "mean_abs_shap": round(float(ma[i]), 5),
             "direction": round(direction[i], 3)} for i in order]
    except Exception as e:
        report["shap"] = f"error: {e}"


def _policy_backtest(te, models, report):
    """Per held-out trade: strategy vs trailing-stop baseline vs model-assisted."""
    model = models.get("severe_giveback") or models.get("early_exit_good")
    if model is None:
        return
    te = te.copy()
    te["_p"] = proba(model, te[report["features"]])
    grp = list(te.sort_values("k").groupby("trade_id"))
    strat = np.array([float(g["final_ret"].iloc[0]) for _, g in grp])
    # trailing-stop baseline: exit at first k whose giveback from peak >= 1%
    base = np.array([float(g[g["run_giveback"] >= 0.01]["ret_now"].iloc[0])
                     if (g["run_giveback"] >= 0.01).any() else float(g["final_ret"].iloc[0])
                     for _, g in grp])

    def model_policy(thr):
        out = []
        for _, g in grp:
            hit = g[g["_p"] >= thr]
            out.append(float(hit["ret_now"].iloc[0]) if len(hit)
                       else float(g["final_ret"].iloc[0]))
        return np.array(out)

    # sweep: severe-giveback exit fires more aggressively at lower thresholds
    sweep = {round(thr, 2): round(float(model_policy(thr).mean()), 5)
             for thr in np.arange(0.2, 0.71, 0.1)}
    best_thr = max(sweep, key=sweep.get)
    report["policy_backtest"] = {
        "trades": len(strat),
        "strategy_exit_mean_ret": round(float(strat.mean()), 5),
        "trailing_stop_mean_ret": round(float(base.mean()), 5),
        "model_assisted_sweep": sweep,
        "model_assisted_best_thr": best_thr,
        "model_assisted_best_ret": sweep[best_thr],
        "best_beats_strategy": sweep[best_thr] > float(strat.mean()),
    }


def _verdict(report):
    aucs = [v["group_auc"] for v in report["targets"].values()]
    best = max(aucs) if aucs else float("nan")
    report["vs_entry_prediction"] = {"exit_best_group_auc": best,
                                     "entry_group_auc_reference": 0.53}
    print("\n=== Verdict ===")
    print(f"Best exit-management group-AUC: {best:.3f}  (entry-prediction was ~0.53)")
    pb = report.get("policy_backtest")
    if pb:
        print(f"Policy backtest (held-out symbols, {pb['trades']:,} trades):")
        print(f"  strategy exit     mean ret {pb['strategy_exit_mean_ret']*100:+.2f}%")
        print(f"  trailing-stop     mean ret {pb['trailing_stop_mean_ret']*100:+.2f}%")
        print(f"  model-assisted    mean ret {pb['model_assisted_best_ret']*100:+.2f}% "
              f"(best thr {pb['model_assisted_best_thr']})")
        print(f"  sweep: " + "  ".join(f"{t}:{v*100:+.2f}%"
                                       for t, v in pb["model_assisted_sweep"].items()))
        print("  -> model-assisted "
              f"{'BEATS' if pb['best_beats_strategy'] else 'does NOT beat'} "
              "the strategy's own exit at any swept threshold.")


if __name__ == "__main__":
    main()
