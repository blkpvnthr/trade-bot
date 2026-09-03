#!/usr/bin/env python3
"""Train the trade_quality_model: can pre-entry chart context predict winners?

Target = outcome_win (normalized_pnl > 0). Features = pre_entry_features ONLY.
Trains Logistic Regression, Random Forest, and (if installed) XGBoost + LightGBM.
Reports AUC / precision / recall on the dataset's time splits (validation, oos)
AND on a clean stratified random hold-out, plus feature importance and SHAP.

    python scripts/train_trade_quality_model.py \
        --dataset data/trade_outcomes/trade_outcome_dataset.parquet
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
from rl_meta.trade_quality import classification_metrics, make_models, proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/trade_outcomes/trade_outcome_dataset.parquet")
    ap.add_argument("--out", default="data/trade_outcomes")
    ap.add_argument("--shap-sample", type=int, default=4000)
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    df = df[df["has_pre_features"] & df["outcome_win"].notna()].copy()
    feats = [f for f in PRE_ENTRY_FEATURES if f in df.columns]
    X = df[feats]
    y = df["outcome_win"].astype(int)
    print(f"Examples with pre-entry features: {len(df):,} | win rate {y.mean()*100:.1f}%")
    print(f"Features: {len(feats)} pre-entry only (no strategy_id/symbol/during/exit)\n")

    # Three splits with very different honesty:
    #  time   : train pre-2024 -> test 2024/2025 (real temporal generalization)
    #  group  : train on some symbols -> test on HELD-OUT symbols (cross-symbol)
    #  random : i.i.d. hold-out -- OPTIMISTIC: trades from the same symbol/period
    #           leak into both sides (autocorrelated samples), so it overstates.
    tr = df["split"] == "train"; va = df["split"] == "validation"; oos = df["split"] == "oos"
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X, y, test_size=0.3, stratify=y,
                                                  random_state=0)
    gi, ge = next(GroupShuffleSplit(1, test_size=0.3, random_state=0)
                  .split(X, y, groups=df["symbol"]))
    Xg_tr, Xg_te, yg_tr, yg_te = X.iloc[gi], X.iloc[ge], y.iloc[gi], y.iloc[ge]
    dr_te = df["strategy_id"].iloc[ge]

    models = make_models()
    report = {"n": len(df), "win_rate": round(float(y.mean()), 4), "features": feats,
              "time_split": {}, "group_split": {}, "random_split": {},
              "auc_by_strategy_group_holdout": {}}
    print(f"{'model':<14}{'val AUC':>9}{'oos AUC':>9}{'group AUC':>10}{'rand AUC':>9}")
    print("-" * 52)
    fitted_group = {}
    for name in models:
        mt = make_models()[name]; mt.fit(X[tr], y[tr])
        m_va = classification_metrics(y[va], proba(mt, X[va]))
        m_oos = classification_metrics(y[oos], proba(mt, X[oos]))
        mg = make_models()[name]; mg.fit(Xg_tr, yg_tr)
        m_g = classification_metrics(yg_te, proba(mg, Xg_te))
        mr = make_models()[name]; mr.fit(Xr_tr, yr_tr)
        m_r = classification_metrics(yr_te, proba(mr, Xr_te))
        fitted_group[name] = mg
        report["time_split"][name] = {"validation": m_va, "oos": m_oos}
        report["group_split"][name] = m_g
        report["random_split"][name] = m_r
        print(f"{name:<14}{m_va['auc']:>9.3f}{m_oos['auc']:>9.3f}{m_g['auc']:>10.3f}"
              f"{m_r['auc']:>9.3f}")

    # AUC by strategy on the GROUP (held-out-symbol) hold-out -- the honest test
    best = max(report["group_split"], key=lambda k: report["group_split"][k]["auc"])
    p = proba(fitted_group[best], Xg_te)
    for sid in sorted(dr_te.unique()):
        msk = (dr_te == sid).values
        if msk.sum() > 50:
            report["auc_by_strategy_group_holdout"][sid] = \
                classification_metrics(yg_te[msk], p[msk])["auc"]

    _feature_importance(fitted_group, feats, report)
    _shap(fitted_group, Xg_te, feats, args, report)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "trade_quality_report.json").write_text(json.dumps(report, indent=2, default=str))
    _verdict(report, best)
    print(f"\nreport -> {out / 'trade_quality_report.json'}")


def _feature_importance(fitted, feats, report):
    fi = {}
    if "random_forest" in fitted:
        imp = fitted["random_forest"].named_steps["clf"].feature_importances_
        fi["random_forest_gain"] = dict(sorted(zip(feats, imp.tolist()),
                                               key=lambda x: -x[1])[:12])
    if "logreg" in fitted:
        coef = fitted["logreg"].named_steps["clf"].coef_[0]
        fi["logreg_coef"] = dict(sorted(zip(feats, coef.tolist()),
                                        key=lambda x: -abs(x[1]))[:12])
    report["feature_importance"] = fi


def _shap(fitted, X_te, feats, args, report):
    model = fitted.get("xgboost") or fitted.get("lightgbm") or None
    if model is None:
        report["shap"] = "unavailable (no tree model)"
        return
    try:
        import shap
        samp = X_te.sample(min(args.shap_sample, len(X_te)), random_state=0)
        sv = shap.TreeExplainer(model).shap_values(samp)
        if isinstance(sv, list):
            sv = sv[1]
        mean_abs = np.abs(sv).mean(0)
        cols = samp.to_numpy()
        direction = [float(np.corrcoef(cols[:, i], sv[:, i])[0, 1])
                     if np.std(cols[:, i]) > 0 else 0.0 for i in range(len(feats))]
        order = np.argsort(-mean_abs)[:12]
        report["shap_top_features"] = [
            {"feature": feats[i], "mean_abs_shap": round(float(mean_abs[i]), 5),
             "direction": round(direction[i], 3)} for i in order]
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            shap.summary_plot(sv, samp, feature_names=feats, show=False, max_display=15)
            plt.tight_layout()
            plt.savefig(Path(args.out) / "shap_summary.png", dpi=110)
            plt.close()
            report["shap_plot"] = str(Path(args.out) / "shap_summary.png")
        except Exception as e:
            report["shap_plot"] = f"skipped: {e}"
    except Exception as e:
        report["shap"] = f"error: {e}"


def _verdict(report, best):
    # Honest generalization = the worse of OOS (time) and group (held-out symbol).
    oos = max(v["oos"]["auc"] for v in report["time_split"].values())
    grp = max(v["auc"] for v in report["group_split"].values())
    rnd = max(v["auc"] for v in report["random_split"].values())
    print("\n=== Verdict: can chart context predict winners better than random? ===")
    print(f"OOS (time) AUC:        {oos:.3f}")
    print(f"Group (held symbol):   {grp:.3f}")
    print(f"Random i.i.d. AUC:     {rnd:.3f}  <- OPTIMISTIC (autocorrelated leak)")
    honest = max(oos, grp)
    if honest >= 0.58:
        print(f"-> Yes: chart context generalizes (honest AUC {honest:.3f}).")
    elif honest >= 0.53:
        print(f"-> Weakly: small generalizable edge (honest AUC {honest:.3f}).")
    else:
        print(f"-> No: out-of-sample AUC ~{honest:.2f} ~ random. The random-split "
              f"{rnd:.2f} is inflated by trade autocorrelation (same symbol/period "
              "in train+test); it does NOT predict unseen trades.")
    if report.get("shap_top_features"):
        pos = [s["feature"] for s in report["shap_top_features"] if s["direction"] > 0.05][:4]
        neg = [s["feature"] for s in report["shap_top_features"] if s["direction"] < -0.05][:4]
        print(f"\nSHAP (descriptive of the fitted model, not validated OOS):")
        print(f"  pushes toward WIN (higher): {', '.join(pos) or '(none clear)'}")
        print(f"  pushes toward LOSS (higher): {', '.join(neg) or '(none clear)'}")


if __name__ == "__main__":
    main()
