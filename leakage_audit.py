#!/usr/bin/env python3
"""
leakage_audit.py — Generic target-leakage audit for the RL datasets.
====================================================================

Generalizes the alpha_entry leakage investigation (docs/rl_psc_validation_findings.md) into a
reusable, offline, read-only audit that can be run on ANY (dataset, target) pair. For each it:

  1. Univariate signal scan — per feature: |AUC| vs sign(target) (classification framing) and
     linear R² + Spearman² vs target (regression framing).
  2. Near-deterministic / exact-functional detection — flags a feature when the target is an
     (near-)invertible function of it: linear R² > 0.99, |AUC| ≈ 1, or target == a·feat + b.
  3. Baselines — LogisticRegression + RandomForest on ALL features, temporal expanding-window
     CV (no shuffling), reporting OOS AUC (and a RandomForestRegressor OOS R² for the continuous
     target). A model that "nails" a FUTURE target out-of-sample is the macro leak signal.
  4. Permutation importance (RandomForest) — surfaces which feature carries the (possibly leaky)
     signal.

LEAK FLAGS (any ⇒ "LEAKAGE DETECTED"):
  * a feature with univariate |AUC| > AUC_FLAG (0.75) or univariate R² > R2_FLAG (0.50);
  * a feature with near-deterministic relationship to the target (linear R² > 0.99);
  * baseline OOS AUC > AUC_FLAG or baseline OOS R² > R2_FLAG (whole-model leak).

TARGETS MUST NEVER BE DERIVED FROM FEATURES. This tool only DETECTS; it never modifies a dataset
or a model. Writes docs/leakage_audit_<dataset>.md per audit. Deps: numpy + pandas + scikit-learn
(graceful: skips gracefully if sklearn is unavailable).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

import rl_common as rc

logger = logging.getLogger("leakage_audit")

AUC_FLAG = float(os.getenv("AUDIT_AUC_FLAG", "0.75"))
R2_FLAG = float(os.getenv("AUDIT_R2_FLAG", "0.50"))
DET_R2 = 0.99               # near-deterministic linear relationship
N_FOLDS = 5
SEED = 0

# (dataset, target, feature-list-attr, exclude-extra). Targets to audit per RL dataset.
AUDITS = [
    ("alpha_entry_dataset", "realized_R", "ALPHA_ENTRY_FEATURES"),
    ("alpha_exit_dataset", "realized_R", "ALPHA_EXIT_FEATURES"),
    ("buffett_dca_dataset", "fwd_return_60d", "BUFFETT_DCA_FEATURES"),
    ("capital_allocation_dataset", "alpha_bucket_return", "ALLOC_FEATURES"),
]


def _load(dataset: str):
    import pandas as pd
    p = rc.RL_DATA_DIR / f"{dataset}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "date_tag" in df.columns:
        keys = [k for k in ("date_tag", "timestamp", "symbol") if k in df.columns]
        df = df.sort_values(keys).reset_index(drop=True)
    return df


def _univariate(X: np.ndarray, y: np.ndarray, names: List[str]) -> List[Dict[str, Any]]:
    from sklearn.metrics import roc_auc_score
    win = (y > 0).astype(int)
    has_two = len(np.unique(win)) > 1
    out = []
    for j, nm in enumerate(names):
        v = X[:, j]
        if len(np.unique(v)) < 2:
            out.append({"feature": nm, "auc": None, "r2": 0.0, "spearman2": 0.0, "flag": False})
            continue
        auc = None
        if has_two:
            try:
                a = roc_auc_score(win, v); auc = round(max(a, 1 - a), 4)
            except Exception:
                auc = None
        # linear R² (corr²) and rank R² (Spearman², via rank-corr without scipy)
        r = np.corrcoef(v, y)[0, 1]
        r2 = 0.0 if np.isnan(r) else round(float(r * r), 4)
        rv = np.argsort(np.argsort(v)); ry = np.argsort(np.argsort(y))
        rs = np.corrcoef(rv, ry)[0, 1]
        sp2 = 0.0 if np.isnan(rs) else round(float(rs * rs), 4)
        flag = bool((auc is not None and auc > AUC_FLAG) or r2 > R2_FLAG or sp2 > R2_FLAG or r2 > DET_R2)
        reason = []
        if auc is not None and auc > AUC_FLAG: reason.append(f"AUC {auc}>{AUC_FLAG}")
        if r2 > DET_R2: reason.append(f"near-deterministic linear R² {r2}")
        elif r2 > R2_FLAG: reason.append(f"R² {r2}>{R2_FLAG}")
        if sp2 > R2_FLAG: reason.append(f"Spearman² {sp2}>{R2_FLAG}")
        out.append({"feature": nm, "auc": auc, "r2": r2, "spearman2": sp2,
                    "flag": flag, "reason": "; ".join(reason)})
    return sorted(out, key=lambda d: -(d["auc"] or 0) - d["r2"])


def _baselines(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Temporal expanding-window CV: LogReg+RF OOS AUC on sign(y), RF-regressor OOS R² on y."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import roc_auc_score, r2_score
    n = len(X); win = (y > 0).astype(int)
    bounds = [int(n * f) for f in np.linspace(0.5, 1.0, N_FOLDS + 1)]
    aucs_lr, aucs_rf, r2s = [], [], []
    for i in range(N_FOLDS):
        s, e = bounds[i], bounds[i + 1]
        if s < 50 or e - s < 20:
            continue
        Xtr, Xte = X[:s], X[s:e]; ytr, yte = win[:s], win[s:e]; ytr_c, yte_c = y[:s], y[s:e]
        # standardize for logreg
        mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-9] = 1.0
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        if len(np.unique(ytr)) > 1 and len(np.unique(yte)) > 1:
            try:
                lr = LogisticRegression(max_iter=200, random_state=SEED).fit(Ztr, ytr)
                aucs_lr.append(roc_auc_score(yte, lr.predict_proba(Zte)[:, 1]))
            except Exception: pass
            try:
                rf = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
                aucs_rf.append(roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]))
            except Exception: pass
        try:
            rfr = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=SEED, n_jobs=-1).fit(Xtr, ytr_c)
            r2s.append(r2_score(yte_c, rfr.predict(Xte)))
        except Exception: pass
    def ms(a): return {"mean": round(float(np.mean(a)), 4), "std": round(float(np.std(a)), 4), "folds": len(a)} if a else None
    return {"logreg_oos_auc": ms(aucs_lr), "random_forest_oos_auc": ms(aucs_rf), "rf_regressor_oos_r2": ms(r2s)}


def _perm_importance(X: np.ndarray, y: np.ndarray, names: List[str]) -> List[Dict[str, Any]]:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    n = len(X); s = int(n * 0.7)
    try:
        rfr = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=SEED, n_jobs=-1).fit(X[:s], y[:s])
        pi = permutation_importance(rfr, X[s:], y[s:], n_repeats=5, random_state=SEED, n_jobs=-1)
        order = np.argsort(-pi.importances_mean)
        return [{"feature": names[i], "importance": round(float(pi.importances_mean[i]), 5)} for i in order[:8]]
    except Exception as exc:
        logger.debug("perm importance failed: %s", exc)
        return []


def audit(dataset: str, target: str, feat_attr: str) -> Optional[Dict[str, Any]]:
    try:
        import sklearn  # noqa
    except Exception:
        return {"error": "sklearn unavailable"}
    df = _load(dataset)
    if df is None or target not in df.columns:
        return {"dataset": dataset, "error": f"missing dataset/target ({target})"}
    feats = [f for f in getattr(rc, feat_attr) if f in df.columns]
    sub = df[feats + [target]].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = sub[feats].to_numpy(); y = sub[target].to_numpy()
    uni = _univariate(X, y, feats)
    base = _baselines(X, y)
    perm = _perm_importance(X, y, feats)
    flagged = [u for u in uni if u["flag"]]
    macro_leak = ((base.get("logreg_oos_auc") or {}).get("mean", 0) or 0) > AUC_FLAG or \
                 ((base.get("random_forest_oos_auc") or {}).get("mean", 0) or 0) > AUC_FLAG or \
                 ((base.get("rf_regressor_oos_r2") or {}).get("mean", 0) or 0) > R2_FLAG
    verdict = "LEAKAGE DETECTED" if (flagged or macro_leak) else "CLEAN"
    return {"dataset": dataset, "target": target, "rows": len(df), "n_features": len(feats),
            "verdict": verdict, "win_rate": round(float((y > 0).mean()), 4),
            "univariate": uni, "flagged_features": flagged, "baselines": base,
            "permutation_importance": perm, "macro_leak": macro_leak}


def to_markdown(r: Dict[str, Any]) -> str:
    if "error" in r:
        return f"# Leakage audit — {r.get('dataset')}\n\n**ERROR:** {r['error']}\n"
    L = [f"# Leakage audit — `{r['dataset']}`", "",
         f"- **Verdict: {r['verdict']}**", f"- target: `{r['target']}`  | rows: {r['rows']}  | "
         f"features: {r['n_features']}  | win-rate: {r['win_rate']}", ""]
    b = r["baselines"]
    L += ["## Baselines (temporal expanding-window CV, out-of-sample)",
          f"- LogReg OOS AUC: {b.get('logreg_oos_auc')}",
          f"- RandomForest OOS AUC: {b.get('random_forest_oos_auc')}",
          f"- RF-Regressor OOS R²: {b.get('rf_regressor_oos_r2')}",
          f"- macro-leak (model nails a future target OOS): **{r['macro_leak']}**", ""]
    L += [f"## Flagged features ({len(r['flagged_features'])})  — AUC>{AUC_FLAG} / R²>{R2_FLAG} / near-deterministic"]
    if r["flagged_features"]:
        L += ["", "| feature | AUC | R² | Spearman² | reason |", "|---|--:|--:|--:|---|"]
        for u in r["flagged_features"]:
            L.append(f"| `{u['feature']}` | {u['auc']} | {u['r2']} | {u['spearman2']} | {u.get('reason','')} |")
    else:
        L.append("\n_None — no single feature is near-deterministic of the target._")
    L += ["", "## Top univariate signal (all features)", "",
          "| feature | AUC | R² | Spearman² |", "|---|--:|--:|--:|"]
    for u in r["univariate"][:12]:
        L.append(f"| `{u['feature']}` | {u['auc']} | {u['r2']} | {u['spearman2']} |")
    L += ["", "## Permutation importance (RandomForest, top 8)", "",
          "| feature | importance |", "|---|--:|"]
    for pi in r["permutation_importance"]:
        L.append(f"| `{pi['feature']}` | {pi['importance']} |")
    L += ["", "## Interpretation",
          ("**Leakage present.** One or more features (near-)determine the target, and/or a baseline "
           "predicts a future target out-of-sample implausibly well. The target is likely derived "
           "from features (forbidden). Re-target from FUTURE bars before any model use."
           if r["verdict"] == "LEAKAGE DETECTED" else
           "**No leakage detected.** No feature near-determines the target and baselines do not "
           "predict it out-of-sample beyond chance. (Absence of leakage ≠ presence of edge.)"), ""]
    return "\n".join(L)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    docs = rc.PROJECT_ROOT / "docs"
    docs.mkdir(exist_ok=True)
    summary = []
    for dataset, target, feat_attr in AUDITS:
        r = audit(dataset, target, feat_attr)
        if r is None:
            continue
        path = docs / f"leakage_audit_{dataset}.md"
        path.write_text(to_markdown(r), encoding="utf-8")
        v = r.get("verdict", r.get("error", "?"))
        summary.append((dataset, target, v, len(r.get("flagged_features", []))))
        print(f"  {dataset:28s} target={target:18s} -> {v}  (flagged={len(r.get('flagged_features',[]))})  -> {path}")
    rc.write_json(rc.DAILY_OUTPUTS / rc.data_tag() / "leakage_audit_summary.json",
                  {"generated_at_utc": rc.now_iso(),
                   "audits": [{"dataset": d, "target": t, "verdict": v, "flagged": n} for d, t, v, n in summary]})
    print("\nVERDICTS:")
    for d, t, v, n in summary:
        print(f"  {'⚠' if v=='LEAKAGE DETECTED' else '✓'} {d}: {v}")


if __name__ == "__main__":
    main()
