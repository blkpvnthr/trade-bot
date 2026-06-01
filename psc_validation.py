#!/usr/bin/env python3
"""
psc_validation.py — Rigorous validation of the causal-state predictive compression.
===================================================================================

Tests the central hypothesis raised after the PPO paper-mode result:
    "Does the causal-state predictive compression create a materially more informative state
     representation than the raw 34-feature space — and is the edge in the REPRESENTATION
     rather than the PPO?"

It answers four of the pre-promotion questions, leakage-free:

  1. MULTI-FOLD WALK-FORWARD (temporal stability). K expanding time folds over the dataset.
     ⚠ HONESTY: the offline data currently spans only ~6 weeks (2026-04 to 2026-05). This tests
     temporal STABILITY within one recent window — it does NOT test cross-regime robustness
     (bull/bear/2018-2025). That needs a real multi-year backfill; it is a DATA limitation, not
     a code one, and is reported as such.

  2. NON-RL CLASSIFIER BENCHMARK. LogisticRegression / RandomForest / HistGradientBoosting on
     each feature set. If a simple classifier matches the PPO's OOS expectancy, the edge is the
     representation, not the RL algorithm.

  3. FEATURE ABLATION. raw34 · raw+causal · causal_only · raw+causal−future_return_embedding ·
     raw+pls. Isolates how much performance comes from the causal embedding, and specifically
     from `future_return_embedding`.

  4. (caller) PORTFOLIO SIM via rl_strategy_overlay.simulate on the best signal.

LEAKAGE CONTROL (critical): the embeddings are RE-FIT INSIDE EACH FOLD's TRAIN slice and applied
to that fold's TEST. The dataset's baked PSC columns are NOT used (their IS/OOS boundary is fixed
at 70% and would leak under other splits). Every map — standardizer, belief predictor, PLS,
classifier — is fit on TRAIN only; TEST is untouched until scoring. Thresholds are fixed a priori
(τ=0.5) or rank-based (decile spread), never tuned on TEST.

Interpretation note: the causal `future_return_embedding` is literally a supervised model's
E[future R | x] (a HistGBM belief predictor). So "causal compression" is, honestly, a
leakage-safe STACKED return predictor; its OOS decile-spread IS that predictor's OOS power. This
harness measures whether that meta-feature GENERALIZES across folds and beats raw features.

OFFLINE, read-only. No Alpaca, no orders.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

import numpy as np

import rl_common as rc
import predictive_state_compression as psc

logger = logging.getLogger("psc_validation")

N_FOLDS = 5
SEED = 0
TAU = 0.5                    # fixed entry threshold P(win) ≥ τ (NOT tuned on test)
# Features that are OUTCOME-COUPLED in the offline/replay dataset and therefore LEAK the target:
#   probability_of_profit — PoP>0.8 ⇒ 99.4% realized win, PoP<0.45 ⇒ 3.5% (univariate AUC 0.92);
#   expected_R            — the _proxy_realized_R path sets realized_R = 4·pop−1 / = expected_R.
# They are EXCLUDED by default so the validation measures genuine ex-ante signal, not the leak.
# Set PSC_VAL_INCLUDE_LEAKY=1 to include them (reproduces the inflated AUC≈0.999 artifact).
import os as _os
LEAK_FEATURES = set() if _os.getenv("PSC_VAL_INCLUDE_LEAKY") == "1" else {"probability_of_profit", "expected_R"}
LEAK_AUC_FLAG = 0.80         # any feature with univariate |AUC| above this is flagged as suspect


def contamination_scan(X: np.ndarray, y: np.ndarray, names: List[str]) -> List[Dict[str, float]]:
    """Univariate |AUC| of each feature vs win — flags outcome-coupled (leaky) features."""
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return []
    win = (y > 0).astype(int)
    if len(np.unique(win)) < 2:
        return []
    out = []
    for j, nm in enumerate(names):
        v = X[:, j]
        if len(np.unique(v)) < 2:
            continue
        try:
            a = roc_auc_score(win, v); a = max(a, 1 - a)
            out.append({"feature": nm, "auc": round(float(a), 4), "suspect_leak": bool(a > LEAK_AUC_FLAG)})
        except Exception:
            pass
    return sorted(out, key=lambda d: -d["auc"])


def _classifiers() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
        out["logreg"] = lambda: LogisticRegression(max_iter=200, random_state=SEED)
        out["random_forest"] = lambda: RandomForestClassifier(n_estimators=120, max_depth=8,
                                                              random_state=SEED, n_jobs=-1)
        out["hist_gbm"] = lambda: HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08,
                                                                max_depth=4, random_state=SEED)
    except Exception as exc:
        logger.warning("sklearn classifiers unavailable: %s", exc)
    return out


def _fold_features(X34tr, ytr, X34te, names) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Build every ablation feature set for one fold, embeddings RE-FIT on TRAIN only.
    Returns {set_name: (train_matrix, test_matrix)}. Leakage-free by construction."""
    (Xtr, Xte), _, _ = psc._standardize(X34tr, X34te)
    (ybin_tr, _), edges = psc._bin_future(ytr, ytr)
    ybin_te = np.digitize(ytr * 0, edges)  # placeholder; not needed for features
    # causal-state embedding (belief predictor + clustering fit on TRAIN)
    cs = psc.discover_causal_states(Xtr, ytr, ybin_tr, Xte, ytr[:len(Xte)] if len(Xte) else ytr,
                                    np.digitize(ytr[:len(Xte)] if len(Xte) else ytr, edges), names)
    # causal feature block (train + test), matching PSC_FEATURES layout sans the const score
    def causal_block(belief, state, fre, conf):
        ps = belief[:, :3] if belief.shape[1] >= 3 else np.pad(belief, ((0, 0), (0, 3 - belief.shape[1])))
        return np.column_stack([ps, state.reshape(-1, 1), fre.reshape(-1, 1), conf.reshape(-1, 1)])
    caus_tr = causal_block(cs["belief_tr"], cs["state_tr"], cs["belief_tr"] @ psc._bin_centers(ytr, ybin_tr, cs["belief_tr"].shape[1]), np.ones(len(Xtr)) * 0.5)
    caus_te = causal_block(cs["belief_te"], cs["state_te"], cs["future_return_embedding_te"], cs["confidence_te"])
    fre_col = caus_tr.shape[1] - 2  # index of future_return_embedding within the causal block
    # PLS components (fit on TRAIN)
    pls = psc.compress_pls(Xtr, ytr, Xte)
    if pls is not None:
        pls_tr, pls_te, _ = pls
    else:
        pls_tr = np.zeros((len(Xtr), 4)); pls_te = np.zeros((len(Xte), 4))
    sets = {
        "raw34": (Xtr, Xte),
        "raw+causal": (np.column_stack([Xtr, caus_tr]), np.column_stack([Xte, caus_te])),
        "causal_only": (caus_tr, caus_te),
        "raw+causal-FRE": (np.column_stack([Xtr, np.delete(caus_tr, fre_col, 1)]),
                           np.column_stack([Xte, np.delete(caus_te, fre_col, 1)])),
        "raw+pls": (np.column_stack([Xtr, pls_tr]), np.column_stack([Xte, pls_te])),
    }
    return sets


def _policy_metrics(proba: np.ndarray, y_te: np.ndarray) -> Dict[str, float]:
    """Threshold-free decile spread + fixed-τ entry-policy expectancy/Sharpe on TEST."""
    out = {"decile_spread": psc.decile_spread(proba, y_te)}
    enter = proba >= TAU
    entered = y_te[enter]
    out["frac_entered"] = float(enter.mean())
    out["policy_expectancy_R"] = float(entered.mean()) if entered.size else 0.0
    out["policy_sharpe"] = float(entered.mean() / (entered.std() + 1e-9)) if entered.size > 1 else 0.0
    try:
        from sklearn.metrics import roc_auc_score
        out["auc"] = float(roc_auc_score((y_te > 0).astype(int), proba)) if len(np.unique(y_te > 0)) > 1 else 0.5
    except Exception:
        out["auc"] = 0.5
    return out


def run(n_folds: int = N_FOLDS) -> Dict[str, Any]:
    # raw base features (PSC off → load_xy returns the base set), chronologically ordered.
    X, y, names, df = psc.load_xy("alpha_entry_dataset")
    # contamination scan on the FULL base set BEFORE dropping leaks (for the report)
    leak_scan = contamination_scan(X, y, names) if df is None else \
        contamination_scan(df[names].astype(float).fillna(0.0).to_numpy(), y, names)
    # drop baked PSC cols AND outcome-coupled leak features (genuine ex-ante signal only)
    base = [n for n in names if n not in rc.PSC_FEATURES and n not in LEAK_FEATURES]
    if df is not None:
        X = df[base].astype(float).fillna(0.0).to_numpy()
    names = base
    n = len(X)
    if n < 500:
        return {"error": "insufficient rows"}
    clfs = _classifiers()
    if not clfs:
        return {"error": "no sklearn classifiers"}
    win = (y > 0).astype(int)

    # K expanding-window folds: train on [0, s_f), test on [s_f, s_{f+1})
    bounds = [int(n * f) for f in np.linspace(0.5, 1.0, n_folds + 1)]
    folds = [(bounds[i], bounds[i + 1]) for i in range(n_folds)]

    # accumulate per (feature_set, classifier) across folds
    agg: Dict[str, Dict[str, List[float]]] = {}
    for fi, (s, e) in enumerate(folds):
        if e - s < 50 or s < 200:
            continue
        Xtr, ytr, wtr = X[:s], y[:s], win[:s]
        Xte, yte = X[s:e], y[s:e]
        try:
            sets = _fold_features(Xtr, ytr, Xte, names)
        except Exception as exc:
            logger.warning("fold %d feature build failed: %s", fi, exc); continue
        for set_name, (Mtr, Mte) in sets.items():
            for clf_name, make in clfs.items():
                key = f"{set_name}|{clf_name}"
                try:
                    clf = make(); clf.fit(Mtr, wtr[:len(Mtr)])
                    proba = clf.predict_proba(Mte)[:, list(clf.classes_).index(1)] \
                        if 1 in clf.classes_ else np.zeros(len(Mte))
                    m = _policy_metrics(proba, yte)
                except Exception as exc:
                    logger.debug("%s fold %d failed: %s", key, fi, exc); continue
                d = agg.setdefault(key, {k: [] for k in m})
                for k, v in m.items():
                    d[k].append(v)

    # summarize mean ± std across folds
    summary = {}
    for key, md in agg.items():
        summary[key] = {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4),
                            "folds": len(v)} for k, v in md.items()}

    report = {
        "generated_at_utc": rc.now_iso(), "date_tag": rc.data_tag(),
        "rows": n, "n_folds_run": sum(1 for s, e in folds if e - s >= 50 and s >= 200),
        "data_span_warning": "offline data ~6 weeks (single recent regime); tests temporal "
                             "stability, NOT cross-regime robustness — needs a multi-year backfill",
        "leakage_control": "embeddings + classifiers re-fit per-fold on TRAIN only; baked PSC "
                           "columns unused; thresholds fixed a priori",
        "entry_threshold_tau": TAU, "win_definition": "realized_R > 0",
        "excluded_leak_features": sorted(LEAK_FEATURES),
        "contamination_scan_top": leak_scan[:10],
        "feature_sets": ["raw34", "raw+causal", "causal_only", "raw+causal-FRE", "raw+pls"],
        "classifiers": list(clfs.keys()),
        "results": summary,
    }
    rc.write_json(rc.DAILY_OUTPUTS / rc.data_tag() / "psc_validation.json", report)
    return report


def _print(report: Dict[str, Any]) -> None:
    if "error" in report:
        print(report["error"]); return
    print("=" * 92)
    print("  CAUSAL-STATE COMPRESSION — leakage-safe multi-fold ablation × classifier benchmark")
    print("=" * 92)
    print(f"  rows={report['rows']}  folds={report['n_folds_run']}  τ={report['entry_threshold_tau']}")
    print(f"  ⚠ {report['data_span_warning']}")
    print(f"  leakage control: {report['leakage_control']}")
    print(f"  excluded leak features: {report['excluded_leak_features']}")
    susp = [d for d in report.get("contamination_scan_top", []) if d.get("suspect_leak")]
    if susp:
        print(f"  ⚠ SUSPECTED-LEAK features (univariate |AUC|>{LEAK_AUC_FLAG}): "
              + ", ".join(f"{d['feature']}({d['auc']})" for d in susp))
    cols = ["auc", "decile_spread", "policy_expectancy_R", "policy_sharpe", "frac_entered"]
    print(f"\n  {'feature_set | clf':34s}" + "".join(f"{c:>20s}" for c in cols))
    res = report["results"]
    # group by feature set for readability
    order = ["raw34", "raw+causal", "causal_only", "raw+causal-FRE", "raw+pls"]
    for fs in order:
        for clf in report["classifiers"]:
            key = f"{fs}|{clf}"
            if key not in res:
                continue
            r = res[key]
            cells = "".join(f"{r[c]['mean']:>13}±{r[c]['std']:<6}" for c in cols)
            print(f"  {key:34s}" + cells)
    print("=" * 92)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _print(run())
