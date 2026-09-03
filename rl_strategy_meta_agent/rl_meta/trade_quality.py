"""trade_quality_model: can pre-entry chart context predict winning trades?

Target = outcome_win (normalized_pnl > 0). Inputs = pre_entry_features ONLY (no
strategy_id, symbol, during/exit, or label-derived columns -- a strict, leakage-
free entry-quality question). This module holds the pure, testable pieces; the
CLI in scripts/train_trade_quality_model.py orchestrates train/eval/SHAP.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             accuracy_score)


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    pred = (y_prob >= threshold).astype(int)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    return {
        "n": int(len(y_true)),
        "positive_rate": round(float(y_true.mean()), 4),
        "auc": round(auc, 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "baseline_auc": 0.5,                       # random / majority-class AUC
    }


def make_models() -> dict:
    """Logistic Regression + Random Forest always; XGBoost/LightGBM if installed.
    Tree models receive raw X (native NaN handling); pipelines impute first."""
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    models = {
        "logreg": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "random_forest": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=12,
                    class_weight="balanced", n_jobs=-1, random_state=0))]),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, eval_metric="auc", random_state=0)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, class_weight="balanced",
            n_jobs=-1, random_state=0, verbose=-1)
    except ImportError:
        pass
    return models


def proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def _demo():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)   # learnable signal
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression().fit(x[:1500], y[:1500])
    auc = classification_metrics(y[1500:], proba(m, x[1500:]))["auc"]
    assert auc > 0.8, auc
    assert classification_metrics([1, 0, 1, 0], [0.4, 0.6, 0.4, 0.6])["auc"] < 0.5
    assert {"logreg", "random_forest"} <= set(make_models())
    print("trade_quality._demo OK auc", auc)


if __name__ == "__main__":
    _demo()
