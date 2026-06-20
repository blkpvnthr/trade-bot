import numpy as np

from rl_meta.trade_quality import classification_metrics, make_models, proba


def test_metrics_perfect_and_random():
    assert classification_metrics([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])["auc"] == 1.0
    m = classification_metrics([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])
    assert m["auc"] == 0.5 and m["baseline_auc"] == 0.5


def test_metrics_keys_and_baseline():
    m = classification_metrics([1, 1, 1, 0], [0.6, 0.7, 0.4, 0.3])
    for k in ("n", "positive_rate", "auc", "precision", "recall", "accuracy", "baseline_auc"):
        assert k in m
    assert m["positive_rate"] == 0.75


def test_models_present():
    models = make_models()
    assert "logreg" in models and "random_forest" in models   # always available


def test_learnable_signal_recovered():
    rng = np.random.default_rng(1)
    n = 1500
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] - X[:, 1] + 0.4 * rng.normal(size=n) > 0).astype(int)
    m = make_models()["random_forest"]
    m.fit(X[:1000], y[:1000])
    assert classification_metrics(y[1000:], proba(m, X[1000:]))["auc"] > 0.75


def test_handles_nan_features():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 3))
    X[::7, 0] = np.nan                      # imputer in the pipeline handles these
    y = (X[:, 1] > 0).astype(int)
    m = make_models()["logreg"]
    m.fit(X[:300], y[:300])
    assert 0.0 <= classification_metrics(y[300:], proba(m, X[300:]))["auc"] <= 1.0
