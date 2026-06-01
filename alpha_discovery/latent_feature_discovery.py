#!/usr/bin/env python3
"""
alpha_discovery/latent_feature_discovery.py — latent compression of realtime
feature blocks into a small number of validated latent factors.

Compresses a block of realtime numeric features (price + technical + options +
news) into latent_realtime_factor_1..N using PCA (scikit-learn) with a
deterministic numpy SVD fallback. Fit IN-SAMPLE only; transform reused live.

This feeds the hidden-alpha layer; the latent factors are exposed to the RL
state only after validation + explicit gating (see rl_feature_exporter). Never
raises; degrades to neutral zeros.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

_DEFAULT_N = 3


class LatentFeatureDiscovery:
    def __init__(self, feature_names: list[str], n_components: int = _DEFAULT_N,
                 prefix: str = "realtime"):
        self.feature_names = list(feature_names)
        self.n_components = max(1, int(n_components))
        self.prefix = str(prefix or "realtime")
        self._pca = None
        self._svd = None        # (components, mean)
        self._fitted = False

    def _matrix(self, records: list[dict[str, Any]]):
        if np is None:
            return None
        rows = []
        for rec in records:
            row = []
            for f in self.feature_names:
                try:
                    v = float(rec.get(f, 0.0))
                except (TypeError, ValueError):
                    v = 0.0
                row.append(v)
            rows.append(row)
        return np.asarray(rows, dtype=float) if rows else np.zeros((0, len(self.feature_names)))

    def fit(self, records: list[dict[str, Any]]) -> "LatentFeatureDiscovery":
        if np is None or not records:
            return self
        X = self._matrix(records)
        if X is None or X.shape[0] < 2 or X.shape[1] == 0:
            return self
        # standardize columns to avoid scale dominance
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std == 0] = 1.0
        Xs = (X - self._mean) / self._std
        k = min(self.n_components, Xs.shape[1], Xs.shape[0])
        try:
            from sklearn.decomposition import PCA  # type: ignore

            self._pca = PCA(n_components=k, random_state=0)
            self._pca.fit(Xs)
            self._fitted = True
            return self
        except Exception:
            pass
        try:
            _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
            self._svd = Vt[:k]
            self._fitted = True
        except Exception:  # pragma: no cover
            self._fitted = False
        return self

    def transform_one(self, rec: dict[str, Any]) -> dict[str, float]:
        out = {f"latent_{self.prefix}_factor_{i + 1}": 0.0 for i in range(self.n_components)}
        if np is None or not self._fitted:
            return out
        try:
            X = self._matrix([rec])
            Xs = (X - self._mean) / self._std
            if self._pca is not None:
                z = self._pca.transform(Xs)[0]
            elif self._svd is not None:
                z = (Xs @ self._svd.T)[0]
            else:
                return out
            for i in range(min(self.n_components, len(z))):
                out[f"latent_{self.prefix}_factor_{i + 1}"] = round(float(z[i]), 6)
        except Exception:  # pragma: no cover
            return out
        return out


def discover_latent_factors(records: list[dict[str, Any]], feature_names: list[str],
                            n_components: int = _DEFAULT_N,
                            prefix: str = "realtime") -> list[dict[str, float]]:
    """Convenience: fit + transform a batch."""
    eng = LatentFeatureDiscovery(feature_names, n_components, prefix=prefix)
    eng.fit(records)
    return [eng.transform_one(r) for r in records]


# =============================================================================
# Options-flow latent factors (hidden options-state discovery)
# =============================================================================

# Sub-blocks of the options_sentiment family compressed into named latent states.
_OPTIONS_FACTOR_BLOCK = [
    "put_call_pressure_factor", "call_pressure_factor", "put_pressure_factor",
    "volume_oi_activity_factor", "unusual_options_activity_factor",
    "gamma_pressure_factor", "delta_pressure_factor", "iv_skew_factor",
    "iv_term_structure_factor", "option_sentiment_factor",
]
_OPTIONS_SENTIMENT_BLOCK = [
    "put_call_pressure_factor", "call_pressure_factor", "put_pressure_factor",
    "option_sentiment_factor", "volume_oi_activity_factor",
    "unusual_options_activity_factor",
]
_OPTIONS_GAMMA_BLOCK = [
    "gamma_pressure_factor", "delta_pressure_factor",
    "iv_skew_factor", "iv_term_structure_factor",
]


def discover_options_latent_factors(records: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Compress the options_sentiment family into latent options-flow states.

    Produces, per record:
        latent_options_factor_1..3       (PCA of the full options factor block)
        latent_options_sentiment_state   (1st latent of the sentiment/pressure sub-block)
        latent_gamma_state               (1st latent of the gamma/delta/IV sub-block)

    Fit IN-SAMPLE on `records`; neutral zeros when data is degenerate. Never raises.
    """
    n = len(records)
    base = [{
        "latent_options_factor_1": 0.0,
        "latent_options_factor_2": 0.0,
        "latent_options_factor_3": 0.0,
        "latent_options_sentiment_state": 0.0,
        "latent_gamma_state": 0.0,
    } for _ in range(n)]
    if n == 0:
        return base

    factor3 = discover_latent_factors(records, _OPTIONS_FACTOR_BLOCK, 3, prefix="options")
    sent1 = discover_latent_factors(records, _OPTIONS_SENTIMENT_BLOCK, 1, prefix="options_sentiment")
    gamma1 = discover_latent_factors(records, _OPTIONS_GAMMA_BLOCK, 1, prefix="options_gamma")

    for i in range(n):
        base[i]["latent_options_factor_1"] = factor3[i].get("latent_options_factor_1", 0.0)
        base[i]["latent_options_factor_2"] = factor3[i].get("latent_options_factor_2", 0.0)
        base[i]["latent_options_factor_3"] = factor3[i].get("latent_options_factor_3", 0.0)
        base[i]["latent_options_sentiment_state"] = sent1[i].get("latent_options_sentiment_factor_1", 0.0)
        base[i]["latent_gamma_state"] = gamma1[i].get("latent_options_gamma_factor_1", 0.0)
    return base
