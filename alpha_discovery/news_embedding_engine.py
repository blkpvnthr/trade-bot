#!/usr/bin/env python3
"""
alpha_discovery/news_embedding_engine.py — compressed news text embeddings.

Turns news headline + summary + topics into a small set of validated latent
features (compressed_news_embedding_1..N). RAW TEXT IS NEVER fed to the RL
models — only the compressed embedding coordinates are exposed.

Embedding backend (graceful degradation):
    1. sentence-transformers (if installed)         -> dense semantic vector
    2. deterministic hashing bag-of-words fallback   -> always available

Compression:
    PCA (scikit-learn) if available, else a deterministic numpy SVD, else the
    first N hashed dimensions. Fit IN-SAMPLE only; transform is reused live.

All failures degrade to neutral zero embeddings; never raises.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is a hard dep elsewhere
    np = None  # type: ignore[assignment]

_DEFAULT_N_COMPONENTS = 3
_HASH_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _text_of(row: dict[str, Any]) -> str:
    parts = [str(row.get("title") or ""), str(row.get("summary") or "")]
    topics = row.get("topics") or []
    if isinstance(topics, list):
        parts.append(" ".join(str(t) for t in topics))
    return " ".join(parts).lower()


def _hash_embed(text: str, dim: int = _HASH_DIM) -> "np.ndarray":
    """Deterministic hashing bag-of-words embedding (l2-normalized)."""
    vec = [0.0] * dim
    for tok in _TOKEN_RE.findall(text):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vec[idx] += sign
    if np is None:
        return vec  # type: ignore[return-value]
    arr = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr


class NewsEmbeddingEngine:
    """Fit-once / transform-many compressor for news text embeddings."""

    def __init__(self, n_components: int = _DEFAULT_N_COMPONENTS, use_transformer: bool = True):
        self.n_components = max(1, int(n_components))
        self.use_transformer = use_transformer
        self._st_model = None
        self._pca = None
        self._svd_components = None  # numpy fallback (components, mean)
        self._fitted = False
        if use_transformer:
            self._try_load_transformer()

    def _try_load_transformer(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("news_embedding_engine: using sentence-transformers backend")
        except Exception:
            self._st_model = None  # fall back to hashing silently

    def _raw_embed_one(self, row: dict[str, Any]):
        text = _text_of(row)
        if not text.strip():
            return _hash_embed("")
        if self._st_model is not None:
            try:
                v = self._st_model.encode(text, show_progress_bar=False)
                return np.asarray(v, dtype=float) if np is not None else list(v)
            except Exception:  # pragma: no cover - degrade to hashing
                pass
        return _hash_embed(text)

    def _raw_matrix(self, rows: list[dict[str, Any]]):
        if np is None:
            return [self._raw_embed_one(r) for r in rows]
        return np.vstack([self._raw_embed_one(r) for r in rows]) if rows else np.zeros((0, _HASH_DIM))

    def fit(self, rows: list[dict[str, Any]]) -> "NewsEmbeddingEngine":
        """Fit the compressor on a corpus of news rows (in-sample)."""
        if np is None or not rows:
            self._fitted = False
            return self
        X = self._raw_matrix(rows)
        if X.shape[0] < 2:
            self._fitted = False
            return self
        k = min(self.n_components, X.shape[1], X.shape[0])
        try:
            from sklearn.decomposition import PCA  # type: ignore

            self._pca = PCA(n_components=k, random_state=0)
            self._pca.fit(X)
            self._fitted = True
            return self
        except Exception:
            pass
        # numpy SVD fallback.
        try:
            mean = X.mean(axis=0)
            Xc = X - mean
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            self._svd_components = (Vt[:k], mean)
            self._fitted = True
        except Exception:  # pragma: no cover - defensive
            self._fitted = False
        return self

    def transform_one(self, row: dict[str, Any]) -> dict[str, float]:
        """Compress one news row to compressed_news_embedding_1..N (neutral if unfit)."""
        out = {f"compressed_news_embedding_{i + 1}": 0.0 for i in range(self.n_components)}
        if np is None or not self._fitted:
            return out
        try:
            x = self._raw_embed_one(row).reshape(1, -1)
            if self._pca is not None:
                z = self._pca.transform(x)[0]
            elif self._svd_components is not None:
                comps, mean = self._svd_components
                z = (x - mean) @ comps.T
                z = z[0]
            else:
                return out
            for i in range(min(self.n_components, len(z))):
                out[f"compressed_news_embedding_{i + 1}"] = round(float(z[i]), 6)
        except Exception:  # pragma: no cover - defensive
            return out
        return out

    def transform(self, rows: list[dict[str, Any]]) -> list[dict[str, float]]:
        return [self.transform_one(r) for r in rows]


def build_compressed_embeddings(rows: list[dict[str, Any]],
                                n_components: int = _DEFAULT_N_COMPONENTS) -> list[dict[str, float]]:
    """Convenience: fit + transform a batch of news rows in one call."""
    engine = NewsEmbeddingEngine(n_components=n_components)
    engine.fit(rows)
    return engine.transform(rows)
