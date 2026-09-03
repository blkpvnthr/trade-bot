#!/usr/bin/env python3
"""
finbert_scorer.py — local financial-sentiment scoring with FinBERT (ProsusAI/finbert).

Replaces Alpha Vantage's pre-computed news sentiment. Alpaca news is raw text only, so we score
it ourselves: headline + summary -> sentiment score in [-1, 1] + a Positive/Neutral/Negative label.

DESIGN — fail-safe and never blocking (mirrors forecast_engine.py's torch degradation):
  * transformers/torch are imported lazily; if either is missing the scorer returns NEUTRAL.
  * the model is loaded from a LOCAL dir only (local_files_only=True) — never hits the network at
    runtime. Download once with `bootstrap_finbert.py --download`.
  * gated by FINBERT_ENABLED (default off) so a host without the model runs fine on neutral scores.
  * every public call returns a result and never raises; on any failure -> NEUTRAL_RESULT.

NEUTRAL_RESULT maps onto the news schema as overall_sentiment_score=None / label="Neutral",
exactly how the legacy AV path behaves when sentiment is unavailable.

Score convention: ProsusAI/finbert emits {positive, negative, neutral} probabilities;
score = p_positive - p_negative  (in [-1, 1]); label = argmax over the three (read from
model.config.id2label so we never hardcode the index order).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("finbert_scorer")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINBERT_ENABLED = os.getenv("FINBERT_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
FINBERT_MODEL_DIR = Path(os.getenv("FINBERT_MODEL_DIR", str(PROJECT_ROOT / "models" / "nlp" / "finbert")))
FINBERT_MODEL_ID = os.getenv("FINBERT_MODEL_ID", "ProsusAI/finbert")   # used only by bootstrap download
_MAX_LEN = 512

NEUTRAL_RESULT: dict[str, Any] = {"score": None, "label": "Neutral", "probs": None}

# ── lazy, optional deps (mirror forecast_engine.py:67-78) ────────────────────────
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _FINBERT_LIB = True
except Exception:
    torch = None  # type: ignore[assignment]
    AutoModelForSequenceClassification = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    _FINBERT_LIB = False

# Singleton model state: None = not yet attempted; False = load failed; tuple = (tokenizer, model, id2label)
_MODEL: Any = None


def _canon_label(raw: str) -> str:
    """Map finbert's lowercase labels to the schema's Positive/Neutral/Negative."""
    r = str(raw or "").strip().lower()
    if r.startswith("pos"):
        return "Positive"
    if r.startswith("neg"):
        return "Negative"
    return "Neutral"


def _load() -> Any:
    """Lazily load (tokenizer, model, id2label) from the local model dir. Returns False on any failure.

    local_files_only=True — never downloads at runtime; the model must be pre-fetched by
    bootstrap_finbert.py. Cached after first call (success or failure).
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not (FINBERT_ENABLED and _FINBERT_LIB):
        _MODEL = False
        return _MODEL
    try:
        if not (FINBERT_MODEL_DIR / "config.json").exists():
            logger.warning("FinBERT model not found at %s — run bootstrap_finbert.py --download", FINBERT_MODEL_DIR)
            _MODEL = False
            return _MODEL
        tok = AutoTokenizer.from_pretrained(str(FINBERT_MODEL_DIR), local_files_only=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(str(FINBERT_MODEL_DIR), local_files_only=True)
        mdl.eval()
        id2label = {int(k): str(v) for k, v in mdl.config.id2label.items()}
        _MODEL = (tok, mdl, id2label)
        logger.info("FinBERT loaded from %s (labels=%s)", FINBERT_MODEL_DIR, id2label)
    except Exception as e:  # noqa: BLE001 — must never raise into the news path
        logger.warning("FinBERT load failed (%s); scoring neutral", str(e)[:160])
        _MODEL = False
    return _MODEL


def is_available() -> bool:
    """True if FinBERT is enabled, the libs are present, and the local model loads."""
    return bool(_load())


def _probs_to_result(prob_row: Any, id2label: dict[int, str]) -> dict[str, Any]:
    probs = {_canon_label(id2label[i]).lower(): float(prob_row[i]) for i in range(len(prob_row))}
    p_pos = probs.get("positive", 0.0)
    p_neg = probs.get("negative", 0.0)
    score = max(-1.0, min(1.0, p_pos - p_neg))
    top_idx = int(max(range(len(prob_row)), key=lambda i: float(prob_row[i])))
    return {"score": score, "label": _canon_label(id2label[top_idx]), "probs": probs}


def score_batch(texts: list[str]) -> list[dict[str, Any]]:
    """Score a list of texts. Returns one result dict per input; NEUTRAL_RESULT for blanks/failures.

    Always returns len(texts) results and never raises.
    """
    n = len(texts)
    out = [dict(NEUTRAL_RESULT) for _ in range(n)]
    if n == 0:
        return out
    loaded = _load()
    if not loaded:
        return out
    tok, mdl, id2label = loaded
    # Score only the non-empty texts; leave blanks neutral.
    idx = [i for i, t in enumerate(texts) if str(t or "").strip()]
    if not idx:
        return out
    try:
        batch = [str(texts[i]) for i in idx]
        enc = tok(batch, return_tensors="pt", truncation=True, max_length=_MAX_LEN, padding=True)
        with torch.no_grad():
            logits = mdl(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for k, i in enumerate(idx):
            out[i] = _probs_to_result(probs[k], id2label)
    except Exception as e:  # noqa: BLE001
        logger.warning("FinBERT scoring failed (%s); returning neutral", str(e)[:160])
    return out


def score_text(text: str) -> dict[str, Any]:
    """Score a single text. NEUTRAL_RESULT for blank/failure. Never raises."""
    return score_batch([text])[0]


if __name__ == "__main__":  # quick manual smoke test
    import json
    samples = [
        "Company beats earnings expectations and raises full-year guidance",
        "Firm slashes outlook amid weak demand and rising costs",
        "Board to meet next week to discuss routine matters",
        "",
    ]
    print("FINBERT_ENABLED=%s  available=%s  dir=%s" % (FINBERT_ENABLED, is_available(), FINBERT_MODEL_DIR))
    for s, r in zip(samples, score_batch(samples)):
        print(json.dumps({"text": s[:50], **r}))
