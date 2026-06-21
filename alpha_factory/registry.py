"""Librarian Agent - maintains the alpha registry (append-only JSONL + accepted-prediction store).

Dedupe is by PREDICTION hash, not formula. Near-duplicates (high correlation) are caught at
evaluation time via correlation_to_accepted(), which Pearson-correlates a new prediction against
every accepted prediction on their shared (date,symbol) support.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .schema import ACCEPTED, AlphaRecord

_ROOT = os.path.join(os.path.dirname(__file__), "storage")


class AlphaRegistry:
    def __init__(self, root: str = _ROOT):
        self.root = root
        self.jsonl = os.path.join(root, "alpha_registry.jsonl")
        self.preds = os.path.join(root, "alpha_results")
        os.makedirs(self.preds, exist_ok=True)

    # --- read -------------------------------------------------------------
    def all(self) -> List[AlphaRecord]:
        if not os.path.exists(self.jsonl):
            return []
        with open(self.jsonl) as f:
            return [AlphaRecord.from_json(ln) for ln in f if ln.strip()]

    def accepted(self) -> List[AlphaRecord]:
        return [r for r in self.all() if r.decision == ACCEPTED]

    def rejected(self) -> List[AlphaRecord]:
        return [r for r in self.all() if r.decision == "rejected"]

    def hashes(self) -> set:
        return {r.prediction_hash for r in self.all()}

    # --- write ------------------------------------------------------------
    def add(self, rec: AlphaRecord, pred: Optional[pd.Series] = None) -> bool:
        """Append a record. Returns False (skipped) if this exact prediction already exists."""
        if rec.prediction_hash in self.hashes():
            return False
        with open(self.jsonl, "a") as f:
            f.write(rec.to_json() + "\n")
        if rec.decision == ACCEPTED and pred is not None:
            pred.to_pickle(os.path.join(self.preds, f"{rec.alpha_id}.pkl"))
        return True

    # --- crowding ---------------------------------------------------------
    def correlation_to_accepted(self, pred: pd.Series) -> float:
        """Max |Pearson| of pred against any stored accepted prediction (0 if none / no overlap)."""
        best = 0.0
        for fn in os.listdir(self.preds):
            if not fn.endswith(".pkl"):
                continue
            other = pd.read_pickle(os.path.join(self.preds, fn))
            j = pred.align(other, join="inner")
            if len(j[0]) < 30 or j[0].std() == 0 or j[1].std() == 0:
                continue
            c = abs(float(np.corrcoef(j[0].to_numpy(), j[1].to_numpy())[0, 1]))
            best = max(best, c if c == c else 0.0)
        return best
