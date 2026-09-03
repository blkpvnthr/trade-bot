"""AlphaRecord - one JSONL row per alpha evaluation. Stored by the Librarian (registry.py)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ACCEPTED = "accepted"
REJECTED = "rejected"
NEEDS_MORE_DATA = "needs_more_data"


def prediction_vector(signals: pd.DataFrame) -> pd.Series:
    """Canonical prediction = cross-sectional percentile-rank, flattened, NaN dropped.

    Rank (not raw value) because rank-IC is invariant to any monotone transform, so two formulas
    that differ only by a monotone wrap are the SAME alpha (dedupe by PREDICTION, not formula).
    """
    s = signals.rank(axis=1, pct=True).stack(future_stack=True).dropna()
    s.index = [f"{d.isoformat()}|{sym}" for d, sym in s.index]
    return s.sort_index()


def prediction_hash(vec: pd.Series) -> str:
    payload = "|".join(f"{k}={v:.4f}" for k, v in vec.items())
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


@dataclass
class AlphaRecord:
    alpha_id: str
    strategy_name: str
    source: str                      # alpha101 | mutation | combination | manual
    formula: str
    symbols: List[str]
    timeframe: str
    created_at: str
    evaluated_at: str
    prediction_hash: str
    in_sample_metrics: Dict[str, Any] = field(default_factory=dict)
    out_of_sample_metrics: Dict[str, Any] = field(default_factory=dict)
    walk_forward_metrics: Dict[str, Any] = field(default_factory=dict)
    cost_model: Dict[str, Any] = field(default_factory=dict)
    correlation_to_accepted: float = 0.0
    promotion: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    score_components: Dict[str, float] = field(default_factory=dict)
    decision: str = NEEDS_MORE_DATA
    rejection_reason: str = ""
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=_jsonable)

    @classmethod
    def from_json(cls, line: str) -> "AlphaRecord":
        return cls(**json.loads(line))


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not jsonable: {type(o)}")
