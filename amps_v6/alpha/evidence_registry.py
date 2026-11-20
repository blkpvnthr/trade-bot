"""Versioned entry-alpha evidence registry (V6 spec sections 14, 44).

Every tradeable claim must point to a dated evidence record with an explicit lifecycle state. Mode
gating is strict: research candidates may only be used in research; paper entries need paper-approved
or stronger; live pilot/live need the corresponding approval. Promotion is an explicit transition,
never implicit. Drift monitoring can demote LIVE_APPROVED -> LIVE_SUSPENDED -> REVALIDATION_REQUIRED.

Pure + JSON-persisted; no network. This is the single source of truth for "is there evidence?".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "logs" / "execution" / "amps_v6_evidence_registry.json"

# lifecycle states (ordered by strength)
RESEARCH_CANDIDATE = "RESEARCH_CANDIDATE"
PAPER_CANDIDATE = "PAPER_CANDIDATE"
LIVE_PILOT_CANDIDATE = "LIVE_PILOT_CANDIDATE"
LIVE_APPROVED = "LIVE_APPROVED"
LIVE_SUSPENDED = "LIVE_SUSPENDED"
REVALIDATION_REQUIRED = "REVALIDATION_REQUIRED"

_STRENGTH = {RESEARCH_CANDIDATE: 0, PAPER_CANDIDATE: 1, LIVE_PILOT_CANDIDATE: 2, LIVE_APPROVED: 3,
             LIVE_SUSPENDED: -1, REVALIDATION_REQUIRED: -1}

# minimum state required to act in each mode
_MODE_MIN = {"BACKTEST": RESEARCH_CANDIDATE, "OBSERVE_ONLY": RESEARCH_CANDIDATE,
             "SHADOW": RESEARCH_CANDIDATE, "PAPER": PAPER_CANDIDATE,
             "LIVE_PILOT": LIVE_PILOT_CANDIDATE, "LIVE": LIVE_APPROVED}


@dataclass
class Evidence:
    evidence_id: str
    setup_family: str
    trading_horizon: str
    instrument_class: str
    state: str = RESEARCH_CANDIDATE
    # measured metrics (populated by the diagnostic / validation)
    net_expectancy_bps: Optional[float] = None
    mfe_mae_ratio: Optional[float] = None
    edge_vs_control_bps: Optional[float] = None
    n_trades: Optional[int] = None
    classification: Optional[str] = None            # VALIDATED/CONTINUE_RESEARCH/REJECTED/INSUFFICIENT_EVIDENCE
    created_at: str = ""
    updated_at: str = ""
    feature_distribution_version: str = "v1"
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class EvidenceRegistry:
    def __init__(self, records: Optional[dict] = None):
        self._rec: dict[str, Evidence] = records or {}

    def put(self, ev: Evidence) -> None:
        self._rec[ev.evidence_id] = ev

    def get(self, evidence_id: str) -> Optional[Evidence]:
        return self._rec.get(evidence_id)

    def all(self) -> list[Evidence]:
        return list(self._rec.values())

    def promote(self, evidence_id: str, to_state: str, updated_at: str) -> Evidence:
        ev = self._rec[evidence_id]
        if to_state not in _STRENGTH:
            raise ValueError(f"unknown state {to_state!r}")
        # forward promotion only allowed one step at a time to LIVE_APPROVED; demotions always allowed
        if _STRENGTH[to_state] > _STRENGTH.get(ev.state, 0) + 1 and _STRENGTH[to_state] >= 0:
            raise ValueError(f"cannot skip states {ev.state} -> {to_state}")
        ev.state = to_state
        ev.updated_at = updated_at
        return ev

    def suspend(self, evidence_id: str, updated_at: str, reason: str = "") -> Evidence:
        ev = self._rec[evidence_id]
        ev.state = LIVE_SUSPENDED
        ev.updated_at = updated_at
        ev.notes = (ev.notes + f" | SUSPENDED: {reason}").strip(" |")
        return ev

    def permits_entry(self, evidence_id: str, mode: str) -> tuple[bool, str]:
        ev = self._rec.get(evidence_id)
        if ev is None:
            return False, "no_evidence_record"
        if ev.state in (LIVE_SUSPENDED, REVALIDATION_REQUIRED):
            return False, f"evidence_state={ev.state}"
        need = _MODE_MIN.get(mode, LIVE_APPROVED)
        if _STRENGTH.get(ev.state, 0) < _STRENGTH[need]:
            return False, f"evidence {ev.state} < required {need} for mode {mode}"
        return True, "ok"

    # ---- persistence ----
    def save(self, path: Optional[Path] = None) -> Path:
        p = path or REGISTRY_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({k: v.as_dict() for k, v in self._rec.items()}, indent=2))
        return p

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "EvidenceRegistry":
        p = path or REGISTRY_PATH
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text())
        return cls({k: Evidence(**v) for k, v in raw.items()})
