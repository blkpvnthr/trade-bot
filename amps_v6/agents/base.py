"""Specialist-agent interface + strict recommendation schema (V6 spec section 4).

Agents ANALYZE / RECOMMEND / CLASSIFY / RANK / VETO. They NEVER talk to the broker -- only the
deterministic execution gateway submits/cancels/replaces orders, and protective exits run
independently of any agent or LLM. Default implementations are deterministic; LLM-backed variants
sit behind feature flags and are never required for latency-sensitive decisions.

Every recommendation is validated against a strict schema. Malformed, expired, unsupported, or
untrusted recommendations are REJECTED (never silently trusted) -- external text is untrusted input.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

VALID_ACTIONS = {"ENTER_LONG", "HOLD", "REDUCE", "EXIT", "NO_TRADE", "VETO", "PROMOTE", "REJECT_PROMOTION"}
VALID_HORIZONS = {"MICRO_SCALP", "INTRADAY_MOMENTUM", "EXTENDED_INTRADAY", "SHORT_SWING",
                  "EXTENDED_SWING", "NONE"}


@dataclass
class Recommendation:
    agent_id: str
    recommendation_id: str
    symbol: str
    underlying: str
    timestamp: str                 # ISO 8601 UTC
    data_version: str
    strategy_version: str
    feature_snapshot_id: str
    recommended_action: str
    trading_horizon: str
    confidence: float              # [0, 1]
    evidence_id: Optional[str]
    reason_codes: list = field(default_factory=list)
    invalidation_conditions: list = field(default_factory=list)
    expiration_timestamp: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


class SchemaError(ValueError):
    pass


def validate(rec: Recommendation, now: Optional[datetime] = None) -> Recommendation:
    """Strict validation. Raises SchemaError on any violation; returns rec unchanged if valid."""
    now = now or datetime.now(timezone.utc)
    if not rec.agent_id or not rec.recommendation_id:
        raise SchemaError("missing agent_id/recommendation_id")
    if not rec.symbol:
        raise SchemaError("missing symbol")
    if rec.recommended_action not in VALID_ACTIONS:
        raise SchemaError(f"unsupported action {rec.recommended_action!r}")
    if rec.trading_horizon not in VALID_HORIZONS:
        raise SchemaError(f"unsupported horizon {rec.trading_horizon!r}")
    if not (0.0 <= float(rec.confidence) <= 1.0):
        raise SchemaError(f"confidence out of range: {rec.confidence}")
    # timestamps parse?
    try:
        datetime.fromisoformat(rec.timestamp.replace("Z", "+00:00"))
    except Exception as e:
        raise SchemaError(f"bad timestamp: {e}")
    if rec.expiration_timestamp:
        try:
            exp = datetime.fromisoformat(rec.expiration_timestamp.replace("Z", "+00:00"))
        except Exception as e:
            raise SchemaError(f"bad expiration_timestamp: {e}")
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if exp < now:
            raise SchemaError("recommendation expired")
    # an actionable entry/promote must cite evidence
    if rec.recommended_action in ("ENTER_LONG", "PROMOTE") and not rec.evidence_id:
        raise SchemaError(f"{rec.recommended_action} requires evidence_id")
    return rec


def is_valid(rec: Recommendation, now: Optional[datetime] = None) -> bool:
    try:
        validate(rec, now)
        return True
    except SchemaError:
        return False


class SpecialistAgent:
    """Common base. Subclasses implement analyze() and return a (validated) Recommendation or None.
    Deterministic by default; `llm_backed` stays False unless a flag enables it."""
    agent_id: str = "base"
    llm_backed: bool = False

    def analyze(self, context: dict) -> Optional[Recommendation]:  # pragma: no cover - interface
        raise NotImplementedError

    def safe_analyze(self, context: dict) -> Optional[Recommendation]:
        """Run analyze(); on any failure return None (fail safe -- never crash the pipeline, never
        emit an unvalidated recommendation)."""
        try:
            rec = self.analyze(context)
        except Exception:
            return None
        if rec is None:
            return None
        return rec if is_valid(rec) else None
