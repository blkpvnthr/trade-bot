"""Deterministic specialist agents (V6 spec section 4) wrapping already-validated repo logic.

These are the latency-safe defaults: no LLM, no network. They emit strictly-validated
Recommendations. LLM-backed variants (research/regime-narrative/adversarial review) would subclass
SpecialistAgent behind a feature flag and are intentionally NOT required here.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

import amps_v4_regime as R
from amps_v6.agents.base import SpecialistAgent, Recommendation
from amps_v6.alpha.evidence_registry import EvidenceRegistry, LIVE_SUSPENDED, REVALIDATION_REQUIRED


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MarketRegimeAgent(SpecialistAgent):
    """Classify broad-market regime from reference rows using the validated amps_v4_regime model.
    context: {ref_rows: {SPY:{...},...}, feature_snapshot_id, data_version}."""
    agent_id = "market_regime_agent"

    def __init__(self, cfg: Optional[R.RegimeConfig] = None):
        self.cfg = cfg or R.RegimeConfig()

    def analyze(self, context: dict) -> Optional[Recommendation]:
        ref_rows = context.get("ref_rows") or {}
        label, score, conf = R.market_regime(ref_rows, self.cfg)
        return Recommendation(
            agent_id=self.agent_id, recommendation_id=f"{self.agent_id}:{_now_iso()}",
            symbol="_MARKET_", underlying="_MARKET_", timestamp=_now_iso(),
            data_version=context.get("data_version", "unknown"),
            strategy_version="amps_v6", feature_snapshot_id=context.get("feature_snapshot_id", "na"),
            recommended_action="HOLD", trading_horizon="NONE",
            confidence=float(conf), evidence_id=None,
            reason_codes=[f"regime={label}", f"score={score}"],
            invalidation_conditions=["regime flip"], expiration_timestamp=None)


class EntryAlphaAgent(SpecialistAgent):
    """The mandatory entry gate as an agent. Given a candidate's (setup_family, horizon,
    instrument_class), look up evidence and emit ENTER_LONG only on VALIDATED/LIVE_APPROVED evidence;
    otherwise VETO. No other agent may override an alpha rejection (enforced by decision_fusion)."""
    agent_id = "entry_alpha_agent"

    def __init__(self, registry: EvidenceRegistry, mode: str = "OBSERVE_ONLY"):
        self.registry = registry
        self.mode = mode

    def analyze(self, context: dict) -> Optional[Recommendation]:
        symbol = context["symbol"]
        underlying = context.get("underlying", symbol)
        evidence_id = context.get("evidence_id")
        horizon = context.get("trading_horizon", "NONE")
        ev = self.registry.get(evidence_id) if evidence_id else None
        permit, reason = (self.registry.permits_entry(evidence_id, self.mode)
                          if evidence_id else (False, "no_evidence_id"))
        validated = bool(ev and ev.classification == "VALIDATED")
        action = "ENTER_LONG" if (permit and validated) else "VETO"
        # VETO carries no evidence_id requirement; ENTER_LONG must cite evidence (schema-enforced)
        return Recommendation(
            agent_id=self.agent_id, recommendation_id=f"{self.agent_id}:{symbol}:{_now_iso()}",
            symbol=symbol, underlying=underlying, timestamp=_now_iso(),
            data_version=context.get("data_version", "unknown"), strategy_version="amps_v6",
            feature_snapshot_id=context.get("feature_snapshot_id", "na"),
            recommended_action=action, trading_horizon=horizon if action == "ENTER_LONG" else "NONE",
            confidence=0.8 if action == "ENTER_LONG" else 0.0,
            evidence_id=evidence_id if action == "ENTER_LONG" else None,
            reason_codes=[reason, f"classification={getattr(ev,'classification',None)}"],
            invalidation_conditions=["evidence suspended", "out of distribution"],
            expiration_timestamp=(datetime.now(timezone.utc) + timedelta(hours=6)).isoformat())
