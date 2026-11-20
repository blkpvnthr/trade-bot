"""Decision-to-execution traceability (V6-Integrated spec section 18; both papers' observability).

One record per candidate decision captures the FULL path: raw agent output -> effective signal ->
regime -> eligibility -> entry-alpha -> requested/allocator-approved/risk-approved/final weight ->
clamp reasons -> order intent -> broker id -> fills -> exit reason -> position/pnl -> data/security
state. This is the auditable distinction between model INTENT and system ACTION.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_PATH = PROJECT_ROOT / "logs" / "execution" / "amps_v6_decision_trace.jsonl"


@dataclass
class DecisionTrace:
    symbol: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # context
    market_regime: str = ""
    regime_lock_status: str = ""
    eligibility: str = ""
    # agent / signal
    raw_agent_output: dict = field(default_factory=dict)
    effective_signal: str = ""
    # entry alpha
    entry_alpha_decision: str = ""
    entry_alpha_evidence_id: Optional[str] = None
    # weights across the pipeline (intent vs action)
    requested_target_weight: Optional[float] = None
    allocator_approved_weight: Optional[float] = None
    risk_approved_weight: Optional[float] = None
    final_execution_weight: Optional[float] = None
    clamp_reasons: list = field(default_factory=list)
    # execution
    order_intent: dict = field(default_factory=dict)
    broker_order_id: Optional[str] = None
    fill_events: list = field(default_factory=list)
    # exits / position / health
    exit_stack_reason: str = ""
    position_state: dict = field(default_factory=dict)
    pnl: Optional[float] = None
    data_quality: str = ""
    security_state: str = ""
    # final outcome
    decision: str = ""          # e.g. ENTRY_AUTHORIZED | NO_TRADE | BLOCKED:<gate> | OBSERVE_ONLY
    blocked_gates: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def append(trace: DecisionTrace, path: Optional[Path] = None) -> None:
    p = path or TRACE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(trace.as_dict(), default=str) + "\n")
