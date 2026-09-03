"""rl_trading.sleeves — paper-only, validation-gated sleeve framework.

A sleeve may submit (paper) orders ONLY after its information source has cleared the standing
gate: PIT -> event/IC study -> bootstrap + placebo -> OOS + OOD -> attribution ->
not-concentrated. Until then it is SCANNER-ONLY. LIVE TRADING IS PERMANENTLY BLOCKED.

Honest invariant (enforced in code, not just docs): no sleeve currently has a passing validation
record, so nothing can trade — exactly the truth of the research program.
"""
from .core import (
    GateResult,
    RiskLimits,
    Sleeve,
    SleeveMode,
    ValidationRecord,
    validation_gate,
)
from .paper import Order, PaperExecutor
from .promote import (
    CANONICAL_EXTENDED_VALIDATION,
    CANONICAL_STATUS,
    PROVENANCE_CAVEAT,
    from_event_study,
    from_extended_validation,
)
from .registry import SleeveRegistry, render_markdown, status_report

__all__ = [
    "GateResult", "RiskLimits", "Sleeve", "SleeveMode", "ValidationRecord", "validation_gate",
    "Order", "PaperExecutor", "SleeveRegistry", "status_report", "render_markdown",
    "from_event_study", "from_extended_validation", "CANONICAL_EXTENDED_VALIDATION",
    "CANONICAL_STATUS", "PROVENANCE_CAVEAT",
]
