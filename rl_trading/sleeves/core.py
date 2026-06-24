"""Core sleeve interface + validation gate.

Design principle: the gate is the authority. A sleeve does not decide whether it may trade — its
attached ValidationRecord must independently clear EVERY gate criterion. A missing or partial
record => SCANNER_ONLY. Live is never an option (no field can turn it on).
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence, Tuple


class SleeveMode(str, enum.Enum):
    DISABLED = "disabled"          # no validation record at all
    SCANNER_ONLY = "scanner_only"  # has a record but it does not clear the gate, or paper not enabled
    PAPER = "paper"                # gate cleared AND paper explicitly enabled (live still impossible)
    # NOTE: there is intentionally no LIVE mode.


@dataclass(frozen=True)
class ValidationRecord:
    """Evidence that an information source cleared the standing gate.

    Each gate field is tri-state: True (passed), False (failed/tested-and-rejected), or None
    (NOT TESTED). None is treated as a failure by the gate — absence of evidence is not evidence.
    """
    source: str
    ledger_id: Optional[str] = None
    # bootstrap CI on the edge metric (e.g. excess return vs SPY). Must be present AND lo > 0.
    bootstrap_ci: Optional[Tuple[float, float]] = None
    placebo_passed: Optional[bool] = None        # beat its own random-shifted-date placebo
    oos_passed: Optional[bool] = None            # held out-of-sample (walk-forward)
    ood_passed: Optional[bool] = None            # held out-of-domain (years/regimes/caps/sectors/...)
    attribution_passed: Optional[bool] = None    # PnL attributable to the claimed signal
    concentration_passed: Optional[bool] = None  # not carried by one name/year/domain
    metric: str = "excess_return_vs_spy"
    value: Optional[float] = None
    notes: str = ""


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: Tuple[str, ...]
    checked: Tuple[str, ...]

    @property
    def next_step(self) -> str:
        """The single highest-priority validation step still required (for scanner output)."""
        order = [
            "no_validation_record", "bootstrap_ci_missing", "bootstrap_ci_not_positive",
            "placebo_not_passed", "oos_not_passed", "ood_not_passed",
            "attribution_not_passed", "concentration_not_passed",
        ]
        for key in order:
            if key in self.failures:
                return key
        return "none"


# Ordered gate criteria. Each maps a ValidationRecord to (ok, failure_key).
def _gate_bootstrap(r: ValidationRecord):
    if r.bootstrap_ci is None:
        return False, "bootstrap_ci_missing"
    lo, _hi = r.bootstrap_ci
    return (lo > 0.0), ("bootstrap_ci_not_positive" if lo <= 0.0 else "")


def validation_gate(record: Optional[ValidationRecord]) -> GateResult:
    """Pure, deterministic gate. Passes ONLY if every criterion is satisfied."""
    checked = (
        "bootstrap_ci_positive", "placebo_passed", "oos_passed", "ood_passed",
        "attribution_passed", "concentration_passed",
    )
    if record is None:
        return GateResult(False, ("no_validation_record",), checked)

    failures: List[str] = []
    ok, key = _gate_bootstrap(record)
    if not ok:
        failures.append(key)
    if record.placebo_passed is not True:
        failures.append("placebo_not_passed")
    if record.oos_passed is not True:
        failures.append("oos_not_passed")
    if record.ood_passed is not True:
        failures.append("ood_not_passed")
    if record.attribution_passed is not True:
        failures.append("attribution_not_passed")
    if record.concentration_passed is not True:
        failures.append("concentration_not_passed")

    return GateResult(len(failures) == 0, tuple(failures), checked)


@dataclass(frozen=True)
class RiskLimits:
    """Hard paper-trading caps. All in account currency unless noted."""
    max_notional: float = 10_000.0       # total deployable paper notional across this sleeve
    per_order_notional: float = 2_000.0  # cap on any single order
    max_position_pct: float = 0.10       # per-symbol cap as a fraction of equity
    max_gross_exposure: float = 1.0      # gross exposure cap as a fraction of equity
    max_concurrent: int = 10             # max concurrent open positions
    max_daily_loss: float = 500.0        # realized+unrealized daily loss kill threshold (positive $)

    def __post_init__(self):
        for f in ("max_notional", "per_order_notional", "max_position_pct",
                  "max_gross_exposure", "max_daily_loss"):
            if getattr(self, f) < 0:
                raise ValueError(f"{f} must be >= 0")
        if self.max_concurrent < 0:
            raise ValueError("max_concurrent must be >= 0")


@dataclass
class Sleeve:
    """A paper-tradeable strategy unit, tradeable only after its source clears the gate."""
    name: str
    signal_source: str
    native_execution_type: str           # 'trend_breakout' | 'event_hold' | 'timing_overlay' | ...
    eligible_symbols: Sequence[str]
    entry_rules: Mapping[str, object]
    exit_rules: Mapping[str, object]
    sizing_rules: Mapping[str, object]
    risk_limits: RiskLimits
    required_validation: Optional[ValidationRecord] = None
    paper_enabled: bool = False
    # Live is a hard, immutable False. Not an __init__ argument — cannot be set by callers.
    live_enabled: bool = field(default=False, init=False)

    def gate(self) -> GateResult:
        return validation_gate(self.required_validation)

    def mode(self) -> SleeveMode:
        if self.required_validation is None:
            return SleeveMode.DISABLED
        if self.gate().passed and self.paper_enabled:
            return SleeveMode.PAPER
        return SleeveMode.SCANNER_ONLY

    def can_trade(self) -> bool:
        """True only if validated AND paper explicitly enabled. Live is irrelevant — always off."""
        return bool(self.gate().passed and self.paper_enabled and not self.live_enabled)

    def why_not_tradeable(self) -> str:
        if self.can_trade():
            return ""
        g = self.gate()
        if not g.passed:
            return "gate_failed:" + ",".join(g.failures)
        if not self.paper_enabled:
            return "paper_not_enabled"
        return "unknown"

    def scan(self, candidates: Sequence[Mapping[str, object]]) -> List[dict]:
        """Scanner output — ALWAYS available, even when trading is blocked.

        Each candidate dict should carry at least: symbol, ts (PIT timestamp), strength.
        We never gate the scanner; we only annotate why it is not (yet) tradeable.
        """
        g = self.gate()
        reason = "tradeable" if self.can_trade() else self.why_not_tradeable()
        horizon = self.entry_rules.get("native_horizon_days", self.exit_rules.get("horizon_days"))
        out: List[dict] = []
        for c in candidates:
            out.append({
                "sleeve": self.name,
                "symbol": c.get("symbol"),
                "pit_timestamp": c.get("ts"),
                "signal_strength": c.get("strength"),
                "proposed_native_horizon_days": horizon,
                "tradeable": self.can_trade(),
                "reason_not_tradeable": reason,
                "required_validation_next_step": g.next_step,
            })
        return out
