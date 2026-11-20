"""Integrated AMPS V6 pipeline (V6-Integrated spec sections 2 & 16).

Wires the components in the mandated order:

  Market Data -> Validation -> Multi-TF Features -> Regime Engine -> Specialist Policy Agents ->
  Candidate Signal Engine -> Entry-Alpha Agent -> Event-Driven Allocator ->
  Independent Risk & Exit Stack -> Execution Gateway -> Broker -> Position Manager -> Monitoring.

Two hard invariants enforced here:
  1. ENTRY conjunction (section 16) -- a new entry needs EVERY gate; no majority voting.
  2. EXITS are always available (section 13) -- position management runs regardless of entry gates.

Default posture is OBSERVE_ONLY (spec section 19): the pipeline computes intents + full decision
traces (model intent vs system action) but does NOT submit orders. The execution gateway/broker and
streaming manager are referenced as staged reuse (alpaca_executor.py / trade.py / streaming_engine.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from amps_v6.regime import regime_engine as RE
from amps_v6.alpha import entry_alpha as EA
from amps_v6.alpha.evidence_registry import EvidenceRegistry
from amps_v6.strategy import allocator as ALLOC
from amps_v6.execution import risk_gateway as RG
from amps_v6.execution.order_intent import OrderIntent
from amps_v6.risk import exit_stack as XS
from amps_v6.monitoring.decision_trace import DecisionTrace


@dataclass
class EntryGateInputs:
    trusted_current_data: bool
    regime_locked_and_eligible: bool
    eligible_symbol: bool
    valid_policy_setup: bool
    entry_alpha_decision: str            # must == "ALPHA_CONFIRMED"
    adversarial_veto: bool
    security_veto: bool
    allocation_above_minimum: bool
    deterministic_risk_approval: bool


def entry_authorized(g: EntryGateInputs) -> tuple[bool, list[str]]:
    """Section 16 conjunction. Returns (authorized, failed_gates). NOT a vote -- all must pass."""
    checks = {
        "trusted_current_data": g.trusted_current_data,
        "locked_eligible_regime": g.regime_locked_and_eligible,
        "eligible_symbol": g.eligible_symbol,
        "valid_policy_setup": g.valid_policy_setup,
        "alpha_confirmed": g.entry_alpha_decision == EA.ALPHA_CONFIRMED,
        "no_adversarial_veto": not g.adversarial_veto,
        "no_security_veto": not g.security_veto,
        "allocation_above_minimum": g.allocation_above_minimum,
        "deterministic_risk_approval": g.deterministic_risk_approval,
    }
    failed = [k for k, ok in checks.items() if not ok]
    return (len(failed) == 0, failed)


@dataclass
class PipelineConfig:
    observe_only: bool = True            # spec section 19 default; never submits when True
    min_allocation_weight: float = 0.01
    alloc: ALLOC.AllocConfig = None
    risk_limits: RG.RiskLimits = None

    def __post_init__(self):
        if self.alloc is None:
            self.alloc = ALLOC.AllocConfig()
        if self.risk_limits is None:
            self.risk_limits = RG.RiskLimits()


def evaluate_entry(ctx: dict, regime_state: RE.RegimeState, perms: dict,
                   evidence: EvidenceRegistry, platform_state, cfg: PipelineConfig,
                   mode: str = "OBSERVE_ONLY") -> DecisionTrace:
    """Run one candidate through regime->policy->alpha->allocator->risk and produce a DecisionTrace.
    Never submits. Returns the trace with decision in {ENTRY_AUTHORIZED, OBSERVE_ONLY, NO_TRADE,
    BLOCKED:<gates>}."""
    sym = ctx.get("symbol", "?")
    tr = DecisionTrace(symbol=sym, market_regime=regime_state.regime,
                       regime_lock_status=regime_state.lock_status,
                       data_quality=ctx.get("data_quality", "UNKNOWN"),
                       security_state=ctx.get("security_state", "OK"))

    # regime eligibility: must be locked AND permit this side's universe
    orientation = ctx.get("orientation", "bull")
    universe_ok = perms.get("inverse_universe", False) if orientation == "bear" else perms.get("long_universe", False)
    regime_ok = regime_state.is_locked and perms.get("new_entries_allowed", False) and universe_ok
    tr.eligibility = "regime_eligible" if regime_ok else f"regime_blocked({regime_state.lock_status})"

    # policy agents -> effective signal (advisory)
    raw = ctx.get("agent_outputs", {})
    tr.raw_agent_output = raw
    valid_setup = any(v == "ENTER_LONG" for v in raw.values()) if raw else bool(ctx.get("valid_setup", False))
    tr.effective_signal = "ENTER_LONG" if valid_setup else "NO_SETUP"

    # entry-alpha gate
    eid = ctx.get("evidence_id")
    ev = evidence.get(eid) if eid else None
    alpha = (EA.ALPHA_CONFIRMED if (ev and ev.classification == "VALIDATED"
             and evidence.permits_entry(eid, mode)[0]) else ctx.get("entry_alpha_decision", EA.NO_ALPHA_EVIDENCE))
    tr.entry_alpha_decision = alpha
    tr.entry_alpha_evidence_id = eid

    # allocation (only matters if alpha confirmed; otherwise target flat)
    cand = ALLOC.Candidate(symbol=sym, direction=orientation,
                           alpha_confirmed=(alpha == EA.ALPHA_CONFIRMED),
                           alpha_confidence=ctx.get("alpha_confidence", 0.6),
                           regime_confidence=regime_state.last_confidence,
                           volatility=ctx.get("volatility", 0.0),
                           current_weight=ctx.get("current_weight", 0.0))
    alloc = ALLOC.allocate([cand], cfg.alloc)[0]
    tr.requested_target_weight = alloc.target_weight
    tr.allocator_approved_weight = alloc.approved_weight
    tr.clamp_reasons = alloc.clamp_reasons
    allocation_ok = alloc.approved_weight >= cfg.min_allocation_weight

    # deterministic risk approval (sizing/kill-switch via the independent gateway)
    price = ctx.get("price", 0.0)
    intent = OrderIntent(symbol=sym, side="buy", qty=0, reason="amps_v6_integrated",
                         approved_trading_horizon=ctx.get("horizon", "INTRADAY_MOMENTUM"),
                         protective_stop=price * 0.98 if price else None,
                         effective_leverage=ctx.get("effective_leverage", 1.0),
                         overnight_permission=ctx.get("overnight_permission", False))
    pctx = RG.PortfolioContext(equity=ctx.get("equity", 0.0), buying_power=ctx.get("buying_power", 0.0),
                               open_positions=ctx.get("open_positions", 0),
                               opposing_held=tuple(ctx.get("opposing_held", ())))
    rdec = RG.evaluate(intent, price, pctx, platform_state, cfg.risk_limits) if price > 0 else RG.RiskDecision(False, "no_price")
    tr.risk_approved_weight = (rdec.sized_notional / ctx["equity"]) if (rdec.approved and ctx.get("equity")) else 0.0

    g = EntryGateInputs(
        trusted_current_data=(ctx.get("data_quality", "UNKNOWN") == "OK"),
        regime_locked_and_eligible=regime_ok,
        eligible_symbol=bool(ctx.get("eligible_symbol", regime_ok)),
        valid_policy_setup=valid_setup,
        entry_alpha_decision=alpha,
        adversarial_veto=ctx.get("adversarial_veto", False),
        security_veto=ctx.get("security_veto", False),
        allocation_above_minimum=allocation_ok,
        deterministic_risk_approval=rdec.approved)
    authorized, failed = entry_authorized(g)

    if not authorized:
        # Distinguish a clean NO_TRADE (no alpha / no setup / no allocation -- nothing warrants an
        # entry) from a safety BLOCK (data/regime/eligibility/veto/risk failure). A missing
        # allocation is a *consequence* of no alpha, so it is soft, not a block.
        hard_gates = {"trusted_current_data", "locked_eligible_regime", "eligible_symbol",
                      "no_adversarial_veto", "no_security_veto", "deterministic_risk_approval"}
        hard_failed = [f for f in failed if f in hard_gates]
        tr.final_execution_weight = 0.0
        if hard_failed:
            tr.decision = "BLOCKED:" + ",".join(hard_failed)
            tr.blocked_gates = hard_failed
        else:
            tr.decision = "NO_TRADE"
            tr.blocked_gates = failed
        return tr

    # authorized: in OBSERVE_ONLY we record the intent but do NOT submit
    tr.final_execution_weight = min(alloc.approved_weight, tr.risk_approved_weight or alloc.approved_weight)
    tr.order_intent = intent.as_dict()
    tr.decision = "OBSERVE_ONLY" if cfg.observe_only else "ENTRY_AUTHORIZED"
    return tr


def manage_position(ctx: dict, exit_cfg: Optional[XS.ExitConfig] = None) -> DecisionTrace:
    """Exit path -- ALWAYS available, independent of entry gates/regime/agent availability."""
    sym = ctx.get("symbol", "?")
    tr = DecisionTrace(symbol=sym, market_regime=ctx.get("regime", ""))
    xctx = XS.ExitContext(
        price=ctx.get("price", 0.0), avg_entry=ctx.get("avg_entry", 0.0), atr=ctx.get("atr", 0.0),
        high_since_entry=ctx.get("high_since_entry", 0.0), bars_held=ctx.get("bars_held", 0),
        regime_permits_hold=ctx.get("regime_permits_hold", True), thesis_valid=ctx.get("thesis_valid", True),
        reduce_only=ctx.get("reduce_only", False), end_of_day=ctx.get("end_of_day", False),
        account_daily_pnl_pct=ctx.get("account_daily_pnl_pct", 0.0),
        account_drawdown_pct=ctx.get("account_drawdown_pct", 0.0), emergency=ctx.get("emergency", False))
    xd = XS.evaluate(xctx, exit_cfg)
    tr.exit_stack_reason = xd.reason
    tr.decision = f"EXIT_STACK:{xd.action}"
    return tr
