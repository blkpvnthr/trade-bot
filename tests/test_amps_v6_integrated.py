"""Tests for the AMPS V6 Integrated regime-aware agentic architecture (spec section 21).

Covers: regime classification + lock state machine + permissions, event-driven allocation, signed
weight clamping (increase AND reduction), inverse/hedge conflicts, new-entry blocking with exits
still available, order state machine (transitions/idempotency/ambiguity/resubmit guard), canonical
position manager (fills/exposure), exit stack (all reasons + account dominance), the section-16 entry
conjunction, policy-agent contracts, canonical-state events, decision trace, NO_TRADE/NO_ALPHA.

Pure/offline. No broker, no network.
"""
from __future__ import annotations

import pytest

from amps_v6.regime import regime_engine as RE
from amps_v6.strategy import allocator as AL
from amps_v6.execution import order_state_machine as OSM
from amps_v6.execution.position_manager import PositionManager, Position
from amps_v6.risk import exit_stack as XS
from amps_v6.state import canonical_state as CS
from amps_v6.agents import policy_agents as PA
from amps_v6.agents import base as AB
from amps_v6 import pipeline as PIPE
from amps_v6.configuration import platform_state as PS
from amps_v6.alpha import entry_alpha as EA
from amps_v6.alpha.evidence_registry import EvidenceRegistry, Evidence
from amps_v6.monitoring import decision_trace as DT


# ---------------------------------------------------------------- regime classification
def _proxy(pdi, mdi, adx=25, vwap_dist=0.001, short_mom=0.1, rv=0.001):
    return {"pdi": pdi, "mdi": mdi, "adx": adx, "vwap_dist": vwap_dist,
            "vwap_slope": vwap_dist, "bb_dist": vwap_dist, "short_mom": short_mom, "realized_vol": rv}


def test_regime_bull_bear_transition_crisis_indeterminate():
    cfg = RE.RegimeConfig()
    bull = {p: _proxy(30, 5, vwap_dist=0.01, short_mom=1) for p in RE.BREADTH_PROXIES}
    assert RE.classify(bull, cfg).regime == RE.BULL
    bear = {p: _proxy(5, 30, vwap_dist=-0.01, short_mom=-1) for p in RE.BREADTH_PROXIES}
    assert RE.classify(bear, cfg).regime == RE.BEAR
    crisis = {p: _proxy(5, 30, rv=0.02) for p in RE.BREADTH_PROXIES}
    assert RE.classify(crisis, cfg).regime == RE.CRISIS   # vol dominates
    thin = {"SPY": _proxy(30, 5)}
    assert RE.classify(thin, cfg).regime == RE.INDETERMINATE   # < min_proxies
    mixed = {"SPY": _proxy(30, 5, vwap_dist=0.01), "QQQ": _proxy(5, 30, vwap_dist=-0.01),
             "DIA": _proxy(30, 5, vwap_dist=0.01), "IWM": _proxy(5, 30, vwap_dist=-0.01)}
    assert RE.classify(mixed, cfg).regime == RE.TRANSITION   # no agreement


def test_regime_permissions():
    assert RE.permissions(RE.BULL)["long_universe"] and not RE.permissions(RE.BULL)["inverse_universe"]
    assert RE.permissions(RE.BEAR)["inverse_universe"] and not RE.permissions(RE.BEAR)["long_universe"]
    assert not RE.permissions(RE.TRANSITION)["new_entries_allowed"]
    assert RE.permissions(RE.CRISIS)["reduce_only"] and not RE.permissions(RE.CRISIS)["new_entries_allowed"]
    assert not RE.permissions(RE.INDETERMINATE)["new_entries_allowed"]
    assert RE.permissions(RE.BULL, hedge_mode=True)["inverse_universe"]   # hedge override


def test_regime_fast_lock_and_confirmed_lock():
    cfg = RE.RegimeConfig()
    st = RE.RegimeState()
    hi = RE.RegimeAssessment(RE.BULL, 40.0, 0.9, 1.0, 0.001, "x")   # high confidence
    st.update(hi, in_opening_window=True, cfg=cfg)
    assert st.lock_status == RE.FAST_LOCK and st.is_locked
    # confirmed lock path: after window, persistent moderate confidence
    st2 = RE.RegimeState()
    mod = RE.RegimeAssessment(RE.BULL, 30.0, 0.6, 0.75, 0.001, "x")
    st2.update(mod, in_opening_window=False, cfg=cfg)   # tick 1 -> PRELIMINARY
    st2.update(mod, in_opening_window=False, cfg=cfg)   # tick 2 -> CONFIRMED
    assert st2.lock_status == RE.CONFIRMED_LOCK


def test_regime_deterioration_and_reversal():
    cfg = RE.RegimeConfig()
    st = RE.RegimeState(lock_status=RE.CONFIRMED_LOCK, regime=RE.BULL, candidate=RE.BULL)
    st.update(RE.RegimeAssessment(RE.BULL, 5.0, 0.1, 0.5, 0.001, "x"), in_opening_window=False, cfg=cfg)
    assert st.lock_status == RE.UNLOCKED and st.lock_reason == "deterioration"
    st2 = RE.RegimeState(lock_status=RE.CONFIRMED_LOCK, regime=RE.BULL, candidate=RE.BULL)
    st2.update(RE.RegimeAssessment(RE.BEAR, -40.0, 0.8, 0.0, 0.001, "x"), in_opening_window=False, cfg=cfg)
    assert st2.lock_status == RE.PRELIMINARY and "reversal" in st2.lock_reason


def test_crisis_takes_over_immediately():
    st = RE.RegimeState(lock_status=RE.CONFIRMED_LOCK, regime=RE.BULL)
    st.update(RE.RegimeAssessment(RE.CRISIS, 0.0, 0.9, 0.0, 0.02, "x"), in_opening_window=False)
    assert st.regime == RE.CRISIS and st.lock_status == RE.CONFIRMED_LOCK


# ---------------------------------------------------------------- signed weight clamping (key)
def test_clamp_weight_increase_capped_by_step():
    approved, reasons = AL.clamp_weight(0.0, 0.5, step_cap=0.1, minimum_weight=0.0, maximum_weight=0.2)
    assert approved == pytest.approx(0.1) and any("step_cap" in r for r in reasons)


def test_clamp_weight_reduction_is_correct_not_min():
    # reducing 0.15 -> 0.0: a naive min(target-cur, step, max-cur)=min(-0.15,0.1,0.05)=-0.15 would be
    # unclamped by step; the signed clip must cap the DOWN move to -step_cap too.
    approved, reasons = AL.clamp_weight(0.15, 0.0, step_cap=0.05, minimum_weight=0.0, maximum_weight=0.2)
    assert approved == pytest.approx(0.10)   # 0.15 - 0.05 step-capped reduction
    assert any("down" in r for r in reasons)


def test_clamp_weight_respects_bounds():
    approved, _ = AL.clamp_weight(0.18, 0.30, step_cap=0.10, minimum_weight=0.0, maximum_weight=0.2)
    assert approved == pytest.approx(0.2)   # bounded at max even though step would allow 0.28


def test_adaptive_exposure_scale_bounds():
    cfg = AL.AllocConfig()
    assert AL.adaptive_exposure_scale(0.0, 0.0, cfg) == cfg.min_exposure_scale
    assert AL.adaptive_exposure_scale(1.0, 1.0, cfg) == cfg.max_exposure_scale


def test_allocate_unconfirmed_targets_flat():
    cfg = AL.AllocConfig()
    cands = [AL.Candidate("TQQQ", "long", alpha_confirmed=False, alpha_confidence=0.9,
                          regime_confidence=0.9, current_weight=0.12)]
    a = AL.allocate(cands, cfg)[0]
    assert a.target_weight == 0.0 and a.approved_weight < 0.12   # reduces toward 0


def test_allocate_gross_cap_scales_down():
    cfg = AL.AllocConfig(max_gross=0.15, base_target=0.2, max_weight=0.2, step_cap=1.0)
    cands = [AL.Candidate(f"S{i}", "long", True, 1.0, 1.0) for i in range(3)]
    allocs = AL.allocate(cands, cfg)
    assert sum(a.approved_weight for a in allocs) <= cfg.max_gross + 1e-6


# ---------------------------------------------------------------- event-driven allocation
def test_diff_events_detects_reallocation_triggers():
    prev = CS.CanonicalState(tradable_universe=["TQQQ"], market_regime="BULL",
                             regime_lock_status="CONFIRMED_LOCK")
    curr = CS.CanonicalState(tradable_universe=["TQQQ", "SOXL"], market_regime="BEAR",
                             regime_lock_status="CONFIRMED_LOCK")
    ev = CS.diff_events(prev, curr)
    assert CS.EV_REGIME_CHANGE in ev and CS.EV_ELIGIBLE_SET_CHANGE in ev


def test_diff_events_quiet_when_nothing_changes():
    s = CS.CanonicalState(tradable_universe=["TQQQ"], market_regime="BULL")
    assert CS.diff_events(s, CS.CanonicalState(tradable_universe=["TQQQ"], market_regime="BULL")) == []


def test_canonical_state_roundtrip(tmp_path):
    s = CS.CanonicalState(market_regime="BEAR", reduce_only=True, buying_power=1234.0)
    p = tmp_path / "cs.json"; s.save(p)
    assert CS.CanonicalState.load(p).market_regime == "BEAR" and CS.CanonicalState.load(p).reduce_only


# ---------------------------------------------------------------- order state machine
def test_osm_valid_and_invalid_transitions():
    m = OSM.OrderStateMachine()
    m.create("c1", "TQQQ", "buy", 10)
    assert m.advance("c1", OSM.OrderState.VALIDATED)
    assert m.advance("c1", OSM.OrderState.RISK_APPROVED)
    assert not m.advance("c1", OSM.OrderState.FILLED)   # illegal jump
    assert m.advance("c1", OSM.OrderState.SUBMITTING)


def test_osm_idempotent_create():
    m = OSM.OrderStateMachine()
    a = m.create("c1", "TQQQ", "buy", 10)
    m.advance("c1", OSM.OrderState.VALIDATED)
    b = m.create("c1", "TQQQ", "buy", 10)   # same cid -> same record, not reset
    assert a is b and b.state == OSM.OrderState.VALIDATED.value


def test_osm_ambiguity_and_resubmit_guard():
    m = OSM.OrderStateMachine()
    m.create("c1", "TQQQ", "buy", 10)
    m.advance("c1", OSM.OrderState.VALIDATED); m.advance("c1", OSM.OrderState.RISK_APPROVED)
    assert m.resubmittable("c1")
    m.advance("c1", OSM.OrderState.SUBMITTING)
    assert not m.resubmittable("c1")                 # never resubmit a submitting order
    m.advance("c1", OSM.OrderState.RECONCILIATION_REQUIRED)
    assert "c1" in m.needs_reconciliation()
    assert not m.resubmittable("c1")


def test_osm_persistence(tmp_path):
    m = OSM.OrderStateMachine(); m.create("c1", "X", "buy", 5)
    m.advance("c1", OSM.OrderState.VALIDATED)
    p = tmp_path / "osm.json"; m.save(p)
    assert OSM.OrderStateMachine.load(p).get("c1").state == OSM.OrderState.VALIDATED.value


# ---------------------------------------------------------------- position manager
def test_position_manager_buy_average_and_sell_realize():
    pm = PositionManager()
    pm.apply_fill("TQQQ", 10, 100.0, "buy", regime="BULL", evidence_id="ev1")
    pm.apply_fill("TQQQ", 10, 110.0, "buy")
    p = pm.get("TQQQ")
    assert p.qty == 20 and p.avg_entry == pytest.approx(105.0)
    assert p.regime_at_entry == "BULL" and p.entry_alpha_evidence_id == "ev1"
    pm.apply_fill("TQQQ", 20, 120.0, "sell")
    assert pm.get("TQQQ").qty == 0 and pm.get("TQQQ").realized_pnl == pytest.approx((120 - 105) * 20)


def test_position_manager_exposure_and_helpers():
    pm = PositionManager()
    pm.apply_fill("SOXL", 5, 40.0, "buy", effective_leverage=3.0)
    p = pm.get("SOXL")
    assert p.effective_exposure(40.0) == pytest.approx(5 * 40 * 3)
    assert pm.effective_exposure({"SOXL": 40.0}) == pytest.approx(600.0)


# ---------------------------------------------------------------- exit stack
def test_exit_stack_each_reason():
    assert XS.evaluate(XS.ExitContext(price=94, avg_entry=100)).reason == "HARD_STOP"
    assert XS.evaluate(XS.ExitContext(price=108.1, avg_entry=100)).reason == "PROFIT_TARGET"
    assert XS.evaluate(XS.ExitContext(price=100, avg_entry=100, high_since_entry=105)).reason == "TRAILING_STOP"
    assert XS.evaluate(XS.ExitContext(price=100, avg_entry=100, thesis_valid=False)).reason == "THESIS_INVALIDATION"
    assert XS.evaluate(XS.ExitContext(price=100, avg_entry=100, regime_permits_hold=False)).reason == "REGIME_EXIT"
    assert XS.evaluate(XS.ExitContext(price=100, avg_entry=100, end_of_day=True)).reason == "END_OF_DAY"
    assert XS.evaluate(XS.ExitContext(price=100, avg_entry=100, reduce_only=True)).action == XS.REDUCE


def test_exit_stack_account_level_dominates():
    # even a profitable position is flattened on a daily-loss shutdown
    d = XS.evaluate(XS.ExitContext(price=109, avg_entry=100, account_daily_pnl_pct=-0.05))
    assert d.reason == "DAILY_LOSS_SHUTDOWN" and d.action == XS.EXIT
    assert XS.evaluate(XS.ExitContext(price=109, avg_entry=100, emergency=True)).reason == "EMERGENCY_LIQUIDATION"


def test_exits_available_even_when_entries_blocked():
    assert XS.entries_blocked_but_exits_available(new_entries_allowed=False) is True
    # the exit path produces an action regardless of any entry permission
    tr = PIPE.manage_position({"symbol": "TQQQ", "price": 94, "avg_entry": 100})
    assert tr.decision == "EXIT_STACK:EXIT" and tr.exit_stack_reason == "HARD_STOP"


# ---------------------------------------------------------------- section-16 entry conjunction
def _gate(**kw):
    base = dict(trusted_current_data=True, regime_locked_and_eligible=True, eligible_symbol=True,
                valid_policy_setup=True, entry_alpha_decision="ALPHA_CONFIRMED", adversarial_veto=False,
                security_veto=False, allocation_above_minimum=True, deterministic_risk_approval=True)
    base.update(kw)
    return PIPE.EntryGateInputs(**base)


def test_entry_conjunction_all_pass():
    ok, failed = PIPE.entry_authorized(_gate())
    assert ok and failed == []


def test_entry_conjunction_requires_alpha_confirmed():
    ok, failed = PIPE.entry_authorized(_gate(entry_alpha_decision="ALPHA_WEAK"))
    assert not ok and "alpha_confirmed" in failed


def test_entry_conjunction_any_veto_blocks():
    assert not PIPE.entry_authorized(_gate(adversarial_veto=True))[0]
    assert not PIPE.entry_authorized(_gate(security_veto=True))[0]
    assert not PIPE.entry_authorized(_gate(deterministic_risk_approval=False))[0]
    assert not PIPE.entry_authorized(_gate(regime_locked_and_eligible=False))[0]


# ---------------------------------------------------------------- pipeline evaluate_entry
def _plat_open():
    st = PS.default_state(); st.clear("global"); st.new_entries_enabled = True
    return st


def test_pipeline_no_alpha_evidence_is_no_trade():
    ev = EvidenceRegistry()
    rs = RE.RegimeState(lock_status=RE.CONFIRMED_LOCK, regime=RE.BULL)
    rs.last_confidence = 0.8
    perms = RE.permissions(RE.BULL)
    ctx = {"symbol": "TQQQ", "orientation": "bull", "data_quality": "OK", "price": 100.0,
           "equity": 100000.0, "buying_power": 100000.0, "agent_outputs": {"trend": "ENTER_LONG"},
           "entry_alpha_decision": EA.NO_ALPHA_EVIDENCE, "eligible_symbol": True}
    tr = PIPE.evaluate_entry(ctx, rs, perms, ev, _plat_open(), PIPE.PipelineConfig())
    assert tr.decision == "NO_TRADE"
    assert tr.entry_alpha_decision == EA.NO_ALPHA_EVIDENCE


def test_pipeline_confirmed_alpha_observe_only():
    ev = EvidenceRegistry()
    ev.put(Evidence("ev1", "trend", "INTRADAY_MOMENTUM", "lev_index", state="LIVE_APPROVED",
                    classification="VALIDATED"))
    rs = RE.RegimeState(lock_status=RE.CONFIRMED_LOCK, regime=RE.BULL); rs.last_confidence = 0.8
    ctx = {"symbol": "TQQQ", "orientation": "bull", "data_quality": "OK", "price": 100.0,
           "equity": 100000.0, "buying_power": 100000.0, "agent_outputs": {"trend": "ENTER_LONG"},
           "evidence_id": "ev1", "eligible_symbol": True, "alpha_confidence": 0.8}
    tr = PIPE.evaluate_entry(ctx, rs, RE.permissions(RE.BULL), ev, _plat_open(),
                             PIPE.PipelineConfig(observe_only=True), mode="LIVE")
    assert tr.decision == "OBSERVE_ONLY"   # authorized but not submitted (fail-safe default)
    assert tr.final_execution_weight and tr.final_execution_weight > 0


def test_pipeline_blocks_entry_but_exit_still_runs_in_transition():
    ev = EvidenceRegistry()
    rs = RE.RegimeState(lock_status=RE.UNLOCKED, regime=RE.TRANSITION)   # no new entries
    ctx = {"symbol": "TQQQ", "orientation": "bull", "data_quality": "OK", "price": 100.0,
           "equity": 100000.0, "buying_power": 100000.0, "agent_outputs": {"trend": "ENTER_LONG"},
           "evidence_id": "ev1", "eligible_symbol": True}
    tr = PIPE.evaluate_entry(ctx, rs, RE.permissions(RE.TRANSITION), ev, _plat_open(), PIPE.PipelineConfig())
    assert tr.decision.startswith("BLOCKED") and "locked_eligible_regime" in tr.blocked_gates
    # exit path remains available in the same transition regime
    xt = PIPE.manage_position({"symbol": "TQQQ", "price": 94, "avg_entry": 100, "regime": "TRANSITION"})
    assert xt.decision == "EXIT_STACK:EXIT"


# ---------------------------------------------------------------- policy agents
def test_policy_agents_return_valid_recommendations():
    ctx = {"symbol": "TQQQ", "features": {"pdi": 30, "mdi": 5, "adx": 28, "vwap_dist": 0.01,
                                          "rel_strength": 1}, "horizon": "INTRADAY_MOMENTUM",
           "evidence_id": "ev1"}
    rec = PA.TrendPolicyAgent().analyze(ctx)
    assert AB.is_valid(rec) and rec.recommended_action == "ENTER_LONG"


def test_order_flow_agent_data_unavailable_vetoes():
    rec = PA.OrderFlowAgent().analyze({"symbol": "TQQQ", "order_flow": {"status": "DATA_UNAVAILABLE"}})
    assert rec.recommended_action == "VETO" and "DATA_UNAVAILABLE" in rec.reason_codes


def test_defensive_agent_reduces_in_crisis():
    rec = PA.DefensivePolicyAgent().analyze({"symbol": "TQQQ", "regime": "CRISIS", "features": {}})
    assert rec.recommended_action == "REDUCE"


# ---------------------------------------------------------------- decision trace
def test_decision_trace_append(tmp_path):
    p = tmp_path / "trace.jsonl"
    DT.append(DT.DecisionTrace(symbol="TQQQ", decision="NO_TRADE"), p)
    DT.append(DT.DecisionTrace(symbol="SOXL", decision="OBSERVE_ONLY"), p)
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2 and "TQQQ" in lines[0]
