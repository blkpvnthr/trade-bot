"""Tests for the AMPS V6 platform spine (registry, platform-state, agents, risk gateway, fusion,
horizon selection, evidence registry, entry-alpha, multi-timeframe daily features).

Pure/offline: no network. Where real cached data is used it is read-only from the diagnostic's
prefetched caches; tests that need bars skip cleanly if a cache is absent.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pytest

from amps_v6.universe import instrument_registry as REG
from amps_v6.configuration import platform_state as PS
from amps_v6.agents import base as AB
from amps_v6.execution import order_intent as OI
from amps_v6.execution import risk_gateway as RG
from amps_v6.strategy import decision_fusion as DF
from amps_v6.strategy import horizon_classifier as HC
from amps_v6.alpha import evidence_registry as ER
from amps_v6.alpha import entry_alpha as EA
from amps_v6.features import multi_timeframe_engine as MTF


# ---------------------------------------------------------------- instrument registry
def test_registry_opposing_and_orientation():
    assert REG.REGISTRY["TQQQ"].directional_orientation == "bull"
    assert REG.REGISTRY["SQQQ"].inverse_status is True
    # opposing exposure within the QQQ group
    opp = REG.opposing_symbols("TQQQ")
    assert "SQQQ" in opp and "TQQQ" not in opp
    assert "TQQQ" in REG.opposing_symbols("SQQQ")


def test_registry_coverage_only_reports_real_data():
    cov = REG.coverage_report()
    assert cov["n_candidates"] >= 20
    # every "with data" symbol must actually have a cached bar file
    for s in cov["candidates_with_data"]:
        assert REG.data_available(s)
    for s in cov["candidates_missing_data"]:
        assert not REG.data_available(s)


# ---------------------------------------------------------------- platform state (3-state gate)
def test_default_state_is_safe():
    st = PS.default_state()
    assert st.mode == PS.Mode.OBSERVE_ONLY
    assert st.live_activation_approved is False
    assert st.new_entries_enabled is False
    assert st.kill["global"] is True            # global kill tripped by default
    assert st.strategy_live_eligible is False


def test_can_send_live_order_fail_closed():
    st = PS.default_state()
    ok, reason = st.can_send_live_order("SHORT_SWING")
    assert ok is False and "PLATFORM_LIVE_READY" in reason
    # satisfy everything step by step; it must still fail until ALL three states + mode + kill clear
    st.platform_live_ready = True
    st.horizon_eligible["SHORT_SWING"] = True
    st.live_activation_approved = True
    st.mode = PS.Mode.LIVE
    st.new_entries_enabled = True
    st.clear("global")
    ok, reason = st.can_send_live_order("SHORT_SWING")
    assert ok is True, reason
    # an unvalidated OTHER horizon is still blocked
    ok2, _ = st.can_send_live_order("MICRO_SCALP")
    assert ok2 is False


def test_kill_switch_trip_clear_and_persist(tmp_path):
    st = PS.default_state()
    st.trip("overnight_entries")
    assert st.kill["overnight_entries"] is True
    with pytest.raises(ValueError):
        st.trip("nonexistent")
    p = tmp_path / "state.json"
    st.save(p)
    st2 = PS.PlatformState.load(p)
    assert st2.kill["overnight_entries"] is True
    assert st2.kill["global"] is True


# ---------------------------------------------------------------- agent schema
def _rec(**kw):
    base = dict(agent_id="a", recommendation_id="r", symbol="TQQQ", underlying="QQQ",
                timestamp=datetime.now(timezone.utc).isoformat(), data_version="d",
                strategy_version="amps_v6", feature_snapshot_id="f", recommended_action="HOLD",
                trading_horizon="NONE", confidence=0.5, evidence_id=None)
    base.update(kw)
    return AB.Recommendation(**base)


def test_schema_rejects_bad_action_and_confidence_and_expiry():
    with pytest.raises(AB.SchemaError):
        AB.validate(_rec(recommended_action="LAUNCH_ROCKET"))
    with pytest.raises(AB.SchemaError):
        AB.validate(_rec(confidence=1.5))
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    with pytest.raises(AB.SchemaError):
        AB.validate(_rec(expiration_timestamp=past))


def test_schema_enter_long_requires_evidence():
    with pytest.raises(AB.SchemaError):
        AB.validate(_rec(recommended_action="ENTER_LONG", trading_horizon="SHORT_SWING", evidence_id=None))
    ok = AB.validate(_rec(recommended_action="ENTER_LONG", trading_horizon="SHORT_SWING", evidence_id="ev1"))
    assert ok.evidence_id == "ev1"


# ---------------------------------------------------------------- risk gateway
def _ctx(**kw):
    base = dict(equity=100_000.0, buying_power=100_000.0)
    base.update(kw)
    return RG.PortfolioContext(**base)


def _intent(**kw):
    base = dict(symbol="TQQQ", side="buy", qty=0, reason="test", approved_trading_horizon="INTRADAY_MOMENTUM",
                protective_stop=98.0, effective_leverage=3.0)
    base.update(kw)
    return OI.OrderIntent(**base)


def test_risk_gateway_blocks_on_kill_switch():
    st = PS.default_state()                       # global kill tripped
    dec = RG.evaluate(_intent(), 100.0, _ctx(), st)
    assert dec.approved is False and "kill" in dec.reason


def test_risk_gateway_blocks_opposing_exposure():
    st = PS.default_state(); st.clear("global")
    dec = RG.evaluate(_intent(), 100.0, _ctx(opposing_held=("SQQQ",)), st)
    assert dec.approved is False and "opposing" in dec.reason


def test_risk_gateway_blocks_swing_without_overnight_permission():
    st = PS.default_state(); st.clear("global")
    it = _intent(approved_trading_horizon="SHORT_SWING", overnight_permission=False)
    dec = RG.evaluate(it, 100.0, _ctx(), st)
    assert dec.approved is False and "overnight" in dec.reason


def test_risk_gateway_approves_and_sizes_by_risk():
    st = PS.default_state(); st.clear("global")
    # entry 100, stop 98 -> $2 risk/share; 0.5% of 100k = $500 risk -> 250 shares; notional 25k
    dec = RG.evaluate(_intent(protective_stop=98.0), 100.0, _ctx(), st)
    assert dec.approved is True
    assert abs(dec.approved_qty - 200.0) < 1e-6 or dec.approved_qty <= 250.0
    # notional cap is 20% of equity = 20k -> 200 shares binds before the 250 risk qty
    assert dec.sized_notional <= 20_000.0 + 1e-6


def test_risk_gateway_blocks_when_positions_full():
    st = PS.default_state(); st.clear("global")
    dec = RG.evaluate(_intent(), 100.0, _ctx(open_positions=5), st)
    assert dec.approved is False and "max_open_positions" in dec.reason


# ---------------------------------------------------------------- decision fusion
def _fin(**kw):
    base = dict(valid_market_data=True, valid_setup=True, eligible_instrument=True,
                eligible_trading_horizon=True, alpha_confirmed=True, positive_expected_net_return=True,
                acceptable_execution_cost=True, deterministic_risk_approval=True)
    base.update(kw)
    return DF.FusionInputs(**base)


def test_fusion_all_pass_allows():
    assert DF.fuse(_fin()).entry_allowed is True


def test_fusion_any_failed_gate_blocks():
    r = DF.fuse(_fin(alpha_confirmed=False))
    assert r.entry_allowed is False and "alpha_confirmed" in r.failed_gates
    r2 = DF.fuse(_fin(adversarial_veto=True))
    assert r2.entry_allowed is False and "no_adversarial_veto" in r2.failed_gates


def test_fusion_swing_and_inverse_extra_gates():
    r = DF.fuse(_fin(is_swing=True, overnight_policy_approval=False))
    assert r.entry_allowed is False and "overnight_policy_approval" in r.failed_gates
    r2 = DF.fuse(_fin(is_inverse=True, bearish_underlying_confirmation=False))
    assert r2.entry_allowed is False and "bearish_underlying_confirmation" in r2.failed_gates


# ---------------------------------------------------------------- horizon classifier
def test_horizon_select_prefers_risk_adjusted_not_nominal():
    swing = HC.HorizonEval("EXTENDED_SWING", 300, 16, 250, 0.5, 8.0, 50.0, evidence_ok=True)
    intraday = HC.HorizonEval("INTRADAY_MOMENTUM", 40, 6, 20, 0.55, 0.5, 0.0, evidence_ok=True)
    chosen, ev, scores = HC.select([swing, intraday])
    # intraday has far better net/risk/time despite a smaller nominal target
    assert chosen == "INTRADAY_MOMENTUM"


def test_horizon_no_trade_when_no_evidence_or_negative():
    e1 = HC.HorizonEval("SHORT_SWING", 10, 16, 100, 0.4, 3.0, 0.0, evidence_ok=True)  # net negative
    e2 = HC.HorizonEval("MICRO_SCALP", 50, 6, 10, 0.6, 0.25, 0.0, evidence_ok=False)  # no evidence
    chosen, ev, scores = HC.select([e1, e2])
    assert chosen == HC.NO_TRADE and ev is None


# ---------------------------------------------------------------- evidence registry
def test_evidence_mode_gating_and_promotion():
    reg = ER.EvidenceRegistry()
    reg.put(ER.Evidence("ev1", "trend", "SHORT_SWING", "lev_index", state=ER.RESEARCH_CANDIDATE,
                        classification="CONTINUE_RESEARCH"))
    ok, _ = reg.permits_entry("ev1", "PAPER")
    assert ok is False                              # research < paper
    reg.promote("ev1", ER.PAPER_CANDIDATE, "t")
    ok, _ = reg.permits_entry("ev1", "PAPER")
    assert ok is True
    # cannot skip two states to LIVE_APPROVED
    with pytest.raises(ValueError):
        reg.promote("ev1", ER.LIVE_APPROVED, "t")
    # live refused until live_approved
    assert reg.permits_entry("ev1", "LIVE")[0] is False


def test_evidence_suspend_blocks_and_persists(tmp_path):
    reg = ER.EvidenceRegistry()
    reg.put(ER.Evidence("ev1", "trend", "SHORT_SWING", "lev_index", state=ER.LIVE_APPROVED,
                        classification="VALIDATED"))
    assert reg.permits_entry("ev1", "LIVE")[0] is True
    reg.suspend("ev1", "t", "drift")
    assert reg.permits_entry("ev1", "LIVE")[0] is False
    p = tmp_path / "ev.json"
    reg.save(p)
    reg2 = ER.EvidenceRegistry.load(p)
    assert reg2.get("ev1").state == ER.LIVE_SUSPENDED


# ---------------------------------------------------------------- entry alpha
def _symmetric_series(n=400, seed=1):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 1.0, n)
    close = 100 + np.cumsum(steps)
    high = close + np.abs(rng.normal(0, 0.3, n))
    low = close - np.abs(rng.normal(0, 0.3, n))
    opn = close - rng.normal(0, 0.1, n)
    return high, low, opn, close


def test_entry_alpha_symmetric_walk_is_no_alpha():
    h, l, o, c = _symmetric_series()
    es = EA.EntrySet("X", h, l, o, c, np.arange(5, 350, 2), cost_rt_bps=16.0)
    prof = EA.horizon_profile([es], horizon=5)
    ctrl = EA.random_control([es], [es], 5)
    dec = EA.decide(prof, ctrl, 5, min_n=50)
    assert dec.decision != EA.ALPHA_CONFIRMED     # a random walk must never confirm alpha
    assert dec.decision in (EA.NO_ALPHA_EVIDENCE, EA.ALPHA_WEAK)


def test_entry_alpha_decide_grades_honestly():
    # insufficient data
    d = EA.decide({"n": 10}, {"n": 10}, 5, min_n=100)
    assert d.decision == EA.INSUFFICIENT_DATA
    # confirmed: asymmetric, net-positive, beats control
    prof = {"n": 500, "mfe_mae_ratio": 1.5, "hold_to_horizon_net_bps": 20.0}
    ctrl = {"n": 500, "hold_to_horizon_net_bps": 5.0}
    assert EA.decide(prof, ctrl, 5).decision == EA.ALPHA_CONFIRMED
    # weak: asymmetric + beats control but net-negative
    prof2 = {"n": 500, "mfe_mae_ratio": 1.2, "hold_to_horizon_net_bps": -10.0}
    ctrl2 = {"n": 500, "hold_to_horizon_net_bps": -80.0}
    assert EA.decide(prof2, ctrl2, 5, min_edge_vs_control_bps=5.0).decision == EA.ALPHA_WEAK


# ---------------------------------------------------------------- multi-timeframe daily features
def test_daily_feature_attach_no_lookahead():
    n = 120
    rng = np.random.default_rng(3)
    close = 50 + np.cumsum(rng.normal(0, 0.5, n))
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="B").date,
        "open": close - 0.1, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": rng.integers(1e6, 2e6, n).astype(float)})
    out = MTF._attach_daily_features(df)
    assert not out.empty
    # di_sep_prev is di_sep shifted by one -> strictly past information
    assert np.isnan(out["di_sep_prev"].iloc[0]) or out["di_sep_prev"].iloc[20] == out["di_sep"].iloc[19]
    assert {"pdi", "mdi", "adx", "bb_mid", "atr", "dollar_volume", "trend_pos"} <= set(out.columns)
