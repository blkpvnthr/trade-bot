"""Tests for amps_v2.py (unified AMPS V2 engine) + trade.py integration.

Covers: BB-approach detection (approach/above/failed; false-approach rejection), entry-quality
scoring + renormalization when order flow is absent, mandatory gates, the four entry setups,
cost-adjusted breakeven, monotonic adaptive stops, exit-state transitions + reasons, EOD
precedence, and the disabled-by-default / shadow trade.py integration."""
from datetime import datetime, time

import amps_v2 as A


def _cfg(**over):
    return A.AmpsV2Config.from_dict({**A.AmpsV2Config().as_dict(), **over})


# ----------------------------------------------------- BB approach
def test_bb_distance_and_states():
    assert abs(A.bb_distance(101, 100) - 0.01) < 1e-9
    cfg = _cfg(bb_entry_policy="combined", bb_approach_distance_bps=10)
    assert A.bb_approach_state(100.5, 100.0, 0.005, 0.004, 0.1, cfg) == A.BB_ABOVE
    # below, within 10bps, gap closing, momentum up -> approaching
    assert A.bb_approach_state(99.95, 100.0, -0.0005, -0.0009, 0.5, cfg) == A.BB_APPROACHING
    # gap WIDENING -> not approaching (false approach rejected)
    assert A.bb_approach_state(99.95, 100.0, -0.0009, -0.0005, 0.5, cfg) == A.BB_NONE
    # declining momentum -> not approaching
    assert A.bb_approach_state(99.95, 100.0, -0.0005, -0.0009, -0.5, cfg) == A.BB_NONE
    # too far below (beyond 10bps) -> not approaching
    assert A.bb_approach_state(99.0, 100.0, -0.01, -0.012, 0.5, cfg) == A.BB_NONE


def test_bb_policy_ok():
    assert A.bb_policy_ok(A.BB_ABOVE, _cfg(bb_entry_policy="confirmed"))
    assert not A.bb_policy_ok(A.BB_APPROACHING, _cfg(bb_entry_policy="confirmed"))
    assert A.bb_policy_ok(A.BB_APPROACHING, _cfg(bb_entry_policy="approaching"))
    assert A.bb_policy_ok(A.BB_ABOVE, _cfg(bb_entry_policy="combined"))
    assert A.bb_policy_ok(A.BB_APPROACHING, _cfg(bb_entry_policy="combined"))


# ----------------------------------------------------- entry quality
def test_entry_quality_renormalizes_without_order_flow():
    cfg = _cfg()
    s, diag = A.entry_quality_score(cfg, di_plus=30, di_minus=15, di_sep_prev=10, adx=25,
                                    adx_prev=23, adx_prev2=22, bb_dist=0.0005, bb_slope=1,
                                    vwap_dist=0.001, cvd_accel=None, buying_pressure=None,
                                    rel_volume=2.0, expected_move_bps=25, cost_bps=6)
    assert 0 <= s <= 100 and diag["order_flow_available"] is False
    # with order flow present, score uses all 8 components
    s2, diag2 = A.entry_quality_score(cfg, 30, 15, 10, 25, 23, 22, 0.0005, 1, 0.001,
                                      cvd_accel=0.9, buying_pressure=0.9, rel_volume=2.0,
                                      expected_move_bps=25, cost_bps=6)
    assert diag2["order_flow_available"] is True and 0 <= s2 <= 100


def test_mandatory_gates():
    cfg = _cfg()
    ok, fails = A.mandatory_gates(cfg, di_plus=30, di_minus=15, adx=25, adx_prev=23,
                                  bb_state=A.BB_ABOVE, above_vwap=True, spread_bps=3.0,
                                  data_fresh=True, liquidity_ok=True, cost_feasible=True, risk_ok=True)
    assert ok and fails == []
    ok2, fails2 = A.mandatory_gates(cfg, 15, 30, 25, 23, A.BB_ABOVE, True, 3.0, True, True, True, True)
    assert not ok2 and "di_plus_le_di_minus" in fails2
    ok3, fails3 = A.mandatory_gates(cfg, 30, 15, 25, 23, A.BB_ABOVE, True, 9.0, True, True, True, True)
    assert "spread_too_wide" in fails3


def test_expected_net_return():
    # 30% win * 30bps - 70% * 20bps - 6bps cost = 9 - 14 - 6 = -11 bps
    assert abs(A.expected_net_return_bps(0.3, 30, 20, 6) - (-11.0)) < 1e-9


def test_setups():
    assert A.setup_momentum_ignition(True, 12, 8, 25, 23, A.BB_ABOVE, 1.3, cvd_accel=None)
    assert not A.setup_momentum_ignition(False, 5, 8, 25, 23, A.BB_NONE, 1.0, cvd_accel=None)
    assert A.setup_compression_breakout(True, True, True, 1.5, 3.0, 5.0)
    assert not A.setup_compression_breakout(False, True, True, 1.5, 3.0, 5.0)  # vol expansion alone no entry
    # order-flow divergence requires ticks -> None CVD => not identifiable
    assert not A.setup_order_flow_divergence(None, 1.2, 0.3, True, A.BB_ABOVE)
    assert A.setup_order_flow_divergence(0.3, 1.2, 0.3, True, A.BB_ABOVE)


# ----------------------------------------------------- exit state machine
def test_cost_adjusted_breakeven():
    assert A.cost_adjusted_breakeven(100.0, 6.0) == 100.0 * (1 + 6 / 1e4)


def test_new_position_initial_stop():
    cfg = _cfg(initial_stop_bps=10)
    pos = A.new_amps_position("X", 100.0, "t", 10, cfg, 0.2, A.BB_ABOVE)
    assert abs(pos.effective_stop - 100.0 * (1 - 10 / 1e4)) < 1e-9 and pos.stage == "INITIAL"


def test_target_exit():
    cfg = _cfg(final_target_bps=30)
    pos = A.new_amps_position("X", 100.0, "t", 10, cfg, 0.2, A.BB_ABOVE)
    ex, r = A.amps_exit_decision(pos, 100.4, datetime(2026, 11, 18, 11, 0), cfg, time(15, 55), 1.0)
    assert ex and r == A.PROFIT_TARGET


def test_trailing_stop_monotonic_never_widens():
    cfg = _cfg()
    pos = A.new_amps_position("Y", 100.0, "t", 10, cfg, 0.2, A.BB_ABOVE)
    A.amps_exit_decision(pos, 100.25, datetime(2026, 11, 18, 11, 0), cfg, time(15, 55), 1.0)  # peak 25bps
    s1 = pos.effective_stop
    A.amps_exit_decision(pos, 100.10, datetime(2026, 11, 18, 11, 1), cfg, time(15, 55), 2.0)  # pullback
    assert pos.effective_stop >= s1  # never widens
    A.amps_exit_decision(pos, 100.40, datetime(2026, 11, 18, 11, 2), cfg, time(15, 55), 3.0)  # new high
    assert pos.effective_stop >= s1


def test_breakeven_stage_and_reason():
    cfg = _cfg(breakeven_activation_bps=10, trailing_activation_bps=20, min_profit_lock_activation_bps=15)
    pos = A.new_amps_position("Z", 100.0, "t", 10, cfg, 0.2, A.BB_ABOVE)
    # push peak to +12bps -> breakeven stage, stop at cost-adjusted breakeven
    A.amps_exit_decision(pos, 100.12, datetime(2026, 11, 18, 11, 0), cfg, time(15, 55), 1.0)
    assert pos.stage == "BREAKEVEN" and pos.effective_stop >= 100.0
    ex, r = A.amps_exit_decision(pos, pos.effective_stop - 0.001, datetime(2026, 11, 18, 11, 1), cfg, time(15, 55), 2.0)
    assert ex and r == A.BREAKEVEN_STOP


def test_max_hold_and_eod_precedence():
    cfg = _cfg(max_hold_minutes=5)
    pos = A.new_amps_position("A", 100.0, "t", 10, cfg, 0.2, A.BB_ABOVE)
    ex, r = A.amps_exit_decision(pos, 100.1, datetime(2026, 11, 18, 11, 0), cfg, time(15, 55), 5.0)
    assert ex and r == A.MAX_HOLD_EXIT
    pos2 = A.new_amps_position("B", 100.0, "t", 10, cfg, 0.2, A.BB_ABOVE)
    ex2, r2 = A.amps_exit_decision(pos2, 100.1, datetime(2026, 11, 18, 15, 56), cfg, time(15, 55), 1.0)
    assert ex2 and r2 == A.EOD_EXIT  # EOD takes precedence over everything


def test_momentum_deterioration_exit():
    cfg = _cfg(bearish_confirmation_count=2)
    pos = A.new_amps_position("C", 100.0, "t", 10, cfg, 0.2, A.BB_ABOVE)
    ex, r = A.amps_exit_decision(pos, 100.05, datetime(2026, 11, 18, 11, 0), cfg, time(15, 55), 1.0, bearish_signals=2)
    assert ex and r == A.MOMENTUM_EXIT
    pos2 = A.new_amps_position("D", 100.0, "t", 10, cfg, 0.2, A.BB_ABOVE)
    ex2, _ = A.amps_exit_decision(pos2, 100.05, datetime(2026, 11, 18, 11, 0), cfg, time(15, 55), 1.0, bearish_signals=1)
    assert not ex2  # single bearish signal insufficient


# ----------------------------------------------------- trade.py integration
def test_trade_integration_disabled_by_default():
    import trade
    c = trade.ScannerConfig()
    assert c.amps_v2_enabled is False and c.amps_v2_shadow_mode is True
    assert hasattr(trade.ProductionOrderExecutor, "_run_amps_v2_overlay")


def test_cli_strategy_amps_v2_sets_env(monkeypatch):
    import sys, trade
    monkeypatch.delenv("AMPS_V2", raising=False)
    monkeypatch.setattr(sys, "argv", ["trade.py", "--strategy", "amps-v2", "--paper"])
    trade._apply_cli_overrides()
    import os
    assert os.environ.get("AMPS_V2") == "1"
    cfg = trade.build_default_config().scanner
    assert cfg.amps_v2_enabled is True
