"""Tests for profit_exit_engine.py -- exit scoring, adaptive trailing stop monotonicity,
net-profit checks, confirmations, hysteresis/cooldown/persistence, shadow isolation,
coordinator priority/deferral, partial vs full exits, and restart recovery."""
import order_flow_engine as of
import profit_exit_engine as pe


# ------------------------------------------------------------------ pure functions
def test_exit_score_weights():
    assert abs(pe.exit_score(1, 1, 1, 1, 1) - 100.0) < 1e-9
    assert pe.exit_score(0, 0, 0, 0, 0) == 0.0
    assert abs(pe.exit_score(1, 0, 0, 0, 0) - 30.0) < 1e-9   # sell_pressure weight 0.30
    assert abs(pe.exit_score(0, 1, 0, 0, 0) - 20.0) < 1e-9   # cvd_deterioration 0.20
    assert abs(pe.exit_score(0, 0, 1, 0, 0) - 15.0) < 1e-9   # buyer_exhaustion 0.15
    assert abs(pe.exit_score(0, 0, 0, 1, 0) - 20.0) < 1e-9   # structure_breakdown 0.20
    assert abs(pe.exit_score(0, 0, 0, 0, 1) - 15.0) < 1e-9   # profit_giveback 0.15
    # inputs clamped
    assert pe.exit_score(5, 5, 5, 5, 5) == 100.0


def test_structure_breakdown_score():
    assert pe.structure_breakdown_score(False, False, False, False, False) == 0.0
    assert abs(pe.structure_breakdown_score(True, True, True, True, True) - 1.0) < 1e-9
    assert pe.structure_breakdown_score(True, True, False, False, False) == 0.5


def test_profit_giveback_score():
    assert pe.profit_giveback_score(100, 110, 110) == 0.0           # at peak
    assert pe.profit_giveback_score(100, 100, 110) > 0.9            # gave back the whole gain
    assert pe.profit_giveback_score(100, 105, 110) == 0.5           # half back
    assert pe.profit_giveback_score(100, 100.5, 100.5, peak_trigger_pct=0.03) == 0.0  # peak < trigger
    assert pe.profit_giveback_score(0, 1, 1) == 0.0                 # bad entry


def test_decision_from_score_bands():
    cfg = pe.ProfitExitConfig()
    assert pe.decision_from_score(10, cfg) == pe.HOLD_STRONG
    assert pe.decision_from_score(45, cfg) == pe.HOLD_CAUTION
    assert pe.decision_from_score(65, cfg) == pe.PROTECT_PROFIT
    assert pe.decision_from_score(80, cfg) == pe.PARTIAL_TAKE_PROFIT
    assert pe.decision_from_score(95, cfg) == pe.FULL_EXIT


def test_adaptive_trailing_stop_monotonic():
    assert pe.adaptive_trailing_stop(110, 2.0, 2.0, 0.0) == 106.0
    # only tightens: existing higher stop is kept
    assert pe.adaptive_trailing_stop(110, 2.0, 2.0, 107.0) == 107.0
    # higher peak raises the stop
    assert pe.adaptive_trailing_stop(120, 2.0, 2.0, 107.0) == 116.0
    # invalid atr -> unchanged
    assert pe.adaptive_trailing_stop(110, 0.0, 2.0, 107.0) == 107.0


def test_atr_multiplier_for_pressure():
    cfg = pe.ProfitExitConfig()
    assert pe.atr_multiplier_for_pressure("STRONG_BUY", cfg) == 2.5
    assert pe.atr_multiplier_for_pressure("NEUTRAL", cfg) == 2.0
    assert pe.atr_multiplier_for_pressure("STRONG_SELL", cfg) == 1.25


def test_net_profit_estimate():
    net, gross, cost = pe.net_profit_estimate(100, 10.0, 11.0, adv=1e7, cfg=pe.BASELINE_COSTS)
    assert gross == 100.0                      # (11-10)*100
    assert cost > 0 and net < gross            # costs reduce it
    # wider spread override raises cost
    net2, _g, cost2 = pe.net_profit_estimate(100, 10.0, 11.0, adv=1e7, cfg=pe.BASELINE_COSTS,
                                             spread_bps_override=100.0)
    assert cost2 > cost and net2 < net


def test_count_confirmations():
    cfg = pe.ProfitExitConfig()
    n, conf = pe.count_confirmations(95, cfg, sell_pressure=0.8, cvd_deterioration=0.6,
                                     buyer_exhaustion=0.6, structure_breakdown=0.6,
                                     cvd_divergence=-0.3, pressure="STRONG_SELL", stop_breached=True)
    assert n >= 3 and "sell_pressure" in conf and "trailing_stop_breached" in conf
    n0, _ = pe.count_confirmations(95, cfg, 0.1, 0.1, 0.1, 0.1, 0.0, "BUY", False)
    assert n0 == 0


# ------------------------------------------------------------------ engine behaviour
class _Snap:
    """Minimal order-flow snapshot stand-in."""
    def __init__(self, ok=True, status="OK", sell_pressure=0.0, cvd_deterioration=0.0,
                 buyer_exhaustion=0.0, cvd_divergence=0.0, pressure="NEUTRAL", spread_bps=10.0):
        self.ok = ok; self.status = status; self.sell_pressure = sell_pressure
        self.cvd_deterioration = cvd_deterioration; self.buyer_exhaustion = buyer_exhaustion
        self.cvd_divergence = cvd_divergence; self.pressure = pressure; self.spread_bps = spread_bps


_STRUCT_BREAK = {"lower_high": True, "lower_low": True, "lost_vwap": True,
                 "failed_breakout": True, "below_fast_slow": True}
_STRUCT_OK = {"lower_high": False, "lower_low": False, "lost_vwap": False,
              "failed_breakout": False, "below_fast_slow": False}


def _strong_sell_snap():
    return _Snap(sell_pressure=0.9, cvd_deterioration=0.8, buyer_exhaustion=0.8,
                 cvd_divergence=-0.4, pressure="STRONG_SELL")


def _full_exit_snap():
    # Strong enough that, with profit giveback + structure breakdown, the score clears 90.
    return _Snap(sell_pressure=0.95, cvd_deterioration=0.9, buyer_exhaustion=0.9,
                 cvd_divergence=-0.5, pressure="STRONG_SELL")


def test_insufficient_data_never_acts():
    eng = pe.ProfitExitEngine(pe.ProfitExitConfig())
    rec = eng.evaluate("AAA", qty=100, avg_entry=10.0, current_price=11.0,
                       snapshot=_Snap(ok=False, status="STALE"), structure=_STRUCT_BREAK,
                       atr=0.2, adv=1e7, now=1000.0)
    assert rec.decision == pe.INSUFFICIENT_DATA and not rec.enforce


def test_full_exit_requires_confirmations_and_persistence():
    # A full-exit score (>=90) needs profit giveback, so first establish a peak at 11.0,
    # then retrace to 10.2. ATR=1.0 keeps the trailing stop (9.75) below price so the
    # test isolates the score+persistence path (not the stop-breach path).
    cfg = pe.ProfitExitConfig(persistence_cycles=2, min_confirmations=3)
    eng = pe.ProfitExitEngine(cfg)
    snap = _full_exit_snap()
    eng.evaluate("AAA", 100, 10.0, 11.0, _Snap(), _STRUCT_OK, atr=1.0, adv=1e7, now=990.0)  # set peak
    # cycle 1: score >= full but band has not persisted -> gated down to PROTECT_PROFIT
    r1 = eng.evaluate("AAA", 100, 10.0, 10.2, snap, _STRUCT_BREAK, atr=1.0, adv=1e7, now=1000.0)
    assert r1.exit_score >= cfg.score_full
    assert r1.decision == pe.PROTECT_PROFIT and not r1.enforce
    # cycle 2: persisted + >=3 confirmations -> FULL_EXIT
    r2 = eng.evaluate("AAA", 100, 10.0, 10.2, snap, _STRUCT_BREAK, atr=1.0, adv=1e7, now=1005.0)
    assert r2.confirmation_count >= 3
    assert r2.decision == pe.FULL_EXIT and r2.enforce and r2.fraction == 1.0


def test_no_liquidation_on_score_alone():
    # High score but only ONE confirmation -> must NOT full-exit even after persistence.
    cfg = pe.ProfitExitConfig(persistence_cycles=1, min_confirmations=3)
    eng = pe.ProfitExitEngine(cfg)
    snap = _Snap(sell_pressure=0.9, cvd_deterioration=0.1, buyer_exhaustion=0.1,
                 cvd_divergence=0.0, pressure="NEUTRAL")  # only sell_pressure fires
    r = eng.evaluate("AAA", 100, 10.0, 10.3, snap, _STRUCT_OK, atr=0.1, adv=1e7, now=1000.0)
    assert r.decision != pe.FULL_EXIT


def test_partial_requires_net_profit():
    cfg = pe.ProfitExitConfig(persistence_cycles=1, min_confirmations=3)
    eng = pe.ProfitExitEngine(cfg)
    # score in partial band via structure + moderate flow
    snap = _Snap(sell_pressure=0.7, cvd_deterioration=0.6, buyer_exhaustion=0.5,
                 cvd_divergence=-0.2, pressure="SELL")
    # underwater position -> not net profitable -> must downgrade to PROTECT
    r = eng.evaluate("AAA", 100, 10.0, 9.0, snap, {"lower_high": True, "lower_low": True,
                     "lost_vwap": True, "failed_breakout": False, "below_fast_slow": False},
                     atr=0.1, adv=1e7, now=1000.0)
    assert r.decision in (pe.PROTECT_PROFIT, pe.HOLD_CAUTION, pe.FULL_EXIT)
    assert r.decision != pe.PARTIAL_TAKE_PROFIT


def test_cooldown_suppresses_repeat_action():
    cfg = pe.ProfitExitConfig(persistence_cycles=1, min_confirmations=3, cooldown_s=120.0)
    eng = pe.ProfitExitEngine(cfg)
    snap = _full_exit_snap()
    eng.evaluate("AAA", 100, 10.0, 11.0, _Snap(), _STRUCT_OK, atr=1.0, adv=1e7, now=990.0)  # peak, no action
    r1 = eng.evaluate("AAA", 100, 10.0, 10.2, snap, _STRUCT_BREAK, atr=1.0, adv=1e7, now=1000.0)
    assert r1.decision == pe.FULL_EXIT and r1.enforce
    # 10s later, still in cooldown -> must not re-fire a full exit
    r2 = eng.evaluate("AAA", 100, 10.0, 10.2, snap, _STRUCT_BREAK, atr=1.0, adv=1e7, now=1010.0)
    assert r2.decision != pe.FULL_EXIT


def test_restart_recovery(tmp_path):
    path = tmp_path / "pe_state.json"
    cfg = pe.ProfitExitConfig(persistence_cycles=1)
    eng = pe.ProfitExitEngine(cfg, state_path=path)
    # establish a peak at 12.0
    eng.evaluate("AAA", 100, 10.0, 12.0, _Snap(), _STRUCT_OK, atr=0.2, adv=1e7, now=1000.0)
    assert path.exists()
    # new engine instance loads persisted peak + stop (restart-safe)
    eng2 = pe.ProfitExitEngine(cfg, state_path=path)
    st = eng2._get_state("AAA", 10.0, 11.0)
    assert st.peak_price >= 12.0 and st.stop_level > 0


# ------------------------------------------------------------------ coordinator
class _Pos:
    def __init__(self, symbol, qty, avg_entry, current):
        self.symbol = symbol; self.qty = qty
        self.avg_entry_price = avg_entry; self.current_price = current
        self.market_value = qty * current


def _coord(cfg=None):
    return pe.ProfitExitCoordinator(pe.ProfitExitEngine(cfg or pe.ProfitExitConfig(persistence_cycles=1)))


def test_coordinator_shadow_isolation():
    """SHADOW mode must produce NO enforcement rows even on a confirmed full exit."""
    coord = _coord()
    pos = _Pos("AAA", 100, 10.0, 10.3)
    snaps = {"AAA": _strong_sell_snap()}
    recs, rows = coord.evaluate_positions([pos], set(), snaps, {"AAA": _STRUCT_BREAK},
                                          {"AAA": 0.1}, {"AAA": 1e7}, now=1000.0, shadow=True)
    assert rows == []                      # shadow: zero broker-bound rows
    assert len(recs) == 1


def test_coordinator_enforce_full_row():
    coord = _coord(pe.ProfitExitConfig(persistence_cycles=1, min_confirmations=3, cooldown_s=0.0))
    # establish a peak at 11.0 via a neutral eval (no action), then retrace to 10.2 -> FULL
    coord.engine.evaluate("AAA", 100, 10.0, 11.0, _Snap(), _STRUCT_OK, atr=1.0, adv=1e7, now=1000.0)
    pos = _Pos("AAA", 100, 10.0, 10.2)
    recs, rows = coord.evaluate_positions([pos], set(), {"AAA": _full_exit_snap()},
                                          {"AAA": _STRUCT_BREAK}, {"AAA": 1.0}, {"AAA": 1e7},
                                          now=1005.0, shadow=False)
    assert len(rows) == 1 and rows[0]["side"] == "sell"
    assert rows[0]["reason"] in ("profit_exit_full", "profit_exit_risk")
    assert rows[0]["delta_shares"] == 100   # full exit sells entire qty


def test_coordinator_enforce_partial_row():
    coord = _coord(pe.ProfitExitConfig(persistence_cycles=1, min_confirmations=3,
                                       cooldown_s=0.0, partial_fraction=0.5))
    # partial-band score (~75): strong sell at a profit, no peak giveback, stop not breached
    pos = _Pos("AAA", 100, 10.0, 10.3)
    recs, rows = coord.evaluate_positions([pos], set(), {"AAA": _strong_sell_snap()},
                                          {"AAA": _STRUCT_BREAK}, {"AAA": 0.1}, {"AAA": 1e7},
                                          now=1000.0, shadow=False)
    assert len(rows) == 1 and rows[0]["reason"] == "profit_exit_partial"
    assert rows[0]["delta_shares"] == 50    # half the position


def test_coordinator_defers_to_protective_layer():
    """A symbol already exiting via the protective layer must NOT get a duplicate row."""
    coord = _coord()
    pos = _Pos("AAA", 100, 10.0, 10.3)
    snap = _strong_sell_snap()
    coord.engine.evaluate("AAA", 100, 10.0, 10.3, snap, _STRUCT_BREAK, 0.1, 1e7, now=1000.0)
    recs, rows = coord.evaluate_positions([pos], {"AAA"}, {"AAA": snap}, {"AAA": _STRUCT_BREAK},
                                          {"AAA": 0.1}, {"AAA": 1e7}, now=1005.0, shadow=False)
    assert rows == []                       # deferred -> no duplicate exit
    assert recs[0].enforce is False
    assert "deferred" in recs[0].reason
