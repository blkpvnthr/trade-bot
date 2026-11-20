"""Tests for buy_opportunity_engine.py + order_flow buy-side signals -- score/normalization,
entry-setup classification, seller-exhaustion confirmations, stale/insufficient data,
scanner-eligibility enforcement, hysteresis/cooldown, coordination (pending exit / held /
reentry), duplicate-entry prevention, shadow isolation, restart recovery."""
import order_flow_engine as of
import buy_opportunity_engine as bo


# ------------------------------------------------ order-flow buy-side signals
def test_order_flow_buy_side_signals():
    sof = of.SymbolOrderFlow("AAA", of.OrderFlowConfig(min_trades_for_signal=4, stale_trade_s=1e9))
    t0 = 1_000_000.0
    sof.on_quote(t0, bid=100.0, ask=100.1, bid_size=500, ask_size=100)
    for i in range(12):
        sof.on_trade(t0 + i, price=100.1, size=100)  # lift ask -> aggressive buys
    s = sof.snapshot(now=t0 + 12)
    assert s.ok and s.pressure in ("BUY", "STRONG_BUY")
    assert s.buy_pressure > 0.6 and s.cvd > 0 and s.cvd_trend > 0


def test_norm_buy_side_functions():
    assert of.norm_buy_pressure(1.0, 1.0, 100, 100, 0) > 0.8
    assert of.norm_buy_pressure(-1.0, -1.0, -100, 0, 100) < 0.2
    assert of.norm_cvd_improvement(cvd=0.0, cvd_trough=-1000.0, delta_30s=50, delta_300s=-200) > 0.5
    assert of.norm_cvd_improvement(cvd=-1000.0, cvd_trough=-1000.0, delta_30s=-5, delta_300s=-200) == 0.0
    assert of.norm_seller_exhaustion(cvd_div=0.4, delta_10s=-10, delta_60s=-120, buy_sell_r=1.5) > 0.3
    assert of.norm_seller_exhaustion(cvd_div=-0.4, delta_10s=-100, delta_60s=-120, buy_sell_r=0.3) < 0.2


def test_subscription_prioritization_and_rotation():
    eng = of.OrderFlowEngine(of.OrderFlowConfig(max_subscriptions=5))
    groups = [["POS1", "POS2"], ["PEND1"], ["C1", "C2", "C3"], ["U1", "U2", "U3", "U4", "U5"]]
    s1 = eng.select_priority_symbols(groups)
    assert s1[:3] == ["POS1", "POS2", "PEND1"] and len(s1) == 5   # mandatory kept, cap honored
    s2 = eng.select_priority_symbols(groups)
    assert s2[:3] == ["POS1", "POS2", "PEND1"]                    # mandatory stable
    assert s1 != s2                                               # overflow rotates fairly


# ------------------------------------------------ pure scoring
def test_buy_score_weights():
    assert abs(bo.buy_score(1, 1, 1, 1, 1) - 100.0) < 1e-9
    assert bo.buy_score(0, 0, 0, 0, 0) == 0.0
    assert abs(bo.buy_score(1, 0, 0, 0, 0) - 30.0) < 1e-9   # buy_pressure 0.30
    assert abs(bo.buy_score(0, 1, 0, 0, 0) - 20.0) < 1e-9   # cvd_improvement 0.20
    assert abs(bo.buy_score(0, 0, 1, 0, 0) - 15.0) < 1e-9   # seller_exhaustion 0.15
    assert abs(bo.buy_score(0, 0, 0, 1, 0) - 20.0) < 1e-9   # bullish_structure 0.20
    assert abs(bo.buy_score(0, 0, 0, 0, 1) - 15.0) < 1e-9   # entry_quality 0.15
    assert bo.buy_score(5, 5, 5, 5, 5) == 100.0             # clamped


def test_bullish_structure_and_entry_quality():
    assert bo.bullish_structure_score(False, False, False, False, False, False, False) == 0.0
    assert abs(bo.bullish_structure_score(True, True, True, True, True, True, True) - 1.0) < 1e-9
    # missing spread -> entry_quality 0 (never a bullish freebie)
    assert bo.entry_quality_score(None, 15.0, "tight", False) == 0.0
    assert bo.entry_quality_score(5.0, 15.0, "tight", False) > 0.9
    assert bo.entry_quality_score(5.0, 15.0, "tight", True) < 0.5   # over-extended penalized
    assert bo.entry_quality_score(20.0, 15.0, "wide", False) < 0.1  # spread beyond max -> ~0


def test_decision_bands_and_setup_classification():
    cfg = bo.BuyOpportunityConfig()
    assert bo.decision_from_buy_score(10, cfg) == bo.NO_BUY
    assert bo.decision_from_buy_score(50, cfg) == bo.WATCH
    assert bo.decision_from_buy_score(65, cfg) == bo.BUY_SETUP
    assert bo.decision_from_buy_score(80, cfg) == bo.BUY_READY
    assert bo.decision_from_buy_score(95, cfg) == bo.STRONG_BUY
    assert bo.classify_entry_setup(0.3, 0.6, {"breakout_confirmed": True}) == bo.SETUP_BREAKOUT
    assert bo.classify_entry_setup(0.0, 0.1, {"above_sma50": True, "pulled_back": True,
                                              "reclaimed_vwap": True}) == bo.SETUP_PULLBACK
    assert bo.classify_entry_setup(0.3, 0.6, {"higher_low": True}) == bo.SETUP_REVERSAL
    assert bo.classify_entry_setup(0.0, 0.0, {}) == bo.SETUP_NONE


def test_confirmations_count():
    cfg = bo.BuyOpportunityConfig()
    struct = {"price_stops_lower_low": True, "reclaimed_vwap": True, "di_plus_over_minus": True,
              "adx_rising": True, "reclaimed_sma50": True, "failed_breakdown_reversal": True}
    n, conf = bo.count_buy_confirmations(cfg, buy_pressure=0.7, cvd_improvement=0.6,
                                         seller_exhaustion=0.6, cvd_divergence=0.3,
                                         executed_imbalance=0.1, delta_10s=20, spread_bps=6.0,
                                         structure=struct)
    assert n >= 3 and "bullish_cvd_divergence" in conf and "liquidity_improved" in conf
    n0, _ = bo.count_buy_confirmations(cfg, 0.1, 0.0, 0.0, -0.3, -0.5, -10, None, {})
    assert n0 == 0


# ------------------------------------------------ engine
class _Snap:
    def __init__(self, ok=True, status="OK", buy_pressure=0.9, cvd_improvement=0.8,
                 seller_exhaustion=0.7, cvd_divergence=0.3, executed_imbalance=0.2,
                 delta_10s=50.0, spread_bps=6.0, liquidity="tight", pressure="STRONG_BUY",
                 cvd_trend=0.8, buy_vol_60s=1000.0, sell_vol_60s=200.0):
        for k, v in locals().items():
            if k != "self":
                setattr(self, k, v)


_STRUCT_BULL = {"higher_low": True, "price_stops_lower_low": True, "reclaimed_vwap": True,
                "reclaimed_sma50": True, "di_plus_over_minus": True, "adx_rising": True,
                "failed_breakdown_reversal": True, "breakout_confirmed": False,
                "above_sma50": True, "pulled_back": True, "over_extended": False}


def test_insufficient_data_never_bullish():
    eng = bo.BuyOpportunityEngine(bo.BuyOpportunityConfig())
    r = eng.evaluate("AAA", _Snap(ok=False, status="STALE"), _STRUCT_BULL, scanner_eligible=True, now=1000.0)
    assert r.decision == bo.INSUFFICIENT_DATA and not r.enforce


def test_buy_ready_requires_persistence_and_confirmations():
    cfg = bo.BuyOpportunityConfig(persistence_cycles=2, min_confirmations=3)
    eng = bo.BuyOpportunityEngine(cfg)
    r1 = eng.evaluate("NVDA", _Snap(), _STRUCT_BULL, scanner_eligible=True, now=1000.0)
    assert r1.buy_score >= cfg.score_ready
    assert not r1.enforce                       # not persisted yet -> de-escalated
    r2 = eng.evaluate("NVDA", _Snap(), _STRUCT_BULL, scanner_eligible=True, now=1005.0)
    assert r2.decision in (bo.BUY_READY, bo.STRONG_BUY) and r2.enforce and r2.confirmation_count >= 3


def test_high_score_does_not_authorize_without_scanner_eligibility():
    cfg = bo.BuyOpportunityConfig(persistence_cycles=1, require_existing_entry_signal=True)
    eng = bo.BuyOpportunityEngine(cfg)
    eng.evaluate("AMD", _Snap(), _STRUCT_BULL, scanner_eligible=False, now=1000.0)
    r = eng.evaluate("AMD", _Snap(), _STRUCT_BULL, scanner_eligible=False, now=1005.0)
    assert not r.enforce   # a high score alone must never authorize an entry


def test_over_extended_blocks_enforcement():
    cfg = bo.BuyOpportunityConfig(persistence_cycles=1)
    eng = bo.BuyOpportunityEngine(cfg)
    struct = dict(_STRUCT_BULL); struct["over_extended"] = True
    eng.evaluate("TSLA", _Snap(), struct, scanner_eligible=True, now=1000.0)
    r = eng.evaluate("TSLA", _Snap(), struct, scanner_eligible=True, now=1005.0)
    assert not r.enforce   # do not chase an over-extended price


def test_wide_spread_blocks_enforcement():
    cfg = bo.BuyOpportunityConfig(persistence_cycles=1, max_spread_bps=15.0)
    eng = bo.BuyOpportunityEngine(cfg)
    snap = _Snap(spread_bps=40.0, liquidity="wide")
    eng.evaluate("AAA", snap, _STRUCT_BULL, scanner_eligible=True, now=1000.0)
    r = eng.evaluate("AAA", snap, _STRUCT_BULL, scanner_eligible=True, now=1005.0)
    assert not r.enforce


def test_restart_recovery(tmp_path):
    path = tmp_path / "buy_state.json"
    cfg = bo.BuyOpportunityConfig(persistence_cycles=2)
    eng = bo.BuyOpportunityEngine(cfg, state_path=path)
    eng.evaluate("NVDA", _Snap(), _STRUCT_BULL, scanner_eligible=True, now=1000.0)
    assert path.exists()
    eng2 = bo.BuyOpportunityEngine(cfg, state_path=path)
    st = eng2._get_state("NVDA")
    assert st.band_streak >= 1 and st.last_band in (bo.BUY_READY, bo.STRONG_BUY)


# ------------------------------------------------ coordinator
def _coord(cfg=None):
    return bo.BuyOpportunityCoordinator(bo.BuyOpportunityEngine(cfg or bo.BuyOpportunityConfig(persistence_cycles=1)))


def test_coordinator_shadow_isolation():
    coord = _coord()
    mon = {"NVDA": _Snap()}
    recs, confirmed = coord.evaluate(mon, {"NVDA": _STRUCT_BULL}, scanner_candidates={"NVDA"},
                                     scanner_eligible={"NVDA"}, held=set(), pending_exit=set(),
                                     reentry_block=set(), now=1000.0, shadow=True)
    assert confirmed == set() and len(recs) == 1   # shadow: never confirms an order


def test_coordinator_confirms_only_eligible_candidate():
    coord = _coord()
    mon = {"NVDA": _Snap(), "AMD": _Snap(), "AAPL": _Snap()}
    sb = {k: _STRUCT_BULL for k in mon}
    recs, confirmed = coord.evaluate(mon, sb, scanner_candidates={"NVDA", "AMD", "AAPL"},
                                     scanner_eligible={"NVDA", "AMD", "AAPL"},
                                     held={"AMD"}, pending_exit={"AAPL"}, reentry_block=set(),
                                     now=1000.0, shadow=False)
    assert confirmed == {"NVDA"}          # AMD held (no pyramid), AAPL pending exit (no conflict)


def test_coordinator_reentry_cooldown_blocks():
    coord = _coord()
    mon = {"NVDA": _Snap()}
    recs, confirmed = coord.evaluate(mon, {"NVDA": _STRUCT_BULL}, scanner_candidates={"NVDA"},
                                     scanner_eligible={"NVDA"}, held=set(), pending_exit=set(),
                                     reentry_block={"NVDA"}, now=1000.0, shadow=False)
    assert confirmed == set()
    assert any("reentry_cooldown" in r.reason for r in recs)


def test_coordinator_not_scanner_candidate_is_monitor_only():
    coord = _coord()
    mon = {"XYZ": _Snap()}  # strong flow but scanner never flagged it
    recs, confirmed = coord.evaluate(mon, {"XYZ": _STRUCT_BULL}, scanner_candidates=set(),
                                     scanner_eligible=set(), held=set(), pending_exit=set(),
                                     reentry_block=set(), now=1000.0, shadow=False)
    assert confirmed == set()
    assert any("not_scanner_candidate" in r.reason for r in recs)
