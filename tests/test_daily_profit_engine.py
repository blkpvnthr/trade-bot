"""Unit tests for the profit-first daily opportunity router (daily_profit_engine.py).

Covers the v2 spec test plan: exits never gated, safe-mode on feed timeout/stale,
bearish/cutoff blocks, ranking prefers higher prob/expected-R, daily/option/tactical
caps, and the overnight share + call gates.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

from daily_profit_engine import (
    TradeOpportunity, DailyOpportunityRouter, RouterConfig, is_bearish_context,
)

ET = ZoneInfo("America/New_York")
OPEN = datetime(2026, 6, 11, 10, 0, tzinfo=ET)   # before 15:30 cutoff
LATE = datetime(2026, 6, 11, 15, 45, tzinfo=ET)  # past cutoff
BULL = {"trade_permission": "normal", "market_bias": "bullish", "risk_level": "medium"}
BEAR = {"trade_permission": "halt_new_buys", "market_bias": "bearish", "risk_level": "high"}


def _opp(**kw):
    base = dict(symbol="AAA", side="buy", strategy_id="alpaca_daily_income",
                setup_type="VWAP_RECLAIM", score=80, probability=0.70, expected_r=2.0,
                liquidity_score=0.8, regime_alignment=0.8, risk_score=0.2,
                reference_price=100.0, metadata={"tactical": True, "trend_intact": True})
    base.update(kw)
    return TradeOpportunity(**base)


def _router(**cfg):
    return DailyOpportunityRouter(RouterConfig(**cfg))


def test_exits_never_gated():
    r = _router()
    assert r.exit_allowed("sell") is True
    assert r.exit_allowed("sell_short") is True
    assert r.exit_allowed("buy") is False


def test_safe_mode_blocks_entries_not_exits():
    r = _router()
    for kw, why in [({"feed_timeout": True}, "feed_timeout"),
                    ({"stale_state": True}, "stale_state"),
                    ({"missing_data": True}, "missing_data")]:
        res = r.route([_opp()], BULL, now_et=OPEN, **kw)
        assert res.safe_mode and res.safe_mode_reason == why
        assert res.approved == []
        assert all(x.blocked_by == "setup_not_validated" for x in res.rejected)
    assert r.exit_allowed("sell") is True  # exits unaffected by safe mode


def test_bearish_context_blocks_entries():
    assert is_bearish_context(BEAR) and not is_bearish_context(BULL)
    res = _router().route([_opp()], BEAR, now_et=OPEN)
    assert res.approved == []


def test_past_cutoff_blocks():
    res = _router().route([_opp()], BULL, now_et=LATE)
    assert res.approved == []
    assert all(x.reason == "past_entry_cutoff" for x in res.rejected)


def test_ranking_prefers_higher_prob_and_expected_r():
    r = _router(max_new_entries=1)
    low = _opp(symbol="LOW", probability=0.55, expected_r=1.6, score=62)
    high = _opp(symbol="HIGH", probability=0.88, expected_r=3.0, score=93)
    res = r.route([low, high], BULL, now_et=OPEN)
    assert [o.symbol for o in res.approved] == ["HIGH"]
    assert any(x.symbol == "LOW" and x.blocked_by == "daily_trade_cap" for x in res.rejected)


def test_min_expected_r_floor():
    res = _router(min_expected_r=1.5).route([_opp(symbol="WEAK", expected_r=1.0)], BULL, now_et=OPEN)
    assert res.approved == [] and any(x.blocked_by == "setup_not_validated" for x in res.rejected)


def test_daily_cap_max_new_entries():
    r = _router(max_new_entries=2, max_intraday_tactical_open=10)
    opps = [_opp(symbol="S%d" % i, probability=0.70 + 0.01 * i) for i in range(5)]
    res = r.route(opps, BULL, now_et=OPEN)
    assert len(res.approved) == 2
    assert sum(1 for x in res.rejected if x.blocked_by == "daily_trade_cap") == 3


def test_option_cap():
    r = _router(max_option_entries=1, max_new_entries=10, max_intraday_tactical_open=10)
    opps = [_opp(symbol="O%d" % i, asset_class="option",
                 metadata={"tactical": True, "trend_intact": True}) for i in range(3)]
    res = r.route(opps, BULL, now_et=OPEN)
    assert sum(1 for o in res.approved if o.asset_class == "option") == 1


def test_intraday_tactical_cap():
    r = _router(max_intraday_tactical_open=2, max_new_entries=10)
    opps = [_opp(symbol="T%d" % i) for i in range(5)]
    res = r.route(opps, BULL, now_et=OPEN, open_intraday_tactical_positions=0)
    assert len(res.approved) == 2


def test_overnight_call_gate_rejects_bad_and_allows_good():
    r = _router()
    base = dict(asset_class="option", overnight_candidate=True, score=93)
    bad = [
        _opp(symbol="C1", probability=0.70, **base,  # low prob
             ),
    ]
    # build explicitly to vary metadata
    low_prob = _opp(symbol="C1", asset_class="option", overnight_candidate=True, probability=0.70, score=93,
                    metadata={"dte": 20, "iv_crush_risk": 10, "protected_or_partialed": True, "active_exit_ladder": True})
    high_crush = _opp(symbol="C2", asset_class="option", overnight_candidate=True, probability=0.85, score=93,
                      metadata={"dte": 20, "iv_crush_risk": 80, "protected_or_partialed": True, "active_exit_ladder": True})
    near_exp = _opp(symbol="C3", asset_class="option", overnight_candidate=True, probability=0.85, score=93,
                    metadata={"dte": 3, "iv_crush_risk": 10, "protected_or_partialed": True, "active_exit_ladder": True})
    unprotected = _opp(symbol="C4", asset_class="option", overnight_candidate=True, probability=0.85, score=70,
                       metadata={"dte": 20, "iv_crush_risk": 10, "protected_or_partialed": False, "active_exit_ladder": True})
    good = _opp(symbol="C5", asset_class="option", overnight_candidate=True, probability=0.85, score=93,
                metadata={"dte": 20, "iv_crush_risk": 10, "protected_or_partialed": True, "active_exit_ladder": True})
    for o in (low_prob, high_crush, near_exp, unprotected):
        assert r.overnight_allowed(o, BULL)[0] is False
    assert r.overnight_allowed(good, BULL)[0] is True


def test_overnight_share_gate():
    r = _router()
    good = _opp(symbol="S1", overnight_candidate=True, metadata={"profitable": True, "trend_intact": True})
    assert r.overnight_allowed(good, BULL)[0] is True
    assert r.overnight_allowed(good, BEAR)[0] is False           # bearish blocks overnight
    bad = _opp(symbol="S2", overnight_candidate=True, metadata={"profitable": False, "trend_intact": True})
    assert r.overnight_allowed(bad, BULL)[0] is False            # not profitable, no thesis
