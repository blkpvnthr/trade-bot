import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from daily_profit_engine import DailyOpportunityRouter, RouterConfig, TradeOpportunity


def _opp(symbol, **kwargs):
    base = {
        "symbol": symbol,
        "asset_class": "equity",
        "side": "buy",
        "strategy_id": "test",
        "setup_type": "VWAP_RECLAIM",
        "score": 80.0,
        "probability": 0.70,
        "expected_r": 2.0,
        "liquidity_score": 0.80,
        "regime_alignment": 0.80,
        "risk_score": 0.20,
        "freshness_score": 1.0,
        "qty": 1,
        "reference_price": 10.0,
        "reason": "test",
    }
    base.update(kwargs)
    return TradeOpportunity(**base)


def test_ranking_prefers_edge_over_quota_filling():
    router = DailyOpportunityRouter(RouterConfig(max_new_entries=2))
    weak = _opp("WEAK", probability=0.55, expected_r=1.5, score=60, liquidity_score=0.5)
    strong = _opp("STRONG", probability=0.90, expected_r=3.0, score=90, liquidity_score=0.9)
    mid = _opp("MID", probability=0.75, expected_r=2.0, score=75, liquidity_score=0.7)

    result = router.route([weak, strong, mid], market_context={"overall_market_bias": "bullish"})

    assert [o.symbol for o in result.approved] == ["STRONG", "MID"]
    assert any(r.symbol == "WEAK" and r.blocked_by == "opportunity_rank" for r in result.rejected)


def test_daily_cap_blocks_buys_but_not_sell_permission():
    router = DailyOpportunityRouter(RouterConfig(max_new_entries=1))
    result = router.route(
        [_opp("A"), _opp("B", probability=0.80, expected_r=2.5)],
        market_context={"overall_market_bias": "bullish"},
    )

    assert len(result.approved) == 1
    assert any(r.blocked_by == "daily_trade_cap" for r in result.rejected)
    assert router.exit_allowed(
        side="sell",
        stale_state=True,
        market_context={"overall_market_bias": "bearish"},
        overnight_rejected=True,
    )


def test_stale_or_timeout_state_returns_no_approved_entries():
    router = DailyOpportunityRouter(RouterConfig())
    for kwargs, reason in (
        ({"stale_state": True}, "stale_state"),
        ({"feed_timeout": True}, "feed_timeout"),
        ({"missing_data": True}, "missing_data"),
    ):
        result = router.route([_opp("SAFE")], market_context={"overall_market_bias": "bullish"}, **kwargs)
        assert result.approved == []
        assert result.safe_mode is True
        assert result.safe_mode_reason == reason
        assert result.rejected[0].blocked_by == "setup_not_validated"


def test_bearish_context_rejects_new_entries_not_exits():
    router = DailyOpportunityRouter(RouterConfig())
    result = router.route([_opp("BEAR")], market_context={"overall_market_bias": "bearish"})

    assert result.approved == []
    assert result.rejected[0].blocked_by == "setup_not_validated"
    assert router.exit_allowed(side="sell", market_context={"overall_market_bias": "bearish"})


@pytest.mark.parametrize(
    "top_level, metadata",
    [
        ({"probability": 0.79}, {}),
        ({}, {"iv_crush_risk": 80}),
        ({}, {"dte": 3}),
        ({"score": 89}, {"protected_or_partialed": False}),
    ],
)
def test_overnight_call_gate_rejects_low_quality_calls(top_level, metadata):
    router = DailyOpportunityRouter(RouterConfig())
    md = {"dte": 30, "iv_crush_risk": 10, "protected_or_partialed": True, "active_exit_ladder": True}
    md.update(metadata)
    call_kwargs = {
        "asset_class": "option",
        "probability": 0.85,
        "expected_r": 3.0,
        "score": 95,
        "overnight_candidate": True,
        "metadata": md,
    }
    call_kwargs.update(top_level)
    call = _opp(
        "AAPL260117C00200000",
        **call_kwargs,
    )

    allowed, reason = router.overnight_allowed(call, {"overall_market_bias": "bullish"})

    assert allowed is False
    assert reason == "overnight_gate"


def test_overnight_share_gate_rejects_bearish_regime_break():
    router = DailyOpportunityRouter(RouterConfig())
    share = _opp(
        "MSFT",
        overnight_candidate=True,
        metadata={"profitable": True, "trend_intact": True, "thesis_backed": False},
    )

    allowed, reason = router.overnight_allowed(share, {"overall_market_bias": "bearish"})

    assert allowed is False
    assert reason == "overnight_gate"


def test_funnel_log_contains_required_metrics(tmp_path):
    router = DailyOpportunityRouter(RouterConfig(max_new_entries=1))
    result = router.route([_opp("A"), _opp("B")], market_context={"overall_market_bias": "bullish"})
    path = tmp_path / "funnel.json"

    router.write_funnel(path, result, actual_trades=1, realized_next_day_outcome={"checked": False})

    data = json.loads(path.read_text())
    for key in (
        "candidates",
        "approved_entries",
        "rejected_entries",
        "actual_trades",
        "overnight_approved_shares",
        "overnight_approved_calls",
        "realized_next_day_outcome",
    ):
        assert key in data


def test_funnel_logs_safe_mode_reason_blocked_by_zero_candidate_and_stale_feed(tmp_path):
    router = DailyOpportunityRouter(RouterConfig())

    zero = router.route([], market_context={"overall_market_bias": "bullish"})
    zero_path = router.write_funnel(tmp_path / "zero.json", zero)
    zero_data = json.loads(zero_path.read_text())
    assert zero_data["candidates"] == 0
    assert zero_data["zero_candidate_day"] is True
    assert zero_data["no_trade_reason"] == "zero_candidates"

    stale = router.route([_opp("STALE")], market_context={"overall_market_bias": "bullish"}, stale_state=True)
    stale_path = router.write_funnel(tmp_path / "stale.json", stale)
    stale_data = json.loads(stale_path.read_text())
    assert stale_data["safe_mode"] is True
    assert stale_data["safe_mode_reason"] == "stale_state"
    assert stale_data["no_trade_reason"] == "safe_mode:stale_state"
    assert stale_data["rejected"][0]["blocked_by"] == "setup_not_validated"
