"""Tests for the deterministic (rules-based) market-context classifier.

There is no external LLM API: classify_market_context derives bias / permission /
risk from the raw signals via fallback_classification, validates event-flag
integrity, and tags the result with source="rules" + a timestamp. No network, no
API key -- the trading engine runs the same with or without any credentials.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import market_context_builder as mcb


def test_classify_is_deterministic_and_tagged_rules():
    ctx = {
        "overnight_futures": {"ES": {"change_pct": 0.4}, "NQ": {"change_pct": 0.6},
                              "VIX": {"change_pct": -3.0}},
        "key_etf_risk_signals": {"VXX": {"change_pct": -1.0}},
        "sector_strength": {"XLK": {"change_pct": 0.9}},
    }
    a = mcb.classify_market_context(ctx)
    b = mcb.classify_market_context(ctx)
    assert a["classification_source"] == "rules"
    assert a.get("classified_at")
    # Deterministic bias/permission/risk (ignore the timestamp field).
    for k in ("overall_market_bias", "trade_permission", "risk_level"):
        assert a[k] == b[k]


def test_no_api_key_needed_no_network():
    # Even with no credentials set, classification succeeds (no LLM call).
    import os
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        os.environ.pop(var, None)
    out = mcb.classify_market_context({"overnight_futures": {}})
    assert out["classification_source"] == "rules"
    # Required downstream fields the engine consumes are always present.
    for k in ("overall_market_bias", "trade_permission", "risk_level"):
        assert k in out


def test_event_flags_not_fabricated():
    # No evidence supplied -> every macro event flag must be False.
    out = mcb.classify_market_context({"overnight_futures": {"ES": {"change_pct": -1.0}}})
    flags = out.get("event_flags", {})
    assert flags, "event_flags should be present"
    assert not any(bool(v) for v in flags.values()), f"no flag should be set without evidence: {flags}"


def test_negative_futures_are_not_bullish():
    # Sharp risk-off futures should not classify as bullish/normal.
    out = mcb.classify_market_context({
        "overnight_futures": {"ES": {"change_pct": -2.5}, "NQ": {"change_pct": -3.0},
                              "VIX": {"change_pct": 20.0}},
    })
    assert out["overall_market_bias"] != "bullish"


def test_no_llm_symbols_remain_in_module():
    # The Anthropic/OpenAI integration is fully removed.
    for removed in ("classify_with_claude", "classify_with_openai", "CLAUDE_MODEL",
                    "MARKET_CONTEXT_DISABLE_LLM", "MARKET_CONTEXT_FAIL_LLM_CLOSED"):
        assert not hasattr(mcb, removed), f"{removed} should have been removed"
    assert mcb.CLASSIFIER_NAME == "deterministic_rules"
