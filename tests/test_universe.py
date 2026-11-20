"""Tests for universe.py: validated, deterministic universe construction.

Covers Objective 1 acceptance criteria:
  - no invalid or duplicate symbol survives universe construction
  - the size cap is applied after normalization/dedup/validation, not before
  - ordinary equities, index/leveraged/inverse ETFs are classified correctly
  - correlated instruments (QQQ/TQQQ/SQQQ, SOXX/SOXL/SOXS) share an exposure group
  - exclusions are logged with a reason and never raise
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import universe
from universe import (
    InstrumentType,
    build_universe,
    classify_symbol,
    default_normalize,
    log_excluded,
)


def test_duplicate_symbol_is_excluded_not_counted_twice():
    result = build_universe(["AMD", "AAPL", "AMD"], max_symbols=30)
    assert result.symbols.count("AMD") == 1
    assert result.symbols == ["AMD", "AAPL"]
    assert any(e.symbol == "AMD" and e.reason == "duplicate" for e in result.excluded)


def test_known_invalid_symbol_nvd_is_excluded():
    result = build_universe(["NVDA", "NVD", "AAPL"], max_symbols=30)
    assert "NVD" not in result.symbols
    assert "NVDA" in result.symbols
    reasons = {e.symbol: e.reason for e in result.excluded}
    assert "NVD" in reasons and "invalid" in reasons["NVD"]


def test_malformed_symbol_is_excluded():
    result = build_universe(["AAPL", "HPE/PC", "", None, "TOOLONGTICKER"], max_symbols=30)
    assert result.symbols == ["AAPL"]
    reasons = {e.symbol for e in result.excluded}
    assert "HPE-PC" in reasons or "HPE/PC" in reasons or len(result.excluded) >= 2


def test_cap_applied_after_dedup_and_validation_not_before():
    """Regression test for the original bug: cap-before-dedup let a duplicate
    AMD and an invalid NVD consume universe slots that distinct valid symbols
    should have had, silently shrinking the realized universe below the cap.
    """
    candidates = [
        "QQQ", "TQQQ", "SQQQ", "SPY", "DIA", "IWM", "XLE", "SOXX", "SMH",
        "AMD", "TECL", "SOXL", "AAPL", "MU", "AMD", "NVD", "ATOM", "TSLL",
        "TECS", "JEPQ", "QTUM", "HUMA", "RGTI", "QBTS", "DRIP", "IONQ", "LASE",
        "LUNR", "IAG", "SOXS", "TTD",
    ]
    result = build_universe(candidates, max_symbols=30)

    assert "NVD" not in result.symbols
    assert result.symbols.count("AMD") == 1
    # 31 raw candidates - 1 duplicate - 1 invalid = 29 distinct valid symbols,
    # all of which fit under the cap of 30 (nothing legitimate gets truncated).
    assert len(result.symbols) == 29
    assert "TTD" in result.symbols  # would have been dropped by the old cap-before-dedup bug


def test_higher_quality_symbol_not_displaced_by_cap():
    """A later, otherwise-valid distinct symbol should only be dropped by the
    cap if the universe already has max_symbols distinct valid entries --
    never because an earlier duplicate or invalid entry wasted a slot."""
    candidates = ["A", "A", "INVALID1/", "B", "C"]
    result = build_universe(candidates, max_symbols=2)
    assert result.symbols == ["A", "B"]
    assert any(e.symbol == "C" and "max_symbols" in e.reason for e in result.excluded)


@pytest.mark.parametrize(
    "symbol,expected_type",
    [
        ("QQQ", InstrumentType.INDEX_ETF),
        ("SPY", InstrumentType.INDEX_ETF),
        ("TQQQ", InstrumentType.LEVERAGED_ETF),
        ("SOXL", InstrumentType.LEVERAGED_ETF),
        ("SQQQ", InstrumentType.INVERSE_ETF),
        ("SOXS", InstrumentType.INVERSE_ETF),
        ("AAPL", InstrumentType.EQUITY),
        ("ZZZZ", InstrumentType.EQUITY),  # unknown defaults to equity
    ],
)
def test_classification(symbol, expected_type):
    assert classify_symbol(symbol).instrument_type == expected_type


def test_leveraged_and_inverse_share_exposure_group_with_underlying():
    result = build_universe(["QQQ", "TQQQ", "SQQQ", "AAPL"], max_symbols=30)
    groups = result.exposure_groups()
    assert set(groups["nasdaq100"]) == {"QQQ", "TQQQ", "SQQQ"}
    assert groups["AAPL"] == ["AAPL"]  # unknown equity gets its own singleton group


def test_semis_leveraged_and_inverse_share_exposure_group():
    result = build_universe(["SOXX", "SOXL", "SOXS", "SMH"], max_symbols=30)
    groups = result.exposure_groups()
    assert set(groups["semiconductors"]) == {"SOXX", "SOXL", "SOXS", "SMH"}


def test_default_normalize_handles_case_whitespace_and_slash():
    assert default_normalize(" aapl ") == "AAPL"
    assert default_normalize("hpe/pc") == "HPE-PC"


def test_log_excluded_does_not_raise_and_calls_log_fn():
    calls = []
    result = build_universe(["AMD", "AMD", "NVD"], max_symbols=30)
    log_excluded(result.excluded, log_fn=calls.append)
    assert len(calls) == len(result.excluded) == 2
    assert all("excluded" in c for c in calls)


def test_zero_max_symbols_means_unbounded():
    result = build_universe(["A", "B", "C"], max_symbols=0)
    assert result.symbols == ["A", "B", "C"]


def test_unregistered_equity_defaults_to_singleton_group():
    universe.register_equity_sectors({})
    assert universe.classify_symbol("ZZZZ").exposure_group == "ZZZZ"


def test_registered_equity_sectors_group_by_sector():
    universe.register_equity_sectors({"AAPL": "Information Technology", "MSFT": "Information Technology", "JPM": "Financials"})
    try:
        aapl = universe.classify_symbol("AAPL")
        msft = universe.classify_symbol("MSFT")
        jpm = universe.classify_symbol("JPM")
        assert aapl.exposure_group == msft.exposure_group == "sector:Information Technology"
        assert jpm.exposure_group == "sector:Financials"
        assert aapl.instrument_type == InstrumentType.EQUITY
    finally:
        universe.register_equity_sectors({})


def test_register_equity_sectors_clears_previous_mapping():
    universe.register_equity_sectors({"AAPL": "Information Technology"})
    universe.register_equity_sectors({"MSFT": "Information Technology"})
    try:
        assert universe.classify_symbol("AAPL").exposure_group == "AAPL"  # no longer registered -> singleton
        assert universe.classify_symbol("MSFT").exposure_group == "sector:Information Technology"
    finally:
        universe.register_equity_sectors({})


def test_etf_registry_takes_precedence_over_sector_map():
    universe.register_equity_sectors({"QQQ": "Information Technology"})
    try:
        info = universe.classify_symbol("QQQ")
        assert info.instrument_type == InstrumentType.INDEX_ETF
        assert info.exposure_group == "nasdaq100"  # not the injected sector
    finally:
        universe.register_equity_sectors({})


def test_register_equity_sectors_normalizes_keys():
    universe.register_equity_sectors({" aapl ": "Information Technology"})
    try:
        assert universe.classify_symbol("AAPL").exposure_group == "sector:Information Technology"
    finally:
        universe.register_equity_sectors({})
