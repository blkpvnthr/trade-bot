"""Tests for the persistent baseline universe (Stage A4)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_scan import baseline


_EXPECTED = (
    "ANT", "GPN", "DVA", "HUM", "UNH",
    "HCA", "UHS", "GIS", "WMT", "COST",
    "NVDA", "AMD", "AVGO", "AAPL", "MU",
    "TSLA", "SPY", "QQQ", "DIA", "IWM",
    "XLP", "XLV", "GLD", "SQQQ", "SOXS",
    "SPXU", "SDOW", "TZA", "TQQQ", "SOXL",
)


def test_baseline_universe_is_the_exact_30_tickers():
    assert baseline.BASELINE_UNIVERSE == _EXPECTED
    assert len(baseline.BASELINE_UNIVERSE) == 30


def test_baseline_has_no_duplicates():
    assert len(set(baseline.BASELINE_UNIVERSE)) == 30


def test_active_universe_size_is_30():
    assert baseline.ACTIVE_UNIVERSE_SIZE == 30


def test_baseline_set_is_uppercase_normalized():
    assert baseline.BASELINE_SET == {s.upper() for s in _EXPECTED}


def test_is_baseline_is_case_insensitive():
    assert baseline.is_baseline("nvda") is True
    assert baseline.is_baseline("NVDA") is True
    assert baseline.is_baseline(" spy ") is True       # tolerant of whitespace
    assert baseline.is_baseline("FAKE") is False
    assert baseline.is_baseline("") is False
    assert baseline.is_baseline(None) is False
