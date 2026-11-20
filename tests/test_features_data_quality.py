"""Tests for features.py's data-quality screening: detect_duplicate_bars
and flag_extreme_returns. Both SURFACE candidates for review; neither
excludes anything -- that decision requires evidence (see the MRNA
investigation: a genuine, news-corroborated +177% day must never be
silently dropped merely for being extreme).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import features as feat


def _bars(rows):
    return pd.DataFrame(rows, columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])


def test_detect_duplicate_bars_finds_repeated_symbol_timestamp():
    df = _bars([
        ("AAPL", "2024-01-01", 1, 1, 1, 100, 1000),
        ("AAPL", "2024-01-01", 1, 1, 1, 101, 1000),  # duplicate timestamp
        ("AAPL", "2024-01-02", 1, 1, 1, 102, 1000),
    ])
    dupes = feat.detect_duplicate_bars(df)
    assert len(dupes) == 2
    assert set(dupes["close"]) == {100, 101}


def test_detect_duplicate_bars_empty_when_clean():
    df = _bars([
        ("AAPL", "2024-01-01", 1, 1, 1, 100, 1000),
        ("AAPL", "2024-01-02", 1, 1, 1, 102, 1000),
    ])
    assert feat.detect_duplicate_bars(df).empty


def test_flag_extreme_returns_finds_above_threshold():
    df = _bars([
        ("MRNA", "2026-08-17", 1, 1, 1, 64.45, 140_533),
        ("MRNA", "2026-08-18", 1, 1, 1, 62.93, 129_956),
        ("MRNA", "2026-08-19", 1, 1, 1, 174.27, 3_673_339),  # +176.9%
        ("MRNA", "2026-08-20", 1, 1, 1, 133.23, 1_739_618),
    ])
    flagged = feat.flag_extreme_returns(df, threshold=0.5)
    assert len(flagged) == 1
    assert flagged.iloc[0]["timestamp"] == "2026-08-19"
    assert flagged.iloc[0]["return"] > 1.5


def test_flag_extreme_returns_does_not_flag_moderate_moves():
    df = _bars([
        ("AAPL", "2024-01-01", 1, 1, 1, 100, 1000),
        ("AAPL", "2024-01-02", 1, 1, 1, 105, 1000),  # +5%
    ])
    assert feat.flag_extreme_returns(df, threshold=0.5).empty


def test_flag_extreme_returns_never_excludes_anything_itself():
    """This function only surfaces candidates -- the input row count must
    never shrink as a side effect of calling it."""
    df = _bars([
        ("MRNA", "2026-08-18", 1, 1, 1, 62.93, 1),
        ("MRNA", "2026-08-19", 1, 1, 1, 174.27, 1),
    ])
    feat.flag_extreme_returns(df, threshold=0.5)
    assert len(df) == 2  # original untouched
