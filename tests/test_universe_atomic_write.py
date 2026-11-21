"""Tests for universe.write_canonical_universe — the atomic universe.txt writer.

The real-time scanner rewrites the canonical universe continuously. The loader
(universe.load_canonical_universe) is fail-closed: a missing or empty file
raises. Therefore the writer MUST be atomic (temp + os.replace) and MUST never
leave an empty or torn file. These tests pin that contract.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import universe
from universe import load_canonical_universe, write_canonical_universe


def test_round_trips_through_loader_preserving_order(tmp_path):
    p = tmp_path / "universe.txt"
    written = write_canonical_universe(["NVDA", "AMD", "GPN", "SOXS"], p)
    assert written == ["NVDA", "AMD", "GPN", "SOXS"]
    assert load_canonical_universe(p) == ["NVDA", "AMD", "GPN", "SOXS"]


def test_normalizes_and_dedupes_first_occurrence_wins(tmp_path):
    p = tmp_path / "universe.txt"
    written = write_canonical_universe(["'nvda'", "amd", "NVDA", " gpn ", "BRK/B"], p)
    assert written == ["NVDA", "AMD", "GPN", "BRK-B"]
    assert load_canonical_universe(p) == ["NVDA", "AMD", "GPN", "BRK-B"]


def test_empty_after_normalization_raises_and_leaves_existing_file_untouched(tmp_path):
    p = tmp_path / "universe.txt"
    write_canonical_universe(["NVDA", "AMD"], p)
    before = p.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        write_canonical_universe(["", None, "  ", "'  '"], p)
    # Fail-closed: the prior good universe must survive a rejected write.
    assert p.read_text(encoding="utf-8") == before
    assert load_canonical_universe(p) == ["NVDA", "AMD"]


def test_write_is_atomic_no_temp_file_left_behind(tmp_path):
    p = tmp_path / "universe.txt"
    write_canonical_universe(["NVDA", "AMD", "GPN"], p)
    leftovers = [q.name for q in tmp_path.iterdir() if q.name != "universe.txt"]
    assert leftovers == [], f"temp artifacts left behind: {leftovers}"


def test_overwrite_replaces_previous_universe_not_appends(tmp_path):
    p = tmp_path / "universe.txt"
    write_canonical_universe(["AAA", "BBB", "CCC"], p)
    write_canonical_universe(["XXX", "YYY"], p)
    assert load_canonical_universe(p) == ["XXX", "YYY"]


def test_min_symbols_floor_rejects_short_universe(tmp_path):
    p = tmp_path / "universe.txt"
    with pytest.raises(ValueError):
        write_canonical_universe(["AAA"], p, min_symbols=2)


def test_creates_parent_dir_if_missing(tmp_path):
    p = tmp_path / "nested" / "dir" / "universe.txt"
    write_canonical_universe(["NVDA"], p)
    assert load_canonical_universe(p) == ["NVDA"]
