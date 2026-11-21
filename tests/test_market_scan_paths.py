"""Tests for canonical, cwd-independent market_scan output path resolution."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_scan import paths
import market_scan.scanner as scanner

REPO = Path(__file__).resolve().parents[1]


def test_live_context_path_canonical_location_and_mode():
    assert paths.live_context_path("paper") == REPO / "market_scan" / "live_context_paper.json"
    assert paths.live_context_path("live") == REPO / "market_scan" / "live_context_live.json"
    # mode normalization: anything not "live" -> paper (never an unexpected file)
    assert paths.live_context_path("PAPER") == paths.live_context_path("paper")
    assert paths.live_context_path("weird") == paths.live_context_path("paper")


def test_universe_path_is_repo_root():
    assert paths.universe_path() == REPO / "universe.txt"


def test_research_dir_is_not_nested_and_distinct_from_production():
    rd = paths.research_output_dir()
    assert rd == REPO / "market_scan" / "research"
    # NEVER the old nested market_scan/market_scan bug
    assert rd != REPO / "market_scan" / "market_scan"
    # research output must not collide with the canonical production context file
    assert (rd / "live_context.json") != paths.live_context_path("paper")
    assert (rd / "live_context.json") != paths.live_context_path("live")
    assert rd.is_dir()


def test_paths_are_cwd_independent(tmp_path):
    before = (paths.live_context_path("paper"), paths.universe_path(),
              paths.research_output_dir())
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        after = (paths.live_context_path("paper"), paths.universe_path(),
                 paths.research_output_dir())
    finally:
        os.chdir(cwd)
    assert before == after  # resolution does not depend on the working directory


def test_scanner_delegates_to_canonical_paths():
    assert scanner._live_context_path("paper") == paths.live_context_path("paper")
    assert scanner._live_context_path("live") == paths.live_context_path("live")
    assert Path(scanner.DEFAULT_UNIVERSE_PATH) == paths.universe_path()
