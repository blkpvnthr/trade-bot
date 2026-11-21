"""Consolidation tests: single scanner / single publisher ownership.

Pins the architectural invariant that market_scan.scanner.MarketScanner (started
by trade.py) is the ONE production scanner + market-context publisher, and the
standalone stream_market_analyzer is research-only and non-competing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import market_scan
from market_scan import paths

REPO = Path(__file__).resolve().parents[1]


def test_production_scanner_entrypoint_exists():
    assert hasattr(market_scan, "start_market_scanner")


def test_trade_py_wires_production_scanner_and_not_the_research_analyzer():
    src = (REPO / "trade.py").read_text(encoding="utf-8")
    # The production scanner IS started by trade.py...
    assert "market_scan.start_market_scanner(" in src
    # ...and the standalone research analyzer is NOT launched by the engine.
    assert "stream_market_analyzer" not in src


def test_single_publisher_research_output_is_distinct_from_production_context():
    research = paths.research_output_dir()
    for mode in ("paper", "live"):
        prod = paths.live_context_path(mode)
        assert research != prod.parent / prod.name
        assert (research / "live_context.json") != prod   # no overlap on the canonical file


def test_research_analyzer_publishes_only_under_research_dir():
    import market_scan.stream_market_analyzer as sma
    import inspect
    src = inspect.getsource(sma.main)
    # main() must route output through the research dir helper, never a hardcoded
    # canonical/nested path.
    assert "research_output_dir()" in src
    assert "parent / 'market_scan'" not in src   # the old nested-dir bug is gone


def test_scanner_publishes_to_canonical_live_context(tmp_path):
    # The production publisher writes the canonical live_context path (by mode).
    from market_scan.publisher import UniversePublisher
    from market_scan.config import MarketScanConfig
    import universe
    p = UniversePublisher(tmp_path / "universe.txt", paths.live_context_path("paper"),
                          MarketScanConfig(), mode="paper")
    # path identity is the contract under test (no write needed)
    assert p.context_path == paths.live_context_path("paper")
