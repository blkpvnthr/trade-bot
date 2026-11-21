"""Canonical, working-directory-independent output paths for all market_scan
components.

Single source of truth so the production scanner (scanner.py), the loss monitor
and hot-reload in trade.py, and the standalone research analyzer never disagree,
never create nested `market_scan/market_scan/` directories, and never compete for
the canonical market-context files. Every path is anchored to this file's
location (``Path(__file__)``), so results do not depend on the current working
directory.
"""

from __future__ import annotations

from pathlib import Path

MARKET_SCAN_DIR = Path(__file__).resolve().parent          # .../<repo>/market_scan
PROJECT_ROOT = MARKET_SCAN_DIR.parent                      # .../<repo>


def universe_path() -> Path:
    """The single canonical trading universe file at the repo root."""
    return PROJECT_ROOT / "universe.txt"


def live_context_path(mode: str) -> Path:
    """Canonical PRODUCTION market-context file, namespaced by paper/live mode.

    Written by the production scanner (scanner.py) and consumed by trade.py.
    """
    m = "live" if str(mode).lower() == "live" else "paper"
    return MARKET_SCAN_DIR / f"live_context_{m}.json"


def research_output_dir() -> Path:
    """Separate output namespace for the standalone research/diagnostic analyzer.

    Deliberately distinct from :func:`live_context_path` so the research analyzer
    NEVER overwrites or competes with the production scanner's canonical
    market-context files. Created on demand. (Fixes the prior
    ``market_scan/market_scan/`` nested-directory bug.)
    """
    d = MARKET_SCAN_DIR / "research"
    d.mkdir(parents=True, exist_ok=True)
    return d
