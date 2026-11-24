"""The persistent baseline universe for the autonomous rotating scanner.

These 30 securities are the engine's permanent watchlist core. They are ALWAYS
members of the master universe and can never be *permanently* removed by the
Opportunity Finder -- they may only rotate out of the bounded active subscription
window and are re-evaluated for reinsertion every rotation. The master universe is
``BASELINE ∪ discovered ∪ monitoring`` and is not capped at 30; only the active
IEX subscription window is (``ACTIVE_UNIVERSE_SIZE``).
"""

from __future__ import annotations

BASELINE_UNIVERSE: tuple[str, ...] = (
    # ELV (Elevance Health, formerly Anthem/ANTM) replaces the original "ANT":
    # ANT is tradable on Alpaca but has NO IEX market data, so it could never
    # become eligible and would waste a baseline rotation slot. ELV is the
    # intended healthcare-insurer exposure (with HUM/UNH), liquid, IEX-covered.
    "ELV", "GPN", "DVA", "HUM", "UNH",
    "HCA", "UHS", "GIS", "WMT", "COST",
    "NVDA", "AMD", "AVGO", "AAPL", "MU",
    "TSLA", "SPY", "QQQ", "DIA", "IWM",
    "XLP", "XLV", "GLD", "SQQQ", "SOXS",
    "SPXU", "SDOW", "TZA", "TQQQ", "SOXL",
)

# Hard cap on the active IEX subscription window (NOT the master universe size).
ACTIVE_UNIVERSE_SIZE: int = 30

# Uppercase-normalized membership set for O(1) baseline checks.
BASELINE_SET: frozenset[str] = frozenset(s.upper() for s in BASELINE_UNIVERSE)


def is_baseline(symbol: object) -> bool:
    """True if ``symbol`` is a baseline member (case/whitespace-insensitive)."""
    if not isinstance(symbol, str):
        return False
    s = symbol.strip().upper()
    return bool(s) and s in BASELINE_SET
