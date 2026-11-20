"""Instrument registry (V6 spec sections 5-6).

A persistent, versioned catalog of candidate instruments with their underlying exposure,
orientation, leverage, and -- crucially -- liquidity/eligibility fields that are populated by
measurement, never assumed. Every mapping in DIRECTIONAL_GROUPS is a CANDIDATE requiring
verification (tradability, inception, liquidity, broker eligibility) before it can be used.

This extends the validated V4 REGISTRY (amps_v4_regime.REGISTRY, which is already exercised by
the V4/V5 backtests) with the V5/V6 single-stock and index ETF candidates. We re-export the V4
REGISTRY entries verbatim so downstream V4/V5 code keeps working unchanged.

No network I/O here. `data_available()` reports which symbols actually have cached bars so the
diagnostics never silently fabricate coverage.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BARS_DIR = PROJECT_ROOT / "data" / "cache" / "bars_multi_tf"

# Re-use the already-validated V4 registry as the authoritative core.
import amps_v4_regime as _R

ORIENT_BULL = "bull"
ORIENT_BEAR = "bear"


@dataclass
class Instrument:
    """One registry row. Liquidity/eligibility fields default to None = 'not yet measured'
    (never assumed true). last_validated_at is set when a measurement populates them."""
    symbol: str
    underlying_symbol: str          # the thing whose direction drives this instrument
    benchmark: str                  # reference used for signals (often == underlying)
    directional_orientation: str    # 'bull' | 'bear'
    daily_leverage: float           # signed (e.g. +3, -3, +2, -1)
    category: str                   # 'index' | 'semi' | 'single' | 'equity'
    pair: str                       # opposing-exposure group key (prevents simultaneous conflict)
    # measured / verified fields (None until populated)
    asset_class: str = "us_equity"
    tradable: Optional[bool] = None
    fractionable: Optional[bool] = None
    shortable: Optional[bool] = None
    inception_date: Optional[str] = None
    average_daily_volume: Optional[float] = None
    average_daily_dollar_volume: Optional[float] = None
    median_spread_bps: Optional[float] = None
    liquidity_score: Optional[float] = None
    intraday_eligible: Optional[bool] = None
    swing_eligible: Optional[bool] = None
    overnight_eligible: Optional[bool] = None
    last_validated_at: Optional[str] = None

    @property
    def inverse_status(self) -> bool:
        return self.daily_leverage < 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["inverse_status"] = self.inverse_status
        return d


# ---------------------------------------------------------------- directional exposure groups (sec 6)
# Every entry is a CANDIDATE requiring verification. (underlying, benchmark) drive signals; the ETF
# itself is what we would trade. Leverage signs: bull>0, bear<0.
# cat: index | semi | single. These supersede nothing in V4 -- V4 entries are re-exported below.
_SPEC = [
    # symbol, underlying, benchmark, orient, lev, cat, pair
    ("UPRO", "SPY", "SPY", ORIENT_BULL, 3, "index", "SPY"),
    ("SPXL", "SPY", "SPY", ORIENT_BULL, 3, "index", "SPY"),
    ("SPXU", "SPY", "SPY", ORIENT_BEAR, -3, "index", "SPY"),
    ("SPXS", "SPY", "SPY", ORIENT_BEAR, -3, "index", "SPY"),
    ("SDS", "SPY", "SPY", ORIENT_BEAR, -2, "index", "SPY"),
    ("SH", "SPY", "SPY", ORIENT_BEAR, -1, "index", "SPY"),
    ("TQQQ", "QQQ", "QQQ", ORIENT_BULL, 3, "index", "QQQ"),
    ("QLD", "QQQ", "QQQ", ORIENT_BULL, 2, "index", "QQQ"),
    ("SQQQ", "QQQ", "QQQ", ORIENT_BEAR, -3, "index", "QQQ"),
    ("QID", "QQQ", "QQQ", ORIENT_BEAR, -2, "index", "QQQ"),
    ("PSQ", "QQQ", "QQQ", ORIENT_BEAR, -1, "index", "QQQ"),
    ("UDOW", "DIA", "DIA", ORIENT_BULL, 3, "index", "DIA"),
    ("DDM", "DIA", "DIA", ORIENT_BULL, 2, "index", "DIA"),
    ("SDOW", "DIA", "DIA", ORIENT_BEAR, -3, "index", "DIA"),
    ("DXD", "DIA", "DIA", ORIENT_BEAR, -2, "index", "DIA"),
    ("DOG", "DIA", "DIA", ORIENT_BEAR, -1, "index", "DIA"),
    ("TNA", "IWM", "IWM", ORIENT_BULL, 3, "index", "IWM"),
    ("UWM", "IWM", "IWM", ORIENT_BULL, 2, "index", "IWM"),
    ("TZA", "IWM", "IWM", ORIENT_BEAR, -3, "index", "IWM"),
    ("TWM", "IWM", "IWM", ORIENT_BEAR, -2, "index", "IWM"),
    ("RWM", "IWM", "IWM", ORIENT_BEAR, -1, "index", "IWM"),
    ("SOXL", "SMH", "SMH", ORIENT_BULL, 3, "semi", "SMH"),
    ("SOXS", "SMH", "SMH", ORIENT_BEAR, -3, "semi", "SMH"),
    ("TECL", "XLK", "XLK", ORIENT_BULL, 3, "sector", "XLK"),
    ("TECS", "XLK", "XLK", ORIENT_BEAR, -3, "sector", "XLK"),
    ("NVDL", "NVDA", "NVDA", ORIENT_BULL, 2, "single", "NVDA"),
    ("NVD", "NVDA", "NVDA", ORIENT_BEAR, -1, "single", "NVDA"),
    ("NVDD", "NVDA", "NVDA", ORIENT_BEAR, -1, "single", "NVDA"),
    ("TSLL", "TSLA", "TSLA", ORIENT_BULL, 2, "single", "TSLA"),
    ("TSLQ", "TSLA", "TSLA", ORIENT_BEAR, -1, "single", "TSLA"),
    ("AAPU", "AAPL", "AAPL", ORIENT_BULL, 2, "single", "AAPL"),
    ("AAPD", "AAPL", "AAPL", ORIENT_BEAR, -1, "single", "AAPL"),
    ("AMZU", "AMZN", "AMZN", ORIENT_BULL, 2, "single", "AMZN"),
    ("AMZD", "AMZN", "AMZN", ORIENT_BEAR, -1, "single", "AMZN"),
    ("MSFU", "MSFT", "MSFT", ORIENT_BULL, 2, "single", "MSFT"),
    ("MSFD", "MSFT", "MSFT", ORIENT_BEAR, -1, "single", "MSFT"),
    ("MSTU", "MSTR", "MSTR", ORIENT_BULL, 2, "single", "MSTR"),
    ("MSTZ", "MSTR", "MSTR", ORIENT_BEAR, -1, "single", "MSTR"),
    ("AMDL", "AMD", "AMD", ORIENT_BULL, 2, "single", "AMD"),
    ("AMDS", "AMD", "AMD", ORIENT_BEAR, -1, "single", "AMD"),
    ("CONL", "COIN", "COIN", ORIENT_BULL, 2, "single", "COIN"),
    ("CONI", "COIN", "COIN", ORIENT_BEAR, -1, "single", "COIN"),
    ("DRIP", "XOP", "XOP", ORIENT_BEAR, -2, "sector", "XOP"),
]

REGISTRY: dict[str, Instrument] = {}
for sym, und, bench, orient, lev, cat, pair in _SPEC:
    REGISTRY[sym] = Instrument(symbol=sym, underlying_symbol=und, benchmark=bench,
                               directional_orientation=orient, daily_leverage=float(lev),
                               category=cat, pair=pair)

# References/underlyings used for signals only (never traded as ETF candidates here).
REFERENCES = sorted({ins.benchmark for ins in REGISTRY.values()} |
                    {ins.underlying_symbol for ins in REGISTRY.values()})

# Opposing-exposure groups: pair key -> {bull:[...], bear:[...]} (sec 6, sec 25 conflict prevention).
DIRECTIONAL_GROUPS: dict[str, dict[str, list[str]]] = {}
for ins in REGISTRY.values():
    g = DIRECTIONAL_GROUPS.setdefault(ins.pair, {"bull": [], "bear": []})
    g[ins.directional_orientation].append(ins.symbol)


def data_available(symbol: str) -> bool:
    """True iff cached 5m bars exist for `symbol` (never fabricate coverage)."""
    return bool(glob.glob(str(BARS_DIR / f"{symbol}_5m_*.csv")))


def tradable_universe(require_data: bool = True) -> list[str]:
    """Candidate ETFs, optionally restricted to those with cached bars."""
    syms = [s for s in REGISTRY if not require_data or data_available(s)]
    return sorted(syms)


def opposing_symbols(symbol: str) -> list[str]:
    """Symbols in the same exposure group with the OPPOSITE orientation (conflict set)."""
    if symbol not in REGISTRY:
        return []
    ins = REGISTRY[symbol]
    other = "bear" if ins.directional_orientation == "bull" else "bull"
    return sorted(DIRECTIONAL_GROUPS.get(ins.pair, {}).get(other, []))


def coverage_report() -> dict:
    """Honest coverage: which candidate symbols/underlyings actually have cached bars."""
    etfs = {s: data_available(s) for s in REGISTRY}
    refs = {r: data_available(r) for r in REFERENCES}
    return {
        "n_candidates": len(REGISTRY),
        "n_candidates_with_data": sum(etfs.values()),
        "n_references": len(REFERENCES),
        "n_references_with_data": sum(refs.values()),
        "candidates_with_data": sorted(s for s, ok in etfs.items() if ok),
        "candidates_missing_data": sorted(s for s, ok in etfs.items() if not ok),
        "references_with_data": sorted(r for r, ok in refs.items() if ok),
        "references_missing_data": sorted(r for r, ok in refs.items() if not ok),
    }
