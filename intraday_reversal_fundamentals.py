#!/usr/bin/env python3
"""Fundamental gate for the intraday reversal engine — fundamentals IN ADDITION to the
technical (analyst/oscillator + price-reclaim) gates when deciding a position.

A long reversal should be a fundamentally-OK name, not a falling knife. This adds an
optional confirmation on a few core fundamentals (revenue growth, net margin, leverage,
valuation), reusing buffett_accumulation_layer's AV-backed FundamentalData fetch.

  * ``fundamentals_ok`` is PURE/testable (takes a dict of fundamentals + cfg).
  * ``fetch_fundamentals`` is the only impure part (lazy buffett/AV fetch, read-only).
  * ``maybe_enrich_fundamentals`` is the gated entry point (no-op unless the gate is on).

Confirmation filter only — never places orders, never sizes, never bypasses the engine
gates or the v2 router. Fail-OPEN: missing fundamentals do not block a trade.
Gated by INTRADAY_REVERSAL_FUNDAMENTAL_GATE (default off); thresholds on EngineConfig.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("intraday_reversal_fundamentals")

_FIELDS = ("revenue_growth_yoy", "eps_growth_yoy", "net_margin", "operating_margin",
           "debt_to_equity", "pe_ratio", "roe", "free_cash_flow")


def fundamentals_ok(f: Optional[dict], cfg: Any) -> tuple[bool, str]:
    """Should we take the position on fundamentals? Pure. Blocks on clearly weak reads;
    every threshold has a lenient default (only non-declining revenue is enforced by
    default). Fail-OPEN when fundamentals are missing."""
    if not f:
        return True, "no_fundamentals"
    rg = f.get("revenue_growth_yoy")
    if rg is not None and rg < cfg.min_revenue_growth:
        return False, f"weak_revenue_growth:{rg:.3f}<{cfg.min_revenue_growth}"
    nm = f.get("net_margin")
    if nm is not None and cfg.require_positive_net_margin and nm <= 0:
        return False, f"negative_net_margin:{nm:.3f}"
    de = f.get("debt_to_equity")
    if de is not None and cfg.max_debt_to_equity > 0 and de > cfg.max_debt_to_equity:
        return False, f"over_leveraged_de:{de:.2f}>{cfg.max_debt_to_equity}"
    pe = f.get("pe_ratio")
    if pe is not None and cfg.max_pe_ratio > 0 and 0 < pe > cfg.max_pe_ratio:
        return False, f"rich_valuation_pe:{pe:.1f}>{cfg.max_pe_ratio}"
    return True, "fundamentals_ok"


def fetch_fundamentals(symbol: str) -> Optional[dict]:
    """Read-only fundamentals for ``symbol`` via buffett_accumulation_layer (AV OVERVIEW,
    cached). Returns a small dict of the gate-relevant fields, or None on failure."""
    try:
        import buffett_accumulation_layer as bal
        fd = bal.get_fundamentals(symbol, bal.BALConfig())
        if fd is None:
            return None
        return {k: getattr(fd, k) for k in _FIELDS if hasattr(fd, k)}
    except Exception as exc:
        logger.debug("fundamentals fetch failed for %s: %s", symbol, exc)
        return None


def maybe_enrich_fundamentals(rows: list, cfg: Any,
                              fetch_fn: Optional[Any] = None) -> list:
    """Attach fundamentals to each row when the fundamental gate is on (so score_candidate
    can apply it). No-op when disabled; rows already carrying ``fundamentals`` are left as-is.
    Fail-safe; ``fetch_fn`` is injectable for tests."""
    if not rows or not getattr(cfg, "fundamental_gate_enabled", False):
        return rows
    fn = fetch_fn or fetch_fundamentals
    for inp in rows:
        if getattr(inp, "fundamentals", None) is None:
            try:
                inp.fundamentals = fn(inp.symbol)
            except Exception as exc:
                logger.debug("fundamental enrich failed for %s: %s", inp.symbol, exc)
    return rows
