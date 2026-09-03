#!/usr/bin/env python3
"""Top-losers universe source — find the day's biggest losers / most-active names from
AlphaVantage TOP_GAINERS_LOSERS and TRACK them as a reversal watchlist.

The reversal engine wants WEAK names that subsequently reclaim. The losers list is that
weak universe; the watchlist persists them across days so a name that dropped today can be
re-checked later for a green-today reclaim. This module:
  * PURE: parse_top_losers (payload -> rows) and to_reversal_inputs (rows -> ReversalInput).
  * impure: fetch_top_losers (read-only AV; never trades).
  * stateful: track_losers / tracked_symbols persist a JSON watchlist.

A top-losers row carries symbol/price/chg/volume only — analyst/oscillator ratings,
perf_1w, and the reclaim triggers must still be added by the screener cross-ref + bar
enrichment before a row can pass the engine gates. This is a candidate SOURCE, not a
decision: no orders, no broker, nothing bypasses the engine gates or the v2 router.

Env: INTRADAY_REVERSAL_TOP_LOSERS_ENABLED, INTRADAY_REVERSAL_TOP_LOSERS_WATCHLIST,
     INTRADAY_REVERSAL_TOP_LOSERS_SECTIONS (csv of top_losers,most_actively_traded).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from intraday_reversal_engine import ReversalInput
from intraday_reversal_universe_feed import _parse_number, _parse_pct

logger = logging.getLogger("intraday_reversal_top_losers")


def parse_top_losers(payload: dict, sections: tuple[str, ...] = ("top_losers",)) -> list[dict]:
    """Extract {ticker, price, chg_pct, volume} rows from the chosen TOP_GAINERS_LOSERS
    sections (top_losers and/or most_actively_traded). Pure; tolerant of missing fields."""
    rows: list[dict] = []
    for sec in sections:
        for r in (payload.get(sec) or []):
            sym = str(r.get("ticker") or "").upper().strip()
            if not sym:
                continue
            rows.append({"ticker": sym, "price": _parse_number(r.get("price")),
                         "chg_pct": _parse_pct(r.get("change_percentage")),
                         "volume": _parse_number(r.get("volume")), "section": sec})
    return rows


def to_reversal_inputs(rows: list[dict]) -> list[ReversalInput]:
    """Map parsed loser rows to ReversalInput (universe seeds). analyst/os ratings, perf_1w,
    and reclaim triggers are intentionally left empty — the screener cross-ref + bar
    enrichment fill them before the engine gates run. Skips rows missing price/volume."""
    out: list[ReversalInput] = []
    for r in rows:
        if r.get("price") in (None, 0) or r.get("volume") in (None, 0):
            continue
        out.append(ReversalInput(
            symbol=r["ticker"], price=float(r["price"]), volume=float(r["volume"]),
            avg_volume=0.0, chg_pct=float(r.get("chg_pct") or 0.0), perf_1w=0.0,
            analyst_rating="", os_rating=""))
    return out


def fetch_top_losers(sections: tuple[str, ...] = ("top_losers",),
                     entitlement: Optional[str] = None) -> list[ReversalInput]:
    """Read-only AV fetch of TOP_GAINERS_LOSERS -> ReversalInput losers. entitlement in
    {None(historical/EOD), 'realtime', 'delayed'}. Returns [] on failure."""
    try:
        import av_catalyst
        params = {"function": "TOP_GAINERS_LOSERS"}
        if entitlement:
            params["entitlement"] = entitlement
        payload = av_catalyst._get(params, f"top_gainers_losers_{entitlement or 'eod'}")
        if not payload:
            return []
        return to_reversal_inputs(parse_top_losers(payload, sections))
    except Exception as exc:
        logger.debug("top-losers fetch failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# watchlist tracking (persist losers across days for reversal follow-up)
# ---------------------------------------------------------------------------
def track_losers(rows: list[ReversalInput], path: str, today: str) -> dict:
    """Merge today's losers into the JSON watchlist at ``path`` (symbol -> {first_seen,
    last_seen, times_seen, last_chg_pct, last_price}). Returns the updated watchlist."""
    p = Path(path)
    wl: dict[str, Any] = {}
    if p.exists():
        try:
            wl = json.loads(p.read_text())
        except Exception:
            wl = {}
    for inp in rows:
        e = wl.get(inp.symbol, {"first_seen": today, "times_seen": 0})
        e.update({"last_seen": today, "times_seen": int(e.get("times_seen", 0)) + 1,
                  "last_chg_pct": inp.chg_pct, "last_price": inp.price})
        wl[inp.symbol] = e
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(wl, indent=2, default=str))
    logger.info("top-losers watchlist: %d tracked (%d added/updated today)", len(wl), len(rows))
    return wl


def tracked_symbols(path: str) -> list[str]:
    """Symbols currently on the reversal watchlist."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        return sorted(json.loads(p.read_text()).keys())
    except Exception:
        return []


def maybe_top_losers_universe() -> list[ReversalInput]:
    """Gated entry point: fetch + track today's losers, return them as universe seeds.
    No-op ([]) unless INTRADAY_REVERSAL_TOP_LOSERS_ENABLED."""
    from intraday_reversal_engine import _env_bool
    if not _env_bool("INTRADAY_REVERSAL_TOP_LOSERS_ENABLED", False):
        return []
    secs = tuple(s.strip() for s in os.getenv(
        "INTRADAY_REVERSAL_TOP_LOSERS_SECTIONS", "top_losers").split(",") if s.strip())
    rows = fetch_top_losers(sections=secs)
    wl_path = os.getenv("INTRADAY_REVERSAL_TOP_LOSERS_WATCHLIST", "")
    if rows and wl_path:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        track_losers(rows, wl_path, datetime.now(ZoneInfo("America/New_York")).date().isoformat())
    return rows
