#!/usr/bin/env python3
"""Live universe-feed adapter for the intraday reversal engine.

Converts a TradingView / generic screener export (CSV or JSON) into a list of
``intraday_reversal_engine.ReversalInput`` so the already-wired v2 router can scan
real candidates. This module ONLY normalizes — it makes NO trade decisions, places
NO orders, touches NO broker. The engine's strict universe gates and the v2 router
remain the sole decision/execution path.

Sources (first that exists wins):
  1. CSV  via ``INTRADAY_REVERSAL_UNIVERSE_CSV``
  2. JSON via ``INTRADAY_REVERSAL_UNIVERSE_JSON``  (list, or {"rows": [...]})
  3. inline JSON via ``INTRADAY_REVERSAL_UNIVERSE_INLINE``

Column names are matched case/spacing/punctuation-insensitively against the aliases
below; values tolerate "$", commas, "K/M/B/T" suffixes, "%", and rating casing.

CLI (dry-run, prints normalized rows; NO orders):
  python intraday_reversal_universe_feed.py --csv export.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from intraday_reversal_engine import ReversalInput, _finite, rel_vol_5m_of

logger = logging.getLogger("intraday_reversal_universe_feed")

# canonical ReversalInput field -> set of NORMALIZED column aliases (see _norm_key)
_ALIASES: dict[str, set[str]] = {
    "symbol": {"symbol", "ticker", "name"},
    "price": {"last", "price", "close", "lastprice"},
    "volume": {"volume", "vol", "vol24h"},
    "chg_pct": {"change", "chg", "changepct", "chgpct", "changepercent", "percentchange",
                "chgpercent", "change1d", "chg1d"},
    "perf_1w": {"perfw", "perf1w", "1w", "1wperf", "weeklyperformance", "perfweek", "perfwk",
                "performanceweek", "perf1week", "1week", "weekperf", "perfweekly"},
    "analyst_rating": {"analystrating", "analyst", "analystrec", "recommendation"},
    "os_rating": {"osrating", "os", "technicalrating", "oscillatorsrating", "oscillatorrating",
                  "technical", "techrating", "oscrating"},
    "relative_volume": {"relativevolume", "rvol", "relvolume", "relativevolume10dcalc",
                        "relvol", "relativevol"},
    "vol_5m": {"5mvolume", "volume5m", "vol5m", "5minvolume", "5minutevolume", "fiveminvolume",
               "5minvol", "volume5min"},
    "avg_volume": {"averagevolume", "avgvolume", "avgvol", "averagevolume10day", "avgvolume10d",
                   "avgvolume10dcalc", "averagevol"},
    "rel_vol_5m": {"relvol5m", "5mrelvol", "relativevolume5m", "5mrelativevolume", "relvolume5m"},
}
_KEY2FIELD: dict[str, str] = {alias: field for field, aliases in _ALIASES.items() for alias in aliases}

REQUIRED = ("symbol", "price", "volume", "chg_pct", "perf_1w", "analyst_rating", "os_rating")
_NULLS = {"", "n/a", "na", "nan", "none", "null", "-", "—", "–"}

# Optional engine signal fields passed through verbatim WHEN an (enriched / bar-derived)
# export already carries them. A plain screener export won't have these — the live
# system attaches intraday bars (or these fields) before the engine detects reclaims.
_PASS_BOOL = ("vwap_reclaim", "rsi_rising", "rsi_cross_40", "close_gt_open",
              "sweep_reclaim", "orb_reclaim", "above_ema20_5m", "recovering")
_PASS_FLOAT = ("prev_day_high", "atr", "extended_atr", "near_low_atr", "pdh", "previous_day_high")
_PASS_FIELD_BY_KEY = {**{_n: _n for _n in _PASS_BOOL},
                      **{re.sub(r"[^a-z0-9]", "", f): f for f in _PASS_BOOL},
                      **{re.sub(r"[^a-z0-9]", "", f): ("prev_day_high" if f in ("pdh", "previous_day_high") else f)
                         for f in _PASS_FLOAT}}


# ---------------------------------------------------------------------------
# parsing helpers (tolerant; never raise)
# ---------------------------------------------------------------------------
def _norm_key(k: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(k).strip().lower())


def _parse_number(v: Any) -> Optional[float]:
    """'3.4M' -> 3_400_000; '820K' -> 820_000; '12,500,000' -> 12_500_000; '$5.2' -> 5.2."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "").replace("$", "").replace("+", "")
    if s.lower() in _NULLS:
        return None
    m = re.fullmatch(r"(-?\d*\.?\d+)\s*([kmbt])?", s, flags=re.I)
    if m:
        mult = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}.get((m.group(2) or "").lower(), 1.0)
        val = float(m.group(1)) * mult
        return float(round(val)) if mult > 1.0 else val   # K/M/B counts are whole; avoid fp dust
    try:
        return float(s)
    except Exception:
        return None


def _parse_pct(v: Any) -> Optional[float]:
    """'-3.25%' -> -3.25; '+1.4' -> 1.4."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace("%", "").replace(",", "").replace("+", "")
    if s.lower() in _NULLS:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _parse_bool(v: Any) -> Optional[bool]:
    if v is None or isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in _NULLS:
        return None
    if s in {"1", "true", "yes", "y", "t"}:
        return True
    if s in {"0", "false", "no", "n", "f"}:
        return False
    return None


def _parse_rating(v: Any) -> str:
    """'Strong Buy' / 'strong_buy' / 'STRONG BUY' / 'StrongBuy' -> 'strong buy'."""
    if v is None:
        return ""
    s = re.sub(r"[\s_\-]+", " ", str(v).strip().lower()).strip()
    if s in _NULLS:
        return ""
    # split camel/glued "strongbuy" -> "strong buy"
    if s in {"strongbuy"}:
        return "strong buy"
    if s in {"strongsell"}:
        return "strong sell"
    return s


# ---------------------------------------------------------------------------
# row -> ReversalInput
# ---------------------------------------------------------------------------
def _normalize_row(raw: dict) -> dict:
    """Map arbitrary export columns onto canonical field names."""
    norm: dict[str, Any] = {}
    for k, v in raw.items():
        field = _KEY2FIELD.get(_norm_key(k))
        if field and norm.get(field) in (None, ""):
            norm[field] = v
    return norm


def _row_to_input(raw: dict) -> tuple[Optional[ReversalInput], list[str]]:
    norm = _normalize_row(raw)
    sym = str(norm.get("symbol", "") or "").strip().upper()
    if ":" in sym:                       # TradingView "NASDAQ:AAPL" -> "AAPL"
        sym = sym.split(":")[-1].strip()
    vals = {
        "symbol": sym,
        "price": _parse_number(norm.get("price")),
        "volume": _parse_number(norm.get("volume")),
        "chg_pct": _parse_pct(norm.get("chg_pct")),
        "perf_1w": _parse_pct(norm.get("perf_1w")),
        "analyst_rating": _parse_rating(norm.get("analyst_rating")),
        "os_rating": _parse_rating(norm.get("os_rating")),
    }
    missing = [f for f in REQUIRED if vals.get(f) in (None, "")]
    if missing:
        return None, missing
    inp = ReversalInput(
        symbol=sym, price=float(vals["price"]), volume=float(vals["volume"]),
        avg_volume=_parse_number(norm.get("avg_volume")) or 0.0,
        chg_pct=float(vals["chg_pct"]), perf_1w=float(vals["perf_1w"]),
        analyst_rating=vals["analyst_rating"], os_rating=vals["os_rating"],
        rvol=_parse_number(norm.get("relative_volume")),
        vol_5m=_parse_number(norm.get("vol_5m")),
        rel_vol_5m=_parse_number(norm.get("rel_vol_5m")))
    # Pass through any pre-computed engine signal fields present in an enriched export.
    for k, v in raw.items():
        field = _PASS_FIELD_BY_KEY.get(_norm_key(k))
        if not field or v in (None, ""):
            continue
        if field in _PASS_BOOL:
            b = _parse_bool(v)
            if b is not None:
                setattr(inp, field, b)
        else:
            f = _parse_number(v)
            if f is not None:
                setattr(inp, field, f)
    return inp, []


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def _csv_rows(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def _json_rows(path_or_text: str, is_path: bool) -> list[dict]:
    data = json.loads(Path(path_or_text).read_text() if is_path else path_or_text)
    rows = data.get("rows", data) if isinstance(data, dict) else data
    return list(rows) if isinstance(rows, list) else []


def _vol5_key(i: ReversalInput) -> float:
    return -_finite(rel_vol_5m_of(i))   # 5-minute RELATIVE volume desc


def load_universe(csv_path: Optional[str] = None, json_path: Optional[str] = None,
                  inline: Optional[str] = None) -> list[ReversalInput]:
    """Load + normalize a screener export into ReversalInput objects, sorted by
    5-minute RELATIVE volume desc. Pure: no trade decisions, no orders, no broker. The
    engine's strict gates still run downstream. Returns [] when no source exists."""
    csv_path = csv_path if csv_path is not None else os.getenv("INTRADAY_REVERSAL_UNIVERSE_CSV", "")
    json_path = json_path if json_path is not None else os.getenv("INTRADAY_REVERSAL_UNIVERSE_JSON", "")
    inline = inline if inline is not None else os.getenv("INTRADAY_REVERSAL_UNIVERSE_INLINE", "")

    src = ""
    raw: list[dict] = []
    try:
        if csv_path and Path(csv_path).exists():
            src, raw = csv_path, _csv_rows(csv_path)
        elif json_path and Path(json_path).exists():
            src, raw = json_path, _json_rows(json_path, is_path=True)
        elif inline:
            src, raw = "<inline env>", _json_rows(inline, is_path=False)
        else:
            logger.debug("universe feed: no source configured (CSV/JSON/inline)")
            return []
    except Exception as exc:
        logger.warning("universe feed: failed to read %s: %s", src or "source", exc)
        return []

    inputs: list[ReversalInput] = []
    missing_by_sym: dict[str, list[str]] = {}
    for r in raw:
        inp, missing = _row_to_input(r)
        if inp is None:
            label = str(r.get("Symbol") or r.get("Ticker") or r.get("symbol") or "?")
            missing_by_sym[label] = missing
            continue
        inputs.append(inp)
    inputs.sort(key=_vol5_key)

    logger.info("universe feed: source=%s loaded=%d normalized=%d skipped=%d",
                src, len(raw), len(inputs), len(missing_by_sym))
    if missing_by_sym:
        logger.info("universe feed: rows missing required fields: %s",
                    dict(list(missing_by_sym.items())[:20]))
    logger.info("universe feed: top 10 by rel_vol_5m: %s",
                [(i.symbol, round(rel_vol_5m_of(i), 2)) for i in inputs[:10]])
    return inputs


# ---------------------------------------------------------------------------
# CLI (dry-run)
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Intraday reversal universe-feed adapter (dry-run)")
    p.add_argument("--csv", default="")
    p.add_argument("--json", default="")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    rows = load_universe(csv_path=args.csv or None, json_path=args.json or None)
    print(f"normalized {len(rows)} ReversalInput rows (sorted by 5m volume desc):")
    for i in rows[:25]:
        print(f"  {i.symbol:<8} px={i.price:<8} vol={int(i.volume):>12,} 5mVol="
              f"{('%d' % i.vol_5m) if i.vol_5m is not None else '-':>10} chg={i.chg_pct:<7} "
              f"1w={i.perf_1w:<7} analyst={i.analyst_rating!r} os={i.os_rating!r}")
    print("(dry-run — no orders, no broker; the engine's gates + v2 router decide trades)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
