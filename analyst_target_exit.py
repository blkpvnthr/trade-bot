#!/usr/bin/env python3
"""
analyst_target_exit.py — analyst price-target logic as an OPTIONS EXIT FILTER (not an entry signal).

For an OPEN long call, compares the current UNDERLYING price to analyst targets (high/avg/low) to
decide hold / tighten-trail / exit. Runs AFTER emergency exits (hard/theta stops) but BEFORE
profit-trailing. Fail-safe: if targets are stale/missing or < 3 estimates → do nothing (the
existing exit engine governs). Also exposes a downside check used to SKIP new entries.

Exit rules (long calls):
  price ≤ target_low                 → exit "below_low_analyst_target" (invalidation)
  price ≥ target_high * 0.98         → exit "near_high_analyst_target" (extreme/full profit)
  price ≥ target_avg * 0.98          → exit "near_avg_analyst_target" (profit exhaustion)
  upside_to_avg < 0.05               → exit "analyst_upside_exhausted"
  target_avg ≤ price < target_avg*.98→ tighten_trail "above_avg_target_tighten_trail"
  else                               → hold "targets_allow_hold"
"""
from __future__ import annotations

import json
import os
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "analyst_targets"
_CTX = ssl.create_default_context()
try:
    _CTX.load_default_certs()
except Exception:
    pass


def pct_to_target(target: Optional[float], price: float) -> Optional[float]:
    if target is None or price <= 0:
        return None
    return target / price - 1.0


def analyst_target_exit(price: float, target_high: Optional[float], target_avg: Optional[float],
                        target_low: Optional[float], estimates: Optional[int] = None) -> dict:
    """Decide hold / tighten-trail / exit for an OPEN long call.

    REVISED (analyst targets are noisy → give winners room): the AVERAGE target only TIGHTENS the
    trailing stop; HARD exits are reserved for the extremes.
        price ≤ low                 → exit  "below_low_analyst_target"   (invalidation)
        price ≥ high*0.98           → exit  "near_high_analyst_target"    (extreme/full profit)
        price ≥ avg                 → TIGHTEN trail "above_avg_target_tighten_trail" (NOT exit)
        else                        → hold
    (Profit-exhaustion <5% upside is now an ENTRY filter, not an exit — see predicts_downside.)
    """
    if price <= 0:
        return {"exit": False, "tighten_trail": False, "reason": "invalid_price"}
    if estimates is not None and estimates < 3:
        return {"exit": False, "tighten_trail": False, "reason": "not_enough_estimates"}
    if target_avg is None or target_high is None or target_low is None:
        return {"exit": False, "tighten_trail": False, "reason": "missing_targets"}

    u_avg = pct_to_target(target_avg, price)
    u_high = pct_to_target(target_high, price)
    d_low = pct_to_target(target_low, price)
    base = {"upside_to_avg": u_avg, "upside_to_high": u_high, "downside_to_low": d_low}

    if price <= target_low:
        return {"exit": True, "tighten_trail": False, "reason": "below_low_analyst_target", **base}
    if price >= target_high * 0.98:
        return {"exit": True, "tighten_trail": False, "reason": "near_high_analyst_target", **base}
    if price >= target_avg:
        return {"exit": False, "tighten_trail": True, "reason": "above_avg_target_tighten_trail", **base}
    return {"exit": False, "tighten_trail": False, "reason": "targets_allow_hold", **base}


def predicts_downside(price: float, target_avg: Optional[float], target_low: Optional[float],
                      estimates: Optional[int] = None, min_upside: float = 0.0) -> bool:
    """ENTRY filter: True if analysts predict downside (avg target ≤ price·(1+min_upside) or price
    already below the low target), with ≥3 estimates. False (don't skip) if targets missing."""
    if estimates is not None and estimates < 3:
        return False
    if target_avg is None or price <= 0:
        return False
    if target_low is not None and price <= target_low:
        return True
    return (target_avg / price - 1.0) <= min_upside


# ── fail-safe analyst-target fetcher (yfinance; cached per symbol/day) ───────────────────────────
def get_analyst_targets(symbol: str, max_age_hours: float = 24.0) -> Optional[dict]:
    """Return {'high','avg','low','n'} analyst price targets, or None on any failure/staleness.
    Source: yfinance (targetHighPrice/targetMeanPrice/targetLowPrice/numberOfAnalystOpinions)."""
    symbol = symbol.upper().strip()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{symbol}.json"
    if cache.exists():
        try:
            obj = json.loads(cache.read_text())
            if (time.time() - obj.get("_ts", 0)) <= max_age_hours * 3600:
                return obj.get("data")
        except Exception:
            pass
    data = None
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).get_info()
        hi = info.get("targetHighPrice"); avg = info.get("targetMeanPrice")
        lo = info.get("targetLowPrice"); n = info.get("numberOfAnalystOpinions")
        if avg and hi and lo:
            data = {"high": float(hi), "avg": float(avg), "low": float(lo),
                    "n": int(n) if n else None}
    except Exception:
        data = None
    try:
        cache.write_text(json.dumps({"_ts": time.time(), "data": data}))
    except Exception:
        pass
    return data


if __name__ == "__main__":
    # unit tests of the decision logic (no network)
    P = 100.0
    cases = [
        ("below low", analyst_target_exit(100, 130, 115, 100, 10)),
        ("at low", analyst_target_exit(95, 130, 115, 100, 10)),
        ("near high", analyst_target_exit(128, 130, 115, 95, 10)),
        ("near avg", analyst_target_exit(114, 130, 115, 95, 10)),
        ("upside exhausted", analyst_target_exit(112, 116, 115, 95, 10)),
        ("above avg trail", analyst_target_exit(116, 130, 115, 95, 10)),
        ("hold", analyst_target_exit(105, 130, 115, 95, 10)),
        ("too few est", analyst_target_exit(105, 130, 115, 95, 2)),
        ("missing", analyst_target_exit(105, None, 115, 95, 10)),
    ]
    for name, r in cases:
        print(f"{name:18s} exit={r['exit']!s:5} trail={r['tighten_trail']!s:5} {r['reason']}")
    print("downside(price=120,avg=115):", predicts_downside(120, 115, 95, 10))
    print("downside(price=100,avg=130):", predicts_downside(100, 130, 95, 10))
