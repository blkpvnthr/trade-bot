#!/usr/bin/env python3
"""
av_historical_options.py — cached fetcher for Alpha Vantage HISTORICAL_OPTIONS (premium).

Returns the REAL historical option chain for (symbol, date): actual bid/ask, OI, volume,
IV and greeks (delta/gamma/theta/vega/rho) — 15+ yrs back. Used to re-validate the EV
options engine on real chains (real skew/spreads/discrete strikes) instead of BS-synthetic
ones, with realized P&L read from the actual chain at exit (no model).

Disk cache: data/cache/av_historical_options/{SYMBOL}_{YYYY-MM-DD}.json (chains are immutable
history, so cached forever). Rate-limited + fail-safe (returns [] on any error/quota).

API: get_chain(symbol, date) -> list[dict] normalised to the engine's contract shape.
"""
from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date as _date, datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "av_historical_options"
_CTX = ssl.create_default_context()
try:
    _CTX.load_default_certs()
except Exception:
    pass

_last_req = 0.0
_RATE = float(os.environ.get("AV_HIST_OPT_RATE_SEC", "0.9"))   # premium-safe (~66/min)
_quota_hit = False


def _api_key() -> str:
    key = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()
    if key:
        return key
    env = PROJECT_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ALPHA_VANTAGE_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def _norm(c: dict, asof: str) -> Optional[dict]:
    """Normalise a raw AV contract → engine shape; compute dte. None on bad row."""
    def f(k, d=0.0):
        try:
            return float(c.get(k, d) or d)
        except Exception:
            return d
    try:
        exp = str(c.get("expiration") or "")[:10]
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - datetime.strptime(asof, "%Y-%m-%d").date()).days
        bid, ask = f("bid"), f("ask")
        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else f("mark")
        return {
            "symbol": str(c.get("contractID") or ""), "type": str(c.get("type") or "").lower(),
            "expiration_date": exp, "dte": dte, "strike_price": f("strike"),
            "delta": f("delta"), "gamma": f("gamma"), "theta": f("theta"),
            "vega": f("vega"), "rho": f("rho"), "implied_volatility": f("implied_volatility"),
            "bid_price": bid, "ask_price": ask, "mid": mid, "mark": f("mark"),
            "open_interest": int(f("open_interest")), "volume": int(f("volume")),
        }
    except Exception:
        return None


def get_chain(symbol: str, date: str, calls_only: bool = False,
              use_cache: bool = True) -> List[dict]:
    """Real historical option chain for (symbol, date='YYYY-MM-DD'). [] on failure/quota."""
    global _last_req, _quota_hit
    symbol = symbol.upper().strip()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{symbol}_{date}.json"
    if use_cache and cache.exists():
        try:
            rows = json.loads(cache.read_text())
            return [r for r in rows if (not calls_only or r.get("type") == "call")]
        except Exception:
            pass
    if _quota_hit:
        return []
    key = _api_key()
    if not key:
        return []
    wait = _RATE - (time.time() - _last_req)
    if wait > 0:
        time.sleep(wait)
    url = "https://www.alphavantage.co/query?" + urllib.parse.urlencode(
        {"function": "HISTORICAL_OPTIONS", "symbol": symbol, "date": date, "apikey": key})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "QuantumQuant/1.0"})
        with urllib.request.urlopen(req, timeout=30, context=_CTX) as r:
            payload = json.loads(r.read().decode("utf-8", "replace"))
        _last_req = time.time()
    except Exception:
        _last_req = time.time()
        return []
    if payload.get("Note") or payload.get("Information"):
        _quota_hit = True
        return []
    raw = payload.get("data") or []
    rows = [n for c in raw if isinstance(c, dict) for n in [_norm(c, date)] if n]
    try:
        cache.write_text(json.dumps(rows))
    except Exception:
        pass
    return [r for r in rows if (not calls_only or r.get("type") == "call")]


def lookup_contract(symbol: str, date: str, contract_id: str) -> Optional[dict]:
    """Find one contractID in the (symbol, date) chain — for realized exit pricing."""
    for c in get_chain(symbol, date):
        if c.get("symbol") == contract_id:
            return c
    return None


if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "IBM"
    dt = sys.argv[2] if len(sys.argv) > 2 else "2024-06-03"
    ch = get_chain(sym, dt, calls_only=True)
    print(f"{sym} {dt}: {len(ch)} calls")
    if ch:
        c = ch[len(ch) // 2]
        print(json.dumps(c, indent=2))
