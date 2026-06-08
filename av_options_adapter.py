#!/usr/bin/env python3
"""av_options_adapter.py — real Alpha Vantage option chains for call contract selection.

Unifies two AV sources into the exact chain format the Alpha Call Engine consumes
(so `evaluate_call_contracts` / `select_best_contract` select from REAL contracts with
REAL greeks + IV, instead of synthetic Black-Scholes ones):

    • REALTIME_OPTIONS   → live selection (reuses options.py)
    • HISTORICAL_OPTIONS → backtest selection, as-of any past date (15+ yrs, with greeks)

Engine chain-entry keys produced (robust to the engine's accepted aliases):
    symbol, type, expiration_date, strike, bid_price/bid, ask_price/ask,
    delta, implied_volatility, open_interest, volume  (+ gamma/theta/vega/rho passthrough)

Historical responses are cached to backtest_cache/avopt/<SYM>_<YYYYMMDD>.json so repeat
backtests never re-hit the API. options.py's score() is also exposed as an optional
liquidity tiebreaker on top of the engine's own selection.
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

AV_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
AV_URL = "https://www.alphavantage.co/query"
_CACHE_DIR = PROJECT_ROOT / "backtest_cache" / "avopt"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_MEM: Dict[str, List[dict]] = {}   # in-process memo keyed by f"{sym}:{date or 'rt'}"


def _f(v, d=0.0):
    try:
        return float(v) if v not in (None, "") else d
    except Exception:
        return d


def _i(v, d=0):
    try:
        return int(float(v)) if v not in (None, "") else d
    except Exception:
        return d


def _row_to_engine_contract(row: dict) -> Optional[dict]:
    """Map one AV option row → engine chain-entry dict. Calls only; None if not a call."""
    typ = str(row.get("type") or row.get("option_type") or row.get("contract_type") or "").lower()
    if "call" not in typ:
        return None
    exp = row.get("expiration") or row.get("expiration_date")
    strike = _f(row.get("strike") or row.get("strike_price"), None)
    if not exp or strike in (None, 0):
        return None
    bid, ask = _f(row.get("bid") or row.get("bid_price")), _f(row.get("ask") or row.get("ask_price"))
    return {
        "symbol": row.get("contractID") or row.get("contract") or row.get("symbol"),
        "type": "call",
        "expiration_date": str(exp)[:10],
        "strike": float(strike),
        "bid_price": bid, "bid": bid,
        "ask_price": ask, "ask": ask,
        "delta": _f(row.get("delta")),
        "implied_volatility": _f(row.get("implied_volatility")),
        "open_interest": _i(row.get("open_interest") or row.get("openInterest")),
        "volume": _i(row.get("volume")),
        "gamma": _f(row.get("gamma")), "theta": _f(row.get("theta")),
        "vega": _f(row.get("vega")), "rho": _f(row.get("rho")),
    }


def _av_get(params: dict, retries: int = 3) -> dict:
    if not AV_KEY:
        raise RuntimeError("ALPHA_VANTAGE_API_KEY not set in .env")
    params = {**params, "apikey": AV_KEY}
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(AV_URL, params=params, timeout=30)
            r.raise_for_status()
            d = r.json()
            for bad in ("Error Message", "Note", "Information"):
                if bad in d:
                    raise RuntimeError(f"{bad}: {str(d[bad])[:120]}")
            return d
        except Exception as e:
            last = e
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"AV request failed after {retries} tries: {last}")


def historical_chain(symbol: str, on_date, calls_only: bool = True) -> List[dict]:
    """Real historical call chain for `symbol` as-of `on_date` (date or 'YYYY-MM-DD').

    One AV call returns the full chain (all expiries) for that date. Cached to disk.
    Returns engine-format contract dicts.
    """
    if isinstance(on_date, (date, datetime)):
        ds = on_date.strftime("%Y-%m-%d")
    else:
        ds = str(on_date)[:10]
    mem_key = f"{symbol}:{ds}"
    if mem_key in _MEM:
        return _MEM[mem_key]
    cache_f = _CACHE_DIR / f"{symbol}_{ds.replace('-', '')}.json"
    if cache_f.exists():
        rows = json.loads(cache_f.read_text())
    else:
        d = _av_get({"function": "HISTORICAL_OPTIONS", "symbol": symbol, "date": ds})
        rows = d.get("data") or []
        if not isinstance(rows, list):
            rows = []
        cache_f.write_text(json.dumps(rows))
    chain = []
    for row in rows:
        c = _row_to_engine_contract(row)
        if c and (not calls_only or c["type"] == "call"):
            chain.append(c)
    _MEM[mem_key] = chain
    return chain


def realtime_chain(symbol: str, calls_only: bool = True) -> List[dict]:
    """Real-time call chain via options.py (REALTIME_OPTIONS), engine-format."""
    mem_key = f"{symbol}:rt"
    if mem_key in _MEM:
        return _MEM[mem_key]
    import options as O
    rows = O.get_realtime_options(symbol)
    chain = []
    for row in rows:
        c = _row_to_engine_contract(row)
        if c and (not calls_only or c["type"] == "call"):
            chain.append(c)
    _MEM[mem_key] = chain
    return chain


def engine_chain(symbol: str, on_date=None) -> List[dict]:
    """Engine-format call chain: historical if `on_date` given, else realtime."""
    return historical_chain(symbol, on_date) if on_date else realtime_chain(symbol)


def liquidity_score(c: dict, underlying_price: float = 0.0) -> float:
    """options.py-style liquidity score (volume/OI · delta · spread · ATM) as a tiebreaker."""
    oi = c.get("open_interest") or 0
    vol = c.get("volume") or 0
    voi = (vol / oi) if oi > 0 else 0.0
    delta = abs(c.get("delta") or 0.0)
    mid = ((c.get("bid") or 0) + (c.get("ask") or 0)) / 2
    spread_pct = ((c.get("ask", 0) - c.get("bid", 0)) / mid) if mid > 0 else 1.0
    spread_q = max(0.0, 1 - spread_pct)
    atm_q = 1.0
    if underlying_price > 0 and c.get("strike"):
        atm_q = max(0.0, 1 - abs(c["strike"] - underlying_price) / underlying_price)
    return round(voi * 40 + delta * 30 + spread_q * 20 + atm_q * 10, 4)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Fetch a real AV call chain (historical or realtime) in engine format.")
    p.add_argument("symbol")
    p.add_argument("--date", default=None, help="YYYY-MM-DD for historical; omit for realtime")
    p.add_argument("--min-dte", type=int, default=21)
    p.add_argument("--max-dte", type=int, default=60)
    a = p.parse_args()
    chain = engine_chain(a.symbol, a.date)
    ref = date.fromisoformat(a.date) if a.date else date.today()
    print(f"{a.symbol} {'@'+a.date if a.date else 'REALTIME'}: {len(chain)} call contracts")
    band = [c for c in chain
            if a.min_dte <= (date.fromisoformat(c['expiration_date']) - ref).days <= a.max_dte]
    band.sort(key=lambda c: -liquidity_score(c))
    print(f"  in {a.min_dte}-{a.max_dte} DTE: {len(band)}; top by liquidity score:")
    for c in band[:8]:
        dte = (date.fromisoformat(c['expiration_date']) - ref).days
        print(f"   {c['symbol']:<22} K={c['strike']:<8} dte={dte:<3} delta={c['delta']:.3f} "
              f"iv={c['implied_volatility']:.3f} bid={c['bid']:.2f} ask={c['ask']:.2f} "
              f"oi={c['open_interest']} vol={c['volume']} score={liquidity_score(c)}")
