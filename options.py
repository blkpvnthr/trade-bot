#!/usr/bin/env python3
import os
import csv
import time
import math
import argparse
import pandas as pd
from datetime import datetime, date
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
AV_BASE_URL = "https://www.alphavantage.co/query"

WATCHLIST = [
   "TECS", "SOXS", "NVD", "SQQQ", "TZA", "SPXS", "SPXU",
]
#pd.read_csv("negbeta_2026-06-06.csv")["Symbol"].tolist()



def die(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def av_get(params: Dict[str, Any]) -> Dict[str, Any]:
    if not ALPHA_VANTAGE_API_KEY:
        die("Missing ALPHA_VANTAGE_API_KEY in .env")

    params = dict(params)
    params["apikey"] = ALPHA_VANTAGE_API_KEY

    r = requests.get(AV_BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if "Error Message" in data:
        raise RuntimeError(data["Error Message"])

    if "Note" in data:
        raise RuntimeError(data["Note"])

    if "Information" in data:
        raise RuntimeError(data["Information"])

    return data


def get_realtime_options(symbol: str) -> List[Dict[str, Any]]:
    data = av_get({
        "function": "REALTIME_OPTIONS",
        "symbol": symbol,
        "require_greeks": "true",
    })

    rows = data.get("data") or data.get("options") or []

    if not isinstance(rows, list):
        return []

    return rows


def parse_expiration(row: Dict[str, Any]) -> Optional[date]:
    raw = row.get("expiration") or row.get("expiration_date")
    if not raw:
        return None

    try:
        return datetime.strptime(str(raw), "%Y-%m-%d").date()
    except Exception:
        return None


def get_underlying_price(rows: List[Dict[str, Any]]) -> Optional[float]:
    for row in rows:
        for key in ["underlying_price", "underlyingPrice", "last_underlying_price", "stock_price"]:
            px = to_float(row.get(key))
            if px and px > 0:
                return px

    calls = []
    for row in rows:
        cp = str(row.get("type") or row.get("option_type") or row.get("contract_type") or "").lower()
        strike = to_float(row.get("strike") or row.get("strike_price"))
        delta = to_float(row.get("delta"))
        if "call" in cp and strike and delta and 0.45 <= delta <= 0.95:
            calls.append(strike)

    if calls:
        return sorted(calls)[len(calls) // 2]

    strikes = []
    for row in rows:
        strike = to_float(row.get("strike") or row.get("strike_price"))
        if strike:
            strikes.append(strike)

    if strikes:
        return sorted(strikes)[len(strikes) // 2]

    return None


def is_call(row: Dict[str, Any]) -> bool:
    typ = str(row.get("type") or row.get("option_type") or row.get("contract_type") or "").lower()
    contract = str(row.get("contractID") or row.get("contract") or row.get("symbol") or "")

    if "call" in typ:
        return True

    if len(contract) >= 15 and "C" in contract[-9:]:
        return True

    return False


def normalize_contract(symbol: str, row: Dict[str, Any], underlying_price: float) -> Optional[Dict[str, Any]]:
    if not is_call(row):
        return None

    contract = row.get("contractID") or row.get("contract") or row.get("symbol")
    expiration = parse_expiration(row)
    strike = to_float(row.get("strike") or row.get("strike_price"))
    bid = to_float(row.get("bid") or row.get("bid_price"))
    ask = to_float(row.get("ask") or row.get("ask_price"))
    delta = to_float(row.get("delta"))
    volume = to_int(row.get("volume"))
    open_interest = to_int(row.get("open_interest") or row.get("openInterest"))

    if not contract or not expiration or strike is None or bid is None or ask is None:
        return None

    if bid <= 0 or ask <= 0 or ask < bid:
        return None

    mid = (bid + ask) / 2
    spread = ask - bid
    spread_pct = spread / mid if mid > 0 else math.inf
    dte = (expiration - date.today()).days
    atm_distance = abs(strike - underlying_price)
    atm_distance_pct = atm_distance / underlying_price if underlying_price > 0 else math.inf

    return {
        "underlying": symbol,
        "contract": contract,
        "expiration": expiration.isoformat(),
        "dte": dte,
        "type": "call",
        "strike": round(strike, 4),
        "underlying_price": round(underlying_price, 4),
        "atm_distance": round(atm_distance, 4),
        "atm_distance_pct": round(atm_distance_pct, 6),
        "bid": round(bid, 4),
        "ask": round(ask, 4),
        "mid": round(mid, 4),
        "spread": round(spread, 4),
        "spread_pct": round(spread_pct, 6),
        "delta": round(delta, 6) if delta is not None else None,
        "volume": volume,
        "open_interest": open_interest,
        "volume_oi_ratio": round(volume / open_interest, 6) if open_interest > 0 else None,
    }


def passes_filters(row: Dict[str, Any], args: argparse.Namespace) -> bool:
    if row["dte"] < args.min_dte or row["dte"] > args.max_dte:
        return False

    if row["atm_distance_pct"] > args.atm_pct:
        return False

    if row["delta"] is None or row["delta"] <= args.min_delta:
        return False

    if row["volume"] <= row["open_interest"]:
        return False

    if row["volume"] < args.min_volume:
        return False

    if row["open_interest"] < args.min_open_interest:
        return False

    if row["mid"] < args.min_price or row["mid"] > args.max_price:
        return False

    if row["spread_pct"] > args.max_spread_pct:
        return False

    if row["spread"] > args.max_spread_abs:
        return False

    return True


def score(row: Dict[str, Any]) -> float:
    voi = row["volume_oi_ratio"] or 0
    delta = row["delta"] or 0
    spread_quality = max(0, 1 - row["spread_pct"])
    atm_quality = max(0, 1 - row["atm_distance_pct"])

    return round((voi * 40) + (delta * 30) + (spread_quality * 20) + (atm_quality * 10), 6)


def scan_symbol(symbol: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    print(f"Scanning {symbol}...")

    try:
        raw_rows = get_realtime_options(symbol)
    except Exception as e:
        print(f"  ERROR: {e}")
        return []

    if not raw_rows:
        print(f"  No options returned")
        return []

    underlying_price = get_underlying_price(raw_rows)

    if not underlying_price:
        print(f"  Could not determine underlying price")
        return []

    contracts = []

    for raw in raw_rows:
        normalized = normalize_contract(symbol, raw, underlying_price)
        if normalized and passes_filters(normalized, args):
            normalized["score"] = score(normalized)
            contracts.append(normalized)

    contracts.sort(key=lambda r: (-r["score"], r["atm_distance_pct"], r["spread_pct"], -(r["volume_oi_ratio"] or 0)))
    print(f"  Found {len(contracts)} matching contracts")
    return contracts


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    fieldnames = [
        "underlying",
        "contract",
        "expiration",
        "dte",
        "type",
        "strike",
        "underlying_price",
        "atm_distance",
        "atm_distance_pct",
        "bid",
        "ask",
        "mid",
        "spread",
        "spread_pct",
        "delta",
        "volume",
        "open_interest",
        "volume_oi_ratio",
        "score",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    print(f"Wrote {len(rows)} contracts to {path}")


def parse_symbols(args: argparse.Namespace) -> List[str]:
    if args.symbols:
        return [s.upper().strip() for s in args.symbols if s.strip()]

    if args.watchlist_file:
        with open(args.watchlist_file, "r") as f:
            text = f.read()

        symbols = []
        for part in text.replace("\n", ",").split(","):
            s = part.strip().upper().replace('"', "").replace("'", "")
            if s:
                symbols.append(s)

        return symbols

    return WATCHLIST


def main() -> None:
    parser = argparse.ArgumentParser(description="Find ATM call options where volume > OI, delta > 0.47, tight bid/ask, and price $0.05-$2.50.")
    parser.add_argument("symbols", nargs="*", help="Optional symbols. If omitted, built-in WATCHLIST is scanned.")
    parser.add_argument("--watchlist-file", default=None, help="Optional comma/newline-separated ticker file.")
    parser.add_argument("--min-dte", type=int, default=7)
    parser.add_argument("--max-dte", type=int, default=225)
    parser.add_argument("--atm-pct", type=float, default=0.5)
    parser.add_argument("--min-delta", type=float, default=0.47)
    parser.add_argument("--min-price", type=float, default=0.20)
    parser.add_argument("--max-price", type=float, default=5.00)
    parser.add_argument("--max-spread-pct", type=float, default=0.10)
    parser.add_argument("--max-spread-abs", type=float, default=0.10)
    parser.add_argument("--min-volume", type=int, default=500)
    parser.add_argument("--min-open-interest", type=int, default=1000)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--out", default="atm_call_contracts.csv")

    args = parser.parse_args()

    symbols = parse_symbols(args)

    print(f"Scanning {len(symbols)} symbols")

    all_rows = []

    for symbol in symbols:
        rows = scan_symbol(symbol, args)
        all_rows.extend(rows)
        time.sleep(args.sleep)

    all_rows.sort(key=lambda r: (-r["score"], r["spread_pct"], r["atm_distance_pct"]))

    write_csv(all_rows, args.out)

    for row in all_rows[:25]:
        print(row)


if __name__ == "__main__":
    main()