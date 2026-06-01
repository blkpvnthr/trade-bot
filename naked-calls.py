#!/usr/bin/env python3
"""
av_alpaca_long_call_scanner.py

Finds the best long call contracts for small Alpaca accounts using only:

Data sources:
    - Alpha Vantage REALTIME_OPTIONS
    - Alpha Vantage REALTIME_PUT_CALL_RATIO
    - Alpha Vantage REALTIME_VOLUME_OPEN_INTEREST_RATIO

Execution:
    - Alpaca only

Important:
    This script scans BUY-to-open long calls.
    It does NOT sell naked calls because short naked calls have unlimited risk.

Required .env:
    ALPHA_VANTAGE_API_KEY=...
    APCA_API_KEY_ID=...
    APCA_API_SECRET_KEY=...
    ALPACA_PAPER=true
"""

import os
import time
import math
import datetime as dt
from typing import Dict, Any, List, Optional

import requests
import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


# ============================================================
# ENV / CLIENTS
# ============================================================

load_dotenv(".env")

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

if not ALPHA_VANTAGE_API_KEY:
    raise RuntimeError("Missing ALPHA_VANTAGE_API_KEY in .env")

if not APCA_API_KEY_ID or not APCA_API_SECRET_KEY:
    raise RuntimeError("Missing Alpaca keys in .env")

alpaca = TradingClient(
    APCA_API_KEY_ID,
    APCA_API_SECRET_KEY,
    paper=ALPACA_PAPER,
)


# ============================================================
# CONFIG
# ============================================================

WATCHLIST = [
    "SPY",
    "QQQ",
    "IWM",
    "AAPL",
    "NVDA",
    "AMD",
    "TSLA",
    "META",
    "MSFT",
    "PLTR",
    "SOFI",
    "RGTI",
    "QBTS",
    "IONQ",
]

ACCOUNTS = {
    "INDIVIDUAL_5632": 195.42,
    "BLKPVNTHR_TRUST_6418": 653.25,
    "INDIVIDUAL_9803": 114.25,
    "BLKPVNTHR_LLC_1767": 164.73,
    "ROTH_IRA_1514": 396.08,
}

MIN_DTE = 7
MAX_DTE = 45

MIN_DELTA = 0.30
MAX_DELTA = 0.70

MIN_VOLUME = 25
MIN_OPEN_INTEREST = 50

MAX_SPREAD_PCT = 0.20
MAX_IV = 2.50

REQUEST_SLEEP_SECONDS = 0.85

DRY_RUN = True


# ============================================================
# HELPERS
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def max_option_budget(account_equity: float) -> float:
    """
    Dynamic sizing for small accounts.
    """
    if account_equity < 250:
        return round(account_equity * 0.15, 2)

    if account_equity < 500:
        return round(account_equity * 0.20, 2)

    return round(account_equity * 0.25, 2)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


# ============================================================
# ALPHA VANTAGE CLIENT
# ============================================================

class AlphaVantageOptionsClient:
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(params)
        params["apikey"] = self.api_key

        response = requests.get(
            self.BASE_URL,
            params=params,
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()

        if "Note" in data:
            raise RuntimeError(f"Alpha Vantage rate limit message: {data['Note']}")

        if "Information" in data:
            raise RuntimeError(f"Alpha Vantage info message: {data['Information']}")

        if "Error Message" in data:
            raise RuntimeError(f"Alpha Vantage error: {data['Error Message']}")

        return data

    def realtime_options(self, symbol: str) -> pd.DataFrame:
        data = self._get({
            "function": "REALTIME_OPTIONS",
            "symbol": symbol,
            "require_greeks": "true",
        })

        rows = (
            data.get("data")
            or data.get("options")
            or data.get("option_chain")
            or []
        )

        if not rows:
            return pd.DataFrame()

        return normalize_columns(pd.DataFrame(rows))

    def realtime_put_call_ratio(self, symbol: str) -> Optional[float]:
        data = self._get({
            "function": "REALTIME_PUT_CALL_RATIO",
            "symbol": symbol,
        })

        rows = data.get("data") or data.get("put_call_ratio") or data

        if isinstance(rows, list) and rows:
            row = rows[0]
        elif isinstance(rows, dict):
            row = rows
        else:
            return None

        for key in [
            "put_call_ratio",
            "ratio",
            "pcr",
            "total_put_call_ratio",
        ]:
            if key in row:
                return safe_float(row[key], default=None)

        return None

    def realtime_volume_oi_ratio(self, symbol: str) -> Optional[float]:
        data = self._get({
            "function": "REALTIME_VOLUME_OPEN_INTEREST_RATIO",
            "symbol": symbol,
        })

        rows = data.get("data") or data.get("volume_open_interest_ratio") or data

        if isinstance(rows, list) and rows:
            row = rows[0]
        elif isinstance(rows, dict):
            row = rows
        else:
            return None

        for key in [
            "volume_open_interest_ratio",
            "volume_oi_ratio",
            "ratio",
        ]:
            if key in row:
                return safe_float(row[key], default=None)

        return None


av = AlphaVantageOptionsClient(ALPHA_VANTAGE_API_KEY)


# ============================================================
# SCORING
# ============================================================

def put_call_score(pcr: Optional[float]) -> float:
    if pcr is None:
        return 0

    if pcr <= 0.35:
        return 1.0

    if pcr <= 0.60:
        return 3.0

    if pcr <= 0.85:
        return 1.5

    if pcr <= 1.00:
        return 0.0

    if pcr <= 1.50:
        return -2.0

    return -0.5


def volume_oi_score(voi: Optional[float]) -> float:
    if voi is None:
        return 0

    if voi >= 3.0:
        return 3.0

    if voi >= 2.0:
        return 2.5

    if voi >= 1.0:
        return 2.0

    if voi >= 0.50:
        return 1.0

    return 0.0


def score_contract(
    *,
    delta: float,
    gamma: float,
    theta: float,
    iv: float,
    volume: int,
    open_interest: int,
    spread_pct: float,
    pcr: Optional[float],
    voi: Optional[float],
) -> float:
    delta_quality = 3.0 - abs(delta - 0.50) * 6.0
    gamma_quality = min(gamma * 20.0, 3.0)
    liquidity_score = min(volume / 500.0, 3.0) + min(open_interest / 1000.0, 3.0)

    theta_penalty = min(abs(theta) * 4.0, 4.0)
    spread_penalty = spread_pct * 8.0
    iv_penalty = max(iv - 1.00, 0.0) * 1.5

    score = (
        put_call_score(pcr)
        + volume_oi_score(voi)
        + delta_quality
        + gamma_quality
        + liquidity_score
        - theta_penalty
        - spread_penalty
        - iv_penalty
    )

    return round(score, 4)


# ============================================================
# SCANNER
# ============================================================

def extract_contract_symbol(row: pd.Series) -> Optional[str]:
    for key in ["contract", "symbol", "contractid", "contract_id"]:
        value = row.get(key)
        if value:
            return str(value)
    return None


def extract_option_type(row: pd.Series) -> str:
    for key in ["type", "contract_type", "option_type"]:
        value = row.get(key)
        if value:
            return str(value).lower()
    return ""


def extract_expiration(row: pd.Series) -> Optional[dt.date]:
    for key in ["expiration", "expiration_date"]:
        value = row.get(key)
        if value:
            try:
                return dt.date.fromisoformat(str(value)[:10])
            except Exception:
                return None
    return None


def scan_symbol(symbol: str) -> List[Dict[str, Any]]:
    print(f"Scanning {symbol}...")

    options_df = av.realtime_options(symbol)
    time.sleep(REQUEST_SLEEP_SECONDS)

    if options_df.empty:
        print(f"No realtime options returned for {symbol}")
        return []

    try:
        pcr = av.realtime_put_call_ratio(symbol)
    except Exception as e:
        print(f"Could not fetch PCR for {symbol}: {e}")
        pcr = None

    time.sleep(REQUEST_SLEEP_SECONDS)

    try:
        voi = av.realtime_volume_oi_ratio(symbol)
    except Exception as e:
        print(f"Could not fetch volume/OI ratio for {symbol}: {e}")
        voi = None

    today = dt.date.today()
    results = []

    for _, row in options_df.iterrows():
        option_type = extract_option_type(row)

        if option_type and "call" not in option_type:
            continue

        expiration = extract_expiration(row)
        if not expiration:
            continue

        dte = (expiration - today).days
        if dte < MIN_DTE or dte > MAX_DTE:
            continue

        contract = extract_contract_symbol(row)
        if not contract:
            continue

        strike = safe_float(row.get("strike") or row.get("strike_price"))
        bid = safe_float(row.get("bid"))
        ask = safe_float(row.get("ask"))
        last = safe_float(row.get("last") or row.get("last_price"))

        if bid > 0 and ask > 0:
            mid = round((bid + ask) / 2.0, 2)
            spread_pct = (ask - bid) / mid if mid > 0 else 999
        else:
            mid = last
            spread_pct = MAX_SPREAD_PCT

        if mid <= 0:
            continue

        if spread_pct > MAX_SPREAD_PCT:
            continue

        cost = round(mid * 100.0, 2)

        volume = safe_int(row.get("volume"))
        open_interest = safe_int(row.get("open_interest"))

        if volume < MIN_VOLUME:
            continue

        if open_interest < MIN_OPEN_INTEREST:
            continue

        delta = abs(safe_float(row.get("delta")))
        gamma = safe_float(row.get("gamma"))
        theta = safe_float(row.get("theta"))
        iv = safe_float(row.get("implied_volatility") or row.get("iv"))

        if delta < MIN_DELTA or delta > MAX_DELTA:
            continue

        if iv > MAX_IV:
            continue

        score = score_contract(
            delta=delta,
            gamma=gamma,
            theta=theta,
            iv=iv,
            volume=volume,
            open_interest=open_interest,
            spread_pct=spread_pct,
            pcr=pcr,
            voi=voi,
        )

        results.append({
            "underlying": symbol,
            "contract": contract,
            "expiration": expiration.isoformat(),
            "dte": dte,
            "strike": strike,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "cost": cost,
            "delta": round(delta, 4),
            "gamma": round(gamma, 4),
            "theta": round(theta, 4),
            "iv": round(iv, 4),
            "volume": volume,
            "open_interest": open_interest,
            "spread_pct": round(spread_pct, 4),
            "put_call_ratio": pcr,
            "volume_oi_ratio": voi,
            "score": score,
        })

    return results


def scan_market() -> pd.DataFrame:
    all_results = []

    for symbol in WATCHLIST:
        try:
            all_results.extend(scan_symbol(symbol))
        except Exception as e:
            print(f"ERROR scanning {symbol}: {e}")

        time.sleep(REQUEST_SLEEP_SECONDS)

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ============================================================
# ACCOUNT ASSIGNMENT
# ============================================================

def assign_best_contracts_to_accounts(ranked: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for account_name, equity in ACCOUNTS.items():
        max_budget = max_option_budget(equity)

        if ranked.empty:
            rows.append({
                "account": account_name,
                "equity": equity,
                "max_budget": max_budget,
                "action": "NO_TRADE",
                "reason": "No ranked contracts available",
            })
            continue

        affordable = ranked[ranked["cost"] <= max_budget].copy()

        if affordable.empty:
            rows.append({
                "account": account_name,
                "equity": equity,
                "max_budget": max_budget,
                "action": "NO_TRADE",
                "reason": "No contract fits account budget",
            })
            continue

        best = affordable.iloc[0].to_dict()
        best["account"] = account_name
        best["equity"] = equity
        best["max_budget"] = max_budget
        best["contracts_to_buy"] = 1
        best["action"] = "BUY_TO_OPEN_LONG_CALL"
        rows.append(best)

    return pd.DataFrame(rows)


# ============================================================
# ALPACA EXECUTION
# ============================================================

def execute_long_call(contract_symbol: str, qty: int = 1):
    if DRY_RUN:
        print(f"[DRY RUN] Would buy {qty} {contract_symbol}")
        return None

    order = MarketOrderRequest(
        symbol=contract_symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )

    submitted = alpaca.submit_order(order)
    print(f"Submitted order: {submitted}")
    return submitted


def execute_account_picks(account_picks: pd.DataFrame):
    for _, pick in account_picks.iterrows():
        if pick.get("action") != "BUY_TO_OPEN_LONG_CALL":
            continue

        contract = pick.get("contract")
        qty = int(pick.get("contracts_to_buy", 1))

        if not contract:
            continue

        execute_long_call(contract, qty)


# ============================================================
# MAIN
# ============================================================

def main():
    print("Starting Alpha Vantage + Alpaca long call scanner...")
    print(f"Alpaca paper mode: {ALPACA_PAPER}")
    print(f"Dry run: {DRY_RUN}")

    ranked = scan_market()

    if ranked.empty:
        print("No qualifying long call contracts found.")
        return

    ranked.to_csv("av_ranked_long_calls.csv", index=False)

    account_picks = assign_best_contracts_to_accounts(ranked)
    account_picks.to_csv("av_account_call_picks.csv", index=False)

    print("\nTop ranked contracts:")
    print(ranked.head(25).to_string(index=False))

    print("\nAccount picks:")
    print(account_picks.to_string(index=False))

    execute_account_picks(account_picks)


if __name__ == "__main__":
    main()