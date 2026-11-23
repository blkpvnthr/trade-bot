#!/usr/bin/env python3
from __future__ import annotations

import re
import json
import math
import os
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI

try:
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, OptionChainRequest
    from alpaca.data.timeframe import TimeFrame
except Exception:
    DataFeed = None
    StockHistoricalDataClient = None
    OptionHistoricalDataClient = None
    StockBarsRequest = None
    OptionChainRequest = None
    TimeFrame = None


# =============================================================================
# Paths / Environment
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


ET = ZoneInfo("America/New_York")
TODAY_TAG = datetime.now(ET).strftime("%Y%m%d")

OUT_DIR = PROJECT_ROOT / "system-data" / f"{TODAY_TAG}"

OUT_DIR.mkdir(parents=True, exist_ok=True)
DAILY_OUTPUT_DIR = PROJECT_ROOT / "daily_system_outputs" /  TODAY_TAG
EXECUTION_LOG_DIR = PROJECT_ROOT / "execution_logs" / TODAY_TAG

LLM_UNIVERSE_PATH = PROJECT_ROOT / f"{TODAY_TAG}_latest_llm_sentiment_universe.json"
LOCKED_UNIVERSE_PATH = PROJECT_ROOT / f"{TODAY_TAG}_locked_trading_universe.txt"

MARKET_CONTEXT_PATH = OUT_DIR / f"{TODAY_TAG}_market_context.json"

DAILY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXECUTION_LOG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Config
# =============================================================================

OPENAI_MODEL = os.getenv("OPENAI_MARKET_CONTEXT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_MARKET_CONTEXT_TIMEOUT_SECONDS", "30"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MARKET_CONTEXT_MAX_RETRIES", "1"))
# Initial output-token budget and the number of application-level classification
# attempts. A response truncated by the token cap (finish_reason='length') is the
# documented root cause of the "Unterminated string" json.loads crash, so a
# truncated or unparseable response is retried (truncation escalates the budget)
# before falling back deterministically.
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MARKET_CONTEXT_MAX_TOKENS", "2600"))
OPENAI_MAX_OUTPUT_TOKENS_CEIL = int(os.getenv("OPENAI_MARKET_CONTEXT_MAX_TOKENS_CEIL", "8000"))
OPENAI_CLASSIFY_MAX_ATTEMPTS = max(1, int(os.getenv("OPENAI_MARKET_CONTEXT_MAX_ATTEMPTS", "2")))
MARKET_CONTEXT_DISABLE_OPENAI = os.getenv("MARKET_CONTEXT_DISABLE_OPENAI", "0").lower() in {"1", "true", "yes"}
MARKET_CONTEXT_FAIL_OPENAI_CLOSED = os.getenv("MARKET_CONTEXT_FAIL_OPENAI_CLOSED", "0").lower() in {"1", "true", "yes"}
from market_news import search_news
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex").lower().strip()
ALPACA_OPTIONS_MAX_CHAIN_CONTRACTS = int(os.getenv("ALPACA_OPTIONS_MAX_CHAIN_CONTRACTS", "800"))
ALPACA_OPTIONS_MIN_DTE = int(os.getenv("ALPACA_OPTIONS_MIN_DTE", "7"))
ALPACA_OPTIONS_MAX_DTE = int(os.getenv("ALPACA_OPTIONS_MAX_DTE", "45"))
ALPACA_OPTIONS_MIN_VOLUME = int(os.getenv("ALPACA_OPTIONS_MIN_VOLUME", "0"))
ALPACA_OPTIONS_MIN_OPEN_INTEREST = int(os.getenv("ALPACA_OPTIONS_MIN_OPEN_INTEREST", "0"))
ALPACA_OPTIONS_MAX_SPREAD_PCT = float(os.getenv("ALPACA_OPTIONS_MAX_SPREAD_PCT", "0.35"))

MAX_SEARCH_RESULTS = int(os.getenv("MARKET_CONTEXT_MAX_SEARCH_RESULTS", "8"))
MAX_SYMBOL_OPTIONS = int(os.getenv("MARKET_CONTEXT_MAX_SYMBOL_OPTIONS", "40"))

INDEX_SYMBOLS = ["SPY", "QQQ", "DIA", "IWM"]

FUTURES_SYMBOLS = {
    "ES": "ES=F",
    "NQ": "NQ=F",
    "YM": "YM=F",
    "RTY": "RTY=F",
    "VIX": "^VIX",
    "CRUDE": "CL=F",
}

SECTOR_ETFS = {
    "Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
}

KEY_RISK_ETFS = {
    "SPY": "broad_market_direction",
    "QQQ": "growth_tech_direction",
    "VXX": "volatility_risk",
    "TLT": "bond_rate_risk",
    "UUP": "dollar_liquidity_risk",
}

EVENT_KEYS = [
    "fed_events_today",
    "jobs_report_today",
    "inflation_report_today",
    "pce_report_today",
    "opec_events_today",
    "presidential_trade_speech_today",
]


# =============================================================================
# Helpers
# =============================================================================

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def today_et() -> str:
    return datetime.now(ET).date().isoformat()


def normalize_symbol(value: Any) -> str:
    return str(value or "").upper().strip().replace(".", "-")


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return int(v)
    except Exception:
        return default


def safe_json(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(safe_json(payload), indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {path}")


def load_universe() -> list[str]:
    if LLM_UNIVERSE_PATH.exists():
        try:
            payload = json.loads(LLM_UNIVERSE_PATH.read_text(encoding="utf-8"))
            return [normalize_symbol(x) for x in payload.get("symbols", []) if normalize_symbol(x)]
        except Exception:
            pass

    if LOCKED_UNIVERSE_PATH.exists():
        raw = LOCKED_UNIVERSE_PATH.read_text(encoding="utf-8")
        tokens = re.split(r"[\s,]+", raw.replace("\"", "").replace("'", ""))
        return [normalize_symbol(x) for x in tokens if normalize_symbol(x)]

    return []


# =============================================================================
# Alpaca / Market Data
# =============================================================================

def get_alpaca_client() -> StockHistoricalDataClient | None:
    if StockHistoricalDataClient is None:
        return None

    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")

    if not key or not secret:
        return None

    return StockHistoricalDataClient(key, secret)


def get_alpaca_option_client() -> OptionHistoricalDataClient | None:
    if OptionHistoricalDataClient is None:
        return None

    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")

    if not key or not secret:
        return None

    try:
        return OptionHistoricalDataClient(key, secret)
    except Exception as exc:
        print(f"[options] failed to initialize Alpaca option client: {exc}")
        return None


OCC_OPTION_SYMBOL_RE = re.compile(r"^([A-Z][A-Z0-9-]*?)(\d{6})([CP])(\d{8})$")


def parse_occ_option_symbol(symbol: Any) -> dict[str, Any] | None:
    raw = normalize_symbol(symbol)
    match = OCC_OPTION_SYMBOL_RE.match(raw)
    if not match:
        return None

    underlying, yymmdd, cp, strike_raw = match.groups()

    try:
        expiration = datetime.strptime(yymmdd, "%y%m%d").date()
        strike = int(strike_raw) / 1000.0
    except Exception:
        return None

    return {
        "symbol": raw,
        "underlying": underlying,
        "expiration": expiration,
        "type": "call" if cp == "C" else "put",
        "strike": strike,
    }


def obj_get(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def option_snapshot_metrics(snapshot: Any) -> dict[str, Any]:
    quote = obj_get(snapshot, "latest_quote", "latestQuote", default=None)
    greeks = obj_get(snapshot, "greeks", default=None)

    bid = to_float(obj_get(quote, "bid_price", "bp", "bid", default=0.0))
    ask = to_float(obj_get(quote, "ask_price", "ap", "ask", default=0.0))
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
    spread_pct = (ask - bid) / mid if mid > 0 else None

    return {
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread_pct": spread_pct,
        "volume": to_int(obj_get(snapshot, "volume", default=0)),
        "open_interest": to_int(obj_get(snapshot, "open_interest", "openInterest", default=0)),
        "implied_volatility": to_float(obj_get(snapshot, "implied_volatility", "iv", default=0.0)),
        "delta": to_float(obj_get(greeks, "delta", default=0.0)),
        "gamma": to_float(obj_get(greeks, "gamma", default=0.0)),
        "theta": to_float(obj_get(greeks, "theta", default=0.0)),
        "vega": to_float(obj_get(greeks, "vega", default=0.0)),
    }


def get_alpaca_option_chain(symbol: str) -> dict[str, Any]:
    client = get_alpaca_option_client()

    if client is None or OptionChainRequest is None:
        return {}

    try:
        request = OptionChainRequest(underlying_symbol=normalize_symbol(symbol))
        chain = client.get_option_chain(request)

        if isinstance(chain, dict):
            return chain
        if hasattr(chain, "data") and isinstance(chain.data, dict):
            return chain.data

        try:
            return dict(chain) if chain is not None else {}
        except Exception:
            return {}

    except Exception as exc:
        print(f"[options] Alpaca option chain failed for {symbol}: {exc}")
        return {}



def get_feed() -> Any:
    if DataFeed is None:
        return None
    return DataFeed.SIP if ALPACA_DATA_FEED == "sip" else DataFeed.IEX


def previous_session_levels() -> dict[str, Any]:
    client = get_alpaca_client()

    if client is None or StockBarsRequest is None:
        return previous_session_levels_yfinance()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=10)

    try:
        kwargs = {
            "symbol_or_symbols": INDEX_SYMBOLS,
            "timeframe": TimeFrame.Day,
            "start": start,
            "end": end,
        }

        feed = get_feed()
        if feed is not None:
            kwargs["feed"] = feed

        bars = client.get_stock_bars(StockBarsRequest(**kwargs)).df

        if bars is None or bars.empty:
            return previous_session_levels_yfinance()

        df = bars.reset_index()
        out: dict[str, Any] = {}

        for symbol in INDEX_SYMBOLS:
            sdf = df[df["symbol"] == symbol].sort_values("timestamp")
            if sdf.empty:
                continue

            row = sdf.iloc[-1]
            open_price = to_float(row.get("open"))
            close_price = to_float(row.get("close"))

            out[symbol] = {
                "open": open_price,
                "close": close_price,
                "high": to_float(row.get("high")),
                "low": to_float(row.get("low")),
                "volume": to_int(row.get("volume")),
                "change_pct": close_price / open_price - 1.0 if open_price > 0 else 0.0,
                "source": "alpaca",
            }

        return out

    except Exception as exc:
        print(f"[market-context] Alpaca previous session failed: {exc}")
        return previous_session_levels_yfinance()


def previous_session_levels_yfinance() -> dict[str, Any]:
    out: dict[str, Any] = {}

    for symbol in INDEX_SYMBOLS:
        try:
            hist = yf.Ticker(symbol).history(period="10d", interval="1d")
            if hist.empty:
                continue

            row = hist.iloc[-1]
            open_price = to_float(row.get("Open"))
            close_price = to_float(row.get("Close"))

            out[symbol] = {
                "open": open_price,
                "close": close_price,
                "high": to_float(row.get("High")),
                "low": to_float(row.get("Low")),
                "volume": to_int(row.get("Volume")),
                "change_pct": close_price / open_price - 1.0 if open_price > 0 else 0.0,
                "source": "yfinance",
            }

        except Exception as exc:
            out[symbol] = {"error": str(exc), "source": "yfinance"}

    return out


def overnight_futures() -> dict[str, Any]:
    out: dict[str, Any] = {}

    current_date = datetime.now(ET).date()
    start_et = datetime.combine(current_date, time(0, 0), tzinfo=ET)
    end_et = datetime.combine(current_date, time(9, 30), tzinfo=ET)

    for label, yf_symbol in FUTURES_SYMBOLS.items():
        try:
            hist = yf.Ticker(yf_symbol).history(period="2d", interval="5m", prepost=True)

            if hist.empty:
                out[label] = {"symbol": yf_symbol, "error": "no_data"}
                continue

            if hist.index.tz is None:
                hist.index = hist.index.tz_localize("UTC")

            hist.index = hist.index.tz_convert(ET)
            window = hist[(hist.index >= start_et) & (hist.index <= end_et)]

            if window.empty:
                out[label] = {"symbol": yf_symbol, "error": "no_overnight_window"}
                continue

            open_price = to_float(window.iloc[0].get("Open"))
            last_price = to_float(window.iloc[-1].get("Close"))

            out[label] = {
                "symbol": yf_symbol,
                "midnight_price": open_price,
                "latest_price": last_price,
                "high": to_float(window["High"].max()),
                "low": to_float(window["Low"].min()),
                "change_pct": last_price / open_price - 1.0 if open_price > 0 else 0.0,
                "source": "yfinance_5m_midnight_to_930_et",
            }

        except Exception as exc:
            out[label] = {"symbol": yf_symbol, "error": str(exc)}

    return out


def get_etf_snapshot(symbols: list[str], period: str = "15d") -> dict[str, Any]:
    out: dict[str, Any] = {}

    for symbol in symbols:
        try:
            hist = yf.Ticker(symbol).history(period=period, interval="1d")

            if hist.empty or len(hist) < 2:
                out[symbol] = {"error": "insufficient_data"}
                continue

            latest = hist.iloc[-1]
            previous = hist.iloc[-2]

            open_price = to_float(latest.get("Open"))
            close = to_float(latest.get("Close"))
            previous_close = to_float(previous.get("Close"))

            out[symbol] = {
                "open": open_price,
                "close": close,
                "previous_close": previous_close,
                "day_change_pct": close / open_price - 1.0 if open_price > 0 else 0.0,
                "prev_close_change_pct": close / previous_close - 1.0 if previous_close > 0 else 0.0,
                "volume": to_int(latest.get("Volume")),
                "source": "yfinance_daily",
            }

        except Exception as exc:
            out[symbol] = {"error": str(exc)}

    return out


def sector_strength_context() -> dict[str, Any]:
    snapshot = get_etf_snapshot(list(SECTOR_ETFS.values()), period="15d")
    rows: list[dict[str, Any]] = []

    for sector, etf in SECTOR_ETFS.items():
        data = snapshot.get(etf, {})
        rows.append(
            {
                "sector": sector,
                "etf": etf,
                "change_pct": to_float(data.get("prev_close_change_pct")),
                "close": data.get("close"),
                "volume": data.get("volume"),
                "source": data.get("source", "unknown"),
            }
        )

    ranked = sorted(rows, key=lambda x: to_float(x.get("change_pct")), reverse=True)
    leading = ranked[:3]
    lagging = ranked[-3:]
    avg = sum(to_float(x.get("change_pct")) for x in ranked) / max(len(ranked), 1)

    leading_etfs = {x["etf"] for x in leading}

    if leading_etfs & {"XLK", "XLC", "XLY"}:
        rotation_bias = "risk_on_growth"
    elif leading_etfs & {"XLU", "XLP", "XLV"}:
        rotation_bias = "defensive"
    elif leading_etfs & {"XLE", "XLF", "XLI", "XLB"}:
        rotation_bias = "cyclical_value"
    else:
        rotation_bias = "mixed"

    return {
        "ranked_sectors": ranked,
        "leading_sectors": leading,
        "lagging_sectors": lagging,
        "average_sector_change_pct": avg,
        "sector_rotation_bias": rotation_bias,
    }


def key_etf_risk_context() -> dict[str, Any]:
    snapshot = get_etf_snapshot(list(KEY_RISK_ETFS.keys()), period="15d")

    spy = to_float(snapshot.get("SPY", {}).get("prev_close_change_pct"))
    qqq = to_float(snapshot.get("QQQ", {}).get("prev_close_change_pct"))
    vxx = to_float(snapshot.get("VXX", {}).get("prev_close_change_pct"))
    tlt = to_float(snapshot.get("TLT", {}).get("prev_close_change_pct"))
    uup = to_float(snapshot.get("UUP", {}).get("prev_close_change_pct"))

    risk_flags: list[str] = []

    if spy < -0.005:
        risk_flags.append("SPY weak")
    if qqq < -0.005:
        risk_flags.append("QQQ weak")
    if vxx > 0.02:
        risk_flags.append("VXX rising")
    if tlt < -0.007:
        risk_flags.append("TLT falling/rates pressure")
    if uup > 0.004:
        risk_flags.append("UUP rising/dollar pressure")

    if spy > 0 and qqq > 0 and vxx <= 0:
        bias = "risk_on"
    elif spy < 0 and qqq < 0 and vxx > 0:
        bias = "risk_off"
    elif len(risk_flags) >= 2:
        bias = "cautious"
    else:
        bias = "mixed"

    return {
        "etfs": snapshot,
        "risk_flags": risk_flags,
        "risk_signal_bias": bias,
        "interpretation": KEY_RISK_ETFS,
    }


# =============================================================================
# SerpApi Context
# =============================================================================

def rss_search(query: str, tbm: str | None = None) -> list[dict[str, Any]]:
    return search_news(query, limit=MAX_SEARCH_RESULTS)


def collect_serpapi_context() -> dict[str, Any]:
    date = today_et()

    queries = {
        "previous_day_market_news": f"stock market news yesterday S&P 500 Nasdaq Dow {date}",
        "fed_events_today": f"Federal Reserve FOMC Powell Fed speaker today {date}",
        "jobs_report_today": f"jobs report unemployment nonfarm payrolls jobless claims today {date}",
        "inflation_report_today": f"CPI PPI inflation report today {date}",
        "pce_report_today": f"PCE price index report today {date}",
        "opec_events_today": f"OPEC meeting oil production decision today {date}",
        "presidential_trade_speech_today": f"President trade tariff speech today markets {date}",
        "earnings_today": f"earnings calendar today major companies {date}",
        "economic_calendar_today": f"economic calendar today CPI PCE Fed jobs report {date}",
    }

    results = {}

    for key, query in queries.items():
        results[key] = rss_search(query, tbm="nws" if key == "previous_day_market_news" else None)

    return {
        "date": date,
        "queries": queries,
        "results": results,
        "source": "google_news_rss",
    }


# =============================================================================
# Options Context
# =============================================================================

def get_options_context_for_symbol(symbol: str) -> dict[str, Any]:
    """
    Build options sentiment/flow context from Alpaca option-chain snapshots.

    Output shape intentionally mirrors the old Alpaca-based context:
        put_volume
        call_volume
        put_call_ratio
        options_bias
        source

    Additional Alpaca fields are included for diagnostics:
        call_open_interest / put_open_interest
        call_premium_flow / put_premium_flow
        call_put_premium_ratio
        avg_call_iv / avg_put_iv
        top_call_contracts / top_put_contracts
    """
    symbol = normalize_symbol(symbol)
    chain = get_alpaca_option_chain(symbol)

    if not chain:
        return {
            "put_volume": None,
            "call_volume": None,
            "put_call_ratio": None,
            "options_bias": "unknown",
            "source": "alpaca_options_no_data",
        }

    today = datetime.now(ET).date()

    call_volume = 0
    put_volume = 0
    call_oi = 0
    put_oi = 0
    call_premium_flow = 0.0
    put_premium_flow = 0.0
    call_iv_values: list[float] = []
    put_iv_values: list[float] = []
    call_rows: list[dict[str, Any]] = []
    put_rows: list[dict[str, Any]] = []

    scanned = 0

    for contract_symbol, snapshot in chain.items():
        if scanned >= ALPACA_OPTIONS_MAX_CHAIN_CONTRACTS:
            break
        scanned += 1

        parsed = parse_occ_option_symbol(contract_symbol)
        if not parsed:
            continue

        dte = (parsed["expiration"] - today).days
        if dte < ALPACA_OPTIONS_MIN_DTE or dte > ALPACA_OPTIONS_MAX_DTE:
            continue

        metrics = option_snapshot_metrics(snapshot)
        volume = to_int(metrics.get("volume"))
        oi = to_int(metrics.get("open_interest"))
        mid = to_float(metrics.get("mid"))
        spread_pct = metrics.get("spread_pct")

        if volume < ALPACA_OPTIONS_MIN_VOLUME:
            continue
        if oi < ALPACA_OPTIONS_MIN_OPEN_INTEREST:
            continue
        if spread_pct is not None and spread_pct > ALPACA_OPTIONS_MAX_SPREAD_PCT:
            continue

        premium_flow = volume * mid * 100.0 if volume > 0 and mid > 0 else 0.0

        row = {
            "option_symbol": normalize_symbol(contract_symbol),
            "expiration": parsed["expiration"].isoformat(),
            "dte": dte,
            "strike": parsed["strike"],
            "volume": volume,
            "open_interest": oi,
            "bid": metrics.get("bid"),
            "ask": metrics.get("ask"),
            "mid": mid,
            "spread_pct": spread_pct,
            "premium_flow": round(premium_flow, 2),
            "implied_volatility": metrics.get("implied_volatility"),
            "delta": metrics.get("delta"),
            "gamma": metrics.get("gamma"),
            "theta": metrics.get("theta"),
        }

        iv = to_float(metrics.get("implied_volatility"), 0.0)

        if parsed["type"] == "call":
            call_volume += volume
            call_oi += oi
            call_premium_flow += premium_flow
            if iv > 0:
                call_iv_values.append(iv)
            call_rows.append(row)
        elif parsed["type"] == "put":
            put_volume += volume
            put_oi += oi
            put_premium_flow += premium_flow
            if iv > 0:
                put_iv_values.append(iv)
            put_rows.append(row)

    put_call_ratio = put_volume / call_volume if call_volume > 0 else None
    call_put_premium_ratio = call_premium_flow / put_premium_flow if put_premium_flow > 0 else None

    if put_call_ratio is None:
        bias = "unknown"
    elif put_call_ratio >= 1.25:
        bias = "bearish"
    elif put_call_ratio <= 0.75:
        bias = "bullish"
    else:
        bias = "neutral"

    # If premium flow is heavily call-dominant, upgrade neutral/unknown to bullish.
    if call_put_premium_ratio is not None and call_put_premium_ratio >= 1.75 and bias in {"neutral", "unknown"}:
        bias = "bullish"
    elif call_put_premium_ratio is not None and call_put_premium_ratio <= 0.60 and bias in {"neutral", "unknown"}:
        bias = "bearish"

    top_calls = sorted(call_rows, key=lambda x: to_float(x.get("premium_flow")), reverse=True)[:5]
    top_puts = sorted(put_rows, key=lambda x: to_float(x.get("premium_flow")), reverse=True)[:5]

    return {
        "expiration_window_dte": {
            "min": ALPACA_OPTIONS_MIN_DTE,
            "max": ALPACA_OPTIONS_MAX_DTE,
        },
        "put_volume": put_volume,
        "call_volume": call_volume,
        "put_open_interest": put_oi,
        "call_open_interest": call_oi,
        "put_call_ratio": put_call_ratio,
        "call_premium_flow": round(call_premium_flow, 2),
        "put_premium_flow": round(put_premium_flow, 2),
        "call_put_premium_ratio": call_put_premium_ratio,
        "avg_call_iv": sum(call_iv_values) / len(call_iv_values) if call_iv_values else None,
        "avg_put_iv": sum(put_iv_values) / len(put_iv_values) if put_iv_values else None,
        "options_bias": bias,
        "contracts_scanned": scanned,
        "contracts_used": len(call_rows) + len(put_rows),
        "top_call_contracts": top_calls,
        "top_put_contracts": top_puts,
        "source": "alpaca_options_chain",
    }


def options_context(symbols: list[str]) -> dict[str, Any]:
    targets = list(dict.fromkeys(INDEX_SYMBOLS + symbols))[:MAX_SYMBOL_OPTIONS]
    symbol_data = {s: get_options_context_for_symbol(s) for s in targets}

    ratios = [
        to_float(v.get("put_call_ratio"))
        for v in symbol_data.values()
        if isinstance(v.get("put_call_ratio"), (int, float))
    ]

    premium_ratios = [
        to_float(v.get("call_put_premium_ratio"))
        for v in symbol_data.values()
        if isinstance(v.get("call_put_premium_ratio"), (int, float))
    ]

    market_ratio = sum(ratios) / len(ratios) if ratios else None
    market_call_put_premium_ratio = sum(premium_ratios) / len(premium_ratios) if premium_ratios else None

    total_call_volume = sum(to_int(v.get("call_volume")) for v in symbol_data.values() if isinstance(v, dict))
    total_put_volume = sum(to_int(v.get("put_volume")) for v in symbol_data.values() if isinstance(v, dict))
    total_call_premium_flow = sum(to_float(v.get("call_premium_flow")) for v in symbol_data.values() if isinstance(v, dict))
    total_put_premium_flow = sum(to_float(v.get("put_premium_flow")) for v in symbol_data.values() if isinstance(v, dict))

    if market_ratio is None:
        bias = "unknown"
    elif market_ratio >= 1.25:
        bias = "bearish"
    elif market_ratio <= 0.75:
        bias = "bullish"
    else:
        bias = "neutral"

    aggregate_call_put_premium_ratio = (
        total_call_premium_flow / total_put_premium_flow
        if total_put_premium_flow > 0
        else None
    )

    if aggregate_call_put_premium_ratio is not None and aggregate_call_put_premium_ratio >= 1.75 and bias in {"neutral", "unknown"}:
        bias = "bullish"
    elif aggregate_call_put_premium_ratio is not None and aggregate_call_put_premium_ratio <= 0.60 and bias in {"neutral", "unknown"}:
        bias = "bearish"

    return {
        "put_call_ratio_market": market_ratio,
        "call_put_premium_ratio_market": market_call_put_premium_ratio,
        "aggregate_call_put_premium_ratio": aggregate_call_put_premium_ratio,
        "options_bias_market": bias,
        "total_call_volume": total_call_volume,
        "total_put_volume": total_put_volume,
        "total_call_premium_flow": round(total_call_premium_flow, 2),
        "total_put_premium_flow": round(total_put_premium_flow, 2),
        "symbols": symbol_data,
        "source": "alpaca_options_chain" if get_alpaca_option_client() else "alpaca_options_not_configured",
    }


# =============================================================================
# Classification
# =============================================================================

def empty_event_evidence() -> dict[str, list[dict[str, Any]]]:
    return {key: [] for key in EVENT_KEYS}


def validate_event_flags(classification: dict[str, Any]) -> dict[str, Any]:
    flags = classification.get("event_flags", {})
    evidence = classification.get("event_evidence", {})

    if not isinstance(flags, dict):
        flags = {}

    if not isinstance(evidence, dict):
        evidence = empty_event_evidence()

    clean_evidence = empty_event_evidence()

    for key in EVENT_KEYS:
        items = evidence.get(key, [])
        if not isinstance(items, list):
            items = []
        clean_evidence[key] = [x for x in items if isinstance(x, dict)]

    clean_flags = {
        "fed_events_today": bool(flags.get("fed_events_today")) and len(clean_evidence["fed_events_today"]) > 0,
        "opec_events_today": bool(flags.get("opec_events_today")) and len(clean_evidence["opec_events_today"]) > 0,
        "jobs_report_today": bool(flags.get("jobs_report_today")) and len(clean_evidence["jobs_report_today"]) > 0,
        "inflation_report_today": bool(flags.get("inflation_report_today")) and len(clean_evidence["inflation_report_today"]) > 0,
        "pce_report_today": bool(flags.get("pce_report_today")) and len(clean_evidence["pce_report_today"]) > 0,
        "presidential_trade_speech_today": bool(flags.get("presidential_trade_speech_today")) and len(clean_evidence["presidential_trade_speech_today"]) > 0,
        "major_earnings_today": bool(flags.get("major_earnings_today")),
    }

    classification["event_flags"] = clean_flags
    classification["event_evidence"] = clean_evidence
    return classification


def classify_with_openai(raw_context: dict[str, Any], *, client: Any = None,
                         max_attempts: int | None = None) -> dict[str, Any]:
    """Classify the market context via OpenAI, resilient to truncated/malformed
    responses.

    ``client`` is injectable for testing (defaults to a real OpenAI client).
    Truncated (finish_reason='length') or unparseable responses are retried up to
    ``max_attempts`` (truncation escalates the output-token budget); on exhaustion
    the result is the deterministic :func:`fallback_classification` -- so core
    trading execution never depends on the LLM. Every return is tagged with
    ``classification_source`` ('openai' or 'fallback') for clear logging. With
    ``MARKET_CONTEXT_FAIL_OPENAI_CLOSED`` set, persistent failure raises instead.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key and client is None:
        return fallback_classification(raw_context, "OPENAI_API_KEY missing")

    if MARKET_CONTEXT_DISABLE_OPENAI:
        return fallback_classification(raw_context, "MARKET_CONTEXT_DISABLE_OPENAI enabled")

    if client is None:
        client = OpenAI(
            api_key=api_key,
            timeout=OPENAI_TIMEOUT_SECONDS,
            max_retries=OPENAI_MAX_RETRIES,
        )

    prompt = f"""
You are a strict market context classifier for an automated trading system.

Analyze the supplied market data, futures, sector strength, key ETF risk signals, RSS news results, calendar context, and options put/call data.

Return ONLY valid JSON.

Required schema:
{{
  "overall_market_bias": "bullish | bearish | mixed | risk_off",
  "trade_permission": "normal | cautious | halt_new_buys",
  "risk_level": "low | medium | high",
  "summary": "brief summary",
  "decision_reasons": ["reason 1", "reason 2"],
  "event_evidence": {{
    "fed_events_today": [],
    "jobs_report_today": [],
    "inflation_report_today": [],
    "pce_report_today": [],
    "opec_events_today": [],
    "presidential_trade_speech_today": []
  }},
  "event_flags": {{
    "fed_events_today": false,
    "opec_events_today": false,
    "jobs_report_today": false,
    "inflation_report_today": false,
    "pce_report_today": false,
    "presidential_trade_speech_today": false,
    "major_earnings_today": false
  }},
  "sector_context_summary": {{
    "leading_sectors": [],
    "lagging_sectors": [],
    "sector_rotation_bias": "risk_on_growth | defensive | cyclical_value | mixed"
  }},
  "key_etf_context_summary": {{
    "risk_signal_bias": "risk_on | risk_off | cautious | mixed",
    "risk_flags": []
  }},
  "news_context_summary": {{
    "previous_day_market_news": [],
    "scheduled_market_events_today": [],
    "earnings_today": []
  }}
}}

Strict event-evidence rules:
- Only set an event flag true if event_evidence for that exact flag contains at least one specific supporting result.
- Each evidence item must include title, source, date, link, and why_it_matters.
- A generic result like "economic calendar today" is not enough.
- If a result says CPI, PPI, PCE, jobs, Fed, OPEC, or an earnings event is scheduled for a future date, do NOT mark it true for today.
- For jobs_report_today, require explicit language like "jobs report today", "nonfarm payrolls today", "unemployment report today", "jobless claims today", or a dated official/economic-calendar result.
- For fed_events_today, require explicit mention of "Powell speaks today", "Fed speaker today", "FOMC today", or "Federal Reserve announcement today".
- For inflation_report_today, require explicit mention of CPI/PPI/inflation data being released today.
- For pce_report_today, require explicit mention of PCE price index data being released today.
- If evidence is ambiguous, set the flag false and explain uncertainty in decision_reasons.
- Missing data should increase caution, but must not create fake event flags.

Market interpretation rules:
- Sector strength: identify whether money is rotating into growth, defensive, cyclical/value, or mixed leadership.
- SPY and QQQ indicate broad market and growth/tech direction.
- VXX rising means volatility/risk stress.
- TLT falling can indicate rate pressure.
- UUP rising can indicate dollar/liquidity pressure.
- Be conservative around confirmed Fed, CPI, PPI, PCE, jobs reports, OPEC, or presidential trade/tariff events.
- If futures are sharply negative or VIX/VXX is rising hard, use cautious or halt_new_buys.
- If SPY/QQQ are weak, VXX is rising, and futures are negative, use halt_new_buys.
- If sector rotation is risk_on_growth, SPY/QQQ are constructive, and no major macro event is confirmed, normal is allowed.
- If risk is unclear, trade_permission should be cautious.

Raw context:
{json.dumps(safe_json(raw_context), separators=(",", ":"))}
""".strip()

    attempts = max_attempts if max_attempts is not None else OPENAI_CLASSIFY_MAX_ATTEMPTS
    attempts = max(1, int(attempts))
    token_budget = OPENAI_MAX_OUTPUT_TOKENS
    last_err = "no attempts made"

    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Return only valid JSON. No markdown. Do not infer events without evidence.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=token_budget,
            )

            choice = response.choices[0]
            content = getattr(choice.message, "content", None) or "{}"
            finish_reason = str(getattr(choice, "finish_reason", "") or "")

            # Truncation detection: a response cut off at the token cap is the
            # root cause of the "Unterminated string" json.loads crash. Escalate
            # the budget and retry rather than parse a guaranteed-invalid body.
            if finish_reason == "length":
                last_err = (f"response truncated (finish_reason=length) "
                            f"at max_tokens={token_budget}")
                if attempt < attempts:
                    token_budget = min(token_budget * 2, OPENAI_MAX_OUTPUT_TOKENS_CEIL)
                    print(f"[market-context] {last_err}; retrying "
                          f"(attempt {attempt + 1}/{attempts}, max_tokens={token_budget})")
                    continue
                print(f"[market-context] {last_err}; attempts exhausted")
                break

            parsed = json.loads(content)
            result = validate_event_flags(parsed)
            result["classification_source"] = "openai"
            print(f"[market-context] classified via OpenAI model={OPENAI_MODEL} "
                  f"(attempt {attempt}/{attempts})")
            return result

        except json.JSONDecodeError as exc:
            # Malformed (but not length-truncated) JSON: retry once more, then fall back.
            last_err = f"OpenAI returned unparseable JSON: {exc}"
            if attempt < attempts:
                print(f"[market-context] {last_err}; retrying "
                      f"(attempt {attempt + 1}/{attempts})")
                continue
            print(f"[market-context] {last_err}; attempts exhausted")
            break

        except Exception as exc:
            # Transport/auth/other errors: the SDK already retried transient
            # failures (max_retries); an app-level retry would not help, so fall
            # back immediately (or fail closed if configured).
            msg = f"OpenAI classification failed: {exc}"
            print(f"[market-context] {msg}; using fallback classification")
            if MARKET_CONTEXT_FAIL_OPENAI_CLOSED:
                raise RuntimeError(msg) from exc
            return fallback_classification(raw_context, msg)

    msg = f"OpenAI classification failed after {attempts} attempt(s): {last_err}"
    print(f"[market-context] {msg}; using fallback classification")
    if MARKET_CONTEXT_FAIL_OPENAI_CLOSED:
        raise RuntimeError(msg)
    return fallback_classification(raw_context, msg)


def fallback_classification(raw_context: dict[str, Any], reason: str) -> dict[str, Any]:
    futures = raw_context.get("overnight_futures", {})
    key_etfs = raw_context.get("key_etf_risk_signals", {})
    sector = raw_context.get("sector_strength", {})

    es = to_float(futures.get("ES", {}).get("change_pct"))
    nq = to_float(futures.get("NQ", {}).get("change_pct"))
    vix = to_float(futures.get("VIX", {}).get("change_pct"))

    risk_signal_bias = key_etfs.get("risk_signal_bias", "mixed")
    sector_rotation_bias = sector.get("sector_rotation_bias", "mixed")

    score = 0.0
    reasons = [reason]

    if es > 0.003 and nq > 0.003:
        score += 1.0
        reasons.append("ES and NQ futures are positive overnight.")
    elif es < -0.004 or nq < -0.004:
        score -= 1.5
        reasons.append("ES or NQ futures are negative overnight.")

    if vix > 0.03:
        score -= 1.0
        reasons.append("VIX is rising.")

    if risk_signal_bias == "risk_on":
        score += 0.75
        reasons.append("Key ETF risk signals are risk-on.")
    elif risk_signal_bias == "risk_off":
        score -= 1.0
        reasons.append("Key ETF risk signals are risk-off.")
    elif risk_signal_bias == "cautious":
        score -= 0.5
        reasons.append("Key ETF risk signals are cautious.")

    if sector_rotation_bias == "risk_on_growth":
        score += 0.5
        reasons.append("Growth sectors are leading.")
    elif sector_rotation_bias == "defensive":
        score -= 0.5
        reasons.append("Defensive sectors are leading.")

    if score >= 1.0:
        bias = "bullish"
        permission = "normal"
        risk = "low"
    elif score <= -1.5:
        bias = "risk_off"
        permission = "halt_new_buys"
        risk = "high"
    else:
        bias = "mixed"
        permission = "cautious"
        risk = "medium"

    return {
        "overall_market_bias": bias,
        "trade_permission": permission,
        "risk_level": risk,
        "classification_source": "fallback",
        "summary": "Fallback market classification using futures, sector strength, and key ETF risk signals.",
        "decision_reasons": reasons,
        "event_evidence": empty_event_evidence(),
        "event_flags": {
            "fed_events_today": False,
            "opec_events_today": False,
            "jobs_report_today": False,
            "inflation_report_today": False,
            "pce_report_today": False,
            "presidential_trade_speech_today": False,
            "major_earnings_today": False,
        },
        "sector_context_summary": {
            "leading_sectors": sector.get("leading_sectors", []),
            "lagging_sectors": sector.get("lagging_sectors", []),
            "sector_rotation_bias": sector_rotation_bias,
        },
        "key_etf_context_summary": {
            "risk_signal_bias": risk_signal_bias,
            "risk_flags": key_etfs.get("risk_flags", []),
        },
        "news_context_summary": {
            "previous_day_market_news": [],
            "scheduled_market_events_today": [],
            "earnings_today": [],
        },
    }


# =============================================================================
# Main
# =============================================================================

def build_market_context() -> dict[str, Any]:
    symbols = load_universe()

    raw_context = {
        "created_at_utc": now_utc(),
        "market_context_date_et": today_et(),
        "universe_symbol_count": len(symbols),
        "previous_session": previous_session_levels(),
        "overnight_futures": overnight_futures(),
        "sector_strength": sector_strength_context(),
        "key_etf_risk_signals": key_etf_risk_context(),
        "serpapi_context": collect_serpapi_context(),
        "options_context": options_context(symbols),
        "calendar_context": {
            "date": today_et(),
            "weekday": datetime.now(ET).strftime("%A"),
            "is_friday": datetime.now(ET).weekday() == 4,
        },
        "data_sources": {
            "alpaca": bool(get_alpaca_client()),
            "yfinance": True,
            "rss_news": True,
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "alpaca_options": bool(get_alpaca_option_client()),
        },
    }

    try:
        classification = validate_event_flags(classify_with_openai(raw_context))
    except Exception as exc:
        print(f"[market-context] classifier failed hard; using fallback: {exc}")
        classification = validate_event_flags(fallback_classification(raw_context, f"classifier hard failure: {exc}"))

    payload = {
        **raw_context,
        "overall_market_bias": classification.get("overall_market_bias", "mixed"),
        "trade_permission": classification.get("trade_permission", "cautious"),
        "risk_level": classification.get("risk_level", "medium"),
        "summary": classification.get("summary", ""),
        "decision_reasons": classification.get("decision_reasons", []),
        "event_evidence": classification.get("event_evidence", empty_event_evidence()),
        "event_flags": classification.get("event_flags", {}),
        "sector_context_summary": classification.get("sector_context_summary", {}),
        "key_etf_context_summary": classification.get("key_etf_context_summary", {}),
        "news_context_summary": classification.get("news_context_summary", {}),
        "classifier_model": OPENAI_MODEL,
    }

    write_json(MARKET_CONTEXT_PATH, payload)
    return payload


if __name__ == "__main__":
    context = build_market_context()
    print(
        json.dumps(
            {
                "overall_market_bias": context.get("overall_market_bias"),
                "trade_permission": context.get("trade_permission"),
                "risk_level": context.get("risk_level"),
                "summary": context.get("summary"),
                "event_flags": context.get("event_flags"),
                "sector_context_summary": context.get("sector_context_summary"),
                "key_etf_context_summary": context.get("key_etf_context_summary"),
                "decision_reasons": context.get("decision_reasons"),
            },
            indent=2,
        )
    )