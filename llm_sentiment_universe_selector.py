#!/usr/bin/env python3
"""
llm_sentiment_universe_selector.py

CSV-driven premarket universe selector.

This version does NOT generate tickers from Yahoo gainers.
It only researches/ranks symbols from:

    data/bullish_YYYY-MM-DD.csv

Expected CSV column:
    Symbol

Data sources:
    - yfinance 1-minute extended-hours bars for premarket volume
    - yfinance daily bars for recent max/average daily volume
    - yfinance Ticker.info / fast_info for company metadata
    - OpenAI LLM to rank only the supplied CSV tickers

Outputs:
    system-data/YYYYMMDD/YYYYMMDD_latest_llm_sentiment_universe.json
    system-data/YYYYMMDD/YYYYMMDD_latest_llm_candidate_discovery.json
    YYYYMMDD_locked_trading_universe.txt

Default:
    python llm_sentiment_universe_selector.py
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI

try:
    from market_context_builder import TODAY_TAG
except Exception:
    TODAY_TAG = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")


# =============================================================================
# Paths / environment
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

ET = ZoneInfo("America/New_York")

TODAY_TAG = datetime.now(ET).strftime("%Y%m%d")
YESTERDAY_TAG = (datetime.now(ET) - timedelta(days=1)).strftime("%Y%m%d")
DATA_TAG = os.getenv("RL_DATA_DATE", TODAY_TAG)
TODAY_DASH = (datetime.now(ET).strftime("%Y-%m-%d"))
CSV_UNIVERSE_PATH = Path(
    os.getenv(
        "BULLISH_UNIVERSE_CSV",
        str(PROJECT_ROOT / "data" / f"bullish_2026-05-15.csv"),
    )
)

OUT_DIR = PROJECT_ROOT / "system-data" / f"{TODAY_TAG}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = OUT_DIR / f"{TODAY_TAG}_latest_llm_sentiment_universe.json"
DISCOVERY_PATH = OUT_DIR / f"{TODAY_TAG}_latest_llm_candidate_discovery.json"
LOCKED_UNIVERSE_PATH = PROJECT_ROOT / f"{TODAY_TAG}_locked_trading_universe.txt"


# =============================================================================
# Config
# =============================================================================

MODEL = os.getenv("OPENAI_SENTIMENT_UNIVERSE_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))

TARGET_COUNT = int(os.getenv("TARGET_UNIVERSE_COUNT", "30"))
LLM_CANDIDATE_LIMIT = int(os.getenv("LLM_CANDIDATE_LIMIT", "80"))

OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "3500"))

MIN_PRICE = float(os.getenv("YF_MIN_PRICE", "2"))
MAX_PRICE = float(os.getenv("YF_MAX_PRICE", "500"))
MIN_MARKET_CAP = int(os.getenv("YF_MIN_MARKET_CAP", "0"))

MIN_PREMARKET_VOLUME = int(os.getenv("MIN_PREMARKET_VOLUME", "1000000"))
PREMARKET_TO_MAX_DAILY_VOL_RATIO = float(os.getenv("PREMARKET_TO_MAX_DAILY_VOL_RATIO", "0.75"))
PREMARKET_TO_AVG_DAILY_VOL_RATIO = float(os.getenv("PREMARKET_TO_AVG_DAILY_VOL_RATIO", "1.25"))
DAILY_VOLUME_LOOKBACK_DAYS = int(os.getenv("DAILY_VOLUME_LOOKBACK_DAYS", "90"))

PREMARKET_START_ET = os.getenv("PREMARKET_START_ET", "04:00")
PREMARKET_END_ET = os.getenv("PREMARKET_END_ET", "09:30")

YF_BATCH_SIZE = int(os.getenv("YF_BATCH_SIZE", "25"))
YF_INTRADAY_PERIOD = os.getenv("YF_INTRADAY_PERIOD", "5d")
YF_DAILY_PERIOD = os.getenv("YF_DAILY_PERIOD", "6mo")

CATALYST_TYPES = {
    "technical_gainer",
    "financial_strength",
    "analyst_upside",
    "sector_catalyst",
    "short_squeeze",
    "liquidity_momentum",
    "earnings_growth",
    "revenue_growth",
    "premarket_news_move",
    "premarket_volume_spike",
    "macro_catalyst",
}

SYSTEM_PROMPT = """
You are an institutional premarket stock selection engine.

You are given a fixed ticker universe from a local CSV file.
Your job is to research and rank ONLY those supplied tickers.

Primary priority:
Prioritize stocks with unusual premarket volume.

A stock has high premarket interest when:
- yfinance 1-minute extended-hours premarket volume is high
- premarket volume is close to or above the highest daily volume in the recent lookback window
- premarket volume is unusually high versus average daily volume
- premarket bar count confirms real extended-hours activity

Return ONLY valid JSON.

Schema:
{
  "market_tone": "bullish | bearish | mixed | risk_off | risk_on",
  "summary": "One concise sentence describing the overall opportunity set.",
  "selections": [
    {
      "rank": 1,
      "symbol": "NVDA",
      "catalyst_type": "technical_gainer | financial_strength | analyst_upside | sector_catalyst | short_squeeze | liquidity_momentum | earnings_growth | revenue_growth | premarket_news_move | premarket_volume_spike | macro_catalyst",
      "reason": "Compelling 1-2 sentence explanation using concrete supplied data.",
      "confidence": 0.0
    }
  ]
}

Rules:
- Rank from 1 upward.
- Use "symbol", not "ticker".
- Symbols must be uppercase.
- Do not add, invent, or substitute tickers.
- No duplicate symbols.
- Every reason must include at least TWO concrete supplied details when available.
- Prefer symbols with confirmed unusual yfinance premarket volume.
- Do not invent facts not present in the supplied data.
- Confidence must be between 0 and 1.
- No markdown.
- No extra commentary.
""".strip()


# =============================================================================
# Helpers
# =============================================================================

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_symbol(value: Any) -> str:
    return str(value or "").upper().strip().replace(".", "-")


def is_valid_symbol(symbol: Any) -> bool:
    symbol = normalize_symbol(symbol)
    return bool(re.fullmatch(r"[A-Z][A-Z0-9-]{0,9}", symbol))


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return int(value)
    except Exception:
        return default


def safe_json_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: safe_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json_value(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(safe_json_value(payload), indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {path}")


def fmt_pct(value: Any) -> str:
    return f"{to_float(value):.2f}%"


def fmt_money(value: Any) -> str:
    v = to_float(value)

    if v >= 1_000_000_000_000:
        return f"${v / 1_000_000_000_000:.2f}T"
    if v >= 1_000_000_000:
        return f"${v / 1_000_000_000:.2f}B"
    if v >= 1_000_000:
        return f"${v / 1_000_000:.2f}M"
    if v > 0:
        return f"${v:,.0f}"
    return "N/A"


def clean_reason(reason: Any) -> str:
    text = str(reason or "").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:700] or "Selected from CSV ticker universe using yfinance premarket volume diagnostics."


def normalize_catalyst_type(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return raw if raw in CATALYST_TYPES else "technical_gainer"


def parse_hhmm(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute), tzinfo=ET)


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i: i + size] for i in range(0, len(items), size)]


def flatten_yf_download(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """
    Normalize yfinance output to columns:
        timestamp, symbol, open, high, low, close, volume
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out_frames: list[pd.DataFrame] = []

    # Multi-symbol download usually returns MultiIndex columns:
    #   Price level: Open/High/Low/Close/Volume
    #   Ticker level: SYMBOL
    if isinstance(df.columns, pd.MultiIndex):
        level0 = [str(x).lower() for x in df.columns.get_level_values(0)]
        if any(x in {"open", "high", "low", "close", "volume"} for x in level0):
            price_level = 0
            ticker_level = 1
        else:
            price_level = 1
            ticker_level = 0

        tickers = sorted(set(str(x) for x in df.columns.get_level_values(ticker_level)))

        for symbol in tickers:
            if not is_valid_symbol(symbol):
                continue
            try:
                sdf = df.xs(symbol, axis=1, level=ticker_level).copy()
                sdf = sdf.rename(columns={c: str(c).lower().replace(" ", "_") for c in sdf.columns})
                sdf = sdf.reset_index()
                sdf["symbol"] = normalize_symbol(symbol)
                out_frames.append(sdf)
            except Exception:
                continue
    else:
        # Single-symbol download.
        symbol = normalize_symbol(symbols[0]) if symbols else ""
        sdf = df.copy()
        sdf = sdf.rename(columns={c: str(c).lower().replace(" ", "_") for c in sdf.columns})
        sdf = sdf.reset_index()
        sdf["symbol"] = symbol
        out_frames.append(sdf)

    if not out_frames:
        return pd.DataFrame()

    out = pd.concat(out_frames, ignore_index=True)

    # yfinance index column may be Datetime or Date.
    timestamp_col = None
    for col in ["datetime", "date", "timestamp"]:
        if col in out.columns:
            timestamp_col = col
            break
    if timestamp_col is None:
        timestamp_col = out.columns[0]

    out = out.rename(
        columns={
            timestamp_col: "timestamp",
            "adj_close": "adj_close",
        }
    )

    required_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }

    for col in required_map:
        if col not in out.columns:
            out[col] = 0.0

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["symbol"] = out["symbol"].map(normalize_symbol)

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out = out.dropna(subset=["timestamp"])
    out = out[out["symbol"].map(is_valid_symbol)].copy()

    return out[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]


# =============================================================================
# CSV universe
# =============================================================================

def load_csv_universe(path: Path = CSV_UNIVERSE_PATH) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Ticker CSV not found: {path}")

    df = pd.read_csv(path)

    if "Symbol" not in df.columns:
        raise ValueError(f"{path} must contain a 'Symbol' column")

    symbols = (
        df["Symbol"]
        .dropna()
        .astype(str)
        .map(normalize_symbol)
        .tolist()
    )

    symbols = sorted(set(s for s in symbols if is_valid_symbol(s)))

    if not symbols:
        raise RuntimeError(f"No valid symbols found in {path}")

    print(f"[csv-universe] loaded {len(symbols)} symbols from {path}")
    return symbols


# =============================================================================
# yfinance data enrichment
# =============================================================================

def fetch_yfinance_premarket_bars(symbols: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for group in chunked(symbols, YF_BATCH_SIZE):
        tickers = " ".join(group)
        try:
            df = yf.download(
                tickers=tickers,
                period=YF_INTRADAY_PERIOD,
                interval="1m",
                prepost=True,
                group_by="column",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            normalized = flatten_yf_download(df, group)
            if not normalized.empty:
                frames.append(normalized)
        except Exception as exc:
            print(f"[yfinance] 1m prepost fetch failed for {group}: {exc}")

    if not frames:
        return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def fetch_yfinance_daily_bars(symbols: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for group in chunked(symbols, YF_BATCH_SIZE):
        tickers = " ".join(group)
        try:
            df = yf.download(
                tickers=tickers,
                period=YF_DAILY_PERIOD,
                interval="1d",
                prepost=False,
                group_by="column",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            normalized = flatten_yf_download(df, group)
            if not normalized.empty:
                frames.append(normalized)
        except Exception as exc:
            print(f"[yfinance] daily fetch failed for {group}: {exc}")

    if not frames:
        return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def default_premarket_payload() -> dict[str, Any]:
    return {
        "premarket_volume": 0,
        "premarket_bar_count": 0,
        "premarket_first_bar_utc": None,
        "premarket_last_bar_utc": None,
        "premarket_open": None,
        "premarket_high": None,
        "premarket_low": None,
        "premarket_last": None,
        "premarket_change_pct": 0.0,
        "max_daily_volume_lookback": 0,
        "avg_daily_volume_lookback": 0,
        "premarket_to_max_daily_volume_ratio": 0.0,
        "premarket_to_avg_daily_volume_ratio": 0.0,
        "has_unusual_premarket_volume": False,
        "premarket_data_source": "yfinance_1m_prepost",
    }


def enrich_premarket_volume_with_yfinance(symbols: list[str]) -> dict[str, dict[str, Any]]:
    results = {symbol: default_premarket_payload() for symbol in symbols}

    minute_bars = fetch_yfinance_premarket_bars(symbols)
    daily_bars = fetch_yfinance_daily_bars(symbols)

    today_et = datetime.now(ET).date()
    premarket_start = datetime.combine(today_et, parse_hhmm(PREMARKET_START_ET))
    premarket_end = datetime.combine(today_et, parse_hhmm(PREMARKET_END_ET))

    if not minute_bars.empty:
        minute_bars = minute_bars.copy()
        minute_bars["timestamp_et"] = minute_bars["timestamp"].dt.tz_convert(ET)
        minute_bars["session_date_et"] = minute_bars["timestamp_et"].dt.date

        pre = minute_bars[
            (minute_bars["session_date_et"] == today_et)
            & (minute_bars["timestamp_et"] >= premarket_start)
            & (minute_bars["timestamp_et"] < premarket_end)
        ].copy()

        for symbol, sdf in pre.groupby("symbol"):
            symbol = normalize_symbol(symbol)
            if symbol not in results or sdf.empty:
                continue

            sdf = sdf.sort_values("timestamp")
            volume_sum = int(pd.to_numeric(sdf["volume"], errors="coerce").fillna(0).sum())
            first_open = to_float(sdf["open"].iloc[0])
            last_close = to_float(sdf["close"].iloc[-1])

            change_pct = (
                (last_close / first_open - 1.0) * 100.0
                if first_open > 0 and last_close > 0
                else 0.0
            )

            results[symbol].update(
                {
                    "premarket_volume": volume_sum,
                    "premarket_bar_count": int(len(sdf)),
                    "premarket_first_bar_utc": str(sdf["timestamp"].min()),
                    "premarket_last_bar_utc": str(sdf["timestamp"].max()),
                    "premarket_open": first_open,
                    "premarket_high": to_float(sdf["high"].max()),
                    "premarket_low": to_float(sdf["low"].min()),
                    "premarket_last": last_close,
                    "premarket_change_pct": round(change_pct, 4),
                }
            )

    if not daily_bars.empty:
        daily_bars = daily_bars.copy()
        daily_bars["timestamp_et"] = daily_bars["timestamp"].dt.tz_convert(ET)
        daily_bars = daily_bars[daily_bars["timestamp_et"].dt.date < today_et].copy()

        for symbol, sdf in daily_bars.groupby("symbol"):
            symbol = normalize_symbol(symbol)
            if symbol not in results or sdf.empty:
                continue

            recent = sdf.tail(DAILY_VOLUME_LOOKBACK_DAYS)
            volumes = pd.to_numeric(recent["volume"], errors="coerce").fillna(0)

            max_daily_volume = int(volumes.max())
            avg_daily_volume = int(volumes.mean())
            premarket_volume = to_int(results[symbol].get("premarket_volume"))

            ratio_to_max = (
                premarket_volume / max_daily_volume
                if premarket_volume > 0 and max_daily_volume > 0
                else 0.0
            )
            ratio_to_avg = (
                premarket_volume / avg_daily_volume
                if premarket_volume > 0 and avg_daily_volume > 0
                else 0.0
            )

            has_unusual = (
                premarket_volume >= MIN_PREMARKET_VOLUME
                or ratio_to_max >= PREMARKET_TO_MAX_DAILY_VOL_RATIO
                or ratio_to_avg >= PREMARKET_TO_AVG_DAILY_VOL_RATIO
            )

            results[symbol].update(
                {
                    "max_daily_volume_lookback": max_daily_volume,
                    "avg_daily_volume_lookback": avg_daily_volume,
                    "premarket_to_max_daily_volume_ratio": round(ratio_to_max, 4),
                    "premarket_to_avg_daily_volume_ratio": round(ratio_to_avg, 4),
                    "has_unusual_premarket_volume": bool(has_unusual),
                }
            )

    return results


def enrich_company_details(symbols: list[str]) -> dict[str, dict[str, Any]]:
    enriched: dict[str, dict[str, Any]] = {}

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)

            info: dict[str, Any] = {}
            try:
                info = ticker.get_info() or {}
            except Exception:
                info = {}

            fast_info: dict[str, Any] = {}
            try:
                fast_info = dict(ticker.fast_info or {})
            except Exception:
                fast_info = {}

            market_cap = info.get("marketCap") or fast_info.get("market_cap")
            current_price = info.get("currentPrice") or fast_info.get("last_price")
            average_volume = info.get("averageVolume") or fast_info.get("three_month_average_volume")

            enriched[symbol] = {
                "company_name": info.get("shortName") or info.get("longName") or symbol,
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": market_cap,
                "enterprise_value": info.get("enterpriseValue"),
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "profit_margins": info.get("profitMargins"),
                "gross_margins": info.get("grossMargins"),
                "operating_margins": info.get("operatingMargins"),
                "beta": info.get("beta"),
                "float_shares": info.get("floatShares"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "short_ratio": info.get("shortRatio"),
                "short_percent_of_float": info.get("shortPercentOfFloat"),
                "average_volume": average_volume,
                "current_price": current_price,
                "target_mean_price": info.get("targetMeanPrice"),
                "target_high_price": info.get("targetHighPrice"),
                "recommendation": info.get("recommendationKey"),
                "number_of_analyst_opinions": info.get("numberOfAnalystOpinions"),
            }

        except Exception as exc:
            print(f"[yfinance] enrichment failed for {symbol}: {exc}")
            enriched[symbol] = {"company_name": symbol}

    return enriched


# =============================================================================
# Scoring
# =============================================================================

def calc_relative_volume(row: dict[str, Any]) -> float | None:
    premarket_volume = to_float(row.get("premarket_volume"))
    avg_volume = to_float(row.get("avg_daily_volume_lookback") or row.get("average_volume"))

    if premarket_volume > 0 and avg_volume > 0:
        return premarket_volume / avg_volume

    return None


def calc_discovery_score(row: dict[str, Any]) -> float:
    premarket_volume = to_float(row.get("premarket_volume"))
    premarket_ratio = to_float(row.get("premarket_to_max_daily_volume_ratio"))
    premarket_avg_ratio = to_float(row.get("premarket_to_avg_daily_volume_ratio"))
    premarket_change_pct = max(to_float(row.get("premarket_change_pct")), 0.0)
    rel_volume = to_float(row.get("relative_volume"), 0.0)
    market_cap = to_float(row.get("market_cap"))
    has_unusual = bool(row.get("has_unusual_premarket_volume"))

    score = 0.0

    if premarket_volume >= MIN_PREMARKET_VOLUME:
        score += 70.0

    score += min(premarket_volume / max(MIN_PREMARKET_VOLUME, 1), 3.0) * 25.0
    score += min(max(premarket_ratio, 0.0), 3.0) * 60.0
    score += min(max(premarket_avg_ratio, 0.0), 6.0) * 18.0
    score += min(max(rel_volume, 0.0), 6.0) * 8.0
    score += min(premarket_change_pct, 40.0) * 1.5

    if has_unusual:
        score += 90.0

    if market_cap >= 50_000_000_000:
        score += 20.0
    elif market_cap >= 10_000_000_000:
        score += 16.0
    elif market_cap >= 1_000_000_000:
        score += 10.0
    elif market_cap >= 100_000_000:
        score += 4.0

    return round(score, 4)


def determine_data_catalyst_type(row: dict[str, Any]) -> str:
    premarket_volume = to_int(row.get("premarket_volume"))
    premarket_ratio = to_float(row.get("premarket_to_max_daily_volume_ratio"))
    premarket_avg_ratio = to_float(row.get("premarket_to_avg_daily_volume_ratio"))
    has_unusual = bool(row.get("has_unusual_premarket_volume"))

    revenue_growth = to_float(row.get("revenue_growth"))
    earnings_growth = to_float(row.get("earnings_growth"))
    short_ratio = to_float(row.get("short_ratio"))
    target = to_float(row.get("target_mean_price"))
    price = to_float(row.get("price") or row.get("current_price"))
    sector = str(row.get("sector") or "").lower()

    if has_unusual or premarket_volume >= MIN_PREMARKET_VOLUME or premarket_ratio >= 0.5 or premarket_avg_ratio >= 1.0:
        return "premarket_volume_spike"
    if earnings_growth > 0.15:
        return "earnings_growth"
    if revenue_growth > 0.15:
        return "revenue_growth"
    if target > 0 and price > 0 and target / price - 1.0 > 0.15:
        return "analyst_upside"
    if short_ratio >= 5:
        return "short_squeeze"
    if sector:
        return "sector_catalyst"

    return "technical_gainer"


# =============================================================================
# Discovery / LLM
# =============================================================================

def discover_candidates() -> dict[str, Any]:
    symbols = load_csv_universe()
    symbols = symbols[:LLM_CANDIDATE_LIMIT]

    print(f"[discovery] researching {len(symbols)} CSV symbols with yfinance premarket data")

    company_details = enrich_company_details(symbols)
    premarket_details = enrich_premarket_volume_with_yfinance(symbols)

    candidates: list[dict[str, Any]] = []

    for symbol in symbols:
        company = company_details.get(symbol, {})
        premarket = premarket_details.get(symbol, default_premarket_payload())

        price = company.get("current_price") or premarket.get("premarket_last")
        avg_volume = company.get("average_volume") or premarket.get("avg_daily_volume_lookback")

        row = {
            "symbol": symbol,
            "name": company.get("company_name") or symbol,
            "price": price,
            "change": None,
            "change_pct": premarket.get("premarket_change_pct"),
            "volume": premarket.get("premarket_volume"),
            "avg_volume": avg_volume,
            "relative_volume": None,
            "market_cap": company.get("market_cap"),
            "pe_ratio": company.get("trailing_pe"),
            "exchange": None,
            "quote_type": "EQUITY",
            "source": "csv_bullish_universe_yfinance_premarket",
            **company,
            **premarket,
        }

        row["relative_volume"] = calc_relative_volume(row)
        row["data_catalyst_type"] = determine_data_catalyst_type(row)
        row["discovery_score"] = calc_discovery_score(row)

        if MIN_PRICE > 0 or MAX_PRICE > 0:
            price_check = to_float(row.get("price"))
            if price_check > 0 and (price_check < MIN_PRICE or price_check > MAX_PRICE):
                row["price_filter_failed"] = True
            else:
                row["price_filter_failed"] = False

        if MIN_MARKET_CAP > 0:
            market_cap = to_float(row.get("market_cap"))
            row["market_cap_filter_failed"] = bool(market_cap > 0 and market_cap < MIN_MARKET_CAP)
        else:
            row["market_cap_filter_failed"] = False

        candidates.append(row)

    candidates = sorted(
        candidates,
        key=lambda x: (
            bool(x.get("has_unusual_premarket_volume")),
            to_float(x.get("premarket_volume")),
            to_float(x.get("premarket_to_max_daily_volume_ratio")),
            to_float(x.get("premarket_to_avg_daily_volume_ratio")),
            to_float(x.get("premarket_change_pct")),
            to_float(x.get("discovery_score")),
        ),
        reverse=True,
    )

    discovery = {
        "created_at_utc": now_utc(),
        "source": "csv_bullish_universe_with_yfinance_premarket_research",
        "input_file": str(CSV_UNIVERSE_PATH),
        "candidate_count": len(candidates),
        "filters": {
            "target_count": TARGET_COUNT,
            "llm_candidate_limit": LLM_CANDIDATE_LIMIT,
            "min_price": MIN_PRICE,
            "max_price": MAX_PRICE,
            "min_market_cap": MIN_MARKET_CAP,
            "min_premarket_volume": MIN_PREMARKET_VOLUME,
            "premarket_to_max_daily_volume_ratio": PREMARKET_TO_MAX_DAILY_VOL_RATIO,
            "premarket_to_avg_daily_volume_ratio": PREMARKET_TO_AVG_DAILY_VOL_RATIO,
            "daily_volume_lookback_days": DAILY_VOLUME_LOOKBACK_DAYS,
            "premarket_start_et": PREMARKET_START_ET,
            "premarket_end_et": PREMARKET_END_ET,
            "yfinance_intraday_period": YF_INTRADAY_PERIOD,
            "yfinance_daily_period": YF_DAILY_PERIOD,
            "yfinance_batch_size": YF_BATCH_SIZE,
        },
        "candidates": candidates[:LLM_CANDIDATE_LIMIT],
    }

    write_json(DISCOVERY_PATH, discovery)

    unusual_count = sum(1 for c in discovery["candidates"] if c.get("has_unusual_premarket_volume"))

    print(
        f"[discovery] csv_symbols={len(symbols)} "
        f"candidates={len(candidates)} "
        f"unusual_premarket={unusual_count}"
    )

    return discovery


def compact_candidates_for_llm(discovery: dict[str, Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []

    for row in discovery.get("candidates", [])[:LLM_CANDIDATE_LIMIT]:
        compact.append(
            {
                "symbol": row.get("symbol"),
                "name": row.get("name") or row.get("company_name"),
                "price": row.get("price") or row.get("current_price"),
                "premarket_change_pct": row.get("premarket_change_pct"),
                "premarket_volume": row.get("premarket_volume"),
                "premarket_bar_count": row.get("premarket_bar_count"),
                "premarket_open": row.get("premarket_open"),
                "premarket_high": row.get("premarket_high"),
                "premarket_low": row.get("premarket_low"),
                "premarket_last": row.get("premarket_last"),
                "max_daily_volume_lookback": row.get("max_daily_volume_lookback"),
                "avg_daily_volume_lookback": row.get("avg_daily_volume_lookback"),
                "premarket_to_max_daily_volume_ratio": row.get("premarket_to_max_daily_volume_ratio"),
                "premarket_to_avg_daily_volume_ratio": row.get("premarket_to_avg_daily_volume_ratio"),
                "relative_volume": row.get("relative_volume"),
                "has_unusual_premarket_volume": row.get("has_unusual_premarket_volume"),
                "premarket_data_source": row.get("premarket_data_source"),
                "market_cap": row.get("market_cap"),
                "sector": row.get("sector"),
                "industry": row.get("industry"),
                "trailing_pe": row.get("trailing_pe") or row.get("pe_ratio"),
                "forward_pe": row.get("forward_pe"),
                "revenue_growth": row.get("revenue_growth"),
                "earnings_growth": row.get("earnings_growth"),
                "profit_margins": row.get("profit_margins"),
                "short_ratio": row.get("short_ratio"),
                "short_percent_of_float": row.get("short_percent_of_float"),
                "target_mean_price": row.get("target_mean_price"),
                "recommendation": row.get("recommendation"),
                "analyst_count": row.get("number_of_analyst_opinions"),
                "discovery_score": row.get("discovery_score"),
                "data_catalyst_type": row.get("data_catalyst_type"),
            }
        )

    return compact


def build_prompt(discovery: dict[str, Any]) -> str:
    candidates = compact_candidates_for_llm(discovery)

    return f"""
Today is {datetime.now(ET).strftime("%Y-%m-%d")}.

You are given a fixed ticker universe from:
{discovery.get("input_file")}

Research and rank ONLY the supplied tickers.
Do NOT add, invent, or substitute tickers.

Use the supplied yfinance company data and yfinance 1-minute extended-hours premarket diagnostics.
Select the TOP {TARGET_COUNT} candidates for today's monitoring universe.

Primary priority:
Rank stocks with unusually high premarket volume.

Premarket volume is especially important when:
- premarket_volume >= {MIN_PREMARKET_VOLUME}
- premarket_to_max_daily_volume_ratio >= {PREMARKET_TO_MAX_DAILY_VOL_RATIO}
- premarket_to_avg_daily_volume_ratio >= {PREMARKET_TO_AVG_DAILY_VOL_RATIO}
- premarket volume is close to or above max_daily_volume_lookback
- premarket_bar_count confirms real extended-hours activity

For each selection, explain why it deserves monitoring today using at least TWO concrete fields when available.

Candidates:
{json.dumps(safe_json_value(candidates), separators=(",", ":"))}

Return the required JSON only.
""".strip()


def call_llm(discovery: dict[str, Any]) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")

    client = OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT_SECONDS)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(discovery)},
        ],
        temperature=0.2,
        max_tokens=OPENAI_MAX_TOKENS,
    )

    content = response.choices[0].message.content or ""
    return json.loads(content.strip())


# =============================================================================
# Output normalization
# =============================================================================

def fallback_reason(row: dict[str, Any]) -> str:
    symbol = row.get("symbol")
    parts = [f"{symbol} was selected from the CSV universe using yfinance premarket diagnostics"]

    premarket_volume = to_int(row.get("premarket_volume"))
    max_daily_volume = to_int(row.get("max_daily_volume_lookback"))
    avg_daily_volume = to_int(row.get("avg_daily_volume_lookback"))

    if premarket_volume:
        parts.append(f"premarket volume of {premarket_volume:,} shares")
    if max_daily_volume:
        parts.append(f"max recent daily volume of {max_daily_volume:,} shares")
    if avg_daily_volume:
        parts.append(f"average recent daily volume of {avg_daily_volume:,} shares")
    if row.get("premarket_to_max_daily_volume_ratio"):
        parts.append(f"premarket/max daily volume ratio of {to_float(row.get('premarket_to_max_daily_volume_ratio')):.2f}x")
    if row.get("premarket_to_avg_daily_volume_ratio"):
        parts.append(f"premarket/average daily volume ratio of {to_float(row.get('premarket_to_avg_daily_volume_ratio')):.2f}x")
    if row.get("premarket_change_pct") is not None:
        parts.append(f"{fmt_pct(row.get('premarket_change_pct'))} premarket move")
    if row.get("relative_volume"):
        parts.append(f"relative premarket volume of {to_float(row.get('relative_volume')):.2f}x")
    if row.get("market_cap"):
        parts.append(f"market cap of {fmt_money(row.get('market_cap'))}")
    if row.get("sector"):
        parts.append(f"{row.get('sector')} sector exposure")
    if row.get("revenue_growth"):
        parts.append(f"revenue growth of {fmt_pct(to_float(row.get('revenue_growth')) * 100)}")
    if row.get("earnings_growth"):
        parts.append(f"earnings growth of {fmt_pct(to_float(row.get('earnings_growth')) * 100)}")
    if row.get("recommendation"):
        parts.append(f"analyst recommendation of {row.get('recommendation')}")

    return clean_reason(", ".join(parts) + ".")


def selection_from_discovery_row(row: dict[str, Any], rank: int, confidence: float = 0.50) -> dict[str, Any]:
    return {
        "rank": rank,
        "symbol": normalize_symbol(row.get("symbol")),
        "catalyst_type": row.get("data_catalyst_type") or "technical_gainer",
        "reason": fallback_reason(row),
        "confidence": confidence,
        "price": row.get("price") or row.get("current_price"),
        "change_pct": row.get("premarket_change_pct"),
        "volume": row.get("premarket_volume"),
        "avg_volume": row.get("avg_volume") or row.get("average_volume"),
        "relative_volume": row.get("relative_volume"),
        "premarket_volume": row.get("premarket_volume"),
        "premarket_bar_count": row.get("premarket_bar_count"),
        "premarket_first_bar_utc": row.get("premarket_first_bar_utc"),
        "premarket_last_bar_utc": row.get("premarket_last_bar_utc"),
        "premarket_open": row.get("premarket_open"),
        "premarket_high": row.get("premarket_high"),
        "premarket_low": row.get("premarket_low"),
        "premarket_last": row.get("premarket_last"),
        "premarket_change_pct": row.get("premarket_change_pct"),
        "max_daily_volume_lookback": row.get("max_daily_volume_lookback"),
        "avg_daily_volume_lookback": row.get("avg_daily_volume_lookback"),
        "premarket_to_max_daily_volume_ratio": row.get("premarket_to_max_daily_volume_ratio"),
        "premarket_to_avg_daily_volume_ratio": row.get("premarket_to_avg_daily_volume_ratio"),
        "has_unusual_premarket_volume": row.get("has_unusual_premarket_volume"),
        "premarket_data_source": row.get("premarket_data_source"),
        "market_cap": row.get("market_cap"),
        "sector": row.get("sector"),
        "industry": row.get("industry"),
        "trailing_pe": row.get("trailing_pe") or row.get("pe_ratio"),
        "forward_pe": row.get("forward_pe"),
        "revenue_growth": row.get("revenue_growth"),
        "earnings_growth": row.get("earnings_growth"),
        "profit_margins": row.get("profit_margins"),
        "short_ratio": row.get("short_ratio"),
        "short_percent_of_float": row.get("short_percent_of_float"),
        "target_mean_price": row.get("target_mean_price"),
        "recommendation": row.get("recommendation"),
        "discovery_score": row.get("discovery_score"),
        "source": row.get("source"),
    }


def normalize_payload(payload: dict[str, Any], discovery: dict[str, Any]) -> dict[str, Any]:
    raw_selections = payload.get("selections", [])
    discovery_by_symbol = {
        normalize_symbol(row["symbol"]): row
        for row in discovery.get("candidates", [])
        if row.get("symbol")
    }

    selections: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in raw_selections:
        symbol = normalize_symbol(row.get("symbol"))

        if not is_valid_symbol(symbol) or symbol in seen or symbol not in discovery_by_symbol:
            continue

        d = discovery_by_symbol[symbol]
        base = selection_from_discovery_row(d, len(selections) + 1, to_float(row.get("confidence"), 0.0))
        base["catalyst_type"] = normalize_catalyst_type(row.get("catalyst_type") or d.get("data_catalyst_type"))
        base["reason"] = clean_reason(row.get("reason") or fallback_reason(d))
        base["confidence"] = max(0.0, min(1.0, to_float(row.get("confidence"), 0.0)))

        selections.append(base)
        seen.add(symbol)

        if len(selections) >= TARGET_COUNT:
            break

    # If LLM omits some names or fails to fill target count, backfill from deterministic ranking.
    for row in discovery.get("candidates", []):
        if len(selections) >= TARGET_COUNT:
            break

        symbol = normalize_symbol(row.get("symbol"))
        if not is_valid_symbol(symbol) or symbol in seen:
            continue

        selections.append(selection_from_discovery_row(row, len(selections) + 1, 0.50))
        seen.add(symbol)

    return {
        "created_at_utc": now_utc(),
        "model": MODEL,
        "target_count": TARGET_COUNT,
        "market_tone": str(payload.get("market_tone", "mixed")).strip() or "mixed",
        "summary": str(
            payload.get(
                "summary",
                "CSV universe ranked by yfinance premarket volume diagnostics.",
            )
        ).strip(),
        "symbols": [row["symbol"] for row in selections],
        "quoted_csv": ",".join(f'"{row["symbol"]}"' for row in selections),
        "selections": selections,
        "discovery": {
            "source": discovery.get("source"),
            "candidate_count": discovery.get("candidate_count", 0),
            "source_file": str(DISCOVERY_PATH),
            "input_file": discovery.get("input_file"),
        },
    }


def write_outputs(payload: dict[str, Any]) -> None:
    write_json(JSON_PATH, payload)

    LOCKED_UNIVERSE_PATH.write_text(
        "\n".join(payload["symbols"]) + "\n",
        encoding="utf-8",
    )

    print(f"[saved] {LOCKED_UNIVERSE_PATH}")
    print(payload["quoted_csv"])


def build_sentiment_universe() -> dict[str, Any]:
    discovery = discover_candidates()

    try:
        raw = call_llm(discovery)
    except Exception as exc:
        print(f"[llm-universe] LLM failed, using deterministic yfinance ranking: {exc}")
        raw = {
            "market_tone": "mixed",
            "summary": "LLM unavailable; using CSV universe ranked by yfinance premarket volume, relative volume, liquidity, and metadata.",
            "selections": [
                {
                    "rank": i + 1,
                    "symbol": row["symbol"],
                    "catalyst_type": row.get("data_catalyst_type") or "technical_gainer",
                    "reason": fallback_reason(row),
                    "confidence": 0.50,
                }
                for i, row in enumerate(discovery.get("candidates", [])[:TARGET_COUNT])
            ],
        }

    payload = normalize_payload(raw, discovery)
    write_outputs(payload)
    return payload


if __name__ == "__main__":
    build_sentiment_universe()
