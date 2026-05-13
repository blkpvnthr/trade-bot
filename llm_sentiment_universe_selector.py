#!/usr/bin/env python3
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

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from market_context_builder import TODAY_TAG

try:
    from alpaca.data.enums import DataFeed
except Exception:
    DataFeed = None


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

OUT_DIR = PROJECT_ROOT / "daily_system_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = OUT_DIR / f"{TODAY_TAG}_latest_llm_sentiment_universe.json"
DISCOVERY_PATH = OUT_DIR / f"{TODAY_TAG}_latest_llm_candidate_discovery.json"
LOCKED_UNIVERSE_PATH = PROJECT_ROOT / f"{TODAY_TAG}_locked_trading_universe.txt"

ET = ZoneInfo("America/New_York")

MODEL = os.getenv("OPENAI_SENTIMENT_UNIVERSE_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))

TARGET_COUNT = int(os.getenv("TARGET_UNIVERSE_COUNT", "30"))
YAHOO_GAINERS_COUNT = int(os.getenv("YAHOO_GAINERS_COUNT", "50"))
LLM_CANDIDATE_LIMIT = int(os.getenv("LLM_CANDIDATE_LIMIT", "50"))

OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "3500"))

MIN_PRICE = float(os.getenv("YAHOO_MIN_PRICE", "2"))
MAX_PRICE = float(os.getenv("YAHOO_MAX_PRICE", "500"))
MIN_VOLUME = int(os.getenv("YAHOO_MIN_VOLUME", "500000"))
MIN_MARKET_CAP = int(os.getenv("YAHOO_MIN_MARKET_CAP", "100000000"))

MIN_PREMARKET_VOLUME = int(os.getenv("MIN_PREMARKET_VOLUME", "1000000"))
PREMARKET_TO_MAX_DAILY_VOL_RATIO = float(os.getenv("PREMARKET_TO_MAX_DAILY_VOL_RATIO", "0.75"))
DAILY_VOLUME_LOOKBACK_DAYS = int(os.getenv("DAILY_VOLUME_LOOKBACK_DAYS", "90"))

ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex").lower().strip()
ALPACA_CHUNK_SIZE = int(os.getenv("ALPACA_CHUNK_SIZE", "25"))

PREMARKET_START_ET = os.getenv("PREMARKET_START_ET", "04:00")
PREMARKET_END_ET = os.getenv("PREMARKET_END_ET", "09:30")

FALLBACK = [
    "NVDA", "AMD", "SMCI", "AVGO", "ARM", "PLTR", "TSLA", "META", "MSFT", "GOOGL",
    "AMZN", "AAPL", "COIN", "MARA", "RIOT", "CLSK", "IREN", "IONQ", "RGTI", "QBTS",
    "QUBT", "SOUN", "AI", "BBAI", "OKLO", "SMR", "CEG", "VST", "SOXX", "QQQ",
]

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

You receive Yahoo Finance day-gainers, company financial details, and Alpaca premarket volume diagnostics.
Your job is to select the TOP candidates for today's monitoring universe.

Primary Step 1:
Prioritize stocks with unusual premarket volume.

A stock has high premarket interest when:
- Alpaca 1-minute extended-hours premarket volume is at least 1,000,000 shares
- premarket volume is close to or above the highest daily volume in the recent lookback window
- premarket volume is unusually high versus average daily volume

If premarket volume is already near or above the stock's highest daily volume day, treat it as a major catalyst because it suggests elevated probability of a large regular-session move.

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
- No duplicate symbols.
- Every reason must include at least TWO concrete details when available.
- Prefer symbols with confirmed unusual Alpaca premarket volume.
- Do not invent facts not present in the supplied data.
- Confidence must be between 0 and 1.
- No markdown.
- No extra commentary.
""".strip()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_symbol(value: Any) -> str:
    return str(value or "").upper().strip().replace(".", "-")


def is_valid_symbol(symbol: Any) -> bool:
    symbol = normalize_symbol(symbol)
    return bool(re.fullmatch(r"[A-Z][A-Z0-9-]{1,9}", symbol))


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


def safe_json_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: safe_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json_value(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(safe_json_value(payload), indent=2) + "\n", encoding="utf-8")


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
    return text[:700] or "Selected from Alpaca premarket, Yahoo gainers, and company financial data."


def normalize_catalyst_type(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return raw if raw in CATALYST_TYPES else "technical_gainer"


def parse_hhmm(value: str) -> time:
    h, m = value.split(":")
    return time(int(h), int(m), tzinfo=ET)


def get_alpaca_data_client() -> StockHistoricalDataClient | None:
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")

    if not key or not secret:
        print("[alpaca] missing APCA_API_KEY_ID / APCA_API_SECRET_KEY; premarket volume will be zero")
        return None

    return StockHistoricalDataClient(key, secret)


def get_alpaca_feed() -> Any:
    if DataFeed is None:
        return None

    if ALPACA_DATA_FEED == "sip":
        return DataFeed.SIP

    return DataFeed.IEX


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def fetch_yahoo_gainers(limit: int = YAHOO_GAINERS_COUNT) -> pd.DataFrame:
    try:
        result = yf.screen("day_gainers", count=limit)
        quotes = result.get("quotes", []) if isinstance(result, dict) else []

        rows: list[dict[str, Any]] = []

        for q in quotes:
            symbol = normalize_symbol(q.get("symbol"))

            if not is_valid_symbol(symbol):
                continue

            rows.append(
                {
                    "symbol": symbol,
                    "name": q.get("shortName") or q.get("longName") or "",
                    "price": q.get("regularMarketPrice"),
                    "change": q.get("regularMarketChange"),
                    "change_pct": q.get("regularMarketChangePercent"),
                    "volume": q.get("regularMarketVolume"),
                    "avg_volume": q.get("averageDailyVolume3Month") or q.get("averageDailyVolume10Day"),
                    "market_cap": q.get("marketCap"),
                    "pe_ratio": q.get("trailingPE"),
                    "exchange": q.get("exchange"),
                    "quote_type": q.get("quoteType"),
                }
            )

        return pd.DataFrame(rows)

    except Exception as exc:
        print(f"[yahoo] failed to fetch day_gainers screen: {exc}")
        return pd.DataFrame()


def passes_basic_filters(row: dict[str, Any]) -> bool:
    price = to_float(row.get("price"))
    volume = to_int(row.get("volume"))
    market_cap = to_int(row.get("market_cap"))

    if not is_valid_symbol(row.get("symbol")):
        return False
    if price < MIN_PRICE or price > MAX_PRICE:
        return False
    if volume < MIN_VOLUME:
        return False
    if market_cap and market_cap < MIN_MARKET_CAP:
        return False

    return True


def enrich_company_details(symbols: list[str]) -> dict[str, dict[str, Any]]:
    enriched: dict[str, dict[str, Any]] = {}

    for symbol in symbols:
        try:
            info = yf.Ticker(symbol).get_info() or {}

            enriched[symbol] = {
                "company_name": info.get("shortName") or info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
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
                "average_volume": info.get("averageVolume"),
                "current_price": info.get("currentPrice"),
                "target_mean_price": info.get("targetMeanPrice"),
                "target_high_price": info.get("targetHighPrice"),
                "recommendation": info.get("recommendationKey"),
                "number_of_analyst_opinions": info.get("numberOfAnalystOpinions"),
            }

        except Exception as exc:
            print(f"[yahoo] enrichment failed for {symbol}: {exc}")
            enriched[symbol] = {}

    return enriched


def default_premarket_payload() -> dict[str, Any]:
    return {
        "premarket_volume": 0,
        "premarket_bar_count": 0,
        "premarket_first_bar_utc": None,
        "premarket_last_bar_utc": None,
        "max_daily_volume_lookback": 0,
        "avg_daily_volume_lookback": 0,
        "premarket_to_max_daily_volume_ratio": 0.0,
        "premarket_to_avg_daily_volume_ratio": 0.0,
        "has_unusual_premarket_volume": False,
        "premarket_data_source": "alpaca_1min_extended_hours",
    }


def enrich_premarket_volume_with_alpaca(symbols: list[str]) -> dict[str, dict[str, Any]]:
    results = {symbol: default_premarket_payload() for symbol in symbols}
    client = get_alpaca_data_client()

    if client is None:
        return results

    today_et = datetime.now(ET).date()
    premarket_start = datetime.combine(today_et, parse_hhmm(PREMARKET_START_ET)).astimezone(timezone.utc)
    premarket_end = datetime.combine(today_et, parse_hhmm(PREMARKET_END_ET)).astimezone(timezone.utc)

    daily_end = datetime.combine(today_et, time(0, 0), tzinfo=ET).astimezone(timezone.utc)
    daily_start = daily_end - timedelta(days=max(DAILY_VOLUME_LOOKBACK_DAYS * 2, 120))

    feed = get_alpaca_feed()

    for group in chunked(symbols, ALPACA_CHUNK_SIZE):
        try:
            minute_kwargs = {
                "symbol_or_symbols": group,
                "timeframe": TimeFrame.Minute,
                "start": premarket_start,
                "end": premarket_end,
            }
            if feed is not None:
                minute_kwargs["feed"] = feed

            minute_request = StockBarsRequest(**minute_kwargs)
            minute_bars = client.get_stock_bars(minute_request).df

            if minute_bars is not None and not minute_bars.empty:
                df = minute_bars.reset_index()

                for symbol in group:
                    sdf = df[df["symbol"] == symbol] if "symbol" in df.columns else pd.DataFrame()

                    if sdf.empty:
                        continue

                    volume_sum = int(sdf["volume"].fillna(0).sum()) if "volume" in sdf.columns else 0

                    results[symbol]["premarket_volume"] = volume_sum
                    results[symbol]["premarket_bar_count"] = int(len(sdf))

                    if "timestamp" in sdf.columns and not sdf["timestamp"].empty:
                        results[symbol]["premarket_first_bar_utc"] = str(sdf["timestamp"].min())
                        results[symbol]["premarket_last_bar_utc"] = str(sdf["timestamp"].max())

        except Exception as exc:
            print(f"[alpaca] premarket 1Min fetch failed for {group}: {exc}")

        try:
            daily_kwargs = {
                "symbol_or_symbols": group,
                "timeframe": TimeFrame.Day,
                "start": daily_start,
                "end": daily_end,
            }
            if feed is not None:
                daily_kwargs["feed"] = feed

            daily_request = StockBarsRequest(**daily_kwargs)
            daily_bars = client.get_stock_bars(daily_request).df

            if daily_bars is not None and not daily_bars.empty:
                df = daily_bars.reset_index()

                for symbol in group:
                    sdf = df[df["symbol"] == symbol] if "symbol" in df.columns else pd.DataFrame()

                    if sdf.empty or "volume" not in sdf.columns:
                        continue

                    recent = sdf.tail(DAILY_VOLUME_LOOKBACK_DAYS)
                    max_daily_volume = int(recent["volume"].fillna(0).max())
                    avg_daily_volume = int(recent["volume"].fillna(0).mean())

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
                        and ratio_to_max >= PREMARKET_TO_MAX_DAILY_VOL_RATIO
                    )

                    results[symbol].update(
                        {
                            "max_daily_volume_lookback": max_daily_volume,
                            "avg_daily_volume_lookback": avg_daily_volume,
                            "premarket_to_max_daily_volume_ratio": round(ratio_to_max, 4),
                            "premarket_to_avg_daily_volume_ratio": round(ratio_to_avg, 4),
                            "has_unusual_premarket_volume": has_unusual,
                        }
                    )

        except Exception as exc:
            print(f"[alpaca] daily volume fetch failed for {group}: {exc}")

    return results


def calc_relative_volume(row: dict[str, Any]) -> float | None:
    volume = to_float(row.get("volume"))
    avg_volume = to_float(row.get("avg_volume"))

    if volume > 0 and avg_volume > 0:
        return volume / avg_volume

    return None


def calc_discovery_score(row: dict[str, Any]) -> float:
    change_pct = max(to_float(row.get("change_pct")), 0.0)
    rel_volume = to_float(row.get("relative_volume"), 1.0)
    market_cap = to_float(row.get("market_cap"))

    premarket_volume = to_float(row.get("premarket_volume"))
    premarket_ratio = to_float(row.get("premarket_to_max_daily_volume_ratio"))
    premarket_avg_ratio = to_float(row.get("premarket_to_avg_daily_volume_ratio"))
    has_unusual = bool(row.get("has_unusual_premarket_volume"))

    score = 0.0
    score += min(change_pct, 80.0) * 2.2
    score += min(max(rel_volume, 0.0), 15.0) * 9.0

    if premarket_volume >= MIN_PREMARKET_VOLUME:
        score += 45

    score += min(premarket_ratio, 2.5) * 55
    score += min(premarket_avg_ratio, 5.0) * 12

    if has_unusual:
        score += 80

    if market_cap >= 50_000_000_000:
        score += 22
    elif market_cap >= 10_000_000_000:
        score += 18
    elif market_cap >= 1_000_000_000:
        score += 12
    elif market_cap >= 100_000_000:
        score += 5

    return round(score, 4)


def determine_data_catalyst_type(row: dict[str, Any]) -> str:
    premarket_volume = to_int(row.get("premarket_volume"))
    premarket_ratio = to_float(row.get("premarket_to_max_daily_volume_ratio"))
    has_unusual = bool(row.get("has_unusual_premarket_volume"))

    revenue_growth = to_float(row.get("revenue_growth"))
    earnings_growth = to_float(row.get("earnings_growth"))
    short_ratio = to_float(row.get("short_ratio"))
    target = to_float(row.get("target_mean_price"))
    price = to_float(row.get("price") or row.get("current_price"))
    rel_volume = to_float(row.get("relative_volume"))
    change_pct = to_float(row.get("change_pct"))
    sector = str(row.get("sector") or "").lower()

    if has_unusual or (premarket_volume >= MIN_PREMARKET_VOLUME and premarket_ratio >= 0.5):
        return "premarket_volume_spike"
    if earnings_growth > 0.15:
        return "earnings_growth"
    if revenue_growth > 0.15:
        return "revenue_growth"
    if target > 0 and price > 0 and target / price - 1.0 > 0.15:
        return "analyst_upside"
    if short_ratio >= 5:
        return "short_squeeze"
    if rel_volume >= 2 or change_pct >= 8:
        return "liquidity_momentum"
    if sector:
        return "sector_catalyst"

    return "technical_gainer"


def build_fallback_rows() -> list[dict[str, Any]]:
    return [
        {
            "symbol": s,
            "name": s,
            "price": None,
            "change_pct": None,
            "volume": None,
            "avg_volume": None,
            "market_cap": None,
            "relative_volume": None,
            **default_premarket_payload(),
            "discovery_score": 0.0,
            "source": "fallback",
        }
        for s in FALLBACK
    ]


def discover_candidates() -> dict[str, Any]:
    gainers = fetch_yahoo_gainers(YAHOO_GAINERS_COUNT)

    if gainers.empty:
        rows = build_fallback_rows()
    else:
        raw_rows = gainers.to_dict("records")
        filtered = [r for r in raw_rows if passes_basic_filters(r)]

        if len(filtered) < TARGET_COUNT:
            filtered = raw_rows

        rows = []

        for r in filtered:
            symbol = normalize_symbol(r.get("symbol"))

            if not is_valid_symbol(symbol):
                continue

            row = {
                "symbol": symbol,
                "name": r.get("name"),
                "price": r.get("price"),
                "change": r.get("change"),
                "change_pct": r.get("change_pct"),
                "volume": r.get("volume"),
                "avg_volume": r.get("avg_volume"),
                "relative_volume": calc_relative_volume(r),
                "market_cap": r.get("market_cap"),
                "pe_ratio": r.get("pe_ratio"),
                "exchange": r.get("exchange"),
                "quote_type": r.get("quote_type"),
                "source": "yahoo_day_gainers",
            }

            rows.append(row)

    rows = rows[:max(YAHOO_GAINERS_COUNT, TARGET_COUNT)]
    symbols = [r["symbol"] for r in rows if is_valid_symbol(r.get("symbol"))]
    symbols = symbols[:LLM_CANDIDATE_LIMIT]

    company_details = enrich_company_details(symbols)
    premarket_details = enrich_premarket_volume_with_alpaca(symbols)

    candidates: list[dict[str, Any]] = []

    for row in rows[:LLM_CANDIDATE_LIMIT]:
        symbol = row["symbol"]
        merged = {
            **row,
            **company_details.get(symbol, {}),
            **premarket_details.get(symbol, default_premarket_payload()),
        }

        if not merged.get("market_cap"):
            merged["market_cap"] = row.get("market_cap")

        if not merged.get("price"):
            merged["price"] = merged.get("current_price")

        merged["data_catalyst_type"] = determine_data_catalyst_type(merged)
        merged["discovery_score"] = calc_discovery_score(merged)

        candidates.append(merged)

    candidates = sorted(candidates, key=lambda x: to_float(x.get("discovery_score")), reverse=True)

    if len(candidates) < TARGET_COUNT:
        existing = {c["symbol"] for c in candidates}

        for symbol in FALLBACK:
            if symbol not in existing:
                candidates.append(
                    {
                        "symbol": symbol,
                        "name": symbol,
                        "source": "fallback",
                        "discovery_score": 0.0,
                        "data_catalyst_type": "sector_catalyst",
                        **default_premarket_payload(),
                    }
                )

    discovery = {
        "created_at_utc": now_utc(),
        "source": "yahoo_day_gainers_with_alpaca_premarket_1min_volume",
        "candidate_count": len(candidates),
        "filters": {
            "min_price": MIN_PRICE,
            "max_price": MAX_PRICE,
            "min_volume": MIN_VOLUME,
            "min_market_cap": MIN_MARKET_CAP,
            "min_premarket_volume": MIN_PREMARKET_VOLUME,
            "premarket_to_max_daily_volume_ratio": PREMARKET_TO_MAX_DAILY_VOL_RATIO,
            "daily_volume_lookback_days": DAILY_VOLUME_LOOKBACK_DAYS,
            "alpaca_data_feed": ALPACA_DATA_FEED,
            "premarket_start_et": PREMARKET_START_ET,
            "premarket_end_et": PREMARKET_END_ET,
            "yahoo_gainers_count": YAHOO_GAINERS_COUNT,
            "llm_candidate_limit": LLM_CANDIDATE_LIMIT,
        },
        "candidates": candidates[:LLM_CANDIDATE_LIMIT],
    }

    write_json(DISCOVERY_PATH, discovery)

    unusual_count = sum(1 for c in discovery["candidates"] if c.get("has_unusual_premarket_volume"))

    print(f"[saved] {DISCOVERY_PATH}")
    print(f"[discovery] candidates={len(candidates)} unusual_premarket={unusual_count}")

    return discovery


def compact_candidates_for_llm(discovery: dict[str, Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []

    for row in discovery.get("candidates", [])[:LLM_CANDIDATE_LIMIT]:
        compact.append(
            {
                "symbol": row.get("symbol"),
                "name": row.get("name") or row.get("company_name"),
                "price": row.get("price") or row.get("current_price"),
                "change_pct": row.get("change_pct"),
                "volume": row.get("volume"),
                "avg_volume": row.get("avg_volume") or row.get("average_volume"),
                "relative_volume": row.get("relative_volume"),
                "premarket_volume": row.get("premarket_volume"),
                "premarket_bar_count": row.get("premarket_bar_count"),
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

Use the supplied candidates, Yahoo Finance gainer data, company financial data, and Alpaca premarket 1-minute volume diagnostics.
Select the TOP {TARGET_COUNT} candidates for today's monitoring universe.

Step 1 priority:
Find stocks with unusual premarket volume.

Premarket volume is especially important when:
- premarket_volume >= {MIN_PREMARKET_VOLUME}
- premarket_to_max_daily_volume_ratio >= {PREMARKET_TO_MAX_DAILY_VOL_RATIO}
- premarket volume is close to or above max_daily_volume_lookback
- premarket_to_avg_daily_volume_ratio is elevated
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

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(discovery)},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=OPENAI_MAX_TOKENS,
    )

    content = response.choices[0].message.content or ""
    return json.loads(content.strip())


def fallback_reason(row: dict[str, Any]) -> str:
    symbol = row.get("symbol")
    parts = [f"{symbol} was selected from the Alpaca premarket/Yahoo discovery process"]

    premarket_volume = to_int(row.get("premarket_volume"))
    max_daily_volume = to_int(row.get("max_daily_volume_lookback"))
    avg_daily_volume = to_int(row.get("avg_daily_volume_lookback"))

    if premarket_volume:
        parts.append(f"Alpaca premarket volume of {premarket_volume:,} shares")
    if max_daily_volume:
        parts.append(f"max recent daily volume of {max_daily_volume:,} shares")
    if avg_daily_volume:
        parts.append(f"average recent daily volume of {avg_daily_volume:,} shares")
    if row.get("premarket_to_max_daily_volume_ratio"):
        parts.append(f"premarket/max daily volume ratio of {to_float(row.get('premarket_to_max_daily_volume_ratio')):.2f}x")
    if row.get("premarket_to_avg_daily_volume_ratio"):
        parts.append(f"premarket/average daily volume ratio of {to_float(row.get('premarket_to_avg_daily_volume_ratio')):.2f}x")
    if row.get("change_pct") is not None:
        parts.append(f"{fmt_pct(row.get('change_pct'))} price move")
    if row.get("relative_volume"):
        parts.append(f"relative volume of {to_float(row.get('relative_volume')):.2f}x")
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
        "change_pct": row.get("change_pct"),
        "volume": row.get("volume"),
        "avg_volume": row.get("avg_volume") or row.get("average_volume"),
        "relative_volume": row.get("relative_volume"),
        "premarket_volume": row.get("premarket_volume"),
        "premarket_bar_count": row.get("premarket_bar_count"),
        "premarket_first_bar_utc": row.get("premarket_first_bar_utc"),
        "premarket_last_bar_utc": row.get("premarket_last_bar_utc"),
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

        if not is_valid_symbol(symbol) or symbol in seen:
            continue

        d = discovery_by_symbol.get(symbol, {})
        base = selection_from_discovery_row(d, len(selections) + 1, to_float(row.get("confidence"), 0.0))
        base["catalyst_type"] = normalize_catalyst_type(row.get("catalyst_type") or d.get("data_catalyst_type"))
        base["reason"] = clean_reason(row.get("reason") or fallback_reason(d))
        base["confidence"] = max(0.0, min(1.0, to_float(row.get("confidence"), 0.0)))

        seen.add(symbol)
        selections.append(base)

        if len(selections) >= TARGET_COUNT:
            break

    for row in discovery.get("candidates", []):
        if len(selections) >= TARGET_COUNT:
            break

        symbol = normalize_symbol(row.get("symbol"))

        if not is_valid_symbol(symbol) or symbol in seen:
            continue

        seen.add(symbol)
        selections.append(selection_from_discovery_row(row, len(selections) + 1, 0.50))

    return {
        "created_at_utc": now_utc(),
        "model": MODEL,
        "target_count": TARGET_COUNT,
        "market_tone": str(payload.get("market_tone", "mixed")).strip() or "mixed",
        "summary": str(payload.get("summary", "Premarket tone is mixed.")).strip(),
        "symbols": [row["symbol"] for row in selections],
        "quoted_csv": ",".join(f'"{row["symbol"]}"' for row in selections),
        "selections": selections,
        "discovery": {
            "source": discovery.get("source"),
            "candidate_count": discovery.get("candidate_count", 0),
            "source_file": str(DISCOVERY_PATH),
        },
    }


def write_outputs(payload: dict[str, Any]) -> None:
    write_json(JSON_PATH, payload)

    LOCKED_UNIVERSE_PATH.write_text(
        "\n".join(payload["symbols"]) + "\n",
        encoding="utf-8",
    )

    print(f"[saved] {JSON_PATH}")
    print(f"[saved] {LOCKED_UNIVERSE_PATH}")
    print(payload["quoted_csv"])


def build_sentiment_universe() -> dict[str, Any]:
    discovery = discover_candidates()

    try:
        raw = call_llm(discovery)
    except Exception as exc:
        print(f"[llm-universe] LLM failed, using discovery ranking: {exc}")
        raw = {
            "market_tone": "mixed",
            "summary": "LLM unavailable; using Alpaca premarket volume, Yahoo gainer ranking, liquidity, and financial details.",
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