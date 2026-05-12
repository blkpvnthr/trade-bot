#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

OUT_DIR = PROJECT_ROOT / "daily_system_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = OUT_DIR / "latest_llm_sentiment_universe.json"
DISCOVERY_PATH = OUT_DIR / "latest_llm_candidate_discovery.json"
LOCKED_UNIVERSE_PATH = PROJECT_ROOT / "locked_trading_universe.txt"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

MODEL = os.getenv("OPENAI_SENTIMENT_UNIVERSE_MODEL", "gpt-4.1-mini")
TARGET_COUNT = int(os.getenv("TARGET_UNIVERSE_COUNT", "30"))
YAHOO_GAINERS_COUNT = int(os.getenv("YAHOO_GAINERS_COUNT", "40"))
LLM_CANDIDATE_LIMIT = int(os.getenv("LLM_CANDIDATE_LIMIT", "40"))
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "3500"))

MIN_PRICE = float(os.getenv("YAHOO_MIN_PRICE", "2"))
MAX_PRICE = float(os.getenv("YAHOO_MAX_PRICE", "500"))
MIN_VOLUME = int(os.getenv("YAHOO_MIN_VOLUME", "500000"))
MIN_MARKET_CAP = int(os.getenv("YAHOO_MIN_MARKET_CAP", "100000000"))

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
    "macro_catalyst",
}

SYSTEM_PROMPT = """
You are an institutional premarket stock selection engine.

You receive Yahoo Finance day-gainers plus company financial details.
Your job is to select the TOP 30 candidates for today's monitoring universe.

Identify the most compelling trade-monitoring candidates based on:
- strongest percent gain
- unusual volume / relative volume
- liquidity
- market cap quality
- sector or industry tailwind
- revenue growth
- earnings growth
- profitability
- analyst upside
- favorable recommendation
- short squeeze potential
- business relevance to current market themes

Return ONLY valid JSON.

Schema:
{
  "market_tone": "bullish | bearish | mixed | risk_off | risk_on",
  "summary": "One concise sentence describing the overall opportunity set.",
  "selections": [
    {
      "rank": 1,
      "symbol": "NVDA",
      "catalyst_type": "technical_gainer | financial_strength | analyst_upside | sector_catalyst | short_squeeze | liquidity_momentum | earnings_growth | revenue_growth | premarket_news_move | macro_catalyst",
      "reason": "Compelling 1-2 sentence explanation using concrete data from the supplied Yahoo/financial details.",
      "confidence": 0.0
    }
  ]
}

Rules:
- EXACTLY 30 selections.
- Rank from 1 to 30.
- Use "symbol", not "ticker".
- Symbols must be uppercase.
- No duplicate symbols.
- Every reason must be specific and data-backed.
- Every reason must include at least TWO concrete details when available.
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
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


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
    return text[:700] or "Selected from Yahoo gainers and company financial data."


def normalize_catalyst_type(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return raw if raw in CATALYST_TYPES else "technical_gainer"


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

    score = 0.0
    score += min(change_pct, 80.0) * 2.2
    score += min(max(rel_volume, 0.0), 15.0) * 9.0

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
    revenue_growth = to_float(row.get("revenue_growth"))
    earnings_growth = to_float(row.get("earnings_growth"))
    short_ratio = to_float(row.get("short_ratio"))
    target = to_float(row.get("target_mean_price"))
    price = to_float(row.get("price") or row.get("current_price"))
    rel_volume = to_float(row.get("relative_volume"))
    change_pct = to_float(row.get("change_pct"))
    sector = str(row.get("sector") or "").lower()

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


def discover_candidates() -> dict[str, Any]:
    gainers = fetch_yahoo_gainers(YAHOO_GAINERS_COUNT)

    if gainers.empty:
        rows = [
            {
                "symbol": s,
                "name": s,
                "price": None,
                "change_pct": None,
                "volume": None,
                "avg_volume": None,
                "market_cap": None,
                "relative_volume": None,
                "discovery_score": 0.0,
                "source": "fallback",
            }
            for s in FALLBACK
        ]
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

            row["discovery_score"] = calc_discovery_score(row)
            rows.append(row)

    rows = sorted(rows, key=lambda x: to_float(x.get("discovery_score")), reverse=True)
    rows = rows[:max(YAHOO_GAINERS_COUNT, TARGET_COUNT)]

    symbols = [r["symbol"] for r in rows if is_valid_symbol(r.get("symbol"))]
    enriched = enrich_company_details(symbols[:LLM_CANDIDATE_LIMIT])

    candidates: list[dict[str, Any]] = []

    for row in rows[:LLM_CANDIDATE_LIMIT]:
        symbol = row["symbol"]
        details = enriched.get(symbol, {})
        merged = {**row, **details}

        if not merged.get("market_cap"):
            merged["market_cap"] = row.get("market_cap")

        if not merged.get("price"):
            merged["price"] = merged.get("current_price")

        merged["data_catalyst_type"] = determine_data_catalyst_type(merged)
        candidates.append(merged)

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
                    }
                )

    discovery = {
        "created_at_utc": now_utc(),
        "source": "yahoo_day_gainers",
        "candidate_count": len(candidates),
        "filters": {
            "min_price": MIN_PRICE,
            "max_price": MAX_PRICE,
            "min_volume": MIN_VOLUME,
            "min_market_cap": MIN_MARKET_CAP,
            "yahoo_gainers_count": YAHOO_GAINERS_COUNT,
            "llm_candidate_limit": LLM_CANDIDATE_LIMIT,
        },
        "candidates": candidates[:LLM_CANDIDATE_LIMIT],
    }

    DISCOVERY_PATH.write_text(json.dumps(discovery, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {DISCOVERY_PATH}")
    print(f"[discovery] yahoo_gainers candidates={len(candidates)}")

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
Today is {datetime.now().strftime("%Y-%m-%d")}.

Use the Yahoo Finance day-gainers and company financial data below.
Select the TOP 30 candidates for today's monitoring universe.

For each selection, explain why it deserves monitoring today using at least TWO concrete fields when available.

Concrete fields include:
- percent gain
- volume
- relative volume
- market cap
- sector/industry
- revenue growth
- earnings growth
- profit margins
- P/E or forward P/E
- analyst target
- recommendation
- short ratio

Yahoo gainers and financial details:
{json.dumps(candidates, separators=(",", ":"))}

Return the required JSON only.
""".strip()


def call_llm(discovery: dict[str, Any]) -> dict[str, Any]:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=OPENAI_TIMEOUT_SECONDS,
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=2500,
        timeout=45,
    )

    return json.loads(response.choices[0].message.content.strip())


def fallback_reason(row: dict[str, Any]) -> str:
    symbol = row.get("symbol")
    change_pct = row.get("change_pct")
    volume = row.get("volume")
    rel_volume = row.get("relative_volume")
    market_cap = row.get("market_cap")
    sector = row.get("sector")
    revenue_growth = row.get("revenue_growth")
    earnings_growth = row.get("earnings_growth")
    recommendation = row.get("recommendation")

    parts = [
        f"{symbol} was added from Yahoo's gainers list with a {fmt_pct(change_pct)} move",
        f"volume of {volume:,}" if isinstance(volume, (int, float)) else f"volume of {volume}",
    ]

    if rel_volume:
        parts.append(f"relative volume of {to_float(rel_volume):.2f}x")
    if market_cap:
        parts.append(f"market cap of {fmt_money(market_cap)}")
    if sector:
        parts.append(f"{sector} sector exposure")
    if revenue_growth:
        parts.append(f"revenue growth of {fmt_pct(to_float(revenue_growth) * 100)}")
    if earnings_growth:
        parts.append(f"earnings growth of {fmt_pct(to_float(earnings_growth) * 100)}")
    if recommendation:
        parts.append(f"analyst recommendation of {recommendation}")

    return clean_reason(", ".join(parts) + ".")


def normalize_payload(payload: dict[str, Any], discovery: dict[str, Any]) -> dict[str, Any]:
    raw_selections = payload.get("selections", [])

    discovery_by_symbol = {
        row["symbol"]: row for row in discovery.get("candidates", []) if row.get("symbol")
    }

    selections: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in raw_selections:
        symbol = normalize_symbol(row.get("symbol"))

        if not is_valid_symbol(symbol) or symbol in seen:
            continue

        d = discovery_by_symbol.get(symbol, {})
        catalyst_type = normalize_catalyst_type(row.get("catalyst_type") or d.get("data_catalyst_type"))
        confidence = max(0.0, min(1.0, to_float(row.get("confidence"), 0.0)))

        seen.add(symbol)

        selections.append(
            {
                "rank": len(selections) + 1,
                "symbol": symbol,
                "catalyst_type": catalyst_type,
                "reason": clean_reason(row.get("reason")),
                "confidence": confidence,
                "price": d.get("price") or d.get("current_price"),
                "change_pct": d.get("change_pct"),
                "volume": d.get("volume"),
                "avg_volume": d.get("avg_volume") or d.get("average_volume"),
                "relative_volume": d.get("relative_volume"),
                "market_cap": d.get("market_cap"),
                "sector": d.get("sector"),
                "industry": d.get("industry"),
                "trailing_pe": d.get("trailing_pe") or d.get("pe_ratio"),
                "forward_pe": d.get("forward_pe"),
                "revenue_growth": d.get("revenue_growth"),
                "earnings_growth": d.get("earnings_growth"),
                "profit_margins": d.get("profit_margins"),
                "short_ratio": d.get("short_ratio"),
                "short_percent_of_float": d.get("short_percent_of_float"),
                "target_mean_price": d.get("target_mean_price"),
                "recommendation": d.get("recommendation"),
                "discovery_score": d.get("discovery_score"),
                "source": d.get("source"),
            }
        )

        if len(selections) >= TARGET_COUNT:
            break

    for row in discovery.get("candidates", []):
        if len(selections) >= TARGET_COUNT:
            break

        symbol = normalize_symbol(row.get("symbol"))

        if not is_valid_symbol(symbol) or symbol in seen:
            continue

        seen.add(symbol)

        selections.append(
            {
                "rank": len(selections) + 1,
                "symbol": symbol,
                "catalyst_type": row.get("data_catalyst_type") or "technical_gainer",
                "reason": fallback_reason(row),
                "confidence": 0.50,
                "price": row.get("price") or row.get("current_price"),
                "change_pct": row.get("change_pct"),
                "volume": row.get("volume"),
                "avg_volume": row.get("avg_volume") or row.get("average_volume"),
                "relative_volume": row.get("relative_volume"),
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
        )

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
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

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
        print(f"[llm-universe] LLM failed, using Yahoo gainer ranking: {exc}")
        raw = {
            "market_tone": "mixed",
            "summary": "LLM unavailable; using Yahoo Finance day-gainers ranked by price move, liquidity, and financial details.",
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