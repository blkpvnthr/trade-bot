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

MODEL = os.getenv("OPENAI_SENTIMENT_UNIVERSE_MODEL", "gpt-4.1-mini")
TARGET_COUNT = 30
YAHOO_GAINERS_COUNT = int(os.getenv("YAHOO_GAINERS_COUNT", "100"))

MIN_PRICE = float(os.getenv("YAHOO_MIN_PRICE", "2"))
MAX_PRICE = float(os.getenv("YAHOO_MAX_PRICE", "500"))
MIN_VOLUME = int(os.getenv("YAHOO_MIN_VOLUME", "500000"))
MIN_MARKET_CAP = int(os.getenv("YAHOO_MIN_MARKET_CAP", "100000000"))

FALLBACK = [
    "NVDA", "AMD", "SMCI", "AVGO", "ARM", "PLTR", "TSLA", "META", "MSFT", "GOOGL",
    "AMZN", "AAPL", "COIN", "MARA", "RIOT", "CLSK", "IREN", "IONQ", "RGTI", "QBTS",
    "QUBT", "SOUN", "AI", "BBAI", "OKLO", "SMR", "CEG", "VST", "SOXX", "QQQ",
]

STRONG_CATALYST_TYPES = {
    "earnings_beat",
    "raised_guidance",
    "upcoming_earnings",
    "major_contract",
    "buyback",
    "analyst_upgrade",
    "price_target_raise",
    "ipo_momentum",
    "acquisition",
    "regulatory_approval",
    "product_launch",
    "strategic_partnership",
    "sector_catalyst",
    "premarket_news_move",
    "macro_catalyst",
    "financial_strength",
    "technical_gainer",
}

SYSTEM_PROMPT = """
You are a premarket institutional universe-selection engine.

You receive Yahoo Finance day-gainer data plus company financial details.
Select EXACTLY 30 liquid US stocks or ETFs for today's monitoring universe.

Prioritize:
- largest positive percent gainers
- high volume and strong liquidity
- meaningful market cap
- strong revenue growth or earnings growth
- analyst upside or favorable recommendation
- strong sector/industry catalyst
- reasonable valuation where applicable
- avoid illiquid low-quality pumps when possible

Each reason must cite at least one concrete data point from the provided data:
- percent gain
- volume
- market cap
- sector/industry
- revenue growth
- earnings growth
- P/E
- analyst target
- recommendation
- business description
- financial strength

Return ONLY valid JSON.

Schema:
{
  "market_tone": "bullish | bearish | mixed | risk_off | risk_on",
  "summary": "One concise sentence describing what kind of trading day this may be.",
  "selections": [
    {
      "symbol": "NVDA",
      "catalyst_type": "technical_gainer | financial_strength | analyst_upgrade | price_target_raise | upcoming_earnings | earnings_beat | raised_guidance | major_contract | buyback | ipo_momentum | acquisition | regulatory_approval | product_launch | strategic_partnership | sector_catalyst | premarket_news_move | macro_catalyst",
      "reason": "Specific one-sentence explanation using concrete data from the supplied Yahoo/financial details.",
      "confidence": 0.0
    }
  ]
}

Rules:
- EXACTLY 30 selections.
- Use "symbol", not "ticker".
- Symbols must be uppercase.
- No duplicate symbols.
- Every selection must have catalyst_type.
- Confidence must be between 0 and 1.
- Do not invent facts not present in the supplied data.
- No markdown.
- No extra commentary.
""".strip()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_symbol(value: Any) -> str:
    return str(value or "").upper().strip().replace(".", "-")


def is_valid_symbol(symbol: str) -> bool:
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


def pct(value: Any) -> float:
    return to_float(value, 0.0)


def clean_reason(reason: Any) -> str:
    text = str(reason or "").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:400] or "Selected from Yahoo gainers and company financial data."


def normalize_catalyst_type(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return raw if raw in STRONG_CATALYST_TYPES else "technical_gainer"


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
    symbol = normalize_symbol(row.get("symbol"))
    price = to_float(row.get("price"))
    volume = to_int(row.get("volume"))
    market_cap = to_int(row.get("market_cap"))

    if not is_valid_symbol(symbol):
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
            ticker = yf.Ticker(symbol)
            info = ticker.get_info() or {}

            enriched[symbol] = {
                "company_name": info.get("shortName") or info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
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
                "business_summary": str(info.get("longBusinessSummary") or "")[:700],
            }

        except Exception as exc:
            print(f"[yahoo] enrichment failed for {symbol}: {exc}")
            enriched[symbol] = {}

    return enriched


def calc_discovery_score(row: dict[str, Any]) -> float:
    change_pct = pct(row.get("change_pct"))
    volume = max(to_float(row.get("volume")), 1.0)
    avg_volume = max(to_float(row.get("avg_volume")), 1.0)
    market_cap = max(to_float(row.get("market_cap")), 1.0)

    rel_volume = volume / avg_volume if avg_volume > 0 else 1.0

    score = 0.0
    score += min(max(change_pct, 0.0), 50.0) * 2.0
    score += min(rel_volume, 10.0) * 8.0

    if market_cap >= 10_000_000_000:
        score += 15
    elif market_cap >= 1_000_000_000:
        score += 10
    elif market_cap >= 100_000_000:
        score += 4

    return round(score, 4)


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

            volume = to_float(r.get("volume"))
            avg_volume = to_float(r.get("avg_volume"))
            rel_volume = volume / avg_volume if avg_volume > 0 else None

            row = {
                "symbol": symbol,
                "name": r.get("name"),
                "price": r.get("price"),
                "change": r.get("change"),
                "change_pct": r.get("change_pct"),
                "volume": r.get("volume"),
                "avg_volume": r.get("avg_volume"),
                "relative_volume": rel_volume,
                "market_cap": r.get("market_cap"),
                "pe_ratio": r.get("pe_ratio"),
                "exchange": r.get("exchange"),
                "quote_type": r.get("quote_type"),
                "discovery_score": calc_discovery_score(r),
                "source": "yahoo_day_gainers",
            }
            rows.append(row)

    rows = sorted(rows, key=lambda x: to_float(x.get("discovery_score")), reverse=True)
    rows = rows[:max(YAHOO_GAINERS_COUNT, TARGET_COUNT)]

    symbols = [r["symbol"] for r in rows if is_valid_symbol(r.get("symbol"))]
    enriched = enrich_company_details(symbols[:80])

    candidates: list[dict[str, Any]] = []

    for row in rows:
        symbol = row["symbol"]
        details = enriched.get(symbol, {})

        merged = {
            **row,
            **details,
        }

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
        },
        "candidates": candidates[:100],
    }

    DISCOVERY_PATH.write_text(json.dumps(discovery, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {DISCOVERY_PATH}")
    print(f"[discovery] yahoo_gainers candidates={len(candidates)}")

    return discovery


def compact_candidates_for_llm(discovery: dict[str, Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []

    for row in discovery.get("candidates", [])[:100]:
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
                "beta": row.get("beta"),
                "short_ratio": row.get("short_ratio"),
                "target_mean_price": row.get("target_mean_price"),
                "recommendation": row.get("recommendation"),
                "analyst_count": row.get("number_of_analyst_opinions"),
                "discovery_score": row.get("discovery_score"),
                "business_summary": row.get("business_summary"),
            }
        )

    return compact


def build_prompt(discovery: dict[str, Any]) -> str:
    candidates = compact_candidates_for_llm(discovery)

    return f"""
Today is {datetime.now().strftime("%Y-%m-%d")}.

Use the Yahoo Finance day-gainers and company financial data below.
Select the 30 best symbols for today's monitoring universe.

You must justify each selection using concrete data such as:
- percent gain
- volume or relative volume
- market cap
- revenue growth
- earnings growth
- profit margins
- P/E or forward P/E
- sector/industry
- analyst target or recommendation
- business summary

Yahoo gainers and financial details:
{json.dumps(candidates, indent=2)}

Return the required JSON only.
""".strip()


def call_llm(discovery: dict[str, Any]) -> dict[str, Any]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.10,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(discovery)},
        ],
    )

    return json.loads(response.choices[0].message.content.strip())


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
        catalyst_type = normalize_catalyst_type(row.get("catalyst_type"))
        confidence = max(0.0, min(1.0, to_float(row.get("confidence"), 0.0)))

        seen.add(symbol)

        selections.append(
            {
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
                "symbol": symbol,
                "catalyst_type": "technical_gainer",
                "reason": (
                    f"Added from Yahoo gainers because it showed a positive move of "
                    f"{row.get('change_pct')}% with volume of {row.get('volume')}."
                ),
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
                    "symbol": row["symbol"],
                    "catalyst_type": "technical_gainer",
                    "reason": (
                        f"Selected from Yahoo gainers with a {row.get('change_pct')}% move, "
                        f"volume of {row.get('volume')}, and market cap of {row.get('market_cap')}."
                    ),
                    "confidence": 0.50,
                }
                for row in discovery.get("candidates", [])[:TARGET_COUNT]
            ],
        }

    payload = normalize_payload(raw, discovery)
    write_outputs(payload)
    return payload


if __name__ == "__main__":
    build_sentiment_universe()