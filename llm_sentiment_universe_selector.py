#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import requests
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

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

MAX_DISCOVERY_CANDIDATES = 80
NEWS_LOOKBACK_HOURS = int(os.getenv("NEWS_LOOKBACK_HOURS", "36"))

# Only fallback if APIs fail or return too few valid candidates.
FALLBACK = [
    "NVDA", "AMD", "SMCI", "AVGO", "ARM", "PLTR", "TSLA", "META", "MSFT", "GOOGL",
    "AMZN", "AAPL", "COIN", "MARA", "RIOT", "CLSK", "IREN", "IONQ", "RGTI", "QBTS",
    "QUBT", "SOUN", "AI", "BBAI", "OKLO", "SMR", "CEG", "VST", "SOXX", "QQQ",
]

STOPWORDS = {
    "CEO", "CFO", "SEC", "USA", "US", "AI", "IPO", "ETF", "EPS", "GDP", "CPI",
    "FOMC", "Fed", "THE", "FOR", "AND", "ARE", "NEW", "OLD", "TOP", "BUY",
}

CATALYST_KEYWORDS = {
    "earnings": 5,
    "beat": 5,
    "raises guidance": 6,
    "guidance": 4,
    "upgrade": 5,
    "upgraded": 5,
    "price target": 4,
    "contract": 4,
    "partnership": 4,
    "launch": 3,
    "approval": 4,
    "breakout": 4,
    "surges": 4,
    "rallies": 4,
    "premarket": 5,
    "before the bell": 5,
    "analyst": 3,
    "revenue": 3,
    "profit": 3,
    "ai": 4,
    "semiconductor": 4,
    "crypto": 3,
    "bitcoin": 3,
    "quantum": 3,
    "nuclear": 3,
    "energy": 2,
}

DISCOVERY_QUERIES = [
    "premarket stock movers today",
    "stocks moving premarket earnings beat guidance",
    "most searched stocks today retail traders",
    "trending stocks today news catalyst",
    "AI stocks premarket news earnings analyst upgrade",
    "semiconductor stocks premarket news catalyst",
    "crypto stocks premarket bitcoin rally news",
    "quantum computing stocks news catalyst",
    "nuclear energy stocks news catalyst",
    "analyst upgrades stocks today",
    "positive earnings expected stocks news",
    "hyped stocks retail traders today",
]


SYSTEM_PROMPT = """
You are a premarket institutional sentiment, catalyst, and hype discovery engine.

You receive discovered ticker candidates from news/search-style queries.
Select EXACTLY 30 liquid US stocks or ETFs that are most relevant for today's trading session.

Prioritize:
- tickers repeatedly mentioned across searches/news
- catalyst news
- positive expected earnings or earnings reactions
- analyst upgrades or raised price targets
- premarket momentum
- retail hype / highly searched / highly discussed names
- liquidity and tradability
- likely intraday continuation potential

Return ONLY valid JSON.

Schema:
{
  "market_tone": "bullish | bearish | mixed | risk_off | risk_on",
  "summary": "One concise sentence describing what kind of trading day this may be.",
  "selections": [
    {
      "symbol": "NVDA",
      "reason": "One concise sentence explaining why this symbol was selected."
    }
  ]
}

Rules:
- EXACTLY 30 selections.
- Use "symbol", not "ticker".
- Symbols must be uppercase.
- No duplicate symbols.
- Reasons must reference the observed catalyst, hype, search interest, news frequency, or earnings/news angle when possible.
- No markdown.
- No extra commentary.
""".strip()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_symbol(value: Any) -> str:
    return str(value).upper().strip().replace(".", "-")


def is_valid_symbol(symbol: str) -> bool:
    if symbol in STOPWORDS:
        return False
    return bool(re.fullmatch(r"[A-Z][A-Z0-9-]{0,9}", symbol))


def clean_reason(reason: Any) -> str:
    text = str(reason or "").replace("\n", " ").strip()
    return text[:300] or "Selected by the premarket LLM sentiment universe engine."


def extract_symbols(text: str) -> list[str]:
    """
    Extracts likely ticker symbols from text.

    Handles:
      $NVDA
      (NVDA)
      NASDAQ:NVDA
      uppercase ticker mentions
    """
    if not text:
        return []

    found: list[str] = []

    patterns = [
        r"\$([A-Z][A-Z0-9.-]{0,9})\b",
        r"\(([A-Z][A-Z0-9.-]{0,9})\)",
        r"\b(?:NASDAQ|NYSE|AMEX):\s*([A-Z][A-Z0-9.-]{0,9})\b",
    ]

    for pat in patterns:
        found.extend(re.findall(pat, text.upper()))

    # Conservative uppercase fallback.
    found.extend(re.findall(r"\b[A-Z]{2,5}\b", text.upper()))

    clean: list[str] = []
    seen: set[str] = set()

    for raw in found:
        sym = normalize_symbol(raw)
        if is_valid_symbol(sym) and sym not in seen:
            seen.add(sym)
            clean.append(sym)

    return clean


def catalyst_score(text: str) -> int:
    lower = text.lower()
    score = 0

    for keyword, weight in CATALYST_KEYWORDS.items():
        if keyword in lower:
            score += weight

    return score


def fetch_newsapi_articles() -> list[dict[str, Any]]:
    if not NEWSAPI_KEY:
        return []

    from_time = (datetime.now(timezone.utc) - timedelta(hours=NEWS_LOOKBACK_HOURS)).isoformat()

    articles: list[dict[str, Any]] = []

    for query in DISCOVERY_QUERIES:
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "from": from_time,
                    "sortBy": "publishedAt",
                    "language": "en",
                    "pageSize": 20,
                    "apiKey": NEWSAPI_KEY,
                },
                timeout=20,
            )
            data = resp.json()

            for article in data.get("articles", []):
                article["discovery_query"] = query
                article["source_api"] = "newsapi"
                articles.append(article)

            time.sleep(0.25)

        except Exception as exc:
            print(f"[discovery] NewsAPI query failed: {query}: {exc}")

    return articles


def fetch_finnhub_market_news() -> list[dict[str, Any]]:
    if not FINNHUB_API_KEY:
        return []

    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/news",
            params={
                "category": "general",
                "token": FINNHUB_API_KEY,
            },
            timeout=20,
        )
        data = resp.json()

        articles: list[dict[str, Any]] = []

        for row in data[:100]:
            articles.append(
                {
                    "title": row.get("headline", ""),
                    "description": row.get("summary", ""),
                    "url": row.get("url", ""),
                    "publishedAt": row.get("datetime", ""),
                    "source": {"name": row.get("source", "Finnhub")},
                    "discovery_query": "finnhub_general_market_news",
                    "source_api": "finnhub",
                }
            )

        return articles

    except Exception as exc:
        print(f"[discovery] Finnhub market news failed: {exc}")
        return []


def discover_candidates() -> dict[str, Any]:
    articles = []
    articles.extend(fetch_newsapi_articles())
    articles.extend(fetch_finnhub_market_news())

    mention_counts: Counter[str] = Counter()
    catalyst_counts: Counter[str] = Counter()
    query_counts: dict[str, set[str]] = defaultdict(set)
    headlines_by_symbol: dict[str, list[str]] = defaultdict(list)
    urls_by_symbol: dict[str, list[str]] = defaultdict(list)

    for article in articles:
        title = str(article.get("title") or "")
        desc = str(article.get("description") or "")
        query = str(article.get("discovery_query") or "")
        url = str(article.get("url") or "")

        text = f"{title}. {desc}"
        symbols = extract_symbols(text)
        score = catalyst_score(text)

        for symbol in symbols:
            mention_counts[symbol] += 1
            catalyst_counts[symbol] += score
            query_counts[symbol].add(query)

            if title and title not in headlines_by_symbol[symbol]:
                headlines_by_symbol[symbol].append(title[:240])

            if url and url not in urls_by_symbol[symbol]:
                urls_by_symbol[symbol].append(url)

    rows = []

    for symbol, mentions in mention_counts.items():
        catalyst = catalyst_counts[symbol]
        query_diversity = len(query_counts[symbol])
        hype_score = mentions * 3 + catalyst + query_diversity * 2

        rows.append(
            {
                "symbol": symbol,
                "mention_count": int(mentions),
                "catalyst_score": int(catalyst),
                "query_diversity": int(query_diversity),
                "hype_score": int(hype_score),
                "headlines": headlines_by_symbol[symbol][:5],
                "urls": urls_by_symbol[symbol][:3],
            }
        )

    rows = sorted(rows, key=lambda r: r["hype_score"], reverse=True)

    if len(rows) < TARGET_COUNT:
        existing = {r["symbol"] for r in rows}
        for symbol in FALLBACK:
            if symbol not in existing:
                rows.append(
                    {
                        "symbol": symbol,
                        "mention_count": 0,
                        "catalyst_score": 0,
                        "query_diversity": 0,
                        "hype_score": 0,
                        "headlines": [],
                        "urls": [],
                    }
                )

    discovery = {
        "created_at_utc": now_utc(),
        "article_count": len(articles),
        "candidate_count": len(rows),
        "candidates": rows[:MAX_DISCOVERY_CANDIDATES],
    }

    DISCOVERY_PATH.write_text(json.dumps(discovery, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {DISCOVERY_PATH}")
    print(f"[discovery] articles={len(articles)} candidates={len(rows)}")

    return discovery


def build_prompt(discovery: dict[str, Any]) -> str:
    candidates = discovery.get("candidates", [])[:MAX_DISCOVERY_CANDIDATES]

    return f"""
Today is {datetime.now().strftime("%Y-%m-%d")}.

Use the discovered candidate data below.
The discovery engine ranks symbols by mention frequency, catalyst keywords,
query diversity, and hype/search-style news relevance.

Select the 30 best symbols for today's premarket monitoring universe.

Discovered candidates:
{json.dumps(candidates, indent=2)}

Return the required JSON only.
""".strip()


def call_llm(discovery: dict[str, Any]) -> dict[str, Any]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.15,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(discovery)},
        ],
    )

    content = response.choices[0].message.content.strip()
    return json.loads(content)


def normalize_payload(payload: dict[str, Any], discovery: dict[str, Any]) -> dict[str, Any]:
    raw_selections = payload.get("selections", [])

    discovery_by_symbol = {
        row["symbol"]: row for row in discovery.get("candidates", []) if row.get("symbol")
    }

    selections: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in raw_selections:
        symbol = normalize_symbol(row.get("symbol") or row.get("ticker") or "")

        if not is_valid_symbol(symbol):
            continue

        if symbol in seen:
            continue

        seen.add(symbol)
        d = discovery_by_symbol.get(symbol, {})

        selections.append(
            {
                "symbol": symbol,
                "reason": clean_reason(row.get("reason")),
                "mention_count": d.get("mention_count", 0),
                "catalyst_score": d.get("catalyst_score", 0),
                "query_diversity": d.get("query_diversity", 0),
                "hype_score": d.get("hype_score", 0),
                "headlines": d.get("headlines", [])[:3],
            }
        )

        if len(selections) >= TARGET_COUNT:
            break

    # Fill from discovery ranking first, then fallback.
    for row in discovery.get("candidates", []):
        if len(selections) >= TARGET_COUNT:
            break

        symbol = normalize_symbol(row.get("symbol", ""))

        if not is_valid_symbol(symbol) or symbol in seen:
            continue

        seen.add(symbol)
        selections.append(
            {
                "symbol": symbol,
                "reason": "Added from discovery ranking due to repeated news/search-style mentions and catalyst relevance.",
                "mention_count": row.get("mention_count", 0),
                "catalyst_score": row.get("catalyst_score", 0),
                "query_diversity": row.get("query_diversity", 0),
                "hype_score": row.get("hype_score", 0),
                "headlines": row.get("headlines", [])[:3],
            }
        )

    for symbol in FALLBACK:
        if len(selections) >= TARGET_COUNT:
            break

        symbol = normalize_symbol(symbol)

        if symbol not in seen and is_valid_symbol(symbol):
            seen.add(symbol)
            selections.append(
                {
                    "symbol": symbol,
                    "reason": "Fallback high-liquidity momentum candidate added because discovery returned fewer than 30 valid symbols.",
                    "mention_count": 0,
                    "catalyst_score": 0,
                    "query_diversity": 0,
                    "hype_score": 0,
                    "headlines": [],
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
            "article_count": discovery.get("article_count", 0),
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
        print(f"[llm-universe] LLM failed, using discovery/fallback ranking: {exc}")
        raw = {
            "market_tone": "mixed",
            "summary": "LLM unavailable; using discovery-ranked and fallback high-liquidity momentum universe.",
            "selections": [
                {
                    "symbol": row["symbol"],
                    "reason": "Selected from discovery ranking based on news/search-style mentions, catalyst keywords, and hype relevance.",
                }
                for row in discovery.get("candidates", [])[:TARGET_COUNT]
            ],
        }

    payload = normalize_payload(raw, discovery)
    write_outputs(payload)
    return payload


if __name__ == "__main__":
    build_sentiment_universe()