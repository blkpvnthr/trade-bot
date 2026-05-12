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

MAX_DISCOVERY_CANDIDATES = 100
NEWS_LOOKBACK_HOURS = int(os.getenv("NEWS_LOOKBACK_HOURS", "36"))

MIN_LLM_CONFIDENCE = float(os.getenv("MIN_LLM_CATALYST_CONFIDENCE", "0.55"))
MIN_DISCOVERY_CATALYST_SCORE = int(os.getenv("MIN_DISCOVERY_CATALYST_SCORE", "4"))

FALLBACK = [
    "NVDA", "AMD", "SMCI", "AVGO", "ARM", "PLTR", "TSLA", "META", "MSFT", "GOOGL",
    "AMZN", "AAPL", "COIN", "MARA", "RIOT", "CLSK", "IREN", "IONQ", "RGTI", "QBTS",
    "QUBT", "SOUN", "AI", "BBAI", "OKLO", "SMR", "CEG", "VST", "SOXX", "QQQ",
]

STOPWORDS = {
    "A", "AN", "AND", "ARE", "AS", "AT", "BE", "BY", "CEO", "CFO", "CPI",
    "DATA", "DO", "ETF", "EPS", "EVERY", "FED", "FOR", "FROM", "GDP",
    "HAS", "HIGH", "IN", "IS", "IT", "ITS", "NEW", "OF", "ON", "OR",
    "PLUS", "SAYS", "SEC", "SHOWS", "STOCK", "TALKS", "TECH", "THE",
    "THIS", "TO", "UP", "USA", "US", "WAR", "WITH", "OLD", "TOP", "BUY",
    "TRUMP", "IRAN", "CHINA", "OIL", "CLUB", "MARKET", "MARKETS", "SHARES",
}

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
}

WEAK_REASON_PHRASES = [
    "general momentum",
    "positive sentiment",
    "strong interest",
    "retail buzz",
    "hype",
    "mentioned often",
    "ai momentum",
    "sector strength",
    "market strength",
    "popular ticker",
    "trending stock",
    "search interest",
    "news frequency",
]

CATALYST_KEYWORDS = {
    "earnings beat": 14,
    "beats estimates": 14,
    "beat estimates": 14,
    "beats expectations": 14,
    "beat expectations": 14,
    "better-than-expected": 12,
    "strong earnings": 10,
    "earnings surprise": 10,

    "raises guidance": 14,
    "raised guidance": 14,
    "guidance raised": 14,
    "raises outlook": 12,
    "raised outlook": 12,
    "strong guidance": 12,
    "boosts forecast": 10,
    "raises forecast": 10,

    "buyback": 12,
    "stock buyback": 14,
    "share buyback": 14,
    "share repurchase": 14,
    "repurchase authorization": 14,

    "new contract": 12,
    "wins contract": 12,
    "major contract": 14,
    "government contract": 12,
    "defense contract": 12,
    "supply agreement": 10,

    "strategic partnership": 12,
    "partnership": 8,
    "collaboration": 7,

    "analyst upgrade": 12,
    "upgraded": 10,
    "upgrades": 10,
    "initiated at buy": 10,
    "initiates at buy": 10,
    "overweight rating": 8,

    "price target raised": 12,
    "raises price target": 12,
    "price target hike": 10,
    "price target": 6,

    "ipo": 10,
    "initial public offering": 10,
    "recent ipo": 10,
    "debut": 8,

    "fda approval": 14,
    "regulatory approval": 12,
    "approved": 8,
    "clearance": 8,

    "acquisition": 12,
    "merger": 12,
    "takeover": 12,
    "deal to buy": 10,

    "product launch": 10,
    "launches": 8,
    "unveils": 8,
    "new product": 8,

    "premarket gain": 10,
    "premarket surge": 12,
    "surges premarket": 12,
    "jumps premarket": 12,
    "rises premarket": 8,
    "before the bell": 8,

    "positive earnings": 8,
    "earnings expected": 8,
    "reports earnings": 7,
    "scheduled to report": 6,

    "ai": 3,
    "semiconductor": 4,
    "crypto": 3,
    "bitcoin": 3,
    "quantum": 3,
    "nuclear": 3,
    "energy": 2,
}

DISCOVERY_QUERIES = [
    "stocks moving premarket earnings beat raised guidance",
    "stocks with earnings beat today premarket",
    "stocks raising guidance today",
    "stocks analyst upgrades price target raised today",
    "stocks buyback authorization today",
    "stocks major contract award today",
    "stocks strategic partnership product launch today",
    "recent IPO stock premarket surge catalyst",
    "positive expected earnings stocks today",
    "AI stocks earnings beat guidance contract today",
    "semiconductor stocks earnings guidance analyst upgrade today",
    "crypto stocks bitcoin rally premarket news catalyst",
    "quantum computing stocks contract partnership news catalyst",
    "nuclear energy stocks contract approval news catalyst",
]


SYSTEM_PROMPT = """
You are a premarket institutional catalyst-selection engine.

You receive discovered ticker candidates from news/search-style queries.
Select EXACTLY 30 liquid US stocks or ETFs with the strongest hard catalysts
for today's trading session.

Hard catalyst examples:
- earnings beat
- raised guidance
- positive upcoming earnings expectations
- major new contract
- acquisition or merger
- big IPO or recent IPO momentum
- stock buyback authorization
- analyst upgrade
- price target raise
- FDA approval
- regulatory approval
- major product launch
- strategic partnership
- unusually strong premarket move tied to news
- sector-wide catalyst with direct exposure
- major macro catalyst directly benefiting the symbol

Weak reasons are NOT enough:
- general hype
- vague AI interest
- vague momentum
- mentioned in news
- popular ticker
- retail buzz without a specific catalyst
- broad market strength alone

Return ONLY valid JSON.

Schema:
{
  "market_tone": "bullish | bearish | mixed | risk_off | risk_on",
  "summary": "One concise sentence describing what kind of trading day this may be.",
  "selections": [
    {
      "symbol": "NVDA",
      "catalyst_type": "earnings_beat | raised_guidance | upcoming_earnings | major_contract | buyback | analyst_upgrade | price_target_raise | ipo_momentum | acquisition | regulatory_approval | product_launch | strategic_partnership | sector_catalyst | premarket_news_move | macro_catalyst",
      "reason": "Specific one-sentence explanation of the catalyst and why it matters today.",
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
- Every reason must identify a specific catalyst.
- Do not select a symbol just because it is hyped.
- Do not select a symbol just because it is mentioned often.
- Do not use vague reasons like AI momentum unless tied to a specific event.
- Confidence must be between 0 and 1.
- Prefer individual stocks with hard catalysts over ETFs.
- ETFs are allowed only as sector_catalyst or macro_catalyst.
- If fewer than 30 hard-catalyst names exist, fill with best liquid sector proxies and label catalyst_type as sector_catalyst.
- No markdown.
- No extra commentary.
""".strip()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_symbol(value: Any) -> str:
    return str(value).upper().strip().replace(".", "-")


def is_valid_symbol(symbol: str) -> bool:
    symbol = normalize_symbol(symbol)

    if symbol in STOPWORDS:
        return False

    if len(symbol) < 2 or len(symbol) > 6:
        return False

    return bool(re.fullmatch(r"[A-Z][A-Z0-9-]{1,5}", symbol))


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def clean_reason(reason: Any) -> str:
    text = str(reason or "").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:350] or "Selected by the premarket LLM catalyst engine."


def normalize_catalyst_type(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in STRONG_CATALYST_TYPES:
        return raw
    return ""


def reason_is_too_weak(reason: str) -> bool:
    lower = reason.lower()
    return any(phrase in lower for phrase in WEAK_REASON_PHRASES)


def extract_symbols(text: str) -> list[str]:
    if not text:
        return []

    upper = text.upper()

    patterns = [
        r"\$([A-Z][A-Z0-9.-]{0,9})\b",
        r"\b(?:NASDAQ|NYSE|AMEX):\s*([A-Z][A-Z0-9.-]{0,9})\b",
        r"\(([A-Z][A-Z0-9.-]{1,9})\)",
    ]

    found: list[str] = []

    for pattern in patterns:
        found.extend(re.findall(pattern, upper))

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


def infer_catalyst_type_from_text(text: str) -> str:
    lower = text.lower()

    rules = [
        ("earnings_beat", ["earnings beat", "beats estimates", "beat estimates", "beats expectations", "beat expectations", "better-than-expected"]),
        ("raised_guidance", ["raises guidance", "raised guidance", "guidance raised", "raises outlook", "raised outlook", "raises forecast"]),
        ("buyback", ["buyback", "stock buyback", "share buyback", "share repurchase", "repurchase authorization"]),
        ("major_contract", ["new contract", "wins contract", "major contract", "government contract", "defense contract", "supply agreement"]),
        ("strategic_partnership", ["strategic partnership", "partnership", "collaboration"]),
        ("analyst_upgrade", ["analyst upgrade", "upgraded", "upgrades", "initiated at buy", "initiates at buy"]),
        ("price_target_raise", ["price target raised", "raises price target", "price target hike"]),
        ("ipo_momentum", ["ipo", "initial public offering", "recent ipo", "debut"]),
        ("regulatory_approval", ["fda approval", "regulatory approval", "approved", "clearance"]),
        ("acquisition", ["acquisition", "merger", "takeover", "deal to buy"]),
        ("product_launch", ["product launch", "launches", "unveils", "new product"]),
        ("premarket_news_move", ["premarket surge", "surges premarket", "jumps premarket", "rises premarket", "premarket gain"]),
        ("upcoming_earnings", ["positive earnings", "earnings expected", "reports earnings", "scheduled to report"]),
    ]

    for catalyst_type, phrases in rules:
        if any(p in lower for p in phrases):
            return catalyst_type

    if any(p in lower for p in ["semiconductor", "bitcoin", "crypto", "ai", "quantum", "nuclear", "energy"]):
        return "sector_catalyst"

    return ""


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
    catalyst_types_by_symbol: dict[str, Counter[str]] = defaultdict(Counter)

    for article in articles:
        title = str(article.get("title") or "")
        desc = str(article.get("description") or "")
        query = str(article.get("discovery_query") or "")
        url = str(article.get("url") or "")

        text = f"{title}. {desc}"
        symbols = extract_symbols(text)
        score = catalyst_score(text)
        inferred_type = infer_catalyst_type_from_text(text)

        for symbol in symbols:
            mention_counts[symbol] += 1
            catalyst_counts[symbol] += score
            query_counts[symbol].add(query)

            if inferred_type:
                catalyst_types_by_symbol[symbol][inferred_type] += 1

            if title and title not in headlines_by_symbol[symbol]:
                headlines_by_symbol[symbol].append(title[:240])

            if url and url not in urls_by_symbol[symbol]:
                urls_by_symbol[symbol].append(url)

    rows = []

    for symbol, mentions in mention_counts.items():
        catalyst = catalyst_counts[symbol]
        query_diversity = len(query_counts[symbol])
        best_catalyst_type = ""

        if catalyst_types_by_symbol[symbol]:
            best_catalyst_type = catalyst_types_by_symbol[symbol].most_common(1)[0][0]

        hard_catalyst_bonus = 12 if best_catalyst_type and best_catalyst_type != "sector_catalyst" else 0
        hype_score = mentions * 2 + catalyst + query_diversity * 2 + hard_catalyst_bonus

        rows.append(
            {
                "symbol": symbol,
                "mention_count": int(mentions),
                "catalyst_score": int(catalyst),
                "query_diversity": int(query_diversity),
                "hype_score": int(hype_score),
                "inferred_catalyst_type": best_catalyst_type,
                "headlines": headlines_by_symbol[symbol][:5],
                "urls": urls_by_symbol[symbol][:3],
            }
        )

    rows = [
        r for r in rows
        if r["catalyst_score"] >= MIN_DISCOVERY_CATALYST_SCORE or r["inferred_catalyst_type"]
    ]

    rows = sorted(
        rows,
        key=lambda r: (
            r["inferred_catalyst_type"] not in ("", "sector_catalyst"),
            r["catalyst_score"],
            r["hype_score"],
            r["mention_count"],
        ),
        reverse=True,
    )

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
                        "inferred_catalyst_type": "sector_catalyst",
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
The discovery engine ranks symbols by hard catalyst keywords, inferred catalyst type,
mention frequency, query diversity, and hype/search-style relevance.

Select the 30 best symbols for today's premarket monitoring universe.

You must prefer candidates with:
- earnings beats
- raised guidance
- positive upcoming earnings expectations
- major contracts
- stock buybacks
- analyst upgrades
- price target raises
- IPO momentum
- acquisitions
- regulatory approvals
- product launches
- strategic partnerships
- specific premarket news moves

Discovered candidates:
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

        if not is_valid_symbol(symbol) or symbol in seen:
            continue

        d = discovery_by_symbol.get(symbol, {})
        reason = clean_reason(row.get("reason"))
        catalyst_type = normalize_catalyst_type(row.get("catalyst_type")) or d.get("inferred_catalyst_type", "")
        confidence = max(0.0, min(1.0, to_float(row.get("confidence"), 0.0)))

        if catalyst_type not in STRONG_CATALYST_TYPES:
            continue

        if confidence < MIN_LLM_CONFIDENCE:
            continue

        if reason_is_too_weak(reason):
            continue

        seen.add(symbol)

        selections.append(
            {
                "symbol": symbol,
                "catalyst_type": catalyst_type,
                "reason": reason,
                "confidence": confidence,
                "mention_count": d.get("mention_count", 0),
                "catalyst_score": d.get("catalyst_score", 0),
                "query_diversity": d.get("query_diversity", 0),
                "hype_score": d.get("hype_score", 0),
                "inferred_catalyst_type": d.get("inferred_catalyst_type", ""),
                "headlines": d.get("headlines", [])[:3],
            }
        )

        if len(selections) >= TARGET_COUNT:
            break

    for row in discovery.get("candidates", []):
        if len(selections) >= TARGET_COUNT:
            break

        symbol = normalize_symbol(row.get("symbol", ""))

        if not is_valid_symbol(symbol) or symbol in seen:
            continue

        catalyst_type = row.get("inferred_catalyst_type") or "sector_catalyst"

        if catalyst_type not in STRONG_CATALYST_TYPES:
            catalyst_type = "sector_catalyst"

        seen.add(symbol)

        headlines = row.get("headlines", [])[:3]
        reason = (
            f"Added from discovery ranking because recent headlines show a {catalyst_type.replace('_', ' ')} catalyst: "
            f"{headlines[0]}"
            if headlines
            else "Added from discovery ranking as a liquid sector proxy because fewer than 30 hard-catalyst names were available."
        )

        selections.append(
            {
                "symbol": symbol,
                "catalyst_type": catalyst_type,
                "reason": clean_reason(reason),
                "confidence": 0.55,
                "mention_count": row.get("mention_count", 0),
                "catalyst_score": row.get("catalyst_score", 0),
                "query_diversity": row.get("query_diversity", 0),
                "hype_score": row.get("hype_score", 0),
                "inferred_catalyst_type": row.get("inferred_catalyst_type", ""),
                "headlines": headlines,
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
                    "catalyst_type": "sector_catalyst",
                    "reason": "Fallback liquid sector or leadership proxy added because fewer than 30 verified hard-catalyst symbols were available.",
                    "confidence": 0.40,
                    "mention_count": 0,
                    "catalyst_score": 0,
                    "query_diversity": 0,
                    "hype_score": 0,
                    "inferred_catalyst_type": "sector_catalyst",
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
            "summary": "LLM unavailable; using discovery-ranked and fallback high-liquidity catalyst universe.",
            "selections": [
                {
                    "symbol": row["symbol"],
                    "catalyst_type": row.get("inferred_catalyst_type") or "sector_catalyst",
                    "reason": (
                        f"Selected from discovery ranking because recent headlines indicate a "
                        f"{(row.get('inferred_catalyst_type') or 'sector_catalyst').replace('_', ' ')} catalyst."
                    ),
                    "confidence": 0.55,
                }
                for row in discovery.get("candidates", [])[:TARGET_COUNT]
            ],
        }

    payload = normalize_payload(raw, discovery)
    write_outputs(payload)
    return payload


if __name__ == "__main__":
    build_sentiment_universe()