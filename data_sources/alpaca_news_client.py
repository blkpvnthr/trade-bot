#!/usr/bin/env python3
"""
data_sources/alpaca_news_client.py — Alpaca news source, schema-compatible with NewsSentimentClient.

Drop-in replacement for the Alpha Vantage NEWS_SENTIMENT path. Alpaca news is RAW TEXT ONLY
(no sentiment, no relevance, no topics), so this client DERIVES the three missing fields and emits
the EXACT 14-key schema that news_sentiment_client.normalize_news_item produces, so every downstream
consumer (live_feature_transformer → news_feature_builder → RL features) works unchanged.

The three derivations:
  * sentiment  → FinBERT on "headline. summary" (data_sources/finbert_scorer); ticker == overall
                 (Alpaca has no per-ticker scores; consumers already fall back overall→ticker).
  * relevance  → heuristic from the article's symbol set (sole-symbol → 1.0; many → diluted; +0.3 if
                 the ticker appears in the headline; 0.0 if the symbol isn't mentioned).
  * topics     → keyword lexicon mapped onto the AV topic vocabulary so the catalyst flags
                 (has_earnings_news / has_merger_news / …) in news_feature_builder still fire.

Advisory-safe: every method returns a list and NEVER raises; on any failure → empty/neutral.
Credentials: APCA_API_KEY_ID / APCA_API_SECRET_KEY (same as alpaca_stream_router / alpaca_executor).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from data_sources.alpha_vantage_client import PROJECT_ROOT, now_utc
from data_sources import finbert_scorer

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_STORE = PROJECT_ROOT / "feature_store" / "news"
_PER_REQUEST_LIMIT = 50   # Alpaca news API max page size

# Keyword → AV topic vocabulary (lowercase_underscore). Only topics the catalyst flags key on
# need to be precise; the rest add coarse context. Mirrors SUPPORTED_TOPICS in news_sentiment_client.
_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "earnings": ("earnings", "eps", "quarterly results", "guidance", "beats estimate", "misses estimate",
                 "revenue", "profit", "results", "outlook"),
    "mergers_and_acquisitions": ("merger", "acquire", "acquisition", "takeover", "buyout", "to buy",
                                 "deal to acquire"),
    "ipo": ("ipo", "public offering", "goes public", "market debut"),
    "economy_monetary": ("federal reserve", "the fed", "interest rate", "rate hike", "rate cut", "fomc",
                         "monetary policy", "powell"),
    "economy_fiscal": ("fiscal", "tariff", "budget", "stimulus", "tax bill", "government spending"),
    "economy_macro": ("gdp", "inflation", "cpi", "unemployment", "jobs report", "recession",
                      "economic growth"),
    "financial_markets": ("stock market", "s&p 500", "nasdaq", "dow", "selloff", "rally", "index"),
    "technology": ("ai ", "artificial intelligence", "semiconductor", "chip", "software", "cloud", "tech"),
    "life_sciences": ("fda", "drug", "clinical trial", "biotech", "pharma", "therapy"),
    "energy_transportation": ("oil", "crude", "natural gas", "energy", "airline", "shipping", "ev "),
    "finance": ("bank", "lending", "loan", "credit", "mortgage rate"),
    "real_estate": ("real estate", "reit", "housing", "home sales"),
    "retail_wholesale": ("retail", "same-store", "consumer spending", "store sales"),
    "manufacturing": ("factory", "manufacturing", "industrial production"),
    "blockchain": ("crypto", "bitcoin", "ethereum", "blockchain"),
}


def infer_topics(text: str) -> list[str]:
    """Infer AV-vocabulary topics from headline+summary via keyword match (approximation)."""
    t = (text or "").lower()
    return [topic for topic, kws in _TOPIC_KEYWORDS.items() if any(k in t for k in kws)]


def _relevance(symbol: str, news_symbols: list[str], headline: str) -> float:
    """Heuristic relevance in [0,1] from the article's symbol set + headline mention."""
    sym = (symbol or "").upper()
    syms = [str(s).upper() for s in (news_symbols or [])]
    if sym not in syms:
        return 0.0
    base = 1.0 if len(syms) <= 1 else 1.0 / len(syms)
    if sym in (headline or "").upper():
        base = min(1.0, base + 0.3)
    return round(base, 4)


def _iso_utc(dt: Any) -> Optional[str]:
    if dt is None:
        return None
    try:
        if isinstance(dt, str):
            return dt
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return None


def normalize_alpaca_news_item(news: Any, symbol: str, sentiment: dict[str, Any]) -> dict[str, Any]:
    """Map an Alpaca ``News`` object + a precomputed FinBERT result to the 14-key contract schema."""
    headline = str(getattr(news, "headline", "") or "")
    summary = str(getattr(news, "summary", "") or "")
    nsyms = list(getattr(news, "symbols", None) or [])
    author = str(getattr(news, "author", "") or "")
    score = sentiment.get("score")
    label = sentiment.get("label") or "Neutral"
    topics = infer_topics(headline + ". " + summary)
    return {
        "symbol": (symbol or "").upper(),
        "published_at": _iso_utc(getattr(news, "created_at", None)),
        "title": headline[:500],
        "summary": summary[:2000],
        "source": str(getattr(news, "source", "") or ""),
        "authors": [author] if author else [],
        "topics": topics,
        "url": str(getattr(news, "url", "") or ""),
        "overall_sentiment_score": score,
        "overall_sentiment_label": label,
        "ticker_sentiment_score": score,    # no per-ticker score from Alpaca → mirror overall
        "ticker_sentiment_label": label,
        "relevance_score": _relevance(symbol, nsyms, headline),
        "time_fetched": now_utc().isoformat(),
    }


def _score_items(items: list[Any]) -> list[dict[str, Any]]:
    """Batch-FinBERT-score a list of Alpaca News objects (headline + summary)."""
    texts = [
        (str(getattr(n, "headline", "") or "") + ". " + str(getattr(n, "summary", "") or "")).strip()
        for n in items
    ]
    return finbert_scorer.score_batch(texts)


class AlpacaNewsClient:
    """Advisory-safe Alpaca news client. Same method surface as NewsSentimentClient; never raises."""

    def __init__(self, news_client: Any = None, feature_store_dir: Optional[Path] = None) -> None:
        self._client = news_client
        self.feature_store_dir = Path(feature_store_dir) if feature_store_dir is not None else DEFAULT_FEATURE_STORE
        try:
            self.feature_store_dir.mkdir(parents=True, exist_ok=True)
        except Exception:  # pragma: no cover
            pass

    @property
    def api_key(self) -> str:
        return os.getenv("APCA_API_KEY_ID", "")

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from alpaca.data.historical.news import NewsClient
            key = os.getenv("APCA_API_KEY_ID", "")
            secret = os.getenv("APCA_API_SECRET_KEY", "")
            self._client = NewsClient(api_key=key, secret_key=secret)
        except Exception as e:  # pragma: no cover
            logger.warning("Alpaca NewsClient init failed (%s); news disabled", str(e)[:160])
            self._client = False
        return self._client

    def _fetch(self, symbol: str, start: Optional[datetime], end: Optional[datetime],
               limit: int, sort: str = "desc") -> list[Any]:
        """Return a list of Alpaca News objects for one symbol over [start, end]; [] on failure."""
        client = self._get_client()
        if not client:
            return []
        try:
            from alpaca.data.requests import NewsRequest
            out: list[Any] = []
            page_token: Optional[str] = None
            while True:
                req = NewsRequest(symbols=symbol.upper(), start=start, end=end,
                                  limit=min(limit, _PER_REQUEST_LIMIT), sort=sort,
                                  include_content=False, exclude_contentless=True,
                                  page_token=page_token)
                res = client.get_news(req)
                data = getattr(res, "data", None) or {}
                for arts in data.values():
                    out.extend(arts)
                page_token = getattr(res, "next_page_token", None)
                if not page_token or len(out) >= limit:
                    break
            # de-dupe by id then url
            seen: set = set()
            uniq: list[Any] = []
            for n in out:
                k = getattr(n, "id", None) or getattr(n, "url", None)
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(n)
            return uniq[:limit] if limit else uniq
        except Exception as e:
            logger.warning("Alpaca news fetch failed for %s (%s)", symbol, str(e)[:160])
            return []

    def get_symbol_news(self, symbol: str, topics: Optional[Iterable[str]] = None,
                        limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Latest normalized news for a symbol (most recent first). `topics` accepted, ignored for
        fetch (Alpaca has no topic filter); topics are derived per item instead."""
        symbol = (symbol or "").upper().strip()
        n = int(limit if limit is not None else 50)
        items = self._fetch(symbol, start=None, end=None, limit=n, sort="desc")
        sents = _score_items(items)
        rows = [normalize_alpaca_news_item(it, symbol, s) for it, s in zip(items, sents)]
        rows.sort(key=lambda r: r.get("published_at") or "", reverse=True)
        return rows

    def window_news(self, symbol: str, start: datetime, end: datetime,
                    limit: int = 1000) -> list[dict[str, Any]]:
        """Normalized news for ``symbol`` over [start, end] (PIT-safe when end <= as-of). Never raises."""
        symbol = (symbol or "").upper().strip()
        items = self._fetch(symbol, start=start, end=end, limit=limit, sort="asc")
        sents = _score_items(items)
        return [normalize_alpaca_news_item(it, symbol, s) for it, s in zip(items, sents)]

    def backfill_symbol(self, symbol: str, days: int = 90, as_of: Optional[datetime] = None,
                        persist: bool = True, **_ignored: Any) -> list[dict[str, Any]]:
        """Historical news for ``symbol`` over [as_of-days, as_of], persisted to
        feature_store/news/<SYMBOL>.jsonl. Pure when as_of is given."""
        symbol = (symbol or "").upper().strip()
        end = as_of or now_utc()
        start = end - timedelta(days=days)
        items = self._fetch(symbol, start=start, end=end, limit=10000, sort="asc")
        sents = _score_items(items)
        rows = [normalize_alpaca_news_item(it, symbol, s) for it, s in zip(items, sents)]
        rows.sort(key=lambda r: r.get("published_at") or "")
        if persist and rows:
            self._persist_symbol(symbol, rows)
        return rows

    def _persist_symbol(self, symbol: str, rows: list[dict[str, Any]]) -> None:
        try:
            path = self.feature_store_dir / f"{symbol.upper()}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, default=str) + "\n")
        except Exception:  # pragma: no cover
            logger.warning("alpaca news persist failed for %s", symbol)

    def load_symbol_history(self, symbol: str) -> list[dict[str, Any]]:
        path = self.feature_store_dir / f"{symbol.upper()}.jsonl"
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        except Exception:  # pragma: no cover
            return []
        return out


_DEFAULT_ALPACA_NEWS_CLIENT: Optional[AlpacaNewsClient] = None


def get_default_alpaca_news_client() -> AlpacaNewsClient:
    global _DEFAULT_ALPACA_NEWS_CLIENT
    if _DEFAULT_ALPACA_NEWS_CLIENT is None:
        _DEFAULT_ALPACA_NEWS_CLIENT = AlpacaNewsClient()
    return _DEFAULT_ALPACA_NEWS_CLIENT


if __name__ == "__main__":  # manual smoke test (needs APCA creds + FINBERT_ENABLED for real scores)
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    rows = get_default_alpaca_news_client().get_symbol_news(sym, limit=5)
    print(f"{sym}: {len(rows)} rows")
    for r in rows[:5]:
        print(json.dumps({k: r[k] for k in ("published_at", "title", "ticker_sentiment_score",
                                            "ticker_sentiment_label", "relevance_score", "topics")}, default=str))
