#!/usr/bin/env python3
"""
data_sources/news_sentiment_client.py — Alpha Vantage NEWS_SENTIMENT client.

Fetches and normalizes Market News & Sentiment for the discovered universe and
serves as a core input to universe discovery, catalyst detection, hidden-alpha
discovery, predictive state compression and the RL options models.

Endpoint
--------
    function=NEWS_SENTIMENT
    Required: function, apikey (env only)
    Optional: tickers, topics, time_from, time_to, sort, limit

Supported topics (validated against this set; unknown topics are dropped):
    blockchain, earnings, ipo, mergers_and_acquisitions, financial_markets,
    economy_fiscal, economy_monetary, economy_macro, energy_transportation,
    finance, life_sciences, manufacturing, real_estate, retail_wholesale,
    technology

Design contract
---------------
    * API key read ONLY from the environment (never hardcoded / logged) — reuses
      AlphaVantageClient for HTTP, retry, throttle, caching and request logging.
    * Advisory-safe: on ANY failure returns an empty list / neutral features and
      logs; NEVER raises. News failures must never interrupt execution.
    * Normalized rows are persisted to feature_store/news/ for the rolling
      historical dataset; reports written to reports/news/.

Normalized fields (per the integration spec):
    symbol, published_at, title, summary, source, authors, topics,
    overall_sentiment_score, overall_sentiment_label, ticker_sentiment_score,
    ticker_sentiment_label, relevance_score, time_fetched
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from data_sources.alpha_vantage_client import (
    AlphaVantageClient,
    PROJECT_ROOT,
    now_utc,
    to_float,
    _env_int,
)

logger = logging.getLogger(__name__)

NEWS_FUNCTION = "NEWS_SENTIMENT"
DEFAULT_FEATURE_STORE = PROJECT_ROOT / "feature_store" / "news"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "news"

SUPPORTED_TOPICS = frozenset({
    "blockchain", "earnings", "ipo", "mergers_and_acquisitions", "financial_markets",
    "economy_fiscal", "economy_monetary", "economy_macro", "energy_transportation",
    "finance", "life_sciences", "manufacturing", "real_estate", "retail_wholesale",
    "technology",
})

# Topic -> catalyst flag mapping for catalyst detection downstream.
_CATALYST_TOPIC_FLAGS = {
    "earnings": "has_earnings_news",
    "mergers_and_acquisitions": "has_merger_news",
    "ipo": "has_merger_news",
    "economy_macro": "has_macro_news",
    "economy_monetary": "has_monetary_policy_news",
    "economy_fiscal": "has_fiscal_policy_news",
}

_AV_TIME_FMT = "%Y%m%dT%H%M"   # Alpha Vantage time_from / time_to format


# =============================================================================
# Config
# =============================================================================

@dataclass
class NewsConfig:
    enabled: bool = True
    default_limit: int = field(default_factory=lambda: _env_int("NEWS_SENTIMENT_LIMIT", 50))
    default_sort: str = "LATEST"          # LATEST | EARLIEST | RELEVANCE
    backfill_days: int = field(default_factory=lambda: _env_int("NEWS_BACKFILL_DAYS", 90))
    backfill_chunk_days: int = field(default_factory=lambda: _env_int("NEWS_BACKFILL_CHUNK_DAYS", 7))


# =============================================================================
# Parsing helpers
# =============================================================================

def _parse_av_time(raw: Any) -> Optional[str]:
    """Parse Alpha Vantage `time_published` (YYYYMMDDTHHMMSS) into ISO-8601 UTC."""
    if not raw:
        return None
    s = str(raw).strip()
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except Exception:
            continue
    # Already ISO?
    try:
        return datetime.fromisoformat(s).isoformat()
    except Exception:
        return None


def _ticker_sentiment_for(item: dict[str, Any], symbol: str) -> tuple[Optional[float], str, Optional[float]]:
    """Return (ticker_sentiment_score, ticker_sentiment_label, relevance_score)
    for ``symbol`` from a feed item's ticker_sentiment list."""
    sym = (symbol or "").upper()
    for ts in (item.get("ticker_sentiment") or []):
        if not isinstance(ts, dict):
            continue
        if str(ts.get("ticker", "")).upper() == sym:
            return (
                to_float(ts.get("ticker_sentiment_score")),
                str(ts.get("ticker_sentiment_label") or "Neutral"),
                to_float(ts.get("relevance_score")),
            )
    return None, "Neutral", None


def normalize_news_item(item: dict[str, Any], symbol: str = "") -> dict[str, Any]:
    """Normalize one NEWS_SENTIMENT feed item for a given symbol.

    When ``symbol`` is provided, ticker-specific sentiment/relevance are pulled
    from the item's ticker_sentiment list; otherwise overall sentiment is used.
    """
    topics: list[str] = []
    for t in (item.get("topics") or []):
        if isinstance(t, dict) and t.get("topic"):
            topics.append(str(t["topic"]).lower().replace(" ", "_").replace("&", "and"))
        elif isinstance(t, str):
            topics.append(t.lower())

    overall_score = to_float(item.get("overall_sentiment_score"))
    overall_label = str(item.get("overall_sentiment_label") or "Neutral")

    tkr_score, tkr_label, relevance = _ticker_sentiment_for(item, symbol)
    if tkr_score is None:
        tkr_score, tkr_label = overall_score, overall_label

    return {
        "symbol": (symbol or "").upper(),
        "published_at": _parse_av_time(item.get("time_published")),
        "title": str(item.get("title") or "")[:500],
        "summary": str(item.get("summary") or "")[:2000],
        "source": str(item.get("source") or ""),
        "authors": list(item.get("authors") or []),
        "topics": topics,
        "url": str(item.get("url") or ""),
        "overall_sentiment_score": overall_score,
        "overall_sentiment_label": overall_label,
        "ticker_sentiment_score": tkr_score,
        "ticker_sentiment_label": tkr_label,
        "relevance_score": relevance,
        "time_fetched": now_utc().isoformat(),
    }


def parse_news_feed(payload: dict[str, Any], symbol: str = "") -> list[dict[str, Any]]:
    """Normalize a full NEWS_SENTIMENT payload's `feed` for a symbol."""
    if not isinstance(payload, dict):
        return []
    feed = payload.get("feed") or []
    if not isinstance(feed, list):
        return []
    out: list[dict[str, Any]] = []
    for item in feed:
        if isinstance(item, dict):
            try:
                out.append(normalize_news_item(item, symbol))
            except Exception:  # pragma: no cover - defensive
                continue
    return out


# =============================================================================
# Client
# =============================================================================

class NewsSentimentClient:
    """Advisory-safe NEWS_SENTIMENT client. Always returns lists; never raises."""

    def __init__(
        self,
        config: Optional[NewsConfig] = None,
        av_client: Optional[AlphaVantageClient] = None,
        feature_store_dir: Optional[Path] = None,
        report_dir: Optional[Path] = None,
    ) -> None:
        self.config = config or NewsConfig()
        self.av = av_client or AlphaVantageClient()
        self.feature_store_dir = Path(feature_store_dir) if feature_store_dir is not None else DEFAULT_FEATURE_STORE
        self.report_dir = Path(report_dir) if report_dir is not None else DEFAULT_REPORT_DIR
        for d in (self.feature_store_dir, self.report_dir):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:  # pragma: no cover - non-fatal
                pass

    @property
    def api_key(self) -> str:
        return self.av.api_key

    def _build_params(
        self,
        tickers: Optional[Iterable[str]] = None,
        topics: Optional[Iterable[str]] = None,
        time_from: Optional[str] = None,
        time_to: Optional[str] = None,
        sort: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict[str, str]:
        params: dict[str, str] = {"function": NEWS_FUNCTION}
        if tickers:
            params["tickers"] = ",".join(sorted({str(t).upper().strip() for t in tickers if t}))
        if topics:
            valid = sorted({str(t).lower().strip() for t in topics if str(t).lower().strip() in SUPPORTED_TOPICS})
            if valid:
                params["topics"] = ",".join(valid)
        if time_from:
            params["time_from"] = str(time_from)
        if time_to:
            params["time_to"] = str(time_to)
        params["sort"] = (sort or self.config.default_sort).upper()
        params["limit"] = str(int(limit if limit is not None else self.config.default_limit))
        return params

    def fetch_news(
        self,
        tickers: Optional[Iterable[str]] = None,
        topics: Optional[Iterable[str]] = None,
        time_from: Optional[str] = None,
        time_to: Optional[str] = None,
        sort: Optional[str] = None,
        limit: Optional[int] = None,
        normalize_for_symbol: str = "",
    ) -> list[dict[str, Any]]:
        """Fetch + normalize news. Empty list on any failure (logged, never raised)."""
        if not self.config.enabled:
            return []
        params = self._build_params(tickers, topics, time_from, time_to, sort, limit)
        try:
            payload = self.av.fetch(params)
        except Exception as exc:  # pragma: no cover - fetch is already guarded
            logger.warning("NEWS_SENTIMENT fetch raised unexpectedly: %s", exc)
            return []
        if not payload:
            return []
        sym = normalize_for_symbol or (next(iter(params.get("tickers", "").split(",")), "") if params.get("tickers") else "")
        return parse_news_feed(payload, sym)

    def get_symbol_news(self, symbol: str, topics: Optional[Iterable[str]] = None,
                        limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Latest normalized news for a single symbol (most recent first)."""
        rows = self.fetch_news(tickers=[symbol], topics=topics, limit=limit, normalize_for_symbol=symbol)
        rows.sort(key=lambda r: r.get("published_at") or "", reverse=True)
        return rows

    # ---- historical backfill ----------------------------------------------

    def backfill_symbol(
        self,
        symbol: str,
        days: Optional[int] = None,
        chunk_days: Optional[int] = None,
        as_of: Optional[datetime] = None,
        persist: bool = True,
    ) -> list[dict[str, Any]]:
        """Build a rolling historical news dataset for ``symbol`` using
        time_from/time_to windows. Persists to feature_store/news/<SYMBOL>.jsonl.

        ``as_of`` defaults to now (UTC). Pure / deterministic when as_of is given.
        """
        symbol = (symbol or "").upper().strip()
        days = days or self.config.backfill_days
        chunk_days = chunk_days or self.config.backfill_chunk_days
        end = as_of or now_utc()
        start = end - timedelta(days=days)

        seen_urls: set[str] = set()
        collected: list[dict[str, Any]] = []
        window_start = start
        while window_start < end:
            window_end = min(window_start + timedelta(days=chunk_days), end)
            rows = self.fetch_news(
                tickers=[symbol],
                time_from=window_start.strftime(_AV_TIME_FMT),
                time_to=window_end.strftime(_AV_TIME_FMT),
                sort="EARLIEST",
                limit=1000,
                normalize_for_symbol=symbol,
            )
            for r in rows:
                key = r.get("url") or f"{r.get('title')}|{r.get('published_at')}"
                if key in seen_urls:
                    continue
                seen_urls.add(key)
                collected.append(r)
            window_start = window_end

        collected.sort(key=lambda r: r.get("published_at") or "")
        if persist and collected:
            self._persist_symbol(symbol, collected)
        return collected

    def _persist_symbol(self, symbol: str, rows: list[dict[str, Any]]) -> None:
        try:
            path = self.feature_store_dir / f"{symbol.upper()}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, default=str) + "\n")
        except Exception:  # pragma: no cover - non-fatal
            logger.warning("news backfill persist failed for %s", symbol)

    def load_symbol_history(self, symbol: str) -> list[dict[str, Any]]:
        """Load the persisted rolling news history for a symbol (empty if none)."""
        path = self.feature_store_dir / f"{symbol.upper()}.jsonl"
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        except Exception:  # pragma: no cover - non-fatal
            return []
        return out


# =============================================================================
# Module-level singleton convenience
# =============================================================================

_DEFAULT_NEWS_CLIENT: Optional[NewsSentimentClient] = None
_DEFAULT_ALPACA_NEWS_CLIENT: Any = None


def get_default_news_client() -> Any:
    """Configured news client. ``NEWS_SOURCE=alpaca`` → Alpaca news + local FinBERT sentiment
    (data_sources.alpaca_news_client); otherwise the Alpha Vantage NEWS_SENTIMENT client (default).

    Both expose get_symbol_news / backfill_symbol / load_symbol_history with the IDENTICAL row
    schema, so callers don't change. The Alpaca module is imported lazily so the AV path never
    pulls in torch/transformers. Source is re-read each call → flip NEWS_SOURCE without restart.
    """
    global _DEFAULT_NEWS_CLIENT, _DEFAULT_ALPACA_NEWS_CLIENT
    import os as _os
    if _os.getenv("NEWS_SOURCE", "av").strip().lower() == "alpaca":
        if _DEFAULT_ALPACA_NEWS_CLIENT is None:
            from data_sources.alpaca_news_client import get_default_alpaca_news_client
            _DEFAULT_ALPACA_NEWS_CLIENT = get_default_alpaca_news_client()
        return _DEFAULT_ALPACA_NEWS_CLIENT
    if _DEFAULT_NEWS_CLIENT is None:
        _DEFAULT_NEWS_CLIENT = NewsSentimentClient()
    return _DEFAULT_NEWS_CLIENT


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    syms = sys.argv[1:] or ["AAPL"]
    client = NewsSentimentClient()
    print(f"API key present: {bool(client.api_key)}")
    for s in syms:
        rows = client.get_symbol_news(s, limit=10)
        print(json.dumps({"symbol": s, "items": len(rows), "latest": rows[0] if rows else None},
                         indent=2, default=str))
