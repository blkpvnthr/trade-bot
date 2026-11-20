"""Cached, rate-limited RSS news discovery. No SerpAPI credentials required."""
from __future__ import annotations
import json, os, time, hashlib
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus
import requests
from bs4 import BeautifulSoup

CACHE_DIR = Path(__file__).resolve().parent / "daily_system_outputs" / "news_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TTL = int(os.getenv("MARKET_NEWS_CACHE_SECONDS", "900"))
MAX_AGE_HOURS = int(os.getenv("MARKET_NEWS_MAX_AGE_HOURS", "72"))
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "13KTradingResearch/1.0 (RSS market monitoring)"})
_last_request = 0.0

def search_news(query: str, limit: int = 8) -> list[dict]:
    global _last_request
    key = hashlib.sha256(query.encode()).hexdigest()[:20]
    cache = CACHE_DIR / f"{key}.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < TTL:
        try: return json.loads(cache.read_text())
        except (OSError, ValueError): pass
    delay = max(0.0, 1.5 - (time.monotonic() - _last_request))
    if delay: time.sleep(delay)
    _last_request = time.monotonic()
    url = "https://news.google.com/rss/search"
    try:
        response = _SESSION.get(url, params={"q": query, "hl":"en-US", "gl":"US", "ceid":"US:en"}, timeout=18)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "xml")
        if soup.find("rss") is None:
            raise ValueError("Unexpected RSS response")
        result = []
        for item in soup.find_all("item"):
            title = item.find("title")
            link = item.find("link")
            date = item.find("pubDate")
            source = item.find("source")
            if not title or not link: continue
            published = date.get_text(strip=True) if date else None
            if published:
                try:
                    age = (datetime.now(timezone.utc) - parsedate_to_datetime(published).astimezone(timezone.utc)).total_seconds()
                    if age < -3600 or age > MAX_AGE_HOURS * 3600: continue
                except (ValueError, TypeError, OverflowError): continue
            else: continue  # do not treat undated articles as fresh
            result.append({"title":title.get_text(" ", strip=True), "source":source.get_text(" ", strip=True) if source else "Google News RSS", "date":published, "snippet":"", "link":link.get_text(strip=True), "query":query, "type":"news"})
            if len(result) >= limit: break
        cache.write_text(json.dumps(result, indent=2))
        return result
    except (requests.RequestException, ValueError, OSError) as exc:
        print(f"[market-news] query failed ({type(exc).__name__}); using recent cache if available")
        if cache.exists() and time.time() - cache.stat().st_mtime < MAX_AGE_HOURS * 3600:
            try: return json.loads(cache.read_text())
            except (OSError, ValueError): pass
        return []
