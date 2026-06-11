"""Tests for the real-time Alpaca news stream router — handler persists normalized rows per
mentioned symbol, dedupes on url, and never raises (offline; redirect FEATURE_STORE to a tmp dir)."""
import asyncio
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alpaca_news_stream_router as r

CONTRACT_KEYS = {
    "symbol", "published_at", "title", "summary", "source", "authors", "topics", "url",
    "overall_sentiment_score", "overall_sentiment_label",
    "ticker_sentiment_score", "ticker_sentiment_label", "relevance_score", "time_fetched",
}


def _news(headline, symbols, nid):
    return SimpleNamespace(id=nid, headline=headline, summary="", symbols=list(symbols),
                          created_at=datetime(2026, 6, 11, 14, tzinfo=timezone.utc),
                          source="Reuters", url=f"http://x/{nid}", author="A")


class NewsStreamHandlerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_store = r.FEATURE_STORE
        r.FEATURE_STORE = Path(self._tmp.name)
        r._SEEN.clear()

    def tearDown(self):
        r.FEATURE_STORE = self._orig_store
        r._SEEN.clear()
        self._tmp.cleanup()

    def _rows(self, sym):
        p = Path(self._tmp.name) / f"{sym}.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []

    def test_persists_per_symbol_with_exact_schema(self):
        asyncio.run(r.news_handler(_news("Apple and Microsoft form AI partnership", ["AAPL", "MSFT"], 1)))
        self.assertEqual(len(self._rows("AAPL")), 1)
        self.assertEqual(len(self._rows("MSFT")), 1)
        self.assertEqual(set(self._rows("AAPL")[0].keys()), CONTRACT_KEYS)

    def test_dedupes_same_url(self):
        asyncio.run(r.news_handler(_news("Apple beats earnings", ["AAPL"], 7)))
        asyncio.run(r.news_handler(_news("Apple beats earnings", ["AAPL"], 7)))  # same url
        self.assertEqual(len(self._rows("AAPL")), 1)

    def test_handler_never_raises_on_bad_input(self):
        # A News-like object missing fields must not crash the stream.
        bad = SimpleNamespace()
        asyncio.run(r.news_handler(bad))  # should swallow + continue

    def test_relevance_and_topics_derived(self):
        asyncio.run(r.news_handler(_news("Apple beats earnings, raises guidance", ["AAPL"], 9)))
        row = self._rows("AAPL")[0]
        self.assertEqual(row["relevance_score"], 1.0)        # sole symbol
        self.assertIn("earnings", row["topics"])


if __name__ == "__main__":
    unittest.main()
