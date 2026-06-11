"""Tests for the Alpaca news client — offline (inject fake News + get_news), assert the exact
14-key schema, the relevance/topic derivations, and a round-trip through build_news_features."""
import os
import sys
import unittest
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_sources import alpaca_news_client as anc

# The exact schema every consumer depends on (news_sentiment_client.normalize_news_item).
CONTRACT_KEYS = {
    "symbol", "published_at", "title", "summary", "source", "authors", "topics", "url",
    "overall_sentiment_score", "overall_sentiment_label",
    "ticker_sentiment_score", "ticker_sentiment_label", "relevance_score", "time_fetched",
}


def _news(headline, summary="", symbols=("AAPL",), created=None, src="Reuters", url="http://x", author="A"):
    return SimpleNamespace(
        id=hash(headline) & 0xffffff, headline=headline, summary=summary, symbols=list(symbols),
        created_at=created or datetime(2026, 6, 11, 14, 0, tzinfo=timezone.utc),
        source=src, url=url, author=author,
    )


class _FakeNewsClient:
    """Stand-in for alpaca.data.historical.news.NewsClient."""
    def __init__(self, items):
        self._items = items
    def get_news(self, req):
        # single page; key the data dict by the requested symbol
        sym = (req.symbols or "AAPL").split(",")[0]
        return SimpleNamespace(data={sym: list(self._items)}, next_page_token=None)


class AlpacaNewsSchemaTests(unittest.TestCase):
    def _client(self, items):
        return anc.AlpacaNewsClient(news_client=_FakeNewsClient(items),
                                    feature_store_dir=None)

    def test_exact_contract_schema(self):
        c = self._client([_news("Apple beats earnings, raises guidance", "Strong quarter.")])
        rows = c.get_symbol_news("AAPL", limit=5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0].keys()), CONTRACT_KEYS)

    def test_sorted_desc_by_published_at(self):
        items = [
            _news("older", created=datetime(2026, 6, 10, tzinfo=timezone.utc)),
            _news("newer", created=datetime(2026, 6, 11, tzinfo=timezone.utc)),
        ]
        rows = self._client(items).get_symbol_news("AAPL", limit=5)
        self.assertEqual(rows[0]["title"], "newer")
        self.assertEqual(rows[1]["title"], "older")

    def test_relevance_sole_symbol_is_one(self):
        rows = self._client([_news("AAPL news", symbols=["AAPL"])]).get_symbol_news("AAPL", limit=1)
        self.assertEqual(rows[0]["relevance_score"], 1.0)

    def test_relevance_diluted_for_many_symbols(self):
        rows = self._client([_news("market roundup", symbols=["AAPL", "MSFT", "GOOG", "AMZN"])]).get_symbol_news("AAPL", limit=1)
        # 1/4 = 0.25 (no ticker mention in headline)
        self.assertAlmostEqual(rows[0]["relevance_score"], 0.25, places=3)

    def test_relevance_zero_when_symbol_absent(self):
        rows = self._client([_news("MSFT only", symbols=["MSFT"])]).get_symbol_news("AAPL", limit=1)
        self.assertEqual(rows[0]["relevance_score"], 0.0)

    def test_topic_inference_earnings_and_ma(self):
        self.assertIn("earnings", anc.infer_topics("Company reports quarterly earnings and raises guidance"))
        self.assertIn("mergers_and_acquisitions", anc.infer_topics("Firm to acquire rival in cash takeover"))
        self.assertEqual(anc.infer_topics("Board meets to discuss routine matters"), [])

    def test_finbert_absent_yields_neutral_schema(self):
        # With FinBERT off (default in CI), sentiment is None/Neutral but schema is intact.
        rows = self._client([_news("Apple beats earnings")]).get_symbol_news("AAPL", limit=1)
        r = rows[0]
        self.assertEqual(set(r.keys()), CONTRACT_KEYS)
        self.assertIn(r["overall_sentiment_label"], {"Positive", "Neutral", "Negative"})
        # ticker mirrors overall
        self.assertEqual(r["ticker_sentiment_score"], r["overall_sentiment_score"])
        self.assertEqual(r["ticker_sentiment_label"], r["overall_sentiment_label"])

    def test_roundtrip_through_build_news_features(self):
        # The whole point: feed our rows into the real feature builder, get a populated dict.
        from alpha_discovery.news_feature_builder import build_news_features
        items = [_news("Apple beats earnings, raises guidance", "Strong demand.",
                       created=datetime.now(timezone.utc) - timedelta(minutes=10))]
        rows = self._client(items).get_symbol_news("AAPL", limit=5)
        feats = build_news_features(rows, as_of=datetime.now(timezone.utc))
        self.assertEqual(feats.get("news_available"), 1)
        self.assertIn("sentiment_mean_60m", feats)
        self.assertIn("has_earnings_news", feats)
        self.assertEqual(feats["has_earnings_news"], 1)


class AlpacaNewsFailSafeTests(unittest.TestCase):
    def test_client_init_failure_returns_empty(self):
        # No injected client + no creds → _get_client builds a real one; if it can't, returns [].
        c = anc.AlpacaNewsClient(news_client=False)  # force "unavailable"
        self.assertEqual(c.get_symbol_news("AAPL", limit=5), [])

    def test_persist_and_load_roundtrip(self, ):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            c = anc.AlpacaNewsClient(news_client=_FakeNewsClient([_news("Apple earnings beat")]),
                                     feature_store_dir=Path(d))
            rows = c.backfill_symbol("AAPL", days=5, as_of=datetime(2026, 6, 11, tzinfo=timezone.utc))
            self.assertTrue((Path(d) / "AAPL.jsonl").exists())
            loaded = c.load_symbol_history("AAPL")
            self.assertEqual(len(loaded), len(rows))
            self.assertEqual(set(loaded[0].keys()), CONTRACT_KEYS)


if __name__ == "__main__":
    unittest.main()
