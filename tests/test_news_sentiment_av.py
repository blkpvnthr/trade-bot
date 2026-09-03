#!/usr/bin/env python3
"""
Tests for NEWS_SENTIMENT client + news feature/embedding builders.

Offline via injected http_get. Covers request generation (tickers/topics/time),
normalization, sentiment aggregation, catalyst detection, news-price
interactions, compressed embeddings, neutral fallback, and the RL schema gating.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_sources.alpha_vantage_client import AlphaVantageClient, AlphaVantageConfig  # noqa: E402
from data_sources.news_sentiment_client import (  # noqa: E402
    NewsSentimentClient, NewsConfig, normalize_news_item, parse_news_feed, SUPPORTED_TOPICS,
)
from alpha_discovery.news_feature_builder import build_news_features, neutral_news_features  # noqa: E402
from alpha_discovery.news_embedding_engine import build_compressed_embeddings  # noqa: E402


def _av(http_get):
    tmp = Path(tempfile.mkdtemp())
    return AlphaVantageClient(config=AlphaVantageConfig(), api_key="K", http_get=http_get,
                              cache_dir=tmp / "c", request_log_path=tmp / "r.csv",
                              status_report_path=tmp / "s.json")


def _news_client(http_get):
    tmp = Path(tempfile.mkdtemp())
    return NewsSentimentClient(config=NewsConfig(), av_client=_av(http_get),
                               feature_store_dir=tmp / "fs", report_dir=tmp / "rep")


_FEED_ITEM = {
    "title": "MegaCorp beats earnings, raises guidance",
    "url": "https://example.com/a",
    "time_published": "20260531T143000",
    "authors": ["Jane Reporter"],
    "summary": "MegaCorp reported strong quarterly earnings and raised guidance.",
    "source": "Example Wire",
    "overall_sentiment_score": 0.35,
    "overall_sentiment_label": "Bullish",
    "topics": [{"topic": "Earnings", "relevance_score": "0.9"},
               {"topic": "Technology", "relevance_score": "0.8"}],
    "ticker_sentiment": [{"ticker": "MEGA", "relevance_score": "0.95",
                          "ticker_sentiment_score": "0.42", "ticker_sentiment_label": "Bullish"}],
}


class TestRequestGeneration(unittest.TestCase):
    def test_tickers_topics_time_params(self):
        captured = {}

        def fake(url, params, timeout):
            captured.update(params)
            return {"feed": []}

        c = _news_client(fake)
        c.fetch_news(tickers=["aapl", "msft"], topics=["earnings", "not_a_topic", "technology"],
                     time_from="20260501T0000", time_to="20260531T0000", sort="EARLIEST", limit=20)
        self.assertEqual(captured["function"], "NEWS_SENTIMENT")
        self.assertEqual(captured["tickers"], "AAPL,MSFT")
        # invalid topic dropped, valid ones sorted
        self.assertEqual(captured["topics"], "earnings,technology")
        self.assertEqual(captured["time_from"], "20260501T0000")
        self.assertEqual(captured["sort"], "EARLIEST")
        self.assertEqual(captured["limit"], "20")

    def test_only_supported_topics_constant(self):
        self.assertIn("mergers_and_acquisitions", SUPPORTED_TOPICS)
        self.assertIn("economy_monetary", SUPPORTED_TOPICS)


class TestNormalization(unittest.TestCase):
    def test_normalize_item_ticker_specific(self):
        n = normalize_news_item(_FEED_ITEM, "MEGA")
        self.assertEqual(n["symbol"], "MEGA")
        self.assertEqual(n["ticker_sentiment_score"], 0.42)
        self.assertEqual(n["relevance_score"], 0.95)
        self.assertIn("earnings", n["topics"])
        self.assertTrue(n["published_at"].startswith("2026-05-31T14:30"))

    def test_parse_feed(self):
        rows = parse_news_feed({"feed": [_FEED_ITEM]}, "MEGA")
        self.assertEqual(len(rows), 1)


class TestFeatureBuilder(unittest.TestCase):
    def _rows(self, n=3, base_score=0.3):
        now = datetime(2026, 5, 31, 15, 0, tzinfo=timezone.utc)
        rows = []
        for i in range(n):
            item = dict(_FEED_ITEM)
            item["time_published"] = (now - timedelta(minutes=5 * i)).strftime("%Y%m%dT%H%M%S")
            item["ticker_sentiment"] = [{"ticker": "MEGA", "relevance_score": "0.9",
                                         "ticker_sentiment_score": str(base_score),
                                         "ticker_sentiment_label": "Bullish"}]
            rows.append(normalize_news_item(item, "MEGA"))
        return rows, now

    def test_windowed_counts_and_sentiment(self):
        rows, now = self._rows(3, 0.3)
        feats = build_news_features(rows, as_of=now)
        self.assertEqual(feats["news_count_15m"], 3)
        self.assertGreater(feats["sentiment_mean_15m"], 0.0)
        self.assertEqual(feats["news_available"], 1)

    def test_catalyst_detection(self):
        rows, now = self._rows(1, 0.3)
        feats = build_news_features(rows, as_of=now)
        self.assertEqual(feats["has_earnings_news"], 1)
        self.assertEqual(feats["has_sector_news"], 1)  # Technology topic
        self.assertGreater(feats["catalyst_strength_score"], 0.0)

    def test_news_price_interactions(self):
        rows, now = self._rows(2, 0.4)
        ctx = {"price_momentum": 0.5, "volume_acceleration": 0.3, "relative_volume": 2.0,
               "vwap_distance": 0.1, "volatility": 0.2, "iv_change": 0.05,
               "options_flow": 0.4, "index_strength": 0.6}
        feats = build_news_features(rows, as_of=now, price_context=ctx)
        self.assertNotEqual(feats["news_sentiment_x_price_momentum"], 0.0)
        self.assertNotEqual(feats["earnings_news_x_iv_change"], 0.0)  # has earnings * iv_change
        self.assertIn("hidden_news_continuation_score", feats)

    def test_neutral_fallback_no_news(self):
        feats = build_news_features([], as_of=datetime.now(timezone.utc))
        self.assertEqual(feats, neutral_news_features())
        self.assertEqual(feats["news_available"], 0)
        self.assertEqual(feats["minutes_since_latest_news"], 1440.0)

    def test_fetch_failure_yields_neutral(self):
        def fake(url, params, timeout):
            raise RuntimeError("network down")

        c = _news_client(fake)
        rows = c.get_symbol_news("MEGA")
        self.assertEqual(rows, [])
        feats = build_news_features(rows)
        self.assertEqual(feats["news_available"], 0)


class TestEmbeddings(unittest.TestCase):
    def test_compressed_embeddings_shape(self):
        rows = [normalize_news_item({**_FEED_ITEM, "title": f"headline {i} merger deal"},
                                    "MEGA") for i in range(5)]
        emb = build_compressed_embeddings(rows, n_components=3)
        self.assertEqual(len(emb), 5)
        self.assertIn("compressed_news_embedding_1", emb[0])
        self.assertIn("compressed_news_embedding_3", emb[0])

    def test_no_raw_text_in_embeddings(self):
        rows = [normalize_news_item(_FEED_ITEM, "MEGA") for _ in range(3)]
        emb = build_compressed_embeddings(rows, n_components=2)
        for e in emb:
            for v in e.values():
                self.assertIsInstance(v, float)


class TestRLSchemaGating(unittest.TestCase):
    def test_news_features_in_schema_when_enabled(self):
        import importlib, os
        os.environ["RL_NEWS_FEATURES"] = "1"
        import rl_common
        importlib.reload(rl_common)
        try:
            self.assertIn("sentiment_mean_60m", rl_common.ALPHA_ENTRY_FEATURES)
            n = len(rl_common.ALPHA_ENTRY_FEATURES)
            v = len(rl_common.alpha_entry_features({"sentiment_mean_60m": 0.2}))
            self.assertEqual(n, v)
        finally:
            os.environ.pop("RL_NEWS_FEATURES", None)
            importlib.reload(rl_common)


if __name__ == "__main__":
    unittest.main()
