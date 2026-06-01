"""Tests for the PremarketLiveFeed live premarket quote-awareness layer.

Covers feed behaviour with the optional `yliveticker` package absent (the default
environment), the fallback path, message parsing, premarket high/low tracking,
staleness, snapshot fields, and the run.py / Alpha / Buffett integration points.
All tests are offline — no websocket is opened and no network is touched.
"""

import inspect
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import premarket_live_feed as plf


def _msg(symbol, price, **kw):
    m = {"id": symbol, "price": price}
    m.update(kw)
    return m


class FeedInitTests(unittest.TestCase):
    def test_initializes_without_yliveticker(self):
        """1. Feed constructs and is usable even when yliveticker is absent."""
        feed = plf.PremarketLiveFeed(["AAPL", "MSFT"], output_dir=None)
        self.assertEqual(feed.symbols, ["AAPL", "MSFT"])
        self.assertEqual(feed.get_snapshot()["quotes"], {})
        self.assertIsNone(feed.get_symbol_state("AAPL"))

    def test_missing_yliveticker_triggers_fallback(self):
        """2. With yliveticker unavailable, start() takes the fallback path (no raise)."""
        saved = plf.YLIVETICKER_AVAILABLE
        try:
            plf.YLIVETICKER_AVAILABLE = False
            feed = plf.PremarketLiveFeed(["AAPL"], output_dir=None)
            feed.start()   # must not raise, must not open a socket
            self.assertEqual(feed.feed_status, "fallback")
            self.assertTrue(feed.fallback_used)
            self.assertEqual(feed.fallback_source, plf.SOURCE_FALLBACK)
        finally:
            plf.YLIVETICKER_AVAILABLE = saved

    def test_feed_failure_does_not_stop(self):
        """10. A streaming failure is contained — start() never raises; feed stays usable."""
        saved_avail, saved_ylt = plf.YLIVETICKER_AVAILABLE, plf._ylt

        class _BoomYlt:
            @staticmethod
            def YLiveTicker(**kwargs):
                raise RuntimeError("simulated stream failure")

        try:
            plf.YLIVETICKER_AVAILABLE = True
            plf._ylt = _BoomYlt()
            feed = plf.PremarketLiveFeed(["AAPL"], output_dir=None)
            feed._reconnect_delay = 0.05
            feed.start()          # spawns the daemon thread; must not raise
            time.sleep(0.2)       # let the thread attempt + fail
            feed.stop()
            self.assertGreaterEqual(feed.error_count, 1)
            self.assertIsInstance(feed.get_snapshot(), dict)   # still usable
        finally:
            plf.YLIVETICKER_AVAILABLE = saved_avail
            plf._ylt = saved_ylt


class MessageParsingTests(unittest.TestCase):
    def test_message_parser_updates_state(self):
        """3. on_message parses a tick and stores per-symbol quote state."""
        feed = plf.PremarketLiveFeed(["AAPL"], output_dir=None)
        feed.on_message(_msg("AAPL", 100.0, previousClose=99.0, dayVolume="50000",
                             bid=99.9, ask=100.1, marketHours="PRE_MARKET"))
        st = feed.get_symbol_state("AAPL")
        self.assertIsNotNone(st)
        self.assertEqual(st["price"], 100.0)
        self.assertEqual(st["previous_close"], 99.0)
        self.assertEqual(st["premarket_volume"], 50000)
        self.assertAlmostEqual(st["bid_ask_spread"], 0.2, places=4)
        self.assertAlmostEqual(st["premarket_change"], 1.0, places=4)
        self.assertAlmostEqual(st["premarket_change_pct"], 1.0 / 99.0 * 100.0, places=4)
        self.assertEqual(st["source"], plf.SOURCE_LIVE)

    def test_premarket_high_low_update(self):
        """4. premarket high/low track the running max/min of observed prices."""
        feed = plf.PremarketLiveFeed(["AAPL"], output_dir=None)
        for px in (100.0, 102.5, 98.0, 101.0):
            feed.on_message(_msg("AAPL", px))
        st = feed.get_symbol_state("AAPL")
        self.assertEqual(st["premarket_high"], 102.5)
        self.assertEqual(st["premarket_low"], 98.0)
        self.assertEqual(st["price"], 101.0)

    def test_ignores_message_without_symbol(self):
        """A tick with no id/symbol is ignored (no fabricated state)."""
        feed = plf.PremarketLiveFeed(["AAPL"], output_dir=None)
        feed.on_message({"price": 100.0})
        self.assertEqual(feed.get_snapshot()["quotes"], {})


class SnapshotTests(unittest.TestCase):
    def test_stale_detection(self):
        """5. A quote older than max_stale_seconds is flagged stale; fresh is not."""
        feed = plf.PremarketLiveFeed(["AAPL"], output_dir=None, max_stale_seconds=30)
        feed.on_message(_msg("AAPL", 100.0))
        self.assertFalse(feed.get_symbol_state("AAPL")["stale"])
        feed.state["AAPL"]["_updated_ts"] -= 10_000.0   # simulate ageing
        st = feed.get_symbol_state("AAPL")
        self.assertTrue(st["stale"])
        self.assertGreater(st["age_seconds"], 30.0)

    def test_snapshot_contains_required_fields(self):
        """6. Snapshot and per-quote entries expose the documented fields."""
        feed = plf.PremarketLiveFeed(["AAPL", "SPY"], output_dir=None)
        feed.on_message(_msg("AAPL", 100.0, previousClose=99.0))
        snap = feed.get_snapshot()
        for field in ("generated_at", "source", "symbols", "quotes",
                      "stale_symbols", "fallback_used"):
            self.assertIn(field, snap)
        for field in ("symbol", "price", "previous_close", "premarket_change",
                      "premarket_change_pct", "premarket_volume", "premarket_high",
                      "premarket_low", "bid", "ask", "bid_ask_spread", "updated_at",
                      "source", "stale", "error"):
            self.assertIn(field, snap["quotes"]["AAPL"])

    def test_health_fields(self):
        """Health report exposes the documented health fields."""
        feed = plf.PremarketLiveFeed(["AAPL"], output_dir=None)
        h = feed.health()
        for field in ("feed_status", "symbols_subscribed", "symbols_updated",
                      "last_message_at", "reconnect_count", "error_count",
                      "fallback_source", "stale_count"):
            self.assertIn(field, h)

    def test_write_snapshot(self):
        """write_snapshot emits the snapshot + health JSON files."""
        import json
        feed = plf.PremarketLiveFeed(["AAPL"], output_dir=None)
        feed.on_message(_msg("AAPL", 100.0, previousClose=99.0))
        with TemporaryDirectory() as d:
            paths = feed.write_snapshot(out_dir=d)
            self.assertTrue(paths["snapshot"].exists())
            self.assertTrue(paths["health"].exists())
            snap = json.loads(paths["snapshot"].read_text())
            self.assertIn("AAPL", snap["quotes"])

    def test_apply_fallback_quotes(self):
        """Fallback quotes (yfinance/Alpaca) populate the same unified snapshot."""
        feed = plf.PremarketLiveFeed(["AAPL"], output_dir=None)
        feed.apply_fallback_quotes({"AAPL": {"price": 100.0, "previous_close": 98.0}})
        st = feed.get_symbol_state("AAPL")
        self.assertEqual(st["price"], 100.0)
        self.assertEqual(st["source"], plf.SOURCE_FALLBACK)
        self.assertTrue(feed.fallback_used)


class IntegrationTests(unittest.TestCase):
    def test_run_py_imports_cleanly(self):
        """7. run.py imports without error and exposes the premarket helpers."""
        import run
        for attr in ("is_premarket", "_premarket_feed", "run_premarket_phase",
                     "_premarket_snapshot", "_stop_premarket_feed"):
            self.assertTrue(hasattr(run, attr), f"run.{attr} missing")

    def test_alpha_accepts_premarket_snapshot(self):
        """8. AlphaCallEngine.evaluate_universe / detect_setup accept & use the snapshot."""
        import alpha_call_engine as ace
        self.assertIn("premarket_snapshot",
                      inspect.signature(ace.AlphaCallEngine.evaluate_universe).parameters)
        self.assertIn("premarket_snapshot",
                      inspect.signature(ace.detect_setup).parameters)
        import pandas as pd
        bars = pd.DataFrame(
            [{"open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 10_000}
             for _ in range(30)],
            index=pd.DatetimeIndex([pd.Timestamp("2026-05-30", tz="UTC") for _ in range(30)]),
        )
        snap = {"quotes": {"AAPL": {"source": "yliveticker", "stale": False,
                                    "premarket_change_pct": 2.5, "premarket_volume": 1_000_000,
                                    "premarket_high": 105.0, "premarket_low": 99.0}},
                "stale_symbols": [], "fallback_used": False}
        ev = ace.detect_setup("AAPL", bars, ace.AlphaCallConfig(),
                              {"overall_market_bias": "bullish"}, None, snap)
        self.assertEqual(ev.premarket_source, "yliveticker")
        self.assertEqual(ev.premarket_gap_pct, 2.5)
        self.assertEqual(ev.premarket_high, 105.0)
        self.assertFalse(ev.premarket_stale)

    def test_buffett_screener_accepts_premarket_snapshot(self):
        """9. buffett_accumulation_screener.screen accepts the snapshot; awareness annotates."""
        import buffett_accumulation_screener as bscr
        self.assertIn("premarket_snapshot", inspect.signature(bscr.screen).parameters)
        result = {"candidates": [], "funnel": {},
                  "detail": [{"symbol": "AAPL", "is_candidate": True}]}
        bscr._annotate_premarket_awareness(
            result, {"quotes": {"AAPL": {"source": "yliveticker", "stale": False,
                                         "premarket_change_pct": 3.0}},
                     "fallback_used": False})
        self.assertTrue(result["funnel"]["premarket_available"])
        self.assertEqual(result["detail"][0]["premarket_gap_pct"], 3.0)
        self.assertEqual(result["detail"][0]["premarket_source"], "yliveticker")

    def test_research_lab_records_premarket_fields(self):
        """Research recorder persists the premarket-source fields onto alpha events."""
        import portfolio_research_lab as prl
        date_tag = "_test_premarket_feed"
        path = prl._path(prl.RESEARCH_EVENTS, date_tag)
        try:
            sig = {"symbol": "AAPL", "setup_type": "ORB", "grade": "A",
                   "premarket_source": "yliveticker", "premarket_gap_pct": 2.1,
                   "premarket_volume": 500000, "premarket_relative_strength": 1.2,
                   "premarket_stale": False, "premarket_high": 104.0, "premarket_low": 99.0}
            prl.record_alpha_signal(sig, market_context={}, placed=True, date_tag=date_tag)
            events = prl.read_events(date_tag)
            self.assertGreaterEqual(len(events), 1)
            ev = events[-1]
            for f in ("premarket_source", "premarket_gap_pct", "premarket_volume",
                      "premarket_relative_strength", "premarket_stale",
                      "premarket_high", "premarket_low"):
                self.assertIn(f, ev)
            self.assertEqual(ev["premarket_source"], "yliveticker")
        finally:
            try:
                path.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
