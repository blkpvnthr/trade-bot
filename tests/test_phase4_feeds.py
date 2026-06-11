"""
Phase 4 tests — Alpaca live-feed integration (IEX + overnight), data quality, no SIP.
Offline: no real sockets/network (fetch is monkeypatched or creds neutralised). Verifies
feed detection + graceful downgrade, overnight metrics, no-SIP guarantee, the engine/feed
data-quality surfaces, research feed fields, and that nothing touches execution / RL gate.
"""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]

import alpaca_feed_manager as afm
import premarket_live_feed as plf
import streaming_indicator_engine as sie
import market_state_engine as mse


def _intraday(n=12, start="2026-05-29 01:00"):
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    px = [100 + i * 0.1 for i in range(n)]
    return pd.DataFrame({"open": px, "high": [p + 0.5 for p in px], "low": [p - 0.5 for p in px],
                         "close": [p + 0.2 for p in px], "volume": [1000 + i for i in range(n)]}, index=idx)


class FeedManagerTests(unittest.TestCase):
    def test_no_sip_in_supported_feeds(self):
        # SIP is now entitlement-gated (Algo Trader Plus). Two postures, one invariant:
        #   - delayed-SIP is forbidden and never supported in ANY mode (hard invariant);
        #   - plain SIP is forbidden+excluded ONLY when the entitlement is OFF.
        self.assertNotIn("delayed_sip", afm.SUPPORTED_FEEDS)
        for f in ("delayed_sip", "v2/delayed_sip"):
            self.assertIn(f, afm.FORBIDDEN_FEEDS)
        if afm.SIP_ENABLED:
            self.assertIn("sip", afm.SUPPORTED_FEEDS)
            self.assertNotIn("sip", afm.FORBIDDEN_FEEDS)
        else:
            self.assertNotIn("sip", afm.SUPPORTED_FEEDS)
            for f in ("sip", "v2/sip"):
                self.assertIn(f, afm.FORBIDDEN_FEEDS)

    def test_forbidden_feed_fetch_refused(self):
        m = afm.AlpacaFeedManager()
        self.assertEqual(m.fetch_bars(["AAPL"], None, None, feed="sip"), {})
        self.assertEqual(m.fetch_bars(["AAPL"], None, None, feed="delayed_sip"), {})

    def test_no_creds_downgrades_gracefully(self):
        m = afm.AlpacaFeedManager()
        orig = afm._creds
        afm._creds = lambda: ("", "")
        try:
            det = m.detect_feeds(force=True)
            self.assertFalse(any(e["feed_available"] for e in det.values()))
            self.assertEqual(m.active_feed("rth"), "fallback")
            self.assertEqual(m.active_feed("premarket"), "fallback")
            diag = m.diagnostic("rth")
            self.assertFalse(diag["sip_used"])
            self.assertFalse(diag["delayed_sip_used"])
        finally:
            afm._creds = orig

    def test_detect_with_mocked_fetch_picks_top_priority_feed(self):
        # With creds + a working fetch, the manager selects the highest-priority AVAILABLE feed
        # (never fallback). When the SIP entitlement is ON, RTH prefers 'sip'; when OFF, 'iex'.
        m = afm.AlpacaFeedManager()
        afm._creds_orig = afm._creds
        afm._creds = lambda: ("k", "s")
        m.fetch_bars = lambda syms, s, e, timeframe="day", feed="iex": {"AAPL": _intraday()}
        try:
            det = m.detect_feeds(force=True)
            self.assertTrue(det["iex"]["feed_available"])
            expected_rth = "sip" if afm.SIP_ENABLED else "iex"
            self.assertEqual(m.active_feed("rth"), expected_rth)
            self.assertIn(m.active_feed("premarket"), ("overnight", "sip", "iex"))
        finally:
            afm._creds = afm._creds_orig

    def test_overnight_metrics_pure(self):
        met = afm.compute_overnight_metrics(_intraday(), prev_close=99.0)
        self.assertTrue(met["overnight_available"])
        for k in ("overnight_volume", "overnight_vwap", "overnight_range_pct",
                  "overnight_trend", "overnight_session_strength"):
            self.assertIn(k, met)
        self.assertEqual(met["overnight_trend"], "up")
        self.assertFalse(afm.compute_overnight_metrics(None)["overnight_available"])

    def test_write_diagnostic(self):
        m = afm.AlpacaFeedManager()
        afm_orig = afm._creds
        afm._creds = lambda: ("", "")
        try:
            with TemporaryDirectory() as d:
                p = m.write_diagnostic(out_dir=Path(d), session="rth")
                self.assertTrue(p.exists())
        finally:
            afm._creds = afm_orig


class PremarketOvernightTests(unittest.TestCase):
    def test_overnight_metrics_stored_and_reported(self):
        f = plf.PremarketLiveFeed(["AAPL"], output_dir=None)
        f.on_message({"id": "AAPL", "price": 100, "previousClose": 99})
        f.set_overnight_metrics("AAPL", {"overnight_available": True, "overnight_volume": 50000,
                                         "overnight_vwap": 99.5, "overnight_trend": "up",
                                         "overnight_session_strength": 0.4})
        q = f.get_symbol_state("AAPL")
        self.assertEqual(q["overnight_volume"], 50000)
        self.assertEqual(q["overnight_trend"], "up")
        with TemporaryDirectory() as d:
            self.assertTrue(f.write_overnight_report(out_dir=Path(d)).exists())


class StreamingFeedHealthTests(unittest.TestCase):
    def test_dmi_and_feed_health(self):
        e = sie.StreamingIndicatorEngine()
        e.set_feed("iex", True)
        rows = [{"timestamp": i, "open": 100 + i * 0.1, "high": 100 + i * 0.1 + 0.5,
                 "low": 100 + i * 0.1 - 0.5, "close": 100 + i * 0.1 + 0.2, "volume": 1000 + i}
                for i in range(60)]
        e.update_many({"AAPL": rows})
        ind = e.get("AAPL")["indicators"]
        for k in ("dmi_plus", "dmi_minus", "adx"):
            self.assertIn(k, ind)
        h = e.health()
        self.assertTrue(h["alpaca_feed_used"])
        self.assertEqual(h["alpaca_feed_type"], "iex")
        self.assertGreaterEqual(h["indicator_updates"], 1)
        self.assertIn("fallback_updates", h)


class MarketStateQualityTests(unittest.TestCase):
    def test_data_quality_fields_and_report(self):
        e = mse.MarketStateEngine()
        st = e.build(market_context={"overall_market_bias": "bullish", "vix": 14},
                     alpaca_quote_count=120, alpaca_bar_count=45, alpaca_data_age=2.1, feed_type="iex")
        self.assertEqual(st["alpaca_quote_count"], 120)
        self.assertEqual(st["feed_type"], "iex")
        self.assertIn("market_state_confidence", st)
        with TemporaryDirectory() as d:
            paths = e.write_report(out_dir=Path(d))
            self.assertTrue(paths["data_quality"].exists())


class ResearchFeedFieldsTests(unittest.TestCase):
    def test_alpha_event_records_feed_fields(self):
        import portfolio_research_lab as prl
        tag = "_test_phase4_feed"
        path = prl._path(prl.RESEARCH_EVENTS, tag)
        try:
            sig = {"symbol": "AAPL", "setup_type": "ORB", "grade": "A",
                   "feed_source": "alpaca", "feed_type": "iex", "feed_latency": 12.5,
                   "data_age": 1.2, "alpaca_feed_health": "ok", "bar_health": "fresh"}
            prl.record_alpha_signal(sig, market_context={}, placed=True, date_tag=tag)
            ev = prl.read_events(tag)[-1]
            for f in ("feed_source", "feed_type", "feed_latency", "data_age",
                      "alpaca_feed_health", "bar_health"):
                self.assertIn(f, ev)
            self.assertEqual(ev["feed_type"], "iex")
        finally:
            try:
                path.unlink()
            except Exception:
                pass


class SafetyInvariantTests(unittest.TestCase):
    def test_feed_manager_no_execution(self):
        src = (ROOT / "alpaca_feed_manager.py").read_text()
        self.assertNotIn("AlpacaGateway(", src)
        for forbidden in ("submit_market_order", "submit_order", "place_order", "TradingClient"):
            self.assertNotIn(forbidden, src)
        for line in src.splitlines():
            if line.strip().startswith(("import ", "from ")):
                self.assertNotIn("alpaca.trading", line)

    def test_sip_is_entitlement_gated(self):
        # SIP is now an entitlement-gated opt-in (Algo Trader Plus): OFF by default, ON only
        # when ALPACA_DATA_FEED=sip / ALPACA_SIP_ENABLED. Two hard invariants remain:
        #   (1) delayed-SIP is NEVER selected, in any mode;
        #   (2) with the flag OFF, the pre-entitlement IEX-only posture is preserved exactly.
        for f in ("alpaca_feed_manager.py", "premarket_live_feed.py", "run.py",
                  "streaming_indicator_engine.py", "market_state_engine.py"):
            self.assertNotIn("DataFeed.DELAYED_SIP", (ROOT / f).read_text(),
                             f"{f} selects delayed-SIP feed")

        import importlib
        import os as _os
        import alpaca_feed_manager as _afm
        saved = dict(_os.environ)
        try:
            # OFF (default): SIP refused, IEX-only.
            _os.environ["ALPACA_DATA_FEED"] = "iex"
            _os.environ.pop("ALPACA_SIP_ENABLED", None)
            off = importlib.reload(_afm)
            self.assertFalse(off.SIP_ENABLED)
            self.assertIn("sip", off.FORBIDDEN_FEEDS)
            self.assertNotIn("sip", off.SUPPORTED_FEEDS)
            self.assertEqual(off.RTH_PRIORITY[0], "iex")
            self.assertFalse(off.AlpacaFeedManager().diagnostic("rth")["delayed_sip_used"])

            # ON (ALPACA_DATA_FEED=sip): SIP allowed + preferred; delayed_sip still forbidden.
            _os.environ["ALPACA_DATA_FEED"] = "sip"
            on = importlib.reload(_afm)
            self.assertTrue(on.SIP_ENABLED)
            self.assertNotIn("sip", on.FORBIDDEN_FEEDS)
            self.assertIn("delayed_sip", on.FORBIDDEN_FEEDS)
            self.assertIn("sip", on.SUPPORTED_FEEDS)
            self.assertEqual(on.RTH_PRIORITY[0], "sip")
        finally:
            _os.environ.clear()
            _os.environ.update(saved)
            importlib.reload(_afm)   # restore module to the ambient environment

    def test_rl_not_promoted(self):
        import os
        # Defaults keep RL out of execution.
        self.assertNotEqual(os.environ.get("ENABLE_RL_SHADOW_MODE", "false").lower(), "true")
        import run
        self.assertTrue(hasattr(run, "_alpaca_feed_cycle"))


if __name__ == "__main__":
    unittest.main()
