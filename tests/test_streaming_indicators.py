"""Tests for the StreamingIndicatorEngine real-time indicator infrastructure layer.

Covers engine behaviour with the optional `streaming_indicators` package absent
(the default CI environment), the public API, staleness detection, static-fallback
sourcing, and the Alpha/Buffett/Research integration points.
"""

import inspect
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streaming_indicator_engine as sie


def _series(n: int, start: float = 100.0) -> list:
    """Deterministic oscillating OHLCV candle series with integer timestamps."""
    candles = []
    px = start
    for i in range(n):
        px = max(1.0, px + math.sin(i / 5.0) * 1.5 + (0.1 if i % 3 == 0 else -0.05))
        candles.append({
            "timestamp": i,
            "open": px - 0.2,
            "high": px + 0.5,
            "low": px - 0.5,
            "close": px,
            "volume": 10_000 + (i * 137) % 5_000,
        })
    return candles


class EngineInitTests(unittest.TestCase):
    def test_initializes_without_package(self):
        """1. Engine constructs and is usable even when streaming_indicators is absent."""
        eng = sie.StreamingIndicatorEngine()
        self.assertIsInstance(eng, sie.StreamingIndicatorEngine)
        # In CI the package is not installed; the engine must still work.
        self.assertEqual(eng.snapshot(), {})
        self.assertEqual(eng.update_errors, 0)

    def test_update_symbol_stores_latest_state(self):
        """2. update_symbol ingests a candle and stores per-symbol state."""
        eng = sie.StreamingIndicatorEngine()
        candle = {"timestamp": 1, "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000}
        eng.update_symbol("MSFT", candle)
        state = eng.get("MSFT")
        self.assertIsNotNone(state)
        self.assertEqual(state["symbol"], "MSFT")
        self.assertEqual(state["bars_seen"], 1)
        self.assertIsNotNone(state["updated_at"])
        # VWAP is computable from a single candle (typical price).
        self.assertAlmostEqual(state["indicators"]["vwap"], (11 + 9 + 10.5) / 3.0, places=4)

    def test_update_many_multiple_symbols(self):
        """3. update_many populates several symbols and computes warm indicators."""
        eng = sie.StreamingIndicatorEngine()
        eng.update_many({"AAPL": _series(60), "NVDA": _series(60, start=400.0)})
        snap = eng.snapshot()
        self.assertEqual(set(snap), {"AAPL", "NVDA"})
        for sym in ("AAPL", "NVDA"):
            ind = snap[sym]["indicators"]
            self.assertIsNotNone(ind["rsi_14"])     # warm after >14 bars
            self.assertIsNotNone(ind["sma_20"])     # warm after >=20 bars
            self.assertIsNotNone(ind["macd"])       # warm after >=26 bars
            self.assertIsNotNone(ind["atr_14"])

    def test_static_sma_value_is_correct(self):
        """Static math sanity: sma_20 equals the mean of the last 20 closes."""
        eng = sie.StreamingIndicatorEngine()
        candles = _series(40)
        eng.update_many({"AAPL": candles})
        closes = [c["close"] for c in candles]
        expected = sum(closes[-20:]) / 20.0
        self.assertAlmostEqual(eng.get("AAPL")["indicators"]["sma_20"], expected, places=4)

    def test_dedupes_already_seen_candles(self):
        """Re-feeding the same timestamps does not double-count bars (incremental)."""
        eng = sie.StreamingIndicatorEngine()
        candles = _series(30)
        eng.update_many({"AAPL": candles})
        first = eng.get("AAPL")["bars_seen"]
        eng.update_many({"AAPL": candles})           # identical timestamps → ignored
        self.assertEqual(eng.get("AAPL")["bars_seen"], first)
        eng.update_many({"AAPL": _series(35)[30:]})  # only the 5 new timestamps ingest
        self.assertEqual(eng.get("AAPL")["bars_seen"], first + 5)


class SnapshotTests(unittest.TestCase):
    def test_snapshot_includes_expected_fields(self):
        """4. Each snapshot entry exposes the documented fields."""
        eng = sie.StreamingIndicatorEngine()
        eng.update_many({"AAPL": _series(40)})
        state = eng.snapshot()["AAPL"]
        for field in ("symbol", "indicators", "source", "fallback_used",
                      "streaming_indicator_available", "updated_at",
                      "indicator_age_seconds", "stale"):
            self.assertIn(field, state)
        for key in ("rsi_14", "sma_20", "sma_50", "ema_20", "vwap", "macd",
                    "atr_14", "stoch_k", "stoch_d", "williams_r", "cci", "mfi",
                    "aroon_osc", "supertrend"):
            self.assertIn(key, state["indicators"])

    def test_stale_detection(self):
        """5. State older than stale_sec is flagged stale; fresh state is not."""
        eng = sie.StreamingIndicatorEngine(stale_sec=120.0)
        eng.update_many({"AAPL": _series(40)})
        self.assertFalse(eng.snapshot()["AAPL"]["stale"])
        # Simulate the state ageing well past the threshold.
        eng.state_by_symbol["AAPL"].updated_ts -= 10_000.0
        stale_state = eng.snapshot()["AAPL"]
        self.assertTrue(stale_state["stale"])
        self.assertGreater(stale_state["indicator_age_seconds"], 120.0)

    def test_fallback_static_when_streaming_unavailable(self):
        """6. With no streaming package, every indicator is sourced static_fallback."""
        eng = sie.StreamingIndicatorEngine()
        eng.update_many({"AAPL": _series(40)})
        state = eng.snapshot()["AAPL"]
        if not sie.STREAMING_PACKAGE_AVAILABLE:
            self.assertEqual(state["source"], sie.SOURCE_STATIC)
            self.assertTrue(state["fallback_used"])
            self.assertFalse(state["streaming_indicator_available"])
            self.assertTrue(all(v == sie.SOURCE_STATIC
                                for v in state["indicator_sources"].values()))

    def test_no_fabrication_without_close(self):
        """Candles without a close are rejected — indicators are never fabricated."""
        eng = sie.StreamingIndicatorEngine()
        eng.update_symbol("BAD", {"timestamp": 1, "open": 1, "high": 2, "low": 0, "volume": 10})
        self.assertIsNone(eng.get("BAD"))

    def test_write_report(self):
        """write_report emits snapshot + health JSON with the documented health fields."""
        import json
        import tempfile
        eng = sie.StreamingIndicatorEngine()
        eng.update_many({"AAPL": _series(40)})
        with tempfile.TemporaryDirectory() as d:
            paths = eng.write_report(out_dir=d)
            self.assertTrue(paths["snapshot"].exists())
            self.assertTrue(paths["health"].exists())
            health = json.loads(paths["health"].read_text())
            for field in ("symbols_tracked", "indicators_tracked",
                          "streaming_package_available", "stale_symbols",
                          "update_errors", "fallback_count"):
                self.assertIn(field, health)
            self.assertEqual(health["symbols_tracked"], 1)


class IntegrationTests(unittest.TestCase):
    def test_alpha_engine_accepts_indicator_snapshot(self):
        """7. AlphaCallEngine.evaluate_universe / check_active_positions accept the kwarg."""
        import alpha_call_engine as ace
        for name in ("evaluate_universe", "check_active_positions"):
            params = inspect.signature(getattr(ace.AlphaCallEngine, name)).parameters
            self.assertIn("indicator_snapshot", params, f"{name} missing indicator_snapshot")
        # detect_setup must tolerate a fresh snapshot and use the streaming VWAP.
        import pandas as pd
        bars = pd.DataFrame(
            [{"open": c["open"], "high": c["high"], "low": c["low"],
              "close": c["close"], "volume": c["volume"]} for c in _series(40)],
            index=pd.DatetimeIndex([pd.Timestamp("2026-05-30", tz="UTC") for _ in range(40)]),
        )
        snap = {"AAPL": {"indicators": {"vwap": 123.45},
                         "stale": False, "streaming_indicator_available": True,
                         "indicator_age_seconds": 1.0, "source": "streaming",
                         "fallback_used": False}}
        mc = {"overall_market_bias": "bullish"}   # pass the regime gate → reach base metrics
        ev = ace.detect_setup("AAPL", bars, ace.AlphaCallConfig(), mc, snap)
        self.assertEqual(ev.indicator_source, "streaming")
        self.assertFalse(ev.fallback_used)
        self.assertAlmostEqual(ev.vwap, 123.45, places=2)

    def test_alpha_detect_setup_falls_back_when_stale(self):
        """Stale streaming state → bar-based VWAP and indicator_source=fallback_static."""
        import alpha_call_engine as ace
        import pandas as pd
        bars = pd.DataFrame(
            [{"open": c["open"], "high": c["high"], "low": c["low"],
              "close": c["close"], "volume": c["volume"]} for c in _series(40)],
            index=pd.DatetimeIndex([pd.Timestamp("2026-05-30", tz="UTC") for _ in range(40)]),
        )
        snap = {"AAPL": {"indicators": {"vwap": 123.45}, "stale": True,
                         "streaming_indicator_available": True,
                         "indicator_age_seconds": 9_999.0}}
        mc = {"overall_market_bias": "bullish"}   # pass the regime gate → reach base metrics
        ev = ace.detect_setup("AAPL", bars, ace.AlphaCallConfig(), mc, snap)
        self.assertEqual(ev.indicator_source, "fallback_static")
        self.assertTrue(ev.fallback_used)
        self.assertNotAlmostEqual(ev.vwap, 123.45, places=2)

    def test_buffett_dca_accepts_indicator_snapshot(self):
        """8. BuffettAccumulationLayer.evaluate_dca accepts the indicator_snapshot kwarg."""
        import buffett_accumulation_layer as bal
        params = inspect.signature(bal.BuffettAccumulationLayer.evaluate_dca).parameters
        self.assertIn("indicator_snapshot", params)
        # An empty-universe call must accept the kwarg and return a list without error.
        layer = bal.BuffettAccumulationLayer(config=bal.BALConfig(), dca_config=bal.DCAConfig())
        out = layer.evaluate_dca(symbols=[], market_context={}, bars_ctx_by_symbol={},
                                 equity=100_000.0, positions={}, available_capital=10_000.0,
                                 indicator_snapshot={"AAPL": {"indicators": {"vwap": 10.0},
                                                              "stale": False}})
        self.assertIsInstance(out, list)

    def test_research_events_record_indicator_source(self):
        """9. Research recorders persist the indicator-source fields onto events."""
        import portfolio_research_lab as prl
        date_tag = "_test_streaming_indicators"
        path = prl._path(prl.RESEARCH_EVENTS, date_tag)
        try:
            sig = {"symbol": "AAPL", "setup_type": "VWAP_RECLAIM", "grade": "A",
                   "indicator_source": "streaming", "indicator_age_seconds": 2.5,
                   "streaming_indicator_available": True, "fallback_used": False}
            prl.record_alpha_signal(sig, market_context={}, placed=True, date_tag=date_tag)
            order = {"symbol": "BRK.B", "tier": "starter", "indicator_source": "fallback_static",
                     "indicator_age_seconds": None, "streaming_indicator_available": False,
                     "fallback_used": True}
            prl.record_bal_order(order, market_context={}, date_tag=date_tag)
            events = prl.read_events(date_tag)
            self.assertGreaterEqual(len(events), 2)
            for ev in events:
                self.assertIn("indicator_source", ev)
                self.assertIn("streaming_indicator_available", ev)
                self.assertIn("fallback_used", ev)
            alpha_ev = next(e for e in events if e["symbol"] == "AAPL")
            self.assertEqual(alpha_ev["indicator_source"], "streaming")
        finally:
            try:
                path.unlink()
            except Exception:
                pass

    def test_run_py_imports_cleanly(self):
        """10. run.py imports without error and exposes the streaming helpers."""
        import run
        self.assertTrue(hasattr(run, "_streaming_engine"))
        self.assertTrue(hasattr(run, "_update_indicators"))
        self.assertTrue(hasattr(run, "_indicator_snapshot"))


if __name__ == "__main__":
    unittest.main()
