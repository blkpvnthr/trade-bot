"""
Tests for the Alpaca bar-context enrichment phase (rl_alpaca_bar_enrichment) and the
related dataset/training changes. All OFFLINE — the BarCache fetch function is injected
with an in-memory fake so no network is touched; no live execution wiring is exercised.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rl_common as rc
import rl_alpaca_bar_enrichment as enr


def _frame(closes, vols=None, tz="UTC"):
    import pandas as pd
    n = len(closes)
    idx = pd.date_range("2026-05-20 13:30", periods=n, freq="5min", tz=tz)
    vols = vols if vols is not None else [1e5] * n
    c = [float(x) for x in closes]
    return pd.DataFrame({"open": c, "high": [x * 1.002 for x in c],
                         "low": [x * 0.998 for x in c], "close": c, "volume": vols}, index=idx)


class BarFeatureTests(unittest.TestCase):
    def test_rising_frame_features(self):
        f5 = _frame([100 + i * 0.1 for i in range(60)],
                    vols=[1e5] * 57 + [6e5] * 3)              # volume spike at the end
        spy = list(range(400, 650))                            # strong uptrend
        out = enr.bar_features({"5m": f5}, spy_daily=spy)
        self.assertGreater(out["mtf_trend_fast"], 0.0)
        self.assertEqual(out["mtf_alignment"], 1.0)
        self.assertGreater(out["rel_volume"], 1.0)
        self.assertGreater(out["catalyst_strength"], 0.0)     # rvol > 1 → catalyst > 0
        self.assertGreater(out["atr_pct"], 0.0)
        self.assertEqual(out["spy_regime"], 1.0)              # price above 50 & 200 DMA

    def test_missing_frames_are_neutral(self):
        out = enr.bar_features({})                            # no frames at all
        self.assertEqual(out, dict(enr.ENRICH_DEFAULTS))
        self.assertEqual(out["mtf_alignment"], 0.0)
        self.assertEqual(out["rel_volume"], 1.0)              # neutral rvol = average

    def test_empty_frame_neutral(self):
        import pandas as pd
        out = enr.bar_features({"5m": pd.DataFrame(columns=["open", "high", "low", "close", "volume"])})
        self.assertEqual(out["mtf_trend_fast"], 0.0)

    def test_regime_score_levels(self):
        self.assertEqual(enr.regime_score(list(range(1, 260))), 1.0)        # rising → bull
        self.assertEqual(enr.regime_score(list(range(260, 1, -1))), -1.0)   # falling → bear
        self.assertEqual(enr.regime_score([5.0]), 0.0)                      # too short → neutral

    def test_never_raises_on_garbage(self):
        self.assertEqual(enr.bar_features({"5m": None, "daily": "nope"})["mtf_alignment"], 0.0)


class BarCacheTests(unittest.TestCase):
    def _fake_fetch_factory(self, counter):
        import pandas as pd

        def fake(symbols, start, end, timeframe="day"):
            counter["n"] += 1
            idx = pd.date_range(start, end, freq="D")
            return {symbols[0]: pd.DataFrame(
                {"open": 1.0, "high": 1.0, "low": 1.0,
                 "close": [float(i + 1) for i in range(len(idx))], "volume": 1.0}, index=idx)}
        return fake

    def test_miss_then_hit(self):
        import pandas as pd
        counter = {"n": 0}
        with tempfile.TemporaryDirectory() as d:
            bc = enr.BarCache(cache_dir=d, fetch_fn=self._fake_fetch_factory(counter))
            s, e = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-01", tz="UTC")
            r1 = bc.get("TEST", "day", s, e)
            r2 = bc.get("TEST", "day", s, e)
            self.assertTrue(len(r1) > 0 and len(r2) > 0)
            self.assertEqual(counter["n"], 1)        # second call served from disk cache
            self.assertEqual((bc.misses, bc.hits), (1, 1))

    def test_fetch_failure_returns_empty_not_raise(self):
        import pandas as pd

        def boom(symbols, start, end, timeframe="day"):
            raise RuntimeError("network down")
        with tempfile.TemporaryDirectory() as d:
            bc = enr.BarCache(cache_dir=d, fetch_fn=boom)
            out = bc.get("TEST", "day", pd.Timestamp("2026-01-01", tz="UTC"),
                         pd.Timestamp("2026-02-01", tz="UTC"))
            self.assertEqual(len(out), 0)            # graceful empty, no exception


class EnrichEventsTests(unittest.TestCase):
    def test_skips_degenerate_symbols(self):
        # numeric / REPLAY symbols cannot be bar-enriched → counted as skipped, left neutral
        events = [{"event_type": "alpha_entry", "symbol": "20260528", "timestamp": "2026-05-28T14:00:00Z"},
                  {"event_type": "alpha_exit", "symbol": "REPLAY", "timestamp": "2026-05-28T14:00:00Z"}]
        counts = enr.enrich_events(events)
        self.assertEqual(counts["skipped_non_symbol"], 2)
        self.assertEqual(counts["enriched"], 0)

    def test_add_regime_false_is_noop(self):
        events = [{"event_type": "alpha_entry", "symbol": "AAPL", "timestamp": "2026-05-28T14:00:00Z"}]
        counts = enr.enrich_events(events, add_regime=False)
        self.assertEqual(counts, {"enriched": 0, "neutral": 0, "skipped_non_symbol": 0})


class SchemaAlignmentTests(unittest.TestCase):
    def test_extractors_match_feature_lists_after_enrichment(self):
        for feats, fn in ((rc.ALPHA_ENTRY_FEATURES, rc.alpha_entry_features),
                          (rc.ALPHA_EXIT_FEATURES, rc.alpha_exit_features),
                          (rc.BUFFETT_DCA_FEATURES, rc.buffett_dca_features)):
            self.assertEqual(len(fn({})), len(feats))

    def test_new_enrichment_features_present_on_alpha(self):
        for f in rc.ENRICH_EXTRA_FEATURES:
            self.assertIn(f, rc.ALPHA_ENTRY_FEATURES)
            self.assertIn(f, rc.ALPHA_EXIT_FEATURES)

    def test_rel_volume_centered_neutral(self):
        # rel_volume default 1.0 (average) → feature value 0.0; absent → also 0.0
        idx = rc.ALPHA_ENTRY_FEATURES.index("rel_volume")
        self.assertAlmostEqual(rc.alpha_entry_features({})[idx], 0.0)
        self.assertAlmostEqual(rc.alpha_entry_features({"rel_volume": 3.0})[idx], 2.0)

    def test_enrichment_values_flow_into_vector(self):
        e = {"vwap_distance": 0.02, "atr_pct": 0.01, "spy_regime": 1.0, "sector_regime": -0.5}
        v = rc.alpha_entry_features(e)
        for f, val in e.items():
            self.assertAlmostEqual(v[rc.ALPHA_ENTRY_FEATURES.index(f)], val)


class PerAgentTimestepTests(unittest.TestCase):
    def test_per_agent_defaults(self):
        import train_rl_agents as tr
        # alpha_entry has the largest obs space and may be tuned upward as features are added;
        # assert the contract (per-agent, distinct, sane) rather than a brittle exact value.
        self.assertGreaterEqual(tr.agent_timesteps("alpha_entry"), 10000)
        self.assertEqual(tr.agent_timesteps("alpha_exit"), 6000)
        self.assertEqual(tr.agent_timesteps("buffett_dca"), 4000)
        self.assertEqual(tr.agent_timesteps("capital_allocation"), 6000)

    def test_env_override_and_global_fallback(self):
        import train_rl_agents as tr
        os.environ["RL_TRAIN_TIMESTEPS_ALPHA_ENTRY"] = "1234"
        try:
            self.assertEqual(tr.agent_timesteps("alpha_entry"), 1234)
        finally:
            del os.environ["RL_TRAIN_TIMESTEPS_ALPHA_ENTRY"]
        # an agent with no per-agent default falls back to the global RL_TRAIN_TIMESTEPS
        self.assertEqual(tr.agent_timesteps("unknown_agent"), tr.RL_TRAIN_TIMESTEPS)


class SafetyTests(unittest.TestCase):
    def test_no_live_execution_wiring_in_enrichment(self):
        src = Path(__file__).resolve().parents[1] / "rl_alpaca_bar_enrichment.py"
        text = src.read_text()
        # No order/gateway IMPORTS or live-trading calls (prose/docstrings may mention them).
        for needle in ("import alpaca.trading", "from alpaca.trading", "AlpacaGateway(",
                       "submit_order", "DataFeed.SIP"):
            self.assertNotIn(needle, text, f"enrichment must not contain {needle!r}")

    def test_default_feed_is_not_sip(self):
        # The enrichment relies on rl_alpaca_backfill.fetch_bars, which defaults to IEX.
        import rl_alpaca_backfill as bf
        import inspect
        src = inspect.getsource(bf.fetch_bars)
        self.assertIn("DataFeed.IEX", src)
        self.assertIn('"iex"', src.lower())   # default branch is iex


if __name__ == "__main__":
    unittest.main()
