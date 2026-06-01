"""
Tests for the RL enhancements adapted from Gregory Roy (2024) MSc thesis,
"Multi-Timeframe Algorithmic Trading Bots using Thick Data Heuristics with Deep
Reinforcement Learning" (TDH-DQN). All offline / advisory — no execution-authority
change, no RL promotion, deployment gate intact.

Covers:
  * multi-timeframe (MTF) trend features + alignment score   (thesis: multi-timeframe core)
  * catalyst-strength feature on alpha entry                 (§3.9 catalyst stocks)
  * APR (Accumulated Percent Returns) metric                 (§3.8)
  * optional DQN+experience-replay algorithm plumbing        (TDH-DQN; default OFF)
  * feature/extractor index-alignment after the additions    (builder maps by index)
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rl_common as rc


class MTFFeatureTests(unittest.TestCase):
    def test_rising_series_aligns_up(self):
        m = rc.mtf_trend_features([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        self.assertEqual(set(m), set(rc.MTF_FEATURES))
        self.assertGreater(m["mtf_trend_fast"], 0.0)
        self.assertEqual(m["mtf_alignment"], 1.0)

    def test_falling_series_aligns_down(self):
        m = rc.mtf_trend_features([20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10])
        self.assertEqual(m["mtf_alignment"], -1.0)
        self.assertLess(m["mtf_trend_fast"], 0.0)

    def test_thin_or_bad_series_is_neutral(self):
        for bad in ([], [5.0], [0.0, 0.0], [None, None], "not-a-series"):
            m = rc.mtf_trend_features(bad)
            self.assertEqual(m["mtf_alignment"], 0.0)
            self.assertTrue(all(v == 0.0 for v in m.values()))

    def test_windows_capped_to_series_length(self):
        # slow window (200) larger than the series must still produce a value, not crash
        m = rc.mtf_trend_features([1, 2, 3, 4, 5], windows=(2, 3, 200))
        self.assertNotEqual(m["mtf_trend_slow"], None)


class FeatureAlignmentTests(unittest.TestCase):
    """The dataset builder maps features by index, so list length must equal extractor length."""

    def test_extractors_match_feature_lists(self):
        for feats, fn in ((rc.ALPHA_ENTRY_FEATURES, rc.alpha_entry_features),
                          (rc.ALPHA_EXIT_FEATURES, rc.alpha_exit_features),
                          (rc.BUFFETT_DCA_FEATURES, rc.buffett_dca_features)):
            self.assertEqual(len(fn({})), len(feats))

    def test_mtf_features_appended_to_each_set(self):
        for feats in (rc.ALPHA_ENTRY_FEATURES, rc.ALPHA_EXIT_FEATURES, rc.BUFFETT_DCA_FEATURES):
            for k in rc.MTF_FEATURES:
                self.assertIn(k, feats)

    def test_catalyst_only_on_entry(self):
        self.assertIn("catalyst_strength", rc.ALPHA_ENTRY_FEATURES)
        self.assertNotIn("catalyst_strength", rc.ALPHA_EXIT_FEATURES)
        self.assertNotIn("catalyst_strength", rc.BUFFETT_DCA_FEATURES)

    def test_entry_extractor_reads_mtf_and_catalyst(self):
        e = {"setup_score": 80, "mtf_trend_fast": 0.5, "mtf_alignment": 1.0,
             "catalyst_strength": 0.9}
        vec = rc.alpha_entry_features(e)
        # catalyst is the last feature; clamped to [0,1]
        self.assertAlmostEqual(vec[rc.ALPHA_ENTRY_FEATURES.index("catalyst_strength")], 0.9)
        self.assertAlmostEqual(vec[rc.ALPHA_ENTRY_FEATURES.index("mtf_alignment")], 1.0)

    def test_catalyst_strength_clamped(self):
        hi = rc.alpha_entry_features({"catalyst_strength": 5.0})
        lo = rc.alpha_entry_features({"catalyst_strength": -3.0})
        idx = rc.ALPHA_ENTRY_FEATURES.index("catalyst_strength")
        self.assertEqual(hi[idx], 1.0)
        self.assertEqual(lo[idx], 0.0)


class APRMetricTests(unittest.TestCase):
    def test_apr_is_accumulated_return(self):
        m = rc.compute_metrics([1.0, -0.5, 2.0])
        self.assertAlmostEqual(m["apr"], 2.5, places=6)

    def test_apr_empty(self):
        self.assertEqual(rc.compute_metrics([])["apr"], 0.0)


class DQNPlumbingTests(unittest.TestCase):
    def test_resolve_algo_precedence(self):
        import train_rl_agents as tr
        self.assertEqual(tr.resolve_algo("alpha_entry", "DQN"), "DQN")     # explicit arg
        self.assertEqual(tr.resolve_algo("alpha_entry", None), "PPO")      # table default unchanged
        os.environ["RL_ALGO_ALPHA_ENTRY"] = "DQN"
        try:
            self.assertEqual(tr.resolve_algo("alpha_entry", None), "DQN")  # per-agent env override
        finally:
            del os.environ["RL_ALGO_ALPHA_ENTRY"]

    def test_global_discrete_algo_override(self):
        import train_rl_agents as tr
        os.environ["RL_DISCRETE_ALGO"] = "DQN"
        try:
            self.assertEqual(tr.resolve_algo("buffett_dca", None), "DQN")
        finally:
            del os.environ["RL_DISCRETE_ALGO"]

    def test_dqn_hyperparams_follow_table_6_1(self):
        import train_rl_agents as tr
        self.assertEqual(tr.RL_DQN_LEARNING_RATE, 1e-3)   # Table 6.1 learning rate
        self.assertEqual(tr.RL_DQN_BATCH_SIZE, 86)        # Table 6.1 batch size
        self.assertEqual(tr.RL_DQN_FINAL_EPS, 0.01)       # Table 6.1 min epsilon

    def test_is_discrete_guards_box_agent(self):
        import train_rl_agents as tr
        import rl_env_alpha_entry as A, rl_env_capital_allocation as D
        self.assertTrue(tr._is_discrete(A.AlphaEntryEnv()))    # entry → Discrete (DQN-eligible)
        self.assertFalse(tr._is_discrete(D.CapitalAllocationEnv()))  # allocation → Box (stays PPO)

    def test_default_algo_table_unchanged(self):
        # All four agents still default to PPO so existing results/recommendations don't move.
        import train_rl_agents as tr
        for a in tr.AGENTS:
            self.assertEqual(tr.AGENTS[a][3], "PPO")


class BackfillPopulationTests(unittest.TestCase):
    def test_replay_buffett_emits_mtf(self):
        try:
            import numpy as np
            import pandas as pd
        except Exception:
            self.skipTest("pandas/numpy unavailable")
        import rl_alpaca_backfill as bf
        n = 300
        idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
        close = np.linspace(100, 160, n)                      # ndarray → no index alignment
        df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                           "close": close, "volume": 1e6}, index=idx)
        ev = bf.replay_buffett("TEST", df, step=20, lookback=200)
        self.assertTrue(ev)
        for k in rc.MTF_FEATURES:
            self.assertIn(k, ev[0])
        self.assertGreater(ev[0]["mtf_alignment"], 0.0)       # uptrend aligns up

    def test_trade_replay_csv_sets_catalyst_from_volume_spike(self):
        import csv as _csv
        import tempfile
        import rl_alpaca_backfill as bf
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "replay.csv"
            with open(p, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=["entry_price", "stop_price", "target_price",
                                                   "outcome", "mfe_pct", "mae_pct",
                                                   "volume_spike_ratio", "symbol", "timestamp"])
                w.writeheader()
                w.writerow({"entry_price": 100, "stop_price": 98, "target_price": 106,
                            "outcome": "win", "mfe_pct": 6, "mae_pct": 1,
                            "volume_spike_ratio": 3.0, "symbol": "ABC",
                            "timestamp": "2026-05-31T14:00:00Z"})
            ev = bf.ingest_trade_replay_csv(p)
            entries = [e for e in ev if e["event_type"] == "alpha_entry"]
            self.assertEqual(len(entries), 1)
            # vsr 3.0 → catalyst (3-1)/2 = 1.0
            self.assertAlmostEqual(entries[0]["catalyst_strength"], 1.0)


if __name__ == "__main__":
    unittest.main()
