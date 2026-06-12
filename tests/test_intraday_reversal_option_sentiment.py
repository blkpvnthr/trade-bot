"""Tests for the option-sentiment buy-gate (AV put/call ratio + vol/OI call-share).
Pure decision logic + the gated propose_option_entries hook. No network (sentiment is
injected); no orders placed."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intraday_reversal_option_sentiment import OptionSentiment, option_sentiment_ok
from intraday_reversal_engine import EngineConfig, ReversalInput, scan, propose_option_entries

CFG = EngineConfig()   # default thresholds: bull<=0.60, bear[1.0,1.5), contra-bull>=1.5, contra-bear<=0.20


class SentimentDecisionTests(unittest.TestCase):
    def test_bullish_pcr_allows(self):
        ok, why = option_sentiment_ok(OptionSentiment(put_call_ratio=0.45), CFG)
        self.assertTrue(ok)
        self.assertIn("bullish", why)

    def test_bearish_pcr_blocks(self):
        ok, why = option_sentiment_ok(OptionSentiment(put_call_ratio=1.2), CFG)
        self.assertFalse(ok)
        self.assertIn("pcr_bearish", why)

    def test_contrarian_bullish_extreme_high_allows(self):
        ok, why = option_sentiment_ok(OptionSentiment(put_call_ratio=1.8), CFG)
        self.assertTrue(ok)
        self.assertIn("contrarian_bullish", why)

    def test_contrarian_bearish_extreme_low_blocks(self):
        ok, why = option_sentiment_ok(OptionSentiment(put_call_ratio=0.15), CFG)
        self.assertFalse(ok)
        self.assertIn("contrarian_bearish", why)

    def test_neutral_allows(self):
        ok, _ = option_sentiment_ok(OptionSentiment(put_call_ratio=0.8), CFG)
        self.assertTrue(ok)

    def test_missing_data_fails_open(self):
        ok, why = option_sentiment_ok(OptionSentiment(put_call_ratio=None), CFG)
        self.assertTrue(ok)
        self.assertIn("unknown", why)

    def test_vol_oi_call_share_floor_blocks(self):
        cfg = EngineConfig(min_vol_oi_call_share=0.55)
        ok, why = option_sentiment_ok(OptionSentiment(put_call_ratio=0.5, vol_oi_call_share=0.30), cfg)
        self.assertFalse(ok)
        self.assertIn("vol_oi_call_share_low", why)
        ok2, _ = option_sentiment_ok(OptionSentiment(put_call_ratio=0.5, vol_oi_call_share=0.70), cfg)
        self.assertTrue(ok2)


def _eligible_scan():
    inp = ReversalInput(symbol="OPT", price=20.0, volume=8e6, avg_volume=3e6, chg_pct=+1.0,
                        perf_1w=-3.0, analyst_rating="strong buy", os_rating="strong buy", rvol=2.0,
                        vwap_reclaim=True, rsi_cross_40=True, close_gt_open=True, above_ema20_5m=True,
                        recovering=True, prev_day_high=22.0, atr=0.4,
                        option_chain=[{"expiry": "2026-06-12", "strike": 20.0, "type": "call",
                                       "bid": 0.80, "ask": 0.88, "dte": 1, "open_interest": 1500,
                                       "volume": 300}])
    return scan([inp], EngineConfig())


class GatedProposeTests(unittest.TestCase):
    def setUp(self):
        self.res = _eligible_scan()
        self.cfg = EngineConfig(option_sentiment_enabled=True, min_score=75.0,
                                max_new_entries_per_day=2, max_open_positions=2)

    def test_bearish_sentiment_blocks_call_buy(self):
        orders = propose_option_entries(self.res, 100_000.0, set(), self.cfg,
                                        sentiment_fn=lambda s: OptionSentiment(put_call_ratio=1.2))
        self.assertEqual(orders, [])

    def test_bullish_sentiment_allows_call_buy(self):
        orders = propose_option_entries(self.res, 100_000.0, set(), self.cfg,
                                        sentiment_fn=lambda s: OptionSentiment(put_call_ratio=0.5))
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].underlying, "OPT")

    def test_disabled_gate_does_not_call_sentiment(self):
        cfg = EngineConfig(option_sentiment_enabled=False, min_score=75.0,
                           max_new_entries_per_day=2, max_open_positions=2)

        def _boom(_sym):
            raise AssertionError("sentiment_fn must not be called when the gate is disabled")

        orders = propose_option_entries(self.res, 100_000.0, set(), cfg, sentiment_fn=_boom)
        self.assertEqual(len(orders), 1)


if __name__ == "__main__":
    unittest.main()
