"""Tests for the fundamental gate (fundamentals IN ADDITION to the technical gates).
Pure decision + the gated score_candidate hook + injected-fetch enrichment. No network."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intraday_reversal_fundamentals import fundamentals_ok, maybe_enrich_fundamentals
from intraday_reversal_engine import EngineConfig, ReversalInput, score_candidate

CFG = EngineConfig()        # gate off, lenient thresholds


def _cand(**kw) -> ReversalInput:
    d = dict(symbol="FUN", price=20.0, volume=8_000_000, avg_volume=3_000_000, chg_pct=+1.0,
             perf_1w=-3.0, analyst_rating="strong buy", os_rating="strong buy", rvol=2.0,
             vwap_reclaim=True, rsi_cross_40=True, close_gt_open=True, above_ema20_5m=True,
             recovering=True, prev_day_high=22.0, atr=0.4)
    d.update(kw)
    return ReversalInput(**d)


class FundamentalsDecisionTests(unittest.TestCase):
    def test_missing_fundamentals_fails_open(self):
        ok, why = fundamentals_ok(None, CFG)
        self.assertTrue(ok)
        self.assertEqual(why, "no_fundamentals")

    def test_weak_revenue_growth_blocks(self):
        ok, why = fundamentals_ok({"revenue_growth_yoy": -0.08}, CFG)   # default min 0.0
        self.assertFalse(ok)
        self.assertIn("weak_revenue_growth", why)

    def test_healthy_fundamentals_pass(self):
        self.assertTrue(fundamentals_ok({"revenue_growth_yoy": 0.18, "net_margin": 0.12}, CFG)[0])

    def test_negative_margin_blocks_when_required(self):
        cfg = EngineConfig(require_positive_net_margin=True)
        self.assertFalse(fundamentals_ok({"revenue_growth_yoy": 0.1, "net_margin": -0.05}, cfg)[0])

    def test_leverage_and_pe_gates(self):
        cfg = EngineConfig(max_debt_to_equity=1.0, max_pe_ratio=50.0)
        self.assertFalse(fundamentals_ok({"revenue_growth_yoy": 0.1, "debt_to_equity": 2.5}, cfg)[0])
        self.assertFalse(fundamentals_ok({"revenue_growth_yoy": 0.1, "pe_ratio": 80.0}, cfg)[0])
        self.assertTrue(fundamentals_ok({"revenue_growth_yoy": 0.1, "debt_to_equity": 0.4,
                                         "pe_ratio": 22.0}, cfg)[0])


class GatedScoreTests(unittest.TestCase):
    def test_gate_off_ignores_fundamentals(self):
        c = score_candidate(_cand(fundamentals={"revenue_growth_yoy": -0.5}), CFG)
        self.assertTrue(c.passed_universe)          # gate off -> weak fundamentals ignored

    def test_gate_on_blocks_weak_fundamentals(self):
        cfg = EngineConfig(fundamental_gate_enabled=True)
        c = score_candidate(_cand(fundamentals={"revenue_growth_yoy": -0.20}), cfg)
        self.assertFalse(c.passed_universe)
        self.assertTrue(c.blocked_by.startswith("fundamental:weak_revenue_growth"))

    def test_gate_on_allows_healthy_and_fails_open_on_missing(self):
        cfg = EngineConfig(fundamental_gate_enabled=True)
        self.assertTrue(score_candidate(_cand(fundamentals={"revenue_growth_yoy": 0.2}), cfg).passed_universe)
        self.assertTrue(score_candidate(_cand(fundamentals=None), cfg).passed_universe)   # fail-open


class EnrichFundamentalsTests(unittest.TestCase):
    def test_enrich_disabled_is_noop(self):
        rows = [_cand()]
        maybe_enrich_fundamentals(rows, CFG, fetch_fn=lambda s: {"revenue_growth_yoy": 0.1})
        self.assertIsNone(rows[0].fundamentals)

    def test_enrich_attaches_when_gate_on(self):
        cfg = EngineConfig(fundamental_gate_enabled=True)
        rows = [_cand()]
        maybe_enrich_fundamentals(rows, cfg, fetch_fn=lambda s: {"revenue_growth_yoy": 0.22})
        self.assertEqual(rows[0].fundamentals["revenue_growth_yoy"], 0.22)


if __name__ == "__main__":
    unittest.main()
