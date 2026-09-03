"""
Tests for the Probability-Weighted Alpha Management Layer. Proves probability adapts
SIZE and MANAGEMENT but NEVER blocks a trade, never overrides RiskManager, and only
moves in the protective direction. Also covers the probability_weighting_report.
"""
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import alpha_probability_model as apm
import alpha_call_engine as ace
import portfolio_research_lab as prl


class ProbabilityModelTests(unittest.TestCase):
    CFG = apm.ProbabilityConfig()

    def test_high_probability_full_size_high_conviction(self):
        hist = {"ORB": {"trades": 30, "win_rate": 0.85, "avg_r_multiple": 1.6}}
        e = apm.evaluate("ORB", 90, "A+", expected_r_bs=2.5, bs_score=85, iv_crush_risk=5,
                         setup_history=hist, config=self.CFG)
        self.assertEqual(e["probability_band"], "80-100")
        self.assertEqual(e["size_multiplier"], 1.00)
        self.assertEqual(e["management_style"], "high_conviction")
        self.assertTrue(e["overnight_allowed"])          # BS agrees
        self.assertEqual(e["stop_tightness"], 1.00)      # normal stop

    def test_low_probability_reduces_size_but_never_zero(self):
        hist = {"PIVOT": {"trades": 25, "win_rate": 0.28, "avg_r_multiple": -0.6}}
        e = apm.evaluate("PIVOT", 45, "B", expected_r_bs=0.7, bs_score=40, iv_crush_risk=60,
                         setup_history=hist, config=self.CFG)
        self.assertEqual(e["probability_band"], "below-50")
        self.assertGreater(e["size_multiplier"], 0.0)     # NEVER zero (no block)
        self.assertLessEqual(e["size_multiplier"], 0.10)
        self.assertEqual(e["management_style"], "micro_exploration")
        self.assertFalse(e["overnight_allowed"])

    def test_adjustments_are_protective_only(self):
        # Across every band, size<=1, stop/trail tightness<=1 (never loosened).
        for p in (0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.20):
            m = apm.management_for_probability(p, self.CFG, bs_score=50, iv_crush_risk=30)
            self.assertLessEqual(m["size_multiplier"], self.CFG.max_size_multiplier)
            self.assertGreaterEqual(m["size_multiplier"], self.CFG.min_size_multiplier)
            self.assertLessEqual(m["stop_tightness"], 1.0)
            self.assertLessEqual(m["trail_tightness_mult"], 1.0)

    def test_bands_are_monotonic_in_size(self):
        sizes = [apm.management_for_probability(p, self.CFG)["size_multiplier"]
                 for p in (0.85, 0.75, 0.65, 0.55, 0.45)]
        self.assertEqual(sizes, sorted(sizes, reverse=True))   # higher prob → >= size

    def test_overnight_requires_bs_agreement(self):
        good = apm.management_for_probability(0.90, self.CFG, bs_score=85, iv_crush_risk=5)
        bad = apm.management_for_probability(0.90, self.CFG, bs_score=30, iv_crush_risk=80)
        self.assertTrue(good["overnight_allowed"])
        self.assertFalse(bad["overnight_allowed"])           # BS disagrees → no overnight

    def test_disabled_exploration_still_never_blocks(self):
        cfg = apm.ProbabilityConfig(exploration_enabled=False)
        m = apm.management_for_probability(0.20, cfg)
        self.assertGreater(m["size_multiplier"], 0.0)        # executes at min, not blocked
        self.assertEqual(m["size_multiplier"], cfg.min_size_multiplier)


class ProbabilityEngineIntegrationTests(unittest.TestCase):
    def _engine(self, history=None):
        eng = ace.AlphaCallEngine(broker=types.SimpleNamespace(), config=ace.AlphaCallConfig())
        eng._active = {}
        return eng

    def test_register_filled_only_tightens_management(self):
        eng = self._engine()
        base = eng.config
        # A high-probability signal: normal management (no tightening).
        sig_hi = ace.CallSignal(symbol="HI", setup_type="ORB", setup_score=90, grade="A+")
        sig_hi.contracts = 2; sig_hi.mid = 4.0; sig_hi.premium_cost = 800.0
        sig_hi.stop_loss_pct_eff = base.stop_loss_pct
        sig_hi.partial_profit_pct_eff = base.partial_profit_pct
        sig_hi.trail_tightness_mult = 1.0
        eng.register_filled(sig_hi)
        hi = eng._active["HI"]
        # A low-probability signal: tighter stop + earlier partial + tighter trail.
        sig_lo = ace.CallSignal(symbol="LO", setup_type="PIVOT", setup_score=45, grade="B")
        sig_lo.contracts = 1; sig_lo.mid = 4.0; sig_lo.premium_cost = 400.0
        sig_lo.stop_loss_pct_eff = base.stop_loss_pct * 0.7
        sig_lo.partial_profit_pct_eff = 0.25
        sig_lo.trail_tightness_mult = 0.4
        eng.register_filled(sig_lo)
        lo = eng._active["LO"]
        # Tighter stop ⇒ hard_stop_value CLOSER to entry (exits a smaller loss = protective).
        self.assertGreater(lo.hard_stop_value / lo.entry_cost, hi.hard_stop_value / hi.entry_cost)
        # Earlier partial ⇒ lower partial_value multiple.
        self.assertLess(lo.partial_value / lo.entry_cost, hi.partial_value / hi.entry_cost)
        # Tighter trail multiplier carried through.
        self.assertLess(lo.trail_tightness_mult, hi.trail_tightness_mult)
        # 3R target NEVER weakened by probability.
        self.assertAlmostEqual(lo.r_target_value / lo.entry_cost, hi.r_target_value / hi.entry_cost, places=6)

    def test_probability_never_blocks_or_bypasses_risk_manager(self):
        # The probability module is advisory only: no order/submit/gate entry points.
        callables = [n for n in dir(apm) if not n.startswith("_") and callable(getattr(apm, n))]
        for n in callables:
            self.assertNotIn("submit", n.lower())
            self.assertNotIn("order", n.lower())
        src = Path(apm.__file__).read_text()
        self.assertNotIn("AlpacaGateway", src)
        self.assertNotIn("submit_market_order", src)


class ProbabilityReportTests(unittest.TestCase):
    def test_probability_weighting_report_generated(self):
        events = [
            {"event_type": "alpha_entry", "symbol": "AAA", "probability_band": "80-100",
             "size_multiplier": 1.0, "management_style": "high_conviction"},
            {"event_type": "alpha_exit", "symbol": "AAA", "probability_band": "80-100",
             "probability_of_profit": 0.85, "pnl": 300, "r_multiple": 3.0,
             "probability_accurate": True, "timestamp": "2026-05-01T14:00:00+00:00"},
            {"event_type": "alpha_exit", "symbol": "BBB", "probability_band": "below-50",
             "probability_of_profit": 0.30, "pnl": -100, "r_multiple": -1.0,
             "probability_accurate": True, "timestamp": "2026-05-02T14:00:00+00:00"},
        ]
        pa = prl.probability_analytics(events)
        self.assertIn("80-100", pa["by_band"])
        self.assertIn("below-50", pa["band_accuracy"])
        self.assertEqual(pa["overall_accuracy"], 1.0)      # both estimates correct
        with TemporaryDirectory() as tmp:
            with patch.object(prl, "DAILY_OUTPUTS", Path(tmp)):
                prl.generate_all_reports(events=events, start_equity=100_000.0)
                self.assertTrue((prl.research_dir() / prl.PROBABILITY_WEIGHTING_REPORT).exists())
                disk = json.loads((prl.research_dir() / prl.PROBABILITY_WEIGHTING_REPORT).read_text())
                self.assertIn("probability_weighting", disk)

    def test_exit_recorder_computes_accuracy(self):
        # A predicted-likely trade that won → accurate True.
        e = prl.record_alpha_exit({"underlying": "AAA", "probability_band": "80-100",
                                   "probability_of_profit": 0.85, "realized_pnl": 200,
                                   "r_multiple": 2.0, "source": "alpha_call_3R_target"},
                                  date_tag="__probtest__")
        self.assertTrue(e["probability_accurate"])
        self.assertEqual(e["realized_R"], 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
