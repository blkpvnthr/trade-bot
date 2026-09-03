"""
Tests for the Adaptive Capital Allocation Engine. Proves it consumes research outputs,
allocates more to proven performers / less to losers, dampens low-sample evidence, and
NEVER exceeds configured caps or bypasses RiskManager (advisory only).
"""
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import adaptive_capital_allocator as aca
import portfolio_research_lab as prl


def _alpha_group(trades, expectancy, pf, r, sharpe, dd, wr=0.6):
    return {"trades": trades, "win_rate": wr, "expectancy": expectancy, "profit_factor": pf,
            "avg_r_multiple": r, "sharpe": sharpe, "max_drawdown": dd}


RESEARCH = {
    "alpha": {
        "totals": _alpha_group(20, 80, 2.5, 1.2, 1.8, -0.08),
        "by_setup_type": {
            "ORB": _alpha_group(15, 100, 3.0, 1.4, 2.0, -0.05),
            "PIVOT_BREAK": _alpha_group(8, -40, 0.6, -0.6, -0.5, -0.30, wr=0.3),
            "RARE": _alpha_group(1, 500, 9.0, 3.0, 0.0, 0.0, wr=1.0),
        },
        "by_sector": {"Technology": _alpha_group(12, 90, 2.8, 1.3, 1.9, -0.06)},
        "by_volatility_regime": {"low": _alpha_group(10, 120, 3.5, 1.6, 2.2, -0.04)},
    },
    "buffett": {
        "totals": {"trades": 6, "win_rate": 0.83, "expectancy": 0.12, "profit_factor": 4.0,
                   "avg_r_multiple": 0.0, "sharpe": 1.5, "max_drawdown": -0.05},
        "by_conviction_bucket": {
            "90-100": {"trades": 5, "expectancy": 0.18, "profit_factor": 6.0, "win_rate": 1.0,
                       "avg_r_multiple": 0.0, "sharpe": 2.0, "max_drawdown": 0.0, "average_return": 0.18},
            "60-74": {"trades": 4, "expectancy": 0.02, "profit_factor": 1.1, "win_rate": 0.5,
                      "avg_r_multiple": 0.0, "sharpe": 0.2, "max_drawdown": -0.12, "average_return": 0.02},
        },
        "by_valuation_bucket": {"65-75%": {"trades": 5, "expectancy": 0.2, "profit_factor": 7.0,
                                "win_rate": 1.0, "avg_r_multiple": 0.0, "sharpe": 2.5, "max_drawdown": 0.0}},
        "by_insider_bucket": {"high(70-100)": {"trades": 3, "expectancy": 0.15, "profit_factor": 5.0,
                              "win_rate": 1.0, "avg_r_multiple": 0.0, "sharpe": 1.0, "max_drawdown": 0.0}},
        "by_tranche": {"panic": {"trades": 2, "expectancy": 0.22, "profit_factor": 8.0, "win_rate": 1.0,
                       "avg_r_multiple": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}},
    },
    "capital_waterfall": {"alpha_generated": 5000.0},
    "source_files": ["research_lab.json"],
}


class AdaptiveAllocatorTests(unittest.TestCase):
    CFG = aca.AllocatorConfig()

    def _plan(self, mc=None):
        return aca.build_allocation_plan(RESEARCH, market_context=mc or {}, config=self.CFG, write=False)

    def test_more_capital_to_proven_winners_less_to_losers(self):
        plan = self._plan()
        orb = plan["setup_allocations"]["ORB"]
        pivot = plan["setup_allocations"]["PIVOT_BREAK"]
        self.assertGreater(orb["capital_score"], pivot["capital_score"])
        self.assertGreater(orb["multiplier"], pivot["multiplier"])
        self.assertGreater(orb["recommended_risk_pct"], pivot["recommended_risk_pct"])
        # winner scales up (mult>1), loser scales down (mult<1)
        self.assertGreater(orb["multiplier"], 1.0)
        self.assertLess(pivot["multiplier"], 1.0)

    def test_low_sample_evidence_is_dampened(self):
        plan = self._plan()
        rare = plan["setup_allocations"]["RARE"]   # 1 trade, extreme numbers
        # Dampened toward neutral: score near 50, multiplier near 1.0 (not max).
        self.assertLess(rare["capital_score"], 70.0)
        self.assertLess(rare["multiplier"], 1.25)

    def test_never_exceeds_configured_caps(self):
        # Even with absurd inputs every recommendation stays within the hard caps.
        extreme = {
            "alpha": {"totals": _alpha_group(1000, 1e9, 999, 50, 99, 0.0),
                      "by_setup_type": {"X": _alpha_group(1000, 1e9, 999, 50, 99, 0.0)},
                      "by_sector": {}, "by_volatility_regime": {}},
            "buffett": {"totals": {"trades": 1000, "expectancy": 9e9, "profit_factor": 999,
                                   "win_rate": 1.0, "avg_r_multiple": 50, "sharpe": 99, "max_drawdown": 0.0},
                        "by_conviction_bucket": {"90-100": {"trades": 1000, "expectancy": 9e9,
                        "profit_factor": 999, "win_rate": 1.0, "avg_r_multiple": 50, "sharpe": 99, "max_drawdown": 0.0}},
                        "by_valuation_bucket": {}, "by_insider_bucket": {}, "by_tranche": {}},
            "capital_waterfall": {"alpha_generated": 1e7},
        }
        plan = aca.build_allocation_plan(extreme, market_context={}, config=self.CFG, write=False)
        for v in plan["setup_allocations"].values():
            self.assertLessEqual(v["recommended_risk_pct"], self.CFG.alpha_risk_cap_pct + 1e-12)
            self.assertLessEqual(v["multiplier"], self.CFG.alpha_max_mult + 1e-9)
        for v in plan["conviction_allocations"].values():
            self.assertLessEqual(v["recommended_target_pct"], self.CFG.max_single_stock_weight + 1e-12)

    def test_transfer_recommendation_within_bounds(self):
        for bias in ("bullish", "bearish", "neutral", ""):
            plan = self._plan({"overall_market_bias": bias})
            t = plan["transfer_recommendation"]
            self.assertGreaterEqual(t["recommended_pct"], self.CFG.transfer_min)
            self.assertLessEqual(t["recommended_pct"], self.CFG.transfer_max)
        # Bearish/value regime transfers more to BAL than bullish/momentum.
        bear = self._plan({"overall_market_bias": "bearish"})["transfer_recommendation"]["recommended_pct"]
        bull = self._plan({"overall_market_bias": "bullish"})["transfer_recommendation"]["recommended_pct"]
        self.assertGreater(bear, bull)
        # transfer amount = pct * realized alpha profit
        plan = self._plan()
        self.assertAlmostEqual(plan["transfer_amount"],
                               plan["transfer_recommendation"]["recommended_pct"] * 5000.0, places=2)

    def test_consumes_research_outputs(self):
        with TemporaryDirectory() as tmp:
            with patch.object(aca, "DAILY_OUTPUTS", Path(tmp)), patch.object(prl, "DAILY_OUTPUTS", Path(tmp)):
                # Research lab writes reports; allocator must consume them.
                prl.generate_all_reports(events=[
                    {"event_type": "alpha_exit", "symbol": "AAA", "setup_type": "ORB",
                     "pnl": 300, "r_multiple": 3.0, "timestamp": "2026-05-01T14:00:00+00:00"},
                    {"event_type": "alpha_exit", "symbol": "AAA", "setup_type": "ORB",
                     "pnl": -100, "r_multiple": -1.0, "timestamp": "2026-05-02T14:00:00+00:00"},
                ], start_equity=100_000.0)
                research = aca.load_research()
                self.assertIn("research_lab.json", research["source_files"])
                self.assertIn("ORB", research["alpha"].get("by_setup_type", {}))
                plan = aca.build_allocation_plan(market_context={"overall_market_bias": "bullish"})
                self.assertTrue((aca._research_dir() / aca.CAPITAL_ALLOCATION_PLAN).exists())
                self.assertTrue((aca._research_dir() / aca.ALLOCATION_DIAGNOSTIC).exists())
                self.assertIn("ORB", plan["setup_allocations"])

    def test_never_bypasses_risk_manager(self):
        # The allocator must be advisory: no order/submit/gate/broker entry points.
        callables = [n for n in dir(aca) if not n.startswith("_") and callable(getattr(aca, n))]
        for n in callables:
            self.assertNotIn("submit", n.lower())
            self.assertNotIn("order", n.lower())
            self.assertNotIn("gate", n.lower())
        src = Path(aca.__file__).read_text()
        self.assertNotIn("AlpacaGateway", src)
        self.assertNotIn("submit_market_order", src)
        # consumption helpers also respect caps as a hard ceiling
        plan = self._plan()
        self.assertLessEqual(aca.recommended_alpha_risk_pct("ORB", plan, self.CFG), self.CFG.alpha_risk_cap_pct)
        self.assertLessEqual(aca.recommended_bal_target_pct("90-100", plan, self.CFG),
                             self.CFG.max_single_stock_weight)

    def test_safe_wrapper_never_raises(self):
        try:
            self.assertIsNotNone(aca.build_allocation_plan_safe(RESEARCH, market_context={}, write=False))
            # Garbage input must not raise.
            aca.build_allocation_plan_safe({"alpha": None, "buffett": None}, write=False)
        except Exception as exc:
            self.fail(f"allocator safe wrapper raised: {exc}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
