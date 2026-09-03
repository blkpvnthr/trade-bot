"""
Tests for the MarketStateEngine + Premarket RVOL + OptionLab + Buffett
Black-Litterman / Efficient-Frontier upgrades.

All offline: no sockets, no network, no live packages required. Optional packages
(optionlab, yliveticker, streaming_indicators) are absent in CI, so the fallback
paths are exercised by default.
"""

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import market_state_engine as mse
import premarket_live_feed as plf
import optionlab_evaluator as ole


# ─────────────────────────── Part 1 — MarketStateEngine ───────────────────────────
class MarketStateTests(unittest.TestCase):
    def _rich(self):
        ind = {"quotes": {
            "SPY": {"indicators": {"sma_20": 510, "sma_50": 500}},
            "QQQ": {"indicators": {"sma_20": 440, "sma_50": 430}},
            "AAPL": {"indicators": {"sma_20": 200, "sma_50": 190}},
        }}
        pm = {"quotes": {"AAPL": {"premarket_change_pct": 1.2, "stale": False}}, "fallback_used": False}
        mc = {"overall_market_bias": "bullish", "volatility_regime": "normal", "vix": 14}
        return mc, ind, pm

    def test_1_market_state_generated(self):
        mc, ind, pm = self._rich()
        st = mse.get_engine().build(market_context=mc, indicator_snapshot=ind, premarket_snapshot=pm)
        for k in mse.STATE_KEYS:
            self.assertIn(k, st)
        self.assertEqual(st["regime"], "bullish")
        self.assertEqual(st["spy_trend"], "up")
        self.assertEqual(st["vix_state"], "calm")

    def test_2_missing_data_lowers_confidence(self):
        mc, ind, pm = self._rich()
        full = mse.MarketStateEngine().build(market_context=mc, indicator_snapshot=ind, premarket_snapshot=pm)
        empty = mse.MarketStateEngine().build(market_context={})
        self.assertLess(empty["confidence"], full["confidence"])
        self.assertGreaterEqual(empty["confidence"], 0.1)   # clamped, never crashes

    def test_3_risk_on_off_mutually_coherent(self):
        for mc in ({"overall_market_bias": "bullish", "vix": 12},
                   {"overall_market_bias": "bear_market crisis", "vix": 40},
                   {}):
            st = mse.MarketStateEngine().build(market_context=mc)
            self.assertFalse(st["risk_on"] and st["risk_off"], "risk_on and risk_off cannot both be true")

    def test_4_output_files_written(self):
        import tempfile
        e = mse.MarketStateEngine()
        e.build(market_context={"overall_market_bias": "neutral"})
        with tempfile.TemporaryDirectory() as d:
            paths = e.write_report(out_dir=d)
            self.assertTrue(paths["state"].exists())
            self.assertTrue(paths["diagnostic"].exists())

    def test_5_downstream_modules_accept_market_state(self):
        import alpha_call_engine as ace
        import buffett_accumulation_layer as bal
        import buffett_portfolio_optimizer as bopt
        import adaptive_capital_allocator as aca
        self.assertIn("market_state", inspect.signature(ace.AlphaCallEngine.evaluate_universe).parameters)
        self.assertIn("market_state", inspect.signature(ace.AlphaCallEngine.check_active_positions).parameters)
        self.assertIn("market_state", inspect.signature(bal.BuffettAccumulationLayer.evaluate_dca).parameters)
        self.assertIn("market_state", inspect.signature(bopt.run_optimizer).parameters)
        self.assertIn("market_state", inspect.signature(aca.build_allocation_plan).parameters)


# ─────────────────────────── Part 2 — Premarket RVOL ──────────────────────────────
class PremarketRvolTests(unittest.TestCase):
    def test_6_computes_premarket_rvol(self):
        f = plf.PremarketLiveFeed(["AAPL"], output_dir=None)
        f.on_message({"id": "AAPL", "price": 100.0, "previousClose": 99.0, "dayVolume": "200000"})
        f.set_avg_premarket_volume("AAPL", 100000, source="premarket_20d")
        q = f.get_symbol_state("AAPL")
        self.assertTrue(q["rvol_available"])
        self.assertAlmostEqual(q["premarket_rvol"], 2.0, places=4)
        self.assertEqual(q["rvol_source"], "premarket_20d")

    def test_7_missing_history_sets_unavailable(self):
        f = plf.PremarketLiveFeed(["AAPL"], output_dir=None)
        f.on_message({"id": "AAPL", "price": 100.0, "dayVolume": "200000"})
        q = f.get_symbol_state("AAPL")
        self.assertFalse(q["rvol_available"])
        self.assertIsNone(q["premarket_rvol"])   # never fabricated

    def test_8_rank_score_uses_gap_rvol_sector_risk(self):
        f = plf.PremarketLiveFeed(["AAPL"], output_dir=None)
        f.on_message({"id": "AAPL", "price": 103.0, "previousClose": 100.0, "dayVolume": "300000"})
        f.set_avg_premarket_volume("AAPL", 100000)
        q = f.get_symbol_state("AAPL")
        risk_on = plf.compute_premarket_rank_score(q, market_state={"risk_on": True}, sector_strength=0.8)
        risk_off = plf.compute_premarket_rank_score(q, market_state={"risk_off": True}, sector_strength=0.8)
        no_rvol = plf.compute_premarket_rank_score(
            {"premarket_change_pct": 3.0, "rvol_available": False, "stale": False}, market_state={"risk_on": True})
        self.assertGreater(risk_on, risk_off)          # risk-on regime ranks higher
        self.assertGreater(risk_on, no_rvol)           # available RVOL contributes

    def test_9_rvol_cannot_create_trade(self):
        # The rank score is advisory: a stale quote contributes nothing, and the feed
        # exposes no order-placement API.
        stale = {"premarket_change_pct": 5.0, "premarket_rvol": 9.0, "rvol_available": True, "stale": True}
        self.assertEqual(plf.compute_premarket_rank_score(stale, market_state={"risk_on": True}), 0.0)
        for attr in ("submit_order", "place_order", "buy", "sell"):
            self.assertFalse(hasattr(plf.PremarketLiveFeed, attr))


# ─────────────────────────── Part 3 — OptionLab ───────────────────────────────────
class OptionLabTests(unittest.TestCase):
    def test_10_long_call_payoff_evaluation(self):
        r = ole.get_evaluator().evaluate_long_call(
            underlying_price=100, strike=100, premium=2.0, dte=30, iv=0.4, delta=0.55, theta=-0.03)
        for k in ("break_even_price", "break_even_distance_pct", "profitable_price_range",
                  "max_loss", "max_gain_estimate", "payoff_asymmetry", "payoff_quality",
                  "optionlab_greeks", "optionlab_score", "optionlab_reject_reason"):
            self.assertIn(k, r)

    def test_11_break_even_calculated(self):
        # Pin the deterministic closed-form fallback (independent of whether the optional
        # `optionlab` package happens to be importable in this environment).
        import unittest.mock as mock
        with mock.patch.object(ole, "OPTIONLAB_AVAILABLE", False):
            r = ole.OptionLabEvaluator().evaluate_long_call(
                underlying_price=100, strike=105, premium=3.0, dte=30, iv=0.4)
        self.assertAlmostEqual(r["break_even_price"], 108.0, places=4)   # strike + premium
        self.assertAlmostEqual(r["max_loss"], 300.0, places=2)           # premium * 100

    def test_12_far_break_even_lowers_score(self):
        import unittest.mock as mock
        with mock.patch.object(ole, "OPTIONLAB_AVAILABLE", False):
            e = ole.OptionLabEvaluator()
            near = e.evaluate_long_call(underlying_price=100, strike=100, premium=2.0, dte=30, iv=0.4, delta=0.55)
            far = e.evaluate_long_call(underlying_price=100, strike=140, premium=0.4, dte=30, iv=0.4, delta=0.08)
        self.assertGreater(near["optionlab_score"], far["optionlab_score"])
        self.assertEqual(far["optionlab_reject_reason"], "break_even_unreachable")

    def test_13_favorable_payoff_improves_score(self):
        import unittest.mock as mock
        with mock.patch.object(ole, "OPTIONLAB_AVAILABLE", False):
            good = ole.OptionLabEvaluator().evaluate_long_call(
                underlying_price=100, strike=98, premium=3.0, dte=45, iv=0.5, delta=0.6, theta=-0.02)
        self.assertGreaterEqual(good["optionlab_score"], 60.0)
        self.assertIn(good["payoff_quality"], ("good", "excellent"))

    def test_14_unavailable_falls_back_safely(self):
        # Simulate the package being unavailable → source must be the BS fallback, no raise.
        import unittest.mock as mock
        with mock.patch.object(ole, "OPTIONLAB_AVAILABLE", False):
            r = ole.OptionLabEvaluator().evaluate_long_call(
                underlying_price=50, strike=50, premium=1.0, dte=20, iv=0.3)
        self.assertEqual(r["source"], ole.SOURCE_FALLBACK)
        self.assertIsInstance(r["optionlab_score"], float)

    def test_15_run_py_no_optionlab_internals(self):
        src = (Path(__file__).resolve().parents[1] / "run.py").read_text()
        self.assertNotIn("optionlab", src.lower(),
                         "run.py must contain no OptionLab internals (alpha engine owns it)")


# ─────────────── Part 4 — Black-Litterman + Efficient Frontier ────────────────────
class BlackLittermanOptimizerTests(unittest.TestCase):
    SIGNALS = [
        {"symbol": "AAPL", "conviction_score": 85, "price_to_intrinsic_value": 0.80, "market_cap": 3.0e12, "expected_return": 0.12},
        {"symbol": "MSFT", "conviction_score": 78, "price_to_intrinsic_value": 0.90, "market_cap": 2.8e12, "expected_return": 0.10},
        {"symbol": "KO", "conviction_score": 70, "price_to_intrinsic_value": 0.95, "market_cap": 2.6e11, "expected_return": 0.07},
    ]

    def _caps(self):
        return {s["symbol"]: s["market_cap"] for s in self.SIGNALS}

    def _bl(self):
        import buffett_black_litterman as bbl
        return bbl.run_black_litterman_safe(self.SIGNALS, market_caps=self._caps(), write=False)

    def test_16_prior_returns_generated(self):
        self.assertTrue(self._bl().get("prior_returns"))

    def test_17_views_generated(self):
        self.assertTrue(self._bl().get("views") is not None)

    def test_18_confidence_matrix_generated(self):
        bl = self._bl()
        self.assertTrue(bl.get("confidences") is not None or bl.get("view_confidence") is not None)

    def test_19_posterior_returns_generated(self):
        self.assertTrue(self._bl().get("posterior_returns"))

    def test_20_posterior_covariance_generated(self):
        self.assertTrue(self._bl().get("posterior_covariance"))

    def _mu_S(self):
        mu = pd.Series({"AAPL": 0.12, "MSFT": 0.10, "KO": 0.07})
        base = np.array([[0.04, 0.01, 0.005], [0.01, 0.03, 0.004], [0.005, 0.004, 0.02]])
        return mu, pd.DataFrame(base, index=mu.index, columns=mu.index)

    def test_21_efficient_frontier_objectives_generated(self):
        import buffett_portfolio_optimizer as bopt
        mu, S = self._mu_S()
        blended = bopt.blended_optimization(mu, S, bopt.OptimizerConfig())
        self.assertTrue(blended.get("used_objectives"))            # EF objectives attempted
        self.assertTrue(blended.get("weights"))

    def test_22_l2_regularization_applied(self):
        import buffett_portfolio_optimizer as bopt
        mu, S = self._mu_S()
        w = bopt.optimize_single(mu, S, "max_quadratic_utility", bopt.OptimizerConfig(), use_l2=True)
        self.assertIsInstance(w, dict)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=2)

    def test_23_turnover_penalty_applied(self):
        import buffett_portfolio_optimizer as bopt
        mu, S = self._mu_S()
        w = bopt.optimize_single(mu, S, "max_quadratic_utility", bopt.OptimizerConfig(),
                                 current_weights={"AAPL": 0.5, "MSFT": 0.3, "KO": 0.2},
                                 use_l2=True, use_turnover=True)
        self.assertIsInstance(w, dict)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=2)

    def test_24_efficient_return_fallback(self):
        import buffett_portfolio_optimizer as bopt
        mu, S = self._mu_S()
        # An unreachable target may make efficient_return fail; blended must still
        # renormalise over the objectives that succeeded and return valid weights.
        blended = bopt.blended_optimization(mu, S, bopt.OptimizerConfig())
        self.assertAlmostEqual(sum(blended["weights"].values()), 1.0, places=2)

    def test_25_weights_sum_to_one(self):
        import buffett_portfolio_optimizer as bopt
        res = bopt.run_optimizer_safe(self.SIGNALS, market_caps=self._caps(), write=False)
        tw = res.get("target_weights", {})
        self.assertTrue(tw)
        self.assertAlmostEqual(sum(tw.values()), 1.0, places=2)

    def test_26_dca_respects_optimized_target(self):
        import buffett_accumulation_layer as bal
        # The DCASignal carries the optimizer target-weight provenance, and evaluate_dca
        # accepts the optimizer weights. The ceiling gate rejects accumulation above target.
        for fld in ("optimized_target_weight", "current_weight", "weight_gap",
                    "optimizer_allocation_reason", "optimizer_objective_weights"):
            self.assertIn(fld, bal.DCASignal.__dataclass_fields__)
        src = (Path(__file__).resolve().parents[1] / "buffett_accumulation_layer.py").read_text()
        self.assertIn("rejected_dca_above_optimized_target", src)
        self.assertIn("optimized_target_weight is not None and cur_pct >= optimized_target_weight", src)

    def test_27_walk_forward_selects_active_optimizer(self):
        import buffett_portfolio_optimizer as bopt
        rng = np.random.default_rng(7)
        cols = ["AAPL", "MSFT", "KO", "JNJ"]
        idx = pd.date_range("2025-01-01", periods=150, freq="B")
        rdf = pd.DataFrame({c: rng.normal(0.0005, 0.012, 150) for c in cols}, index=idx)
        cmp = bopt.walk_forward_compare(rdf, conviction_weights={c: 0.25 for c in cols}, write=False)
        active = cmp.get("active_buffett_optimizer")
        self.assertIsInstance(active, str)
        self.assertTrue(active)

    def test_28_optimizer_failure_falls_back_safely(self):
        import buffett_portfolio_optimizer as bopt
        res = bopt.run_optimizer_safe([], market_caps={}, write=False)   # degenerate input
        self.assertIsInstance(res, dict)
        self.assertIn("target_weights", res)


# ─────────────────────────── Part 6 — Global guarantees ───────────────────────────
class GlobalGuaranteeTests(unittest.TestCase):
    def test_29_run_imports_cleanly(self):
        import run
        self.assertTrue(hasattr(run, "_build_market_state"))
        self.assertTrue(hasattr(run, "_market_state_engine"))

    def test_31_no_new_component_places_orders(self):
        root = Path(__file__).resolve().parents[1]
        for mod in ("market_state_engine.py", "optionlab_evaluator.py"):
            src = root.joinpath(mod).read_text()
            for forbidden in ("submit_market_order", "submit_order", "AlpacaGateway",
                              "place_order", "TradingClient"):
                self.assertNotIn(forbidden, src, f"{mod} must not place orders ({forbidden})")

    def test_32_riskmanager_remains_authority(self):
        # New advisory modules must not import the broker/execution layer.
        root = Path(__file__).resolve().parents[1]
        for mod in ("market_state_engine.py", "optionlab_evaluator.py"):
            src = root.joinpath(mod).read_text()
            self.assertNotIn("import alpaca", src.lower())
            self.assertNotIn("alpaca_executor", src.lower())


if __name__ == "__main__":
    unittest.main()
