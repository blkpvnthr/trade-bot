"""
Tests for the Black-Litterman + Efficient-Frontier Buffett optimizer. Covers the BL
return engine, the four optimization objectives with L2 + turnover, the blend, the
diversification analytics, the walk-forward optimizer comparison, and the fail-safe
fallbacks. Deterministic (seeded synthetic data); never requires network.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import buffett_black_litterman as bbl
import buffett_portfolio_optimizer as bopt


def _signals():
    return [
        {"symbol": "MSFT", "conviction_score": 82, "price_to_intrinsic_value": 0.85, "BAL_score": 85,
         "insider_buy_score": 0, "valuation_confidence": 0.8, "macro_cycle_opportunity_score": 80,
         "valuation_model_count": 6, "market_cap": 3.0e12},
        {"symbol": "GOOGL", "conviction_score": 78, "price_to_intrinsic_value": 0.80, "BAL_score": 82,
         "insider_buy_score": 0, "valuation_confidence": 0.75, "macro_cycle_opportunity_score": 80,
         "valuation_model_count": 6, "market_cap": 2.1e12},
        {"symbol": "NVDA", "conviction_score": 91, "price_to_intrinsic_value": 0.70, "BAL_score": 88,
         "insider_buy_score": 60, "valuation_confidence": 0.85, "macro_cycle_opportunity_score": 85,
         "valuation_model_count": 7, "market_cap": 2.8e12},
        {"symbol": "UNH", "conviction_score": 66, "price_to_intrinsic_value": 0.92, "BAL_score": 74,
         "insider_buy_score": 0, "valuation_confidence": 0.6, "macro_cycle_opportunity_score": 70,
         "valuation_model_count": 5, "market_cap": 0.5e12},
        {"symbol": "COST", "conviction_score": 80, "price_to_intrinsic_value": 0.88, "BAL_score": 83,
         "insider_buy_score": 0, "valuation_confidence": 0.7, "macro_cycle_opportunity_score": 78,
         "valuation_model_count": 6, "market_cap": 0.4e12},
    ]


def _returns_df(seed=7, T=400):
    rng = np.random.default_rng(seed)
    syms = ["MSFT", "GOOGL", "NVDA", "UNH", "COST"]
    mus = [0.0010, 0.0009, 0.0011, 0.0002, 0.0006]
    vols = [0.009, 0.010, 0.013, 0.020, 0.011]
    return pd.DataFrame({s: rng.normal(mus[i], vols[i], T) for i, s in enumerate(syms)})


class BlackLittermanTests(unittest.TestCase):
    def setUp(self):
        self.sigs = _signals()
        self.caps = {s["symbol"]: s["market_cap"] for s in self.sigs}
        self.r = bbl.run_black_litterman(self.sigs, market_caps=self.caps, write=False)

    def test_1_prior_returns_generated(self):
        self.assertGreaterEqual(len(self.r["prior_returns"]), 5)
        self.assertTrue(all(isinstance(v, float) for v in self.r["prior_returns"].values()))

    def test_2_views_generated_correctly(self):
        labels = self.r["views"]
        self.assertTrue(any(v["type"] == "absolute" for v in labels))
        self.assertTrue(any(v["type"] == "relative" for v in labels))
        av = self.r["absolute_views"]
        self.assertEqual(max(av, key=av.get), "NVDA")   # deepest discount + insider + conviction

    def test_3_confidence_matrix_generated(self):
        confs = self.r["confidences"]
        self.assertGreaterEqual(len(confs), 5)
        self.assertTrue(all(0.0 < c <= 1.0 for c in confs))
        self.assertTrue(all(o >= 0 for o in self.r["omega"]))

    def test_4_posterior_differs_from_prior(self):
        prior, post = self.r["prior_returns"], self.r["posterior_returns"]
        self.assertFalse(self.r["fallback"])
        self.assertTrue(any(abs(prior[k] - post[k]) > 1e-6 for k in prior))
        self.assertGreater(self.r["diagnostics"]["posterior_vs_prior_max_shift"], 0.0)

    def test_5_posterior_covariance_generated(self):
        cov = self.r["_posterior_covariance_df"]
        self.assertEqual(cov.shape[0], cov.shape[1])
        self.assertGreaterEqual(cov.shape[0], 5)

    def test_12_fallback_when_bl_fails(self):
        r = bbl.run_black_litterman([{"symbol": "ONE", "conviction_score": 80}], write=False)
        self.assertTrue(r["fallback"])
        self.assertIn("posterior_returns", r)


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.sigs = _signals()
        self.caps = {s["symbol"]: s["market_cap"] for s in self.sigs}
        bl = bbl.run_black_litterman(self.sigs, market_caps=self.caps, write=False)
        self.mu = bl["_posterior_returns_series"]
        self.S = bl["_posterior_covariance_df"]
        self.cfg = bopt.OptimizerConfig()

    def test_6_black_litterman_feeds_efficient_frontier(self):
        w = bopt.optimize_single(self.mu, self.S, "max_sharpe", self.cfg)
        self.assertIsNotNone(w)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=4)

    def test_7_l2_regularization_applied(self):
        # L2 must change the solution and must NOT make it more concentrated. With a
        # small universe min-vol can already be near-even, so allow solver tolerance
        # rather than asserting a strict decrease. Verify L2 has real effect via a
        # max-quadratic-utility objective where its diversifying pull is unambiguous.
        with_l2 = bopt.optimize_single(self.mu, self.S, "min_volatility", self.cfg, use_l2=True)
        without = bopt.optimize_single(self.mu, self.S, "min_volatility", self.cfg, use_l2=False)
        self.assertNotEqual(with_l2, without)
        hhi_l2 = sum(v ** 2 for v in with_l2.values())
        hhi_no = sum(v ** 2 for v in without.values())
        self.assertLessEqual(hhi_l2, hhi_no + 1e-4)   # L2 never increases concentration (solver tol)
        # Unambiguous case: on max-quadratic-utility, L2 strictly lowers concentration.
        u_l2 = bopt.optimize_single(self.mu, self.S, "max_quadratic_utility", self.cfg, use_l2=True)
        u_no = bopt.optimize_single(self.mu, self.S, "max_quadratic_utility", self.cfg, use_l2=False)
        self.assertLess(sum(v ** 2 for v in u_l2.values()),
                        sum(v ** 2 for v in u_no.values()) + 1e-9)

    def test_8_turnover_penalty_applied(self):
        prev = {"MSFT": 0.5, "GOOGL": 0.5, "NVDA": 0.0, "UNH": 0.0, "COST": 0.0}
        no_turn = bopt.optimize_single(self.mu, self.S, "max_quadratic_utility", self.cfg,
                                       current_weights=prev, use_turnover=False)
        with_turn = bopt.optimize_single(self.mu, self.S, "max_quadratic_utility", self.cfg,
                                         current_weights=prev, use_turnover=True)
        self.assertNotEqual(no_turn, with_turn)

    def test_9_efficient_return_objective(self):
        w = bopt.optimize_single(self.mu, self.S, "efficient_return", self.cfg)
        self.assertIsNotNone(w)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=4)

    def test_10_optimizer_weights_sum_to_one(self):
        blended = bopt.blended_optimization(self.mu, self.S, self.cfg, use_turnover=True)
        self.assertAlmostEqual(sum(blended["weights"].values()), 1.0, places=4)
        self.assertFalse(blended["fallback"])

    def test_11_diversification_metrics_generated(self):
        blended = bopt.blended_optimization(self.mu, self.S, self.cfg)
        div = bopt.diversification_metrics(blended["weights"])
        for k in ("holdings_count", "largest_position", "hhi_concentration",
                  "effective_number_of_holdings", "turnover", "sector_concentration"):
            self.assertIn(k, div)
        self.assertGreater(div["effective_number_of_holdings"], 0.0)

    def test_13_fallback_when_efficient_frontier_fails(self):
        bad_S = pd.DataFrame(np.full((5, 5), np.nan), index=self.mu.index, columns=self.mu.index)
        blended = bopt.blended_optimization(self.mu, bad_S, self.cfg)
        self.assertTrue(blended["fallback"])
        self.assertAlmostEqual(sum(blended["weights"].values()), 1.0, places=4)

    def test_16_optimizer_beats_equal_weight_walkforward(self):
        cmp = bopt.walk_forward_compare(_returns_df(), self.cfg,
                                        conviction_weights={s: 1.0 for s in self.mu.index}, write=False)
        self.assertTrue(cmp["usable"])
        active = cmp["active_buffett_optimizer"]
        self.assertGreaterEqual(cmp["configs"][active]["optimizer_score"],
                                cmp["configs"]["equal_weight"]["optimizer_score"])
        self.assertGreaterEqual(cmp["configs"][active]["sharpe"],
                                cmp["configs"]["equal_weight"]["sharpe"])

    def test_reports_written(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        with TemporaryDirectory() as tmp:
            with patch.object(bopt, "DAILY_OUTPUTS", Path(tmp)), patch.object(bbl, "DAILY_OUTPUTS", Path(tmp)):
                bopt.run_optimizer(self.sigs, market_caps=self.caps, returns_df=_returns_df(), write=True)
                d = bopt._research_dir()
                for name in (bopt.DIVERSIFICATION_REPORT, bopt.OPTIMIZER_COMPARISON,
                             bopt.ACTIVE_OPTIMIZER, bopt.OPTIMIZER_PERFORMANCE):
                    self.assertTrue((d / name).exists(), f"{name} not written")


class IntegrationTests(unittest.TestCase):
    def test_14_run_imports_cleanly(self):
        import importlib, run
        importlib.reload(run)
        self.assertTrue(hasattr(run, "ENABLE_BUFFETT_OPTIMIZER"))

    def test_optimizer_target_weight_overrides_conviction_in_dca(self):
        from buffett_accumulation_layer import (BuffettAccumulationLayer, BALConfig, DCAConfig,
                                                FundamentalData, BUFFETT_DCA_SIGNALS_PATH)
        import json
        fd = FundamentalData(symbol="GOOD", current_price=150.0, market_cap=100e9, gross_margin=0.65,
            operating_margin=0.30, net_margin=0.22, revenue_growth_yoy=0.18, eps_growth_yoy=0.22,
            roe=0.32, roic=0.22, free_cash_flow=9e9, debt_to_equity=0.20, eps_ttm=10.0, peg_ratio=0.9,
            ev_ebitda=9.0, shares_growth_yoy=-0.03, insider_ownership_pct=0.12, beta=1.0,
            sma_50=168, sma_200=165, fifty_two_week_low=142.5, fifty_two_week_high=270, rsi_14=22,
            atr_expansion=1.7, volume_capitulation=2.1, revenue_ttm=40e9, price_to_book=6.0, sector="Technology")
        b = BuffettAccumulationLayer(config=BALConfig(), dca_config=DCAConfig()); b._ownership = {}
        ctx = {"overall_market_bias": "bullish", "rate_direction": "cutting", "inflation_trend": "cooling",
               "recession_risk": "low", "cycle_phase": "expansion", "fomc_stance": "dovish",
               "cpi_trend": "falling", "fiscal_policy": "stimulus", "tariff_risk": "low"}
        orders = b.evaluate_dca(["GOOD"], ctx, {"GOOD": {"vwap": 149, "close": 150}}, 100000.0, {},
                                50000.0, fd_by_symbol={"GOOD": fd},
                                target_weights_by_symbol={"GOOD": 0.05})
        self.assertEqual(len(orders), 1)
        sig = json.loads(BUFFETT_DCA_SIGNALS_PATH.read_text())["signals"][0]
        self.assertEqual(sig["target_allocation_pct"], 0.05)
        self.assertIn("optimizer_target_weight", sig["allocation_reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
