"""
Tests for the OOPS-inspired Options Utility Optimization Layer. Proves the simulation /
utility math, that OOPS improves contract selection within liquidity limits, that it is
advisory only (no orders, never overrides the hard stop or RiskManager), and that it
falls back to Black-Scholes selection when simulation data is missing.
"""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import oops_option_optimizer as oops
import alpha_call_engine as ace


def _contract(strike, mid, spread=0.03, oi=800, vol=60, iv=0.5, theta=0.02, sym=None):
    return {"strike": strike, "mid": mid, "dte": 30, "implied_volatility": iv,
            "open_interest": oi, "volume": vol, "spread_pct": spread,
            "theta_pct_per_day": theta, "black_scholes_score": 60.0,
            "contract_symbol": sym or f"X{strike}", "delta": 0.55, "expiry": "2026-07-03",
            "bid": mid * 0.99, "ask": mid * 1.01}


CFG = oops.OOPSConfig(monte_carlo_paths=20000)


class OOPSModelTests(unittest.TestCase):

    # 1 ─────────────────────────────────────────────────────────────────────────
    def test_simulated_call_payoff_is_correct(self):
        payoffs = oops.simulate_option_payoffs(np.array([90.0, 100.0, 110.0, 125.0]), 100.0)
        self.assertEqual(payoffs.tolist(), [0.0, 0.0, 10.0, 25.0])

    # 2 ─────────────────────────────────────────────────────────────────────────
    def test_transaction_costs_reduce_expected_return(self):
        terminal = oops.simulate_underlying_returns(100.0, 30, 0.5, CFG)
        payoffs = oops.simulate_option_payoffs(terminal, 100.0)
        no_cost = oops.compute_option_return_distribution(
            payoffs, 4.0, 0.0, oops.OOPSConfig(transaction_cost_bps=0, slippage_bps=0))
        with_cost = oops.compute_option_return_distribution(
            payoffs, 4.0, 0.0, oops.OOPSConfig(transaction_cost_bps=80, slippage_bps=40))
        self.assertLess(with_cost["expected_return"], no_cost["expected_return"])

    # 3 ─────────────────────────────────────────────────────────────────────────
    def test_high_total_loss_probability_reduces_utility_score(self):
        contract = _contract(100, 4.0)
        good_dist = {"expected_return": 0.2, "median_return": 0.0, "probability_profit": 0.55,
                     "probability_total_loss": 0.30, "skew": 1.0, "kurtosis": 2.0,
                     "downside_tail_risk": -0.8, "returns": np.array([0.1, -0.1, 0.3])}
        bad_dist = dict(good_dist, probability_total_loss=0.90, probability_profit=0.15)
        util = {"expected_utility": -0.2, "certainty_equivalent": 0.99, "certainty_equivalent_return": 0.05}
        good = oops.score_oops_contract(good_dist, util, contract, CFG)
        bad = oops.score_oops_contract(bad_dist, util, contract, CFG)
        self.assertLess(bad["oops_score"], good["oops_score"])

    # 4 ─────────────────────────────────────────────────────────────────────────
    def test_higher_expected_utility_contract_is_selected(self):
        # ATM (convex, lower total-loss prob) vs far-OTM (mostly worthless).
        atm = _contract(100, 4.0, sym="ATM")
        otm = _contract(125, 0.40, sym="OTM")
        ranked = oops.optimize_option_candidates([otm, atm], 100.0, cfg=CFG)
        self.assertEqual(ranked[0]["contract_symbol"], "ATM")
        self.assertGreater(ranked[0]["oops_score"], ranked[1]["oops_score"])
        self.assertLess(ranked[0]["probability_total_loss"], ranked[1]["probability_total_loss"])

    # 5 ─────────────────────────────────────────────────────────────────────────
    def test_wide_spread_reduces_oops_score(self):
        narrow = oops.optimize_option_candidates([_contract(100, 4.0, spread=0.01)], 100.0, cfg=CFG)[0]
        wide = oops.optimize_option_candidates([_contract(100, 4.0, spread=0.05)], 100.0, cfg=CFG)[0]
        self.assertLess(wide["oops_score"], narrow["oops_score"])

    # 6 ─────────────────────────────────────────────────────────────────────────
    def test_oops_cannot_select_illiquid_contract(self):
        liquid = _contract(100, 4.0, oi=900, vol=80, sym="LIQ")
        illiquid = _contract(100, 4.0, oi=5, vol=1, sym="ILQ")     # below OI/volume gates
        ranked = oops.optimize_option_candidates([illiquid, liquid], 100.0, cfg=CFG)
        syms = [r["contract_symbol"] for r in ranked]
        self.assertIn("LIQ", syms)
        self.assertNotIn("ILQ", syms)

    # 10 (model side) ────────────────────────────────────────────────────────────
    def test_missing_data_returns_empty_for_bs_fallback(self):
        # No valid sim inputs (zero strike / zero dte) → no OOPS rows → caller uses BS.
        bad = [{"strike": 0, "mid": 4.0, "dte": 0, "open_interest": 800, "volume": 60, "spread_pct": 0.03}]
        self.assertEqual(oops.optimize_option_candidates(bad, 100.0, cfg=CFG), [])
        self.assertEqual(oops.optimize_option_candidates([], 100.0, cfg=CFG), [])

    # 7 ─────────────────────────────────────────────────────────────────────────
    def test_oops_is_advisory_only(self):
        # No public callable implies order placement.
        for n in [n for n in dir(oops) if not n.startswith("_") and callable(getattr(oops, n))]:
            self.assertNotIn("submit", n.lower())
            self.assertNotIn("place", n.lower())
        # No actual EXECUTION code: it never imports run/the broker or submits orders.
        # (The module docstring may *mention* AlpacaGateway/RiskManager in prose to
        # document that they remain the only execution path — that's allowed.)
        src = Path(oops.__file__).read_text()
        self.assertNotIn("import run", src)
        self.assertNotIn(".submit_market_order(", src)
        self.assertNotIn("AlpacaGateway(", src)      # never instantiates the broker
        self.assertNotIn(".gate(", src)              # never calls RiskManager.gate


class OOPSEngineIntegrationTests(unittest.TestCase):

    # 8 ─────────────────────────────────────────────────────────────────────────
    def test_alpha_order_routes_through_risk_manager_and_gateway(self):
        import run
        risk = run.RiskManager(equity=100_000.0)
        gw = run.AlpacaGateway(risk)
        gw._available = False; gw._mode = "sim"      # simulate; risk gate still applies
        eng = ace.AlphaCallEngine(broker=gw, config=ace.AlphaCallConfig())

        ready = ace.CallSignal(symbol="AAPL", setup_type="ORB", setup_score=88, grade="A+")
        ready.contract_symbol = "AAPL260116C00150000"; ready.contracts = 1
        ready.mid = 4.0; ready.ask = 4.1; ready.premium_cost = 410.0
        ready.selected_by_oops = True; ready.oops_score = 75.0; ready.status = "ready"
        eng.evaluate_universe = lambda **kw: [ready]

        class _WF:
            def consume_alpha(self, amt): return amt

        with patch.object(run, "is_market_open", return_value=True):
            placed = run.run_alpha_entries(eng, gw, _WF(), risk, symbols=["AAPL"],
                                           bars_by_symbol={}, chains_by_symbol={}, market_context={})
        self.assertEqual(placed, 1)
        # Recorded fill on the RiskManager proves it passed risk.gate() via the gateway.
        self.assertIn("AAPL260116C00150000", risk.open_positions)

    def test_alpha_entries_blocked_when_market_closed(self):
        import run
        risk = run.RiskManager(equity=100_000.0)
        gw = run.AlpacaGateway(risk)
        gw._available = False; gw._mode = "sim"
        eng = ace.AlphaCallEngine(broker=gw, config=ace.AlphaCallConfig())
        eng.evaluate_universe = lambda **kw: self.fail("evaluate_universe should not run when market is closed")

        class _WF:
            def consume_alpha(self, amt): return amt

        with patch.object(run, "is_market_open", return_value=False):
            placed = run.run_alpha_entries(eng, gw, _WF(), risk, symbols=["AAPL"],
                                           bars_by_symbol={}, chains_by_symbol={}, market_context={})
        self.assertEqual(placed, 0)
        self.assertEqual(risk.open_positions, {})

    def test_alpha_entries_blocked_by_halt_new_buys(self):
        import run
        risk = run.RiskManager(equity=100_000.0)
        gw = run.AlpacaGateway(risk)
        gw._available = False; gw._mode = "sim"
        eng = ace.AlphaCallEngine(broker=gw, config=ace.AlphaCallConfig())
        eng.evaluate_universe = lambda **kw: self.fail("evaluate_universe should not run during halt_new_buys")

        class _WF:
            def consume_alpha(self, amt): return amt

        with patch.object(run, "is_market_open", return_value=True):
            placed = run.run_alpha_entries(
                eng, gw, _WF(), risk,
                symbols=["AAPL"], bars_by_symbol={}, chains_by_symbol={},
                market_context={"trade_permission": "halt_new_buys"},
            )
        self.assertEqual(placed, 0)
        self.assertEqual(risk.open_positions, {})

    # 9 ─────────────────────────────────────────────────────────────────────────
    def test_oops_exit_cannot_override_hard_stop(self):
        eng = ace.AlphaCallEngine(broker=types.SimpleNamespace(), config=ace.AlphaCallConfig())
        eng._active = {}
        cfg = eng.config
        call = ace.ActiveCall(
            symbol="XYZ", contract_symbol="XYZ260601C00100000", expiry="2026-06-01",
            strike=100.0, contracts_held=2, contracts_original=2, entry_mid=4.0,
            entry_cost=800.0, hard_stop_value=800.0 * (1 - cfg.stop_loss_pct),
            r_target_value=800.0 * (1 + cfg.r_target),
            partial_value=800.0 * (1 + cfg.partial_profit_pct), peak_value=800.0)
        eng._active["XYZ"] = call
        # Deep loss → hard stop must fire first; OOPS exit assessment never runs.
        positions = [{"symbol": "XYZ260601C00100000", "qty": 2, "unrealized_plpc": -0.50}]
        exits = eng._check_active_positions(positions, waterfall_callback=None, market_context={})
        sources = [o["source"] for o in exits]
        self.assertIn("alpha_call_hard_stop", sources)
        self.assertNotIn("alpha_call_oops_exit", sources)

    # 10 (engine side) ────────────────────────────────────────────────────────────
    def test_bs_fallback_when_oops_unavailable(self):
        sig = ace.CallSignal(symbol="AAA", setup_type="ORB", setup_score=80, grade="A")
        # A contract WITHOUT oops fields (simulation unavailable) → BS fallback.
        ace._apply_oops_fields(sig, {"contract_symbol": "AAA260101C00100000"}, selected_by_oops=False)
        self.assertFalse(sig.selected_by_oops)
        self.assertEqual(sig.oops_reject_reason, "oops_unavailable_bs_fallback")
        self.assertEqual(sig.oops_score, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
