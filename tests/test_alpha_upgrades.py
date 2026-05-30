"""
Tests for the Alpha options upgrades: volatility diagnostics (expected move, IV
rank/percentile with graceful fallback, gamma/theta efficiency in scoring, IV crush
risk affecting the overnight-hold score), and the Black-Scholes fallback exit ladder.
Offline — synthetic chains/positions are constructed in-test.
"""
import math
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import alpha_call_engine as ace
from alpha_call_engine import (
    AlphaCallConfig, AlphaCallEngine, ActiveCall,
    compute_volatility_diagnostics, overnight_hold_score, price_call_contract,
    resolve_risk_free_rate,
)


def _call(strike, iv, dte, oi=800, vol=50):
    return {"strike": strike, "implied_volatility": iv, "dte": dte,
            "open_interest": oi, "volume": vol}


class AlphaUpgradeTests(unittest.TestCase):
    CFG = AlphaCallConfig()

    # 7 ─────────────────────────────────────────────────────────────────────────
    def test_expected_move_computed_when_iv_and_price_exist(self):
        S = 100.0
        chain = [_call(95, 0.30, 30), _call(100, 0.30, 30), _call(105, 0.30, 30)]
        d = compute_volatility_diagnostics("XYZ", S, chain, None, self.CFG, {})
        expected = S * 0.30 * math.sqrt(30 / 365.0)
        self.assertGreater(d["expected_move"], 0.0)
        self.assertAlmostEqual(d["expected_move"], expected, places=2)

    # 8 ─────────────────────────────────────────────────────────────────────────
    def test_iv_rank_percentile_fallback_when_history_missing(self):
        S = 100.0
        # No bars (no realized-vol history) but a chain with IV skew across strikes.
        chain = [_call(90, 0.22, 30), _call(100, 0.30, 30), _call(110, 0.42, 30)]
        d = compute_volatility_diagnostics("XYZ", S, chain, None, self.CFG, {})
        self.assertFalse(d["iv_history_available"])
        self.assertIn(d["iv_rank_source"], {"chain_cross_section", "realized_vol_band", "none"})
        self.assertTrue(0.0 <= d["iv_rank"] <= 100.0)
        self.assertTrue(0.0 <= d["iv_percentile"] <= 100.0)
        # No IV anywhere → neutral 50/50, never crashes.
        empty = compute_volatility_diagnostics("XYZ", S, [_call(100, 0.0, 30)], None, self.CFG, {})
        self.assertEqual(empty["iv_rank"], 50.0)
        self.assertEqual(empty["iv_percentile"], 50.0)

    # 9 ─────────────────────────────────────────────────────────────────────────
    def test_gamma_theta_efficiency_affects_contract_scoring(self):
        # Hold every other score input equal (same strike/S/DTE/delta/spread/OI/volume,
        # mid == theoretical so mispricing is 0, both IVs inside the flat IV-score band)
        # and vary ONLY implied vol. Lower IV ⇒ higher gamma/theta efficiency ⇒ the
        # black_scholes_score must be higher. This isolates the gamma/theta term.
        S, K, dte, r = 100.0, 100.0, 40, self.CFG.risk_free_rate
        from alpha_call_engine import black_scholes_call

        def priced(iv):
            theo = black_scholes_call(S, K, dte / 365.0, r, iv, self.CFG.dividend_yield)["price"]
            c = {"strike": K, "mid": round(theo, 4), "dte": dte, "delta": 0.55,
                 "spread_pct": 0.02, "open_interest": 800, "volume": 50,
                 "implied_volatility": iv}
            return price_call_contract(dict(c), S, self.CFG, r, momentum_rvol=2.0)

        efficient = priced(0.25)   # higher gamma/theta efficiency
        inefficient = priced(0.50)
        self.assertGreater(efficient["gamma_theta_ratio"], inefficient["gamma_theta_ratio"])
        self.assertGreater(efficient["black_scholes_score"], inefficient["black_scholes_score"])

    # 10 ────────────────────────────────────────────────────────────────────────
    def test_iv_crush_risk_reduces_overnight_hold_score(self):
        low = {"iv_crush_risk": 5.0, "iv_term_structure": 1.0, "volatility_regime_score": 60.0}
        high = {"iv_crush_risk": 90.0, "iv_term_structure": 1.3, "volatility_regime_score": 30.0}
        self.assertGreater(overnight_hold_score(60.0, low, self.CFG),
                           overnight_hold_score(60.0, high, self.CFG))
        # High crush + backwardation should also flip the exit optimizer to an exit.
        eng = AlphaCallEngine(broker=types.SimpleNamespace(), config=self.CFG)
        call = ActiveCall(symbol="XYZ", contract_symbol="XYZ250101C00100000", expiry="2025-01-01",
                          strike=100.0, contracts_held=1, contracts_original=1, entry_mid=4.0,
                          entry_cost=400.0, hard_stop_value=288.0, r_target_value=1600.0,
                          partial_value=600.0, peak_value=400.0)
        vol_high = {"iv_crush_risk": 95.0, "iv_term_structure": 1.3, "volatility_regime_score": 20.0}
        guide = eng.black_scholes_exit_optimizer(call, current_value=420.0, dte_remaining=25,
                                                 vol_diag=vol_high)
        self.assertEqual(guide["action"], "exit_iv_crush")

    # 11 ────────────────────────────────────────────────────────────────────────
    def test_black_scholes_missing_inputs_still_trigger_fallback_ladder(self):
        # When BS/vol inputs are unavailable, the hard R-multiple exit ladder must
        # still fire. A position deep in the red triggers the hard stop with no BS data.
        eng = AlphaCallEngine(broker=types.SimpleNamespace(), config=self.CFG)
        eng._active = {}
        call = ActiveCall(symbol="XYZ", contract_symbol="XYZ250601C00100000", expiry="2025-06-01",
                          strike=100.0, contracts_held=2, contracts_original=2, entry_mid=4.0,
                          entry_cost=800.0, hard_stop_value=800.0 * (1 - self.CFG.stop_loss_pct),
                          r_target_value=800.0 * (1 + self.CFG.r_target),
                          partial_value=800.0 * (1 + self.CFG.partial_profit_pct), peak_value=800.0)
        eng._active["XYZ"] = call
        positions = [{"symbol": "XYZ250601C00100000", "qty": 2, "unrealized_plpc": -0.50}]
        exits = eng._check_active_positions(positions, waterfall_callback=None, market_context={})
        self.assertTrue(any(o["source"] == "alpha_call_hard_stop" for o in exits),
                        "hard-stop ladder must fire without any Black-Scholes inputs")
        # Solving IV from an impossible market price returns the graceful BS reject,
        # not an exception.
        bad = price_call_contract({"strike": 100.0, "mid": 0.0, "dte": 30, "delta": 0.5,
                                   "spread_pct": 0.02, "open_interest": 500, "volume": 10},
                                  100.0, self.CFG, self.CFG.risk_free_rate)
        self.assertEqual(bad["bs_reject_reason"], "rejected_by_black_scholes_iv")


if __name__ == "__main__":
    unittest.main(verbosity=2)
