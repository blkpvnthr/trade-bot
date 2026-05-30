"""
Priority-0 regression tests for the exposure-notional bug: RiskManager exposure must
never be computed on qty × 0.0. A BUY with no resolvable reference price is rejected,
and BAL orders are gated on real notional.
"""
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run


def _sim_gateway(equity=100_000.0):
    risk = run.RiskManager(equity=equity)
    gw = run.AlpacaGateway(risk)
    gw._available = False      # simulation: no live market-data resolution
    gw._mode = "sim"
    return risk, gw


class ExposureNotionalTests(unittest.TestCase):

    def test_gate_uses_real_notional_not_zero(self):
        risk, gw = _sim_gateway()
        # Near the 80% stock-exposure cap (= $80k on $100k equity).
        risk.stock_exposure = 79_000.0
        # A $5,000 notional buy must be REJECTED (79k + 5k = 84k > 80k cap).
        over = gw.submit_market_order("AAA", "buy", 100, reason="t", reference_price=50.0)
        self.assertIsNone(over, "buy exceeding the exposure cap must be rejected on real notional")
        # A $500 notional buy must be ALLOWED (79k + 0.5k = 79.5k <= 80k).
        under = gw.submit_market_order("AAA", "buy", 10, reason="t", reference_price=50.0)
        self.assertIsNotNone(under, "buy within the exposure cap must pass")

    def test_buy_without_reference_price_is_rejected(self):
        # The old bug: no price → gate saw qty × 0.0 and silently passed. Now: no
        # resolvable price → the BUY is rejected (never gated on zero notional).
        risk, gw = _sim_gateway()
        resp = gw.submit_market_order("BBB", "buy", 100, reason="t")   # no limit/reference, sim → no data
        self.assertIsNone(resp, "a buy with no reference price must be rejected, not zero-gated")

    def test_sell_without_price_still_allowed(self):
        # Exits must never depend on a notional/reference price.
        risk, gw = _sim_gateway()
        resp = gw.submit_market_order("CCC", "sell", 5, reason="exit")
        self.assertIsNotNone(resp, "sells must not be blocked by missing reference price")

    def test_bal_entries_pass_reference_price_and_gate(self):
        risk, gw = _sim_gateway()

        class _WF:
            def consume_bal(self, amt):
                return amt

        # Fake BAL: one priced order (good) + one zero-price order (must be gated out).
        class _BAL:
            def evaluate_dca(self, **kw):
                return [
                    {"symbol": "GOOD", "qty": 10, "price": 150.0, "order_dollars": 1500.0, "reason": "buffett_dca"},
                    {"symbol": "NOPX", "qty": 10, "price": 0.0, "order_dollars": 0.0, "reason": "buffett_dca"},
                ]

        placed = run.run_bal_entries(
            _BAL(), gw, _WF(), risk, symbols=["GOOD", "NOPX"], market_context={},
            bars_ctx_by_symbol={}, equity_positions={}, insider_data_by_symbol={})
        # The priced order is placed (real notional); the zero-price order is rejected.
        self.assertEqual(placed, 1)
        self.assertIn("GOOD", risk.open_positions)
        self.assertNotIn("NOPX", risk.open_positions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
