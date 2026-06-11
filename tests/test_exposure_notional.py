"""
Priority-0 regression tests for the exposure-notional bug: RiskManager exposure must
never be computed on qty × 0.0. A BUY with no resolvable reference price is rejected,
and BAL orders are gated on real notional.
"""
import sys
import os
import subprocess
import types
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run


def _sim_gateway(equity=100_000.0):
    risk = run.RiskManager(equity=equity)
    risk.max_order_dollars = equity
    risk.refresh_dynamic_limits = lambda: None
    gw = run.AlpacaGateway(risk)
    gw._available = False      # simulation: no live market-data resolution
    gw._mode = "sim"
    return risk, gw


class ExposureNotionalTests(unittest.TestCase):

    def test_gate_uses_real_notional_not_zero(self):
        risk, gw = _sim_gateway()
        # Near the 80% stock-exposure cap (= $80k on $100k equity).
        risk.stock_exposure = 79_000.0
        # SIZE-TO-FIT (P1.5): a $5,000 notional buy (100 @ $50) that would breach
        # the cap (79k + 5k = 84k > 80k) is now TRIMMED to the $1k of remaining
        # room (= 20 shares) instead of being rejected outright. The cap itself is
        # never exceeded. (That a trim happens at all proves notional is computed
        # on the REAL price, not qty × 0.0 — a zero notional would never trim.)
        over = gw.submit_market_order("AAA", "buy", 100, reason="t", reference_price=50.0)
        self.assertIsNotNone(over, "over-cap buy should be trimmed to fit, not dropped")
        self.assertEqual(over["qty"], 20, "buy must be trimmed to the room under the cap")
        self.assertLessEqual(risk.stock_exposure, risk.equity * run.MAX_STOCK_EXPOSURE_PCT)
        # A $500 notional buy must pass untrimmed (79k + 0.5k = 79.5k <= 80k).
        risk, gw = _sim_gateway()
        risk.stock_exposure = 79_000.0
        under = gw.submit_market_order("AAA", "buy", 10, reason="t", reference_price=50.0)
        self.assertIsNotNone(under, "buy within the exposure cap must pass")
        self.assertEqual(under["qty"], 10, "in-cap buy must not be trimmed")

    def test_size_to_fit_can_be_disabled(self):
        """With SIZE_TO_FIT_EXPOSURE off, the old hard-reject contract holds."""
        risk, gw = _sim_gateway()
        risk.stock_exposure = 79_000.0
        with unittest.mock.patch.object(run, "SIZE_TO_FIT_EXPOSURE", False):
            over = gw.submit_market_order("AAA", "buy", 100, reason="t", reference_price=50.0)
        self.assertIsNone(over, "with size-to-fit disabled, an over-cap buy is rejected")

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

    def test_sell_allowed_during_drawdown_block_but_kill_switch_still_blocks(self):
        risk, _ = _sim_gateway()
        risk.equity = 85_000.0
        risk.starting_equity = 100_000.0
        allowed, reason = risk.gate("CCC", "sell", 5, 10.0)
        self.assertTrue(allowed, reason)

        risk._killed = True
        allowed, reason = risk.gate("CCC", "sell", 5, 10.0)
        self.assertFalse(allowed)
        self.assertEqual(reason, "rejected_by_kill_switch")

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

    def test_apca_paper_alias_is_honored(self):
        env = dict(os.environ)
        env.pop("ALPACA_PAPER_ONLY", None)
        env["APCA_PAPER"] = "false"
        out = subprocess.check_output(
            [sys.executable, "-c", "import run; print(run.ALPACA_PAPER_ONLY)"],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            text=True,
        ).strip()
        self.assertEqual(out, "False")


if __name__ == "__main__":
    unittest.main(verbosity=2)
