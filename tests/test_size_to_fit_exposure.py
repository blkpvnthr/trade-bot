"""P1 tests: size-to-fit on the stock-exposure cap.

Instead of hard-rejecting a BUY whose notional pushes stock exposure past the
cap, RiskManager.max_affordable_qty returns the largest qty that fits the
remaining room. The cap must never be exceeded, and sells / hard blocks are
unaffected.
"""
import unittest

import run


class MaxAffordableQtyTests(unittest.TestCase):
    def _risk(self, equity=100_000.0, stock_exposure=0.0, max_order_dollars=0.0):
        r = run.RiskManager.__new__(run.RiskManager)
        r.equity = equity
        r.stock_exposure = stock_exposure
        r.options_exposure = 0.0
        r.max_order_dollars = max_order_dollars
        return r

    def test_trims_buy_to_remaining_stock_room(self):
        # cap = 80% * 100k = 80k; already 76k used → 4k room; px=40 → 100 shares.
        r = self._risk(equity=100_000.0, stock_exposure=76_000.0)
        fit = r.max_affordable_qty("buy", price=40.0, asset_class="equity")
        self.assertEqual(fit, 100)
        # The trimmed order must NOT breach the cap.
        self.assertLessEqual(r.stock_exposure + fit * 40.0, r.equity * run.MAX_STOCK_EXPOSURE_PCT)

    def test_zero_room_returns_zero(self):
        r = self._risk(equity=100_000.0, stock_exposure=80_000.0)
        self.assertEqual(r.max_affordable_qty("buy", price=40.0), 0)

    def test_per_order_dollar_cap_is_binding_when_tighter(self):
        # plenty of exposure room, but per-order cap 1000$ at px=40 → 25 shares.
        r = self._risk(equity=100_000.0, stock_exposure=0.0, max_order_dollars=1_000.0)
        self.assertEqual(r.max_affordable_qty("buy", price=40.0), 25)

    def test_sells_are_not_trimmed(self):
        r = self._risk()
        self.assertIsNone(r.max_affordable_qty("sell", price=40.0))

    def test_nonpositive_price_returns_none(self):
        r = self._risk()
        self.assertIsNone(r.max_affordable_qty("buy", price=0.0))

    def test_gate_passes_after_trimming_to_fit(self):
        # Full gate: trim qty to fit, then gate() must allow it (cap respected).
        r = run.RiskManager.__new__(run.RiskManager)
        r.equity = 100_000.0
        r.starting_equity = 100_000.0
        r.stock_exposure = 76_000.0
        r.options_exposure = 0.0
        r.open_positions = {}
        r.daily_pnl = 0.0
        r.pdt_restricted = False
        r._killed = False
        r.max_order_dollars = 0.0
        r.max_order_size = 0
        r.max_positions = 50
        r.equity_floor = 0.0
        r.daily_loss_limit = -1e9
        r.max_drawdown = -1.0
        r.refresh_dynamic_limits = lambda: None
        r._check_kill_switch = lambda: False

        requested = 1000  # 1000 * 40 = 40k → would breach the 4k room
        fit = r.max_affordable_qty("buy", 40.0, asset_class="equity")
        self.assertEqual(fit, 100)
        allowed, reason = r.gate("XYZ", "buy", fit, 40.0, asset_class="equity")
        self.assertTrue(allowed, msg=f"gate rejected trimmed order: {reason}")


if __name__ == "__main__":
    unittest.main()
