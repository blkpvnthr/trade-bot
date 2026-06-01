"""P0 regression tests for trade.py exit-path reliability.

Covers the fix for Alpaca 40310000 "insufficient qty available" SELL errors:
protective bracket legs hold shares (qty_available < qty), so a plain market
sell fails. `_prepare_exit` must cancel THIS symbol's working orders to free the
shares, then size the exit to what is actually available — never blocking an
exit, only enabling/resizing it.
"""
import types
import unittest
from unittest.mock import patch

import trade


class FakeBroker:
    """Stateful broker double. `avail_sequence` is returned by successive
    get_available_qty() calls; cancel returns a fixed list of order IDs and
    records which symbols it was asked to cancel."""

    def __init__(self, avail_sequence, cancel_returns):
        self._avail = list(avail_sequence)
        self._cancel_returns = list(cancel_returns)
        self.cancel_calls = []
        self.avail_calls = []

    def get_available_qty(self, symbol):
        self.avail_calls.append(symbol)
        return self._avail.pop(0) if self._avail else 0.0

    def cancel_open_orders_for_symbol(self, symbol):
        self.cancel_calls.append(symbol)
        return list(self._cancel_returns)


def make_executor(broker):
    ex = trade.ProductionOrderExecutor.__new__(trade.ProductionOrderExecutor)
    ex.broker = broker
    return ex


class PrepareExitTests(unittest.TestCase):
    def setUp(self):
        # No real events / no real sleep during tests.
        self._patchers = [
            patch.object(trade, "write_event", lambda *a, **k: None),
            patch.object(trade.time, "sleep", lambda *a, **k: None),
        ]
        for p in self._patchers:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patchers])

    def test_cancels_and_exits_full_qty_when_cancel_frees_shares(self):
        """Shares held by bracket legs (avail 0) → cancel → freed (avail 1000)
        → exit submits the full intended qty."""
        broker = FakeBroker(avail_sequence=[0.0, 1000.0], cancel_returns=["leg-1", "leg-2"])
        ex = make_executor(broker)

        exit_qty, meta = ex._prepare_exit("RDW", 1000, "cycle1", "monitor_failed_breakout_cut")

        self.assertEqual(exit_qty, 1000)
        self.assertEqual(broker.cancel_calls, ["RDW"])  # cancel happened, per-symbol
        self.assertEqual(meta["orders_canceled_count"], 2)
        self.assertEqual(meta["available_qty_before"], 0.0)
        self.assertEqual(meta["available_qty_after"], 1000.0)
        self.assertFalse(meta["resized"])

    def test_resizes_to_available_when_cancel_only_partially_frees(self):
        """If only part of the position can be freed, the exit is sized to what
        is sellable so risk still drops — it never errors on the full qty."""
        broker = FakeBroker(avail_sequence=[0.0, 600.0], cancel_returns=["leg-1"])
        ex = make_executor(broker)

        exit_qty, meta = ex._prepare_exit("MNTS", 1000, "cycle1", "scanner_stale_position")

        self.assertEqual(exit_qty, 600)
        self.assertTrue(meta["resized"])
        self.assertEqual(meta["exit_qty_intended"], 1000)
        self.assertEqual(meta["exit_qty_submitted"], 600)

    def test_no_cancel_when_shares_already_available(self):
        """When qty_available already covers the exit, do not touch any orders."""
        broker = FakeBroker(avail_sequence=[1000.0], cancel_returns=[])
        ex = make_executor(broker)

        exit_qty, meta = ex._prepare_exit("AAPL", 1000, "cycle1", "take_profit")

        self.assertEqual(exit_qty, 1000)
        self.assertEqual(broker.cancel_calls, [])  # no cancellation needed
        self.assertEqual(len(broker.avail_calls), 1)  # only the before-check ran
        self.assertFalse(meta["resized"])

    def test_returns_zero_qty_when_nothing_available_after_cancel(self):
        """Nothing sellable even after cancel → exit_qty 0 so the caller skips
        (journals exit_no_available_qty_after_cancel) rather than erroring."""
        broker = FakeBroker(avail_sequence=[0.0, 0.0], cancel_returns=["leg-1"])
        ex = make_executor(broker)

        exit_qty, meta = ex._prepare_exit("RDW", 1000, "cycle1", "")

        self.assertEqual(exit_qty, 0)
        self.assertEqual(meta["available_qty_after"], 0.0)

    def test_cancel_is_per_symbol_not_global(self):
        """The cancel must target only the exiting symbol (other symbols' orders
        and their protective brackets must be left intact)."""
        broker = FakeBroker(avail_sequence=[0.0, 500.0], cancel_returns=["leg-1"])
        ex = make_executor(broker)

        ex._prepare_exit("TSLA", 500, "cycle1", "stop")

        self.assertEqual(broker.cancel_calls, ["TSLA"])


class BrokerPerSymbolCancelTests(unittest.TestCase):
    def test_cancel_open_orders_for_symbol_only_cancels_matching_symbol(self):
        """Per-symbol cancel must skip other symbols' orders entirely."""
        canceled_ids = []

        class FakeTrading:
            def cancel_order_by_id(self, oid):
                canceled_ids.append(oid)

        broker = trade.AlpacaBroker.__new__(trade.AlpacaBroker)
        broker.trading = FakeTrading()
        orders = [
            types.SimpleNamespace(symbol="RDW", id="a"),
            types.SimpleNamespace(symbol="MNTS", id="b"),
            types.SimpleNamespace(symbol="RDW", id="c"),
        ]
        with patch.object(trade.AlpacaBroker, "list_open_orders", lambda self: orders):
            result = broker.cancel_open_orders_for_symbol("RDW")

        self.assertEqual(sorted(result), ["a", "c"])
        self.assertEqual(sorted(canceled_ids), ["a", "c"])  # MNTS leg 'b' untouched

    def test_get_available_qty_prefers_qty_available_field(self):
        class FakeTrading:
            def get_all_positions(self):
                return [types.SimpleNamespace(symbol="RDW", qty="1000", qty_available="300")]

        broker = trade.AlpacaBroker.__new__(trade.AlpacaBroker)
        broker.trading = FakeTrading()
        self.assertEqual(broker.get_available_qty("RDW"), 300.0)

    def test_get_available_qty_falls_back_to_qty_when_field_absent(self):
        class FakeTrading:
            def get_all_positions(self):
                pos = types.SimpleNamespace(symbol="RDW", qty="1000")
                return [pos]

        broker = trade.AlpacaBroker.__new__(trade.AlpacaBroker)
        broker.trading = FakeTrading()
        # qty_available attribute absent → falls back to qty
        self.assertEqual(broker.get_available_qty("RDW"), 1000.0)

    def test_get_available_qty_zero_when_no_position(self):
        class FakeTrading:
            def get_all_positions(self):
                return []

        broker = trade.AlpacaBroker.__new__(trade.AlpacaBroker)
        broker.trading = FakeTrading()
        self.assertEqual(broker.get_available_qty("RDW"), 0.0)


if __name__ == "__main__":
    unittest.main()
