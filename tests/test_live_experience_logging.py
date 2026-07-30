import types
import unittest
from unittest.mock import patch

import pandas as pd

import trade as run


class _SubmittedOrder:
    id = "order-1"
    status = "filled"


class _BrokerStub:
    def get_position(self, symbol):
        return run.BrokerPosition(symbol=symbol, qty=0)

    def has_open_order_for_symbol(self, symbol):
        return False

    def submit_market_buy(self, symbol, qty, client_order_id, time_in_force):
        return _SubmittedOrder()

    def cancel_open_orders_for_symbol(self, symbol):
        return None

    def get_all_open_positions(self):
        return []


class LiveExperienceLoggingTests(unittest.TestCase):
    def _make_executor(self):
        executor = run.ProductionOrderExecutor.__new__(run.ProductionOrderExecutor)
        executor.config = run.build_default_config()
        executor.config.scanner.require_direction_confirmation_before_buy = False
        executor.config.scanner.stream_confirmed_append_direction_rejects = False
        executor.config.scanner.stream_confirmed_submit_trailing_stop_after_fill = False
        executor.scanner = None
        executor.broker = _BrokerStub()
        executor._record_account_snapshot = lambda label, cycle_id: {
            "equity": 100000.0,
            "cash": 100000.0,
            "buying_power": 400000.0,
        }
        executor._gross_exposure = lambda equity: 0.0
        executor._latest_feature_by_symbol = lambda: {}
        executor._cash_buffer = lambda equity: 0.0
        executor._max_order_notional = lambda equity: 1_000_000.0
        executor._risk_profile = lambda equity: types.SimpleNamespace(
            max_gross_exposure=1.0,
            min_notional_to_trade=1.0,
        )
        executor._read_kill_state = lambda: {"start_equity": 100000.0}
        executor._mark_order = lambda: None
        executor._poll_fill = lambda order_id, price: {
            "fill_status": "filled",
            "filled_qty": 10,
            "filled_avg_price": 10.0,
            "slippage_pct": 0.0,
        }
        executor._order_log_context = lambda **kwargs: {}
        executor._journal_order = lambda *args, **kwargs: None
        return executor

    def test_logging_failure_does_not_turn_filled_order_into_error(self):
        executor = self._make_executor()
        stages = []

        def failing_log_live_experience(**kwargs):
            stages.append(kwargs["stage"])
            if kwargs["stage"] == "submitted":
                raise AttributeError("'ScannerConfig' object has no attribute 'live_experience_log_orders'")

        executor._log_live_experience = failing_log_live_experience

        plan = pd.DataFrame(
            [
                {
                    "symbol": "HPE",
                    "side": "buy",
                    "order_qty": 10,
                    "price": 10.0,
                    "capped_dollars": 100.0,
                    "reason": "test_entry",
                    "client_order_id": "client-1",
                }
            ]
        )

        with (
            patch.object(run, "log_executed_order", lambda **kwargs: None),
            patch.object(run, "write_event", lambda *args, **kwargs: None),
        ):
            results = executor._execute_order_plan(plan, "cycle-1")

        self.assertEqual(stages, ["decision", "submitted"])
        self.assertEqual(list(results["status"]), ["filled"])
        self.assertEqual(list(results["symbol"]), ["HPE"])


if __name__ == "__main__":
    unittest.main()
