import unittest

import run


def _make_signal(symbol: str = "HPE") -> run.ORBSweepEngulfSignal:
    return run.ORBSweepEngulfSignal(
        symbol=symbol,
        side="long",
        trade_date="2026-07-14",
        timestamp="2026-07-14T14:05:00+00:00",
        opening_timestamp="2026-07-14T13:30:00+00:00",
        opening_high=10.50,
        opening_low=10.00,
        sweep_price=9.80,
        retest_timestamp="2026-07-14T14:00:00+00:00",
        engulf_timestamp="2026-07-14T14:05:00+00:00",
        retest_delta=-100.0,
        engulf_delta=150.0,
        entry_price=10.20,
        stop_loss=9.80,
        take_profit=11.40,
        risk_per_share=0.40,
        reward_per_share=1.20,
        retest_open=9.95, retest_high=10.05, retest_low=9.82, retest_close=9.98,
        engulf_open=9.98, engulf_high=10.22, engulf_low=9.97, engulf_close=10.20,
    )


class _SubmittedOrder:
    id = "order-1"
    status = "accepted"


class _BrokerStub:
    def __init__(self):
        self.submit_calls = 0

    def submit_market_bracket_order(self, **kwargs):
        self.submit_calls += 1
        return _SubmittedOrder()

    def get_account_snapshot(self):
        return run.AccountSnapshot(buying_power=100000.0, equity=100000.0, cash=100000.0, portfolio_value=100000.0)

    def get_all_open_positions(self):
        return []

    def has_open_order_for_symbol(self, symbol):
        return False

    def get_order_by_id(self, order_id):
        return _SubmittedOrder()


class _SignalEngineStub:
    def __init__(self):
        self.marked = []

    def mark_processed(self, signal):
        self.marked.append(signal)


class _JournalStub:
    def __init__(self):
        self.rows = []

    def append(self, row):
        self.rows.append(row)


def _make_executor():
    executor = run.ORBSweepEngulfExecutor.__new__(run.ORBSweepEngulfExecutor)
    executor.config = run.ExecutorConfig()
    executor.journal = _JournalStub()
    executor.broker = _BrokerStub()
    executor.signal_engine = _SignalEngineStub()
    executor.last_order_time = {}
    executor.last_heartbeat_time = 0.0
    executor.account_history = []
    executor._session_ordered = {}
    return executor


class LiveExperienceLoggingTests(unittest.TestCase):
    def test_post_submit_logging_failure_does_not_turn_filled_order_into_error(self):
        executor = _make_executor()
        signal = _make_signal("HPE")

        original_write_event = run.write_event
        recorded_events = []

        def fake_write_event(event, payload=None):
            recorded_events.append(event)
            if event == "orb_sweep_engulf_bracket_order_submitted":
                raise AttributeError("simulated post-submit logging failure")

        run.write_event = fake_write_event
        try:
            result = executor._execute_signal(signal, "cycle-1")
        finally:
            run.write_event = original_write_event

        self.assertEqual(recorded_events, ["orb_sweep_engulf_bracket_order_submitted", "orb_sweep_engulf_post_submit_logging_error"])
        self.assertEqual(executor.broker.submit_calls, 1)
        self.assertNotIn(result["status"], {"error", "api_error"})
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["order_id"], "order-1")
        self.assertEqual(result["symbol"], "HPE")
        # The signal must still be marked processed / session-deduped even
        # though post-submission logging blew up, so we never re-submit.
        self.assertEqual(len(executor.signal_engine.marked), 1)
        self.assertTrue(executor._session_already_ordered("HPE"))


class PerSymbolCooldownTests(unittest.TestCase):
    def test_cooldown_on_one_symbol_does_not_block_a_different_symbol(self):
        executor = _make_executor()
        executor._mark_order("AAPL")

        self.assertTrue(executor._in_cooldown("AAPL"))
        self.assertFalse(executor._in_cooldown("MSFT"))


if __name__ == "__main__":
    unittest.main()
