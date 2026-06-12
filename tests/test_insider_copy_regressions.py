import csv
import contextlib
import types
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import run


class _GatewayStub:
    def __init__(self, response=None):
        self.response = response
        self.submit_calls = []

    def _resolve_reference_price(self, symbol):
        return 33.09

    def submit_market_order(self, symbol, side, qty, **kwargs):
        self.submit_calls.append(
            {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                **kwargs,
            }
        )
        return self.response


class _MorningDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        base = cls(2026, 6, 10, 10, 30, 0)
        return base.replace(tzinfo=tz) if tz is not None else base


class InsiderCopyRegressionTests(unittest.TestCase):
    def test_insider_copier_skips_symbols_already_in_local_state(self):
        gateway = _GatewayStub()
        today = run.datetime.now(run.ET).date().isoformat()
        sec_candidates = [
            {
                "symbol": "RYAN",
                "total_open_market_purchase_value": run.INSIDER_COPY_MIN_VALUE + 500_000,
                "latest_filing_date": today,
                "company": "Ryan Specialty",
                "insider_count": 1,
                "repeated_purchases": False,
            }
        ]

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.multiple(
                    run,
                    ENABLE_INSIDER_COPIER=True,
                    INSIDER_COPY_USE_BACKTEST_REGIME=False,
                    INSIDER_COPY_PF_KILL_ENABLED=False,
                    INSIDER_COPY_QUEUE_ENABLED=False,
                    INSIDER_COPY_REQUIRE_QUALITY_SCREEN=False,
                    INSIDER_COPY_REQUIRE_AV_CONFIRM=False,
                    INSIDER_COPY_INSTRUMENT="share",
                    is_market_open=lambda: True,
                    regime_allows_new_buys=lambda market_context: (True, ""),
                    _insider_vol_liquidity_gate=lambda symbol: (True, "", 0.5, 2_000_000.0),
                    _insider_copy_state_load=lambda: {
                        "RYAN": {
                            "entry_date": today,
                            "underlying": "RYAN",
                            "qty": 21,
                            "instrument": "share",
                        }
                    },
                    _insider_copy_record_open=lambda *args, **kwargs: None,
                    _maybe_pair_insider_call=lambda *args, **kwargs: None,
                    prefetch_insider_vol_gate=lambda symbols: None,
                    log_skip=lambda *args, **kwargs: None,
                    datetime=_MorningDatetime,
                )
            )
            stack.enter_context(patch.object(run, "MAX_ORDER_DOLLARS", 120.0, create=True))
            placed = run.run_insider_trade_copier(
                gateway=gateway,
                risk=types.SimpleNamespace(equity=10_000.0),
                sec_candidates=sec_candidates,
                market_context={},
                equity_positions={},
                option_positions=[],
            )

        self.assertEqual(placed, 0)
        self.assertEqual(gateway.submit_calls, [])

    def test_log_executed_order_persists_explicit_order_metadata(self):
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            with patch.object(run, "LOG_DIR", log_dir), patch.object(run, "_in_test", return_value=False):
                run.log_executed_order(
                    event="insider_copy_buy",
                    engine="insider_copier",
                    symbol="RYAN",
                    side="buy",
                    qty=21,
                    limit_price=33.09,
                    client_order_id="coid-123",
                    order_id="order-123",
                    order_status="accepted",
                    filled_qty=21,
                    filled_avg_price=33.05,
                    slippage_pct=-0.0012,
                )

            journal_path = log_dir / f"{run.DATA_TAG}_executor_journal.csv"
            rows = list(csv.DictReader(journal_path.open()))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["client_order_id"], "coid-123")
        self.assertEqual(rows[0]["order_id"], "order-123")
        self.assertEqual(rows[0]["order_status"], "accepted")
        self.assertEqual(rows[0]["filled_qty"], "21")
        self.assertEqual(rows[0]["filled_avg_price"], "33.05")

    def test_log_executed_order_skips_disk_writes_during_tests(self):
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            with patch.object(run, "LOG_DIR", log_dir), patch.object(run, "_in_test", return_value=True):
                run.log_executed_order(
                    event="insider_copy_buy",
                    engine="insider_copier",
                    symbol="RYAN",
                    side="buy",
                    qty=21,
                    limit_price=33.09,
                    order_id="order-123",
                )
                run.write_event("executed_order", {"symbol": "RYAN"})

            self.assertFalse((log_dir / f"{run.DATA_TAG}_executor_journal.csv").exists())
            self.assertFalse((log_dir / f"{run.DATA_TAG}_execution_events.jsonl").exists())

    def test_insider_copier_uses_risk_max_order_dollars_for_share_clamp(self):
        gateway = _GatewayStub(response={"id": "order-1", "status": "accepted"})
        today = run.datetime.now(run.ET).date().isoformat()
        sec_candidates = [
            {
                "symbol": "RYAN",
                "total_open_market_purchase_value": run.INSIDER_COPY_MIN_VALUE + 500_000,
                "latest_filing_date": today,
                "company": "Ryan Specialty",
                "insider_count": 2,
                "repeated_purchases": False,
            }
        ]
        captured = []

        with patch.multiple(
            run,
            ENABLE_INSIDER_COPIER=True,
            INSIDER_COPY_USE_BACKTEST_REGIME=False,
            INSIDER_COPY_PF_KILL_ENABLED=False,
            INSIDER_COPY_QUEUE_ENABLED=False,
            INSIDER_COPY_REQUIRE_QUALITY_SCREEN=False,
            INSIDER_COPY_REQUIRE_AV_CONFIRM=False,
            INSIDER_COPY_INSTRUMENT="share",
            INSIDER_COPY_BREADTH_SHADOW=False,
            is_market_open=lambda: True,
            regime_allows_new_buys=lambda market_context: (True, ""),
            _insider_vol_liquidity_gate=lambda symbol: (True, "", 0.5, 2_000_000.0),
            _insider_copy_state_load=lambda: {},
            _insider_copy_record_open=lambda *args, **kwargs: None,
            _maybe_pair_insider_call=lambda *args, **kwargs: None,
            prefetch_insider_vol_gate=lambda symbols: None,
            log_skip=lambda *args, **kwargs: None,
            log_executed_order=lambda **kwargs: captured.append(kwargs),
            datetime=_MorningDatetime,
        ):
            placed = run.run_insider_trade_copier(
                gateway=gateway,
                risk=types.SimpleNamespace(equity=10_000.0, max_order_dollars=120.0),
                sec_candidates=sec_candidates,
                market_context={},
                equity_positions={},
                option_positions=[],
            )

        self.assertEqual(placed, 1)
        self.assertEqual(len(gateway.submit_calls), 1)
        self.assertEqual(gateway.submit_calls[0]["qty"], 3)
        self.assertEqual(captured[0]["order_id"], "order-1")


if __name__ == "__main__":
    unittest.main()
