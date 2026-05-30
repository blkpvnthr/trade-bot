import json
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import run
from ensure_runtime_outputs import ensure_runtime_outputs


class RuntimeOutputRecoveryTests(unittest.TestCase):
    def test_ensure_runtime_outputs_repairs_one_byte_csvs_with_headers(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            system_dir = root / "system-data" / "20260527"
            log_dir = root / "logs" / "execution" / "20260527"
            system_dir.mkdir(parents=True)
            log_dir.mkdir(parents=True)
            bad = system_dir / "20260527_daily_walkforward_emulated_rebalance_orders.csv"
            bad.write_text("\n", encoding="utf-8")

            result = ensure_runtime_outputs("20260527", root=root)

            df = pd.read_csv(bad)
            self.assertEqual(len(df), 0)
            self.assertIn("symbol", df.columns)
            self.assertIn("stop_price", df.columns)
            self.assertIn("target_price", df.columns)
            self.assertTrue((log_dir / "20260527_execution_events.jsonl").exists())
            self.assertGreaterEqual(len(result["csv_initialized"]), 1)


class PlannerOrderRecoveryTests(unittest.TestCase):
    def _make_executor(self):
        executor = run.ProductionOrderExecutor.__new__(run.ProductionOrderExecutor)
        executor.config = run.build_default_config()
        executor.config.max_order_dollars = 1500.0
        executor.config.max_single_order_qty = 100000
        executor.config.min_notional_to_trade = 1.0
        executor.config.scanner.use_rl_timing_gate = True
        executor.broker = types.SimpleNamespace(get_all_open_positions=lambda: [])
        return executor

    def test_valid_premarket_plans_generate_executable_bracket_rows(self):
        executor = self._make_executor()
        payload = json.loads(Path("system-data/20260527/20260527_premarket_trade_plan.json").read_text())

        rows = executor._premarket_plan_to_order_rows(payload, equity=13000.0)

        by_symbol = {row["symbol"]: row for row in rows}
        for symbol in ["MNTS", "PTON", "SLF", "MDA", "CPSH", "HLIT", "ASPI", "GRRR"]:
            self.assertIn(symbol, by_symbol)
            row = by_symbol[symbol]
            self.assertEqual(row["side"], "buy")
            self.assertGreater(row["order_qty"], 0)
            self.assertGreater(row["price"], row["stop_price"])
            self.assertGreater(row["target_price"], row["price"])
            self.assertTrue(run.row_has_complete_bracket_plan(row))

        self.assertEqual(by_symbol["MNTS"]["price"], 19.5195)
        self.assertEqual(by_symbol["MNTS"]["stop_price"], 19.3925)
        self.assertEqual(by_symbol["MNTS"]["target_r_multiple"], 5.0)
        self.assertAlmostEqual(by_symbol["MNTS"]["target_price"], 20.1545, places=4)

    def test_rl_wait_does_not_remove_complete_bracket_buy(self):
        executor = self._make_executor()
        executor.config.scanner.rl_shadow_mode = False
        executor._download_latest_bars_for_rl = lambda symbols: pd.DataFrame()
        executor._load_rl_structure_map = lambda: {}
        executor._predict_rl_signal = lambda symbol, bars, structure, positions: {
            "symbol": symbol,
            "action": "HOLD",
            "confidence": 0.0,
            "reason": "test_wait",
        }

        rebalance = pd.DataFrame(
            [
                {
                    "symbol": "MNTS",
                    "side": "buy",
                    "delta_shares": 10,
                    "delta_dollars": 195.195,
                    "price": 19.5195,
                    "entry_price": 19.5195,
                    "stop_price": 19.4025,
                    "target_price": 21.45,
                    "stop_loss": 19.4025,
                    "take_profit": 21.45,
                    "reason": "premarket_trade_plan",
                }
            ]
        )

        with patch.object(run, "write_event", lambda *args, **kwargs: None), patch.object(run, "save_df", lambda *args, **kwargs: None):
            out, exits = executor._apply_rl_timing_gate(rebalance, pd.DataFrame())

        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["symbol"], "MNTS")
        self.assertEqual(out.iloc[0]["rl_warning"], "rl_timing_not_confirmed")
        self.assertTrue(exits.empty)


class BracketSubmissionTests(unittest.TestCase):
    def test_complete_buy_uses_bracket_and_never_naked_market_by_default(self):
        executor = run.ProductionOrderExecutor.__new__(run.ProductionOrderExecutor)
        executor.config = run.build_default_config()
        executor.config.scanner.require_direction_confirmation_before_buy = False
        executor.config.scanner.stream_confirmed_append_direction_rejects = False
        executor.config.scanner.stream_confirmed_submit_trailing_stop_after_fill = False
        executor.config.min_notional_to_trade = 1.0
        executor.scanner = None

        calls = {"bracket": 0, "market": 0}

        class Broker:
            def get_position(self, symbol):
                return run.BrokerPosition(symbol=symbol, qty=0)

            def has_open_order_for_symbol(self, symbol):
                return False

            def submit_bracket_buy(self, symbol, qty, stop_price, target_price, client_order_id, tif):
                calls["bracket"] += 1
                self.payload = {
                    "symbol": symbol,
                    "qty": qty,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "client_order_id": client_order_id,
                }
                return types.SimpleNamespace(id="order-1", status="submitted")

            def submit_market_buy(self, *args, **kwargs):
                calls["market"] += 1
                return types.SimpleNamespace(id="market-1", status="submitted")

            def cancel_open_orders_for_symbol(self, symbol):
                return None

            def get_all_open_positions(self):
                return []

        broker = Broker()
        executor.broker = broker
        executor._record_account_snapshot = lambda label, cycle_id: {
            "equity": 13000.0,
            "cash": 13000.0,
            "buying_power": 26000.0,
        }
        executor._gross_exposure = lambda equity: 0.0
        executor._latest_feature_by_symbol = lambda: {}
        executor._cash_buffer = lambda equity: 0.0
        executor._max_order_notional = lambda equity: 1500.0
        executor._risk_profile = lambda equity: types.SimpleNamespace(max_gross_exposure=1.0, min_notional_to_trade=1.0)
        executor._read_kill_state = lambda: {"start_equity": 13000.0}
        executor._mark_order = lambda: None
        executor._poll_fill = lambda order_id, price: {"fill_status": "submitted", "filled_qty": 0, "filled_avg_price": 0.0, "slippage_pct": 0.0}
        executor._order_log_context = lambda **kwargs: {}
        executor._journal_order = lambda *args, **kwargs: None
        executor._safe_log_live_experience = lambda **kwargs: None

        plan = pd.DataFrame(
            [
                {
                    "symbol": "MNTS",
                    "side": "buy",
                    "order_qty": 10,
                    "price": 19.5195,
                    "capped_dollars": 195.195,
                    "stop_price": 19.4025,
                    "target_price": 21.45,
                    "client_order_id": "qq_20260527_test_MNTS_buy",
                    "reason": "test_bracket",
                }
            ]
        )

        with patch.object(run, "_BRACKET_ORDERS_AVAILABLE", True), patch.object(run, "log_executed_order", lambda **kwargs: None), patch.object(run, "write_event", lambda *args, **kwargs: None):
            results = executor._execute_order_plan(plan, "cycle-1")

        self.assertEqual(calls["bracket"], 1)
        self.assertEqual(calls["market"], 0)
        self.assertEqual(broker.payload["stop_price"], 19.4025)
        self.assertEqual(broker.payload["target_price"], 21.45)
        self.assertTrue(bool(results.iloc[0]["bracket_attached"]))


if __name__ == "__main__":
    unittest.main()
