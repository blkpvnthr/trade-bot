import types
import unittest
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run


ET = ZoneInfo("America/New_York")


def _bar(symbol: str, hhmm: str, open_: float, high: float, low: float, close: float, volume: int = 1000):
    ts = datetime.strptime(f"2026-05-28 {hhmm}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    return {
        "symbol": symbol,
        "timestamp": ts.astimezone(ZoneInfo("UTC")),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


class FirstCandleRuleTests(unittest.TestCase):
    def _scanner(self, bars: pd.DataFrame, enabled: bool = True):
        scanner = run.MomentumSetupScanner.__new__(run.MomentumSetupScanner)
        scanner.config = run.ScannerConfig(
            first_candle_rule_enabled=enabled,
            first_candle_rule_require_volume_confirmation=False,
        )
        scanner.executor_config = types.SimpleNamespace(
            max_position_weight=0.10,
            min_weight_delta=0.0001,
            min_notional_to_trade=1.0,
        )
        scanner._intraday_bar_cache = {}
        scanner._download_intraday_bars = lambda symbols, minutes: bars.copy()
        return scanner

    def test_bullish_first_candle_creates_3r_bracket_buy(self):
        bars = pd.DataFrame(
            [
                _bar("AAPL", "09:30", 10.40, 10.48, 10.35, 10.45, 2000),
                _bar("AAPL", "09:31", 10.45, 10.55, 10.42, 10.50, 2100),
                _bar("AAPL", "09:32", 10.50, 10.60, 10.48, 10.57, 2200),
                _bar("AAPL", "09:33", 10.57, 10.68, 10.54, 10.63, 2300),
                _bar("AAPL", "09:34", 10.63, 10.72, 10.60, 10.60, 2400),
            ]
        )
        scanner = self._scanner(bars)

        rows = scanner._build_first_candle_rule_orders(["AAPL"], {}, 13000.0)

        self.assertEqual(len(rows), 1)
        row = rows.iloc[0]
        self.assertEqual(row["symbol"], "AAPL")
        self.assertEqual(row["side"], "buy")
        self.assertEqual(row["source"], "first_candle_rule_935")
        self.assertEqual(row["entry_type"], "first_5m_close")
        self.assertAlmostEqual(row["price"], 10.60, places=4)
        self.assertAlmostEqual(row["stop_price"], 10.34, places=4)
        self.assertAlmostEqual(row["target_price"], 11.38, places=4)
        self.assertAlmostEqual(row["risk_reward_ratio"], 3.0, places=4)
        self.assertTrue(run.row_has_complete_bracket_plan(row))

    def test_bearish_first_candle_is_blocked_by_first_candle_rule(self):
        bars = pd.DataFrame(
            [
                _bar("AAPL", "09:30", 10.60, 10.65, 10.45, 10.55),
                _bar("AAPL", "09:31", 10.55, 10.58, 10.40, 10.50),
                _bar("AAPL", "09:32", 10.50, 10.52, 10.35, 10.42),
                _bar("AAPL", "09:33", 10.42, 10.45, 10.31, 10.38),
                _bar("AAPL", "09:34", 10.38, 10.40, 10.30, 10.34),
            ]
        )
        scanner = self._scanner(bars)

        sig = scanner._first_candle_rule_signal_for_symbol("AAPL", bars)
        rows = scanner._build_first_candle_rule_orders(["AAPL"], {}, 13000.0)

        self.assertFalse(sig["available"])
        self.assertEqual(sig["blocked_by"], "first_candle_rule")
        self.assertEqual(sig["reason"], "first_candle_not_bullish")
        self.assertTrue(rows.empty)

    def test_incomplete_first_candle_creates_no_order(self):
        bars = pd.DataFrame(
            [
                _bar("AAPL", "09:30", 10.40, 10.48, 10.35, 10.45),
                _bar("AAPL", "09:31", 10.45, 10.55, 10.42, 10.50),
                _bar("AAPL", "09:32", 10.50, 10.60, 10.48, 10.57),
                _bar("AAPL", "09:33", 10.57, 10.68, 10.54, 10.63),
            ]
        )
        scanner = self._scanner(bars)

        rows = scanner._build_first_candle_rule_orders(["AAPL"], {}, 13000.0)

        self.assertTrue(rows.empty)

    def test_stop_distance_outside_limits_blocks_setup(self):
        bars = pd.DataFrame(
            [
                _bar("AAPL", "09:30", 10.400, 10.402, 10.399, 10.401),
                _bar("AAPL", "09:31", 10.401, 10.403, 10.400, 10.402),
                _bar("AAPL", "09:32", 10.402, 10.404, 10.401, 10.403),
                _bar("AAPL", "09:33", 10.403, 10.405, 10.402, 10.404),
                _bar("AAPL", "09:34", 10.404, 10.405, 10.403, 10.405),
            ]
        )
        scanner = self._scanner(bars)

        sig = scanner._first_candle_rule_signal_for_symbol("AAPL", bars)

        self.assertFalse(sig["available"])
        self.assertIn(sig["reason"], {"first_candle_range_too_tight", "stop_distance_too_small"})

    def test_disabled_feature_returns_no_rows(self):
        bars = pd.DataFrame(
            [
                _bar("AAPL", "09:30", 10.40, 10.48, 10.35, 10.45),
                _bar("AAPL", "09:31", 10.45, 10.55, 10.42, 10.50),
                _bar("AAPL", "09:32", 10.50, 10.60, 10.48, 10.57),
                _bar("AAPL", "09:33", 10.57, 10.68, 10.54, 10.63),
                _bar("AAPL", "09:34", 10.63, 10.72, 10.60, 10.60),
            ]
        )
        scanner = self._scanner(bars, enabled=False)

        rows = scanner._build_first_candle_rule_orders(["AAPL"], {}, 13000.0)

        self.assertTrue(rows.empty)


if __name__ == "__main__":
    unittest.main()
