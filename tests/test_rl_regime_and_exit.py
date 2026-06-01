"""
Tests for the two validated 'move-the-needle' levers:
  * rl_regime_study  — a SPY-200DMA regime gate cuts drawdown in a bear regime.
  * rl_dynamic_exit  — an asymmetric scale-out/trail exit beats the fixed 3:1 on reversals,
                       and the bracket is path-conservative (stop checked before target).
Pure/offline — synthetic bars only, no network.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rl_regime_study as rg
import rl_dynamic_exit as dx


def _frame(rows):
    import pandas as pd
    idx = pd.date_range("2026-05-20 13:35", periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(rows, index=idx)


class DynamicExitTests(unittest.TestCase):
    def test_fixed_hits_target_and_stop(self):
        risk = 1.0
        up = _frame([{"open": 100, "high": 100 + 3.0 * (i + 1), "low": 100, "close": 100} for i in range(3)])
        self.assertEqual(dx.fixed_bracket(100.0, risk, up), 3.0)
        down = _frame([{"open": 100, "high": 100, "low": 100 - 1.5 * (i + 1), "close": 99} for i in range(3)])
        self.assertEqual(dx.fixed_bracket(100.0, risk, down), -1.0)

    def test_fixed_stop_checked_before_target(self):
        # a single bar that spans both the stop and the target → conservative = stop (−1R)
        bar = _frame([{"open": 100, "high": 104, "low": 98.9, "close": 100}])
        self.assertEqual(dx.fixed_bracket(100.0, 1.0, bar), -1.0)

    def test_dynamic_beats_fixed_on_reversal(self):
        # pop to +1.2R then reverse through entry: fixed → −1R; dynamic locks +1R on 1/3, rest @ breakeven
        risk = 1.0
        bars = _frame([{"open": 100, "high": 101.2, "low": 100, "close": 101},   # +1.2R high → scale 1/3, stop→BE
                       {"open": 101, "high": 101, "low": 99.0, "close": 99}])     # reverses to BE/stop
        fixed = dx.fixed_bracket(100.0, risk, bars)
        dyn = dx.dynamic_exit(100.0, risk, bars)
        self.assertEqual(fixed, -1.0)
        self.assertGreater(dyn, fixed)
        self.assertAlmostEqual(dyn, 1.0 / 3.0, places=3)     # +1R on first third, remainder at breakeven

    def test_dynamic_clean_loser_is_about_minus_one(self):
        bars = _frame([{"open": 100, "high": 100.0, "low": 98.0, "close": 98.5}])
        self.assertLessEqual(dx.dynamic_exit(100.0, 1.0, bars), -0.9)


class RegimeGateTests(unittest.TestCase):
    def _daily(self, closes):
        import pandas as pd
        idx = pd.date_range("2020-01-01", periods=len(closes), freq="D", tz="UTC")
        return pd.DataFrame({"open": closes, "high": [c * 1.001 for c in closes],
                             "low": [c * 0.999 for c in closes], "close": closes,
                             "volume": [1e6] * len(closes)}, index=idx)

    def test_gate_goes_to_cash_below_spy_200dma(self):
        # Hand-built book: the name keeps signalling long, but SPY is below its 200-DMA on the
        # down days. The gate must go to CASH there → it avoids the losses → smaller drawdown.
        import datetime as _dt
        d = [_dt.date(2022, 1, i + 1) for i in range(3)]
        book = {
            "all_dates": d,
            "sym_signal": {"AAA": {d[0]: (True, 0.01), d[1]: (True, -0.05), d[2]: (True, -0.05)}},
            "spy_px": {d[0]: 110.0, d[1]: 90.0, d[2]: 88.0},     # below 200DMA on d1,d2
            "spy_sma200": {d[0]: 100.0, d[1]: 100.0, d[2]: 100.0},
        }
        take_all = rg.simulate_book(book, regime_gate=False)
        gated = rg.simulate_book(book, regime_gate=True)
        self.assertLess(gated["max_drawdown_pct"], take_all["max_drawdown_pct"])
        self.assertEqual(gated["invested_days"], 1)             # only the bull day was traded

    def test_year_drawdowns_structure(self):
        import datetime as _dt
        dated = [(_dt.date(2022, 1, 1), -0.1), (_dt.date(2022, 6, 1), 0.05),
                 (_dt.date(2023, 1, 1), 0.02)]
        yd = rg._year_drawdowns(dated)
        self.assertIn("2022", yd)
        self.assertGreaterEqual(yd["2022"], 0.0)


if __name__ == "__main__":
    unittest.main()
