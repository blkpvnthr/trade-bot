"""
Unit tests for tjl_signal.py — the deterministic Trend Join Long engine.

Hermetic: imports ONLY tjl_signal (stdlib + zoneinfo). Does NOT import run.py,
so it never triggers the production .env auto-load. Runs under both
`python -m pytest tests/test_tjl_signal.py` and `python -m unittest`.

Covers (per spec): trend filter, stochastic momentum filter, EMA crossover,
RTH guard, stop loss, take profit, EOD flatten, no pyramiding — plus config
parsing, Pine-faithful EMA seeding, and blocked-reason precedence.
"""

import os
import sys
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tjl_signal as T  # noqa: E402

NY = ZoneInfo("America/New_York")


def _ny(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=NY)


# Crafted 4-bar momentum data (stoch_len=3) — raw stoch = [None, None, 60, 70].
# With smooth_k=1, smooth_d=2: K[-1]=70, D[-1]=65 -> K>D and K<80 -> momentum OK.
_MOMO_HI = [11.0, 12.0, 13.0, 14.0]
_MOMO_LO = [8.0, 9.0, 10.0, 11.0]
_MOMO_CL_OK = [9.0, 11.0, 11.0, 12.5]    # raw3 = 70 (rising, <80)
_MOMO_CL_FALLING = [9.0, 11.0, 11.0, 11.5]   # raw3 = 50 (<60) -> K<D
_MOMO_CL_OB = [9.0, 11.0, 11.0, 13.25]   # raw3 = 85 (>80) -> overbought

# Entry series: flat then a jump up -> fast EMA crosses above slow EMA on last bar.
_ENTRY_CROSS = [100.0] * 30 + [110.0]
_ENTRY_NOCROSS = [100.0] * 31           # fast == slow everywhere -> no cross
_ENTRY_DOWN = [100.0] * 30 + [90.0]     # fast dips below slow -> no up-cross

# A small cfg matching the crafted momentum data, for composite tests.
_CFG = T.TJLConfig(stoch_len=3, smooth_k=1, smooth_d=2, stoch_ob=80,
                   trend_sma_len=3, fast_ema=9, slow_ema=21,
                   stop_pct=0.015, tp_pct=0.03, eod_flat=True)

_TREND_UP = [1.0, 2.0, 3.0]       # last 3.0 > sma3(=2.0)
_TREND_DOWN = [3.0, 2.0, 1.0]     # last 1.0 < sma3(=2.0)


class TestTrendFilter(unittest.TestCase):
    def test_trend_up_true(self):
        self.assertTrue(T.trend_up(list(range(1, 201)), 200))   # last 200 > sma 100.5

    def test_trend_up_false_below_sma(self):
        self.assertFalse(T.trend_up(list(range(200, 0, -1)), 200))  # last 1 < sma 100.5

    def test_trend_insufficient_data_is_false(self):
        self.assertFalse(T.trend_up([1.0, 2.0, 3.0], 200))   # < 200 bars -> conservative False


class TestMomentumFilter(unittest.TestCase):
    def test_momentum_ok(self):
        self.assertTrue(T.momentum_ok(_MOMO_HI, _MOMO_LO, _MOMO_CL_OK, 3, 1, 2, 80))

    def test_momentum_blocked_when_k_below_d(self):
        self.assertFalse(T.momentum_ok(_MOMO_HI, _MOMO_LO, _MOMO_CL_FALLING, 3, 1, 2, 80))

    def test_momentum_blocked_when_overbought(self):
        self.assertFalse(T.momentum_ok(_MOMO_HI, _MOMO_LO, _MOMO_CL_OB, 3, 1, 2, 80))

    def test_stoch_raw_values(self):
        raw = T.stoch_raw_series(_MOMO_HI, _MOMO_LO, _MOMO_CL_OK, 3)
        self.assertEqual(raw[0], None)
        self.assertEqual(raw[1], None)
        self.assertAlmostEqual(raw[2], 60.0)
        self.assertAlmostEqual(raw[3], 70.0)

    def test_stoch_zero_range_is_none(self):
        flat = [5.0, 5.0, 5.0]
        raw = T.stoch_raw_series(flat, flat, flat, 3)
        self.assertIsNone(raw[-1])   # HH==LL -> Pine NA


class TestEmaCrossover(unittest.TestCase):
    def test_ema_pine_seeding(self):
        # First EMA value is seeded with the source (Pine: na(sum[1]) ? src).
        self.assertEqual(T.ema_series([42.0, 1.0, 2.0], 9)[0], 42.0)

    def test_crossover_true(self):
        self.assertTrue(T.ema_crossover_up(_ENTRY_CROSS, 9, 21))

    def test_crossover_false_flat(self):
        self.assertFalse(T.ema_crossover_up(_ENTRY_NOCROSS, 9, 21))

    def test_crossover_false_on_downmove(self):
        self.assertFalse(T.ema_crossover_up(_ENTRY_DOWN, 9, 21))

    def test_crossover_last_semantics(self):
        # a crosses above b: a[1] <= b[1] and a[0] > b[0]
        self.assertTrue(T.crossover_last([1.0, 3.0], [2.0, 2.0]))
        self.assertFalse(T.crossover_last([3.0, 4.0], [2.0, 2.0]))  # already above last bar


class TestRthGuard(unittest.TestCase):
    def test_inside_rth(self):
        self.assertTrue(T.is_rth(_ny(2026, 6, 12, 10, 0)))    # Fri 10:00

    def test_open_edge_inclusive(self):
        self.assertTrue(T.is_rth(_ny(2026, 6, 12, 9, 30)))

    def test_before_open(self):
        self.assertFalse(T.is_rth(_ny(2026, 6, 12, 9, 29)))

    def test_close_edge_exclusive(self):
        self.assertFalse(T.is_rth(_ny(2026, 6, 12, 16, 0)))

    def test_weekend(self):
        self.assertFalse(T.is_rth(_ny(2026, 6, 13, 10, 0)))   # Saturday


class TestStopLoss(unittest.TestCase):
    def test_stop_hit(self):
        self.assertTrue(T.hit_stop(98.5, 100.0, 0.015))   # threshold = 98.5

    def test_stop_not_hit(self):
        self.assertFalse(T.hit_stop(98.6, 100.0, 0.015))

    def test_stop_ignored_without_entry_price(self):
        self.assertFalse(T.hit_stop(98.5, 0.0, 0.015))


class TestTakeProfit(unittest.TestCase):
    def test_tp_hit(self):
        self.assertTrue(T.hit_take_profit(103.0, 100.0, 0.03))   # threshold = 103.0

    def test_tp_not_hit(self):
        self.assertFalse(T.hit_take_profit(102.99, 100.0, 0.03))


class TestEodFlatten(unittest.TestCase):
    def test_session_end_true(self):
        self.assertTrue(T.is_session_end(_ny(2026, 6, 12, 15, 55), _ny(2026, 6, 12, 16, 0)))

    def test_session_end_false_midday(self):
        self.assertFalse(T.is_session_end(_ny(2026, 6, 12, 10, 0), _ny(2026, 6, 12, 10, 5)))

    def test_eod_exit_via_evaluate(self):
        sig = T.evaluate_exit(_CFG, price=100.0, avg_entry=100.0,
                              prev_ts=_ny(2026, 6, 12, 15, 55),
                              now_ts=_ny(2026, 6, 12, 16, 0), in_position=True)
        self.assertTrue(sig.exit)
        self.assertEqual(sig.reason, T.EOD_FLAT)


class TestNoPyramiding(unittest.TestCase):
    def test_already_in_position_blocks_entry(self):
        sig = T.evaluate_entry(_CFG, _TREND_UP, _MOMO_HI, _MOMO_LO, _MOMO_CL_OK,
                               _ENTRY_CROSS, _ny(2026, 6, 12, 10, 0),
                               in_position=True, log=False)
        self.assertFalse(sig.enter)
        self.assertEqual(sig.reason, T.ALREADY_IN_POSITION)


class TestEvaluateEntryComposite(unittest.TestCase):
    """All gates pass -> ENTER; each failing gate -> its reason, in precedence order."""

    def _entry(self, trend, mhi, mlo, mcl, entry, ts, in_pos):
        return T.evaluate_entry(_CFG, trend, mhi, mlo, mcl, entry, ts, in_pos, log=False)

    def test_happy_path_enters(self):
        sig = self._entry(_TREND_UP, _MOMO_HI, _MOMO_LO, _MOMO_CL_OK, _ENTRY_CROSS,
                          _ny(2026, 6, 12, 10, 0), False)
        self.assertTrue(sig.enter)
        self.assertEqual(sig.reason, T.ENTER)

    def test_outside_rth_precedence(self):
        sig = self._entry(_TREND_UP, _MOMO_HI, _MOMO_LO, _MOMO_CL_OK, _ENTRY_CROSS,
                          _ny(2026, 6, 13, 10, 0), False)   # Saturday
        self.assertEqual(sig.reason, T.OUTSIDE_RTH)

    def test_trend_not_up(self):
        sig = self._entry(_TREND_DOWN, _MOMO_HI, _MOMO_LO, _MOMO_CL_OK, _ENTRY_CROSS,
                          _ny(2026, 6, 12, 10, 0), False)
        self.assertEqual(sig.reason, T.TREND_NOT_UP)

    def test_momentum_not_ok(self):
        sig = self._entry(_TREND_UP, _MOMO_HI, _MOMO_LO, _MOMO_CL_FALLING, _ENTRY_CROSS,
                          _ny(2026, 6, 12, 10, 0), False)
        self.assertEqual(sig.reason, T.MOMENTUM_NOT_OK)

    def test_no_ema_cross(self):
        sig = self._entry(_TREND_UP, _MOMO_HI, _MOMO_LO, _MOMO_CL_OK, _ENTRY_NOCROSS,
                          _ny(2026, 6, 12, 10, 0), False)
        self.assertEqual(sig.reason, T.NO_EMA_CROSS)


class TestExitPrecedence(unittest.TestCase):
    def test_stop_before_tp(self):
        sig = T.evaluate_exit(_CFG, price=98.0, avg_entry=100.0,
                              prev_ts=_ny(2026, 6, 12, 10, 0),
                              now_ts=_ny(2026, 6, 12, 10, 5), in_position=True)
        self.assertTrue(sig.exit)
        self.assertEqual(sig.reason, T.STOP_LOSS)

    def test_take_profit(self):
        sig = T.evaluate_exit(_CFG, price=103.0, avg_entry=100.0,
                              prev_ts=_ny(2026, 6, 12, 10, 0),
                              now_ts=_ny(2026, 6, 12, 10, 5), in_position=True)
        self.assertEqual(sig.reason, T.TAKE_PROFIT)

    def test_no_exit_when_flat(self):
        sig = T.evaluate_exit(_CFG, price=50.0, avg_entry=100.0,
                              prev_ts=_ny(2026, 6, 12, 10, 0),
                              now_ts=_ny(2026, 6, 12, 10, 5), in_position=False)
        self.assertFalse(sig.exit)


class TestConfigFromEnv(unittest.TestCase):
    def test_defaults_match_pine(self):
        cfg = T.TJLConfig.from_env({})
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.trend_sma_len, 200)
        self.assertEqual(cfg.fast_ema, 9)
        self.assertEqual(cfg.slow_ema, 21)
        self.assertAlmostEqual(cfg.stop_pct, 0.015)
        self.assertAlmostEqual(cfg.tp_pct, 0.03)
        self.assertEqual(cfg.trend_tf_min, 60)
        self.assertEqual(cfg.momo_tf_min, 15)

    def test_env_overrides(self):
        env = {
            "TJL_ENABLED": "true", "TJL_TREND_TF": "30m", "TJL_TREND_SMA_LEN": "150",
            "TJL_MOMO_TF": "5m", "TJL_STOCH_LEN": "21", "TJL_SMOOTH_K": "5",
            "TJL_SMOOTH_D": "5", "TJL_STOCH_OB": "75", "TJL_FAST_EMA": "8",
            "TJL_SLOW_EMA": "20", "TJL_STOP_PCT": "0.02", "TJL_TP_PCT": "0.05",
            "TJL_EOD_FLAT": "false",
        }
        cfg = T.TJLConfig.from_env(env)
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.trend_tf_min, 30)
        self.assertEqual(cfg.trend_sma_len, 150)
        self.assertEqual(cfg.momo_tf_min, 5)
        self.assertEqual(cfg.stoch_ob, 75)
        self.assertAlmostEqual(cfg.stop_pct, 0.02)
        self.assertAlmostEqual(cfg.tp_pct, 0.05)
        self.assertFalse(cfg.eod_flat)


if __name__ == "__main__":
    unittest.main(verbosity=2)
