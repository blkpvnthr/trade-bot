"""
Tests for the trade-selection + risk-management overlay (rl_strategy_overlay). Pure-function
tests on synthetic trades — no dataset, no network. They assert the MECHANICS (gating,
sizing bounds, daily loss cap, no-look-ahead), not a backtest outcome.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rl_strategy_overlay as so


class QualityScoreTests(unittest.TestCase):
    def test_monotonic_and_bounded(self):
        low = so.quality_score({"rel_volume": 1.0, "mtf_alignment": 0.0, "catalyst_strength": 0.0})
        high = so.quality_score({"rel_volume": 3.0, "mtf_alignment": 1.0, "catalyst_strength": 1.0,
                                 "sector_regime": 1.0, "setup_score": 100})
        self.assertLess(low, high)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)

    def test_uses_only_exante_fields(self):
        # realized_R / peak_R must NOT change the score (no look-ahead)
        base = {"rel_volume": 2.0, "mtf_alignment": 0.5}
        self.assertEqual(so.quality_score(base),
                         so.quality_score({**base, "realized_R": 3.0, "peak_R": 9.9}))


class PositionRiskTests(unittest.TestCase):
    def test_quality_sizing_band(self):
        p = dict(so.DEFAULT_PARAMS)
        lo = so.position_risk(p, quality=0.0, atr_pct=p["target_atr"], equity_dd=0.0)
        hi = so.position_risk(p, quality=1.0, atr_pct=p["target_atr"], equity_dd=0.0)
        self.assertAlmostEqual(lo, p["base_risk"] * 0.5, places=6)
        self.assertAlmostEqual(hi, p["base_risk"] * 1.5, places=6)

    def test_drawdown_derisk_reduces_size(self):
        p = dict(so.DEFAULT_PARAMS)
        flat = so.position_risk(p, 0.5, p["target_atr"], equity_dd=0.0)
        drawn = so.position_risk(p, 0.5, p["target_atr"], equity_dd=0.2)
        self.assertLess(drawn, flat)
        self.assertGreaterEqual(drawn, 0.0)

    def test_vol_target_caps(self):
        p = dict(so.DEFAULT_PARAMS)
        high_vol = so.position_risk(p, 0.5, atr_pct=1.0, equity_dd=0.0)   # huge atr → min 0.5x
        self.assertGreater(high_vol, 0.0)


class SelectTests(unittest.TestCase):
    def _trades(self, n, q_field="rel_volume"):
        # alternating quality, all same day
        return [{"date_tag": "20260520", "timestamp": f"t{i:03d}",
                 q_field: 1.0 + (i % 5) * 0.5, "realized_R": 1.0} for i in range(n)]

    def test_quality_gate_filters(self):
        trades = self._trades(50)
        sel = so.select(trades, quality_min=0.5, max_per_day=999)
        self.assertTrue(all(so.quality_score(t) >= 0.5 for t in sel))

    def test_per_day_cap(self):
        trades = self._trades(50)
        sel = so.select(trades, quality_min=0.0, max_per_day=5)
        self.assertEqual(len(sel), 5)        # one day, capped at 5


class SimulateTests(unittest.TestCase):
    def test_daily_loss_cap_stops_the_day(self):
        # Five −1R trades same day at 0.5% risk: a −1% daily cap should stop after ~2 losers.
        trades = [{"date_tag": "20260520", "timestamp": f"t{i}", "realized_R": -1.0,
                   "rel_volume": 1.0, "_q": 0.0} for i in range(5)]
        p = {**so.DEFAULT_PARAMS, "base_risk": 0.005, "daily_loss_cap": 0.01,
             "quality_sizing": False, "vol_target": False, "derisk_k": 0.0}
        m = so.simulate(trades, p)
        self.assertLess(m["trades"], 5)      # cap halted the day before all 5 ran
        self.assertGreaterEqual(m["trades"], 1)

    def test_metrics_sane(self):
        trades = [{"date_tag": "20260520", "timestamp": f"t{i}",
                   "realized_R": 3.0 if i % 3 else -1.0, "rel_volume": 1.5, "_q": 0.4}
                  for i in range(30)]
        m = so.simulate(trades, dict(so.DEFAULT_PARAMS))
        self.assertEqual(m["trades"], 30)
        self.assertTrue(0.0 <= m["win_rate"] <= 1.0)
        self.assertGreaterEqual(m["max_drawdown_pct"], 0.0)


class NoLeakTests(unittest.TestCase):
    def test_threshold_from_percentile_in_distribution(self):
        trades = [{"rel_volume": 1.0 + i * 0.1} for i in range(20)]
        thr = so.quality_threshold_for_percentile(trades, 0.5)
        qs = sorted(so.quality_score(t) for t in trades)
        self.assertGreaterEqual(thr, qs[0])
        self.assertLessEqual(thr, qs[-1])

    def test_tune_returns_grid_value(self):
        trades = [{"date_tag": "20260520", "rel_volume": 1.0 + (i % 4) * 0.5,
                   "realized_R": 3.0 if i % 2 else -1.0} for i in range(400)]
        pct = so.tune_quality_percentile(trades, grid=(0.0, 0.5), min_trades=10)
        self.assertIn(pct, (0.0, 0.5))


if __name__ == "__main__":
    unittest.main()
