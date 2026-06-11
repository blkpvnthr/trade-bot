"""Tests for intraday_reversal_engine — the isolated intraday reversal candidate source.

Covers the universe scanner gates, the pure scorer's component weights (each
"scores higher when X" lever), the reversal-trigger requirement, and the
order-emission floor (no orders below min score), plus a bar-derived sanity check.
All offline — no Alpaca, no network.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import intraday_reversal_engine as ire
from intraday_reversal_engine import (
    EngineConfig, ReversalInput, score_candidate, scan, propose_entries, live_orders_allowed,
)

CFG = EngineConfig(enabled=True, paper_only=True, min_score=75.0,
                   max_new_entries_per_day=2, max_open_positions=2)


def _base(**kw) -> ReversalInput:
    """A clean candidate that clears the universe filter and scores ~100."""
    d = dict(symbol="TST", price=20.0, volume=4_000_000, avg_volume=1_500_000,
             chg_pct=-1.0, perf_1w=-3.0, analyst_rating="strong buy", os_rating="strong buy",
             rvol=1.6, vwap_reclaim=True, rsi_rising=True, rsi_cross_40=False,
             close_gt_open=True, sweep_reclaim=False, orb_reclaim=False, above_ema20_5m=True,
             recovering=True, extended_atr=0.5, atr=0.4)
    d.update(kw)
    return ReversalInput(**d)


class UniverseGateTests(unittest.TestCase):
    def test_passes_full_universe_filter(self):
        c = score_candidate(_base(), CFG)
        self.assertTrue(c.passed_universe)
        self.assertNotIn(c.blocked_by, ("price_below_5", "volume_below_3m", "chg_not_negative",
                                        "perf_1w_not_negative", "analyst_not_strong_buy",
                                        "os_not_strong_buy", "untradable_or_halted"))
        self.assertGreater(c.score, 0)

    def test_rejects_price_le_5(self):
        c = score_candidate(_base(price=5.0), CFG)
        self.assertFalse(c.passed_universe)
        self.assertEqual(c.blocked_by, "price_below_5")

    def test_rejects_volume_le_3m(self):
        c = score_candidate(_base(volume=3_000_000), CFG)
        self.assertFalse(c.passed_universe)
        self.assertEqual(c.blocked_by, "volume_below_3m")

    def test_rejects_chg_ge_0(self):
        c = score_candidate(_base(chg_pct=0.0), CFG)
        self.assertFalse(c.passed_universe)
        self.assertEqual(c.blocked_by, "chg_not_negative")

    def test_rejects_perf_1w_ge_0(self):
        c = score_candidate(_base(perf_1w=0.0), CFG)
        self.assertFalse(c.passed_universe)
        self.assertEqual(c.blocked_by, "perf_1w_not_negative")

    def test_rejects_analyst_not_strong_buy(self):
        c = score_candidate(_base(analyst_rating="buy"), CFG)
        self.assertFalse(c.passed_universe)
        self.assertEqual(c.blocked_by, "analyst_not_strong_buy")

    def test_rejects_os_not_strong_buy(self):
        c = score_candidate(_base(os_rating="neutral"), CFG)
        self.assertFalse(c.passed_universe)
        self.assertEqual(c.blocked_by, "os_not_strong_buy")

    def test_rejects_halted(self):
        c = score_candidate(_base(halted=True), CFG)
        self.assertFalse(c.passed_universe)
        self.assertEqual(c.blocked_by, "untradable_or_halted")


class ScoreLeverTests(unittest.TestCase):
    def test_rvol_above_2_scores_higher(self):
        lo = score_candidate(_base(rvol=1.6), CFG).score
        hi = score_candidate(_base(rvol=2.5), CFG).score
        self.assertGreater(hi, lo)

    def test_vwap_reclaim_scores_higher(self):
        without = score_candidate(_base(vwap_reclaim=False), CFG).score
        with_ = score_candidate(_base(vwap_reclaim=True), CFG).score
        self.assertGreater(with_, without)

    def test_rsi_cross_40_scores_higher(self):
        without = score_candidate(_base(rsi_cross_40=False), CFG).score
        with_ = score_candidate(_base(rsi_cross_40=True), CFG).score
        self.assertGreater(with_, without)

    def test_orb_reclaim_after_sweep_scores_higher(self):
        without = score_candidate(_base(sweep_reclaim=False, orb_reclaim=False), CFG).score
        with_ = score_candidate(_base(sweep_reclaim=True, orb_reclaim=True), CFG).score
        self.assertGreater(with_, without)

    def test_over_extension_penalizes(self):
        normal = score_candidate(_base(extended_atr=0.5), CFG).score
        extended = score_candidate(_base(extended_atr=4.5), CFG).score
        self.assertLess(extended, normal)

    def test_components_recorded(self):
        c = score_candidate(_base(rvol=2.5), CFG)
        self.assertIn("vwap_reclaim", c.score_components)
        self.assertIn("rvol_gt_2_0", c.score_components)
        self.assertEqual(c.score_components["vwap_reclaim"], 20)
        self.assertEqual(c.score, round(sum(c.score_components.values()), 2))


class TriggerAndEmissionTests(unittest.TestCase):
    def _low_score(self) -> ReversalInput:
        # Clears universe + has a VWAP-reclaim trigger, but only ~45 pts (< min 75).
        return _base(avg_volume=500_000, rvol=1.0, perf_1w=-1.0, recovering=False,
                     rsi_rising=False, rsi_cross_40=False, close_gt_open=False,
                     above_ema20_5m=False, vwap_reclaim=True)

    def test_no_trigger_is_not_a_setup(self):
        c = score_candidate(_base(vwap_reclaim=False, sweep_reclaim=False, orb_reclaim=False,
                                  rsi_cross_40=False), CFG)
        self.assertFalse(c.passed_setup)
        self.assertEqual(c.blocked_by, "no_reversal_trigger")

    def test_below_min_score_marked_blocked(self):
        c = score_candidate(self._low_score(), CFG)
        self.assertTrue(c.passed_setup)        # has a trigger
        self.assertLess(c.score, CFG.min_score)
        self.assertFalse(c.eligible)
        self.assertEqual(c.blocked_by, "below_min_score")

    def test_does_not_emit_orders_below_min_score(self):
        res = scan([self._low_score()], CFG)
        orders = propose_entries(res, equity=25_000.0, open_symbols=set(), cfg=CFG)
        self.assertEqual(orders, [])

    def test_emits_marketable_limit_order_when_eligible(self):
        res = scan([_base(symbol="GOOD")], CFG)
        orders = propose_entries(res, equity=25_000.0, open_symbols=set(), cfg=CFG)
        self.assertEqual(len(orders), 1)
        o = orders[0]
        self.assertEqual((o.symbol, o.side, o.order_type), ("GOOD", "buy", "limit"))
        self.assertGreater(o.limit_price, _base().price)   # marketable (above last)
        self.assertGreater(o.qty, 0)

    def test_never_averages_into_open_symbol(self):
        res = scan([_base(symbol="HELD")], CFG)
        orders = propose_entries(res, equity=25_000.0, open_symbols={"HELD"}, cfg=CFG)
        self.assertEqual(orders, [])

    def test_respects_engine_caps(self):
        rows = [_base(symbol=f"S{i}", volume=4_000_000 + i) for i in range(5)]
        res = scan(rows, CFG)
        orders = propose_entries(res, equity=25_000.0, open_symbols=set(), cfg=CFG)
        self.assertLessEqual(len(orders), CFG.max_new_entries_per_day)
        self.assertLessEqual(len(orders), CFG.max_open_positions)


class SortingAndFunnelTests(unittest.TestCase):
    def test_sorted_by_volume_desc_then_score(self):
        rows = [_base(symbol="LOWV", volume=3_500_000),
                _base(symbol="HIGHV", volume=9_000_000)]
        res = scan(rows, CFG)
        self.assertEqual([c.symbol for c in res.candidates], ["HIGHV", "LOWV"])

    def test_funnel_has_required_fields(self):
        res = scan([_base(), _base(symbol="X", chg_pct=0.5)], CFG)
        f = res.funnel()
        for k in ("scanned", "passed_universe", "passed_setup", "eligible_for_entry",
                  "rejected_by", "top_candidates"):
            self.assertIn(k, f)
        self.assertEqual(f["scanned"], 2)
        self.assertIn("chg_not_negative", f["rejected_by"])
        self.assertIn("score_components", f["top_candidates"][0])


class PaperGateTests(unittest.TestCase):
    def test_paper_only_blocks_live(self):
        cfg = EngineConfig(paper_only=True)
        self.assertFalse(live_orders_allowed(cfg, account_mode="live", system_paper_only=False))

    def test_live_requires_live_mode_and_paper_only_false(self):
        cfg = EngineConfig(paper_only=False)
        self.assertTrue(live_orders_allowed(cfg, "live", False))
        self.assertFalse(live_orders_allowed(cfg, "paper", False))
        self.assertFalse(live_orders_allowed(cfg, "live", True))


class BarDerivationTests(unittest.TestCase):
    def test_vwap_reclaim_derived_from_bars(self):
        import pandas as pd
        # Build bars that dip below VWAP then reclaim above it on the last bar.
        idx = pd.date_range("2026-05-04 13:30", periods=20, freq="1min", tz="UTC")
        # 10 bars flat at 100 (pulls VWAP to ~100), a dip to 98 below VWAP, then a
        # decisive reclaim to 102 on the last bar.
        close = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                 98, 98, 98, 98, 98, 98, 98, 98, 98, 102]
        df = pd.DataFrame({"open": [c - 0.1 for c in close], "high": [c + 0.2 for c in close],
                           "low": [c - 0.2 for c in close], "close": close,
                           "volume": [50000] * 20}, index=idx)
        inp = _base(symbol="BARS", bars=df, vwap_reclaim=None, rsi_rising=None,
                    rsi_cross_40=None, close_gt_open=None, sweep_reclaim=None, orb_reclaim=None,
                    above_ema20_5m=None, recovering=None, extended_atr=None, atr=None)
        c = score_candidate(inp, CFG)
        self.assertTrue(c.passed_universe)
        self.assertTrue(c.metrics["vwap_reclaim"])     # reclaimed VWAP on the recovery leg
        self.assertIn("vwap_reclaim", c.score_components)


if __name__ == "__main__":
    unittest.main()
