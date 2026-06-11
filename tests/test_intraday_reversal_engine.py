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

from datetime import datetime
from zoneinfo import ZoneInfo

import intraday_reversal_engine as ire
from intraday_reversal_engine import (
    EngineConfig, ReversalInput, score_candidate, scan, propose_entries, live_orders_allowed,
    select_call, estimate_option_target_premium, size_option, propose_option_entries,
    propose_for_engine, previous_day_high, _occ_symbol,
    is_active_window, manage_eod, next_day_exit_decision, ManagedPosition,
)

_ET = ZoneInfo("America/New_York")

CFG = EngineConfig(enabled=True, paper_only=True, min_score=75.0,
                   max_new_entries_per_day=2, max_open_positions=2)


def _base(**kw) -> ReversalInput:
    """A clean candidate that clears the universe filter (green today, weak 1w, vol>1M,
    rel_vol_5m>1, price>5, analyst+os strong buy) and scores ~100."""
    d = dict(symbol="TST", price=20.0, volume=4_000_000, avg_volume=1_500_000,
             chg_pct=+1.0, perf_1w=-3.0, analyst_rating="strong buy", os_rating="strong buy",
             rvol=1.6, rel_vol_5m=1.5, vwap_reclaim=True, rsi_rising=True, rsi_cross_40=False,
             close_gt_open=True, sweep_reclaim=False, orb_reclaim=False, above_ema20_5m=True,
             recovering=True, extended_atr=0.5, atr=0.4)
    d.update(kw)
    return ReversalInput(**d)


class UniverseGateTests(unittest.TestCase):
    def test_passes_full_universe_filter(self):
        c = score_candidate(_base(), CFG)
        self.assertTrue(c.passed_universe)
        self.assertNotIn(c.blocked_by, ("price_below_5", "volume_below_1m", "chg_not_positive",
                                        "perf_1w_not_negative", "rel_vol_5m_below_1",
                                        "analyst_not_strong_buy", "os_not_strong_buy",
                                        "untradable_or_halted"))
        self.assertGreater(c.score, 0)

    def test_rejects_price_le_5(self):
        c = score_candidate(_base(price=5.0), CFG)
        self.assertFalse(c.passed_universe)
        self.assertEqual(c.blocked_by, "price_below_5")

    def test_rejects_volume_le_1m(self):
        c = score_candidate(_base(volume=1_000_000), CFG)
        self.assertFalse(c.passed_universe)
        self.assertEqual(c.blocked_by, "volume_below_1m")

    def test_rejects_chg_le_0_not_green(self):
        self.assertEqual(score_candidate(_base(chg_pct=0.0), CFG).blocked_by, "chg_not_positive")
        self.assertEqual(score_candidate(_base(chg_pct=-1.5), CFG).blocked_by, "chg_not_positive")

    def test_rejects_rel_vol_5m_le_1(self):
        c = score_candidate(_base(rel_vol_5m=0.9), CFG)
        self.assertFalse(c.passed_universe)
        self.assertEqual(c.blocked_by, "rel_vol_5m_below_1")

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
    def test_sorted_by_5min_volume_desc(self):
        # 5-min volume drives the order even when DAY volume disagrees: B has lower
        # day volume but higher 5-min volume, so it ranks first.
        rows = [_base(symbol="A", volume=9_000_000, vol_5m=100_000),
                _base(symbol="B", volume=3_500_000, vol_5m=500_000)]
        res = scan(rows, CFG)
        self.assertEqual([c.symbol for c in res.candidates], ["B", "A"])

    def test_sort_falls_back_to_day_volume_when_no_5min(self):
        rows = [_base(symbol="LOWV", volume=3_500_000, vol_5m=None),
                _base(symbol="HIGHV", volume=9_000_000, vol_5m=None)]
        res = scan(rows, CFG)
        self.assertEqual([c.symbol for c in res.candidates], ["HIGHV", "LOWV"])

    def test_funnel_has_required_fields(self):
        res = scan([_base(vol_5m=300_000), _base(symbol="X", chg_pct=-0.5)], CFG)
        f = res.funnel()
        for k in ("scanned", "passed_universe", "passed_setup", "eligible_for_entry",
                  "rejected_by", "sorted_by", "top_candidates"):
            self.assertIn(k, f)
        self.assertEqual(f["scanned"], 2)
        self.assertEqual(f["sorted_by"], "vol_5m_desc_then_score")
        self.assertIn("chg_not_positive", f["rejected_by"])
        self.assertIn("score_components", f["top_candidates"][0])
        self.assertIn("vol_5m", f["top_candidates"][0])


class PaperGateTests(unittest.TestCase):
    def test_paper_only_blocks_live(self):
        cfg = EngineConfig(paper_only=True)
        self.assertFalse(live_orders_allowed(cfg, account_mode="live", system_paper_only=False))

    def test_live_requires_live_mode_and_paper_only_false(self):
        cfg = EngineConfig(paper_only=False)
        self.assertTrue(live_orders_allowed(cfg, "live", False))
        self.assertFalse(live_orders_allowed(cfg, "paper", False))
        self.assertFalse(live_orders_allowed(cfg, "live", True))


def _chain():
    return [
        {"expiry": "2026-06-12", "strike": 20.0, "type": "call", "bid": 0.80, "ask": 0.88,
         "dte": 1, "open_interest": 1500, "volume": 300},
        {"expiry": "2026-06-12", "strike": 21.0, "type": "call", "bid": 0.45, "ask": 0.52,
         "dte": 1, "open_interest": 1200, "volume": 200},
        {"expiry": "2026-06-19", "strike": 20.0, "type": "call", "bid": 1.20, "ask": 1.30,
         "dte": 7, "open_interest": 3000, "volume": 200},
        {"expiry": "2026-06-12", "strike": 20.0, "type": "put", "bid": 0.50, "ask": 0.58,
         "dte": 1, "open_interest": 1000, "volume": 100},
        {"expiry": "2026-07-17", "strike": 20.0, "type": "call", "bid": 2.0, "ask": 2.2,
         "dte": 35, "open_interest": 5000, "volume": 50},   # out of 1-7 DTE window
    ]


class OptionOverlayTests(unittest.TestCase):
    def test_prev_day_high_explicit_and_derived(self):
        self.assertEqual(previous_day_high(_base(prev_day_high=22.0)), 22.0)
        import pandas as pd
        dh = pd.DataFrame({"high": [21.0, 22.5], "low": [20, 21], "close": [20.5, 22.0]})
        self.assertEqual(previous_day_high(_base(prev_day_high=None, daily_history=dh)), 22.5)

    def test_occ_symbol_format(self):
        self.assertEqual(_occ_symbol("AAPL", "2026-01-16", "C", 150.0), "AAPL260116C00150000")

    def test_select_call_picks_1_7_dte_atm_call(self):
        c, why = select_call(20.0, _chain(), CFG)
        self.assertIsNone(why)
        self.assertEqual(str(c["type"]).lower(), "call")
        self.assertTrue(CFG.option_min_dte <= c["dte"] <= CFG.option_max_dte)
        self.assertEqual(c["strike"], 20.0)   # nearest ATM, shortest DTE

    def test_select_call_rejects_when_only_out_of_window_or_puts(self):
        chain = [c for c in _chain() if c["type"] == "put" or c["dte"] > 7]
        c, why = select_call(20.0, chain, CFG)
        self.assertIsNone(c)
        self.assertIn(why, ("no_call_in_dte_window", "no_liquid_call", "no_call_chain"))

    def test_select_call_rejects_illiquid(self):
        wide = [{"expiry": "2026-06-12", "strike": 20.0, "type": "call", "bid": 0.20, "ask": 1.20,
                 "dte": 1, "open_interest": 5, "volume": 1}]   # huge spread + tiny OI
        c, why = select_call(20.0, wide, CFG)
        self.assertIsNone(c)
        self.assertEqual(why, "no_liquid_call")

    def test_target_premium_intrinsic_plus_extrinsic_and_monotonic(self):
        # strike 20, price 20 (intrinsic 0, extrinsic = mid), pdh 22 -> 2 + 1 = 3.0
        self.assertEqual(estimate_option_target_premium(20.0, 22.0, 20.0, 1.0), 3.0)
        hi = estimate_option_target_premium(20.0, 24.0, 20.0, 1.0)
        lo = estimate_option_target_premium(20.0, 22.0, 20.0, 1.0)
        self.assertGreater(hi, lo)

    def test_size_option_respects_risk_and_notional(self):
        # 0.5% of 100k = $500 risk; premium $1.00 -> $100/contract -> 5 contracts.
        self.assertEqual(size_option(100_000.0, 1.00, CFG), 5)
        self.assertEqual(size_option(0.0, 1.0, CFG), 0)
        self.assertEqual(size_option(100_000.0, 0.0, CFG), 0)

    def test_propose_option_entries_targets_pdh_with_trailing_stop(self):
        row = _base(symbol="OPT", price=20.0, prev_day_high=22.0, option_chain=_chain())
        res = scan([row], CFG)
        orders = propose_option_entries(res, equity=100_000.0, open_symbols=set(), cfg=CFG)
        self.assertEqual(len(orders), 1)
        o = orders[0]
        self.assertEqual((o.underlying, o.side, o.order_type), ("OPT", "buy", "limit"))
        self.assertEqual(o.target_underlying, 22.0)              # PDH target
        self.assertAlmostEqual(o.trailing_stop_pct, 0.0075)      # 0.75% trailing stop
        self.assertTrue(1 <= o.dte <= 7)
        self.assertTrue(o.occ_symbol.startswith("OPT"))
        self.assertGreater(o.qty, 0)
        self.assertGreater(o.est_target_premium, o.est_premium)  # PDH is above -> ITM gain

    def test_option_overlay_skips_below_min_score(self):
        low = _base(symbol="LOW", price=20.0, prev_day_high=22.0, option_chain=_chain(),
                    avg_volume=500_000, rvol=1.0, perf_1w=-1.0, recovering=False, rsi_rising=False,
                    rsi_cross_40=False, close_gt_open=False, above_ema20_5m=False, vwap_reclaim=True)
        res = scan([low], CFG)
        self.assertEqual(propose_option_entries(res, 100_000.0, set(), CFG), [])

    def test_option_overlay_never_averages_into_held_underlying(self):
        res = scan([_base(symbol="HELD", price=20.0, prev_day_high=22.0, option_chain=_chain())], CFG)
        self.assertEqual(propose_option_entries(res, 100_000.0, {"HELD"}, CFG), [])

    def test_option_overlay_skips_when_pdh_not_above_price(self):
        res = scan([_base(symbol="ABOVE", price=23.0, prev_day_high=22.0, option_chain=_chain())], CFG)
        self.assertEqual(propose_option_entries(res, 100_000.0, set(), CFG), [])

    def test_propose_for_engine_routes_by_options_flag(self):
        row = _base(symbol="RT", price=20.0, prev_day_high=22.0, option_chain=_chain())
        res = scan([row], CFG)
        opt_cfg = EngineConfig(min_score=75.0, max_new_entries_per_day=2, max_open_positions=2,
                               options_enabled=True)
        eq_cfg = EngineConfig(min_score=75.0, max_new_entries_per_day=2, max_open_positions=2,
                              options_enabled=False)
        self.assertEqual(type(propose_for_engine(res, 100_000.0, set(), opt_cfg)[0]).__name__,
                         "OptionEntryOrder")
        self.assertEqual(type(propose_for_engine(res, 100_000.0, set(), eq_cfg)[0]).__name__,
                         "EntryOrder")

    def test_share_order_carries_target_and_trailing_stop(self):
        res = scan([_base(symbol="SH", prev_day_high=30.0)], CFG)
        o = propose_entries(res, 100_000.0, set(), CFG)[0]
        self.assertEqual(o.target_underlying, 30.0)
        self.assertAlmostEqual(o.trailing_stop_pct, 0.0075)


class ScheduleWindowTests(unittest.TestCase):
    def test_active_only_after_start_before_cutoff_on_weekday(self):
        # Thu 2026-06-11 is a weekday.
        self.assertFalse(is_active_window(datetime(2026, 6, 11, 13, 0, tzinfo=_ET), CFG))   # before 13:30
        self.assertTrue(is_active_window(datetime(2026, 6, 11, 13, 30, tzinfo=_ET), CFG))   # at start
        self.assertTrue(is_active_window(datetime(2026, 6, 11, 14, 45, tzinfo=_ET), CFG))   # mid-window
        self.assertFalse(is_active_window(datetime(2026, 6, 11, 16, 0, tzinfo=_ET), CFG))   # past cutoff

    def test_inactive_on_weekend(self):
        self.assertFalse(is_active_window(datetime(2026, 6, 13, 14, 0, tzinfo=_ET), CFG))   # Saturday


class TowardPdhSelectionTests(unittest.TestCase):
    def test_toward_pdh_adds_score_and_flag(self):
        # price below PDH, reachable, recovering -> toward-PDH bonus.
        toward = score_candidate(_base(price=20.0, prev_day_high=22.0, atr=0.4, recovering=True), CFG)
        none_ = score_candidate(_base(price=20.0, prev_day_high=None, recovering=True), CFG)
        self.assertTrue(toward.metrics["toward_pdh"])
        self.assertIn("toward_pdh", toward.score_components)
        self.assertGreater(toward.score, none_.score)

    def test_not_toward_when_already_above_pdh(self):
        c = score_candidate(_base(price=23.0, prev_day_high=22.0, atr=0.4, recovering=True), CFG)
        self.assertFalse(c.metrics["toward_pdh"])
        self.assertNotIn("toward_pdh", c.score_components)


class EodManagementTests(unittest.TestCase):
    def test_pdh_hit_is_target(self):
        a = manage_eod(ManagedPosition("X", 100, 44.6, 44.5, unrealized_pl=200.0,
                                       trailing_stop_pct=0.0075, pdh_hit=True), CFG)
        self.assertEqual(a.action, "target_hit")

    def test_up_and_pdh_not_hit_tightens_and_carries(self):
        a = manage_eod(ManagedPosition("X", 100, 43.9, 44.5, unrealized_pl=150.0,
                                       trailing_stop_pct=0.0075, pdh_hit=False), CFG)
        self.assertEqual(a.action, "carry_overnight_tighten")
        self.assertEqual(a.exit_plan, "sell_at_next_day_high")
        self.assertLessEqual(a.new_trailing_stop_pct, CFG.overnight_trailing_stop_pct)
        self.assertLess(a.new_trailing_stop_pct, 0.0075)        # strictly tighter than entry trail

    def test_down_and_pdh_not_hit_exits_at_close(self):
        a = manage_eod(ManagedPosition("X", 100, 42.0, 44.5, unrealized_pl=-80.0,
                                       trailing_stop_pct=0.0075, pdh_hit=False), CFG)
        self.assertEqual(a.action, "exit_eod")
        self.assertEqual(a.exit_plan, "exit_at_close")

    def test_carry_overnight_disabled_exits(self):
        cfg = EngineConfig(carry_overnight_if_up=False)
        a = manage_eod(ManagedPosition("X", 100, 43.9, 44.5, unrealized_pl=150.0,
                                       trailing_stop_pct=0.0075, pdh_hit=False), cfg)
        self.assertEqual(a.action, "exit_eod")


class NextDayExitTests(unittest.TestCase):
    def test_sell_when_pdh_target_reached(self):
        d = next_day_exit_decision(price=44.6, running_high=44.6, prev_day_high=44.5, cfg=CFG)
        self.assertTrue(d.sell)
        self.assertEqual(d.reason, "pdh_target_reached")

    def test_sell_when_retraced_from_next_day_high(self):
        # high 44.0, trail 0.40% -> sell at <= 43.824; price 43.7 triggers.
        d = next_day_exit_decision(price=43.70, running_high=44.0, prev_day_high=44.5, cfg=CFG)
        self.assertTrue(d.sell)
        self.assertIn("next_day_high_trail_hit", d.reason)

    def test_hold_when_near_high_and_below_pdh(self):
        d = next_day_exit_decision(price=43.98, running_high=44.0, prev_day_high=44.5, cfg=CFG)
        self.assertFalse(d.sell)


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
