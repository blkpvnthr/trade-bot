"""
Tests for the Portfolio Research Lab (Evidence System): expectancy / profit-factor
math, grouping, valuation/conviction/insider buckets, self-improvement rankings,
walk-forward simulation, event recording, and full report generation. Offline.
"""
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import portfolio_research_lab as prl


def _alpha_exit(symbol, setup, sector, regime, vol, pnl, r, ts):
    return {"event_type": "alpha_exit", "symbol": symbol, "setup_type": setup,
            "sector": sector, "regime": regime, "vol_regime": vol,
            "pnl": pnl, "r_multiple": r, "timestamp": ts}


def _bal(symbol, etype, tier, p2iv, conv, insider, dollars, ret, pnl, hold, ts):
    return {"event_type": etype, "symbol": symbol, "tier": tier,
            "price_to_intrinsic_value": p2iv, "conviction_score": conv,
            "insider_buy_score": insider, "order_dollars": dollars,
            "return_pct": ret, "pnl": pnl, "holding_period_days": hold, "timestamp": ts}


EVENTS = [
    _alpha_exit("AAA", "ORB", "Technology", "bullish", "low", 300, 3.0, "2026-05-01T14:00:00+00:00"),
    _alpha_exit("AAA", "ORB", "Technology", "bullish", "low", -100, -1.0, "2026-05-02T14:00:00+00:00"),
    _alpha_exit("BBB", "VWAP_RECLAIM", "Energy", "bullish", "normal", 200, 2.0, "2026-05-03T14:00:00+00:00"),
    _alpha_exit("CCC", "PIVOT_BREAK", "Financials", "neutral", "high", -150, -1.5, "2026-05-04T14:00:00+00:00"),
    {"event_type": "alpha_entry", "symbol": "DDD", "setup_type": "ORB"},
    {"event_type": "alpha_rejection", "symbol": "EEE", "setup_type": "ORB",
     "reject_reason": "rejected_by_setup_quality"},
    _bal("GOOD", "bal_buy", "starter", 0.64, 86, 73, 2000, 0.12, 240, 30, "2026-05-01T15:00:00+00:00"),
    _bal("GOOD", "bal_dca", "deep", 0.80, 80, 0, 2000, 0.08, 160, 15, "2026-05-10T15:00:00+00:00"),
    _bal("VAL2", "bal_dca", "panic", 0.72, 92, 90, 2000, 0.20, 400, 20, "2026-05-12T15:00:00+00:00"),
    {"event_type": "bal_rejection", "symbol": "WEAK", "BAL_score": 40, "journal_reason": "rejected_dca_BAL_score"},
    {"event_type": "insider_discovery", "symbol": "GOOD", "insider_buy_score": 73, "insider_count": 2},
]


class ResearchLabTests(unittest.TestCase):

    def test_trade_metrics_expectancy_and_profit_factor(self):
        exits = [e for e in EVENTS if e["event_type"] == "alpha_exit"]
        m = prl.trade_metrics(exits)
        self.assertEqual(m["trades"], 4)
        self.assertEqual(m["wins"], 2)
        self.assertEqual(m["losses"], 2)
        self.assertAlmostEqual(m["win_rate"], 0.5, places=4)
        # avg_win = (300+200)/2 = 250 ; avg_loss = (100+150)/2 = 125
        # expectancy = 0.5*250 - 0.5*125 = 62.5 ; profit_factor = 500/250 = 2.0
        self.assertAlmostEqual(m["avg_win"], 250.0, places=2)
        self.assertAlmostEqual(m["avg_loss"], 125.0, places=2)
        self.assertAlmostEqual(m["expectancy"], 62.5, places=2)
        self.assertAlmostEqual(m["profit_factor"], 2.0, places=3)

    def test_alpha_analytics_grouping_and_funnel(self):
        a = prl.alpha_analytics(EVENTS)
        self.assertIn("ORB", a["by_setup_type"])
        self.assertIn("VWAP_RECLAIM", a["by_setup_type"])
        self.assertIn("Technology", a["by_sector"])
        self.assertIn("bullish", a["by_market_regime"])
        self.assertIn("high", a["by_volatility_regime"])
        self.assertEqual(a["funnel"], {"entries": 1, "exits": 4, "rejections": 1})
        # ORB: +300/-100 → expectancy 0.5*300 - 0.5*100 = 100
        self.assertAlmostEqual(a["by_setup_type"]["ORB"]["expectancy"], 100.0, places=2)

    def test_buffett_buckets_and_tranches(self):
        b = prl.buffett_analytics(EVENTS)
        self.assertIn("starter", b["by_tranche"])
        self.assertIn("deep", b["by_tranche"])
        self.assertIn("panic", b["by_tranche"])
        # valuation buckets present
        self.assertTrue(set(b["by_valuation_bucket"]) & {"<65%", "65-75%", "75-85%"})
        # conviction buckets present
        self.assertTrue(set(b["by_conviction_bucket"]) & {"90-100", "75-89"})
        # insider buckets present
        self.assertTrue(set(b["by_insider_bucket"]) & {"none", "high(70-100)"})
        self.assertEqual(b["funnel"]["accumulations"], 3)
        self.assertEqual(b["funnel"]["rejections"], 1)
        # tranche extras
        self.assertIn("average_return", b["by_tranche"]["starter"])
        self.assertIn("contribution_to_total_return", b["by_tranche"]["starter"])

    def test_deeper_undervaluation_question_is_measurable(self):
        # The lab answers "does deeper undervaluation create better returns?" by bucket.
        b = prl.buffett_analytics(EVENTS)
        deep_val = b["by_valuation_bucket"].get("<65%", {})
        self.assertGreaterEqual(deep_val.get("trades", 0), 1)

    def test_rankings_identify_best_and_worst(self):
        a = prl.alpha_analytics(EVENTS)
        b = prl.buffett_analytics(EVENTS)
        r = prl.rankings(a, b)
        top = [x["key"] for x in r["top_alpha_setups"]["by_expectancy"]]
        worst = [x["key"] for x in r["worst_alpha_setups"]["by_expectancy"]]
        # PIVOT_BREAK (only a loser) must rank worst; a winner must rank best.
        self.assertEqual(worst[0], "PIVOT_BREAK")
        self.assertIn(top[0], {"VWAP_RECLAIM", "ORB"})
        self.assertIn("valuation_buckets", r["top_buffett_characteristics"])

    def test_walk_forward_simulation(self):
        wf = prl.walk_forward(EVENTS, start_equity=100_000.0)
        # Net pnl of closed events = 300-100+200-150 + 240+160+400 = 1050
        self.assertAlmostEqual(wf["end_equity"], 101_050.0, places=2)
        self.assertEqual(wf["trades"], 7)
        self.assertGreater(wf["total_return"], 0.0)
        for key in ("cagr", "sharpe", "max_drawdown", "profit_factor", "win_rate", "equity_curve"):
            self.assertIn(key, wf)
        self.assertEqual(wf["equity_curve"][0], 100_000.0)

    def test_waterfall_analytics(self):
        wf = {"realized_today": 1000.0, "allocation_today": {"bal": 600.0}, "bal_reserve": 200.0}
        w = prl.waterfall_analytics(EVENTS, wf, bal_current_value=7000.0)
        self.assertEqual(w["bal_invested"], 6000.0)        # 2000*3
        self.assertEqual(w["bal_current_value"], 7000.0)
        self.assertAlmostEqual(w["waterfall_return"], (7000 - 6000) / 6000, places=4)
        self.assertEqual(w["alpha_generated"], 1000.0)

    def test_event_recording_and_full_report_generation(self):
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            with patch.object(prl, "DAILY_OUTPUTS", tmpdir):
                # record one of each major event
                prl.record_alpha_signal({"symbol": "AAA", "setup_type": "ORB", "grade": "A"},
                                        {"overall_market_bias": "bullish"}, placed=True)
                prl.record_alpha_exit({"underlying": "AAA", "setup_type": "ORB", "realized_pnl": 300,
                                       "r_multiple": 3.0, "source": "alpha_call_3R_target"},
                                      {"overall_market_bias": "bullish"})
                prl.record_bal_order({"symbol": "GOOD", "tier": "starter", "price": 150,
                                      "order_dollars": 2000, "conviction_score": 86, "action": "normal"},
                                     {"overall_market_bias": "bullish"})
                prl.record_insider_discovery({"symbol": "GOOD", "insider_buy_score": 73, "insider_count": 2})

                events = prl.read_events()
                self.assertEqual(len(events), 4)
                types = {e["event_type"] for e in events}
                self.assertEqual(types, {"alpha_entry", "alpha_exit", "bal_buy", "insider_discovery"})

                rl = prl.generate_all_reports(start_equity=100_000.0)
                # all five report files exist
                for name in (prl.RESEARCH_LAB, prl.ALPHA_EXPECTANCY_REPORT,
                             prl.BUFFETT_PERFORMANCE_REPORT, prl.CAPITAL_WATERFALL_PERFORMANCE,
                             prl.WALKFORWARD_REPORT):
                    self.assertTrue((prl.research_dir() / name).exists(), f"{name} not written")
                self.assertIn("alpha", rl)
                self.assertIn("walkforward", rl)
                # research_lab.json is valid JSON with the analytics sections
                disk = json.loads((prl.research_dir() / prl.RESEARCH_LAB).read_text())
                self.assertIn("rankings", disk)
                self.assertIn("capital_waterfall", disk)

    def test_bal_rejections_are_deduped_by_symbol_and_reason(self):
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            with patch.object(prl, "DAILY_OUTPUTS", tmpdir):
                prl.record_bal_rejection(
                    {
                        "symbol": "ADBE",
                        "BAL_score": 69.5,
                        "conviction_score": 73.7,
                        "price_to_intrinsic_value": 0.6512,
                        "composite_dca_score": 56.8,
                        "journal_reason": "rejected_dca_cash_buffer",
                        "reject_reason": "order_too_small:$64",
                    },
                    {"overall_market_bias": "mixed"},
                )
                prl.record_bal_rejection(
                    {
                        "symbol": "ADBE",
                        "BAL_score": 69.5,
                        "conviction_score": 73.7,
                        "price_to_intrinsic_value": 0.6512,
                        "composite_dca_score": 56.8,
                        "journal_reason": "rejected_dca_cash_buffer",
                        "reject_reason": "order_too_small:$64",
                    },
                    {"overall_market_bias": "bullish"},
                )
                prl.record_bal_rejection(
                    {
                        "symbol": "ADBE",
                        "BAL_score": 69.5,
                        "conviction_score": 73.7,
                        "price_to_intrinsic_value": 0.6512,
                        "composite_dca_score": 56.8,
                        "journal_reason": "rejected_dca_concentration",
                        "reject_reason": "order_too_small:$40",
                    },
                    {"overall_market_bias": "bullish"},
                )

                events = prl.read_events()
                self.assertEqual(len(events), 2)
                self.assertEqual(events[0]["journal_reason"], "rejected_dca_cash_buffer")
                self.assertEqual(events[1]["journal_reason"], "rejected_dca_concentration")

    def test_safe_wrappers_never_raise(self):
        # Bad inputs must never raise (trading-loop safety) — they degrade gracefully.
        try:
            prl.record_alpha_exit_safe(None)            # None handled → returns an event
            prl.record_alpha_signal_safe(None, None, True)
            self.assertIsNone(prl.record_bal_order_safe(12345))   # wrong type → swallowed → None
            self.assertIsNotNone(prl.generate_all_reports_safe(events=[]))
        except Exception as exc:
            self.fail(f"safe wrapper raised: {exc}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
