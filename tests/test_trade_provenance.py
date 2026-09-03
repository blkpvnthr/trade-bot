"""Tests for trade-provenance classification and its propagation through the round-trip FIFO.

Covers the real situation that motivated this: the bot CLOSING a manually-opened options
position (option_safety_dte) must keep that round-trip in the MANUAL bucket, because the
manual buy opened it.
"""
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trade_provenance as tp
from position_ownership import make_bot_coid, is_bot_coid
from order_emailer import round_trips


def _fill(sym, side, qty, price, t, coid=""):
    return {"symbol": sym, "side": side, "qty": qty, "price": price, "time": t, "coid": coid}


T0 = datetime(2026, 6, 8, 14, 0, tzinfo=timezone.utc)


class TestClassifier(unittest.TestCase):
    def test_origin_bot_vs_manual(self):
        self.assertEqual(tp.classify_origin(make_bot_coid("insider")), tp.ORIGIN_PRODUCTION)
        self.assertEqual(tp.classify_origin(make_bot_coid()), tp.ORIGIN_PRODUCTION)
        self.assertEqual(tp.classify_origin(""), tp.ORIGIN_MANUAL)
        self.assertEqual(tp.classify_origin("a1b2c3-manual-ui"), tp.ORIGIN_MANUAL)
        self.assertEqual(tp.classify_origin(None), tp.ORIGIN_MANUAL)

    def test_engine_from_coid(self):
        self.assertEqual(tp.engine_from_coid(make_bot_coid("insider")), "insider")
        self.assertEqual(tp.engine_from_coid(make_bot_coid("alphacall")), "alphacall")
        # legacy bot coid with no engine segment → unattributed
        self.assertEqual(tp.engine_from_coid(make_bot_coid()), tp.UNATTRIBUTED)
        # manual coid → unattributed
        self.assertEqual(tp.engine_from_coid("some-manual-coid"), tp.UNATTRIBUTED)
        self.assertEqual(tp.engine_from_coid(""), tp.UNATTRIBUTED)

    def test_strategy_from_engine(self):
        self.assertEqual(tp.strategy_from_engine("insider"), "insider_swing")
        self.assertEqual(tp.strategy_from_engine("buffett"), "buffett_accumulation")
        self.assertEqual(tp.strategy_from_engine("alphacall"), "alpha_call")
        self.assertEqual(tp.strategy_from_engine(tp.UNATTRIBUTED), tp.UNATTRIBUTED)
        self.assertEqual(tp.strategy_from_engine(""), tp.UNATTRIBUTED)

    def test_make_coid_is_recognized(self):
        self.assertTrue(is_bot_coid(make_bot_coid("buffett")))

    def test_provenance_for_coid(self):
        p = tp.provenance_for_coid(make_bot_coid("insider"))
        self.assertEqual(p, {"origin": tp.ORIGIN_PRODUCTION, "engine": "insider",
                             "strategy_id": "insider_swing"})


class TestRoundTripProvenance(unittest.TestCase):
    def test_bot_open_bot_close_is_automated(self):
        coid = make_bot_coid("insider")
        fills = [_fill("RYAN", "buy", 3, 33.0, T0, coid),
                 _fill("RYAN", "sell", 3, 35.0, T0 + timedelta(hours=2), make_bot_coid("insider"))]
        trips = round_trips(fills)
        self.assertEqual(len(trips), 1)
        self.assertEqual(trips[0]["origin"], tp.ORIGIN_PRODUCTION)
        self.assertEqual(trips[0]["engine"], "insider")
        self.assertAlmostEqual(trips[0]["pnl"], 6.0)

    def test_manual_open_bot_close_stays_manual(self):
        # The real case: user buys SOXS calls manually; bot closes them (option_safety_dte).
        sym = "SOXS260612C00006500"
        fills = [_fill(sym, "buy", 20, 0.50, T0, ""),                       # manual open (no coid)
                 _fill(sym, "sell", 20, 0.20, T0 + timedelta(days=2), make_bot_coid("optionsafety"))]
        trips = round_trips(fills)
        self.assertEqual(len(trips), 1)
        self.assertEqual(trips[0]["origin"], tp.ORIGIN_MANUAL)
        self.assertEqual(trips[0]["engine"], tp.UNATTRIBUTED)
        self.assertLess(trips[0]["pnl"], 0)   # a loss, attributed to manual

    def test_bot_open_manual_close_is_automated(self):
        # Opener decides attribution.
        fills = [_fill("AAPL", "buy", 1, 300.0, T0, make_bot_coid("alphacall")),
                 _fill("AAPL", "sell", 1, 310.0, T0 + timedelta(hours=5), "")]
        trips = round_trips(fills)
        self.assertEqual(trips[0]["origin"], tp.ORIGIN_PRODUCTION)
        self.assertEqual(trips[0]["engine"], "alphacall")

    def test_legacy_fills_without_coid_are_manual(self):
        fills = [_fill("MSFT", "buy", 1, 400.0, T0),
                 _fill("MSFT", "sell", 1, 405.0, T0 + timedelta(hours=1))]
        trips = round_trips(fills)
        self.assertEqual(trips[0]["origin"], tp.ORIGIN_MANUAL)


if __name__ == "__main__":
    unittest.main()
