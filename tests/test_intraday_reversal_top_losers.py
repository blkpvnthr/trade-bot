"""Tests for the top-losers universe source + reversal watchlist. Pure parse/map + the
JSON watchlist persistence. No network."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intraday_reversal_top_losers import (
    parse_top_losers, to_reversal_inputs, track_losers, tracked_symbols,
)

_PAYLOAD = {
    "metadata": "Top gainers, losers, and most actively traded US tickers",
    "top_gainers": [{"ticker": "GAIN", "price": "5.00", "change_amount": "1.0",
                     "change_percentage": "25.0%", "volume": "4000000"}],
    "top_losers": [
        {"ticker": "LOSE", "price": "12.50", "change_amount": "-2.0",
         "change_percentage": "-13.79%", "volume": "8000000"},
        {"ticker": "DROP", "price": "45.00", "change_amount": "-5.0",
         "change_percentage": "-10.0%", "volume": "3500000"},
        {"ticker": "BADV", "price": "", "change_amount": "-1.0",      # missing price -> skipped
         "change_percentage": "-9.0%", "volume": "2000000"},
    ],
    "most_actively_traded": [{"ticker": "ACTV", "price": "30.0", "change_amount": "-0.5",
                              "change_percentage": "-1.6%", "volume": "50000000"}],
}


class ParseMapTests(unittest.TestCase):
    def test_parse_top_losers_only(self):
        rows = parse_top_losers(_PAYLOAD, sections=("top_losers",))
        self.assertEqual([r["ticker"] for r in rows], ["LOSE", "DROP", "BADV"])
        lose = rows[0]
        self.assertEqual(lose["price"], 12.50)
        self.assertEqual(lose["chg_pct"], -13.79)
        self.assertEqual(lose["volume"], 8_000_000)

    def test_parse_includes_most_active_when_requested(self):
        rows = parse_top_losers(_PAYLOAD, sections=("top_losers", "most_actively_traded"))
        self.assertIn("ACTV", [r["ticker"] for r in rows])

    def test_to_reversal_inputs_maps_and_skips_malformed(self):
        inputs = to_reversal_inputs(parse_top_losers(_PAYLOAD))
        syms = {i.symbol for i in inputs}
        self.assertEqual(syms, {"LOSE", "DROP"})          # BADV (no price) skipped
        lose = next(i for i in inputs if i.symbol == "LOSE")
        self.assertEqual(lose.chg_pct, -13.79)            # RED today (loser)
        self.assertEqual(lose.analyst_rating, "")         # ratings left for downstream enrichment
        self.assertEqual(lose.perf_1w, 0.0)


class WatchlistTests(unittest.TestCase):
    def test_track_and_load_roundtrip_and_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "wl.json")
            rows = to_reversal_inputs(parse_top_losers(_PAYLOAD))
            wl = track_losers(rows, path, today="2026-06-11")
            self.assertEqual(set(tracked_symbols(path)), {"LOSE", "DROP"})
            self.assertEqual(wl["LOSE"]["times_seen"], 1)
            self.assertEqual(wl["LOSE"]["first_seen"], "2026-06-11")
            # Re-track next day: existing symbol increments, first_seen preserved.
            wl2 = track_losers(rows, path, today="2026-06-12")
            self.assertEqual(wl2["LOSE"]["times_seen"], 2)
            self.assertEqual(wl2["LOSE"]["first_seen"], "2026-06-11")
            self.assertEqual(wl2["LOSE"]["last_seen"], "2026-06-12")


if __name__ == "__main__":
    unittest.main()
