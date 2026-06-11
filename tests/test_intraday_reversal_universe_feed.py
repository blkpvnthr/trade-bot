"""Tests for intraday_reversal_universe_feed — the screener-export -> ReversalInput
normalizer. Pure normalization (no trade decisions, no orders); covers TradingView
CSV/JSON column mapping, value parsing (suffixes/percent/ratings), 5m-volume sorting,
malformed-row tolerance, and the collect_reversal_opportunities feed integration.
"""
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import intraday_reversal_universe_feed as feed
from intraday_reversal_universe_feed import (
    load_universe, _parse_number, _parse_pct, _parse_rating, _row_to_input,
)
import intraday_reversal_engine as ire

_ET = ZoneInfo("America/New_York")

_TV_CSV = (
    "Ticker,Last,Volume,Change %,Perf.W,Analyst Rating,Technical Rating,"
    "Relative Volume,5m Volume,Average Volume\n"
    'NASDAQ:AAOI,42.10,"8.2M",-1.80%,-7.40%,Strong Buy,Strong Buy,2.6,420K,3.1M\n'
    'NYSE:RVRS,18.55,"5,400,000",-0.90%,-3.10%,Strong Buy,strong_buy,1.7,610K,1.5M\n'
    "NASDAQ:EXTD,120.00,4.1M,-0.40%,-1.20%,STRONG BUY,Strong Buy,1.6,180K,900K\n"
)


def _write(tmp, name, text):
    p = Path(tmp) / name
    p.write_text(text, encoding="utf-8")
    return str(p)


class ParseHelperTests(unittest.TestCase):
    def test_number_suffixes_and_commas(self):
        self.assertEqual(_parse_number("3.4M"), 3_400_000)
        self.assertEqual(_parse_number("820K"), 820_000)
        self.assertEqual(_parse_number("12,500,000"), 12_500_000)
        self.assertEqual(_parse_number("1.2B"), 1_200_000_000)
        self.assertEqual(_parse_number("$5.20"), 5.20)
        self.assertEqual(_parse_number("8.2M"), 8_200_000)        # no fp dust
        self.assertIsNone(_parse_number(""))
        self.assertIsNone(_parse_number("N/A"))

    def test_percent_strings(self):
        self.assertEqual(_parse_pct("-3.25%"), -3.25)
        self.assertEqual(_parse_pct("+1.4"), 1.4)
        self.assertEqual(_parse_pct("0.00%"), 0.0)
        self.assertIsNone(_parse_pct("—"))

    def test_rating_normalization(self):
        for v in ("Strong Buy", "strong_buy", "STRONG BUY", "strong  buy", "StrongBuy", "Strong-Buy"):
            self.assertEqual(_parse_rating(v), "strong buy")
        self.assertEqual(_parse_rating("Buy"), "buy")
        self.assertEqual(_parse_rating(""), "")


class CsvJsonParsingTests(unittest.TestCase):
    def test_parses_tradingview_csv_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = load_universe(csv_path=_write(tmp, "u.csv", _TV_CSV))
        by = {r.symbol: r for r in rows}
        self.assertEqual(set(by), {"AAOI", "RVRS", "EXTD"})       # exchange prefix stripped
        a = by["AAOI"]
        self.assertEqual(a.price, 42.10)
        self.assertEqual(a.volume, 8_200_000)
        self.assertEqual(a.chg_pct, -1.80)
        self.assertEqual(a.perf_1w, -7.40)
        self.assertEqual(a.analyst_rating, "strong buy")
        self.assertEqual(a.os_rating, "strong buy")               # "Technical Rating" -> os_rating
        self.assertEqual(a.rvol, 2.6)
        self.assertEqual(a.vol_5m, 420_000)
        self.assertEqual(a.avg_volume, 3_100_000)

    def test_parses_json_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            j = '[{"symbol":"ABC","price":10.5,"volume":"4M","chg_pct":"-1.2%","perf_1w":"-3%",' \
                '"analyst_rating":"Strong Buy","os_rating":"Strong Buy","vol_5m":"250K"}]'
            rows = load_universe(json_path=_write(tmp, "u.json", j))
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0].symbol, rows[0].volume, rows[0].vol_5m), ("ABC", 4_000_000, 250_000))

    def test_sorts_by_vol_5m_desc(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = load_universe(csv_path=_write(tmp, "u.csv", _TV_CSV))
        self.assertEqual([r.symbol for r in rows], ["RVRS", "AAOI", "EXTD"])   # 610K,420K,180K

    def test_skips_malformed_rows_without_crashing(self):
        bad = _TV_CSV + 'NASDAQ:MISS,,3.2M,-2.10%,-5.00%,Strong Buy,Strong Buy,1.9,300K,1.2M\n'
        with tempfile.TemporaryDirectory() as tmp:
            rows = load_universe(csv_path=_write(tmp, "u.csv", bad))
        self.assertNotIn("MISS", {r.symbol for r in rows})        # missing price -> skipped
        self.assertEqual(len(rows), 3)                            # the 3 valid rows survive

    def test_missing_required_field_reported(self):
        inp, missing = _row_to_input({"Ticker": "ZZ", "Last": "", "Volume": "1M",
                                      "Change %": "-1%", "Perf.W": "-2%",
                                      "Analyst Rating": "Strong Buy", "Technical Rating": "Strong Buy"})
        self.assertIsNone(inp)
        self.assertIn("price", missing)

    def test_no_source_returns_empty(self):
        self.assertEqual(load_universe(csv_path="", json_path="", inline=""), [])


class CollectIntegrationTests(unittest.TestCase):
    # Rows RED today (clear the engine's chg<0 gate) WITH enriched reversal-trigger columns
    # (a screener-only export has no triggers, so the live system attaches bars/signals
    # first). Proves collect_reversal_opportunities consumes the feed end-to-end.
    _RED_CSV = (
        "Ticker,Last,Volume,Change %,Perf.W,Analyst Rating,Technical Rating,Relative Volume,"
        "5m Volume,Average Volume,vwap_reclaim,rsi_cross_40,close_gt_open,above_ema20_5m,"
        "recovering,prev_day_high,atr\n"
        "NASDAQ:GRN,42.10,8.2M,-1.80%,-7.40%,Strong Buy,Strong Buy,2.6,420K,3.1M,"
        "true,true,true,true,true,44.5,0.9\n"
    )
    _IN_WINDOW = datetime(2026, 6, 11, 14, 0, tzinfo=_ET)

    def test_collect_uses_feed_when_env_path_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "u.csv", self._RED_CSV)
            saved = os.environ.get("INTRADAY_REVERSAL_UNIVERSE_CSV")
            os.environ["INTRADAY_REVERSAL_UNIVERSE_CSV"] = path
            try:
                cfg = ire.EngineConfig(enabled=True)
                opps = ire.collect_reversal_opportunities(cfg=cfg, now_et=self._IN_WINDOW)
            finally:
                if saved is None:
                    os.environ.pop("INTRADAY_REVERSAL_UNIVERSE_CSV", None)
                else:
                    os.environ["INTRADAY_REVERSAL_UNIVERSE_CSV"] = saved
        self.assertTrue(any(o.symbol == "GRN" for o in opps))
        self.assertTrue(all(o.strategy_id == "intraday_reversal" for o in opps))


if __name__ == "__main__":
    unittest.main()
