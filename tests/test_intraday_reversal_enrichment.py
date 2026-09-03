"""Tests for intraday_reversal_enrichment — intraday-bar signal derivation that turns a
screener-only universe row into a scorable reversal candidate. Pure derivation (no broker,
no orders): covers VWAP reclaim, ORB reclaim, liquidity sweep/reclaim, RSI rising/cross-40,
PDH from daily history, field attachment, and that an enriched red screener row can become
eligible. The Alpaca provider (network) is not exercised; a pure provider is injected.
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import intraday_reversal_enrichment as enr
from intraday_reversal_enrichment import derive_signals, enrich_input, enrich_universe, maybe_enrich
from intraday_reversal_engine import ReversalInput, scan, EngineConfig


def _bars(closes, vol=50_000, start="2026-06-11 13:30"):
    idx = pd.date_range(start, periods=len(closes), freq="1min", tz="UTC")
    return pd.DataFrame({"open": [c - 0.05 for c in closes], "high": [c + 0.2 for c in closes],
                         "low": [c - 0.2 for c in closes], "close": closes,
                         "volume": [vol] * len(closes)}, index=idx)


_DAILY = pd.DataFrame({"high": [110.0, 108.0, 112.0], "low": [100.0, 101.0, 102.0],
                       "close": [105.0, 104.0, 106.0]})


class DeriveSignalTests(unittest.TestCase):
    def test_vwap_reclaim(self):
        closes = [100] * 10 + [98] * 9 + [102]           # flat, dip below VWAP, reclaim on last bar
        self.assertTrue(derive_signals(_bars(closes))["vwap_reclaim"])

    def test_orb_reclaim(self):
        closes = [100] * 15 + [97, 97, 99, 100, 101]     # OR(15m) low broken then reclaimed
        self.assertTrue(derive_signals(_bars(closes), orb_minutes=15)["orb_reclaim"])

    def test_sweep_reclaim(self):
        closes = [100] * 14 + [100, 96, 100.5]           # sweep a prior low on bar -2, reclaim on -1
        self.assertTrue(derive_signals(_bars(closes))["sweep_reclaim"])

    def test_rsi_rising(self):
        closes = list(range(100, 89, -1)) + list(range(90, 100))   # V-shape: down then up
        self.assertTrue(derive_signals(_bars([float(c) for c in closes]))["rsi_rising"])

    def test_rsi_cross_40(self):
        # Decline (RSI < 40), then a sharp recovery on the last bars crosses 40 from below.
        closes = ([100, 98, 96, 94, 92, 90, 88, 86, 84, 82, 80, 79, 78, 77, 76]
                  + [76, 76, 76, 84, 92])
        sig = derive_signals(_bars([float(c) for c in closes]))
        self.assertTrue(sig["rsi_cross_40"])

    def test_pdh_from_daily(self):
        sig = derive_signals(_bars([100.0] * 20), _DAILY)
        self.assertEqual(sig["prev_day_high"], 112.0)          # last completed session high

    def test_atr_and_vol5m_present(self):
        sig = derive_signals(_bars([100.0] * 20, vol=40_000), _DAILY)
        self.assertGreater(sig["atr"], 0)
        self.assertEqual(sig["vol_5m"], 200_000)               # last 5 1-min bars * 40k


class EnrichInputTests(unittest.TestCase):
    def test_enrich_input_attaches_fields(self):
        inp = ReversalInput(symbol="X", price=102.0, volume=4e6, avg_volume=2e6, chg_pct=-1.0,
                            perf_1w=-3.0, analyst_rating="strong buy", os_rating="strong buy")
        enrich_input(inp, _bars([100] * 10 + [98] * 9 + [102]), _DAILY)
        self.assertTrue(inp.vwap_reclaim)
        self.assertIsNotNone(inp.atr)
        self.assertEqual(inp.prev_day_high, 112.0)
        self.assertIsNotNone(inp.vol_5m)


class EnrichUniverseTests(unittest.TestCase):
    def _eligible_bars(self):
        # A reclaim stack: dip below VWAP then reclaim, off the lows + rising.
        return _bars([102] * 8 + [99, 98, 98, 99, 100, 101, 102, 103], vol=60_000)

    def test_enriched_screener_row_becomes_eligible(self):
        # Green-today screener row with NO triggers -> enrich -> eligible in the engine scan.
        inp = ReversalInput(symbol="RVR", price=103.0, volume=8e6, avg_volume=3e6, chg_pct=+1.5,
                            perf_1w=-6.0, analyst_rating="strong buy", os_rating="strong buy")
        self.assertIsNone(inp.vwap_reclaim)                    # no trigger before enrichment
        provider = lambda sym: (self._eligible_bars(), _DAILY)
        rows = enrich_universe([inp], provider, orb_minutes=15, bars_per_min=1.0)
        self.assertTrue(rows[0].vwap_reclaim)                  # trigger derived
        res = scan(rows, EngineConfig())
        self.assertTrue(any(c.symbol == "RVR" and c.eligible for c in res.candidates))

    def test_maybe_enrich_disabled_is_noop(self):
        os.environ.pop("INTRADAY_REVERSAL_ENRICH_BARS", None)
        rows = [ReversalInput(symbol="X", price=10, volume=4e6, avg_volume=2e6, chg_pct=-1,
                              perf_1w=-2, analyst_rating="strong buy", os_rating="strong buy")]
        self.assertIs(maybe_enrich(rows), rows)
        self.assertIsNone(rows[0].vwap_reclaim)               # untouched

    def test_maybe_enrich_uses_injected_provider_when_enabled(self):
        saved = os.environ.get("INTRADAY_REVERSAL_ENRICH_BARS")
        os.environ["INTRADAY_REVERSAL_ENRICH_BARS"] = "true"
        try:
            inp = ReversalInput(symbol="Y", price=103.0, volume=8e6, avg_volume=3e6, chg_pct=+1.2,
                                perf_1w=-4.0, analyst_rating="strong buy", os_rating="strong buy")
            out = maybe_enrich([inp], bar_provider=lambda s: (self._eligible_bars(), _DAILY))
            self.assertTrue(out[0].vwap_reclaim)
        finally:
            if saved is None:
                os.environ.pop("INTRADAY_REVERSAL_ENRICH_BARS", None)
            else:
                os.environ["INTRADAY_REVERSAL_ENRICH_BARS"] = saved


if __name__ == "__main__":
    unittest.main()
