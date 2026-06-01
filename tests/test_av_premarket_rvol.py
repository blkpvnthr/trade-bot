"""
Tests for the Alpha Vantage premarket RVOL adapter (data_sources/alpha_vantage_premarket)
and the provider-consistent wiring in PremarketLiveFeed.

Covers the required cases:
  * AV timestamps treated as US/Eastern, NOT UTC
  * fewer than 5 prior sessions  => rvol_available=False (no fabrication)
  * exactly 5 prior sessions     => valid average
  * current and average come from the SAME provider (no mixing)
  * malformed / throttled / raising client fails safe (no crash, unavailable)
  * no fabricated fallback values when unavailable
All offline (bars injected; no network).
"""

import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data_sources.alpha_vantage_premarket as avpm
from premarket_live_feed import PremarketLiveFeed


def _bar(d: str, hhmm: str, vol: float) -> dict:
    return {"timestamp": f"{d} {hhmm}:00", "open": 1.0, "high": 1.0, "low": 1.0,
            "close": 1.0, "volume": vol}


def _sessions(day_vol_pairs) -> list:
    """One premarket bar (08:00 ET) per (date, volume) pair."""
    return [_bar(d, "08:00", v) for d, v in day_vol_pairs]


class TimestampEasternTests(unittest.TestCase):
    def test_timestamps_treated_as_eastern_not_utc(self):
        # 07:00 ET is premarket (incl.); 13:00 ET is regular hours (excl.). If these were
        # misread as UTC→ET, 07:00 would become 03:00 (excluded) and 13:00 would become 09:00
        # (included) — so the exact value 100 proves Eastern handling in BOTH directions.
        bars = [_bar("2026-05-20", "07:00", 100), _bar("2026-05-20", "13:00", 999)]
        by_day = avpm._premarket_volume_by_session(bars)
        self.assertEqual(by_day, {date(2026, 5, 20): 100.0})

    def test_premarket_window_boundaries(self):
        bars = [_bar("2026-05-20", "04:00", 10),   # 04:00 incl (start)
                _bar("2026-05-20", "09:25", 20),   # 09:25 incl
                _bar("2026-05-20", "09:30", 40),   # 09:30 EXCL (open)
                _bar("2026-05-20", "03:59", 80)]   # 03:59 EXCL (before 4am)
        self.assertEqual(avpm._premarket_volume_by_session(bars), {date(2026, 5, 20): 30.0})


class SessionCountTests(unittest.TestCase):
    def test_fewer_than_5_prior_sessions_unavailable(self):
        # 4 prior + 1 current = 4 prior sessions < 5
        bars = _sessions([("2026-05-18", 100), ("2026-05-19", 100), ("2026-05-20", 100),
                          ("2026-05-21", 100), ("2026-05-22", 300)])
        r = avpm.compute_premarket_rvol("AAA", bars=bars)
        self.assertFalse(r["rvol_available"])
        self.assertIsNone(r["premarket_rvol"])
        self.assertIsNone(r["avg_premarket_volume"])           # never fabricated
        self.assertTrue(r["rvol_reason"].startswith("insufficient_premarket_sessions"))
        self.assertEqual(r["historical_sessions"], 4)

    def test_exactly_5_prior_sessions_valid_average(self):
        bars = _sessions([("2026-05-15", 100), ("2026-05-18", 100), ("2026-05-19", 100),
                          ("2026-05-20", 100), ("2026-05-21", 100),     # 5 priors @ 100
                          ("2026-05-22", 300)])                          # current @ 300
        r = avpm.compute_premarket_rvol("AAA", bars=bars, min_sessions=5)
        self.assertTrue(r["rvol_available"])
        self.assertEqual(r["sessions_used"], 5)
        self.assertEqual(r["avg_premarket_volume"], 100.0)
        self.assertEqual(r["current_premarket_volume"], 300.0)
        self.assertEqual(r["premarket_rvol"], 3.0)
        self.assertEqual(r["rvol_reason"], "ok")


class ProviderConsistencyTests(unittest.TestCase):
    def test_current_and_average_same_provider(self):
        bars = _sessions([("2026-05-15", 200), ("2026-05-18", 200), ("2026-05-19", 200),
                          ("2026-05-20", 200), ("2026-05-21", 200), ("2026-05-22", 400)])
        r = avpm.compute_premarket_rvol("AAA", bars=bars)
        self.assertEqual(r["rvol_provider"], "alpha_vantage")
        # ratio is exactly current/avg from the same bar set
        self.assertAlmostEqual(r["premarket_rvol"], r["current_premarket_volume"] / r["avg_premarket_volume"])

    def test_feed_does_not_mix_live_numerator_with_av_average(self):
        feed = PremarketLiveFeed(["AAA"])
        feed.set_premarket_rvol("AAA", current_volume=300, avg_volume=100,
                                provider="alpha_vantage", sessions=5, reason="ok", available=True)
        # simulate a DIFFERENT-provider live numerator landing in state
        feed.state["AAA"]["premarket_volume"] = 999999
        q = feed.get_symbol_state("AAA")
        self.assertEqual(q["premarket_rvol"], 3.0)              # AV 300/100, NOT 999999/100
        self.assertTrue(q["rvol_available"])
        self.assertEqual(q["rvol_provider"], "alpha_vantage")


class FailSafeTests(unittest.TestCase):
    def test_empty_bars_unavailable(self):
        r = avpm.compute_premarket_rvol("AAA", bars=[])
        self.assertFalse(r["rvol_available"])
        self.assertIsNone(r["premarket_rvol"])
        self.assertEqual(r["rvol_reason"], "no_data_or_entitlement_unavailable")

    def test_malformed_bars_fail_safe(self):
        bars = [{"garbage": 1}, {"timestamp": "not-a-date", "volume": None}, {"volume": 5}]
        r = avpm.compute_premarket_rvol("AAA", bars=bars)
        self.assertFalse(r["rvol_available"])
        self.assertIsNone(r["premarket_rvol"])

    def test_raising_client_fails_safe(self):
        class Boom:
            def get_intraday_extended(self, *a, **k):
                raise RuntimeError("throttled / 429")
        r = avpm.compute_premarket_rvol("AAA", client=Boom())
        self.assertFalse(r["rvol_available"])
        self.assertIsNone(r["premarket_rvol"])
        self.assertTrue(r["rvol_reason"].startswith("error:"))

    def test_no_fabricated_fallback_in_feed(self):
        feed = PremarketLiveFeed(["AAA"])
        feed.set_premarket_rvol("AAA", current_volume=None, avg_volume=None,
                                provider="alpha_vantage", reason="no_premarket_bars", available=False)
        q = feed.get_symbol_state("AAA")
        self.assertFalse(q["rvol_available"])
        self.assertIsNone(q["premarket_rvol"])
        self.assertIsNone(q["avg_premarket_volume_20d"])
        self.assertEqual(q["rvol_reason"], "no_premarket_bars")


class TimeMatchedAndSettleTests(unittest.TestCase):
    PRIORS = ["2026-05-15", "2026-05-18", "2026-05-19", "2026-05-20", "2026-05-21"]

    def test_time_matched_vs_full_session(self):
        # Priors have heavy LATE premarket volume (full=1000); today is only early-to-date (200).
        bars = []
        for d in self.PRIORS:
            bars += [_bar(d, "04:00", 100), _bar(d, "05:00", 100), _bar(d, "09:00", 800)]
        bars += [_bar("2026-05-22", "04:00", 100), _bar("2026-05-22", "05:00", 100)]  # cutoff 05:00
        tm = avpm.compute_premarket_rvol("AAA", bars=bars, time_matched=True)
        fs = avpm.compute_premarket_rvol("AAA", bars=bars, time_matched=False)
        self.assertEqual(tm["rvol_mode"], "time_matched")
        self.assertEqual(tm["as_of_et"], "05:00")
        self.assertAlmostEqual(tm["premarket_rvol"], 1.0)    # 200 today / 200 prior-to-05:00 → apples-to-apples
        self.assertAlmostEqual(fs["premarket_rvol"], 0.2)    # 200 today / 1000 prior-FULL → the misleading artifact

    def test_warmup_settle_holds_rvol(self):
        bars = [_bar(d, "04:00", 100) for d in self.PRIORS] + [_bar(d, "04:05", 100) for d in self.PRIORS]
        bars += [_bar("2026-05-22", "04:05", 100)]           # cutoff 04:05 → elapsed 5 min < 10
        r = avpm.compute_premarket_rvol("AAA", bars=bars, time_matched=True)
        self.assertFalse(r["rvol_available"])
        self.assertIsNone(r["premarket_rvol"])
        self.assertTrue(r["rvol_reason"].startswith("warmup_settling"))

    def test_rolling_window_excludes_early_spike(self):
        old = avpm.RVOL_ROLLING_MIN
        avpm.RVOL_ROLLING_MIN = 30
        try:
            bars = []
            for d in self.PRIORS + ["2026-05-22"]:
                bars += [_bar(d, "04:00", 999999), _bar(d, "05:00", 50), _bar(d, "05:25", 50)]
            r = avpm.compute_premarket_rvol("AAA", bars=bars, time_matched=True)
            self.assertTrue(r["rvol_available"])
            self.assertEqual(r["rvol_mode"], "time_matched_rolling30m")
            self.assertEqual(r["current_premarket_volume"], 100.0)   # (295,325] window — 04:00 spike excluded
            self.assertAlmostEqual(r["premarket_rvol"], 1.0)
        finally:
            avpm.RVOL_ROLLING_MIN = old


if __name__ == "__main__":
    unittest.main()
