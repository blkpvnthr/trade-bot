"""Offline tests for scripts/smoke_check_market_data_packages.py.

These validate the SCRIPT logic only, by stubbing the optional packages. No real
sockets are opened and no live packages are required. (The live smoke check itself
is run manually — it is intentionally not part of this suite.)
"""

import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import smoke_check_market_data_packages as smoke


# ── stubs ──────────────────────────────────────────────────────────────────────
def _good_streaming_module():
    m = types.ModuleType("streaming_indicators_stub")

    class _Ind:
        def __init__(self, *a, **k):
            self.value = None
            self._n = 0

        def update(self, v):
            self._n += 1
            self.value = float(v)
            return self.value

    for name in ("RSI", "SMA", "EMA", "VWAP", "ATR", "SuperTrend"):
        setattr(m, name, type(name, (_Ind,), {}))
    return m


class _FakeWS:
    def close(self):
        pass


def _ticking_ylt_module():
    """yliveticker stub whose YLiveTicker immediately delivers one dict tick (no socket)."""
    m = types.ModuleType("yliveticker_stub")

    class YLiveTicker:
        def __init__(self, on_ticker=None, ticker_names=None, reconnect=5,
                     ping_interval=15, ping_timeout=10):
            if on_ticker:
                on_ticker(_FakeWS(), {"id": "AAPL", "price": 100.0,
                                      "previousClose": 99.0, "dayVolume": "1000"})
            # returns immediately — no blocking, no socket
    m.YLiveTicker = YLiveTicker
    return m


def _silent_ylt_module():
    """yliveticker stub that delivers no ticks (simulates outside-market-hours)."""
    m = types.ModuleType("yliveticker_stub")

    class YLiveTicker:
        def __init__(self, on_ticker=None, ticker_names=None, reconnect=5,
                     ping_interval=15, ping_timeout=10):
            pass
    m.YLiveTicker = YLiveTicker
    return m


def _bad_param_ylt_module():
    """yliveticker stub with the removed on_open/on_error/on_close params."""
    m = types.ModuleType("yliveticker_stub")

    class YLiveTicker:
        def __init__(self, on_ticker=None, ticker_names=None,
                     on_open=None, on_error=None, on_close=None):
            pass
    m.YLiveTicker = YLiveTicker
    return m


# ── tests ────────────────────────────────────────────────────────────────────────
class SmokeCheckTests(unittest.TestCase):
    def test_1_missing_streaming_indicators_handled(self):
        r = smoke.run_streaming_indicators_check(None)
        self.assertFalse(r["available"])
        self.assertEqual(r["status"], "missing")

    def test_2_missing_yliveticker_handled(self):
        r = smoke.run_yliveticker_check(None, ["AAPL"], live_seconds=0.0)
        self.assertFalse(r["available"])
        self.assertEqual(r["ticks_received"], 0)

    def test_3_signature_check_catches_bad_params(self):
        r = smoke.run_yliveticker_check(_bad_param_ylt_module(), ["AAPL"], live_seconds=0.0)
        self.assertFalse(r["signature_ok"])
        self.assertTrue(any("on_open" in e or "on_error" in e or "on_close" in e or "unexpected" in e
                            for e in r["errors"]))

    def test_4_callback_parser_accepts_dict_messages(self):
        r = smoke.run_yliveticker_check(_ticking_ylt_module(), ["AAPL"], live_seconds=0.5)
        self.assertTrue(r["callback_ok"])
        self.assertGreaterEqual(r["ticks_received"], 1)
        self.assertIn("id", r["message_keys"])
        self.assertTrue(r["signature_ok"])   # the good stub has the expected params

    def test_5_output_json_written(self):
        si = smoke.run_streaming_indicators_check(_good_streaming_module())
        ylt = smoke.run_yliveticker_check(_ticking_ylt_module(), ["AAPL"], live_seconds=0.5)
        report = smoke.build_report(si, ylt)
        for k in ("generated_at", "streaming_indicators_available", "streaming_indicators_status",
                  "streaming_indicators_errors", "yliveticker_available", "yliveticker_signature",
                  "yliveticker_signature_ok", "yliveticker_callback_ok", "yliveticker_ticks_received",
                  "yliveticker_message_keys", "yliveticker_errors", "recommendations"):
            self.assertIn(k, report)
        with TemporaryDirectory() as d:
            p = smoke.write_report(report, out_dir=d)
            self.assertIsNotNone(p)
            self.assertTrue(p.exists())

    def test_6_allow_missing_returns_success(self):
        # Deterministically simulate BOTH optional packages absent (independent of the
        # local environment) → with --allow-missing, main() must exit 0.
        import unittest.mock as mock
        with mock.patch.object(smoke, "_import_optional", return_value=None):
            code = smoke.main(["--allow-missing", "--symbols", "AAPL", "--live-seconds", "0"])
        self.assertEqual(code, 0)

    def test_6b_missing_without_allow_missing_fails(self):
        # Same simulated-absent packages, but WITHOUT --allow-missing → exit 1.
        import unittest.mock as mock
        with mock.patch.object(smoke, "_import_optional", return_value=None):
            code = smoke.main(["--symbols", "AAPL", "--live-seconds", "0"])
        self.assertEqual(code, 1)

    def test_7_require_tick_fails_on_zero_ticks(self):
        r = smoke.run_yliveticker_check(_silent_ylt_module(), ["AAPL"],
                                        live_seconds=0.3, require_tick=True)
        self.assertEqual(r["ticks_received"], 0)
        self.assertTrue(any("no ticks" in e.lower() for e in r["errors"]))

    def test_good_streaming_module_passes(self):
        r = smoke.run_streaming_indicators_check(_good_streaming_module())
        self.assertTrue(r["available"])
        self.assertEqual(r["status"], "ok")


if __name__ == "__main__":
    unittest.main()
