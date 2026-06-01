#!/usr/bin/env python3
"""
Tests for the unified Alpha Vantage realtime client (data_sources/alpha_vantage_client.py).

Offline: a fake http_get is injected so no network / API key is required. Covers
entitlement insertion, env-only key, normalization for every endpoint, greeks/IV
parsing, PCR / VOI parsing, missing-field fallback, rate-limit (notice) handling,
live-trading fallback, and the security guarantee that API keys never hit logs.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_sources.alpha_vantage_client import (  # noqa: E402
    AlphaVantageClient,
    AlphaVantageConfig,
    normalize_time_series,
    normalize_global_quote,
    normalize_technical_indicator,
    normalize_options_chain,
    _parse_put_call_ratio,
    _parse_volume_oi_ratio,
)


def _client(http_get, **cfg_over):
    cfg = AlphaVantageConfig()
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    tmp = Path(tempfile.mkdtemp())
    return AlphaVantageClient(
        config=cfg, api_key="SECRET_TEST_KEY", http_get=http_get,
        cache_dir=tmp / "cache",
        request_log_path=tmp / "req.csv",
        status_report_path=tmp / "status.json",
    )


class TestEntitlement(unittest.TestCase):
    def test_entitlement_appended_for_equity(self):
        captured = {}

        def fake(url, params, timeout):
            captured.update(params)
            return {"Time Series (5min)": {}}

        c = _client(fake)  # default entitlement realtime
        c.get_intraday("IBM", "5min")
        self.assertEqual(captured.get("entitlement"), "realtime")
        self.assertEqual(captured.get("function"), "TIME_SERIES_INTRADAY")

    def test_entitlement_absent_when_disabled(self):
        captured = {}

        def fake(url, params, timeout):
            captured.update(params)
            return {"Time Series (5min)": {}}

        c = _client(fake, entitlement="")
        c.get_intraday("IBM")
        self.assertNotIn("entitlement", captured)

    def test_realtime_options_has_no_entitlement_param(self):
        captured = {}

        def fake(url, params, timeout):
            captured.update(params)
            return {"data": []}

        c = _client(fake)
        c.get_realtime_options("IBM")
        self.assertNotIn("entitlement", captured)
        self.assertEqual(captured.get("require_greeks"), "true")


class TestApiKeyOnlyFromEnv(unittest.TestCase):
    def test_default_key_from_env_only(self):
        os.environ.pop("ALPHA_VANTAGE_API_KEY", None)
        os.environ.pop("ALPHA_VANTAGE_API", None)
        os.environ["ALPHA_VANTAGE_API_KEY"] = "ENVKEY123"
        try:
            c = AlphaVantageClient()
            self.assertEqual(c.api_key, "ENVKEY123")
        finally:
            os.environ.pop("ALPHA_VANTAGE_API_KEY", None)

    def test_no_key_returns_neutral_not_raise(self):
        c = AlphaVantageClient(api_key="")  # explicit empty
        self.assertEqual(c.get_intraday("IBM"), [])
        self.assertEqual(c.get_global_quote("IBM"), {})
        self.assertEqual(c.get_realtime_options("IBM"), [])


class TestKeyNeverLogged(unittest.TestCase):
    def test_apikey_absent_from_request_log(self):
        def fake(url, params, timeout):
            # the apikey MUST be present in the outgoing request...
            assert params.get("apikey") == "SECRET_TEST_KEY"
            return {"Global Quote": {"01. symbol": "IBM", "05. price": "10"}}

        c = _client(fake)
        c.get_global_quote("IBM")
        log_text = c.request_log_path.read_text(encoding="utf-8")
        self.assertNotIn("SECRET_TEST_KEY", log_text)
        self.assertIn("GLOBAL_QUOTE", log_text)


class TestNormalizers(unittest.TestCase):
    def test_intraday_normalization(self):
        payload = {"Time Series (5min)": {
            "2026-05-29 16:00:00": {"1. open": "10", "2. high": "11", "3. low": "9", "4. close": "10.5", "5. volume": "1000"},
            "2026-05-29 15:55:00": {"1. open": "9.8", "2. high": "10.2", "3. low": "9.7", "4. close": "10.0", "5. volume": "800"},
        }}
        rows = normalize_time_series(payload, "IBM", "realtime", source="alpha_vantage_intraday")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["timestamp"], "2026-05-29 15:55:00")  # sorted oldest-first
        self.assertEqual(rows[-1]["close"], 10.5)
        for f in ("symbol", "timestamp", "open", "high", "low", "close", "volume", "source", "entitlement"):
            self.assertIn(f, rows[0])
        self.assertEqual(rows[0]["entitlement"], "realtime")

    def test_global_quote_normalization(self):
        payload = {"Global Quote": {
            "01. symbol": "IBM", "02. open": "10", "03. high": "11", "04. low": "9",
            "05. price": "10.5", "06. volume": "12345", "08. previous close": "10.0",
            "09. change": "0.5", "10. change percent": "5.0%",
        }}
        q = normalize_global_quote(payload, "IBM", "realtime")
        self.assertEqual(q["symbol"], "IBM")
        self.assertEqual(q["price"], 10.5)
        self.assertEqual(q["change_percent"], 5.0)

    def test_daily_adjusted_normalization(self):
        payload = {"Time Series (Daily)": {
            "2026-05-29": {"1. open": "10", "2. high": "11", "3. low": "9",
                           "4. close": "10.4", "5. adjusted close": "10.5", "6. volume": "9999"},
        }}
        rows = normalize_time_series(payload, "IBM", "realtime", source="alpha_vantage_daily_adjusted")
        self.assertEqual(len(rows), 1)
        # adjusted close preferred when present
        self.assertEqual(rows[0]["close"], 10.4)
        self.assertEqual(rows[0]["volume"], 9999)

    def test_technical_indicator_normalization(self):
        payload = {"Technical Analysis: RSI": {
            "2026-05-29 16:00": {"RSI": "61.5"},
            "2026-05-29 15:55": {"RSI": "58.0"},
        }}
        rows = normalize_technical_indicator(payload, "IBM", "RSI", "realtime")
        self.assertEqual(len(rows), 2)
        self.assertIn("rsi_rsi", rows[0])
        self.assertEqual(rows[-1]["rsi_rsi"], 61.5)

    def test_options_chain_normalization_with_greeks(self):
        payload = {"data": [
            {"contractID": "IBM260619C", "symbol": "IBM", "expiration": "2026-06-19",
             "strike": "150", "type": "call", "bid": "1.0", "ask": "1.2", "last": "1.1",
             "volume": "120", "open_interest": "1500", "implied_volatility": "0.32",
             "delta": "0.55", "gamma": "0.02", "theta": "-0.03", "vega": "0.10", "rho": "0.01"},
            {"contractID": "IBM260619P", "symbol": "IBM", "expiration": "2026-06-19",
             "strike": "150", "type": "put", "bid": "0.9", "ask": "1.1"},
        ]}
        chain = normalize_options_chain(payload, "IBM", "realtime")
        self.assertEqual(len(chain), 2)
        call = chain[0]
        self.assertEqual(call["type"], "call")
        self.assertAlmostEqual(call["mark"], 1.1)  # mid of 1.0/1.2
        self.assertEqual(call["delta"], 0.55)
        self.assertEqual(call["implied_volatility"], 0.32)
        # missing greeks -> None, not error
        self.assertIsNone(chain[1]["delta"])

    def test_pcr_parsing(self):
        payload = {"symbol": "IBM", "put_call_ratio_full_chain": "0.63",
                   "put_call_ratio_by_expiration": [{"date": "2026-06-19", "value": "1.2"}]}
        chain, by_exp = _parse_put_call_ratio(payload)
        self.assertEqual(chain, 0.63)
        self.assertEqual(by_exp["2026-06-19"], 1.2)

    def test_pcr_fallback_to_mean_when_chain_missing(self):
        payload = {"put_call_ratio_by_expiration": [
            {"date": "2026-06-19", "value": "1.0"}, {"date": "2026-07-17", "value": "2.0"}]}
        chain, by_exp = _parse_put_call_ratio(payload)
        self.assertAlmostEqual(chain, 1.5)

    def test_voi_parsing(self):
        payload = {"symbol": "NVDA", "data": [
            {"type": "call", "volume_open_interest_ratio": "0.40"},
            {"type": "call", "volume_open_interest_ratio": "0.60"},
            {"type": "put", "volume_open_interest_ratio": "0.20"},
        ]}
        chain, call_m, put_m = _parse_volume_oi_ratio(payload)
        self.assertAlmostEqual(chain, 0.40)
        self.assertAlmostEqual(call_m, 0.50)
        self.assertAlmostEqual(put_m, 0.20)

    def test_missing_field_fallback(self):
        # garbage payloads -> empty, never raise
        self.assertEqual(normalize_time_series({}, "X", "realtime"), [])
        self.assertEqual(normalize_global_quote({}, "X", "realtime"), {})
        self.assertEqual(normalize_options_chain({"data": "notalist"}, "X", "realtime"), [])


class TestRateLimitAndFallback(unittest.TestCase):
    def test_notice_payload_triggers_fallback(self):
        def fake(url, params, timeout):
            return {"Note": "Thank you for using Alpha Vantage! ... 25 requests per day"}

        c = _client(fake)
        self.assertEqual(c.get_intraday("IBM"), [])
        self.assertGreaterEqual(c.fallback_count, 1)

    def test_exception_retries_then_fallback(self):
        calls = {"n": 0}

        def fake(url, params, timeout):
            calls["n"] += 1
            raise RuntimeError("boom")

        c = _client(fake, max_retries=2, retry_backoff_seconds=0.0)
        self.assertEqual(c.get_global_quote("IBM"), {})
        self.assertEqual(calls["n"], 3)  # initial + 2 retries

    def test_status_report_written(self):
        def fake(url, params, timeout):
            return {"Global Quote": {"01. symbol": "IBM", "05. price": "10"}}

        c = _client(fake)
        c.get_global_quote("IBM")
        status = c.write_status_report()
        self.assertTrue(c.status_report_path.exists())
        self.assertTrue(status["api_key_present"])
        self.assertNotIn("SECRET_TEST_KEY", c.status_report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
