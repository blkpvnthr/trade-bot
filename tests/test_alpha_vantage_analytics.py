#!/usr/bin/env python3
"""
Tests for the Alpha Vantage Advanced Analytics integration (§11).

Offline: a fake http_get is injected so no network / API key is required. Covers
fixed/sliding request generation, repeated RANGE + intraday RANGE, calculation
string + window-size validation, response normalization (scalar / matrix /
autocorrelation / histogram), local + neutral fallbacks, API-key redaction, and
the RL-schema / universe / risk integration of the analytics feature family.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_sources.alpha_vantage_analytics_client import (  # noqa: E402
    AlphaVantageAnalyticsClient,
    AnalyticsConfig,
    build_analytics_params,
    normalize_analytics_result,
    parse_matrix_calc,
    parse_scalar_calc,
    parse_histogram,
    compute_local_analytics,
    neutral_analytics_result,
    FIXED_WINDOW_FUNCTION,
    SLIDING_WINDOW_FUNCTION,
)
from alpha_discovery.analytics_feature_builder import (  # noqa: E402
    build_analytics_features,
    build_fixed_window_features,
    relationship_value,
    neutral_analytics_features,
)
from alpha_discovery.analytics_universe_scorer import (  # noqa: E402
    rank_universe,
    compute_risk_flags,
    correlation_clusters,
    write_correlation_reports,
    run_universe_analytics,
)


def _client(http_get, **cfg_over):
    cfg = AnalyticsConfig()
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    tmp = Path(tempfile.mkdtemp())
    return AlphaVantageAnalyticsClient(
        config=cfg, api_key="SECRET_TEST_KEY", http_get=http_get,
        cache_dir=tmp / "cache",
        request_log_path=tmp / "req.csv",
        status_report_path=tmp / "status.json",
    )


# A representative ANALYTICS_FIXED_WINDOW payload.
_FIXED_PAYLOAD = {
    "meta_data": {"symbols": "AAPL,MSFT,SPY", "interval": "DAILY", "ohlc": "close"},
    "payload": {"RETURNS_CALCULATIONS": {
        "MEAN": {"AAPL": 0.001, "MSFT": 0.002, "SPY": 0.0008},
        "STDDEV(annualized=True)": {"AAPL": 0.2, "MSFT": 0.25, "SPY": 0.15},
        "MAX_DRAWDOWN": {"AAPL": -0.1, "MSFT": -0.08, "SPY": -0.05},
        "CUMULATIVE_RETURN": {"AAPL": 0.06, "MSFT": 0.09, "SPY": 0.03},
        "AUTOCORRELATION(lag=1)": {"AAPL": 0.3, "MSFT": -0.2, "SPY": 0.1},
        "HISTOGRAM(bins=20)": {"AAPL": {"bins": [-0.02, 0.0, 0.02], "counts": [5, 20, 7]}},
        "CORRELATION(method=PEARSON)": {
            "index": ["AAPL", "MSFT", "SPY"],
            "correlation": [[1.0, 0.7, 0.9], [0.7, 1.0, 0.85], [0.9, 0.85, 1.0]],
        },
        "COVARIANCE": {
            "index": ["AAPL", "MSFT", "SPY"],
            "covariance": [[0.0004, 0.0003, 0.0002], [0.0003, 0.0006, 0.00025], [0.0002, 0.00025, 0.0002]],
        },
    }},
}


class TestRequestGeneration(unittest.TestCase):
    def test_fixed_window_request_generation(self):
        p = build_analytics_params(FIXED_WINDOW_FUNCTION, ["aapl", "msft"], ["MEAN", "STDDEV"], "60day", "DAILY")
        self.assertEqual(p["function"], FIXED_WINDOW_FUNCTION)
        self.assertEqual(p["SYMBOLS"], "AAPL,MSFT")
        self.assertEqual(p["CALCULATIONS"], "MEAN,STDDEV")
        self.assertEqual(p["RANGE"], "60day")
        self.assertEqual(p["INTERVAL"], "DAILY")
        self.assertNotIn("WINDOW_SIZE", p)

    def test_sliding_window_request_generation(self):
        p = build_analytics_params(SLIDING_WINDOW_FUNCTION, "AAPL", ["MEAN"], "3month", "5min", window_size=20)
        self.assertEqual(p["function"], SLIDING_WINDOW_FUNCTION)
        self.assertEqual(p["WINDOW_SIZE"], "20")

    def test_repeated_range_parameter(self):
        p = build_analytics_params(FIXED_WINDOW_FUNCTION, "AAPL", ["MEAN"], ["2023-07-01", "2023-08-31"], "DAILY")
        self.assertEqual(p["RANGE"], ["2023-07-01", "2023-08-31"])  # list -> repeated param

    def test_intraday_range_and_interval(self):
        for iv in ("1min", "5min", "15min", "30min", "60min"):
            p = build_analytics_params(SLIDING_WINDOW_FUNCTION, "AAPL", ["MEAN"], "1month", iv, window_size=50)
            self.assertEqual(p["INTERVAL"], iv)

    def test_calculation_string_formatting(self):
        p = build_analytics_params(
            FIXED_WINDOW_FUNCTION, "AAPL",
            ["MEAN", "VARIANCE(annualized=True)", "AUTOCORRELATION(lag=2)", "CORRELATION(method=KENDALL)"],
            "full", "DAILY")
        self.assertEqual(
            p["CALCULATIONS"],
            "MEAN,VARIANCE(annualized=True),AUTOCORRELATION(lag=2),CORRELATION(method=KENDALL)")

    def test_ohlc_param(self):
        p = build_analytics_params(FIXED_WINDOW_FUNCTION, "AAPL", ["MEAN"], "60day", "DAILY", ohlc="high")
        self.assertEqual(p["OHLC"], "high")
        p2 = build_analytics_params(FIXED_WINDOW_FUNCTION, "AAPL", ["MEAN"], "60day", "DAILY", ohlc="garbage")
        self.assertNotIn("OHLC", p2)  # invalid silently dropped

    def test_window_size_validation(self):
        with self.assertRaises(ValueError):
            build_analytics_params(SLIDING_WINDOW_FUNCTION, "AAPL", ["MEAN"], "3month", "5min", window_size=5)
        with self.assertRaises(ValueError):
            build_analytics_params(SLIDING_WINDOW_FUNCTION, "AAPL", ["MEAN"], "3month", "5min", window_size=None)

    def test_invalid_interval_rejected(self):
        with self.assertRaises(ValueError):
            build_analytics_params(FIXED_WINDOW_FUNCTION, "AAPL", ["MEAN"], "60day", "2min")

    def test_empty_symbols_or_calcs_rejected(self):
        with self.assertRaises(ValueError):
            build_analytics_params(FIXED_WINDOW_FUNCTION, [], ["MEAN"], "60day", "DAILY")
        with self.assertRaises(ValueError):
            build_analytics_params(FIXED_WINDOW_FUNCTION, "AAPL", [], "60day", "DAILY")


class TestNormalization(unittest.TestCase):
    def test_analytics_response_normalization(self):
        res = normalize_analytics_result(_FIXED_PAYLOAD, FIXED_WINDOW_FUNCTION, ["AAPL", "MSFT", "SPY"])
        self.assertTrue(res["available"])
        self.assertEqual(res["source"], "alpha_vantage_analytics")
        c = res["calculations"]
        # calc names normalized to base (args stripped)
        self.assertIn("MEAN", c)
        self.assertIn("STDDEV", c)
        self.assertIn("AUTOCORRELATION", c)
        self.assertIn("CORRELATION", c)
        self.assertIn("HISTOGRAM", c)
        self.assertEqual(c["MEAN"]["AAPL"], 0.001)

    def test_correlation_matrix_parsing(self):
        parsed = parse_matrix_calc(_FIXED_PAYLOAD["payload"]["RETURNS_CALCULATIONS"]["CORRELATION(method=PEARSON)"])
        self.assertEqual(parsed["index"], ["AAPL", "MSFT", "SPY"])
        self.assertEqual(parsed["matrix"][0][2], 0.9)

    def test_covariance_matrix_parsing(self):
        parsed = parse_matrix_calc(_FIXED_PAYLOAD["payload"]["RETURNS_CALCULATIONS"]["COVARIANCE"])
        self.assertEqual(parsed["matrix"][1][1], 0.0006)

    def test_lower_triangular_matrix_symmetrized(self):
        # Alpha Vantage returns a lower-triangular matrix; we must symmetrize it.
        node = {"index": ["MSFT", "AAPL", "SPY"],
                "correlation": [[1.0], [0.02, 1.0], [0.11, 0.41, 1.0]]}
        parsed = parse_matrix_calc(node)
        self.assertEqual(len(parsed["matrix"]), 3)
        self.assertTrue(all(len(r) == 3 for r in parsed["matrix"]))
        # mirrored: [AAPL][SPY] (upper) == [SPY][AAPL] (lower) == 0.41
        self.assertEqual(parsed["matrix"][1][2], 0.41)
        self.assertEqual(parsed["matrix"][2][1], 0.41)
        # lookups work in BOTH directions now
        self.assertEqual(relationship_value({"CORRELATION": parsed}, "CORRELATION", "AAPL", "SPY"), 0.41)
        self.assertEqual(relationship_value({"CORRELATION": parsed}, "CORRELATION", "SPY", "AAPL"), 0.41)

    def test_autocorrelation_parsing(self):
        c = normalize_analytics_result(_FIXED_PAYLOAD, FIXED_WINDOW_FUNCTION, ["AAPL", "MSFT"])["calculations"]
        self.assertEqual(c["AUTOCORRELATION"]["AAPL"], 0.3)
        self.assertEqual(c["AUTOCORRELATION"]["MSFT"], -0.2)

    def test_histogram_parsing(self):
        h = parse_histogram(_FIXED_PAYLOAD["payload"]["RETURNS_CALCULATIONS"]["HISTOGRAM(bins=20)"])
        self.assertEqual(h["AAPL"]["counts"], [5, 20, 7])

    def test_sliding_window_running_envelope(self):
        # Real sliding-window shape: RUNNING_<X> envelope + pairwise running correlation.
        # Real shape: RUNNING_<X> alongside a WINDOW_START sibling key.
        payload = {"meta_data": {"interval": "60min"}, "payload": {"RETURNS_CALCULATIONS": {
            "MEAN": {"WINDOW_START": "2026-05-06T16:00:00", "RUNNING_MEAN": {
                "SPY": {"2026-05-06T16:00:00": 0.0006, "2026-05-06T17:00:00": 0.0008},
                "AAPL": {"2026-05-06T16:00:00": 0.0010, "2026-05-06T17:00:00": 0.0015}}},
            "STDDEV": {"WINDOW_START": "2026-05-06T16:00:00", "RUNNING_STDDEV": {
                "AAPL": {"2026-05-06T16:00:00": 0.003, "2026-05-06T17:00:00": 0.004}}},
            "CORRELATION(METHOD=PEARSON)": {"WINDOW_START": "x", "RUNNING_CORRELATION": {
                "SPY;AAPL": {"2026-05-06T16:00:00": 0.68, "2026-05-06T17:00:00": 0.72}}},
        }}}
        res = normalize_analytics_result(payload, SLIDING_WINDOW_FUNCTION, ["AAPL", "SPY"])
        c = res["calculations"]
        # scalar unwrapped + keyed by real symbol (not RUNNING_MEAN)
        self.assertIn("AAPL", c["MEAN"])
        self.assertEqual(c["MEAN"]["AAPL"]["latest"], 0.0015)
        # pairwise running correlation -> symmetric latest matrix + slope matrix
        corr = c["CORRELATION"]
        self.assertEqual(relationship_value(c, "CORRELATION", "AAPL", "SPY"), 0.72)
        self.assertIn("slope_matrix", corr)

    def test_sliding_window_features_from_running(self):
        from alpha_discovery.analytics_feature_builder import build_sliding_window_features
        payload = {"payload": {"RETURNS_CALCULATIONS": {
            "STDDEV": {"RUNNING_STDDEV": {"AAPL": {"t1": 0.003, "t2": 0.004, "t3": 0.006}}},
            "CORRELATION(METHOD=PEARSON)": {"RUNNING_CORRELATION": {
                "AAPL;SPY": {"t1": 0.80, "t2": 0.70, "t3": 0.60}}},
        }}}
        res = normalize_analytics_result(payload, SLIDING_WINDOW_FUNCTION, ["AAPL", "SPY"])
        f = build_sliding_window_features("AAPL", res, spy="SPY")
        self.assertEqual(f["rolling_correlation_to_spy"], 0.60)
        self.assertLess(f["rolling_correlation_slope"], 0.0)   # correlation breaking down
        self.assertGreater(f["rolling_volatility_slope"], 0.0)  # vol expanding

    def test_sliding_series_summary(self):
        node = {"AAPL": [0.01, 0.02, 0.03, 0.05]}
        parsed = parse_scalar_calc(node)
        self.assertEqual(parsed["AAPL"]["latest"], 0.05)
        self.assertAlmostEqual(parsed["AAPL"]["slope"], (0.05 - 0.01) / 3)

    def test_garbage_payload_is_neutral_not_raise(self):
        res = normalize_analytics_result({"nonsense": 1}, FIXED_WINDOW_FUNCTION, ["AAPL"])
        self.assertFalse(res["available"])
        self.assertEqual(res["calculations"], {})


class TestFetchAndFallback(unittest.TestCase):
    def test_full_fetch_normalizes(self):
        captured = {}

        def fake(url, params, timeout):
            captured.update(params)
            return _FIXED_PAYLOAD

        c = _client(fake)
        res = c.get_fixed_window(["AAPL", "MSFT", "SPY"], ["MEAN", "CORRELATION"], range_="60day", interval="DAILY")
        self.assertTrue(res["available"])
        self.assertEqual(captured.get("function"), FIXED_WINDOW_FUNCTION)
        self.assertEqual(captured.get("SYMBOLS"), "AAPL,MSFT,SPY")

    def test_local_fallback_calculation(self):
        # disabled client -> fetch None -> local computation from price series
        c = _client(lambda *a, **k: None, enabled=False)
        res = c.get_fixed_window(
            ["AAPL", "SPY"], ["MEAN", "STDDEV", "MAX_DRAWDOWN", "CUMULATIVE_RETURN", "CORRELATION"],
            local_series={"AAPL": [100, 101, 102, 101, 103, 105], "SPY": [400, 401, 402, 403, 402, 404]})
        self.assertEqual(res["source"], "local_fallback")
        self.assertTrue(res["available"])
        self.assertIn("MEAN", res["calculations"])
        self.assertIn("CORRELATION", res["calculations"])

    def test_neutral_fallback_when_no_series(self):
        c = _client(lambda *a, **k: None, enabled=False)
        res = c.get_fixed_window(["AAPL"], ["MEAN"])
        self.assertEqual(res["source"], "neutral")
        self.assertFalse(res["available"])
        self.assertEqual(res["calculations"], {})

    def test_notice_payload_triggers_fallback(self):
        def fake(url, params, timeout):
            return {"Note": "Thank you for using Alpha Vantage! ... 25 requests per day"}

        c = _client(fake)
        res = c.get_fixed_window(["AAPL"], ["MEAN"], local_series={"AAPL": [10, 11, 12]})
        self.assertEqual(res["source"], "local_fallback")
        self.assertGreaterEqual(c.fallback_count, 1)

    def test_exception_retries_then_fallback(self):
        calls = {"n": 0}

        def fake(url, params, timeout):
            calls["n"] += 1
            raise RuntimeError("boom")

        c = _client(fake, max_retries=2, retry_backoff_seconds=0.0)
        res = c.get_fixed_window(["AAPL"], ["MEAN"])
        self.assertEqual(calls["n"], 3)  # initial + 2 retries
        self.assertEqual(res["source"], "neutral")

    def test_sliding_drops_unsupported_calcs(self):
        # MAX_DRAWDOWN/MIN/MAX break the whole sliding request -> must be dropped.
        captured = {}

        def fake(url, params, timeout):
            captured.update(params)
            return {"payload": {"RETURNS_CALCULATIONS": {"MEAN": {"RUNNING_MEAN": {"AAPL": {"t": 0.01}}}}}}

        c = _client(fake)
        c.get_sliding_window(["AAPL"], ["MEAN", "MAX_DRAWDOWN", "MIN", "MAX", "STDDEV"], window_size=20)
        sent = captured.get("CALCULATIONS", "")
        self.assertIn("MEAN", sent)
        self.assertIn("STDDEV", sent)
        self.assertNotIn("MAX_DRAWDOWN", sent)
        self.assertNotIn("MIN", sent)
        self.assertNotIn("MAX", sent)

    def test_bad_window_size_degrades_gracefully(self):
        c = _client(lambda *a, **k: _FIXED_PAYLOAD)
        res = c.get_sliding_window(["AAPL"], ["MEAN"], window_size=3)  # invalid -> no request
        self.assertIn(res["source"], ("neutral", "local_fallback"))

    def test_local_analytics_math(self):
        calcs = compute_local_analytics(
            ["AAPL"], ["MEAN", "MAX_DRAWDOWN", "CUMULATIVE_RETURN"],
            {"AAPL": [100, 110, 99]})
        self.assertAlmostEqual(calcs["CUMULATIVE_RETURN"]["AAPL"], 99 / 100 - 1.0)
        self.assertLess(calcs["MAX_DRAWDOWN"]["AAPL"], 0.0)


class TestApiKeySecurity(unittest.TestCase):
    def test_key_from_env_only(self):
        os.environ.pop("ALPHA_VANTAGE_API_KEY", None)
        os.environ.pop("ALPHA_VANTAGE_API", None)
        os.environ["ALPHA_VANTAGE_API_KEY"] = "ENVKEY999"
        try:
            self.assertEqual(AlphaVantageAnalyticsClient().api_key, "ENVKEY999")
        finally:
            os.environ.pop("ALPHA_VANTAGE_API_KEY", None)

    def test_apikey_absent_from_request_log(self):
        def fake(url, params, timeout):
            assert params.get("apikey") == "SECRET_TEST_KEY"  # must be in outgoing request
            return _FIXED_PAYLOAD

        c = _client(fake)
        c.get_fixed_window(["AAPL"], ["MEAN"], range_="60day", interval="DAILY")
        log_text = c.request_log_path.read_text(encoding="utf-8")
        self.assertNotIn("SECRET_TEST_KEY", log_text)
        self.assertIn(FIXED_WINDOW_FUNCTION, log_text)

    def test_status_report_redacted(self):
        c = _client(lambda *a, **k: _FIXED_PAYLOAD)
        c.get_fixed_window(["AAPL"], ["MEAN"])
        status = c.write_status_report()
        self.assertTrue(status["api_key_present"])
        self.assertNotIn("SECRET_TEST_KEY", c.status_report_path.read_text(encoding="utf-8"))

    def test_request_log_schema(self):
        c = _client(lambda *a, **k: _FIXED_PAYLOAD)
        c.get_fixed_window(["AAPL"], ["MEAN"], range_="60day", interval="DAILY")
        header = c.request_log_path.read_text(encoding="utf-8").splitlines()[0]
        for col in ("endpoint", "symbols", "range", "interval", "window_size",
                    "calculations", "ohlc", "status", "latency", "rows_returned",
                    "fallback_used", "error_message"):
            self.assertIn(col, header)


class TestFeatureBuilder(unittest.TestCase):
    def test_relationship_value(self):
        res = normalize_analytics_result(_FIXED_PAYLOAD, FIXED_WINDOW_FUNCTION, ["AAPL", "MSFT", "SPY"])
        v = relationship_value(res["calculations"], "CORRELATION", "AAPL", "SPY")
        self.assertEqual(v, 0.9)

    def test_build_fixed_window_features(self):
        res = normalize_analytics_result(_FIXED_PAYLOAD, FIXED_WINDOW_FUNCTION, ["AAPL", "MSFT", "SPY"])
        f = build_fixed_window_features("AAPL", res, spy="SPY", qqq="MSFT")  # MSFT as stand-in benchmark
        self.assertEqual(f["fixed_correlation_to_spy"], 0.9)
        self.assertEqual(f["fixed_cumulative_return"], 0.06)
        self.assertEqual(f["fixed_max_drawdown"], -0.1)

    def test_full_feature_block_schema_and_bounds(self):
        res = normalize_analytics_result(_FIXED_PAYLOAD, FIXED_WINDOW_FUNCTION, ["AAPL", "MSFT", "SPY"])
        feats = build_analytics_features("AAPL", res, None, event={"price_momentum": 0.5, "rsi": 70})
        self.assertEqual(feats["analytics_available"], 1)
        for k in neutral_analytics_features():
            self.assertIn(k, feats)
        for k in ("hidden_market_independence_score", "hidden_risk_adjusted_momentum_score"):
            self.assertGreaterEqual(feats[k], -1.0)
            self.assertLessEqual(feats[k], 1.0)
        self.assertIn("ix_autocorr_x_price_momentum", feats)

    def test_neutral_features_when_empty(self):
        feats = build_analytics_features("AAPL", None, None)
        self.assertEqual(feats["analytics_available"], 0)
        self.assertEqual(feats["hidden_market_independence_score"], 0.0)


class TestUniverseAndRisk(unittest.TestCase):
    def _fixed_result(self):
        return normalize_analytics_result(_FIXED_PAYLOAD, FIXED_WINDOW_FUNCTION, ["AAPL", "MSFT", "SPY"])

    def test_universe_scoring_includes_analytics(self):
        # qqq=SPY so both candidates (AAPL, MSFT) remain in the ranked set.
        rows = rank_universe(["AAPL", "MSFT"], self._fixed_result(), None, spy="SPY", qqq="SPY")
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertIn("universe_score", r)
            self.assertIn("cumulative_return_rank", r)
            self.assertIn("risk_adjusted_momentum_rank", r)
            self.assertIn("max_drawdown_risk_flag", r)
        # MSFT has higher cumulative return + better drawdown -> should outrank AAPL
        self.assertEqual(rows[0]["symbol"], "MSFT")

    def test_risk_flags_are_advisory_ints(self):
        f = build_fixed_window_features("AAPL", self._fixed_result(), spy="SPY", qqq="MSFT")
        flags = compute_risk_flags(f, {})
        self.assertTrue(all(v in (0, 1) for v in flags.values()))
        # correlation_to_spy 0.9 > crowding 0.85 -> flagged
        self.assertEqual(flags["correlation_crowding_flag"], 1)

    def test_correlation_clusters(self):
        corr = self._fixed_result()["calculations"]["CORRELATION"]
        clusters = correlation_clusters(corr, threshold=0.8)
        # AAPL-SPY (0.9) and MSFT-SPY (0.85) link all three
        self.assertEqual(len(clusters[0]), 3)

    def test_correlation_reports_written(self):
        tmp = Path(tempfile.mkdtemp())
        art = write_correlation_reports(self._fixed_result(), 0.8, tmp)
        for name in ("latest_correlation_matrix.csv", "latest_covariance_matrix.csv",
                     "latest_correlation_clusters.csv"):
            self.assertTrue((tmp / name).exists())
        self.assertTrue(art["clusters"])

    def test_run_universe_analytics_offline(self):
        # injected client that returns the canned payload for both endpoints
        c = _client(lambda url, params, timeout: _FIXED_PAYLOAD)
        tmp = Path(tempfile.mkdtemp())
        out = run_universe_analytics(
            ["AAPL", "MSFT"], analytics_client=c, spy="SPY", qqq="SPY",
            write_reports=True, report_dir=tmp, risk_report_dir=tmp)
        self.assertTrue(out["universe"])
        self.assertTrue((tmp / "latest_universe_analytics_scores.csv").exists())
        self.assertTrue((tmp / "latest_fixed_window_analytics.csv").exists())


class TestRLSchema(unittest.TestCase):
    def test_rl_schema_includes_analytics_features(self):
        import rl_common as rc
        self.assertTrue(hasattr(rc, "ANALYTICS_FEATURES"))
        self.assertEqual(len(rc.ANALYTICS_FEATURES), 10)
        self.assertEqual(len(rc._analytics_vals({})), len(rc.ANALYTICS_FEATURES))

    def test_default_off_does_not_change_width(self):
        import rl_common as rc
        # default gate OFF -> analytics not in the entry schema
        if not rc.ANALYTICS_ENABLED:
            self.assertNotIn("hidden_market_independence_score", rc.ALPHA_ENTRY_FEATURES)
            self.assertEqual(len(rc.ALPHA_ENTRY_FEATURES), len(rc.alpha_entry_features({"symbol": "X"})))

    def test_exporter_lists_analytics(self):
        from alpha_discovery.rl_feature_exporter import write_schema_report
        tmp = Path(tempfile.mkdtemp()) / "schema.json"
        report = write_schema_report(tmp)
        self.assertIn("analytics_feature_count", report)
        self.assertEqual(report["analytics_feature_count"], 10)


if __name__ == "__main__":
    unittest.main()
