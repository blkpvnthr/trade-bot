"""
Tests for options_sentiment_av — Alpha Vantage realtime options-sentiment.

Offline: all HTTP is injected via a fake `http_get`; no network. Verifies the
put-call / volume-OI normalizers, payload parsing, caching, graceful neutral
fallback, and that the advisory overlay re-ranks contracts WITHOUT touching the
hard risk gates (sizing / premium-at-risk / probability-of-profit).
"""

import sys
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import options_sentiment_av as av
from data_sources import alpha_vantage_client as avc


# ----------------------------------------------------------------------------
# Fake Alpha Vantage transport (injected into the unified AlphaVantageClient)
# ----------------------------------------------------------------------------

def _pcr_payload(symbol, full_chain, by_exp):
    return {
        "symbol": symbol,
        "put_call_ratio_full_chain": str(full_chain),
        "put_call_ratio_by_expiration": [{"date": d, "value": str(v)} for d, v in by_exp.items()],
    }


def _voi_payload(symbol, contracts):
    # contracts: list of (expiration, type, ratio)
    return {
        "symbol": symbol,
        "data": [
            {"contractID": f"{symbol}{i}", "symbol": symbol, "expiration": exp,
             "strike": "100", "type": typ, "date": exp, "volume_open_interest_ratio": ratio}
            for i, (exp, typ, ratio) in enumerate(contracts)
        ],
    }


class FakeTransport:
    """Counts calls and returns canned payloads keyed by AV function."""

    def __init__(self, pcr=None, voi=None):
        self.pcr = pcr
        self.voi = voi
        self.calls = []

    def __call__(self, url, params=None, timeout=None):
        params = params or {}
        fn = params.get("function")
        self.calls.append(fn)
        if fn == av.PCR_FUNCTION:
            if isinstance(self.pcr, Exception):
                raise self.pcr
            return self.pcr
        if fn == av.VOI_FUNCTION:
            if isinstance(self.voi, Exception):
                raise self.voi
            return self.voi
        return {}


def _av_client(transport, cache_dir, api_key="testkey", **cfg_kw):
    cfg = avc.AlphaVantageConfig(enabled=True, max_retries=0, cache_seconds=cfg_kw.pop("cache_seconds", 30))
    return avc.AlphaVantageClient(
        config=cfg, api_key=api_key, cache_dir=cache_dir,
        request_log_path=Path(cache_dir) / "req.csv",
        status_report_path=Path(cache_dir) / "status.json",
        http_get=transport,
    )


def _client(transport, cache_dir, api_key="testkey", **cfg_kw):
    return av.OptionsSentimentClient(
        config=av.SentimentConfig(enabled=True),
        av_client=_av_client(transport, cache_dir, api_key=api_key, **cfg_kw),
    )


# ----------------------------------------------------------------------------
# Normalizer tests
# ----------------------------------------------------------------------------

class PutCallNormalizerTests(unittest.TestCase):
    def test_bullish_pcr_below_06(self):
        out = av.normalize_put_call_ratio(0.45)
        self.assertGreater(out["directional_score"], 0.0)
        self.assertFalse(out["contrarian_warning"])
        self.assertEqual(out["band"], "bullish_call_demand")

    def test_bearish_pcr_at_or_above_10(self):
        out = av.normalize_put_call_ratio(1.30)
        self.assertLess(out["directional_score"], 0.0)
        self.assertFalse(out["contrarian_warning"])
        self.assertEqual(out["band"], "bearish_put_demand")

    def test_extreme_high_triggers_contrarian_warning(self):
        out = av.normalize_put_call_ratio(2.5)
        self.assertTrue(out["contrarian_warning"])
        # Extremely elevated -> contrarian bullish tilt.
        self.assertGreater(out["directional_score"], 0.0)
        self.assertEqual(out["band"], "extreme_high_contrarian_bullish")

    def test_extreme_low_triggers_contrarian_warning(self):
        out = av.normalize_put_call_ratio(0.05)
        self.assertTrue(out["contrarian_warning"])
        # Extremely low -> contrarian bearish tilt.
        self.assertLess(out["directional_score"], 0.0)
        self.assertEqual(out["band"], "extreme_low_contrarian_bearish")

    def test_neutral_band(self):
        out = av.normalize_put_call_ratio(0.8)
        self.assertAlmostEqual(out["directional_score"], 0.0, places=2)
        self.assertEqual(out["band"], "neutral_mixed")

    def test_invalid_value_is_neutral(self):
        for bad in (None, "x", -1.0, float("nan")):
            out = av.normalize_put_call_ratio(bad)
            self.assertEqual(out["directional_score"], 0.0)
            self.assertEqual(out["confidence_modifier"], 1.0)


class VolumeOiNormalizerTests(unittest.TestCase):
    def test_high_voi_boosts_participation(self):
        out = av.normalize_volume_oi_ratio(0.9)
        self.assertGreaterEqual(out["participation_score"], 0.9)
        self.assertGreater(out["confidence_modifier"], 1.0)
        self.assertEqual(out["band"], "strong_participation")

    def test_low_voi_penalizes_confirmation(self):
        out = av.normalize_volume_oi_ratio(0.02)
        self.assertLessEqual(out["participation_score"], 0.1)
        self.assertLess(out["confidence_modifier"], 1.0)
        self.assertEqual(out["band"], "weak_participation")

    def test_missing_voi_is_neutral(self):
        out = av.normalize_volume_oi_ratio(None)
        self.assertEqual(out["participation_score"], 0.5)
        self.assertEqual(out["confidence_modifier"], 1.0)


# ----------------------------------------------------------------------------
# Parser tests
# ----------------------------------------------------------------------------

class ParserTests(unittest.TestCase):
    """The raw PCR/VOI parsing lives in the unified AlphaVantageClient now."""

    def test_parse_pcr(self):
        payload = _pcr_payload("IBM", 0.23, {"2026-06-19": 1.73, "2026-07-17": 0.45})
        chain, by_exp = avc._parse_put_call_ratio(payload)
        self.assertAlmostEqual(chain, 0.23)
        self.assertAlmostEqual(by_exp["2026-06-19"], 1.73)

    def test_parse_voi_aggregates_call_put(self):
        payload = _voi_payload("NVDA", [
            ("2026-05-29", "call", 0.4), ("2026-05-29", "put", 0.2), ("2026-06-01", "call", 0.6),
        ])
        chain, call_m, put_m = avc._parse_volume_oi_ratio(payload)
        self.assertAlmostEqual(chain, (0.4 + 0.2 + 0.6) / 3)
        self.assertAlmostEqual(call_m, (0.4 + 0.6) / 2)
        self.assertAlmostEqual(put_m, 0.2)

    def test_error_payload_detected(self):
        self.assertIsNotNone(avc.AlphaVantageClient._notice({"Note": "rate limit"}))
        self.assertIsNotNone(avc.AlphaVantageClient._notice({"Information": "premium endpoint"}))
        self.assertIsNone(avc.AlphaVantageClient._notice({"symbol": "IBM"}))


# ----------------------------------------------------------------------------
# Client behavior
# ----------------------------------------------------------------------------

class ClientTests(unittest.TestCase):
    def test_full_bundle_bullish(self):
        with TemporaryDirectory() as d:
            t = FakeTransport(
                pcr=_pcr_payload("IBM", 0.45, {"2026-07-17": 0.45}),
                voi=_voi_payload("IBM", [("2026-07-17", "call", 0.6)]),
            )
            c = _client(t, d)
            bundle = c.get_symbol_sentiment("IBM")
            self.assertTrue(bundle["available"])
            self.assertAlmostEqual(bundle["put_call_ratio_chain"], 0.45)
            self.assertGreater(bundle["sentiment_directional_score"], 0.0)
            self.assertGreater(bundle["participation_score"], 0.5)

    def test_invalid_response_returns_neutral(self):
        with TemporaryDirectory() as d:
            t = FakeTransport(pcr={"Information": "premium"}, voi={"Note": "throttled"})
            c = _client(t, d)
            bundle = c.get_symbol_sentiment("IBM")
            self.assertFalse(bundle["available"])
            self.assertEqual(bundle["source"], "neutral_fallback")
            self.assertEqual(bundle["sentiment_directional_score"], 0.0)
            self.assertEqual(bundle["participation_score"], 0.5)

    def test_network_exception_returns_neutral(self):
        with TemporaryDirectory() as d:
            t = FakeTransport(pcr=RuntimeError("boom"), voi=RuntimeError("boom"))
            c = _client(t, d)
            bundle = c.get_symbol_sentiment("IBM")
            self.assertFalse(bundle["available"])

    def test_no_api_key_returns_neutral_without_calls(self):
        with TemporaryDirectory() as d:
            t = FakeTransport(pcr=_pcr_payload("IBM", 0.4, {}), voi=_voi_payload("IBM", []))
            c = _client(t, d, api_key="")
            bundle = c.get_symbol_sentiment("IBM")
            self.assertFalse(bundle["available"])
            self.assertEqual(len(t.calls), 0)

    def test_cache_hit_avoids_duplicate_api_call(self):
        with TemporaryDirectory() as d:
            t = FakeTransport(
                pcr=_pcr_payload("IBM", 0.45, {"2026-07-17": 0.45}),
                voi=_voi_payload("IBM", [("2026-07-17", "call", 0.6)]),
            )
            c = _client(t, d)
            c.get_symbol_sentiment("IBM")
            first = len(t.calls)
            c.get_symbol_sentiment("IBM")
            second = len(t.calls)
            self.assertEqual(first, 2)  # one PCR + one VOI
            self.assertEqual(second, 2)  # unified-client cache hit -> no extra calls

    def test_shared_av_client_cache_reused_across_wrappers(self):
        with TemporaryDirectory() as d:
            t = FakeTransport(
                pcr=_pcr_payload("IBM", 0.45, {"2026-07-17": 0.45}),
                voi=_voi_payload("IBM", [("2026-07-17", "call", 0.6)]),
            )
            shared = _av_client(t, d)
            av.OptionsSentimentClient(av_client=shared).get_symbol_sentiment("IBM")
            self.assertEqual(len(t.calls), 2)
            # A second wrapper over the SAME av client reuses its in-memory cache.
            av.OptionsSentimentClient(av_client=shared).get_symbol_sentiment("IBM")
            self.assertEqual(len(t.calls), 2)


# ----------------------------------------------------------------------------
# Side-aware advisory evaluation
# ----------------------------------------------------------------------------

class SetupEvaluationTests(unittest.TestCase):
    def _bundle(self, pcr, voi, exp="2026-07-17"):
        return {
            "available": True,
            "put_call_ratio_chain": pcr,
            "put_call_ratio_expiration": {exp: pcr},
            "volume_oi_ratio_chain": voi,
            "volume_oi_ratio_expiration": {exp: voi},
            "sentiment_directional_score": av.normalize_put_call_ratio(pcr)["directional_score"],
            "participation_score": av.normalize_volume_oi_ratio(voi)["participation_score"],
            "contrarian_warning": av.normalize_put_call_ratio(pcr)["contrarian_warning"],
            "options_sentiment_confidence": 0.6,
        }

    def test_bullish_call_boosts_multiplier(self):
        ev = av.evaluate_setup_sentiment(self._bundle(0.45, 0.6), side="call", technical_bullish=True)
        self.assertGreater(ev["confidence_multiplier"], 1.0)
        self.assertIsNone(ev["safety_warning"])

    def test_bearish_call_penalized_and_warned(self):
        ev = av.evaluate_setup_sentiment(self._bundle(1.4, 0.6), side="call", technical_bullish=True)
        self.assertLess(ev["confidence_multiplier"], 1.0)
        self.assertIsNotNone(ev["safety_warning"])

    def test_multiplier_is_bounded_even_for_extremes(self):
        cfg = av.SentimentConfig()
        # Any input must stay inside [floor, ceiling] — sentiment can only nudge.
        for pcr in (0.01, 0.05, 0.3, 0.6, 1.0, 1.8, 5.0):
            for voi in (0.0, 0.5, 5.0):
                ev = av.evaluate_setup_sentiment(self._bundle(pcr, voi), side="call", technical_bullish=True)
                self.assertGreaterEqual(ev["confidence_multiplier"], cfg.multiplier_floor)
                self.assertLessEqual(ev["confidence_multiplier"], cfg.multiplier_ceiling)

    def test_unavailable_sentiment_is_no_op(self):
        ev = av.evaluate_setup_sentiment({"available": False}, side="call", technical_bullish=True)
        self.assertEqual(ev["confidence_multiplier"], 1.0)
        self.assertFalse(ev["skip_low_confirmation"])

    def test_naked_call_requires_participation_and_warns_on_conflict(self):
        ev = av.evaluate_setup_sentiment(self._bundle(1.4, 0.05), side="naked_call", technical_bullish=False)
        self.assertIsNotNone(ev["safety_warning"])
        self.assertIn("naked_call", ev["safety_warning"])


# ----------------------------------------------------------------------------
# Integration: ranking changes but hard risk gates are untouched
# ----------------------------------------------------------------------------

class BoosterIntegrationTests(unittest.TestCase):
    def setUp(self):
        import options_booster as ob
        self.ob = ob
        self.row = pd.Series({"continuation_confidence": 0.7, "source": "test"})
        self.parsed = {"strike": 100.0, "expiration": date(2026, 7, 17), "type": "call"}
        self.metrics = {
            "bid": 2.0, "ask": 2.1, "mid": 2.05, "bid_size": 10, "ask_size": 10,
            "spread_pct": 0.04, "delta": 0.52, "gamma": 0.02, "theta": -0.03,
            "vega": 0.1, "iv": 0.3, "volume": 500, "open_interest": 1000,
        }

    def _bundle(self, pcr, voi, exp="2026-07-17"):
        return {
            "available": True,
            "put_call_ratio_chain": pcr,
            "put_call_ratio_expiration": {exp: pcr},
            "volume_oi_ratio_chain": voi,
            "volume_oi_ratio_expiration": {exp: voi},
            "sentiment_directional_score": av.normalize_put_call_ratio(pcr)["directional_score"],
            "participation_score": av.normalize_volume_oi_ratio(voi)["participation_score"],
            "contrarian_warning": av.normalize_put_call_ratio(pcr)["contrarian_warning"],
            "options_sentiment_confidence": 0.6,
        }

    def test_ranking_changes_with_sentiment(self):
        neutral = self.ob.score_contract(self.row, 100.0, self.parsed, 47, dict(self.metrics), sentiment=None)
        bull = self.ob.score_contract(self.row, 100.0, self.parsed, 47, dict(self.metrics), sentiment=self._bundle(0.45, 0.6))
        bear = self.ob.score_contract(self.row, 100.0, self.parsed, 47, dict(self.metrics), sentiment=self._bundle(1.4, 0.6))
        self.assertGreater(bull["option_score"], neutral["option_score"])
        self.assertGreater(neutral["option_score"], bear["option_score"])

    def test_sentiment_does_not_alter_hard_risk_fields(self):
        # Sentiment must NOT change probability-of-profit or premium-at-risk: those
        # feed sizing / risk gates, which sentiment is forbidden to override.
        neutral = self.ob.score_contract(self.row, 100.0, self.parsed, 47, dict(self.metrics), sentiment=None)
        bull = self.ob.score_contract(self.row, 100.0, self.parsed, 47, dict(self.metrics), sentiment=self._bundle(0.45, 0.9))
        for risk_field in ("probability_of_profit", "premium_at_risk_per_contract",
                           "expected_value_dollars", "expected_option_price"):
            self.assertAlmostEqual(neutral[risk_field], bull[risk_field], places=9,
                                   msg=f"sentiment changed risk field {risk_field}")

    def test_advisory_fields_present(self):
        s = self.ob.score_contract(self.row, 100.0, self.parsed, 47, dict(self.metrics), sentiment=self._bundle(0.45, 0.6))
        for f in ("put_call_ratio_chain", "put_call_ratio_expiration", "volume_oi_ratio_chain",
                  "volume_oi_ratio_expiration", "sentiment_directional_score", "participation_score",
                  "contrarian_warning", "options_sentiment_confidence",
                  "options_sentiment_decision_impact"):
            self.assertIn(f, s)


if __name__ == "__main__":
    unittest.main()
