#!/usr/bin/env python3
"""
Tests for the realtime feature builders (price / technical / options) and the
RL feature exporter + schema gating.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpha_discovery.price_feature_builder import build_price_features, neutral_price_features  # noqa: E402
from alpha_discovery.technical_feature_builder import build_technical_features, neutral_technical_features  # noqa: E402
from alpha_discovery.options_feature_builder import (  # noqa: E402
    build_options_features, build_pcr_features, build_voi_features, build_chain_features,
    score_contract, select_best_contract, neutral_options_features,
)


def _bars(n=8, start=10.0, vol=1000):
    rows = []
    px = start
    for i in range(n):
        px = px * 1.001
        rows.append({"symbol": "X", "timestamp": f"2026-05-29 15:{i:02d}:00",
                     "open": px, "high": px * 1.002, "low": px * 0.998, "close": px,
                     "volume": vol + i * 100, "source": "s", "entitlement": "realtime"})
    return rows


class TestPriceFeatures(unittest.TestCase):
    def test_build_and_fields(self):
        feats = build_price_features(_bars(), {"price": 10.5, "change_percent": 1.2})
        self.assertEqual(feats["realtime_price_available"], 1)
        self.assertEqual(feats["realtime_quote_change_pct"], 1.2)
        self.assertIn("realtime_intraday_vwap_distance", feats)
        self.assertIn("realtime_intraday_volume_acceleration", feats)

    def test_neutral_when_empty(self):
        self.assertEqual(build_price_features([], None), neutral_price_features())


class TestTechnicalFeatures(unittest.TestCase):
    def test_alignment_score(self):
        indicators = {
            "RSI": [{"timestamp": "t", "rsi_rsi": 65.0}],
            "SMA": [{"timestamp": "t", "sma_sma": 10.0}],
            "EMA": [{"timestamp": "t", "ema_ema": 10.2}],
            "MACD": [{"timestamp": "t", "macd_macd": 0.3, "macd_macd_signal": 0.1, "macd_macd_hist": 0.2}],
            "BBANDS": [{"timestamp": "t", "bbands_real_upper_band": 11.0,
                        "bbands_real_lower_band": 9.0, "bbands_real_middle_band": 10.0}],
        }
        feats = build_technical_features(indicators, last_price=10.5)
        self.assertEqual(feats["realtime_technical_available"], 1)
        self.assertGreater(feats["realtime_technical_alignment_score"], 0.0)  # bullish setup
        self.assertEqual(feats["realtime_rsi"], 65.0)
        self.assertIsNotNone(feats["realtime_bbands_pctb"])

    def test_neutral_when_empty(self):
        self.assertEqual(build_technical_features({}), neutral_technical_features())


class TestOptionsFeatures(unittest.TestCase):
    def _chain(self):
        return [
            {"contract_id": "C1", "symbol": "X", "expiration": "2026-06-19", "strike": 150.0,
             "type": "call", "bid": 1.0, "ask": 1.1, "last": 1.05, "volume": 300, "open_interest": 1500,
             "implied_volatility": 0.30, "delta": 0.55, "gamma": 0.02, "theta": -0.03, "vega": 0.1, "rho": 0.01},
            {"contract_id": "P1", "symbol": "X", "expiration": "2026-06-19", "strike": 150.0,
             "type": "put", "bid": 0.9, "ask": 1.0, "last": 0.95, "volume": 100, "open_interest": 1200,
             "implied_volatility": 0.36, "delta": -0.45, "gamma": 0.02, "theta": -0.03, "vega": 0.1, "rho": -0.01},
        ]

    def test_pcr_features(self):
        pcr = {"available": True, "put_call_ratio_chain": 0.5,
               "put_call_ratio_expiration": {"2026-06-19": 0.6}}
        feats = build_pcr_features(pcr)
        self.assertGreater(feats["put_call_ratio_signal"], 0.0)  # low PCR -> bullish
        self.assertEqual(feats["realtime_put_call_ratio_total"], 0.5)

    def test_voi_features(self):
        voi = {"available": True, "volume_oi_ratio_chain": 0.8,
               "volume_oi_ratio_call": 1.0, "volume_oi_ratio_put": 0.4}
        feats = build_voi_features(voi)
        self.assertGreater(feats["unusual_options_activity_score"], 0.0)
        self.assertGreater(feats["options_participation_score"], 0.5)

    def test_chain_iv_skew_and_liquidity(self):
        feats = build_chain_features(self._chain(), spot=150.0)
        # put IV (0.36) > call IV (0.30) -> positive skew
        self.assertAlmostEqual(feats["realtime_option_iv_skew"], 0.06, places=3)
        self.assertGreater(feats["realtime_options_liquidity_score"], 0.0)

    def test_aggregate_and_availability(self):
        feats = build_options_features(self._chain(),
                                       {"available": True, "put_call_ratio_chain": 0.5,
                                        "put_call_ratio_expiration": {}},
                                       {"available": True, "volume_oi_ratio_chain": 0.8,
                                        "volume_oi_ratio_call": 1.0, "volume_oi_ratio_put": 0.4},
                                       spot=150.0)
        self.assertEqual(feats["realtime_options_available"], 1)

    def test_neutral_when_empty(self):
        self.assertEqual(build_options_features(), neutral_options_features())

    def test_contract_selection_prefers_liquid_tight(self):
        from datetime import date
        as_of = date(2026, 5, 29)  # ~21 DTE to 2026-06-19
        best = select_best_contract(self._chain(), spot=150.0, side="call",
                                    target_delta=0.55, min_dte=15, max_dte=60, as_of=as_of)
        self.assertIsNotNone(best)
        self.assertEqual(best["type"], "call")
        self.assertIn("selection", best)
        self.assertTrue(best["selection"]["passes_gates"])

    def test_contract_selection_rejects_illiquid(self):
        from datetime import date
        bad = [{"contract_id": "C2", "type": "call", "expiration": "2026-06-19", "strike": 150.0,
                "bid": 1.0, "ask": 1.5, "volume": 1, "open_interest": 5,  # wide spread, tiny OI
                "implied_volatility": 0.3, "delta": 0.5, "gamma": 0.01, "theta": -0.02}]
        best = select_best_contract(bad, spot=150.0, side="call",
                                    min_dte=15, max_dte=60, as_of=date(2026, 5, 29))
        self.assertIsNone(best)


class TestRLFeatureExporter(unittest.TestCase):
    def test_export_realtime_subset(self):
        from alpha_discovery.rl_feature_exporter import export_rl_features, write_schema_report
        merged = {"realtime_quote_momentum": 0.1, "sentiment_mean_60m": 0.2, "irrelevant": "x"}
        out = export_rl_features(merged)
        self.assertIn("realtime_quote_momentum", out)
        self.assertIn("sentiment_mean_60m", out)
        self.assertNotIn("irrelevant", out)
        # absent declared features default to 0.0
        self.assertEqual(out.get("realtime_option_iv_skew", 0.0), 0.0)
        report = write_schema_report()
        self.assertGreaterEqual(report["av_realtime_feature_count"], 10)


class TestRLSchemaGatingDefaultOff(unittest.TestCase):
    def test_default_off_preserves_width(self):
        import importlib, os
        for k in ("RL_REALTIME_AV_FEATURES", "RL_NEWS_FEATURES", "RL_PSC_FEATURES"):
            os.environ.pop(k, None)
        import rl_common
        importlib.reload(rl_common)
        self.assertNotIn("realtime_quote_momentum", rl_common.ALPHA_ENTRY_FEATURES)
        n = len(rl_common.ALPHA_ENTRY_FEATURES)
        v = len(rl_common.alpha_entry_features({}))
        self.assertEqual(n, v)

    def test_av_features_when_enabled(self):
        import importlib, os
        os.environ["RL_REALTIME_AV_FEATURES"] = "1"
        import rl_common
        importlib.reload(rl_common)
        try:
            self.assertIn("realtime_quote_momentum", rl_common.ALPHA_ENTRY_FEATURES)
            n = len(rl_common.ALPHA_ENTRY_FEATURES)
            v = len(rl_common.alpha_entry_features({"realtime_quote_momentum": 0.1}))
            self.assertEqual(n, v)
        finally:
            os.environ.pop("RL_REALTIME_AV_FEATURES", None)
            importlib.reload(rl_common)


if __name__ == "__main__":
    unittest.main()
