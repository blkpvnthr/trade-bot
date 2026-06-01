"""
Tests for the options-sentiment alpha-discovery research layer.

Offline / deterministic. Covers: the options_sentiment feature family (factors +
interactions + IV term structure), no-regression of the existing options feature
keys, latent options-flow discovery, the RL gating (default-OFF preserves width;
+5 when enabled), validation→promotion→attribution (real signal approved, noise
rejected), the PSC feature group, exporter wiring, and the family report.

Independence invariant: this layer must not import or mutate the live booster
(options_booster.py / options_sentiment_av.py).
"""

import importlib
import os
import sys
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpha_discovery import options_feature_builder as ofb
from alpha_discovery import latent_feature_discovery as lfd
from alpha_discovery import options_alpha_discovery as oad


def _chain():
    return [
        {"type": "call", "implied_volatility": 0.28, "delta": 0.55, "gamma": 0.02,
         "open_interest": 1500, "bid": 1.0, "ask": 1.1, "volume": 300,
         "expiration": "2026-06-19", "strike": 150.0},
        {"type": "put", "implied_volatility": 0.34, "delta": -0.45, "gamma": 0.02,
         "open_interest": 1200, "bid": 1.0, "ask": 1.1, "volume": 200,
         "expiration": "2026-06-19", "strike": 150.0},
        {"type": "call", "implied_volatility": 0.24, "delta": 0.40, "gamma": 0.01,
         "open_interest": 800, "bid": 0.8, "ask": 0.9, "volume": 50,
         "expiration": "2026-09-18", "strike": 150.0},
    ]


_PCR = {"available": True, "put_call_ratio_chain": 0.45, "put_call_ratio_expiration": {"2026-06-19": 0.45}}
_VOI = {"available": True, "volume_oi_ratio_chain": 0.8, "volume_oi_ratio_call": 1.0, "volume_oi_ratio_put": 0.4}


class OptionsSentimentFamilyTests(unittest.TestCase):
    def test_factors_and_interactions_present(self):
        ctx = {"price_momentum": 0.5, "news_sentiment": 0.3, "relative_volume": 2.0,
               "breakout_strength": 0.6, "trend_strength": 0.7, "volatility_expansion": 0.4}
        f = ofb.build_options_sentiment_features(_chain(), _PCR, _VOI, spot=150.0,
                                                 as_of=date(2026, 5, 31), context=ctx)
        for k in ofb.OPTIONS_SENTIMENT_FACTORS + ofb.OPTIONS_SENTIMENT_INTERACTIONS:
            self.assertIn(k, f)
        self.assertEqual(f["options_sentiment_available"], 1)
        # Low PCR (0.45) => call pressure positive, put pressure zero.
        self.assertGreater(f["call_pressure_factor"], 0.0)
        self.assertEqual(f["put_pressure_factor"], 0.0)
        # Put IV (0.34) > call IV (0.28) => positive skew factor.
        self.assertGreater(f["iv_skew_factor"], 0.0)
        # Near IV (0.28/0.34) > far IV (0.24) => positive term structure.
        self.assertGreater(f["iv_term_structure_factor"], 0.0)
        # Interactions are products of the factor and the context value.
        self.assertAlmostEqual(f["options_sentiment_x_price_momentum"],
                               f["option_sentiment_factor"] * 0.5, places=4)

    def test_neutral_when_missing(self):
        n = ofb.build_options_sentiment_features()
        self.assertEqual(n["options_sentiment_available"], 0)
        for k in ofb.OPTIONS_SENTIMENT_FACTORS + ofb.OPTIONS_SENTIMENT_INTERACTIONS:
            self.assertEqual(n[k], 0.0)

    def test_interactions_zero_without_context(self):
        f = ofb.build_options_sentiment_features(_chain(), _PCR, _VOI, spot=150.0)
        for k in ofb.OPTIONS_SENTIMENT_INTERACTIONS:
            self.assertEqual(f[k], 0.0)

    def test_aggregate_no_regression(self):
        agg = ofb.build_options_features(_chain(), _PCR, _VOI, spot=150.0)
        # Legacy keys preserved.
        for legacy in ("realtime_put_call_sentiment_score", "realtime_option_gamma_exposure",
                       "realtime_option_iv_skew", "realtime_options_available"):
            self.assertIn(legacy, agg)
        # New family folded in.
        self.assertIn("put_call_pressure_factor", agg)
        self.assertEqual(agg["realtime_options_available"], 1)


class LatentDiscoveryTests(unittest.TestCase):
    def _factor_records(self, n=40):
        import random
        random.seed(3)
        rows = []
        for _ in range(n):
            rows.append({k: random.uniform(-1, 1) for k in ofb.OPTIONS_SENTIMENT_FACTORS})
        return rows

    def test_options_latent_keys(self):
        out = lfd.discover_options_latent_factors(self._factor_records())
        self.assertEqual(
            set(out[0]),
            {"latent_options_factor_1", "latent_options_factor_2", "latent_options_factor_3",
             "latent_options_sentiment_state", "latent_gamma_state"},
        )

    def test_backward_compatible_default_prefix(self):
        out = lfd.discover_latent_factors(self._factor_records()[:5],
                                          ["put_call_pressure_factor", "gamma_pressure_factor"], 2)
        self.assertIn("latent_realtime_factor_1", out[0])

    def test_empty_records_safe(self):
        self.assertEqual(lfd.discover_options_latent_factors([]), [])


class RLGatingTests(unittest.TestCase):
    def test_default_off_preserves_width(self):
        os.environ.pop("RL_OPTIONS_SENTIMENT_FEATURES", None)
        import rl_common
        importlib.reload(rl_common)
        self.assertNotIn("latent_options_factor_1", rl_common.ALPHA_ENTRY_FEATURES)
        self.assertEqual(len(rl_common.ALPHA_ENTRY_FEATURES),
                         len(rl_common.alpha_entry_features({})))

    def test_options_features_when_enabled(self):
        os.environ["RL_OPTIONS_SENTIMENT_FEATURES"] = "1"
        import rl_common
        importlib.reload(rl_common)
        try:
            self.assertEqual(rl_common.ALPHA_ENTRY_FEATURES[-5:], rl_common.OPTIONS_SENTIMENT_FEATURES)
            n = len(rl_common.ALPHA_ENTRY_FEATURES)
            v = len(rl_common.alpha_entry_features({"latent_options_factor_1": 0.3}))
            self.assertEqual(n, v)
            idx = rl_common.ALPHA_ENTRY_FEATURES.index("latent_options_factor_1")
            self.assertAlmostEqual(rl_common.alpha_entry_features({"latent_options_factor_1": 0.3})[idx], 0.3)
        finally:
            os.environ.pop("RL_OPTIONS_SENTIMENT_FEATURES", None)
            importlib.reload(rl_common)


def _labeled_records(n=120, seed=7):
    import random
    random.seed(seed)
    rows = []
    for _ in range(n):
        p = random.uniform(-1, 1)        # predictive
        noise = random.uniform(-1, 1)    # noise
        base = 0.004 * p + random.gauss(0, 0.003)
        rows.append({
            "put_call_pressure_factor": p,
            "iv_skew_factor": noise,
            "ret_5m": base * 0.5, "ret_15m": base, "ret_30m": base * 1.2,
            "ret_60m": base * 1.4, "ret_eod": base * 2.0,
            "mfe": abs(base) + 0.002, "mae": -(abs(base) * 0.4 + 0.001),
        })
    return rows


class ValidationPromotionTests(unittest.TestCase):
    def setUp(self):
        # Redirect report outputs to a temp dir so the suite never writes repo files.
        self._tmp = TemporaryDirectory()
        d = Path(self._tmp.name)
        self._orig = (oad.REPORT_DIR, oad.VALIDATION_REPORT, oad.APPROVED_REGISTRY_JSON,
                      oad.APPROVED_REGISTRY_CSV, oad.ATTRIBUTION_REPORT)
        oad.REPORT_DIR = d
        oad.VALIDATION_REPORT = d / "options_validation_report.csv"
        oad.APPROVED_REGISTRY_JSON = d / "options_approved_features.json"
        oad.APPROVED_REGISTRY_CSV = d / "options_approved_features.csv"
        oad.ATTRIBUTION_REPORT = d / "options_alpha_contribution.csv"

    def tearDown(self):
        (oad.REPORT_DIR, oad.VALIDATION_REPORT, oad.APPROVED_REGISTRY_JSON,
         oad.APPROVED_REGISTRY_CSV, oad.ATTRIBUTION_REPORT) = self._orig
        self._tmp.cleanup()

    def test_real_signal_approved_noise_rejected(self):
        out = oad.run(records=_labeled_records())
        self.assertEqual(out["status"], "ok")
        self.assertIn("put_call_pressure_factor", out["approved_features"])
        self.assertNotIn("iv_skew_factor", out["approved_features"])
        self.assertTrue(oad.VALIDATION_REPORT.exists())
        self.assertTrue(oad.APPROVED_REGISTRY_JSON.exists())

    def test_attribution_columns_exact(self):
        oad.run(records=_labeled_records())
        header = oad.ATTRIBUTION_REPORT.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(
            header,
            "Feature,Sharpe Contribution,PnL Contribution,Trade Selection Contribution,Timing Contribution",
        )

    def test_no_data_promotes_nothing(self):
        out = oad.run(records=[])
        self.assertEqual(out["status"], "no_data")
        self.assertEqual(out["approved_count"], 0)

    def test_missing_dataset_safe(self):
        recs = oad.load_records_from_dataset("/nonexistent/path.parquet")
        self.assertEqual(recs, [])


class WiringTests(unittest.TestCase):
    def test_psc_group_present(self):
        import predictive_state_compression as psc
        self.assertIn("options_sentiment", psc.FEATURE_GROUPS)
        self.assertIn("gamma_pressure_factor", psc.FEATURE_GROUPS["options_sentiment"])
        self.assertIn("latent_gamma_state", psc.FEATURE_GROUPS["options_sentiment"])

    def test_exporter_includes_family(self):
        from alpha_discovery import rl_feature_exporter as rfe
        report = rfe.write_schema_report(Path(TemporaryDirectory().name) / "schema.json")
        self.assertEqual(report["options_sentiment_feature_count"], 5)

    def test_feature_store_family_report(self):
        from alpha_discovery.feature_store import RealtimeFeatureStore
        with TemporaryDirectory() as d:
            store = RealtimeFeatureStore(store_dir=Path(d) / "s", report_dir=Path(d) / "r")
            rows = [{"symbol": "X", "snapshot_at": "t", "put_call_pressure_factor": 0.4, "noise": 1.0}]
            store.write_family_report(rows, "options_sentiment", names=["put_call_pressure_factor"])
            csv_path = Path(d) / "r" / "latest_options_sentiment_features.csv"
            self.assertTrue(csv_path.exists())
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("put_call_pressure_factor", header)
            self.assertNotIn("noise", header)  # projected to the family columns only

    def test_research_layer_independent_of_booster(self):
        # The research layer may *name* the booster in prose, but must never
        # IMPORT it — the two responsibilities stay separated.
        src = (Path(__file__).resolve().parents[1] / "alpha_discovery" / "options_alpha_discovery.py").read_text()
        for line in src.splitlines():
            s = line.strip()
            if s.startswith(("import ", "from ")):
                self.assertNotIn("options_booster", s)
                self.assertNotIn("options_sentiment_av", s)


if __name__ == "__main__":
    unittest.main()
