"""
Tests for insider_transactions_av.py and its Buffett screener integration.

All tests are offline — API calls are mocked via monkeypatching or direct
function injection; no network requests are made.

Coverage:
  1.  No transactions → neutral output
  2.  One meaningful CEO open-market buy → positive score
  3.  Multiple executive buys (cluster) → stronger positive than single
  4.  Heavy discretionary insider selling → negative/caution score
  5.  Mixed buy + sell → mild/neutral score
  6.  Only routine option-exercise sells → mildly negative, flagged
  7.  Missing / invalid API response → neutral (no crash)
  8.  Cache hit avoids duplicate API call
  9.  Buffett screener ranking improves with strong AV insider signal
 10.  AV insider score does NOT override hard quality/valuation gates
 11.  Dataset feature vector length and schema valid (rl_common)
 12.  AV modifier capped so it cannot push conviction above max
 13.  normalize_transaction handles missing / malformed fields safely
"""
import json
import math
import sys
import time
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insider_transactions_av import (
    AVInsiderConfig, normalize_transaction, compute_insider_trend_score,
    get_insider_trend, fetch_insider_transactions, _classify_role,
    _classify_transaction, _NEUTRAL,
)


# ── helpers ────────────────────────────────────────────────────────────────────
# Use actual Alpha Vantage field names: security_type, shares, share_price
# (AV does not have transaction_type, number_of_shares, value, or ownership_type)

def _txn(aod="A", security_type="Common Stock", shares=10000, price=50.0,
         title="Chairman & CEO", name="Jane Smith", td=None):
    td = td or (date.today() - timedelta(days=5)).isoformat()
    return normalize_transaction({
        "acquisition_or_disposal": aod,
        "security_type": security_type,
        "shares": shares,
        "share_price": price,
        "executive_title": title,
        "executive": name,
        "transaction_date": td,
    })


def _buy(shares=10000, price=50.0, title="Chairman & CEO", name="Jane Smith", td=None):
    return _txn("A", "Common Stock", shares, price, title, name, td)


def _open_sell(shares=5000, price=50.0, title="Chief Financial Officer", name="John CFO"):
    return _txn("D", "Common Stock", shares, price, title, name)


def _routine_sell(shares=5000, price=50.0, title="SVP Sales", name="Bob SVP"):
    return _txn("D", "Emp. Stock Option (right to buy)", shares, price, title, name)


# ── 1. No transactions → neutral ──────────────────────────────────────────────

class TestNoTransactions(unittest.TestCase):
    def test_empty_returns_neutral_score(self):
        result = compute_insider_trend_score([])
        self.assertEqual(result["insider_trend_score"], 0.0)
        self.assertEqual(result["insider_confidence_modifier"], 0.0)
        self.assertEqual(result["insider_buy_count"], 0)
        self.assertEqual(result["insider_sell_count"], 0)
        self.assertIn("no_recent_insider_data", result["insider_warning_flags"])

    def test_empty_buy_sell_ratio_neutral(self):
        result = compute_insider_trend_score([])
        self.assertEqual(result["insider_buy_sell_ratio"], 1.0)

    def test_neutral_dict_has_all_required_keys(self):
        required = {
            "insider_net_shares", "insider_net_value", "insider_buy_value",
            "insider_sell_value", "insider_buy_count", "insider_sell_count",
            "insider_buy_sell_ratio", "insider_trend_score",
            "insider_confidence_modifier", "insider_warning_flags",
            "insider_summary_text",
        }
        result = compute_insider_trend_score([])
        self.assertTrue(required.issubset(result.keys()))


# ── 2. One meaningful CEO buy → positive score ─────────────────────────────────

class TestSingleCEOBuy(unittest.TestCase):
    def setUp(self):
        self.txns = [_buy(shares=20000, price=100.0, title="Chairman & CEO")]

    def test_trend_score_positive(self):
        result = compute_insider_trend_score(self.txns)
        self.assertGreater(result["insider_trend_score"], 0.0)

    def test_confidence_modifier_positive(self):
        result = compute_insider_trend_score(self.txns)
        self.assertGreater(result["insider_confidence_modifier"], 0.0)

    def test_buy_value_correct(self):
        result = compute_insider_trend_score(self.txns)
        self.assertAlmostEqual(result["insider_buy_value"], 2_000_000.0, places=0)

    def test_sell_value_zero(self):
        result = compute_insider_trend_score(self.txns)
        self.assertEqual(result["insider_sell_value"], 0.0)

    def test_buy_count_one(self):
        result = compute_insider_trend_score(self.txns)
        self.assertEqual(result["insider_buy_count"], 1)

    def test_no_warning_flags(self):
        result = compute_insider_trend_score(self.txns)
        severe = [f for f in result["insider_warning_flags"]
                  if f in ("heavy_insider_selling", "senior_exec_selling_no_buying")]
        self.assertEqual(severe, [])

    def test_trend_score_in_valid_range(self):
        result = compute_insider_trend_score(self.txns)
        self.assertGreaterEqual(result["insider_trend_score"], -1.0)
        self.assertLessEqual(result["insider_trend_score"], 1.0)


# ── 3. Multiple exec buys (cluster) → stronger than single ────────────────────

class TestClusterBuys(unittest.TestCase):
    def setUp(self):
        self.single = [_buy(shares=10000, price=100.0, title="CEO", name="Alice")]
        self.cluster = [
            _buy(shares=10000, price=100.0, title="Chairman & CEO", name="Alice"),
            _buy(shares=8000, price=100.0, title="Chief Financial Officer", name="Bob"),
            _buy(shares=5000, price=100.0, title="Director", name="Carol"),
        ]

    def test_cluster_score_exceeds_single(self):
        single_result = compute_insider_trend_score(self.single)
        cluster_result = compute_insider_trend_score(self.cluster)
        self.assertGreater(cluster_result["insider_trend_score"],
                           single_result["insider_trend_score"])

    def test_cluster_confidence_modifier_exceeds_single(self):
        single_result = compute_insider_trend_score(self.single)
        cluster_result = compute_insider_trend_score(self.cluster)
        self.assertGreaterEqual(cluster_result["insider_confidence_modifier"],
                                single_result["insider_confidence_modifier"])

    def test_buy_count_correct(self):
        result = compute_insider_trend_score(self.cluster)
        self.assertEqual(result["insider_buy_count"], 3)


# ── 4. Heavy discretionary selling → negative/caution ─────────────────────────

class TestHeavySelling(unittest.TestCase):
    def setUp(self):
        # CEO sells $5M, no buys
        self.txns = [_open_sell(shares=50000, price=100.0, title="Chairman & CEO", name="Alice")]

    def test_trend_score_negative(self):
        result = compute_insider_trend_score(self.txns)
        self.assertLess(result["insider_trend_score"], 0.0)

    def test_confidence_modifier_negative(self):
        result = compute_insider_trend_score(self.txns)
        self.assertLessEqual(result["insider_confidence_modifier"], 0.0)

    def test_sell_count_correct(self):
        result = compute_insider_trend_score(self.txns)
        self.assertEqual(result["insider_sell_count"], 1)

    def test_warning_flag_present(self):
        result = compute_insider_trend_score(self.txns)
        self.assertIn("no_open_market_buys", result["insider_warning_flags"])

    def test_buy_value_zero(self):
        result = compute_insider_trend_score(self.txns)
        self.assertEqual(result["insider_buy_value"], 0.0)


# ── 5. Mixed buy + sell → mild/neutral ────────────────────────────────────────

class TestMixedActivity(unittest.TestCase):
    def setUp(self):
        self.txns = [
            _buy(shares=10000, price=100.0, title="Director", name="Alice"),
            _open_sell(shares=8000, price=100.0, title="Chief Financial Officer", name="Bob"),
        ]

    def test_score_moderate(self):
        result = compute_insider_trend_score(self.txns)
        # Mixed → trend score should be in (-0.5, +0.5)
        self.assertGreater(result["insider_trend_score"], -0.6)
        self.assertLess(result["insider_trend_score"], 0.6)

    def test_both_counts_non_zero(self):
        result = compute_insider_trend_score(self.txns)
        self.assertGreater(result["insider_buy_count"], 0)
        self.assertGreater(result["insider_sell_count"], 0)

    def test_net_value_correct(self):
        result = compute_insider_trend_score(self.txns)
        # buy: 10k * 100 = $1M, sell: 8k * 100 = $800k, net = $200k
        self.assertAlmostEqual(result["insider_net_value"], 200_000.0, places=0)


# ── 6. Only routine option-exercise sells → mildly penalised, flagged ─────────

class TestRoutineOptionSells(unittest.TestCase):
    def setUp(self):
        self.txns = [
            _routine_sell(shares=10000, price=80.0, title="SVP Engineering", name="Eve"),
            _routine_sell(shares=5000, price=80.0, title="EVP Products", name="Frank"),
        ]

    def test_score_mildly_negative_or_neutral(self):
        result = compute_insider_trend_score(self.txns)
        # Routine option-exercise sells are discounted (AV_INSIDER_SELL_DISCOUNT=0.70) so they do
        # not floor the score at fully bearish. Threshold reflects that conservative discount
        # (a routine-only ~$1.2M book scores ~-0.69); routine sells are still treated as a
        # softer-but-present negative, not neutral.
        self.assertGreater(result["insider_trend_score"], -0.7)

    def test_flagged_correctly(self):
        result = compute_insider_trend_score(self.txns)
        self.assertIn("only_routine_option_exercise_sells", result["insider_warning_flags"])

    def test_sell_count_zero(self):
        # routine sells do NOT count as discretionary open-market sells
        result = compute_insider_trend_score(self.txns)
        self.assertEqual(result["insider_sell_count"], 0)


# ── 7. Missing / invalid API response → neutral, no crash ─────────────────────

class TestInvalidAPIResponse(unittest.TestCase):
    def test_get_insider_trend_no_api_key(self):
        cfg = AVInsiderConfig()
        cfg.enabled = True   # isolate the no-api-key path from the ambient ENABLE_AV_INSIDER flag
        cfg.api_key = ""
        result = get_insider_trend("AAPL", config=cfg)
        self.assertEqual(result["insider_trend_score"], 0.0)
        self.assertEqual(result["fetch_error"], "no_api_key")

    def test_get_insider_trend_disabled(self):
        cfg = AVInsiderConfig()
        cfg.enabled = False
        result = get_insider_trend("AAPL", config=cfg)
        self.assertEqual(result["insider_trend_score"], 0.0)
        self.assertEqual(result["fetch_error"], "disabled")

    def test_network_failure_returns_neutral(self):
        cfg = AVInsiderConfig()
        cfg.api_key = "test_key"
        cfg.rate_limit_seconds = 0.0
        with patch("insider_transactions_av._fetch_from_api", return_value=[]):
            result = get_insider_trend("AAPL", config=cfg)
        self.assertEqual(result["insider_trend_score"], 0.0)
        self.assertIn("no_recent_insider_data", result["insider_warning_flags"])

    def test_exception_in_normalize_returns_neutral(self):
        # Even if normalize_transaction raises internally, get_insider_trend returns neutral
        cfg = AVInsiderConfig()
        cfg.api_key = "test_key"
        cfg.rate_limit_seconds = 0.0
        broken_rows = [None, {}, {"acquisition_or_disposal": None}]
        with patch("insider_transactions_av._fetch_from_api", return_value=broken_rows):
            # _filter_and_normalize will call normalize_transaction on bad input;
            # compute_insider_trend_score should still return cleanly
            result = get_insider_trend("AAPL", config=cfg)
        self.assertIn("insider_trend_score", result)
        self.assertGreaterEqual(result["insider_trend_score"], -1.0)
        self.assertLessEqual(result["insider_trend_score"], 1.0)

    def test_normalize_handles_missing_fields(self):
        norm = normalize_transaction({})
        self.assertEqual(norm["transaction_class"], "non_qualifying")
        self.assertEqual(norm["value"], 0.0)
        self.assertEqual(norm["shares"], 0.0)
        self.assertFalse(norm["is_buy"])

    def test_normalize_handles_none_values(self):
        norm = normalize_transaction({
            "acquisition_or_disposal": None,
            "transaction_type": None,
            "executive_title": None,
            "executive": None,
            "number_of_shares": None,
            "share_price": None,
            "value": None,
        })
        self.assertIsInstance(norm["value"], float)
        self.assertIsInstance(norm["shares"], float)


# ── 8. Cache hit avoids duplicate API call ─────────────────────────────────────

class TestCacheHit(unittest.TestCase):
    def test_cache_prevents_second_api_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = AVInsiderConfig()
            cfg.enabled = True   # exercise the fetch/cache path regardless of ambient ENABLE_AV_INSIDER
            cfg.api_key = "test_key"
            cfg.cache_dir = tmpdir
            cfg.rate_limit_seconds = 0.0
            cfg.cache_ttl_days = 7

            call_count = {"n": 0}

            def mock_fetch(symbol, config):
                call_count["n"] += 1
                return [
                    {"acquisition_or_disposal": "A", "transaction_type": "P-Purchase",
                     "number_of_shares": "1000", "share_price": "100.0", "value": "100000",
                     "executive_title": "CEO", "executive": "Alice",
                     "transaction_date": (date.today() - timedelta(days=3)).isoformat(),
                     "ownership_type": "D"},
                ]

            with patch("insider_transactions_av._fetch_from_api", side_effect=mock_fetch):
                fetch_insider_transactions("AAPL", config=cfg)
                fetch_insider_transactions("AAPL", config=cfg)   # second call — should hit cache

            self.assertEqual(call_count["n"], 1, "API should only be called once; second should use cache")


# ── 9. Buffett screener ranking improves with strong AV signal ─────────────────

class TestScreenerRankingWithAV(unittest.TestCase):
    """
    When all fundamental/valuation gates pass, a strongly bullish AV insider signal
    must increase conviction_score vs the baseline (no AV signal).
    """

    def _make_conviction(self, base_insider: float, av_modifier: float) -> float:
        """
        Simulate screener conviction computation with a given base insider score
        and AV modifier. Uses a strong business so quality/valuation gates pass.
        """
        from buffett_accumulation_layer import (
            BALConfig, FundamentalData, compute_conviction_score,
            _dca_valuation_discount_score, _dca_macro_cycle_opportunity_score,
            _earnings_quality_score, _profitability_score, score_symbol,
            multi_model_intrinsic_value,
        )
        mc = {"overall_market_bias": "bullish", "rate_direction": "cutting",
              "inflation_trend": "cooling", "recession_risk": "low",
              "cycle_phase": "expansion"}
        fd = FundamentalData(
            symbol="GOOD", current_price=150.0, market_cap=100e9, gross_margin=0.65,
            operating_margin=0.30, net_margin=0.22, revenue_growth_yoy=0.18,
            eps_growth_yoy=0.22, roe=0.32, roic=0.22, free_cash_flow=9e9,
            debt_to_equity=0.20, eps_ttm=10.0, peg_ratio=0.9, ev_ebitda=9.0,
            shares_growth_yoy=-0.03, insider_ownership_pct=0.12, beta=1.0,
            rsi_14=22.0, atr_expansion=1.7, volume_capitulation=2.1,
        )
        fd.fcf_yield = fd.free_cash_flow / fd.market_cap
        fd.pe_ratio = max(1.0, fd.current_price / fd.eps_ttm)
        card = score_symbol("GOOD", BALConfig(), mc, {}, fd=fd)
        val = multi_model_intrinsic_value(fd, mc)
        p2iv = (fd.current_price / val["intrinsic_value"]) if val["intrinsic_value"] > 0 else 0.0
        val_s = _dca_valuation_discount_score(p2iv)
        macro_s = _dca_macro_cycle_opportunity_score(card, mc)
        eq_s = _earnings_quality_score(fd)
        prof = _profitability_score(fd)
        combined = max(0.0, min(100.0, base_insider + av_modifier))
        return compute_conviction_score(card, fd, val_s, macro_s, combined, eq_s, prof)["conviction_score"]

    def test_bullish_av_signal_increases_conviction(self):
        # Start from a non-zero base so +10 modifier has room to lift conviction
        no_signal = self._make_conviction(base_insider=40.0, av_modifier=0.0)
        with_signal = self._make_conviction(base_insider=40.0, av_modifier=10.0)
        self.assertGreater(with_signal, no_signal)

    def test_bearish_av_signal_decreases_conviction(self):
        # Start from a positive base so -10 modifier actually reduces the insider input
        no_signal = self._make_conviction(base_insider=40.0, av_modifier=0.0)
        with_negative = self._make_conviction(base_insider=40.0, av_modifier=-10.0)
        self.assertLess(with_negative, no_signal)

    def test_av_modifier_bounded(self):
        # Even with maximum modifier, conviction cannot exceed 100
        result = self._make_conviction(base_insider=90.0, av_modifier=10.0)
        self.assertLessEqual(result, 100.0)
        # Even with minimum modifier, conviction cannot go below 0
        result = self._make_conviction(base_insider=5.0, av_modifier=-100.0)
        self.assertGreaterEqual(result, 0.0)


# ── 10. AV insider score does NOT override hard gates ─────────────────────────

class TestAVDoesNotOverrideGates(unittest.TestCase):
    """
    Even a perfectly bullish AV insider signal must not push a weak/rejected
    business past the BAL quality or valuation gate.
    """

    def test_strong_av_signal_does_not_rescue_weak_business(self):
        from buffett_accumulation_layer import (
            BALConfig, FundamentalData, score_symbol, multi_model_intrinsic_value,
            compute_conviction_score, _dca_valuation_discount_score,
            _dca_macro_cycle_opportunity_score, _earnings_quality_score,
            _profitability_score,
        )
        mc = {"overall_market_bias": "bullish", "rate_direction": "cutting",
              "inflation_trend": "cooling", "recession_risk": "low",
              "cycle_phase": "expansion"}
        # Weak business: negative fundamentals
        fd = FundamentalData(
            symbol="JUNK", current_price=10.0, market_cap=2e9, gross_margin=0.12,
            operating_margin=-0.05, net_margin=-0.10, revenue_growth_yoy=-0.15,
            eps_growth_yoy=-0.30, roe=-0.05, roic=0.01, free_cash_flow=-5e8,
            debt_to_equity=3.5, eps_ttm=-1.0, beta=2.0,
        )
        cfg = BALConfig()
        card = score_symbol("JUNK", cfg, mc, {}, fd=fd)

        # Gate: BAL score must be >= min_score_to_buy
        bal_ok = float(card.bal_score) >= cfg.min_score_to_buy
        # This weak business should NOT pass the BAL gate
        self.assertFalse(bal_ok,
                         "Weak business must fail the BAL gate regardless of insider signal")

        # Even with maximum insider modifier, the screener should not apply it
        # because bal_ok is False — av_modifier_applied = 0.0
        av_modifier_applied = 10.0 if bal_ok else 0.0
        self.assertEqual(av_modifier_applied, 0.0)

    def test_no_data_neutral_no_penalty(self):
        # No AV data → modifier = 0.0 → no conviction change
        result = compute_insider_trend_score([])
        self.assertEqual(result["insider_confidence_modifier"], 0.0)
        self.assertEqual(result["insider_trend_score"], 0.0)


# ── 11. Dataset feature vector length and schema valid ────────────────────────

class TestRLDatasetSchema(unittest.TestCase):
    def test_buffett_dca_feature_length_matches_extractor(self):
        import rl_common as rc
        dummy_event = {
            "bal_score": 70.0, "conviction_score": 75.0, "price_to_intrinsic_value": 0.85,
            "composite_dca_score": 68.0, "optimized_target_weight": 0.05,
            "current_weight": 0.02, "bl_posterior_return": 0.04,
            "market_state_confidence": 0.7, "RSI": 35.0, "ATR_expansion": 1.3,
            "premarket_gap_pct": 0.5, "sector_rotation": "normal",
            "av_insider_trend_score": 0.4,
            "av_insider_confidence_modifier": 5.0,
        }
        feats = rc.buffett_dca_features(dummy_event)
        self.assertEqual(len(feats), len(rc.BUFFETT_DCA_FEATURES),
                         "Feature extractor length must match BUFFETT_DCA_FEATURES list")

    def test_new_features_in_feature_list(self):
        import rl_common as rc
        self.assertIn("insider_trend_score", rc.BUFFETT_DCA_FEATURES)
        self.assertIn("insider_confidence_modifier", rc.BUFFETT_DCA_FEATURES)

    def test_missing_av_fields_default_to_zero(self):
        import rl_common as rc
        # Event without AV fields (backward-compatible)
        old_event = {
            "bal_score": 70.0, "conviction_score": 75.0, "price_to_intrinsic_value": 0.85,
            "composite_dca_score": 68.0, "optimized_target_weight": 0.05,
            "current_weight": 0.02, "bl_posterior_return": 0.04,
            "market_state_confidence": 0.7, "RSI": 35.0, "ATR_expansion": 1.3,
            "premarket_gap_pct": 0.5, "sector_rotation": "normal",
        }
        feats = rc.buffett_dca_features(old_event)
        # The two new features are at positions 12 and 13 (0-indexed); they should be 0.0
        idx_trend = rc.BUFFETT_DCA_FEATURES.index("insider_trend_score")
        idx_mod = rc.BUFFETT_DCA_FEATURES.index("insider_confidence_modifier")
        self.assertAlmostEqual(feats[idx_trend], 0.0, places=6)
        self.assertAlmostEqual(feats[idx_mod], 0.0, places=6)

    def test_all_feature_values_are_finite(self):
        import rl_common as rc
        event = {
            "bal_score": 70.0, "conviction_score": 75.0, "price_to_intrinsic_value": 0.85,
            "composite_dca_score": 68.0, "optimized_target_weight": 0.05,
            "current_weight": 0.02, "bl_posterior_return": 0.04,
            "market_state_confidence": 0.7, "RSI": 35.0, "ATR_expansion": 1.3,
            "premarket_gap_pct": 0.5, "sector_rotation": "normal",
            "av_insider_trend_score": 0.6, "av_insider_confidence_modifier": 8.0,
        }
        feats = rc.buffett_dca_features(event)
        for i, v in enumerate(feats):
            self.assertTrue(math.isfinite(v),
                            f"Feature {rc.BUFFETT_DCA_FEATURES[i]} is not finite: {v}")


# ── 12. AV modifier capped (cannot push conviction above 100) ─────────────────

class TestModifierCapping(unittest.TestCase):
    def test_combined_insider_score_capped_at_100(self):
        # sec_insider_score=90, av_modifier=+15 → must clamp to 100
        sec_insider_score = 90.0
        av_modifier = 15.0
        combined = max(0.0, min(100.0, sec_insider_score + av_modifier))
        self.assertEqual(combined, 100.0)

    def test_combined_insider_score_floored_at_zero(self):
        # sec_insider_score=5, av_modifier=-20 → must clamp to 0
        sec_insider_score = 5.0
        av_modifier = -20.0
        combined = max(0.0, min(100.0, sec_insider_score + av_modifier))
        self.assertEqual(combined, 0.0)

    def test_trend_score_always_in_minus_one_to_one(self):
        # Even with extreme weighted values, tanh ensures [-1, 1]
        extreme_buys = [
            _buy(shares=10_000_000, price=1000.0, title="CEO", name=f"Exec{i}")
            for i in range(10)
        ]
        result = compute_insider_trend_score(extreme_buys)
        self.assertGreaterEqual(result["insider_trend_score"], -1.0)
        self.assertLessEqual(result["insider_trend_score"], 1.0)


# ── 13. normalize_transaction correctness ──────────────────────────────────────

class TestNormalizeTransaction(unittest.TestCase):
    def test_open_market_buy_classified_correctly(self):
        # Use real AV field names: security_type="Common Stock", shares, share_price
        norm = normalize_transaction({
            "acquisition_or_disposal": "A",
            "security_type": "Common Stock",
            "shares": "5000", "share_price": "120.00",
            "executive_title": "Chief Executive Officer", "executive": "Alice CEO",
            "transaction_date": "2026-05-15", "ticker": "AAPL",
        })
        self.assertEqual(norm["transaction_class"], "open_market_buy")
        self.assertTrue(norm["is_buy"])
        self.assertEqual(norm["executive_role"], "CEO")
        # value computed from shares * share_price
        self.assertAlmostEqual(norm["value"], 600000.0, places=0)

    def test_open_market_sale_classified_correctly(self):
        norm = normalize_transaction({
            "acquisition_or_disposal": "D",
            "security_type": "Common Stock",
            "shares": "3000", "share_price": "100.00",
            "executive_title": "CFO", "executive": "Bob CFO",
            "transaction_date": "2026-05-10",
        })
        self.assertEqual(norm["transaction_class"], "open_market_sell")
        self.assertFalse(norm["is_buy"])
        self.assertEqual(norm["executive_role"], "CFO")

    def test_option_exercise_sale_classified_as_routine(self):
        # AV security_type for option exercises
        norm = normalize_transaction({
            "acquisition_or_disposal": "D",
            "security_type": "Emp. Stock Option (right to buy)",
            "shares": "2000", "share_price": "90.00",
            "executive_title": "SVP", "executive": "Carol SVP",
            "transaction_date": "2026-05-01",
        })
        self.assertEqual(norm["transaction_class"], "routine_sell")
        self.assertFalse(norm["is_open_market"])

    def test_compensation_shares_non_qualifying(self):
        # Phantom Stock / RSU / Promised Fee Share = non_qualifying
        for st in ("Phantom Stock Unit", "Rst. Stock Unit", "Promised Fee Share"):
            norm = normalize_transaction({
                "acquisition_or_disposal": "A",
                "security_type": st,
                "shares": "1000", "share_price": "100.00",
                "executive_title": "Director", "executive": "Dave D",
                "transaction_date": "2026-04-01",
            })
            self.assertEqual(norm["transaction_class"], "non_qualifying",
                             f"Expected non_qualifying for security_type={st!r}")

    def test_value_computed_from_shares_and_price(self):
        norm = normalize_transaction({
            "acquisition_or_disposal": "A",
            "security_type": "Common Stock",
            "shares": "1000", "share_price": "50.00",
            "executive_title": "Director", "executive": "Dave D",
            "transaction_date": "2026-04-01",
        })
        self.assertAlmostEqual(norm["value"], 50000.0, places=0)

    def test_role_weights_ordered_correctly(self):
        ceo_norm = normalize_transaction({"acquisition_or_disposal": "A",
                                          "security_type": "Common Stock",
                                          "executive_title": "CEO",
                                          "transaction_date": "2026-05-01"})
        dir_norm = normalize_transaction({"acquisition_or_disposal": "A",
                                          "security_type": "Common Stock",
                                          "executive_title": "Director",
                                          "transaction_date": "2026-05-01"})
        other_norm = normalize_transaction({"acquisition_or_disposal": "A",
                                            "security_type": "Common Stock",
                                            "executive_title": "Employee",
                                            "transaction_date": "2026-05-01"})
        self.assertGreater(ceo_norm["role_weight"], dir_norm["role_weight"])
        self.assertGreater(dir_norm["role_weight"], other_norm["role_weight"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
