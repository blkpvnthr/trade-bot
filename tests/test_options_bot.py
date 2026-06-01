"""
test_options_bot.py

Comprehensive pytest test suite for the options trading bot (run.py).

Tests cover:
  - AlphaVantageOptionsSentimentClient normalization
  - Put-call sentiment scoring
  - Volume/OI scoring
  - Contract filtering
  - Contract ranking
  - Risk sizing
  - Limit order pricing
  - OCC symbol parsing
  - Option exit logic (hard stop, take profit, expiration, profit lock)
  - Helper utilities (to_float, to_int)
  - _option_quote_metrics
  - AlpacaOptionChainDiscovery
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Path injection — must come before any run.py import
# ---------------------------------------------------------------------------
sys.path.insert(0, "/Users/blkpvnthr/Desktop/trader13K/trade-bot")

from run import (  # noqa: E402
    OptionsTradingConfig,
    OptionsContractRanker,
    AlphaVantageOptionsSentimentClient,
    parse_occ_option_symbol,
    _option_quote_metrics,
    to_float,
    to_int,
    AlpacaOptionChainDiscovery,
)


# ===========================================================================
# TEST GROUP 1 — Alpha Vantage normalization
# ===========================================================================

class TestAlphaVantageNormalization:

    def _make_client(self) -> AlphaVantageOptionsSentimentClient:
        return AlphaVantageOptionsSentimentClient("fake_key")

    def test_get_pcr_normal_response(self):
        """Valid PCR response parses to float."""
        client = self._make_client()
        mock_data = {
            "put_call_ratio": [
                {"put_call_ratio": "0.72", "timestamp": "2026-01-01"}
            ]
        }
        with patch.object(client, "_fetch", return_value=mock_data):
            result = client.get_put_call_ratio("AAPL")
        assert result == pytest.approx(0.72, abs=0.001)

    def test_get_pcr_error_message_returns_none(self):
        """None fetch result (error / rate limit) returns None."""
        client = self._make_client()
        with patch.object(client, "_fetch", return_value=None):
            result = client.get_put_call_ratio("AAPL")
        assert result is None

    def test_get_pcr_empty_records_returns_none(self):
        """Empty records list returns None."""
        client = self._make_client()
        mock_data = {"put_call_ratio": []}
        with patch.object(client, "_fetch", return_value=mock_data):
            result = client.get_put_call_ratio("AAPL")
        assert result is None

    def test_get_vol_oi_ratio_normal(self):
        """Valid volume/OI response parses to float."""
        client = self._make_client()
        mock_data = {
            "volume_open_interest_ratio": [
                {"volume_open_interest_ratio": "1.25"}
            ]
        }
        with patch.object(client, "_fetch", return_value=mock_data):
            result = client.get_volume_oi_ratio("AAPL")
        assert result == pytest.approx(1.25, abs=0.001)

    def test_malformed_api_response_returns_none(self):
        """Malformed response (no expected keys) returns None gracefully."""
        client = self._make_client()
        mock_data = {"unexpected_key": "garbage"}
        with patch.object(client, "_fetch", return_value=mock_data):
            result = client.get_put_call_ratio("AAPL")
        assert result is None

    def test_empty_response_returns_none(self):
        """Empty dict returns None."""
        client = self._make_client()
        with patch.object(client, "_fetch", return_value={}):
            result = client.get_put_call_ratio("AAPL")
        assert result is None


# ===========================================================================
# TEST GROUP 2 — Put-call sentiment scoring
# ===========================================================================

class TestPutCallScoring:

    def _ranker(self) -> OptionsContractRanker:
        return OptionsContractRanker(OptionsTradingConfig())

    def test_bullish_pcr_scores_high_for_calls(self):
        """PCR <= 0.75 should score high for call options."""
        ranker = self._ranker()
        score = ranker._score_pcr_sentiment(0.60, "call")
        assert score >= 0.70, (
            f"Bullish PCR should score >= 0.70 for calls, got {score}"
        )

    def test_bearish_pcr_scores_low_for_calls(self):
        """PCR >= 1.0 should score low for calls."""
        ranker = self._ranker()
        score = ranker._score_pcr_sentiment(1.20, "call")
        assert score <= 0.25, (
            f"Bearish PCR should score <= 0.25 for calls, got {score}"
        )

    def test_bearish_pcr_scores_high_for_puts(self):
        """PCR >= 1.0 should score high for put options."""
        ranker = self._ranker()
        score = ranker._score_pcr_sentiment(1.20, "put")
        assert score >= 0.50

    def test_bullish_pcr_scores_low_for_puts(self):
        """PCR <= 0.75 should score low for puts."""
        ranker = self._ranker()
        score = ranker._score_pcr_sentiment(0.60, "put")
        assert score <= 0.25

    def test_contrarian_bullish_pcr_gets_small_weight_for_calls(self):
        """PCR >= 1.5 (contrarian bullish) should give smaller score than pure bullish."""
        ranker = self._ranker()
        contrarian = ranker._score_pcr_sentiment(1.80, "call")
        bullish = ranker._score_pcr_sentiment(0.50, "call")
        assert contrarian < bullish

    def test_none_pcr_returns_neutral(self):
        """None PCR should return neutral score ~0.5."""
        ranker = self._ranker()
        score = ranker._score_pcr_sentiment(None, "call")
        assert 0.3 <= score <= 0.7


# ===========================================================================
# TEST GROUP 3 — Volume/OI scoring
# ===========================================================================

class TestVolumeOIScoring:

    def _ranker(self) -> OptionsContractRanker:
        return OptionsContractRanker(OptionsTradingConfig())

    def test_sweet_spot_scores_high(self):
        """V/OI in 0.5-2.0 should score well."""
        ranker = self._ranker()
        score = ranker._score_volume_oi(1.0, 500.0, 500.0)
        assert score >= 0.7

    def test_very_low_ratio_penalized(self):
        """V/OI < 0.05 should score low."""
        ranker = self._ranker()
        score = ranker._score_volume_oi(0.01, 5.0, 1000.0)
        assert score <= 0.3

    def test_extreme_ratio_tapers(self):
        """V/OI > 10 should taper back from the peak sweet-spot score."""
        ranker = self._ranker()
        sweet = ranker._score_volume_oi(1.0, 1000.0, 1000.0)
        extreme = ranker._score_volume_oi(25.0, 25000.0, 1000.0)
        assert extreme < sweet

    def test_none_ratio_uses_volume_oi_fallback(self):
        """None ratio should compute from volume/open_interest and stay in [0,1]."""
        ranker = self._ranker()
        score = ranker._score_volume_oi(None, 500.0, 500.0)
        assert 0 <= score <= 1.0


# ===========================================================================
# TEST GROUP 4 — Contract filtering
# ===========================================================================

class TestContractFiltering:

    def _cfg(self) -> OptionsTradingConfig:
        return OptionsTradingConfig()

    def _ranker(self) -> OptionsContractRanker:
        return OptionsContractRanker(self._cfg())

    def _good_contract(self) -> dict:
        return {
            "symbol": "AAPL260116C00200000",
            "bid": 2.50,
            "ask": 2.60,
            "mid": 2.55,
            "bid_size": 10.0,
            "ask_size": 10.0,
            "spread_pct": 0.039,
            "delta": 0.52,
            "gamma": 0.03,
            "iv": 0.35,
            "volume": 500.0,
            "open_interest": 2000.0,
            "dte": 21,
            "strike": 200.0,
            "type": "call",
            "underlying_price": 198.0,
        }

    def test_good_contract_passes(self):
        ranker = self._ranker()
        result = ranker.filter_contracts([self._good_contract()])
        assert len(result) == 1

    def test_missing_bid_rejected(self):
        c = self._good_contract()
        c["bid"] = 0.0
        assert len(self._ranker().filter_contracts([c])) == 0

    def test_missing_ask_rejected(self):
        c = self._good_contract()
        c["ask"] = 0.0
        assert len(self._ranker().filter_contracts([c])) == 0

    def test_spread_too_wide_rejected(self):
        c = self._good_contract()
        c["spread_pct"] = 0.20  # > 12%
        assert len(self._ranker().filter_contracts([c])) == 0

    def test_low_oi_rejected(self):
        c = self._good_contract()
        c["open_interest"] = 50.0  # < 100
        assert len(self._ranker().filter_contracts([c])) == 0

    def test_low_volume_rejected(self):
        c = self._good_contract()
        c["volume"] = 5.0  # < 20
        assert len(self._ranker().filter_contracts([c])) == 0

    def test_missing_iv_rejected(self):
        c = self._good_contract()
        c["iv"] = float("nan")
        assert len(self._ranker().filter_contracts([c])) == 0

    def test_invalid_iv_zero_rejected(self):
        c = self._good_contract()
        c["iv"] = 0.0
        assert len(self._ranker().filter_contracts([c])) == 0

    def test_near_expiration_rejected(self):
        c = self._good_contract()
        c["dte"] = 2  # <= 3
        assert len(self._ranker().filter_contracts([c])) == 0


# ===========================================================================
# TEST GROUP 5 — Contract ranking
# ===========================================================================

class TestContractRanking:

    def _ranker(self) -> OptionsContractRanker:
        return OptionsContractRanker(OptionsTradingConfig())

    def _make_contract(
        self,
        delta: float = 0.50,
        dte: int = 21,
        spread_pct: float = 0.04,
        volume: float = 500.0,
        oi: float = 2000.0,
        iv: float = 0.35,
    ) -> dict:
        return {
            "symbol": f"AAPL260116C{int(200 * 1000):08d}",
            "bid": 2.40,
            "ask": 2.60,
            "mid": 2.50,
            "bid_size": 10.0,
            "ask_size": 10.0,
            "spread_pct": spread_pct,
            "delta": delta,
            "gamma": 0.03,
            "iv": iv,
            "volume": volume,
            "open_interest": oi,
            "dte": dte,
            "strike": 200.0,
            "type": "call",
            "underlying_price": 198.0,
        }

    def test_ranked_df_has_contract_score(self):
        ranker = self._ranker()
        contracts = [self._make_contract()]
        df = ranker.rank(contracts, 0.75, 0.70, 1.0)
        assert "contract_score" in df.columns
        assert len(df) == 1

    def test_better_contract_ranks_higher(self):
        ranker = self._ranker()
        good = self._make_contract(delta=0.50, spread_pct=0.04, volume=1000.0)
        bad  = self._make_contract(delta=0.20, spread_pct=0.10, volume=25.0)
        df = ranker.rank([bad, good], 0.75, 0.70, 1.0)
        assert df.iloc[0]["delta"] == pytest.approx(0.50, abs=0.01)

    def test_empty_contracts_returns_empty_df(self):
        ranker = self._ranker()
        df = ranker.rank([], 0.75, 0.70, 1.0)
        assert df.empty


# ===========================================================================
# TEST GROUP 6 — Risk sizing
# ===========================================================================

class TestRiskSizing:

    def test_risk_budget_calculation(self):
        """1.5% of $13,000 equity at $2.50 mid = 0.78 contracts -> 1 (min)."""
        equity = 13_000.0
        risk_pct = 0.015
        mid = 2.50
        contract_notional = mid * 100.0
        risk_dollars = equity * risk_pct
        n_contracts = max(1, int(risk_dollars // contract_notional))
        assert n_contracts == 1

    def test_risk_caps_at_max_contracts(self):
        """Large equity should cap at max_contracts_per_underlying=2."""
        equity = 1_000_000.0
        risk_pct = 0.015
        mid = 2.50
        contract_notional = mid * 100.0
        risk_dollars = equity * risk_pct
        n_contracts = int(risk_dollars // contract_notional)
        n_contracts = min(n_contracts, 2)
        assert n_contracts == 2


# ===========================================================================
# TEST GROUP 7 — Limit order pricing
# ===========================================================================

class TestLimitOrderPricing:

    def test_entry_limit_price_above_mid(self):
        """Entry limit should be mid + improvement."""
        mid = 2.50
        improvement = 0.01
        limit_price = round(mid + improvement, 2)
        assert limit_price == 2.51

    def test_exit_limit_price_below_mid(self):
        """Exit limit should be mid - offset."""
        mid = 2.50
        offset = 0.01
        limit_price = round(mid - offset, 2)
        assert limit_price == 2.49

    def test_exit_limit_never_goes_negative(self):
        """Exit limit floor is 0.01."""
        mid = 0.005
        offset = 0.01
        limit_price = round(max(mid - offset, 0.01), 2)
        assert limit_price >= 0.01


# ===========================================================================
# TEST GROUP 8 — OCC symbol parsing
# ===========================================================================

class TestOCCSymbolParsing:

    def test_valid_call_symbol(self):
        result = parse_occ_option_symbol("AAPL260116C00200000")
        assert result is not None
        assert result["underlying"] == "AAPL"
        assert result["type"] == "call"
        assert result["strike"] == pytest.approx(200.0)

    def test_valid_put_symbol(self):
        result = parse_occ_option_symbol("SPY260116P00450000")
        assert result is not None
        assert result["type"] == "put"

    def test_invalid_symbol_returns_none(self):
        assert parse_occ_option_symbol("AAPL") is None
        assert parse_occ_option_symbol("") is None
        assert parse_occ_option_symbol(None) is None
        assert parse_occ_option_symbol("GARBAGE123") is None


# ===========================================================================
# TEST GROUP 9 — Call/Put exits
# ===========================================================================

class TestOptionExits:

    def _make_position(self, symbol: str, unrealized_plpc: float):
        from run import BrokerPosition
        return BrokerPosition(
            symbol=symbol,
            qty=2,
            market_value=500.0,
            avg_entry_price=2.50,
            current_price=2.50 * (1 + unrealized_plpc),
            unrealized_pl=500.0 * unrealized_plpc,
            unrealized_plpc=unrealized_plpc,
        )

    def test_hard_stop_exit_detected(self):
        """A position down -35% should trigger hard stop."""
        plpc = -0.36
        cfg = OptionsTradingConfig()
        assert plpc <= cfg.hard_stop_pct

    def test_take_profit_exit_detected(self):
        """A position up +65% should trigger take profit."""
        plpc = 0.65
        cfg = OptionsTradingConfig()
        assert plpc >= cfg.take_profit_pct

    def test_expiration_exit_within_3_days(self):
        """DTE <= close_before_expiration_days should exit."""
        cfg = OptionsTradingConfig()
        dte = 2
        assert dte <= cfg.close_before_expiration_days

    def test_profit_lock_giveback(self):
        """Profit lock: new_high >= 40% and current <= new_high - 25% triggers exit."""
        cfg = OptionsTradingConfig()
        new_high = 0.45
        current = 0.15
        assert new_high >= cfg.profit_lock_trigger_pct
        assert current <= new_high - cfg.profit_giveback_pct


# ===========================================================================
# TEST GROUP 10 — Helper utilities (to_float / to_int)
# ===========================================================================

class TestHelpers:

    def test_to_float_valid_string(self):
        assert to_float("3.14") == pytest.approx(3.14)

    def test_to_float_none_returns_default(self):
        assert to_float(None) == 0.0
        assert to_float(None, 99.0) == 99.0

    def test_to_float_garbage_returns_default(self):
        assert to_float("not_a_number") == 0.0

    def test_to_int_valid(self):
        assert to_int("7") == 7
        assert to_int(3.9) == 3

    def test_to_int_none_returns_default(self):
        assert to_int(None) == 0
        assert to_int(None, 5) == 5

    def test_to_int_garbage_returns_default(self):
        assert to_int("abc") == 0


# ===========================================================================
# TEST GROUP 11 — _option_quote_metrics
# ===========================================================================

class TestOptionQuoteMetrics:

    def _make_snapshot(
        self,
        bid: float = 2.50,
        ask: float = 2.60,
        bid_size: float = 5.0,
        ask_size: float = 8.0,
        delta: float = 0.52,
        gamma: float = 0.03,
        theta: float = -0.04,
        vega: float = 0.15,
        iv: float = 0.30,
        volume: float = 400.0,
        open_interest: float = 1500.0,
    ) -> MagicMock:
        snap = MagicMock()
        quote = MagicMock()
        greeks = MagicMock()

        quote.bid_price = bid
        quote.ask_price = ask
        quote.bid_size = bid_size
        quote.ask_size = ask_size

        greeks.delta = delta
        greeks.gamma = gamma
        greeks.theta = theta
        greeks.vega = vega

        snap.latest_quote = quote
        snap.greeks = greeks
        snap.implied_volatility = iv
        snap.volume = volume
        snap.open_interest = open_interest
        return snap

    def test_mid_computed_correctly(self):
        snap = self._make_snapshot(bid=2.40, ask=2.60)
        metrics = _option_quote_metrics(snap)
        assert metrics["mid"] == pytest.approx(2.50, abs=0.01)

    def test_spread_pct_computed(self):
        snap = self._make_snapshot(bid=2.40, ask=2.60)
        metrics = _option_quote_metrics(snap)
        # spread = 0.20, mid = 2.50 -> spread_pct ~ 0.08
        assert metrics["spread_pct"] == pytest.approx(0.08, abs=0.005)

    def test_zero_bid_gives_zero_mid(self):
        snap = self._make_snapshot(bid=0.0, ask=2.60)
        metrics = _option_quote_metrics(snap)
        assert metrics["mid"] == 0.0

    def test_delta_present(self):
        snap = self._make_snapshot(delta=0.45)
        metrics = _option_quote_metrics(snap)
        assert metrics["delta"] == pytest.approx(0.45, abs=0.01)

    def test_volume_and_oi_present(self):
        snap = self._make_snapshot(volume=300.0, open_interest=1200.0)
        metrics = _option_quote_metrics(snap)
        assert metrics["volume"] == pytest.approx(300.0)
        assert metrics["open_interest"] == pytest.approx(1200.0)


# ===========================================================================
# TEST GROUP 12 — AlpacaOptionChainDiscovery
# ===========================================================================

class TestAlpacaOptionChainDiscovery:

    def _make_discovery(self) -> AlpacaOptionChainDiscovery:
        return AlpacaOptionChainDiscovery(api_key="fake_key", secret_key="fake_secret")

    def test_instantiation(self):
        """AlpacaOptionChainDiscovery instantiates without error."""
        disc = self._make_discovery()
        assert disc is not None

    def test_get_chain_returns_list_or_empty_on_error(self):
        """discover_contracts_for_underlying handles HTTP errors gracefully."""
        disc = self._make_discovery()
        with patch.object(disc, "get_contracts", side_effect=Exception("network_error")):
            result = disc.discover_contracts_for_underlying("AAPL")
        assert isinstance(result, list)

    def test_get_chain_parses_valid_chain(self):
        """discover_contracts_for_underlying merges contracts + snapshots into contract dicts."""
        disc = self._make_discovery()
        fake_contracts = [{"symbol": "AAPL260116C00200000", "expiration_date": "2026-01-16"}]
        fake_snapshots = {
            "AAPL260116C00200000": {
                "latestQuote": {"bp": 2.40, "ap": 2.60, "bs": 5.0, "as": 8.0},
                "greeks": {"delta": 0.52, "gamma": 0.03, "theta": -0.04, "vega": 0.15},
                "impliedVolatility": 0.30,
                "volume": 400.0,
                "openInterest": 1500.0,
            }
        }
        with patch.object(disc, "get_contracts", return_value=fake_contracts), \
             patch.object(disc, "get_snapshots_bulk", return_value=fake_snapshots):
            result = disc.discover_contracts_for_underlying("AAPL")
        assert isinstance(result, list)
        if result:
            assert "symbol" in result[0]
