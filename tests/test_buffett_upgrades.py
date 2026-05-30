"""
Tests for the Buffett upgrades: multi-model valuation, conviction score, and the
insider-purchase conviction modifier. Offline — synthetic FundamentalData is injected.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from buffett_accumulation_layer import (
    BALConfig, DCAConfig, BuffettAccumulationLayer, FundamentalData, score_symbol,
    multi_model_intrinsic_value, compute_conviction_score, compute_insider_buy_score,
    conviction_target_allocation, _earnings_quality_score, _profitability_score,
    _dca_valuation_discount_score, _dca_macro_cycle_opportunity_score,
    BUFFETT_DCA_SIGNALS_PATH,
)

BULLISH_CTX = {
    "overall_market_bias": "bullish", "rate_direction": "cutting",
    "inflation_trend": "cooling", "recession_risk": "low",
    "cycle_phase": "expansion", "fomc_stance": "dovish", "cpi_trend": "falling",
    "fiscal_policy": "stimulus", "tariff_risk": "low",
}


def _strong_fd(symbol: str, price: float, *, full: bool = False) -> FundamentalData:
    """High-quality, undervalued business. full=True populates inputs for all 7 models."""
    fd = FundamentalData(symbol=symbol, current_price=price)
    fd.market_cap = 100e9
    fd.gross_margin = 0.65
    fd.operating_margin = 0.30
    fd.net_margin = 0.22
    fd.revenue_growth_yoy = 0.18
    fd.eps_growth_yoy = 0.22
    fd.roe = 0.32
    fd.roic = 0.22
    fd.free_cash_flow = 9e9
    fd.fcf_yield = fd.free_cash_flow / fd.market_cap
    fd.debt_to_equity = 0.20
    fd.eps_ttm = 10.0
    fd.pe_ratio = max(1.0, price / fd.eps_ttm)
    fd.peg_ratio = 0.9
    fd.ev_ebitda = 9.0
    fd.shares_growth_yoy = -0.03
    fd.insider_ownership_pct = 0.12
    fd.beta = 1.0
    fd.sma_50 = price * 1.12
    fd.sma_200 = price * 1.10
    fd.fifty_two_week_low = price * 0.95
    fd.fifty_two_week_high = price * 1.8
    fd.rsi_14 = 22.0
    fd.atr_expansion = 1.7
    fd.volume_capitulation = 2.1
    if full:
        fd.revenue_ttm = 40e9          # enables EV/EBIT
        fd.price_to_book = 6.0         # enables net-debt → EV models
        fd.sector = "Technology"       # enables sector-relative
    return fd


def _weak_fd(symbol: str, price: float) -> FundamentalData:
    fd = FundamentalData(symbol=symbol, current_price=price)
    fd.market_cap = 2e9
    fd.gross_margin = 0.12
    fd.operating_margin = -0.05
    fd.net_margin = -0.10
    fd.revenue_growth_yoy = -0.15
    fd.eps_growth_yoy = -0.30
    fd.roe = -0.05
    fd.roic = 0.01
    fd.free_cash_flow = -5e8
    fd.debt_to_equity = 3.5
    fd.eps_ttm = -1.0
    fd.sma_50 = price * 1.4
    fd.sma_200 = price * 1.8
    fd.fifty_two_week_low = price * 0.9
    fd.fifty_two_week_high = price * 3.0
    fd.rsi_14 = 12.0
    fd.atr_expansion = 2.5
    fd.volume_capitulation = 3.0
    return fd


class _Pos:
    def __init__(self, market_value=0.0, avg_entry_price=0.0):
        self.market_value = market_value
        self.avg_entry_price = avg_entry_price


def _fresh_bal() -> BuffettAccumulationLayer:
    b = BuffettAccumulationLayer(config=BALConfig(), dca_config=DCAConfig())
    b._ownership = {}
    return b


def _conviction_for(fd, mc=BULLISH_CTX, insider_score=0.0):
    card = score_symbol(fd.symbol, BALConfig(), mc, {}, fd=fd)
    detail = multi_model_intrinsic_value(fd, mc)
    iv = float(detail.get("intrinsic_value", 0.0) or 0.0)
    p2iv = (fd.current_price / iv) if iv > 0 else 0.0
    val_s = _dca_valuation_discount_score(round(p2iv, 4))
    macro_s = _dca_macro_cycle_opportunity_score(card, mc)
    eq_s = _earnings_quality_score(fd)
    prof_s = _profitability_score(fd)
    return compute_conviction_score(card, fd, val_s, macro_s, insider_score, eq_s, prof_s)["conviction_score"]


class BuffettUpgradeTests(unittest.TestCase):
    EQUITY = 100_000.0

    # 1 ─────────────────────────────────────────────────────────────────────────
    def test_multi_model_median_uses_only_valid_model_values(self):
        fd = _strong_fd("GOOD", 150.0)   # no sector / price_to_book / revenue → EV & sector skipped
        d = multi_model_intrinsic_value(fd, BULLISH_CTX)
        values = d["valuation_model_values"]
        # only models with available inputs are present (no fabricated zeros)
        self.assertNotIn("sector_relative", values)
        self.assertNotIn("ev_fcf_multiple", values)
        self.assertNotIn("ev_ebit_multiple", values)
        self.assertEqual(d["valuation_model_count"], len(values))
        self.assertTrue(all(v > 0 for v in values.values()))
        # intrinsic value is the median of exactly the valid model values
        s = sorted(values.values())
        n = len(s)
        median = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
        self.assertAlmostEqual(d["intrinsic_value"], round(median, 2), places=2)

    # 2 ─────────────────────────────────────────────────────────────────────────
    def test_valuation_confidence_drops_when_few_models(self):
        full = multi_model_intrinsic_value(_strong_fd("FULL", 150.0, full=True), BULLISH_CTX)
        few_fd = FundamentalData(symbol="FEW", current_price=150.0, market_cap=100e9,
                                 free_cash_flow=9e9, beta=1.0, eps_growth_yoy=0.10)  # fcf-only → dcf+owner
        few = multi_model_intrinsic_value(few_fd, BULLISH_CTX)
        self.assertGreaterEqual(full["valuation_model_count"], 5)
        self.assertLess(few["valuation_model_count"], 3)
        self.assertTrue(few["valuation_fallback_used"])
        self.assertGreater(full["valuation_confidence"], few["valuation_confidence"])

    # 3 ─────────────────────────────────────────────────────────────────────────
    def test_conviction_higher_for_quality_undervalued_business(self):
        strong = _conviction_for(_strong_fd("GOOD", 150.0))
        weak = _conviction_for(_weak_fd("JUNK", 10.0))
        self.assertGreater(strong, weak)
        self.assertGreaterEqual(strong, 60.0)   # eligible for allocation
        self.assertLess(weak, 60.0)             # watchlist only

    # 4 ─────────────────────────────────────────────────────────────────────────
    def test_insider_buy_increases_conviction_only_when_bal_and_valuation_pass(self):
        fd = _strong_fd("GOOD", 150.0)   # high BAL + undervalued → insider is usable
        insider = {"GOOD": {"amount": 2_000_000, "role": "CEO", "cluster": True}}
        b1 = _fresh_bal(); b2 = _fresh_bal()
        b1.evaluate_dca(symbols=["GOOD"], market_context=BULLISH_CTX,
                        bars_ctx_by_symbol={"GOOD": {"vwap": 149, "close": 150}},
                        equity=self.EQUITY, positions={}, available_capital=50_000.0,
                        fd_by_symbol={"GOOD": fd})
        conv_without = json.loads(BUFFETT_DCA_SIGNALS_PATH.read_text())["signals"][0]["conviction_score"]
        b2.evaluate_dca(symbols=["GOOD"], market_context=BULLISH_CTX,
                        bars_ctx_by_symbol={"GOOD": {"vwap": 149, "close": 150}},
                        equity=self.EQUITY, positions={}, available_capital=50_000.0,
                        fd_by_symbol={"GOOD": fd}, insider_data_by_symbol=insider)
        sig = json.loads(BUFFETT_DCA_SIGNALS_PATH.read_text())["signals"][0]
        self.assertTrue(sig["insider_buy_detected"])
        self.assertTrue(sig["insider_buy_used_in_conviction"])
        self.assertGreater(sig["conviction_score"], conv_without)

    # 5 ─────────────────────────────────────────────────────────────────────────
    def test_insider_buy_does_not_trigger_buy_for_weak_business(self):
        fd = _weak_fd("JUNK", 10.0)
        insider = {"JUNK": {"amount": 3_000_000, "role": "CEO", "cluster": True}}
        b = _fresh_bal()
        orders = b.evaluate_dca(symbols=["JUNK"], market_context=BULLISH_CTX,
                                bars_ctx_by_symbol={"JUNK": {"vwap": 9.9, "close": 10.0}},
                                equity=self.EQUITY, positions={}, available_capital=50_000.0,
                                fd_by_symbol={"JUNK": fd}, insider_data_by_symbol=insider)
        self.assertEqual(orders, [], "insider buying must not trigger a buy for a weak business")
        sig = json.loads(BUFFETT_DCA_SIGNALS_PATH.read_text())["signals"][0]
        self.assertTrue(sig["insider_buy_detected"])
        self.assertFalse(sig["insider_buy_used_in_conviction"])
        # A weak/broken business yields a quality rejection (either reason is valid).
        self.assertIn(sig["insider_buy_rejected_reason"], {"weak_fundamentals", "thesis_broken"})

    # 6 ─────────────────────────────────────────────────────────────────────────
    def test_dca_target_allocation_scales_with_conviction(self):
        large, _, _ = conviction_target_allocation(95.0)
        normal, _, _ = conviction_target_allocation(80.0)
        small, _, _ = conviction_target_allocation(65.0)
        watch, _, reason = conviction_target_allocation(50.0)
        self.assertGreater(large, normal)
        self.assertGreater(normal, small)
        self.assertGreater(small, watch)
        self.assertEqual(watch, 0.0)
        self.assertIn("watchlist", reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
