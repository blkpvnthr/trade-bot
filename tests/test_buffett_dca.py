"""
Offline validation for the Buffett Accumulation Layer Composite DCA framework.

Confirms (no network — synthetic FundamentalData is injected):
  * buffett_dca_signals.json is written each cycle
  * RSI-only oversold candidates are rejected when valuation / business quality fail
  * undervalued high-BAL candidates receive a DCA signal + buy order
  * the max-tranche cap prevents endless DCA
  * DCA buy orders route through RiskManager.gate() / AlpacaGateway
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import buffett_accumulation_layer as bal_mod
from buffett_accumulation_layer import (
    BALConfig, DCAConfig, BuffettAccumulationLayer, FundamentalData,
    estimate_intrinsic_value, BUFFETT_DCA_SIGNALS_PATH,
)


# Bullish/expansion macro context → strong macro/policy/cycle sub-scores.
BULLISH_CTX = {
    "overall_market_bias": "bullish", "rate_direction": "cutting",
    "inflation_trend": "cooling", "recession_risk": "low",
    "cycle_phase": "expansion", "fomc_stance": "dovish", "cpi_trend": "falling",
    "fiscal_policy": "stimulus", "tariff_risk": "low",
}


def _high_quality_fd(symbol: str, price: float) -> FundamentalData:
    """A fortress-balance-sheet compounder. BAL score lands in the aggressive band."""
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
    fd.current_ratio = 2.5
    fd.eps_ttm = 10.0
    fd.pe_ratio = max(1.0, price / fd.eps_ttm)
    fd.peg_ratio = 0.9
    fd.ev_ebitda = 9.0
    fd.shares_growth_yoy = -0.03      # net buybacks
    fd.insider_ownership_pct = 0.12
    fd.beta = 1.0
    fd.sma_50 = price * 1.12          # price well below the 50DMA (dislocated)
    fd.sma_200 = price * 1.10
    fd.fifty_two_week_low = price * 0.95
    fd.fifty_two_week_high = price * 1.8
    fd.rsi_14 = 22.0                  # oversold (timing input only)
    fd.atr_pct = 0.03
    fd.atr_expansion = 1.7
    fd.volume_capitulation = 2.1
    return fd


def _junk_fd(symbol: str, price: float) -> FundamentalData:
    """Low-quality, deeply oversold name — RSI screams but the business is weak."""
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
    fd.fcf_yield = fd.free_cash_flow / fd.market_cap
    fd.debt_to_equity = 3.5
    fd.eps_ttm = -1.0
    fd.pe_ratio = 0.0
    fd.sma_50 = price * 1.4
    fd.sma_200 = price * 1.8
    fd.fifty_two_week_low = price * 0.9
    fd.fifty_two_week_high = price * 3.0
    fd.rsi_14 = 12.0                  # extremely oversold
    fd.atr_expansion = 2.5
    fd.volume_capitulation = 3.0
    return fd


class _Pos:
    def __init__(self, market_value=0.0, avg_entry_price=0.0):
        self.market_value = market_value
        self.avg_entry_price = avg_entry_price


def _fresh_bal() -> BuffettAccumulationLayer:
    b = BuffettAccumulationLayer(config=BALConfig(), dca_config=DCAConfig())
    b._ownership = {}          # isolate from any persisted ownership universe
    return b


class CompositeDCATests(unittest.TestCase):
    EQUITY = 100_000.0

    def test_intrinsic_value_discount(self):
        fd = _high_quality_fd("GOOD", price=150.0)
        iv, detail = estimate_intrinsic_value(fd)
        self.assertGreater(iv, fd.current_price, "high-quality cheap name should be undervalued")
        self.assertLess(fd.current_price / iv, 0.95, f"p2iv should clear small tier: {detail}")

    def test_undervalued_high_bal_gets_signal_and_order(self):
        b = _fresh_bal()
        fd = _high_quality_fd("GOOD", price=150.0)
        orders = b.evaluate_dca(
            symbols=["GOOD"], market_context=BULLISH_CTX,
            bars_ctx_by_symbol={"GOOD": {"vwap": 149.0, "close": 150.0}},
            equity=self.EQUITY, positions={}, available_capital=50_000.0,
            fd_by_symbol={"GOOD": fd},
        )
        self.assertEqual(len(orders), 1, "undervalued high-BAL name should produce one order")
        o = orders[0]
        self.assertEqual(o["side"], "buy")
        self.assertIn(o["action"], {"small", "normal", "aggressive"})
        self.assertGreater(o["qty"], 0)
        self.assertTrue(o["journal_reason"].startswith("dca_triggered"))
        # signals file written with the rich field set
        self.assertTrue(BUFFETT_DCA_SIGNALS_PATH.exists())
        payload = json.loads(BUFFETT_DCA_SIGNALS_PATH.read_text())
        sig = next(s for s in payload["signals"] if s["symbol"] == "GOOD")
        self.assertTrue(sig["risk_limits_passed"])
        for f in ("composite_dca_score", "valuation_discount_score", "technical_dislocation_score",
                  "business_quality_score", "macro_cycle_opportunity_score",
                  "sentiment_capitulation_score", "intrinsic_value", "price_to_intrinsic_value",
                  "remaining_dca_tranches", "DCA_tranche_size"):
            self.assertIn(f, sig)

    def test_rsi_only_junk_is_rejected_on_business_quality(self):
        b = _fresh_bal()
        fd = _junk_fd("JUNK", price=10.0)
        orders = b.evaluate_dca(
            symbols=["JUNK"], market_context=BULLISH_CTX,
            bars_ctx_by_symbol={"JUNK": {"vwap": 9.9, "close": 10.0}},
            equity=self.EQUITY, positions={}, available_capital=50_000.0,
            fd_by_symbol={"JUNK": fd},
        )
        self.assertEqual(orders, [], "low-quality oversold name must not DCA on RSI alone")
        payload = json.loads(BUFFETT_DCA_SIGNALS_PATH.read_text())
        sig = next(s for s in payload["signals"] if s["symbol"] == "JUNK")
        self.assertFalse(sig["risk_limits_passed"])
        self.assertIn(sig["journal_reason"], {"rejected_dca_BAL_score", "rejected_dca_thesis_broken"})

    def test_rsi_only_overvalued_existing_holding_rejected_not_undervalued(self):
        # Good business but trading ABOVE intrinsic value, already holding a tranche.
        # Oversold RSI alone must not justify adding.
        b = _fresh_bal()
        # Price set above the multi-model intrinsic estimate so the name is NOT at a
        # DCA-worthy discount (p2iv > 0.95) despite being oversold and high quality.
        fd = _high_quality_fd("RICH", price=550.0)
        iv, _ = estimate_intrinsic_value(fd)
        self.assertGreater(fd.current_price / iv, 0.95, "test setup must be overvalued")
        b._ownership["RICH"] = {"symbol": "RICH", "dca_count": 1,
                                "target_position_pct": 0.10, "last_dca_date": "2000-01-01"}
        orders = b.evaluate_dca(
            symbols=["RICH"], market_context=BULLISH_CTX,
            bars_ctx_by_symbol={"RICH": {"vwap": 549.0, "close": 550.0}},
            equity=self.EQUITY, positions={"RICH": _Pos(market_value=2_000.0, avg_entry_price=520.0)},
            available_capital=50_000.0, fd_by_symbol={"RICH": fd},
        )
        self.assertEqual(orders, [], "overvalued holding must not DCA on RSI alone")
        payload = json.loads(BUFFETT_DCA_SIGNALS_PATH.read_text())
        sig = next(s for s in payload["signals"] if s["symbol"] == "RICH")
        self.assertEqual(sig["journal_reason"], "rejected_dca_not_undervalued")

    def test_max_tranche_cap_prevents_endless_dca(self):
        b = _fresh_bal()
        fd = _high_quality_fd("GOOD", price=150.0)
        b._ownership["GOOD"] = {"symbol": "GOOD", "dca_count": 5,
                                "target_position_pct": 0.10, "last_dca_date": "2000-01-01"}
        orders = b.evaluate_dca(
            symbols=["GOOD"], market_context=BULLISH_CTX,
            bars_ctx_by_symbol={"GOOD": {"vwap": 149.0, "close": 150.0}},
            equity=self.EQUITY, positions={"GOOD": _Pos(market_value=4_000.0, avg_entry_price=150.0)},
            available_capital=50_000.0, fd_by_symbol={"GOOD": fd},
        )
        self.assertEqual(orders, [], "max tranche count must block further DCA")
        payload = json.loads(BUFFETT_DCA_SIGNALS_PATH.read_text())
        sig = next(s for s in payload["signals"] if s["symbol"] == "GOOD")
        self.assertEqual(sig["journal_reason"], "rejected_dca_max_tranches")

    def test_dca_orders_route_through_risk_manager_and_gateway(self):
        import run
        risk = run.RiskManager(equity=self.EQUITY)
        gateway = run.AlpacaGateway(risk)
        # Force simulation so the test never places a live/paper order — the risk gate
        # path is identical; submit_market_order still calls risk.gate() before the
        # sim branch records the fill.
        gateway._available = False
        gateway._mode = "sim"

        b = _fresh_bal()
        fd = _high_quality_fd("GOOD", price=150.0)

        class _WF:
            def consume_bal(self, amt):
                return amt

        # Patch evaluate_dca to use injected synthetic fundamentals (offline).
        orig = b.evaluate_dca
        b.evaluate_dca = lambda **kw: orig(fd_by_symbol={"GOOD": fd}, **kw)

        placed = run.run_bal_entries(
            b, gateway, _WF(), risk, symbols=["GOOD"], market_context=BULLISH_CTX,
            bars_ctx_by_symbol={"GOOD": {"vwap": 149.0, "close": 150.0}},
            equity_positions={},
        )
        self.assertEqual(placed, 1, "DCA order should be placed via the gateway")
        # Gateway records the fill on the RiskManager → proves it passed risk.gate()
        self.assertIn("GOOD", risk.open_positions)
        self.assertGreater(risk.open_positions["GOOD"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
