"""
Tests for the SEC EDGAR Form 4 insider-buy DISCOVERY system and its Buffett
integration. Offline — XML fixtures + injected fundamentals; no network.

Verifies the discovery layer (parse / filter / aggregate / score) AND the integration
rule that SEC Form 4 only EXPANDS the review universe and never generates a buy or
bypasses any Buffett/RiskManager gate.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sec_form4_feed as sec
from sec_form4_feed import (
    parse_form4_xml, extract_open_market_purchases, aggregate_by_symbol,
    score_insider_aggregate, build_candidates, transaction_value,
    SECForm4Config, INSIDER_BUY_CANDIDATES_PATH,
)
from buffett_accumulation_layer import (
    BALConfig, DCAConfig, BuffettAccumulationLayer, FundamentalData, BUFFETT_DCA_SIGNALS_PATH,
)

BULLISH_CTX = {
    "overall_market_bias": "bullish", "rate_direction": "cutting",
    "inflation_trend": "cooling", "recession_risk": "low",
    "cycle_phase": "expansion", "fomc_stance": "dovish", "cpi_trend": "falling",
    "fiscal_policy": "stimulus", "tariff_risk": "low",
}


def _form4(ticker, name, owner, title, code, shares, price, ad="A", di="D",
           sec_title="Common Stock", date="2020-01-15", is_dir="0", is_off="1"):
    return f"""<?xml version="1.0"?>
<ownershipDocument>
 <issuer><issuerCik>0000000001</issuerCik><issuerName>{name}</issuerName>
   <issuerTradingSymbol>{ticker}</issuerTradingSymbol></issuer>
 <reportingOwner><reportingOwnerId><rptOwnerName>{owner}</rptOwnerName></reportingOwnerId>
   <reportingOwnerRelationship><isDirector>{is_dir}</isDirector><isOfficer>{is_off}</isOfficer>
     <officerTitle>{title}</officerTitle></reportingOwnerRelationship></reportingOwner>
 <nonDerivativeTable><nonDerivativeTransaction>
   <securityTitle><value>{sec_title}</value></securityTitle>
   <transactionDate><value>{date}</value></transactionDate>
   <transactionCoding><transactionCode>{code}</transactionCode></transactionCoding>
   <transactionAmounts><transactionShares><value>{shares}</value></transactionShares>
     <transactionPricePerShare><value>{price}</value></transactionPricePerShare>
     <transactionAcquiredDisposedCode><value>{ad}</value></transactionAcquiredDisposedCode></transactionAmounts>
   <ownershipNature><directOrIndirectOwnership><value>{di}</value></directOrIndirectOwnership></ownershipNature>
 </nonDerivativeTransaction></nonDerivativeTable>
</ownershipDocument>"""


def _meta(acc="0000000001-20-000001", url="https://www.sec.gov/x", fdate="2020-01-16"):
    return {"accession": acc, "filing_url": url, "filing_date": fdate}


def _strong_fd(symbol, price):
    fd = FundamentalData(symbol=symbol, current_price=price, market_cap=100e9, gross_margin=0.65,
        operating_margin=0.30, net_margin=0.22, revenue_growth_yoy=0.18, eps_growth_yoy=0.22,
        roe=0.32, roic=0.22, free_cash_flow=9e9, debt_to_equity=0.20, eps_ttm=10.0, peg_ratio=0.9,
        ev_ebitda=9.0, shares_growth_yoy=-0.03, insider_ownership_pct=0.12, beta=1.0,
        sma_50=price*1.12, sma_200=price*1.10, fifty_two_week_low=price*0.95,
        fifty_two_week_high=price*1.8, rsi_14=22.0, atr_expansion=1.7, volume_capitulation=2.1)
    fd.fcf_yield = fd.free_cash_flow / fd.market_cap
    fd.pe_ratio = max(1.0, price / fd.eps_ttm)
    return fd


def _weak_fd(symbol, price):
    return FundamentalData(symbol=symbol, current_price=price, market_cap=2e9, gross_margin=0.12,
        operating_margin=-0.05, net_margin=-0.10, revenue_growth_yoy=-0.15, eps_growth_yoy=-0.30,
        roe=-0.05, roic=0.01, free_cash_flow=-5e8, debt_to_equity=3.5, eps_ttm=-1.0,
        sma_50=price*1.4, sma_200=price*1.8, fifty_two_week_low=price*0.9,
        fifty_two_week_high=price*3.0, rsi_14=12.0, atr_expansion=2.5, volume_capitulation=3.0)


class _Pos:
    def __init__(self, market_value=0.0, avg_entry_price=0.0):
        self.market_value = market_value
        self.avg_entry_price = avg_entry_price


def _fresh_bal():
    b = BuffettAccumulationLayer(config=BALConfig(), dca_config=DCAConfig())
    b._ownership = {}
    return b


CFG = SECForm4Config()


class SECForm4DiscoveryTests(unittest.TestCase):

    # 1 ─────────────────────────────────────────────────────────────────────────
    def test_parses_open_market_purchase(self):
        parsed = parse_form4_xml(_form4("AAPL", "Apple Inc.", "COOK TIMOTHY", "CEO", "P", 800000, 170.0))
        self.assertEqual(parsed["issuer"]["ticker"], "AAPL")
        self.assertEqual(parsed["owner"]["role"], "CEO")
        rows = extract_open_market_purchases(parsed, **_meta())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["transaction_code"], "P")
        self.assertEqual(rows[0]["symbol"], "AAPL")

    # 2 ─────────────────────────────────────────────────────────────────────────
    def test_ignores_sales_grants_exercises(self):
        for code in ("S", "A", "M", "F"):
            parsed = parse_form4_xml(_form4("XOM", "Exxon", "SMITH J", "CFO", code, 5000, 100.0,
                                            ad=("D" if code in ("S", "F") else "A")))
            self.assertEqual(extract_open_market_purchases(parsed, **_meta()), [],
                             f"transaction code {code} must not be an open-market purchase")

    # 3 ─────────────────────────────────────────────────────────────────────────
    def test_transaction_value_calculation(self):
        parsed = parse_form4_xml(_form4("MSFT", "Microsoft", "NADELLA S", "CEO", "P", 12345, 67.89))
        txn = parsed["transactions"][0]
        self.assertAlmostEqual(transaction_value(txn), 12345 * 67.89, places=2)

    # 4 ─────────────────────────────────────────────────────────────────────────
    def test_filters_below_min_transaction_value(self):
        small = parse_form4_xml(_form4("TINY", "Tiny Co", "DOE J", "CEO", "P", 1000, 50.0))  # $50k
        res = build_candidates([(small, _meta())], CFG)
        self.assertEqual(res["candidates"], [], "sub-$100M aggregate must be rejected, not a candidate")
        self.assertEqual(len(res["rejected"]), 1)
        self.assertTrue(res["rejected"][0]["reject_reason"].startswith("below_min_transaction_value"))

    # 5 ─────────────────────────────────────────────────────────────────────────
    def test_aggregates_multiple_filings(self):
        f1 = parse_form4_xml(_form4("AAPL", "Apple Inc.", "COOK TIMOTHY", "CEO", "P", 800000, 170.0))
        f2 = parse_form4_xml(_form4("AAPL", "Apple Inc.", "MAESTRI LUCA", "CFO", "P", 200000, 165.0))
        agg = aggregate_by_symbol(
            extract_open_market_purchases(f1, accession="a1", filing_url="u1", filing_date="2020-01-16")
            + extract_open_market_purchases(f2, accession="a2", filing_url="u2", filing_date="2020-01-17"))
        a = agg["AAPL"]
        self.assertEqual(a["insider_count"], 2)
        self.assertEqual(a["transaction_count"], 2)
        self.assertAlmostEqual(a["total_open_market_purchase_value"], 800000 * 170 + 200000 * 165, places=2)
        self.assertEqual(len(a["filings"]), 2)
        self.assertTrue(a["repeated_purchases"])

    # 6 ─────────────────────────────────────────────────────────────────────────
    def test_computes_insider_buy_score(self):
        f1 = parse_form4_xml(_form4("AAPL", "Apple Inc.", "COOK TIMOTHY", "CEO", "P", 800000, 170.0))
        f2 = parse_form4_xml(_form4("AAPL", "Apple Inc.", "MAESTRI LUCA", "CFO", "P", 200000, 165.0))
        agg = aggregate_by_symbol(extract_open_market_purchases(f1, **_meta("a1"))
                                  + extract_open_market_purchases(f2, **_meta("a2")))["AAPL"]
        score, bd = score_insider_aggregate(agg, CFG)
        # value≥min(+40) + senior role(+15) + multiple insiders(+15) + direct(+10) + repeated(+10)
        self.assertGreaterEqual(score, 85.0)
        self.assertLessEqual(score, 100.0)
        self.assertIn("value>=min", bd)

    # 8 ─────────────────────────────────────────────────────────────────────────
    def test_discovery_does_not_generate_buys(self):
        f1 = parse_form4_xml(_form4("AAPL", "Apple Inc.", "COOK TIMOTHY", "CEO", "P", 800000, 170.0))
        res = build_candidates([(f1, _meta())], CFG)
        self.assertNotIn("orders", res)
        for c in res["candidates"]:
            self.assertNotIn("side", c)
            self.assertNotIn("qty", c)
            self.assertNotIn("order_dollars", c)
        # the module exposes no order-submission entry point
        self.assertFalse(any("submit" in n or "order" in n for n in dir(sec) if not n.startswith("_")
                             and callable(getattr(sec, n))))

    # 7 ─────────────────────────────────────────────────────────────────────────
    def test_sec_candidates_added_to_buffett_universe(self):
        import os
        INSIDER_BUY_CANDIDATES_PATH.write_text(json.dumps({
            "generated_at": "x", "source": "SEC_EDGAR_FORM4", "count": 1,
            "candidates": {"ZZTESTSEC": {"symbol": "ZZTESTSEC", "insider_buy_score": 90.0,
                                         "insider_buy_amount": 150_000_000, "insider_count": 2,
                                         "insider_roles": ["CEO"], "sec_filing_urls": ["u"]}},
        }), encoding="utf-8")
        # Form-4 expansion only applies in MERGE mode. The ambient production .env sets
        # BUFFETT_WATCHLIST_EXCLUSIVE=true (and any test that reloads `run` re-loads it via
        # load_dotenv(override=True), leaking it into os.environ), which suppresses the SEC
        # expansion. Force merge mode so this test exercises the expansion path it is about.
        saved_excl = os.environ.get("BUFFETT_WATCHLIST_EXCLUSIVE")
        os.environ["BUFFETT_WATCHLIST_EXCLUSIVE"] = "false"
        try:
            import importlib
            import buffett_accumulation_screener as bscr
            importlib.reload(bscr)
            self.assertIn("ZZTESTSEC", bscr.build_buffett_universe())
            self.assertIn("ZZTESTSEC", sec.discovered_symbols())
        finally:
            INSIDER_BUY_CANDIDATES_PATH.unlink(missing_ok=True)
            if saved_excl is None:
                os.environ.pop("BUFFETT_WATCHLIST_EXCLUSIVE", None)
            else:
                os.environ["BUFFETT_WATCHLIST_EXCLUSIVE"] = saved_excl

    # 9 ─────────────────────────────────────────────────────────────────────────
    def test_weak_company_with_insider_buy_is_rejected(self):
        b = _fresh_bal()
        fd = _weak_fd("WEAK", 10.0)
        insider = {"WEAK": {"amount": 150_000_000, "role": "CEO", "cluster": True, "score": 95.0, "detected": True}}
        orders = b.evaluate_dca(["WEAK"], BULLISH_CTX, {"WEAK": {"vwap": 9.9, "close": 10.0}},
                                100000.0, {}, 50000.0, fd_by_symbol={"WEAK": fd},
                                insider_data_by_symbol=insider)
        self.assertEqual(orders, [], "weak business must be rejected even with a huge SEC insider buy")
        sig = json.loads(BUFFETT_DCA_SIGNALS_PATH.read_text())["signals"][0]
        self.assertFalse(sig["insider_buy_used_in_conviction"])

    # 10 ────────────────────────────────────────────────────────────────────────
    def test_overvalued_company_with_insider_buy_is_watchlisted(self):
        b = _fresh_bal()
        fd = _strong_fd("RICHSEC", 550.0)        # excellent business, trading above intrinsic
        b._ownership["RICHSEC"] = {"symbol": "RICHSEC", "dca_count": 1,
                                   "target_position_pct": 0.10, "last_dca_date": "2000-01-01"}
        insider = {"RICHSEC": {"amount": 200_000_000, "role": "CEO", "cluster": True, "score": 95.0, "detected": True}}
        orders = b.evaluate_dca(["RICHSEC"], BULLISH_CTX, {"RICHSEC": {"vwap": 549, "close": 550}},
                                100000.0, {"RICHSEC": _Pos(market_value=2000.0, avg_entry_price=520.0)},
                                50000.0, fd_by_symbol={"RICHSEC": fd}, insider_data_by_symbol=insider)
        self.assertEqual(orders, [], "overvalued name must be watchlisted, not accumulated")
        sig = json.loads(BUFFETT_DCA_SIGNALS_PATH.read_text())["signals"][0]
        self.assertEqual(sig["journal_reason"], "rejected_dca_not_undervalued")
        self.assertFalse(sig["insider_buy_used_in_conviction"])
        self.assertEqual(sig["insider_buy_rejected_reason"], "not_undervalued")

    # 11 ────────────────────────────────────────────────────────────────────────
    def test_sec_failure_does_not_stop_buffett_screener(self):
        # Corrupt candidate file → loaders fail-safe to empty; universe still builds.
        INSIDER_BUY_CANDIDATES_PATH.write_text("{ this is not valid json", encoding="utf-8")
        try:
            self.assertEqual(sec.load_insider_candidates(), {})
            import importlib
            import buffett_accumulation_screener as bscr
            importlib.reload(bscr)
            universe = bscr.build_buffett_universe()
            self.assertTrue(len(universe) > 0, "screener universe must still build when SEC data is corrupt")
        finally:
            INSIDER_BUY_CANDIDATES_PATH.unlink(missing_ok=True)
        # The fail-safe driver never raises, even with no network available.
        res = sec.run_sec_form4_discovery(SECForm4Config(lookback_days=0))
        self.assertIn("candidates", res)
        self.assertEqual(res["source"], "SEC_EDGAR_FORM4")

    # 12 ────────────────────────────────────────────────────────────────────────
    def test_alpha_engine_unaffected_by_sec(self):
        import alpha_call_engine
        src = Path(alpha_call_engine.__file__).read_text()
        self.assertNotIn("sec_form4_feed", src, "Alpha engine must not depend on the SEC feed")
        self.assertNotIn("insider", src.lower(), "Alpha engine must remain insider-agnostic")


if __name__ == "__main__":
    unittest.main(verbosity=2)
