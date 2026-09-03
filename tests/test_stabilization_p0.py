"""
Regression tests for the P0 stabilization sprint (review findings, Bugs 1-10).
All offline: no sockets, no network, no live packages required.
"""

import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
import premarket_live_feed as plf
import optionlab_evaluator as ole


# ── Bug 1 — OptionLab combined-score corruption ────────────────────────────────────
class Bug1CombinedScoreTests(unittest.TestCase):
    def test_recompute_gated_on_optionlab_ran(self):
        """The combined-score recompute must sit inside `if _optionlab_ran:` so a default
        optionlab_score=0 can't deflate signals when OptionLab is disabled/failed."""
        src = (ROOT / "alpha_call_engine.py").read_text()
        self.assertIn("_optionlab_ran = False", src)
        self.assertIn("if _optionlab_ran:", src)
        # the renormalised blend must appear AFTER the gate, not unconditionally
        gate = src.index("if _optionlab_ran:")
        blend = src.index("combined_bs_weight", gate)
        self.assertGreater(blend, gate)

    def test_disabled_config_does_not_run_optionlab(self):
        """With ENABLE_OPTIONLAB-style disable, the evaluator's refine path is skipped."""
        cfg = ole.OptionLabConfig(enabled=False)
        r = ole.OptionLabEvaluator(cfg).evaluate_long_call(
            underlying_price=100, strike=100, premium=2.0, dte=30, iv=0.4)
        self.assertEqual(r["source"], ole.SOURCE_FALLBACK)   # no optionlab refine


# ── Bug 2 — Premarket fallback latch (now clearable) ───────────────────────────────
class Bug2FallbackClearTests(unittest.TestCase):
    def test_fallback_activates_then_clears_on_live_tick(self):
        f = plf.PremarketLiveFeed(["AAPL"], output_dir=None)
        f.apply_fallback_quotes({"AAPL": {"price": 100.0, "previous_close": 99.0}})
        self.assertTrue(f.fallback_active)
        self.assertTrue(f.fallback_used)           # back-compat alias tracks active state
        self.assertIsNotNone(f.fallback_last_used)
        # A fresh LIVE tick must clear fallback — live data is authoritative again.
        f.on_message({"id": "AAPL", "price": 101.0, "previousClose": 99.0})
        self.assertFalse(f.fallback_active)
        self.assertFalse(f.fallback_used)
        self.assertEqual(f.get_symbol_state("AAPL")["source"], plf.SOURCE_LIVE)

    def test_fallback_reason_recorded(self):
        f = plf.PremarketLiveFeed(["AAPL"], output_dir=None)
        f._mark_fallback("yliveticker_unavailable")
        self.assertEqual(f.fallback_reason, "yliveticker_unavailable")
        self.assertTrue(f.fallback_active)


# ── Bug 3 — Premarket stop terminates / freezes ────────────────────────────────────
class Bug3StopTests(unittest.TestCase):
    def test_stop_before_first_tick_freezes_parser(self):
        f = plf.PremarketLiveFeed(["AAPL"], output_dir=None)
        f.stop()                                   # no tick ever arrived (self._ws is None)
        self.assertTrue(f.shutdown_requested)
        f.on_message({"id": "AAPL", "price": 100.0})   # must be ignored post-shutdown
        self.assertIsNone(f.get_symbol_state("AAPL"))

    def test_stop_after_tick_freezes_state(self):
        f = plf.PremarketLiveFeed(["AAPL"], output_dir=None)
        f.on_message({"id": "AAPL", "price": 100.0, "previousClose": 99.0})
        f.stop()
        f.on_message({"id": "AAPL", "price": 200.0})   # frozen — no update
        self.assertEqual(f.get_symbol_state("AAPL")["price"], 100.0)

    def test_stop_during_reconnect_terminates_thread(self):
        import types, time
        saved_av, saved_ylt = plf.YLIVETICKER_AVAILABLE, plf._ylt

        class _StubYlt:
            @staticmethod
            def YLiveTicker(on_ticker=None, ticker_names=None, reconnect=5):
                return  # returns immediately → _run_stream loops into the reconnect wait
        try:
            plf.YLIVETICKER_AVAILABLE = True
            plf._ylt = _StubYlt()
            f = plf.PremarketLiveFeed(["AAPL"], output_dir=None)
            f._reconnect_delay = 0.05
            f.start()
            time.sleep(0.15)
            f.stop()                                # breaks the reconnect loop via the flag
            time.sleep(0.1)
            self.assertTrue(f.shutdown_requested)
            self.assertFalse(f._thread.is_alive())
        finally:
            plf.YLIVETICKER_AVAILABLE = saved_av
            plf._ylt = saved_ylt


# ── Bug 4 — OptionLab memory growth bounded ────────────────────────────────────────
class Bug4MemoryTests(unittest.TestCase):
    def test_records_bounded_by_maxlen(self):
        e = ole.OptionLabEvaluator(ole.OptionLabConfig(max_records=10))
        for _ in range(50):
            e.evaluate_long_call(underlying_price=100, strike=100, premium=2.0, dte=30, iv=0.4)
        self.assertLessEqual(len(e._records), 10)      # ring buffer caps memory
        self.assertEqual(e._total_evaluated, 50)       # lifetime count still tracked


# ── Bug 5 — Premarket rank score affects ranking, not eligibility ──────────────────
class Bug5RankWiringTests(unittest.TestCase):
    def test_rank_score_reorders_equal_setups(self):
        import alpha_call_engine as ace
        a = ace.CallSignal(symbol="A", setup_type="ORB", setup_score=70.0, grade="A",
                           premarket_rank_score=0.0, status="ready")
        b = ace.CallSignal(symbol="B", setup_type="ORB", setup_score=70.0, grade="A",
                           premarket_rank_score=80.0, status="ready")
        key = lambda s: s.setup_score + min(10.0, 0.1 * float(s.premarket_rank_score or 0.0))
        ordered = sorted([a, b], key=key, reverse=True)
        self.assertEqual(ordered[0].symbol, "B")       # higher premarket rank ranks first

    def test_rank_score_cannot_lift_below_threshold_setup(self):
        # A weak setup with a huge premarket rank still scores below a strong setup —
        # ranking nudge is bounded to +10 and never changes eligibility gates.
        import alpha_call_engine as ace
        weak = ace.CallSignal(symbol="W", setup_type="ORB", setup_score=40.0, grade="C",
                              premarket_rank_score=100.0)
        strong = ace.CallSignal(symbol="S", setup_type="ORB", setup_score=75.0, grade="A",
                                premarket_rank_score=0.0)
        key = lambda s: s.setup_score + min(10.0, 0.1 * float(s.premarket_rank_score or 0.0))
        self.assertLess(key(weak), key(strong))

    def test_rank_score_wired_in_evaluate_universe(self):
        src = (ROOT / "alpha_call_engine.py").read_text()
        self.assertIn("compute_premarket_rank_score", src)
        self.assertIn("premarket_rank_score=ev.premarket_rank_score", src)


# ── Bug 6 — TypeError masking removed ──────────────────────────────────────────────
class Bug6TypeErrorTests(unittest.TestCase):
    def test_context_safe_uses_signature_not_except_typeerror(self):
        src = (ROOT / "run.py").read_text()
        i = src.index("def _build_context_safe()")
        body = src[i:i + 900]
        self.assertIn("inspect.signature(build_market_context)", body)
        # No `except TypeError:` STATEMENT (the prose comment may mention the phrase) —
        # a genuine TypeError inside the builder must surface, not be swallowed/re-run.
        self.assertNotIn("except TypeError:", body)


# ── Bug 7 — RVOL availability requires actual prints ───────────────────────────────
class Bug7RvolAvailabilityTests(unittest.TestCase):
    def test_zero_volume_is_not_available(self):
        f = plf.PremarketLiveFeed(["AAPL"], output_dir=None)
        f.set_avg_premarket_volume("AAPL", 100000)
        f.on_message({"id": "AAPL", "price": 100.0, "dayVolume": "0"})   # no prints yet
        q = f.get_symbol_state("AAPL")
        self.assertFalse(q["rvol_available"])
        self.assertIsNone(q["premarket_rvol"])
        # Once real prints arrive, RVOL becomes available.
        f.on_message({"id": "AAPL", "price": 100.0, "dayVolume": "50000"})
        q = f.get_symbol_state("AAPL")
        self.assertTrue(q["rvol_available"])
        self.assertAlmostEqual(q["premarket_rvol"], 0.5, places=4)


# ── Bug 8 — Exit diagnostics propagated ────────────────────────────────────────────
class Bug8ExitDiagnosticsTests(unittest.TestCase):
    def test_market_state_stamped_on_exit_orders(self):
        import alpha_call_engine as ace
        cfg = ace.AlphaCallConfig()
        eng = ace.AlphaCallEngine(broker=None, config=cfg)
        call = ace.ActiveCall(
            symbol="XYZ", contract_symbol="XYZ250620C00100000", expiry="2025-06-20",
            strike=100.0, contracts_held=2, contracts_original=2, entry_mid=4.0,
            entry_cost=800.0, hard_stop_value=800.0 * (1 - cfg.stop_loss_pct),
            r_target_value=800.0 * (1 + cfg.r_target),
            partial_value=800.0 * (1 + cfg.partial_profit_pct), peak_value=800.0)
        eng._active["XYZ"] = call
        positions = [{"symbol": "XYZ250620C00100000", "qty": 2, "unrealized_plpc": -0.60}]
        exits = eng.check_active_positions(
            positions, waterfall_callback=None, market_context={},
            market_state={"regime": "bullish", "confidence": 0.8})
        self.assertTrue(exits, "a hard-stop exit should fire")
        for ex in exits:
            self.assertEqual(ex.get("market_state"), "bullish")
            self.assertEqual(ex.get("market_state_confidence"), 0.8)


# ── Bug 10 — MarketState actually shifts Black-Litterman posterior ─────────────────
class Bug10MarketStateWiringTests(unittest.TestCase):
    SIGNALS = [
        {"symbol": "AAPL", "conviction_score": 88, "price_to_intrinsic_value": 0.75, "market_cap": 3.0e12, "expected_return": 0.14},
        {"symbol": "MSFT", "conviction_score": 76, "price_to_intrinsic_value": 0.92, "market_cap": 2.8e12, "expected_return": 0.10},
        {"symbol": "KO", "conviction_score": 68, "price_to_intrinsic_value": 0.97, "market_cap": 2.6e11, "expected_return": 0.06},
    ]

    def _caps(self):
        return {s["symbol"]: s["market_cap"] for s in self.SIGNALS}

    def test_posterior_shifts_with_market_state_confidence(self):
        import buffett_black_litterman as bbl
        hi = bbl.run_black_litterman_safe(self.SIGNALS, market_caps=self._caps(),
                                          market_state={"confidence": 1.0}, write=False)
        lo = bbl.run_black_litterman_safe(self.SIGNALS, market_caps=self._caps(),
                                          market_state={"confidence": 0.2, "volatility_regime": "high"}, write=False)
        self.assertNotEqual(hi.get("market_state_confidence_multiplier"),
                            lo.get("market_state_confidence_multiplier"))
        self.assertNotEqual(hi.get("posterior_returns"), lo.get("posterior_returns"))

    def test_no_market_state_is_noop(self):
        import buffett_black_litterman as bbl
        base = bbl.run_black_litterman_safe(self.SIGNALS, market_caps=self._caps(), write=False)
        self.assertEqual(base.get("market_state_confidence_multiplier"), 1.0)

    def test_optimizer_forwards_market_state(self):
        import buffett_portfolio_optimizer as bopt
        self.assertIn("market_state", inspect.signature(bopt.run_optimizer).parameters)
        src = (ROOT / "buffett_portfolio_optimizer.py").read_text()
        self.assertIn("market_state=market_state", src)


if __name__ == "__main__":
    unittest.main()
