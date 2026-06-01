"""
Tests for the gated wiring of the two strategy levers (rl_engine_levers) and the
safety invariants of the engine hooks. Verifies the levers are OFF by default, sit
behind the deployment gate, are paper-only, are suppressive-only, and that the alpha
engine's HARD exits (stop / 3R / kill switch) are evaluated before — and untouched by —
the dynamic-exit profile.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rl_engine_levers as lv

ROOT = Path(__file__).resolve().parents[1]


def _clean_env(**over):
    base = {k: v for k, v in os.environ.items()
            if not k.startswith(("ENABLE_RL_", "RL_LEVER_", "MODE", "ALPACA_PAPER_ONLY"))}
    base.update(over)
    return mock.patch.dict(os.environ, base, clear=True)


class DefaultOffTests(unittest.TestCase):
    def test_all_levers_off_by_default(self):
        with _clean_env():
            self.assertFalse(lv.dynamic_exit_active())
            self.assertFalse(lv.regime_gate_active("alpha_entry"))
            self.assertFalse(lv.regime_gate_active("buffett_dca"))
            # suppress_entry must be False when the lever is off, even with a bearish tape
            self.assertFalse(lv.suppress_entry("alpha_entry", {"spy_trend": "down"}, None))


class GateTests(unittest.TestCase):
    def test_flag_on_but_gate_blocks(self):
        with _clean_env(ENABLE_RL_DYNAMIC_EXIT="1"), \
             mock.patch.object(lv, "deployment_status", return_value="do_not_deploy"):
            self.assertFalse(lv.dynamic_exit_active())      # gate required by default → blocked

    def test_flag_on_and_gate_allows(self):
        with _clean_env(ENABLE_RL_DYNAMIC_EXIT="1"), \
             mock.patch.object(lv, "deployment_status", return_value="paper_mode_only"):
            self.assertTrue(lv.dynamic_exit_active())

    def test_gate_bypass_flag(self):
        with _clean_env(ENABLE_RL_REGIME_GATE="1", RL_LEVER_REQUIRE_GATE="0"):
            self.assertTrue(lv.regime_gate_active("alpha_entry"))

    def test_unknown_status_blocks_when_gate_required(self):
        with _clean_env(ENABLE_RL_DYNAMIC_EXIT="1"), \
             mock.patch.object(lv, "deployment_status", return_value="unknown"):
            self.assertFalse(lv.dynamic_exit_active())

    def test_live_mode_blocks_even_with_gate_bypass(self):
        with _clean_env(ENABLE_RL_DYNAMIC_EXIT="1", RL_LEVER_REQUIRE_GATE="0", MODE="live"):
            self.assertFalse(lv.dynamic_exit_active())

    def test_paper_only_via_alpaca_flag(self):
        with _clean_env(ENABLE_RL_REGIME_GATE="1", RL_LEVER_REQUIRE_GATE="0",
                        ALPACA_PAPER_ONLY="false"):
            self.assertFalse(lv.regime_gate_active("alpha_entry"))


class RegimeLogicTests(unittest.TestCase):
    def test_confirmed_bull_from_sma200(self):
        self.assertTrue(lv.spy_confirmed_bull(None, {"SPY": {"price": 500, "indicators": {"sma_200": 450}}}))
        self.assertFalse(lv.spy_confirmed_bull(None, {"SPY": {"price": 400, "indicators": {"sma_200": 450}}}))

    def test_confirmed_bull_from_trend_fallback(self):
        self.assertTrue(lv.spy_confirmed_bull({"spy_trend": "up"}, None))
        self.assertFalse(lv.spy_confirmed_bull({"spy_trend": "down"}, None))

    def test_unknown_is_not_bull(self):
        self.assertFalse(lv.spy_confirmed_bull(None, None))          # never assume missing = bullish

    def test_suppress_only_when_active_and_not_bull(self):
        with _clean_env(ENABLE_RL_REGIME_GATE="1", RL_LEVER_REQUIRE_GATE="0"):
            self.assertTrue(lv.suppress_entry("alpha_entry", {"spy_trend": "down"}, None))
            self.assertFalse(lv.suppress_entry("alpha_entry", {"spy_trend": "up"}, None))


class ExitProfileTests(unittest.TestCase):
    def test_validated_profile_values(self):
        p = lv.exit_profile()
        self.assertEqual(p["partial_r"], 1.0)
        self.assertEqual(p["trail_activate_r"], 1.0)
        self.assertEqual(p["trail_offset_r"], 1.0)


class EngineSafetyInvariantTests(unittest.TestCase):
    """Source-level guarantees that hard exits remain first and the dynamic profile only
    re-times the partial/trail — it can never touch the hard stop, 3R cap, or kill switch."""

    def setUp(self):
        self.src = (ROOT / "alpha_call_engine.py").read_text()

    def test_hard_exits_before_dynamic_partial_and_trail(self):
        i_hard = self.src.index("current_value <= call.hard_stop_value")
        i_3r = self.src.index("current_value >= call.r_target_value")
        i_partial = self.src.index("_partial_trigger")
        i_trail = self.src.index("_trail_activate_r")
        self.assertLess(i_hard, i_partial)
        self.assertLess(i_hard, i_trail)
        self.assertLess(i_3r, i_partial)

    def test_hard_stop_and_3r_do_not_use_dynamic_profile(self):
        # The hard stop and 3R target compare to the fixed stored values, never to _dyn.
        hard_line = [l for l in self.src.splitlines() if "current_value <= call.hard_stop_value" in l][0]
        tr_line = [l for l in self.src.splitlines() if "current_value >= call.r_target_value" in l][0]
        self.assertNotIn("_dyn", hard_line)
        self.assertNotIn("_dyn", tr_line)

    def test_dynamic_falls_back_to_config_when_inactive(self):
        # When _dyn is None (lever inactive/default) each trigger falls back to the engine's
        # existing stored value / config — i.e. behavior is unchanged.
        self.assertIn('if _dyn else call.partial_value', self.src)
        self.assertIn('_dyn["trail_activate_r"] if _dyn else self.config.r_trail_activate', self.src)
        self.assertIn('_dyn["trail_offset_r"] if _dyn else self.config.r_trail_offset', self.src)

    def test_regime_gate_is_suppressive_only_in_alpha(self):
        # the regime hook can only set entries_enabled = False, never True
        self.assertIn("entries_enabled = False", self.src)
        self.assertIn('suppress_entry("alpha_entry"', self.src)


class BuffettSafetyInvariantTests(unittest.TestCase):
    def setUp(self):
        self.src = (ROOT / "buffett_accumulation_layer.py").read_text()

    def test_regime_gate_uses_reject_and_continue(self):
        self.assertIn('suppress_entry("buffett_dca"', self.src)
        self.assertIn('_reject("regime_gate:SPY_not_confirmed_above_200DMA", "rejected_dca_regime_gate"); continue', self.src)

    def test_regime_gate_before_quality_gate(self):
        i_regime = self.src.index("regime_gate:SPY_not_confirmed_above_200DMA")
        i_quality = self.src.index("# ── 1. Quality gate")
        self.assertLess(i_regime, i_quality)


if __name__ == "__main__":
    unittest.main()
