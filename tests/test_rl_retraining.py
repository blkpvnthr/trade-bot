"""
Tests for the two-engine RL retraining stack (offline). No sockets, no network, no live
packages required for the assertions, and no SB3 training in the unit suite (heuristic
paths keep it fast). Verifies datasets, environments, rewards, training/eval/gate, shadow
mode, and the offline-safety invariants (no AlpacaGateway, no order submission).
"""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

import rl_common as rc
import rl_training_dataset_builder as builder
import rl_env_alpha_entry as A
import rl_env_alpha_exit as B
import rl_env_buffett_dca as C
import rl_env_capital_allocation as D
import train_rl_agents as trainer
import evaluate_rl_agents as evaluator
import rl_shadow_mode as shadow


SYNTH_EVENTS = [
    {"event_type": "alpha_entry", "symbol": "AAA", "date_tag": "20260101", "setup_score": 80,
     "probability_of_profit": 0.7, "black_scholes_score": 70, "expected_R": 1.6},
    {"event_type": "alpha_exit", "symbol": "AAA", "date_tag": "20260101", "r_multiple": 2.0,
     "realized_R": 2.0, "probability_band": "80-100", "holding_period_days": 0},
    {"event_type": "bal_buy", "symbol": "GOOD", "date_tag": "20260101", "bal_score": 90,
     "conviction_score": 85, "composite_dca_score": 84, "price_to_intrinsic_value": 0.8,
     "optimized_target_weight": 0.08, "current_weight": 0.02},
]


class DatasetBuilderTests(unittest.TestCase):
    def test_1_reads_research_events(self):
        ds = builder.build_alpha_entry_dataset(SYNTH_EVENTS)
        self.assertTrue(ds)
        self.assertIn("realized_R", ds[0])
        self.assertIn("setup_score", ds[0])
        # realized_R for AAA entry should pick up the matched exit (2.0)
        self.assertAlmostEqual(ds[0]["realized_R"], 2.0, places=4)
        self.assertTrue(builder.build_buffett_dca_dataset(SYNTH_EVENTS))

    def test_2_handles_missing_files_safely(self):
        # No events → all builders return empty lists, no raise.
        self.assertEqual(builder.build_alpha_entry_dataset([]), [])
        self.assertEqual(builder.build_alpha_exit_dataset([]), [])
        self.assertEqual(builder.build_buffett_dca_dataset([]), [])
        self.assertEqual(builder.build_capital_allocation_dataset([]), [])
        with TemporaryDirectory() as d:
            info = builder.write_dataset([], "empty_ds", out_dir=Path(d))
            self.assertEqual(info["rows"], 0)
            self.assertTrue(Path(info["path"]).exists())


class EnvironmentTests(unittest.TestCase):
    def _step(self, env):
        obs, info = env.reset()
        self.assertEqual(len(obs), env.observation_space.shape[0])
        a = env.action_space.sample()
        obs2, reward, terminated, truncated, info2 = env.step(a)
        self.assertEqual(len(obs2), env.observation_space.shape[0])
        self.assertIsInstance(float(reward), float)
        self.assertIn(terminated, (True, False))

    def test_3_alpha_entry_env_steps(self):
        self._step(A.AlphaEntryEnv())

    def test_4_alpha_exit_env_steps(self):
        self._step(B.AlphaExitEnv())

    def test_5_buffett_dca_env_steps(self):
        self._step(C.BuffettDCAEnv())

    def test_6_capital_allocation_env_steps(self):
        self._step(D.CapitalAllocationEnv())


class RewardTests(unittest.TestCase):
    def test_7_rewards_compute_correctly(self):
        # entry: enter a winner > skip a winner; skip a loser is positive (avoided loss)
        self.assertGreater(rc.alpha_entry_reward("enter_now", 2.0), rc.alpha_entry_reward("skip", 2.0))
        self.assertGreater(rc.alpha_entry_reward("skip", -1.0), 0.0)
        # exit: avoiding an overnight loss rewards no_overnight over allow_overnight
        self.assertGreater(rc.alpha_exit_reward("no_overnight", -1.0, 0.0, overnight_loss=True),
                           rc.alpha_exit_reward("allow_overnight", -1.0, 0.0, overnight_loss=True))
        # dca: aggressive on a positive forward return beats no_dca
        self.assertGreater(rc.buffett_dca_reward("aggressive_dca", 0.2, 0.3, 0.02, 0.08),
                           rc.buffett_dca_reward("no_dca", 0.2, 0.3, 0.02, 0.08))
        # dca: buying past the optimized target is penalised
        self.assertLess(rc.buffett_dca_reward("aggressive_dca", 0.2, 0.3, 0.079, 0.08),
                        rc.buffett_dca_reward("aggressive_dca", 0.2, 0.3, 0.02, 0.08))
        # allocation reward is finite
        self.assertIsInstance(rc.capital_alloc_reward([0.5, 0.3, 0.2], [0.01, 0.005, 0.0], 0.0, 0.1), float)


class TrainEvalGateTests(unittest.TestCase):
    def test_8_models_are_saved(self):
        with TemporaryDirectory() as d:
            res = trainer.train_one("alpha_entry", algo="heuristic", out_dir=Path(d))
            self.assertTrue(res["saved"])
            self.assertTrue(Path(res["model_path"]).exists())
            self.assertIn("reward_function", res)

    def test_9_evaluation_compares_against_baseline(self):
        r = evaluator.evaluate_agent("buffett_dca")
        self.assertIn("baseline_metrics", r)
        self.assertIn("rl_metrics", r)
        self.assertIn("expectancy", r["baseline_metrics"])
        self.assertIn("deployment_recommendation", r)

    def test_10_gate_blocks_weak_models(self):
        # RL strictly worse expectancy, ample samples → do_not_deploy.
        rec, _ = rc.deployment_recommendation(
            {"expectancy": 0.5, "max_drawdown": 1.0, "win_rate": 0.55},
            {"expectancy": 0.2, "max_drawdown": 1.0, "win_rate": 0.55}, 500)
        self.assertEqual(rec, "do_not_deploy")
        # Materially worse drawdown → do_not_deploy even if expectancy ties.
        rec2, _ = rc.deployment_recommendation(
            {"expectancy": 0.5, "max_drawdown": 1.0, "win_rate": 0.5},
            {"expectancy": 0.6, "max_drawdown": 2.0, "win_rate": 0.5}, 500)
        self.assertEqual(rec2, "do_not_deploy")
        # Small sample → shadow_mode_only regardless of improvement.
        rec3, _ = rc.deployment_recommendation(
            {"expectancy": 0.3, "max_drawdown": 1.0, "win_rate": 0.5},
            {"expectancy": 0.9, "max_drawdown": 0.5, "win_rate": 0.6}, 10)
        self.assertEqual(rec3, "shadow_mode_only")
        # Genuine improvement, ample sample → paper_mode_only.
        rec4, _ = rc.deployment_recommendation(
            {"expectancy": 0.3, "max_drawdown": 2.0, "win_rate": 0.5},
            {"expectancy": 0.6, "max_drawdown": 1.5, "win_rate": 0.55}, 500)
        self.assertEqual(rec4, "paper_mode_only")


class ShadowModeTests(unittest.TestCase):
    def test_11_shadow_logs_recommendations_only(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "rec.jsonl"
            # log directly to a temp file by monkeypatching the path resolver
            orig = shadow._shadow_path
            shadow._shadow_path = lambda: p
            try:
                shadow.log_recommendation("alpha_entry", "AAA", {"setup_score": 0.8}, "enter_now")
                self.assertTrue(p.exists())
                rec = json.loads(p.read_text().splitlines()[0])
                self.assertEqual(rec["recommendation"], "enter_now")
                self.assertTrue(rec["live_action_unchanged"])
                self.assertTrue(rec["shadow_only"])
            finally:
                shadow._shadow_path = orig
        # Disabled by default → run_shadow_cycle is a no-op.
        import os
        os.environ.pop("ENABLE_RL_SHADOW_MODE", None)
        self.assertEqual(shadow.run_shadow_cycle(), {"enabled": False})


class OfflineSafetyTests(unittest.TestCase):
    RL_FILES = ["rl_common.py", "rl_training_dataset_builder.py", "rl_env_alpha_entry.py",
                "rl_env_alpha_exit.py", "rl_env_buffett_dca.py", "rl_env_capital_allocation.py",
                "train_rl_agents.py", "evaluate_rl_agents.py", "rl_shadow_mode.py"]

    def test_12_no_env_imports_alpaca_gateway(self):
        # Scan actual import statements / instantiation (prose docstrings may mention it).
        for f in self.RL_FILES:
            src = (ROOT / f).read_text()
            self.assertNotIn("AlpacaGateway(", src, f"{f} must not instantiate AlpacaGateway")
            for line in src.splitlines():
                s = line.strip()
                if s.startswith(("import ", "from ")):
                    self.assertNotIn("alpaca", s.lower(), f"{f} must not import alpaca: {s}")
                    self.assertNotIn("AlpacaGateway", s, f"{f} must not import AlpacaGateway: {s}")

    def test_13_no_training_script_submits_orders(self):
        for f in self.RL_FILES:
            src = (ROOT / f).read_text()
            for forbidden in ("submit_market_order", "submit_order", "place_order", "TradingClient"):
                self.assertNotIn(forbidden, src, f"{f} must not submit orders ({forbidden})")

    def test_14_run_imports_cleanly(self):
        import run
        self.assertTrue(hasattr(run, "_rl_shadow_cycle"))


if __name__ == "__main__":
    unittest.main()
