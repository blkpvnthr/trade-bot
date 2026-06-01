"""
Tests for the RL enhancements adapted from Yasin & Gill (2024),
"Reinforcement Learning Framework for Quantitative Trading":
  * input normalization of observations (§8)
  * trade-activity overfit guard in the deployment gate (§11.5)
  * paper-aligned configurable hyperparameters (§11.3)
All offline.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rl_common as rc


class NormalizationTests(unittest.TestCase):
    def test_tanh_bounds_and_length(self):
        vec = [0.0, 0.5, 3.0, -2.0, 100.0, -100.0]
        out = rc.normalize_obs(vec, method="tanh")
        self.assertEqual(len(out), len(vec))
        self.assertTrue(all(-1.0 <= x <= 1.0 for x in out))
        self.assertEqual(round(out[0], 6), 0.0)
        self.assertGreater(out[2], 0.99)      # 3.0 → ~0.995
        self.assertLess(out[3], 0.0)

    def test_clip_and_none_modes(self):
        self.assertEqual(rc.normalize_obs([10.0], method="clip"), [1.0])     # clip→/OBS_CLIP capped at 1
        self.assertEqual(rc.normalize_obs([0.3], method="none"), [0.3])

    def test_never_raises_on_bad_input(self):
        self.assertEqual(len(rc.normalize_obs([None, "x", 1.0])), 3)

    def test_envs_emit_normalized_obs(self):
        import rl_env_alpha_entry as A, rl_env_capital_allocation as D
        for cls in (A.AlphaEntryEnv, D.CapitalAllocationEnv):
            obs, _ = cls().reset()
            self.assertTrue(all(-1.0001 <= float(x) <= 1.0001 for x in obs),
                            f"{cls.__name__} obs must be normalized")


class ActivityMetricTests(unittest.TestCase):
    def test_active_trades_counts_nonzero(self):
        m = rc.compute_metrics([2.0, 0.0, 0.0, -1.0, 3.0, 0.0])
        self.assertEqual(m["active_trades"], 3)
        self.assertAlmostEqual(m["activity_ratio"], 0.5, places=4)
        self.assertEqual(rc.compute_metrics([])["active_trades"], 0)


class ActivityGuardTests(unittest.TestCase):
    def test_low_activity_downgrades_positive_rec(self):
        # Looks great (expectancy + drawdown improved) but only 3 actual trades → overfit.
        rec, notes = rc.deployment_recommendation(
            {"expectancy": 0.3, "max_drawdown": 2.0, "win_rate": 0.5},
            {"expectancy": 0.9, "max_drawdown": 1.0, "win_rate": 1.0,
             "active_trades": 3, "activity_ratio": 0.01}, 500)
        self.assertEqual(rec, "shadow_mode_only")
        self.assertTrue(notes["overfit_warning"])
        self.assertEqual(notes["pre_activity_recommendation"], "paper_mode_only")

    def test_healthy_activity_keeps_paper_mode(self):
        rec, _ = rc.deployment_recommendation(
            {"expectancy": 0.3, "max_drawdown": 2.0, "win_rate": 0.5},
            {"expectancy": 0.6, "max_drawdown": 1.5, "win_rate": 0.55,
             "active_trades": 4000, "activity_ratio": 0.9}, 500)
        self.assertEqual(rec, "paper_mode_only")

    def test_absent_activity_metric_is_backward_compatible(self):
        # Hand-built dicts without active_trades must behave exactly as before.
        rec, _ = rc.deployment_recommendation(
            {"expectancy": 0.5, "max_drawdown": 1.0, "win_rate": 0.55},
            {"expectancy": 0.2, "max_drawdown": 1.0, "win_rate": 0.55}, 500)
        self.assertEqual(rec, "do_not_deploy")

    def test_guard_does_not_rescue_negative_rec(self):
        # do_not_deploy stays do_not_deploy regardless of activity.
        rec, _ = rc.deployment_recommendation(
            {"expectancy": 0.5, "max_drawdown": 1.0, "win_rate": 0.5},
            {"expectancy": 0.2, "max_drawdown": 1.0, "win_rate": 0.5,
             "active_trades": 2, "activity_ratio": 0.0}, 500)
        self.assertEqual(rec, "do_not_deploy")


class HyperparameterTests(unittest.TestCase):
    def test_paper_aligned_hyperparams_configurable(self):
        import train_rl_agents as tr
        self.assertEqual(tr.RL_GAMMA, 0.99)                 # paper §11.3 Table 1
        for attr in ("RL_LEARNING_RATE", "RL_BATCH_SIZE", "RL_N_STEPS", "RL_N_EPOCHS"):
            self.assertTrue(hasattr(tr, attr))


if __name__ == "__main__":
    unittest.main()
