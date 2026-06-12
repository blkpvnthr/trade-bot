import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import runtime_guard_report as rgr


class RuntimeGuardReportTests(unittest.TestCase):
    def test_defaults_to_paper_safe_when_env_omits_flag(self):
        status = rgr._resolve_paper_runtime({})
        self.assertIsNone(status["apca_paper"])
        self.assertTrue(status["paper_only_confirmed"])
        self.assertEqual(status["paper_mode_source"], "run.py_default")

    def test_explicit_false_alias_disables_paper_only_confirmation(self):
        status = rgr._resolve_paper_runtime({"APCA_PAPER": "false"})
        self.assertEqual(status["apca_paper"], "false")
        self.assertFalse(status["paper_only_confirmed"])
        self.assertEqual(status["paper_mode_source"], "APCA_PAPER")

    def test_alpaca_paper_only_alias_is_honored(self):
        status = rgr._resolve_paper_runtime({"ALPACA_PAPER_ONLY": "true"})
        self.assertIsNone(status["apca_paper"])
        self.assertTrue(status["paper_only_confirmed"])
        self.assertEqual(status["paper_mode_source"], "ALPACA_PAPER_ONLY")


if __name__ == "__main__":
    unittest.main(verbosity=2)
