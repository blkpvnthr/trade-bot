"""Tests for the FinBERT scorer — must be fail-safe (neutral, never raise) when the model/lib
is absent, and only assert real scoring when the model is actually installed (offline CI-safe)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_sources import finbert_scorer as fs


class FinbertFailSafeTests(unittest.TestCase):
    """These must hold regardless of whether transformers/the model are present."""

    def test_neutral_result_shape(self):
        self.assertEqual(set(fs.NEUTRAL_RESULT), {"score", "label", "probs"})
        self.assertIsNone(fs.NEUTRAL_RESULT["score"])
        self.assertEqual(fs.NEUTRAL_RESULT["label"], "Neutral")

    def test_blank_text_is_neutral(self):
        for t in ("", "   ", None):
            r = fs.score_text(t)  # type: ignore[arg-type]
            self.assertIsNone(r["score"])
            self.assertEqual(r["label"], "Neutral")

    def test_batch_returns_one_per_input_and_never_raises(self):
        texts = ["good earnings beat", "", "guidance cut"]
        out = fs.score_batch(texts)
        self.assertEqual(len(out), len(texts))
        for r in out:
            self.assertIn("score", r)
            self.assertIn("label", r)
        # the blank input is always neutral
        self.assertIsNone(out[1]["score"])

    def test_lib_absent_forces_neutral(self):
        # Simulate transformers/torch missing -> scorer degrades to neutral, no crash.
        orig_lib, orig_model = fs._FINBERT_LIB, fs._MODEL
        try:
            fs._FINBERT_LIB = False
            fs._MODEL = None
            r = fs.score_text("Company beats earnings and raises guidance")
            self.assertIsNone(r["score"])
            self.assertEqual(r["label"], "Neutral")
            self.assertFalse(fs.is_available())
        finally:
            fs._FINBERT_LIB, fs._MODEL = orig_lib, orig_model

    def test_canon_label(self):
        self.assertEqual(fs._canon_label("positive"), "Positive")
        self.assertEqual(fs._canon_label("NEGATIVE"), "Negative")
        self.assertEqual(fs._canon_label("neutral"), "Neutral")
        self.assertEqual(fs._canon_label("whatever"), "Neutral")


@unittest.skipUnless(fs.is_available(), "FinBERT model not installed (run bootstrap_finbert.py --download)")
class FinbertRealScoringTests(unittest.TestCase):
    """Only run when the model is actually present."""

    def test_positive_headline_scores_positive(self):
        r = fs.score_text("Company beats earnings expectations and raises full-year guidance")
        self.assertIsNotNone(r["score"])
        self.assertGreater(r["score"], 0.0)
        self.assertEqual(r["label"], "Positive")

    def test_negative_headline_scores_negative(self):
        r = fs.score_text("Firm slashes outlook amid weak demand and mounting losses")
        self.assertIsNotNone(r["score"])
        self.assertLess(r["score"], 0.0)
        self.assertEqual(r["label"], "Negative")

    def test_score_in_range(self):
        for r in fs.score_batch(["great quarter", "terrible miss", "routine filing"]):
            if r["score"] is not None:
                self.assertGreaterEqual(r["score"], -1.0)
                self.assertLessEqual(r["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
