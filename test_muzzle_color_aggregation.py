"""
test_muzzle_color_aggregation.py — unit tests for _aggregate_muzzle_color,
extracted from /register's original inline logic so /search's multi-photo
query can share the same policy (see main.py's docstring on the function
and CLAUDE.md's muzzle color aggregation note).

allow_reject=True must reproduce /register's exact original behavior
(regression-tested here); allow_reject=False is the new /search path that
never raises.
"""

import os
import unittest

os.environ.setdefault("MODEL_PATH", "dummy.pt")
os.environ.setdefault("FAISS_INDEX_PATH", "dummy.index")

from main import _aggregate_muzzle_color, BYPASS_QUALITY_GATES  # noqa: E402


def _reading(label, confidence):
    return {"label": label, "confidence": confidence}


class TestAggregateMuzzleColor(unittest.TestCase):
    def test_majority_of_two_uses_mean_confidence(self):
        readings = [_reading("BLACK", 0.8), _reading("BLACK", 0.6), _reading("PINK", 0.9)]
        result = _aggregate_muzzle_color(readings, allow_reject=True)
        self.assertEqual(result["label"], "BLACK")
        self.assertAlmostEqual(result["confidence"], 0.7)  # mean of 0.8, 0.6

    def test_unanimous_three_uses_mean_of_all(self):
        readings = [_reading("PINK", 0.5), _reading("PINK", 0.7), _reading("PINK", 0.9)]
        result = _aggregate_muzzle_color(readings, allow_reject=True)
        self.assertEqual(result["label"], "PINK")
        self.assertAlmostEqual(result["confidence"], 0.7)

    def test_no_majority_allow_reject_true_raises_unless_bypassed(self):
        readings = [_reading("BLACK", 0.5), _reading("PINK", 0.6), _reading("MIXED", 0.4)]
        if BYPASS_QUALITY_GATES:
            self.skipTest("BYPASS_QUALITY_GATES is set in this environment; reject path not reachable")
        with self.assertRaises(Exception):
            _aggregate_muzzle_color(readings, allow_reject=True)

    def test_no_majority_allow_reject_false_never_raises(self):
        """The whole point of the /search variant: same no-majority input
        that raises for /register must instead fall back cleanly here."""
        readings = [_reading("BLACK", 0.5), _reading("PINK", 0.9), _reading("MIXED", 0.4)]
        result = _aggregate_muzzle_color(readings, allow_reject=False)
        self.assertEqual(result["label"], "PINK")  # best confidence
        self.assertAlmostEqual(result["confidence"], 0.45)  # halved, same as register's own bypass path

    def test_two_photo_query_majority(self):
        """/search may send only 2 muzzle photos in some paths -- majority
        vote still works for N=2 when they agree."""
        readings = [_reading("BROWN", 0.7), _reading("BROWN", 0.5)]
        result = _aggregate_muzzle_color(readings, allow_reject=False)
        self.assertEqual(result["label"], "BROWN")
        self.assertAlmostEqual(result["confidence"], 0.6)

    def test_two_photo_query_disagreement_falls_back(self):
        readings = [_reading("BROWN", 0.7), _reading("BLACK", 0.9)]
        result = _aggregate_muzzle_color(readings, allow_reject=False)
        self.assertEqual(result["label"], "BLACK")
        self.assertAlmostEqual(result["confidence"], 0.45)


if __name__ == "__main__":
    unittest.main()
