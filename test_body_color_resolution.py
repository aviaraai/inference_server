"""
test_body_color_resolution.py — unit tests for _resolve_disagreeing_body_colors.

The load-bearing cases are the two REAL field failures from 2026-09-07, both
produced by the same black buffalo within 35 minutes on two different handsets,
and each of which hit a DIFFERENT branch of the old 422:

    GREY 0.85 vs SPOTTED 0.51  → old: "conflicting confident readings"
    GREY 0.47 vs SPOTTED 0.42  → old: "neither is decisive"

Seven registrations were refused this way and no retake could have fixed any of
them — the classifier has no stable opinion on a dark coat. Both must now
resolve. If either of these starts raising again, body color is blocking field
registrations a second time.
"""

import os
import unittest

os.environ.setdefault("MODEL_PATH", "dummy.pt")
os.environ.setdefault("FAISS_INDEX_PATH", "dummy.index")

from main import (  # noqa: E402
    _resolve_disagreeing_body_colors,
    BYPASS_QUALITY_GATES,
)


def _reading(label, confidence):
    return {"label": label, "confidence": confidence}


class TestResolveDisagreeingBodyColors(unittest.TestCase):
    # ── The two real field failures ──────────────────────────────────────────

    def test_field_case_bare_majority_cannot_veto_a_dominant_reading(self):
        """GREY 0.85 vs SPOTTED 0.51 — the 2026-09-07 01:27 PM refusal.

        0.51 is a coin flip about where a classifier boundary fell. It must not
        outvote a reading covering 85% of the coat.
        """
        result = _resolve_disagreeing_body_colors(
            [_reading("SPOTTED", 0.51), _reading("GREY", 0.85)]
        )
        self.assertEqual(result["label"], "GREY")
        self.assertAlmostEqual(result["confidence"], 0.425)  # halved
        self.assertIn("SPOTTED", result["reason"])

    def test_field_case_two_weak_readings_store_unknown_and_proceed(self):
        """GREY 0.47 vs SPOTTED 0.42 — the 2026-09-07 01:03 PM refusal.

        Neither claims a majority of the coat, so there is nothing to store —
        but "we don't know" is what extract_body itself returns when it cannot
        read a coat, and two photos agreeing on UNKNOWN have always registered
        fine. Disagreeing weakly must not be treated as worse than that.
        """
        result = _resolve_disagreeing_body_colors(
            [_reading("GREY", 0.47), _reading("SPOTTED", 0.42)]
        )
        self.assertEqual(result["label"], "UNKNOWN")
        self.assertEqual(result["confidence"], 0.0)

    # ── The one case that must still block ───────────────────────────────────

    def test_two_strongly_confident_contradictory_readings_still_raise(self):
        """The case a retake genuinely serves: most plausibly two different
        animals. This is the ONLY path left that blocks on body color."""
        if BYPASS_QUALITY_GATES:
            self.skipTest("BYPASS_QUALITY_GATES is set; reject path unreachable")
        with self.assertRaises(Exception):
            _resolve_disagreeing_body_colors(
                [_reading("BLACK", 0.90), _reading("WHITE", 0.88)]
            )

    def test_exactly_at_the_contradiction_bar_still_raises(self):
        """0.75 is inclusive on both sides — pins the boundary so a later
        refactor cannot quietly turn it into > and reopen the gate."""
        if BYPASS_QUALITY_GATES:
            self.skipTest("BYPASS_QUALITY_GATES is set; reject path unreachable")
        with self.assertRaises(Exception):
            _resolve_disagreeing_body_colors(
                [_reading("BLACK", 0.75), _reading("BROWN", 0.75)]
            )

    def test_just_below_the_contradiction_bar_resolves(self):
        result = _resolve_disagreeing_body_colors(
            [_reading("BLACK", 0.74), _reading("BROWN", 0.74)]
        )
        self.assertEqual(result["label"], "BLACK")  # tie → max() takes the first

    # ── Unchanged behavior: exactly one reading claims a majority ────────────

    def test_single_majority_reading_is_taken_with_halved_confidence(self):
        result = _resolve_disagreeing_body_colors(
            [_reading("BROWN", 0.60), _reading("WHITE", 0.30)]
        )
        self.assertEqual(result["label"], "BROWN")
        self.assertAlmostEqual(result["confidence"], 0.30)
        self.assertTrue(result["reason"].startswith("RESOLVED_DISAGREEMENT"))

    def test_stronger_reading_wins_regardless_of_argument_order(self):
        for pair in (
            [_reading("GREY", 0.85), _reading("SPOTTED", 0.51)],
            [_reading("SPOTTED", 0.51), _reading("GREY", 0.85)],
        ):
            with self.subTest(order=[c["label"] for c in pair]):
                self.assertEqual(_resolve_disagreeing_body_colors(pair)["label"], "GREY")

    def test_accepted_reading_never_reports_full_confidence(self):
        """One of two photos failed to support the label, so the stored value
        must read as less certain than two agreeing photos would."""
        for a, b in ((0.60, 0.30), (0.85, 0.51), (0.74, 0.10)):
            with self.subTest(pair=(a, b)):
                result = _resolve_disagreeing_body_colors(
                    [_reading("BROWN", a), _reading("WHITE", b)]
                )
                self.assertLess(result["confidence"], a)


if __name__ == "__main__":
    unittest.main()
