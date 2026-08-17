"""
cctv/tests/test_panning.py — tests for the automatic panning detector
(cctv/panning.py).

The three REAL_* fixtures below are exact regression cases: the four
checkpoint values (index 0, n//2, int(n*0.75), -1 -- the only four points
detect_panning() ever reads) are copied directly from real metrics.csv runs
in cctv/runs/ (gitignored, so not read from disk here -- the values are
inlined so this test doesn't depend on those files existing). See
CLAUDE.md, "Automatic peak-vs-tracking metric selection" for how each clip
was independently visually confirmed as panning/static before being used to
pick PANNING_RATIO_THRESHOLD.
"""

import unittest

from cctv.panning import (
    MIN_FRAMES_FOR_DETECTION,
    MIN_UNIQUE_IDS_FOR_DETECTION,
    detect_panning,
)


def _series_from_checkpoints(n: int, s0: int, shalf: int, sq3: int, slast: int) -> list[int]:
    """Builds a monotonic series of length n that hits the four exact
    values detect_panning() actually reads (index 0, n//2, int(n*0.75), -1).
    What happens strictly between those checkpoints is irrelevant to the
    function under test, so simple linear interpolation is enough to make a
    valid, monotonic fixture without needing the real per-frame data."""
    half = n // 2
    q3 = int(n * 0.75)

    def lerp_segment(start_idx, end_idx, start_val, end_val):
        span = end_idx - start_idx
        return [
            round(start_val + (end_val - start_val) * (i - start_idx) / span) if span > 0 else start_val
            for i in range(start_idx, end_idx)
        ]

    series = (
        lerp_segment(0, half, s0, shalf)
        + lerp_segment(half, q3, shalf, sq3)
        + lerp_segment(q3, n - 1, sq3, slast)
        + [slast]
    )
    assert len(series) == n, f"fixture builder bug: got {len(series)}, want {n}"
    return series


class TestDetectPanning(unittest.TestCase):
    # ── real, visually-confirmed cases ──────────────────────────────────
    def test_real_panning_clip(self):
        # max_cattle_in_frame=22 vs unique_tracked_cattle=62 on the wire;
        # camera motion confirmed by inspecting frames from 4 different
        # points in the clip (each shows a visibly different part of the
        # shed). Ratio comes out to 2.185 on the real data.
        series = _series_from_checkpoints(480, 6, 30, 51, 77)
        is_panning, ratio = detect_panning(series)
        self.assertTrue(is_panning)
        self.assertAlmostEqual(ratio, 2.185, places=2)

    def test_real_static_clip(self):
        # Fixed close-up on a feeding trough -- same background
        # railing/trough structure from the first sampled frame to the
        # last. Ratio comes out to 0.506 on the real data.
        series = _series_from_checkpoints(320, 4, 16, 22, 25)
        is_panning, ratio = detect_panning(series)
        self.assertFalse(is_panning)
        self.assertAlmostEqual(ratio, 0.506, places=2)

    def test_real_ambiguous_field_clip(self):
        # Open-pasture clip, peak_cattle_in_frame only 6 but 33+ stable IDs
        # minted -- visual inspection showed the camera position genuinely
        # shifts between the first and last sampled frame (different part
        # of the field in view), so classifying this as panning (ratio
        # 1.609) is the visually-supported call, not a misfire.
        series = _series_from_checkpoints(366, 5, 15, 25, 33)
        is_panning, ratio = detect_panning(series)
        self.assertTrue(is_panning)
        self.assertAlmostEqual(ratio, 1.609, places=2)

    # ── synthetic edge cases ─────────────────────────────────────────────
    def test_clean_plateau_is_not_panning(self):
        # Climbs fast early, then flat -- the textbook static-camera shape.
        series = [0, 2, 4, 6, 8, 10] + [10] * 94  # n=100
        is_panning, ratio = detect_panning(series)
        self.assertFalse(is_panning)
        self.assertEqual(ratio, 0.0)

    def test_clean_linear_climb_is_panning(self):
        # Constant rate throughout -- ratio == 1.0, right at the threshold,
        # and the threshold is defined as inclusive (>=), so this must
        # count as panning: a camera revealing new ground at a steady,
        # undiminished rate for the whole clip is exactly the case this
        # exists to catch, not a coin-flip to leave on the "safe" side.
        series = list(range(100))  # n=100, rate exactly 1/frame throughout
        is_panning, ratio = detect_panning(series)
        self.assertTrue(is_panning)
        self.assertAlmostEqual(ratio, 1.0, places=6)

    def test_too_few_frames_defaults_to_not_panning(self):
        series = list(range(MIN_FRAMES_FOR_DETECTION - 1))
        is_panning, ratio = detect_panning(series)
        self.assertFalse(is_panning)
        self.assertEqual(ratio, 0.0)

    def test_too_few_unique_ids_defaults_to_not_panning(self):
        # Plenty of frames, but the herd is tiny (e.g. 2 animals total) --
        # not enough signal to say anything about coverage.
        series = [0] * 10 + [1] * 10 + [MIN_UNIQUE_IDS_FOR_DETECTION - 1] * 30
        is_panning, ratio = detect_panning(series)
        self.assertFalse(is_panning)
        self.assertEqual(ratio, 0.0)

    def test_flat_start_then_late_growth_is_panning(self):
        # Nothing minted in the first half (first_half_rate == 0), but the
        # back half genuinely discovers new individuals from a standing
        # start -- e.g. a camera that starts pointed at an empty stretch of
        # trough and only later pans onto the herd.
        series = [1] * 50 + list(range(1, 51))  # n=100
        is_panning, ratio = detect_panning(series)
        self.assertTrue(is_panning)
        self.assertEqual(ratio, float("inf"))

    def test_flat_throughout_is_not_panning(self):
        series = [1] * 100
        # Below MIN_UNIQUE_IDS_FOR_DETECTION (1 < 4), so this is actually
        # exercising that guard rather than the flat-rate branch -- still
        # correctly "not panning" either way.
        is_panning, ratio = detect_panning(series)
        self.assertFalse(is_panning)


if __name__ == "__main__":
    unittest.main()
