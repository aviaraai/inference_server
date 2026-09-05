"""
pipeline/tests/test_keypoint_color.py — pure-function tests for
pipeline/keypoint_color.py. No model weights needed: _forehead_direction and
_sample_patch are tested directly with synthetic keypoints/patches, same
approach pose_model/tests/test_geometry.py uses for compute_geometry (feed a
synthetic (8,3) array, no trained model required). combine_keypoint_color and
the end-to-end UNKNOWN cascade are covered with synthetic reading dicts /
monkeypatched crop_cattle+predict_keypoints, matching this repo's existing
unittest style (see wildlife/color/tests/test_color.py, cctv/tests/test_panning.py).
"""

import unittest
from unittest.mock import patch

import numpy as np

from pipeline.geometry import CREST, L_EYE, MUZZLE, R_EYE
from pipeline.keypoint_color import (
    KeypointColorStatus,
    _forehead_direction,
    _sample_patch,
    combine_keypoint_color,
    extract_keypoint_forehead_color,
)


def _kpts(overrides: dict[int, tuple[float, float, float]]) -> np.ndarray:
    """(8,3) keypoint array, all-zero/zero-confidence except the given
    {index: (x, y, conf)} overrides."""
    kp = np.zeros((8, 3), dtype=float)
    for idx, (x, y, conf) in overrides.items():
        kp[idx] = [x, y, conf]
    return kp


def _noisy_bgr(base_bgr: tuple[int, int, int], shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    """A synthetic BGR image around `base_bgr` with real per-pixel noise, not
    a flat constant block. check_roi_quality's contrast gate
    (MIN_CONTRAST=10.0) rejects a truly uniform patch -- zero standard
    deviation never happens on a real photo -- so a flat np.full()/np.zeros()
    patch fails quality gating before classification ever runs. Same
    reasoning wildlife/color/tests/test_color.py's _field_scene applies for
    its synthetic scenes.

    The SAME noise value is added to all three channels at each pixel
    (correlated, not three independent draws): independent per-channel noise
    also perturbs a*/b*, and at a std high enough to clear the contrast gate
    it pushes chroma over NEUTRAL_THRESHOLD (9.0) often enough to misclassify
    an intended-achromatic patch as chromatic/UNKNOWN. Real lighting
    variation across a uniformly-colored surface overwhelmingly affects
    luminance, not hue, so this is the more realistic noise model anyway."""
    rng = np.random.RandomState(seed)
    h, w = shape
    noise = rng.normal(0, 20, size=(h, w, 1))
    img = np.clip(np.array(base_bgr, dtype=float) + noise, 0, 255)
    return img.astype(np.uint8)


class TestForeheadDirection(unittest.TestCase):
    """R_EYE=(10,10), L_EYE=(20,10) -> eye_mid=(15,10), ruler=10, eye-line
    horizontal. "Up" in image coordinates is smaller y."""

    def setUp(self):
        self.xy = np.zeros((8, 2))
        self.xy[R_EYE] = [10, 10]
        self.xy[L_EYE] = [20, 10]
        self.eye_mid = np.array([15.0, 10.0])
        self.ruler = 10.0

    def test_muzzle_below_eyes_points_forehead_up(self):
        conf = np.zeros(8)
        conf[MUZZLE] = 1.0
        self.xy[MUZZLE] = [15, 20]  # below the eye-line
        direction = _forehead_direction(self.xy, conf, self.eye_mid, self.ruler)
        self.assertLess(direction[1], 0)  # points toward smaller y = up

    def test_crest_above_eyes_points_forehead_toward_crest(self):
        conf = np.zeros(8)
        conf[CREST] = 1.0
        self.xy[CREST] = [15, 0]  # above the eye-line
        direction = _forehead_direction(self.xy, conf, self.eye_mid, self.ruler)
        self.assertLess(direction[1], 0)

    def test_no_reliable_anchor_falls_back_to_image_up(self):
        conf = np.zeros(8)  # neither muzzle nor crest confident
        direction = _forehead_direction(self.xy, conf, self.eye_mid, self.ruler)
        self.assertLess(direction[1], 0)


class TestSamplePatch(unittest.TestCase):
    def test_dark_noisy_patch_classifies_black(self):
        img = _noisy_bgr((35, 35, 35), (200, 200))
        reasons: dict[str, str] = {}
        result = _sample_patch(img, np.array([100.0, 100.0]), half_side=30.0, reason_key="x", reasons=reasons)
        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "BLACK")
        self.assertNotIn("x", reasons)

    def test_too_small_patch_rejected_not_silently_sampled(self):
        img = _noisy_bgr((35, 35, 35), (200, 200))
        reasons: dict[str, str] = {}
        # half_side chosen so the resulting side is well under MIN_PATCH_SIDE_PX
        result = _sample_patch(img, np.array([100.0, 100.0]), half_side=2.0, reason_key="tiny", reasons=reasons)
        self.assertIsNone(result)
        self.assertIn("tiny", reasons)
        self.assertIn("too small", reasons["tiny"])

    def test_patch_clipped_off_frame_edge_rejected(self):
        img = _noisy_bgr((35, 35, 35), (40, 40))
        reasons: dict[str, str] = {}
        # Center well outside the frame entirely: clipping collapses both
        # dimensions to zero (not just "small").
        result = _sample_patch(img, np.array([-100.0, -100.0]), half_side=10.0, reason_key="corner", reasons=reasons)
        self.assertIsNone(result)
        self.assertIn("corner", reasons)


class TestCombineKeypointColor(unittest.TestCase):
    def test_all_unknown_stays_unknown(self):
        readings = [
            {"status": KeypointColorStatus.UNKNOWN, "reasons": {"status": "no face"}},
            {"status": KeypointColorStatus.UNKNOWN, "reasons": {"status": "no face"}},
        ]
        combined = combine_keypoint_color(readings)
        self.assertEqual(combined["status"], KeypointColorStatus.UNKNOWN)
        self.assertEqual(combined["sources_usable"], 0)

    def test_one_usable_one_unknown_uses_the_usable_one(self):
        readings = [
            {"status": KeypointColorStatus.UNKNOWN, "reasons": {"status": "no face"}},
            {
                "status": KeypointColorStatus.OK,
                "label": "BROWN",
                "confidence": 0.8,
                "inter_eye_distance_px": 50.0,
                "patches": {"forehead_low": {}, "forehead_high": {}, "crest": None},
            },
        ]
        combined = combine_keypoint_color(readings)
        self.assertEqual(combined["status"], KeypointColorStatus.OK)
        self.assertEqual(combined["label"], "BROWN")
        self.assertEqual(combined["sources_usable"], 1)
        self.assertIsNotNone(combined["reason"])  # not a full 2/2 reading

    def test_two_usable_picks_higher_confidence(self):
        readings = [
            {
                "status": KeypointColorStatus.PARTIAL,
                "label": "BLACK",
                "confidence": 0.4,
                "inter_eye_distance_px": 40.0,
                "patches": {"forehead_low": {}, "forehead_high": None, "crest": None},
            },
            {
                "status": KeypointColorStatus.OK,
                "label": "BROWN",
                "confidence": 0.9,
                "inter_eye_distance_px": 60.0,
                "patches": {"forehead_low": {}, "forehead_high": {}, "crest": None},
            },
        ]
        combined = combine_keypoint_color(readings)
        self.assertEqual(combined["label"], "BROWN")
        self.assertEqual(combined["confidence"], 0.9)
        self.assertEqual(combined["inter_eye_distance_px"], 50.0)  # averaged
        self.assertIsNone(combined["reason"])  # full 2/2


class TestExtractKeypointForeheadColorEndToEnd(unittest.TestCase):
    """crop_cattle and predict_keypoints monkeypatched -- no real model
    weights needed, same reasoning as the unit tests above. Patches the
    names at their point of use: predict_keypoints/pose_available are bound
    into pipeline.keypoint_color's namespace at import time (a plain `from
    pipeline.pose import ...`), so they must be patched there; crop_cattle is
    imported fresh inside the function body on every call (a deliberate
    local import -- see the module docstring), so patching its source
    (pipeline.yolo_crop.crop_cattle) is what actually takes effect.
    """

    def test_missing_eye_keypoints_returns_unknown_not_a_fallback_reading(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        kpts = _kpts({R_EYE: (40, 40, 0.9)})  # L_EYE missing entirely
        with patch("pipeline.keypoint_color.pose_available", return_value=True), \
             patch("pipeline.keypoint_color.predict_keypoints", return_value=kpts), \
             patch("pipeline.yolo_crop.crop_cattle", return_value=(img, "OK", 0.9)):
            result = extract_keypoint_forehead_color(img)
        self.assertEqual(result["status"], KeypointColorStatus.UNKNOWN)
        self.assertEqual(result["label"], "UNKNOWN")
        self.assertIsNone(result["inter_eye_distance_px"])

    def test_low_confidence_eye_returns_unknown(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        kpts = _kpts({R_EYE: (40, 40, 0.2), L_EYE: (60, 40, 0.9)})  # right eye below threshold
        with patch("pipeline.keypoint_color.pose_available", return_value=True), \
             patch("pipeline.keypoint_color.predict_keypoints", return_value=kpts), \
             patch("pipeline.yolo_crop.crop_cattle", return_value=(img, "OK", 0.9)):
            result = extract_keypoint_forehead_color(img)
        self.assertEqual(result["status"], KeypointColorStatus.UNKNOWN)

    def test_confident_eyes_on_real_sized_crop_produces_ok_or_partial(self):
        img = _noisy_bgr((30, 30, 30), (400, 400))  # dark, textured, big enough for patches
        kpts = _kpts({
            R_EYE: (150, 200, 0.9),
            L_EYE: (250, 200, 0.9),
            MUZZLE: (200, 280, 0.9),
        })
        with patch("pipeline.keypoint_color.pose_available", return_value=True), \
             patch("pipeline.keypoint_color.predict_keypoints", return_value=kpts), \
             patch("pipeline.yolo_crop.crop_cattle", return_value=(img, "OK", 0.9)):
            result = extract_keypoint_forehead_color(img)
        self.assertIn(result["status"], (KeypointColorStatus.OK, KeypointColorStatus.PARTIAL))
        self.assertIsNotNone(result["inter_eye_distance_px"])
        self.assertEqual(result["inter_eye_distance_px"], 100.0)
        self.assertIsNotNone(result["patches"]["forehead_low"])


if __name__ == "__main__":
    unittest.main()
