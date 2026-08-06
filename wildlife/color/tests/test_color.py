"""
test_color.py — Test suite for color extraction. Computes Confusion Matrix, Accuracy, Precision, Recall, and F1.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

# Adjust python path to allow running script directly or as module
_COLOR_DIR = Path(__file__).resolve().parent.parent
if str(_COLOR_DIR) not in sys.path:
    sys.path.insert(0, str(_COLOR_DIR))

try:
    import color_constants as C
    from body_color import classify_body_color
    from muzzle_color import classify_muzzle_color
except ImportError:
    from . import color_constants as C
    from .body_color import classify_body_color
    from .muzzle_color import classify_muzzle_color


def _field_scene(coat_bgr, bg_bgr, coat_frac=0.40, patch_bgr=None, size=600, seed=0):
    """Build a synthetic photo shaped like a real field capture.

    The coat deliberately occupies a MINORITY of the frame against a
    contrasting background. This matters: a naive test image where the coat
    fills most of the frame passes even against the old buggy classifier and
    therefore proves nothing. The bugs these tests cover only reproduce when
    the animal is a minority of the frame, which is the normal case in real
    field photos (see get_body_roi / the center-crop fallback).
    """
    rs = np.random.RandomState(seed)
    img = np.zeros((size, size, 3), np.uint8)
    img[:, :] = bg_bgr
    side = int(size * coat_frac)
    off = (size - side) // 2
    img[off:off + side, off:off + side] = coat_bgr
    if patch_bgr is not None:
        # A patch ON the animal (upper half of the coat block), i.e. a real
        # two-tone coat rather than a differently-colored background.
        img[off:off + side // 2, off:off + side] = patch_bgr
    return np.clip(img.astype(np.int16) + rs.randint(-12, 12, (size, size, 3)), 0, 255).astype(np.uint8)


class TestColorContract(unittest.TestCase):
    """Sanity checks to ensure classification functions conform to the API contract."""

    def test_body_color_contract(self):
        # Create a gray synthetic image with noise to pass quality checks
        np.random.seed(42)
        noise = np.random.randint(-30, 30, size=(400, 400, 3))
        img = (np.ones((400, 400, 3), dtype=np.uint8) * 128).astype(np.int16)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

        result = classify_body_color(img)

        # Check fields
        self.assertIn("label", result)
        self.assertIn("confidence", result)
        self.assertIn("method", result)
        self.assertIn("reason", result)
        self.assertIn("median_lab", result)
        self.assertIn("dominant_lab", result)

        self.assertEqual(result["method"], "LAB_KMEANS_V2")
        self.assertEqual(result["label"], C.LABEL_GREY)
        self.assertEqual(result["reason"], "OK")


class TestBodyColorRegressions(unittest.TestCase):
    """Regressions for the "white/brown animal classified BLACK" field bug.

    Root cause was two independent defects in the dominant-color extractor:

    1. It returned a HISTOGRAM BIN CENTER as the dominant color rather than a
       real measurement. The neutral point sat on a bin EDGE, so every
       near-neutral coat landed in the bin centered at (+8, +8) and reported
       an identical fabricated chroma of 11.3 whatever its true color. That
       also put chroma permanently below NEUTRAL_THRESHOLD (15), making BROWN
       unreachable for any animal.
    2. Peak-finding ran over a*/b* only, ignoring L*. A black coat and a
       white coat are both achromatic, so they shared a bin and had their
       lightness values median-ed together — and that merged lightness is
       what decided BLACK vs WHITE.

    EVERY test in this class fails against the pre-fix implementation.
    """

    def test_dominant_color_is_a_real_measurement_not_a_bin_center(self):
        """Different-colored coats must not report identical chromaticity."""
        white = classify_body_color(_field_scene((235, 235, 235), (45, 45, 45)))
        brown = classify_body_color(_field_scene((40, 80, 145), (128, 128, 128)))

        white_ab = tuple(white["dominant_lab"][1:])
        brown_ab = tuple(brown["dominant_lab"][1:])
        self.assertNotEqual(
            white_ab, brown_ab,
            "white and brown coats reported identical chromaticity — the "
            "dominant color is a fabricated bin center, not a measurement",
        )
        # The old code pinned every near-neutral coat to exactly (+8, +8).
        self.assertNotEqual(white_ab, (8.0, 8.0))

    def test_white_coat_on_dark_background_is_not_black(self):
        """The exact reported failure: a white animal came back BLACK."""
        result = classify_body_color(_field_scene((235, 235, 235), (45, 45, 45)))
        self.assertEqual(result["label"], C.LABEL_WHITE)

    def test_black_coat_on_bright_background_is_not_white(self):
        """The same defect in the opposite direction."""
        result = classify_body_color(_field_scene((30, 30, 30), (210, 210, 210)))
        self.assertEqual(result["label"], C.LABEL_BLACK)

    def test_brown_is_reachable(self):
        """BROWN was unreachable: fabricated chroma never exceeded NEUTRAL_THRESHOLD."""
        result = classify_body_color(_field_scene((40, 80, 145), (128, 128, 128)))
        self.assertEqual(result["label"], C.LABEL_BROWN)

    def test_coat_color_survives_a_colored_background(self):
        """Foliage/soil behind the animal must not become the reported coat color."""
        result = classify_body_color(_field_scene((235, 235, 235), (60, 110, 50)))
        self.assertEqual(result["label"], C.LABEL_WHITE)

    def test_genuinely_two_tone_coat_is_still_spotted(self):
        """Control: the fix must not work by simply disabling SPOTTED.

        A patch ON the animal (not a differently-colored background) must
        still classify as SPOTTED. The old code got this backwards too,
        reporting a solid color for a genuinely two-tone coat.
        """
        result = classify_body_color(
            _field_scene((240, 240, 240), (70, 70, 70), patch_bgr=(25, 25, 25))
        )
        self.assertEqual(result["label"], C.LABEL_SPOTTED)

    def test_classification_is_deterministic(self):
        """/register rejects (422) when two front photos disagree on body color.

        The classifier clusters pixels, so it must be seeded — an unseeded
        RNG would make the same animal fail registration at random.
        """
        img = _field_scene((235, 235, 235), (45, 45, 45), seed=7)
        results = [classify_body_color(img) for _ in range(5)]
        self.assertEqual(len({r["label"] for r in results}), 1)
        self.assertEqual(len({tuple(r["dominant_lab"]) for r in results}), 1)
        self.assertEqual(len({r["confidence"] for r in results}), 1)


class TestMuzzleColorRegressions(unittest.TestCase):
    """Regressions for muzzle color.

    Two defects, both distinct from the body-color bugs:

    1. The ROI was never localized on the muzzle. main.py passes
       crop_cattle()'s WHOLE-ANIMAL box to extract_muzzle(), and the old
       get_muzzle_roi() took a fixed center crop of it — i.e. the animal's
       neck/chest. On real photos of a brown-hided cow this reported PINK for
       an obviously black muzzle, because it was measuring coat.
    2. _classify_muzzle_lab() returned MIXED as its catch-all for a SINGLE
       color sample. MIXED describes a muzzle carrying two skin colors, which
       is a property of the whole muzzle — one cluster is one color. The
       phantom "MIXED color" then outvoted the real reading.
    """

    def test_single_sample_is_never_classified_mixed(self):
        """MIXED is an aggregate outcome and must never label one color sample."""
        from muzzle_color import _classify_muzzle_lab

        # Sweep the LAB space a real muzzle can occupy.
        for l in range(0, 101, 5):
            for a in range(-20, 41, 5):
                for b in range(-20, 41, 5):
                    self.assertNotEqual(
                        _classify_muzzle_lab([float(l), float(a), float(b)]),
                        C.LABEL_MIXED,
                        f"single sample [{l},{a},{b}] classified MIXED",
                    )

    def test_dark_muzzle_with_lighter_lip_is_black_not_mixed(self):
        """A black nose pad whose lower lip is lighter is still a BLACK muzzle."""
        rs = np.random.RandomState(3)
        img = np.zeros((300, 300, 3), np.uint8)
        img[:, :] = (38, 36, 40)            # dark nose pad
        img[220:, :] = (120, 110, 135)      # lighter lower lip
        img = np.clip(img.astype(np.int16) + rs.randint(-14, 14, (300, 300, 3)), 0, 255).astype(np.uint8)

        self.assertEqual(classify_muzzle_color(img)["label"], C.LABEL_BLACK)

    def test_genuinely_two_tone_muzzle_is_still_mixed(self):
        """Control: MIXED must remain reachable for a real two-color muzzle."""
        rs = np.random.RandomState(4)
        img = np.zeros((300, 300, 3), np.uint8)
        img[:150, :] = (35, 33, 37)         # black half
        img[150:, :] = (150, 140, 225)      # pink half
        img = np.clip(img.astype(np.int16) + rs.randint(-14, 14, (300, 300, 3)), 0, 255).astype(np.uint8)

        self.assertEqual(classify_muzzle_color(img)["label"], C.LABEL_MIXED)

    def test_muzzle_classification_is_deterministic(self):
        """Same reasoning as the body-color determinism test."""
        rs = np.random.RandomState(5)
        img = np.clip(
            np.full((300, 300, 3), 40, np.int16) + rs.randint(-14, 14, (300, 300, 3)), 0, 255
        ).astype(np.uint8)
        results = [classify_muzzle_color(img) for _ in range(5)]
        self.assertEqual(len({r["label"] for r in results}), 1)
        self.assertEqual(len({tuple(r["dominant_lab"]) for r in results}), 1)


class TestColorQualityAndMuzzle(unittest.TestCase):
    """Quality-gate and muzzle contract checks."""

    def test_body_color_quality_gate(self):
        # Create a tiny 50x50 image which should fail size check
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        result = classify_body_color(img)

        self.assertEqual(result["label"], C.LABEL_UNKNOWN)
        self.assertEqual(result["confidence"], 0.0)
        self.assertTrue(result["reason"].startswith("LOW_QUALITY"))

    def test_muzzle_color_contract(self):
        # Create a pink-ish synthetic image with noise to pass quality checks
        np.random.seed(42)
        noise = np.random.randint(-30, 30, size=(300, 300, 3))
        img = (np.ones((300, 300, 3), dtype=np.uint8) * 128).astype(np.int16)
        # Shift channels to be pinkish (high red/blue, lower green)
        img[:, :, 0] = 160  # Blue
        img[:, :, 1] = 130  # Green
        img[:, :, 2] = 230  # Red
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        result = classify_muzzle_color(img)

        self.assertIn("label", result)
        self.assertIn("confidence", result)
        self.assertEqual(result["label"], C.LABEL_PINK)


def run_metrics_evaluation(data_dir: Path, is_muzzle: bool):
    """Run color classification on folders of labeled images and print metrics."""
    # class_name -> list of (predicted_class)
    labels = C.MUZZLE_COLORS if is_muzzle else C.BODY_COLORS
    classify_fn = classify_muzzle_color if is_muzzle else classify_body_color

    # Ground truth vs predicted counts
    # gt_label -> pred_label -> count
    matrix = defaultdict(lambda: defaultdict(int))
    total_samples = 0
    correct_samples = 0

    print(f"Running metrics evaluation on: {data_dir.name} (muzzle={is_muzzle})")

    for class_folder in data_dir.iterdir():
        if not class_folder.is_dir():
            continue
        gt_label = class_folder.name.upper()
        if gt_label not in labels:
            print(f"[WARNING] Skipping folder with unrecognized label: {gt_label}")
            continue

        for img_path in class_folder.glob("*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Run classifier
            res = classify_fn(img)
            pred_label = res["label"]

            matrix[gt_label][pred_label] += 1
            total_samples += 1
            if pred_label == gt_label:
                correct_samples += 1

    if total_samples == 0:
        print("[ERROR] No images processed. Make sure labeled directories match target classes.")
        return

    # Print Confusion Matrix
    print("\n" + "=" * 65)
    print("CONFUSION MATRIX")
    print("=" * 65)
    header = f"{'Actual \\ Pred':<15} | " + " | ".join(f"{l:<8}" for l in labels)
    print(header)
    print("-" * len(header))
    for gt in labels:
        row = f"{gt:<15} | " + " | ".join(f"{matrix[gt][pred]:<8}" for pred in labels)
        print(row)
    print("=" * 65)

    # Compute Per-Class Metrics: Accuracy, Precision, Recall, F1
    print("\nPER-CLASS PERFORMANCE METRICS")
    print("=" * 65)
    print(f"{'Class':<12} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'Accuracy':<8}")
    print("-" * 65)

    macro_precision = []
    macro_recall = []
    macro_f1 = []

    for c in labels:
        # TP, FP, FN, TN
        tp = matrix[c][c]
        fp = sum(matrix[other][c] for other in labels if other != c)
        fn = sum(matrix[c][other] for other in labels if other != c)
        tn = sum(matrix[o1][o2] for o1 in labels if o1 != c for o2 in labels if o2 != c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0

        macro_precision.append(precision)
        macro_recall.append(recall)
        macro_f1.append(f1)

        print(f"{c:<12} | {precision:<9.3f} | {recall:<8.3f} | {f1:<8.3f} | {accuracy:<8.3f}")

    print("-" * 65)
    print(f"{'MACRO-AVG':<12} | {np.mean(macro_precision):<9.3f} | {np.mean(macro_recall):<8.3f} | {np.mean(macro_f1):<8.3f} | {correct_samples / total_samples:<8.3f}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test runner and metrics evaluator for color classification.")
    parser.add_argument("--data-dir", default=None, help="Path to folder containing labeled image subfolders.")
    parser.add_argument("--muzzle", action="store_true", help="Evaluate muzzle color instead of body color.")
    args = parser.parse_args()

    if args.data_dir is not None:
        run_metrics_evaluation(Path(args.data_dir), args.muzzle)
    else:
        # Run standard unit tests
        unittest.main()
