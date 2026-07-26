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

        self.assertEqual(result["method"], "LAB_HISTOGRAM_V1")
        self.assertEqual(result["label"], C.LABEL_GREY)
        self.assertEqual(result["reason"], "OK")

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
