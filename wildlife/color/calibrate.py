"""
calibrate.py — Calibration tool to compute optimal LAB color thresholds from labeled image directories.
"""

import argparse
import math
from pathlib import Path
import cv2
import numpy as np

try:
    from . import color_constants as C
    from .roi import get_body_roi
    from .utils import bgr_to_lab, calculate_median_lab
except ImportError:
    import color_constants as C
    from roi import get_body_roi
    from utils import bgr_to_lab, calculate_median_lab


def get_chroma_and_angle(a_val: float, b_val: float) -> tuple[float, float]:
    """Helper to return chroma and angle (in degrees)."""
    chroma = math.sqrt(a_val**2 + b_val**2)
    angle = math.degrees(math.atan2(b_val, a_val))
    if angle < 0:
        angle += 360.0
    return chroma, angle


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate LAB thresholds from folders containing labeled cattle images (e.g. BLACK/, WHITE/, BROWN/)."
    )
    parser.add_argument(
        "--data-dir", "-d",
        required=True,
        help="Path to folder containing subfolders for each color label.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERROR] Directory not found: {data_dir}")
        return

    # Store stats per class
    # class_name -> list of [L, a, b]
    class_stats = {}

    for class_folder in data_dir.iterdir():
        if not class_folder.is_dir():
            continue
        label = class_folder.name.upper()
        print(f"Reading class: {label} ...")

        stats = []
        for img_path in class_folder.glob("*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Extract body ROI (YOLO-localized when available, else
            # fixed-percentage center crop)
            roi = get_body_roi(img)
            if roi is None or roi.size == 0:
                continue

            # Convert to LAB and calculate median
            img_lab = bgr_to_lab(cv2.medianBlur(roi, 15))
            median_lab = calculate_median_lab(img_lab)
            stats.append(median_lab)

        if stats:
            class_stats[label] = np.array(stats)
            print(f"  Processed {len(stats)} images.")

    if not class_stats:
        print("[ERROR] No class directories or images found.")
        return

    print("\n" + "=" * 50)
    print("CALIBRATION ANALYSIS REPORT")
    print("=" * 50)

    # 1. Neutral/Chroma check (We check white/black/grey chroma to define neutral threshold)
    neutral_chromas = []
    for label in ["BLACK", "WHITE", "GREY"]:
        if label in class_stats:
            arr = class_stats[label]
            chromas = [get_chroma_and_angle(a, b)[0] for a, b in zip(arr[:, 1], arr[:, 2])]
            neutral_chromas.extend(chromas)
            print(f"Achromatic Class {label} Chroma: mean={np.mean(chromas):.2f}, max={np.max(chromas):.2f}")

    if neutral_chromas:
        rec_neutral = float(np.percentile(neutral_chromas, 95))
        print(f"\n---> Recommended NEUTRAL_THRESHOLD (95th percentile): {rec_neutral:.2f}")
    else:
        rec_neutral = C.NEUTRAL_THRESHOLD

    # 2. Black L* threshold
    if "BLACK" in class_stats:
        arr = class_stats["BLACK"]
        black_l = arr[:, 0]
        rec_black = float(np.percentile(black_l, 95))
        print(f"Black L* values: mean={np.mean(black_l):.2f}, 95th percentile={rec_black:.2f}")
        print(f"---> Recommended L_BLACK_MAX: {rec_black:.2f}")

    # 3. White L* threshold
    if "WHITE" in class_stats:
        arr = class_stats["WHITE"]
        white_l = arr[:, 0]
        rec_white = float(np.percentile(white_l, 5))
        print(f"White L* values: mean={np.mean(white_l):.2f}, 5th percentile={rec_white:.2f}")
        print(f"---> Recommended L_WHITE_MIN: {rec_white:.2f}")

    # 4. Brown Angle check
    if "BROWN" in class_stats:
        arr = class_stats["BROWN"]
        angles = [get_chroma_and_angle(a, b)[1] for a, b in zip(arr[:, 1], arr[:, 2])]
        rec_angle_min = float(np.percentile(angles, 5))
        rec_angle_max = float(np.percentile(angles, 95))
        print(f"\nBrown angle values: mean={np.mean(angles):.1f}°, min={np.min(angles):.1f}°, max={np.max(angles):.1f}°")
        print(f"---> Recommended BROWN_ANGLE_MIN: {rec_angle_min:.1f}")
        print(f"---> Recommended BROWN_ANGLE_MAX: {rec_angle_max:.1f}")

    print("=" * 50)


if __name__ == "__main__":
    main()
