"""
muzzle_color.py — Main classifier for cattle muzzle skin color.
"""

import math
import cv2
import numpy as np

try:
    from . import color_constants as C
    from .quality import check_roi_quality
    from .roi import get_muzzle_roi
    from .utils import bgr_to_lab, calculate_median_lab, extract_dominant_lab_features
except ImportError:
    import color_constants as C
    from quality import check_roi_quality
    from roi import get_muzzle_roi
    from utils import bgr_to_lab, calculate_median_lab, extract_dominant_lab_features


def _classify_muzzle_lab(lab: list[float]) -> str:
    """Classify a single LAB coordinate into a primary muzzle color label."""
    l_val, a_val, b_val = lab[0], lab[1], lab[2]

    # Calculate chroma distance
    chroma = math.sqrt(a_val**2 + b_val**2)

    # Black muzzle: dark skin, low chroma
    if l_val <= C.L_BLACK_MAX and chroma < C.NEUTRAL_THRESHOLD:
        return C.LABEL_BLACK

    # Pink muzzle: higher lightness, higher positive a* (red/pink tone)
    # Pink skin is generally L* > 45, and a* > 4
    if l_val > 45.0 and a_val > 4.0:
        return C.LABEL_PINK

    # If it is light but not pinkish, default to MIXED or BLACK based on lightness
    if l_val <= 50.0:
        return C.LABEL_BLACK

    return C.LABEL_MIXED


def classify_muzzle_color(img_bgr: np.ndarray) -> dict:
    """Classify the muzzle skin color of a cow from a BGR image.

    Parameters
    ----------
    img_bgr : np.ndarray
        Full cropped muzzle image.

    Returns
    -------
    dict
        Unified API contract containing label, confidence, method, reason,
        and raw LAB statistics.
    """
    # 1. Slice center region
    roi = get_muzzle_roi(img_bgr)

    # 2. Quality Gate check
    is_ok, quality_reason = check_roi_quality(roi)
    if not is_ok:
        return {
            "label": C.LABEL_UNKNOWN,
            "confidence": 0.0,
            "method": "LAB_HISTOGRAM_V1",
            "reason": f"LOW_QUALITY: {quality_reason}",
            "median_lab": [0.0, 0.0, 0.0],
            "dominant_lab": [0.0, 0.0, 0.0],
        }

    # 3. Median filter to remove hair pattern noise
    blurred_roi = cv2.medianBlur(roi, 11)

    # 4. Conversion to LAB
    img_lab = bgr_to_lab(blurred_roi)

    # 5. Extract raw color statistics
    median_lab = calculate_median_lab(img_lab)
    features = extract_dominant_lab_features(img_lab)
    dominant_lab = features["dominant_lab"]

    # 6. Apply classification rules
    primary_label = _classify_muzzle_lab(dominant_lab)

    # 7. Check for Mixed/Speckled muzzles (Multimodal peak check)
    if features["peak_ratio"] >= C.MIXED_RATIO_MIN:
        # If there's high lightness variance or standard deviation in the ROI,
        # it indicates black/pink speckled spots on the muzzle skin
        gray_roi = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2GRAY)
        std_dev = gray_roi.std()
        if std_dev > 20.0:
            label = C.LABEL_MIXED
        else:
            label = primary_label
    else:
        label = primary_label

    # Adjust confidence based on peak ratio and standard dev
    confidence = features["confidence"]

    return {
        "label": label,
        "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 4),
        "method": "LAB_HISTOGRAM_V1",
        "reason": "OK",
        "median_lab": median_lab,
        "dominant_lab": dominant_lab,
    }
