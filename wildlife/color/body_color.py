"""
body_color.py — Main classifier for cattle body coat color.
"""

import math
import cv2
import numpy as np

try:
    from . import color_constants as C
    from .quality import check_roi_quality
    from .roi import get_body_roi
    from .utils import bgr_to_lab, calculate_median_lab, extract_dominant_lab_features
except ImportError:
    import color_constants as C
    from quality import check_roi_quality
    from roi import get_body_roi
    from utils import bgr_to_lab, calculate_median_lab, extract_dominant_lab_features


def _classify_lab(lab: list[float]) -> str:
    """Classify a single LAB coordinate into a primary color label."""
    l_val, a_val, b_val = lab[0], lab[1], lab[2]

    # Calculate chroma distance from origin (achromatic check)
    chroma = math.sqrt(a_val**2 + b_val**2)

    if chroma < C.NEUTRAL_THRESHOLD:
        # Achromatic categorization
        if l_val <= C.L_BLACK_MAX:
            return C.LABEL_BLACK
        elif l_val >= C.L_WHITE_MIN:
            return C.LABEL_WHITE
        else:
            return C.LABEL_GREY

    # Chromatic categorization (angle in degrees in a*-b* plane)
    angle = math.degrees(math.atan2(b_val, a_val))
    if angle < 0:
        angle += 360.0

    # Brown/Red spectrum (typically first quadrant, positive a* and b*)
    if C.BROWN_ANGLE_MIN <= angle <= C.BROWN_ANGLE_MAX and b_val > 0:
        return C.LABEL_BROWN

    return C.LABEL_UNKNOWN


def classify_body_color(img_bgr: np.ndarray) -> dict:
    """Classify the body coat color of a cow from a BGR image.

    Parameters
    ----------
    img_bgr : np.ndarray
        Full BGR image matrix.

    Returns
    -------
    dict
        Unified API contract containing label, confidence, method, reason,
        and raw LAB statistics.
    """
    # 1. Bounding box ROI
    roi = get_body_roi(img_bgr)

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
    blurred_roi = cv2.medianBlur(roi, 15)

    # 4. Conversion to LAB
    img_lab = bgr_to_lab(blurred_roi)

    # 5. Extract raw color statistics
    median_lab = calculate_median_lab(img_lab)
    features = extract_dominant_lab_features(img_lab)
    dominant_lab = features["dominant_lab"]

    # 6. Apply classification rules
    primary_label = _classify_lab(dominant_lab)

    # 7. Check for Spotted/Mixed coats (Multimodal peak check)
    if features["peak_ratio"] >= C.SPOTTED_RATIO_MIN:
        # Determine if the secondary peak represents a different color class
        # To simulate secondary peak, perturb dominant LAB based on a*-b* direction offsets
        # If the primary label is Black/Grey and secondary represents White, or vice versa,
        # we mark it as SPOTTED.
        # For simplicity, if primary is BLACK or BROWN and it's multimodal, we tend to classify
        # as SPOTTED when the overall contrast or lightness variance is high.
        h, w = img_lab.shape[:2]
        gray_roi = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2GRAY)
        std_dev = gray_roi.std()
        if std_dev > 25.0:  # High local contrast implies spotted pattern
            label = C.LABEL_SPOTTED
        else:
            label = primary_label
    else:
        label = primary_label

    # Adjust confidence mathematically:
    # Scale based on how clearly the dominant color stands out
    confidence = features["confidence"]
    # Adjust for neutral margins
    chroma = math.sqrt(dominant_lab[1]**2 + dominant_lab[2]**2)
    neutral_dist = abs(chroma - C.NEUTRAL_THRESHOLD)
    # If it is extremely close to the neutral boundary, reduce confidence
    if neutral_dist < 3.0:
        confidence *= 0.7

    return {
        "label": label,
        "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 4),
        "method": "LAB_HISTOGRAM_V1",
        "reason": "OK",
        "median_lab": median_lab,
        "dominant_lab": dominant_lab,
    }
