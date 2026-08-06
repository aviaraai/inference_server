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
    from .utils import (
        aggregate_clusters_by_label,
        bgr_to_lab,
        calculate_median_lab,
        extract_dominant_lab_features,
    )
except ImportError:
    import color_constants as C
    from quality import check_roi_quality
    from roi import get_body_roi
    from utils import (
        aggregate_clusters_by_label,
        bgr_to_lab,
        calculate_median_lab,
        extract_dominant_lab_features,
    )


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
    # 1. Bounding box ROI — YOLO-localized when available, else a
    # fixed-percentage center crop.
    roi = get_body_roi(img_bgr)

    # 2. Quality Gate check
    is_ok, quality_reason = check_roi_quality(roi)
    if not is_ok:
        return {
            "label": C.LABEL_UNKNOWN,
            "confidence": 0.0,
            "method": "LAB_KMEANS_V2",
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

    # 6. Classify every cluster and sum weights per label, so a coat split
    # across several lighting clusters is scored as one color. Peripheral
    # clusters are dropped as scene rather than animal.
    ranked = aggregate_clusters_by_label(
        features["clusters"],
        _classify_lab,
        centrality_ratio_min=C.SPOTTED_CENTRALITY_RATIO_MIN,
    )
    if not ranked:
        return {
            "label": C.LABEL_UNKNOWN,
            "confidence": 0.0,
            "method": "LAB_KMEANS_V2",
            "reason": "NO_COLOR_CLUSTERS",
            "median_lab": median_lab,
            "dominant_lab": dominant_lab,
        }

    primary_label, primary_weight = ranked[0]
    runner_up_weight = ranked[1][1] if len(ranked) > 1 else 0.0

    # 7. Spotted coats — a second color class holding a comparable share of
    # the animal, not merely a present one. Nearly every real animal shows
    # some second color (a blaze, a sock, an ear), so requiring only that one
    # exists labels almost everything SPOTTED. Requiring the two to be
    # genuinely comparable reserves SPOTTED for actually two-tone coats.
    label = primary_label
    if primary_weight > 0 and runner_up_weight / primary_weight >= C.SPOTTED_RATIO_MIN:
        label = C.LABEL_SPOTTED

    # Confidence: the winning color's share of the sampled coat.
    confidence = primary_weight / sum(w for _, w in ranked) if ranked else 0.0
    # A coat sitting right on the neutral/chromatic boundary could go either
    # way on the next photo — say so rather than reporting false certainty.
    chroma = math.sqrt(dominant_lab[1]**2 + dominant_lab[2]**2)
    if abs(chroma - C.NEUTRAL_THRESHOLD) < 3.0:
        confidence *= 0.7

    return {
        "label": label,
        "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 4),
        "method": "LAB_KMEANS_V2",
        "reason": "OK",
        "median_lab": median_lab,
        "dominant_lab": dominant_lab,
    }
