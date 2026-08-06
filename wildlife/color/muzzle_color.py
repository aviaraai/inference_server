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
    from .utils import (
        aggregate_clusters_by_label,
        bgr_to_lab,
        calculate_median_lab,
        extract_dominant_lab_features,
    )
except ImportError:
    import color_constants as C
    from quality import check_roi_quality
    from roi import get_muzzle_roi
    from utils import (
        aggregate_clusters_by_label,
        bgr_to_lab,
        calculate_median_lab,
        extract_dominant_lab_features,
    )


def _classify_muzzle_lab(lab: list[float]) -> str:
    """Classify a single LAB coordinate into ONE muzzle skin color.

    Returns only BLACK, PINK, or UNKNOWN — never MIXED. MIXED describes a
    muzzle carrying two different skin colors, which is a property of the
    whole muzzle, not of one color sample; a single cluster is by definition
    one color. This previously returned MIXED as its catch-all, so a real
    muzzle whose lower lip fell outside both the BLACK and PINK rules
    contributed a phantom "MIXED" color that then outvoted the actual reading
    and mislabeled a plainly black muzzle.
    """
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

    if l_val <= 50.0:
        return C.LABEL_BLACK

    # Light but not pinkish — genuinely doesn't match either skin color.
    return C.LABEL_UNKNOWN


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
    is_ok, quality_reason = check_roi_quality(
        roi, C.MIN_MUZZLE_ROI_WIDTH, C.MIN_MUZZLE_ROI_HEIGHT
    )
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
    blurred_roi = cv2.medianBlur(roi, 11)

    # 4. Conversion to LAB
    img_lab = bgr_to_lab(blurred_roi)

    # 5. Extract raw color statistics
    median_lab = calculate_median_lab(img_lab)
    features = extract_dominant_lab_features(img_lab)
    dominant_lab = features["dominant_lab"]

    # 6. Classify every cluster and sum weights per label — same reasoning as
    # body_color: clustering over L* splits a uniform muzzle into lit and
    # shadowed clusters of the same skin color, so the heaviest single
    # cluster is not the dominant color. The muzzle ROI is a fixed center
    # crop with no segmentation, so peripheral clusters (surrounding hair,
    # not skin) are discounted by centrality.
    ranked = aggregate_clusters_by_label(
        features["clusters"],
        _classify_muzzle_lab,
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

    # Clusters that match neither skin color don't get a vote on which color
    # the muzzle is — but they still count against confidence below, so a
    # muzzle that was only partly classifiable doesn't report false certainty.
    total_weight = sum(w for _, w in ranked)
    known = [(lbl, w) for lbl, w in ranked if lbl != C.LABEL_UNKNOWN]
    if not known:
        return {
            "label": C.LABEL_UNKNOWN,
            "confidence": 0.0,
            "method": "LAB_KMEANS_V2",
            "reason": "NO_RECOGNIZED_SKIN_COLOR",
            "median_lab": median_lab,
            "dominant_lab": dominant_lab,
        }

    primary_label, primary_weight = known[0]
    runner_up_weight = known[1][1] if len(known) > 1 else 0.0

    # 7. Mixed/speckled muzzles — BOTH skin colors present in comparable
    # amounts, not merely a few dark flecks on a pink muzzle.
    label = primary_label
    if primary_weight > 0 and runner_up_weight / primary_weight >= C.MIXED_RATIO_MIN:
        label = C.LABEL_MIXED

    confidence = primary_weight / total_weight if total_weight > 0 else 0.0

    return {
        "label": label,
        "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 4),
        "method": "LAB_HISTOGRAM_V1",
        "reason": "OK",
        "median_lab": median_lab,
        "dominant_lab": dominant_lab,
    }
