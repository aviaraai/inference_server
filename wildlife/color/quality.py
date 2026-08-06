"""
quality.py — Gating functions to reject blur, extreme exposure, or low contrast.
"""

import cv2
import numpy as np

try:
    from . import color_constants as C
except ImportError:
    import color_constants as C


def check_roi_quality(
    roi: np.ndarray,
    min_width: int | None = None,
    min_height: int | None = None,
) -> tuple[bool, str]:
    """Perform image quality checks on the BGR Region of Interest (ROI).

    Parameters
    ----------
    min_width, min_height : int, optional
        Override the default minimum ROI size. A detector-localized ROI is
        legitimately smaller than a fixed crop of the whole frame — see
        MIN_MUZZLE_ROI_WIDTH in color_constants.py.

    Returns
    -------
    is_ok : bool
        True if all quality gates pass, False otherwise.
    reason : str
        "OK" or a description of the failed gate.
    """
    if roi is None or roi.size == 0:
        return False, "EMPTY_ROI"

    min_w = C.MIN_ROI_WIDTH if min_width is None else min_width
    min_h = C.MIN_ROI_HEIGHT if min_height is None else min_height

    h, w = roi.shape[:2]

    # 1. Size Check
    if w < min_w or h < min_h:
        return False, f"ROI_TOO_SMALL: {w}x{h} (min {min_w}x{min_h})"

    # Convert to Grayscale for checks
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 2. Brightness (Exposure) Check
    avg_brightness = float(gray.mean())
    if avg_brightness < C.BRIGHTNESS_MIN:
        return False, f"TOO_DARK: brightness={avg_brightness:.1f} (min {C.BRIGHTNESS_MIN})"
    if avg_brightness > C.BRIGHTNESS_MAX:
        return False, f"TOO_BRIGHT: brightness={avg_brightness:.1f} (max {C.BRIGHTNESS_MAX})"

    # 3. Contrast Check (Standard Deviation)
    contrast = float(gray.std())
    if contrast < C.MIN_CONTRAST:
        return False, f"LOW_CONTRAST: contrast={contrast:.1f} (min {C.MIN_CONTRAST})"

    # 4. Blur Check (Laplacian Variance)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < C.BLUR_THRESHOLD:
        return False, f"BLURRY: blur_score={blur_score:.1f} (min {C.BLUR_THRESHOLD})"

    return True, "OK"
