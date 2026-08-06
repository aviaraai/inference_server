"""
pipeline/quality.py — Image quality gate for the inference server.

Ported from src/identify.py:quality_check(). Runs on both /register and
/search images to prevent bad embeddings from entering the FAISS index.

Blur is measured on the central 50% crop of the image, not the full frame.
This avoids penalizing close-up shots where the background is intentionally
blurred (depth-of-field / bokeh), while still catching genuinely blurry
subjects.
"""

import cv2
import numpy as np

from godhaar.config import (
    MIN_SHORT_SIDE,
    BLUR_THRESHOLD,
    MIN_EXPOSURE,
    MIN_EXPOSURE_STD,
    MAX_EXPOSURE,
)

# Fraction of the image (centered) used for blur measurement.
_BLUR_CENTER_FRAC = 0.50


def _center_region(gray: np.ndarray) -> np.ndarray:
    """Return the central crop of a grayscale image for blur measurement.

    Uses the central `_BLUR_CENTER_FRAC` of each axis so that out-of-focus
    backgrounds (bokeh) don't pull the Laplacian variance below the threshold.
    """
    h, w = gray.shape[:2]
    dy = int(h * (1 - _BLUR_CENTER_FRAC) / 2)
    dx = int(w * (1 - _BLUR_CENTER_FRAC) / 2)
    # Guard against degenerate images where crop would be empty
    dy = max(dy, 0)
    dx = max(dx, 0)
    return gray[dy: h - dy or h, dx: w - dx or w]


def _blur_score(gray: np.ndarray) -> float:
    """Sharpness of an already-CROPPED subject: the sharper of two regions.

    The central-50% rule below (_center_region) assumes the subject is central
    and the out-of-focus background peripheral. That holds for a raw camera
    frame, which is why quality_check() still uses it alone. It inverts for a
    tight YOLO crop: the background has ALREADY been cropped away, and what
    now sits dead-center is the animal's smooth hide — the bridge of the nose,
    a flat cheek — while the texture that actually proves focus (the muzzle's
    bead pattern, hair boundaries, horn edges) sits off-center.

    Measured on a real reported field photo: the full frame's central 50%
    scored 511.97 — unambiguously sharp — while the same photo's CROP scored
    17.50 and was rejected as "bad_quality blur". Nothing about the photo was
    blurry; the measurement window had simply moved onto featureless hide.
    This is the same class of defect as the muzzle-color ROI bug (see
    CLAUDE.md): the wrong pixels were being measured, and no change to
    BLUR_THRESHOLD can fix that — lowering it to admit this photo would admit
    genuinely blurry ones too.

    Taking the MAX of the central region and the whole crop asks the question
    the gate actually cares about — "is the subject in focus ANYWHERE?" — and
    is monotonically >= the previous measurement, so it cannot reject any
    image that passes today. It only stops discarding sharp ones.
    """
    center = float(cv2.Laplacian(_center_region(gray), cv2.CV_64F).var())
    whole = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return max(center, whole)


def quality_check(image_bytes: bytes) -> tuple[str, str]:
    """Run quality checks on raw image bytes.

    Parameters
    ----------
    image_bytes : bytes
        Raw image file content.

    Returns
    -------
    (status, reason) : tuple[str, str]
        status is one of "GOOD", "REJECTED_INPUT", "RECAPTURE".
        reason is a human-readable explanation.
    """
    # Decode
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return "REJECTED_INPUT", "cannot_decode_image"

    h, w = img.shape[:2]
    short = min(w, h)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(_center_region(gray), cv2.CV_64F).var())
    exp = float(gray.mean())

    if short < MIN_SHORT_SIDE:
        return "RECAPTURE", f"bad_quality short={short}"
    if blur < BLUR_THRESHOLD:
        return "RECAPTURE", f"bad_quality blur={blur:.2f}"
    if exp > MAX_EXPOSURE:
        return "RECAPTURE", f"bad_quality exposure={exp:.2f}"
    if exp < MIN_EXPOSURE:
        # Low mean brightness alone doesn't distinguish a naturally dark
        # subject (black/dark-brown cattle, common in Indian breeds) from a
        # genuinely underexposed photo. A real underexposed shot is dark AND
        # flat; a well-lit dark animal still has real local contrast. Only
        # reject when both conditions hold.
        contrast = float(gray.std())
        if contrast < MIN_EXPOSURE_STD:
            return (
                "RECAPTURE",
                f"bad_quality exposure={exp:.2f} contrast={contrast:.2f}",
            )

    return "GOOD", "ok"


def quality_check_cv2(img: np.ndarray) -> tuple[str, str]:
    """Run quality checks on an already-decoded BGR image.

    Parameters
    ----------
    img : np.ndarray
        BGR image matrix (from cv2.imread or YOLO crop).

    Returns
    -------
    (status, reason) : tuple[str, str]
    """
    if img is None or img.size == 0:
        return "REJECTED_INPUT", "empty_image"

    h, w = img.shape[:2]
    short = min(w, h)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # This function's callers pass a YOLO crop (main.py register/search), so
    # blur is scored with the crop-aware rule — see _blur_score.
    blur = _blur_score(gray)
    exp = float(gray.mean())

    if short < MIN_SHORT_SIDE:
        return "RECAPTURE", f"bad_quality short={short}"
    if blur < BLUR_THRESHOLD:
        return "RECAPTURE", f"bad_quality blur={blur:.2f}"
    if exp > MAX_EXPOSURE:
        return "RECAPTURE", f"bad_quality exposure={exp:.2f}"
    if exp < MIN_EXPOSURE:
        # Low mean brightness alone doesn't distinguish a naturally dark
        # subject (black/dark-brown cattle, common in Indian breeds) from a
        # genuinely underexposed photo. A real underexposed shot is dark AND
        # flat; a well-lit dark animal still has real local contrast. Only
        # reject when both conditions hold.
        contrast = float(gray.std())
        if contrast < MIN_EXPOSURE_STD:
            return (
                "RECAPTURE",
                f"bad_quality exposure={exp:.2f} contrast={contrast:.2f}",
            )

    return "GOOD", "ok"
