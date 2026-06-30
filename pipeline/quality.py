"""
pipeline/quality.py — Image quality gate for the inference server.

Ported from src/identify.py:quality_check(). Runs on both /register and
/search images to prevent bad embeddings from entering the FAISS index.
"""

import cv2
import numpy as np

from godhaar.config import (
    MIN_SHORT_SIDE,
    BLUR_THRESHOLD,
    MIN_EXPOSURE,
    MAX_EXPOSURE,
)


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
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    exp = float(gray.mean())

    if short < MIN_SHORT_SIDE:
        return "RECAPTURE", f"bad_quality short={short}"
    if blur < BLUR_THRESHOLD:
        return "RECAPTURE", f"bad_quality blur={blur:.2f}"
    if exp < MIN_EXPOSURE or exp > MAX_EXPOSURE:
        return "RECAPTURE", f"bad_quality exposure={exp:.2f}"

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
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    exp = float(gray.mean())

    if short < MIN_SHORT_SIDE:
        return "RECAPTURE", f"bad_quality short={short}"
    if blur < BLUR_THRESHOLD:
        return "RECAPTURE", f"bad_quality blur={blur:.2f}"
    if exp < MIN_EXPOSURE or exp > MAX_EXPOSURE:
        return "RECAPTURE", f"bad_quality exposure={exp:.2f}"

    return "GOOD", "ok"
