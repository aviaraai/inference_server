"""
pipeline/yolo_crop.py — YOLO-based cattle detection and cropping.

Ported from src/identify.py:crop_or_full(). Detects cattle in an image,
validates single-animal constraint, and returns a padded crop.
"""

import logging
from typing import Optional

import numpy as np

from godhaar.config import (
    CROP_PADDING_PX,
    MAX_CATTLE_PER_IMAGE,
    MIN_BBOX_AREA_PCT,
    YOLO_CONF,
    YOLO_COW_CLASS_ID,
    YOLO_MODEL_NAME,
)

log = logging.getLogger("godhaar.yolo_crop")

# Lazy-loaded YOLO model (loaded once on first call or during warmup)
_yolo_model = None


def load_yolo(model_path: Optional[str] = None) -> None:
    """Load the YOLO model into memory. Called during warmup.

    Parameters
    ----------
    model_path : str, optional
        Path to the YOLO .pt file. Defaults to YOLO_MODEL_NAME from config.
    """
    global _yolo_model
    try:
        from ultralytics import YOLO
    except ImportError:
        log.warning("ultralytics not installed. YOLO crop will be unavailable.")
        return

    path = model_path or YOLO_MODEL_NAME
    _yolo_model = YOLO(path)
    log.info(f"YOLO model loaded from '{path}'")


def warmup_yolo() -> None:
    """Run a dummy inference to warm up the YOLO model."""
    if _yolo_model is None:
        return
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    _yolo_model(dummy, verbose=False)
    log.info("YOLO warmup complete.")


def crop_cattle(
    img: np.ndarray, no_crop: bool = False
) -> tuple[Optional[np.ndarray], str, float]:
    """Detect and crop the cattle from a BGR image.

    Parameters
    ----------
    img : np.ndarray
        BGR image.
    no_crop : bool
        If True, skip YOLO and return the full image.

    Returns
    -------
    (crop, status, confidence) : tuple
        crop     : cropped BGR image, or None if detection failed.
        status   : "OK", "FULL_IMAGE", "FULL_IMAGE_NO_YOLO",
                   "RECAPTURE_NO_DETECTION", "RECAPTURE_MULTI_CATTLE".
        confidence : detection confidence (0.0–1.0).
    """
    if no_crop:
        return img, "FULL_IMAGE", 1.0

    if _yolo_model is None:
        log.warning("YOLO model not loaded. Returning full image.")
        return img, "FULL_IMAGE_NO_YOLO", 1.0

    h, w = img.shape[:2]
    img_area = h * w
    log.info(f"crop_cattle: input image {w}x{h} ({img_area} px)")

    results = _yolo_model(img, verbose=False)[0]
    boxes = []

    # ── DEBUG: log ALL raw YOLO detections before filtering ──
    raw_count = len(results.boxes)
    log.info(f"crop_cattle: YOLO returned {raw_count} raw detections")
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = max(0, x2 - x1) * max(0, y2 - y1)
        area_pct = area / img_area if img_area > 0 else 0
        # Log why each detection passes or fails
        if cls != YOLO_COW_CLASS_ID:
            reason = f"SKIP class={cls} (need {YOLO_COW_CLASS_ID})"
        elif conf < YOLO_CONF:
            reason = f"SKIP conf={conf:.3f} < {YOLO_CONF}"
        elif area_pct < MIN_BBOX_AREA_PCT:
            reason = f"SKIP area={area_pct:.4f} < {MIN_BBOX_AREA_PCT}"
        else:
            reason = "ACCEPTED"
            boxes.append((conf, x1, y1, x2, y2))
        log.info(
            f"  det: class={cls} conf={conf:.3f} bbox=({x1},{y1},{x2},{y2}) "
            f"area_pct={area_pct:.4f} → {reason}"
        )

    if len(boxes) == 0:
        log.warning(f"crop_cattle: NO valid boxes after filtering ({raw_count} raw)")
        return None, "RECAPTURE_NO_DETECTION", 0.0

    if len(boxes) > MAX_CATTLE_PER_IMAGE:
        return None, "RECAPTURE_MULTI_CATTLE", max(b[0] for b in boxes)

    # Take the highest-confidence detection
    boxes.sort(reverse=True)
    conf, x1, y1, x2, y2 = boxes[0]

    # Apply padding
    pad = CROP_PADDING_PX
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

    return img[y1:y2, x1:x2].copy(), "OK", conf
