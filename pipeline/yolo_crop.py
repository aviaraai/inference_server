"""
pipeline/yolo_crop.py — YOLO-based cattle detection and cropping.

Ported from src/identify.py:crop_or_full(). Detects cattle in an image,
validates single-animal constraint, and returns a padded crop.
"""

import logging
from typing import Optional

import numpy as np

from godhaar.config import (
    CLOSE_UP_AREA_PCT,
    CROP_PADDING_PX,
    MAX_CATTLE_PER_IMAGE,
    MIN_BBOX_AREA_PCT,
    YOLO_CONF,
    YOLO_CATTLE_CLASS_IDS,
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
    if img.size == 0:
        return None, "RECAPTURE_NO_DETECTION", 0.0

    if no_crop:
        return img, "FULL_IMAGE", 1.0

    if _yolo_model is None:
        log.warning("YOLO model not loaded. Returning full image.")
        return img, "FULL_IMAGE_NO_YOLO", 1.0

    h, w = img.shape[:2]
    img_area = h * w
    log.info(f"crop_cattle: input image {w}x{h} ({img_area} px)")

    try:
        results = _yolo_model(img, verbose=False)[0]
    except Exception as e:
        log.error(f"YOLO inference failed: {e}")
        return None, "RECAPTURE_NO_DETECTION", 0.0
    boxes = []

    # ── DEBUG: log ALL raw YOLO detections before filtering ──
    raw_count = len(results.boxes)
    log.info(f"crop_cattle: YOLO returned {raw_count} raw detections")
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = (int(round(float(v))) for v in box.xyxy[0])
        area = max(0, x2 - x1) * max(0, y2 - y1)
        area_pct = area / img_area
        # Log why each detection passes or fails
        if cls not in YOLO_CATTLE_CLASS_IDS:
            reason = f"SKIP class={cls} (not in {YOLO_CATTLE_CLASS_IDS})"
        elif conf < YOLO_CONF:
            reason = f"SKIP conf={conf:.3f} < {YOLO_CONF}"
        elif area_pct < MIN_BBOX_AREA_PCT:
            reason = f"SKIP area={area_pct:.4f} < {MIN_BBOX_AREA_PCT}"
        else:
            reason = "ACCEPTED"
            boxes.append((conf, x1, y1, x2, y2))
        log.debug(
            f"  det: class={cls} conf={conf:.3f} bbox=({x1},{y1},{x2},{y2}) "
            f"area_pct={area_pct:.4f} → {reason}"
        )

    if len(boxes) == 0:
        log.warning(f"crop_cattle: NO valid boxes after filtering ({raw_count} raw)")
        return None, "RECAPTURE_NO_DETECTION", 0.0

    # ── Cross-class NMS deduplication ───────────────────────────────────────
    # YOLO sometimes fires multiple classes (e.g. "cow" + "horse") on the same
    # buffalo body. If two boxes overlap by more than IOU_MERGE_THRESHOLD of
    # their union area, they refer to the same animal — keep only the
    # highest-confidence one.
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a[1], a[2], a[3], a[4]
        bx1, by1, bx2, by2 = b[1], b[2], b[3], b[4]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)

    IOU_MERGE_THRESHOLD = 0.50
    boxes.sort(reverse=True)          # highest conf first
    kept = []
    for box in boxes:
        if all(_iou(box, k) < IOU_MERGE_THRESHOLD for k in kept):
            kept.append(box)
        else:
            log.info(
                f"crop_cattle: merged duplicate box "
                f"conf={box[0]:.3f} (IoU >= {IOU_MERGE_THRESHOLD})"
            )
    boxes = kept
    log.info(f"crop_cattle: {len(boxes)} box(es) after IoU deduplication")

    if len(boxes) > MAX_CATTLE_PER_IMAGE:
        return None, "RECAPTURE_MULTI_CATTLE", max(b[0] for b in boxes)

    # Take the highest-confidence detection
    boxes.sort(reverse=True)
    conf, x1, y1, x2, y2 = boxes[0]

    # Close-up fallback: if the best box fills most of the frame the animal is
    # too close for a meaningful crop — return the full image instead.
    area_pct = (x2 - x1) * (y2 - y1) / img_area
    if area_pct >= CLOSE_UP_AREA_PCT:
        log.info(
            f"crop_cattle: close-up detected (area_pct={area_pct:.3f} >= "
            f"{CLOSE_UP_AREA_PCT}), returning full image."
        )
        return img, "FULL_IMAGE", conf

    # Apply padding
    pad = CROP_PADDING_PX
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

    return img[y1:y2, x1:x2].copy(), "OK", conf
