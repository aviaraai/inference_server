"""
pipeline/yolo_crop.py — YOLO-based cattle detection and cropping.

Ported from src/identify.py:crop_or_full(). Detects cattle in an image,
validates single-animal constraint, and returns a padded crop.

When YOLO fails to detect on the raw image (common with dark-colored Indian
cattle/buffalo from low-contrast phone cameras), the pipeline retries with
CLAHE contrast enhancement. If that also fails, it falls back to using the
full image rather than rejecting the request outright.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from godhaar.config import (
    CROP_PADDING_PX,
    MAX_CATTLE_PER_IMAGE,
    MIN_BBOX_AREA_PCT,
    YOLO_CONF,
    YOLO_INTERNAL_CONF,
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


# ── Image enhancement for YOLO detection ──────────────────────────────────────

def _enhance_for_detection(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement to help YOLO detect dark cattle.

    Dark-colored Indian cattle (buffalo, Murrah, etc.) photographed by
    various phone cameras often have very low contrast — the animal blends
    into shadows, dark soil, or overcast skies. CLAHE (Contrast Limited
    Adaptive Histogram Equalization) locally boosts contrast so that the
    animal's outline becomes distinct enough for YOLO to pick up.

    The enhancement is applied ONLY for YOLO detection; the original
    unmodified image is still used for the final crop, so downstream
    embedding and color extraction are unaffected.

    Parameters
    ----------
    img : np.ndarray
        BGR image (original).

    Returns
    -------
    np.ndarray
        Contrast-enhanced BGR image for YOLO inference only.
    """
    # Convert to LAB color space — L channel holds luminance
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    # CLAHE on the luminance channel
    # clipLimit=3.0 gives a strong but not over-blown enhancement;
    # tileGridSize 8×8 is the standard for most resolutions.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_chan)

    lab_enhanced = cv2.merge([l_enhanced, a_chan, b_chan])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return enhanced


def _run_yolo(
    img: np.ndarray, img_area: int, **kwargs
) -> list[tuple[float, int, int, int, int]]:
    """Run YOLO on a single image and return filtered boxes.

    Uses YOLO_INTERNAL_CONF (0.10) as the inference threshold so we see
    ALL potential detections, then applies our own YOLO_CONF (0.30) filter.
    This prevents YOLO's default (0.25) from silently dropping dark-cattle
    detections that fall between 0.10 and 0.25.

    Extra **kwargs (e.g. ``imgsz=1280``, ``augment=True``) are forwarded
    directly to the YOLO model call for retry strategies.

    Parameters
    ----------
    img : np.ndarray
        BGR image to run inference on.
    img_area : int
        Total pixel area of the image (for area-based filtering).
    **kwargs
        Extra arguments forwarded to YOLO inference (imgsz, augment, etc.).

    Returns
    -------
    list of (conf, x1, y1, x2, y2) tuples that pass class/conf/area filters.
    """
    extra_desc = ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    try:
        results = _yolo_model(
            img, conf=YOLO_INTERNAL_CONF, verbose=False, **kwargs
        )[0]
    except Exception as e:
        log.error(f"YOLO inference failed: {e}")
        return []

    boxes = []
    raw_count = len(results.boxes)
    tag = f" [{extra_desc}]" if extra_desc else ""
    log.info(
        f"crop_cattle: YOLO returned {raw_count} raw detections"
        f" (conf>={YOLO_INTERNAL_CONF}){tag}"
    )

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = (int(round(float(v))) for v in box.xyxy[0])
        area = max(0, x2 - x1) * max(0, y2 - y1)
        area_pct = area / img_area

        if cls not in YOLO_CATTLE_CLASS_IDS:
            reason = f"SKIP class={cls} (not in {YOLO_CATTLE_CLASS_IDS})"
        elif conf < YOLO_CONF:
            reason = f"SKIP conf={conf:.3f} < {YOLO_CONF}"
        elif area_pct < MIN_BBOX_AREA_PCT:
            reason = f"SKIP area={area_pct:.4f} < {MIN_BBOX_AREA_PCT}"
        else:
            reason = "ACCEPTED"
            boxes.append((conf, x1, y1, x2, y2))

        # Log ALL detections at INFO level so we can diagnose failures
        log.info(
            f"  det: class={cls} conf={conf:.3f} bbox=({x1},{y1},{x2},{y2}) "
            f"area_pct={area_pct:.4f} → {reason}"
        )

    return boxes


def detect_primary_animal(img: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Detect the primary animal's box, for body-color ROI localization only.

    Unlike crop_cattle() (the muzzle-embedding path), this does NOT enforce
    MAX_CATTLE_PER_IMAGE: a real field/goshala photo often has other cattle
    in the background, and refusing to read the subject's coat color because
    a neighbor is also in frame would be wrong for this use case (the
    single-animal constraint exists so muzzle embeddings aren't ambiguous
    about which animal they represent — that reasoning doesn't apply to
    localizing where to sample body color). Among all detected boxes, the
    LARGEST one by area is taken as the subject — not the highest-confidence
    one — since the photographed animal is normally closest to the camera
    and fills more of the frame than anything in the background.

    Uses only the first two (cheapest) detection attempts from crop_cattle's
    ladder — this is a "nice to have" ROI improvement, not a hard gate, so
    callers should fall back to a non-localized crop on a None return rather
    than pay for the expensive imgsz=1280/TTA retries.

    Returns
    -------
    (x1, y1, x2, y2) of the largest detected box, or None if no cattle-like
    animal was detected or the model isn't loaded.
    """
    if img is None or img.size == 0 or _yolo_model is None:
        return None

    h, w = img.shape[:2]
    img_area = h * w

    boxes = _run_yolo(img, img_area)
    if len(boxes) == 0:
        boxes = _run_yolo(_enhance_for_detection(img), img_area)

    if len(boxes) == 0:
        return None

    def _area(box: tuple[float, int, int, int, int]) -> int:
        _, bx1, by1, bx2, by2 = box
        return max(0, bx2 - bx1) * max(0, by2 - by1)

    _, x1, y1, x2, y2 = max(boxes, key=_area)
    return x1, y1, x2, y2


def crop_cattle(
    img: np.ndarray, no_crop: bool = False
) -> tuple[Optional[np.ndarray], str, float]:
    """Detect and crop the cattle from a BGR image.

    Detection strategy (progressive retries, cheapest first):
      1. YOLO on original image (imgsz=640, ~30ms)
      2. YOLO on CLAHE-enhanced image (imgsz=640, ~40ms)
      3. CLAHE + imgsz=1280 — higher resolution preserves detail (~100ms)
      4. CLAHE + augment=True — multi-scale TTA, most robust (~500ms)

    Each retry is more expensive but catches harder cases (dark cattle,
    low-contrast phone cameras, unusual angles). The crop is ALWAYS taken
    from the original image — enhancement is only used to help YOLO find
    the bounding box.

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

    # ── Attempt 1: YOLO on the original image ─────────────────────────────
    boxes = _run_yolo(img, img_area)

    # ── Attempt 2: CLAHE-enhanced image ───────────────────────────────────
    enhanced = None
    if len(boxes) == 0:
        log.info("crop_cattle: attempt 2 — CLAHE enhancement...")
        enhanced = _enhance_for_detection(img)
        boxes = _run_yolo(enhanced, img_area)

    # ── Attempt 3: CLAHE + higher resolution (imgsz=1280) ─────────────────
    if len(boxes) == 0:
        log.info("crop_cattle: attempt 3 — CLAHE + imgsz=1280...")
        if enhanced is None:
            enhanced = _enhance_for_detection(img)
        boxes = _run_yolo(enhanced, img_area, imgsz=1280)

    # ── Attempt 4: CLAHE + test-time augmentation (most expensive) ────────
    if len(boxes) == 0:
        log.info("crop_cattle: attempt 4 — CLAHE + augment (TTA)...")
        if enhanced is None:
            enhanced = _enhance_for_detection(img)
        boxes = _run_yolo(enhanced, img_area, augment=True)

    # ── Log which attempt succeeded ───────────────────────────────────────
    if len(boxes) > 0:
        log.info(f"crop_cattle: {len(boxes)} valid box(es) found")

    # ── All attempts failed — reject ──────────────────────────────────────
    if len(boxes) == 0:
        log.warning(
            "crop_cattle: NO detection after all 4 attempts. "
            "Requesting recapture."
        )
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

    # Always crop tightly to the detected box, padded a few pixels — crop from
    # the ORIGINAL image, not the enhanced one.
    #
    # This used to special-case "close-up" detections (best box covering most
    # of the frame) by returning the full, uncropped photo instead of a bbox
    # crop, on the theory that the animal was "too close for a meaningful
    # crop." That was backwards for this pipeline: crop_cattle()'s only
    # production callers are the MUZZLE embedding path (main.py register/
    # search) — there is no whole-body detection use case here that needed an
    # uncropped fallback. The capture UI tells officers to fill the frame
    # with the muzzle, so a correctly-taken photo routinely triggered this
    # branch — meaning most muzzle photos were embedded as the full raw scene
    # (background, ground, other cattle) instead of a subject-filling crop.
    # preprocess.py then stretches whatever comes out of here to a fixed
    # 518x518 (non-aspect-preserving), so the full-frame case diluted the
    # muzzle even further. GodhaarModel was trained on tight, subject-filling
    # crops — always producing one here, regardless of how large the
    # detection is, is what it actually expects.
    pad = CROP_PADDING_PX
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

    return img[y1:y2, x1:x2].copy(), "OK", conf
