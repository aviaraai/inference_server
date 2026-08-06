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
    SUBJECT_CENTER_WEIGHT_POWER,
    SUBJECT_DOMINANCE_RATIO,
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


def _dominance_score(
    box: tuple[float, int, int, int, int], w: int, h: int
) -> float:
    """Score how strongly a box reads as "the animal this photo is OF".

    Two things make an animal the subject of a hand-held field photo: it fills
    a lot of the frame, and it sits near the middle of it (the photographer
    pointed the phone at it). Neighbouring cattle in a goshala stall can match
    on the first — they are the same size animal, standing just as close — but
    not on the second, because the operator framed the one they meant.

    score = area_fraction * (1 - center_distance) ** SUBJECT_CENTER_WEIGHT_POWER

    center_distance is the box center's distance from the frame center,
    normalized by the half-diagonal so it is 0 at dead center and 1 in a
    corner — resolution- and aspect-independent, so the score means the same
    thing on any phone. Confidence is deliberately NOT a factor: it measures
    how sure YOLO is that something is a cow, not which cow was photographed,
    and on the real photos the background animal often scores HIGHER
    confidence than the subject (0.922 vs 0.861 on the reported front photo)
    because it is unblurred and side-on.
    """
    _, x1, y1, x2, y2 = box
    area_frac = (max(0, x2 - x1) * max(0, y2 - y1)) / float(w * h)

    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    # Normalize by the half-diagonal so distance is 0..1 regardless of shape.
    half_diag = ((w / 2.0) ** 2 + (h / 2.0) ** 2) ** 0.5
    dist = (((cx - w / 2.0) ** 2 + (cy - h / 2.0) ** 2) ** 0.5) / half_diag

    return area_frac * (1.0 - min(1.0, dist)) ** SUBJECT_CENTER_WEIGHT_POWER


def select_dominant_box(
    boxes: list[tuple[float, int, int, int, int]], w: int, h: int
) -> Optional[tuple[float, int, int, int, int]]:
    """Pick the single subject animal out of several detections, or None.

    Returns the top-scoring box only when it beats the runner-up by at least
    SUBJECT_DOMINANCE_RATIO. A None return means no animal stood out — two or
    more are comparably large and comparably centered — and the caller should
    keep rejecting the image rather than guess, since an embedding that could
    belong to either animal is worse than asking for a retake.
    """
    if len(boxes) == 0:
        return None
    if len(boxes) == 1:
        return boxes[0]

    scored = sorted(
        ((_dominance_score(b, w, h), b) for b in boxes),
        key=lambda pair: pair[0],
        reverse=True,
    )
    top_score, top_box = scored[0]
    runner_up_score = scored[1][0]

    for score, b in scored:
        log.info(
            f"  dominance: box=({b[1]},{b[2]},{b[3]},{b[4]}) "
            f"conf={b[0]:.3f} score={score:.4f}"
        )

    if runner_up_score <= 0.0:
        ratio = float("inf")
    else:
        ratio = top_score / runner_up_score

    if ratio < SUBJECT_DOMINANCE_RATIO:
        log.warning(
            f"select_dominant_box: no dominant subject — top/runner-up "
            f"score ratio {ratio:.2f} < {SUBJECT_DOMINANCE_RATIO}"
        )
        return None

    log.info(
        f"select_dominant_box: subject box=({top_box[1]},{top_box[2]},"
        f"{top_box[3]},{top_box[4]}) wins by {ratio:.2f}x"
    )
    return top_box


def detect_primary_animal(img: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Detect the primary animal's box, for body-color ROI localization only.

    Picks the subject with the SAME dominance score crop_cattle() uses
    (_dominance_score: large in frame AND near its center), so every signal
    extracted from one animal's photos describes the same animal. That
    consistency is the point: front photos reach body color through here while
    muzzle photos reach the encoder through crop_cattle(), and a goshala frame
    holds several cattle — if the two used different rules, /register could
    store the neighbour's coat color against the subject's embedding. This
    used to take the LARGEST box, which on the reported front photo picks the
    same animal but for a reason that does not generalize: the white neighbour
    there is 62% the subject's area, so a slightly closer neighbour would flip
    it. Centrality is what actually identifies the animal the operator aimed
    at.

    Unlike crop_cattle() this does NOT enforce MAX_CATTLE_PER_IMAGE, and it
    ignores the dominance RATIO gate — it always returns its best guess when
    anything was detected. Refusing to read a coat color because two animals
    are comparably prominent would help nobody: the fallback is a fixed
    center crop of the whole frame, which is strictly worse than the
    top-scoring animal's box even when that score is a close call.

    Uses only the first two (cheapest) detection attempts from crop_cattle's
    ladder — this is a "nice to have" ROI improvement, not a hard gate, so
    callers should fall back to a non-localized crop on a None return rather
    than pay for the expensive imgsz=1280/TTA retries.

    Returns
    -------
    (x1, y1, x2, y2) of the subject's box, or None if no cattle-like animal
    was detected or the model isn't loaded.
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

    _, x1, y1, x2, y2 = max(boxes, key=lambda b: _dominance_score(b, w, h))
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

    # ── Resolve multiple animals to the one the photo is OF ─────────────────
    # In a goshala the cattle stand adjacent, so a correctly-framed photo of
    # one animal normally has neighbours in frame too. Rejecting all of those
    # outright made registration impossible there. Instead, prefer the
    # dominant subject — largest AND most centered (see select_dominant_box) —
    # and only fall back to RECAPTURE_MULTI_CATTLE when no animal stands out,
    # which is the case the single-animal gate genuinely exists for: two
    # equally prominent animals, no way to know which one was meant.
    if len(boxes) > MAX_CATTLE_PER_IMAGE:
        log.info(
            f"crop_cattle: {len(boxes)} animals in frame — selecting subject"
        )
        subject = select_dominant_box(boxes, w, h)
        if subject is None:
            return None, "RECAPTURE_MULTI_CATTLE", max(b[0] for b in boxes)
        boxes = [subject]

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
