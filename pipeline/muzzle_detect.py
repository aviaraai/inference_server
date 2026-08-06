"""
pipeline/muzzle_detect.py — Muzzle localization for color sampling.

Wraps the app's single-class muzzle detector (`best_float16.tflite`, a
YOLOv8n exported to TFLite, class `cattle_muzzle_3`) — the same weights the
Telangana Android app ships in its assets. It loads through the existing
`ultralytics` dependency via `YOLO(path, task="detect")`, which delegates to
an `ai-edge-litert` backend.

Why this exists: `crop_cattle()` (pipeline/yolo_crop.py) is a COCO detector
that finds the WHOLE ANIMAL, not the muzzle. Muzzle color used to be read
from the center of that whole-animal box, which is the animal's neck/chest —
so it reported the color of the coat rather than the nose, e.g. PINK for an
obviously black muzzle on a brown-hided cow. This module localizes the actual
muzzle first.

Scope: color sampling only. The embedding path (`main.py` register/search)
still feeds `crop_cattle()`'s whole-animal crop to the encoder — changing
that would move stored and query embeddings into a different space and
invalidate every vector already in the FAISS index, which is a separate,
much larger migration.
"""

import logging
import os
from typing import Optional

import numpy as np

from godhaar.config import (
    MUZZLE_DETECT_CONF,
    MUZZLE_MODEL_PATH,
)

log = logging.getLogger("godhaar.muzzle_detect")

_muzzle_model = None


def load_muzzle_detector(model_path: Optional[str] = None) -> None:
    """Load the muzzle detector into memory. Called during warmup.

    Failure is non-fatal and logged, not raised: muzzle localization is an
    accuracy improvement for one returned field, not a request-critical
    dependency. Without it, get_muzzle_roi() falls back to its fixed center
    crop and the service keeps serving.
    """
    global _muzzle_model

    path = model_path or MUZZLE_MODEL_PATH
    if not path or not os.path.isfile(path):
        log.warning(
            f"Muzzle detector not found at '{path}'. Muzzle color will fall "
            "back to a fixed center crop of the whole-animal box, which reads "
            "coat color rather than muzzle color — see pipeline/muzzle_detect.py."
        )
        return

    try:
        from ultralytics import YOLO

        _muzzle_model = YOLO(path, task="detect")
        log.info(f"Muzzle detector loaded from '{path}'")
    except Exception as e:
        log.warning(f"Muzzle detector failed to load ({e}). Falling back to fixed crop.")
        _muzzle_model = None


def warmup_muzzle_detector() -> None:
    """Run a dummy inference so the first real request isn't slowed by lazy init."""
    if _muzzle_model is None:
        return
    try:
        _muzzle_model(np.zeros((224, 224, 3), dtype=np.uint8), verbose=False)
        log.info("Muzzle detector warmup complete.")
    except Exception as e:
        log.warning(f"Muzzle detector warmup failed: {e}")


def available() -> bool:
    """True if the detector is loaded and usable."""
    return _muzzle_model is not None


def detect_muzzle(img: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Locate the muzzle in a BGR image.

    Takes the HIGHEST-CONFIDENCE detection rather than the largest box — the
    opposite of detect_primary_animal()'s choice, and deliberately so. There
    is exactly one muzzle on the subject animal; extra detections are other
    animals' muzzles in the background, and a background animal that happens
    to be closer to the camera would win on size. Confidence is the better
    signal for "this is the muzzle we mean."

    Returns
    -------
    (x1, y1, x2, y2), or None if the detector isn't loaded or found nothing.
    """
    if _muzzle_model is None or img is None or img.size == 0:
        return None

    try:
        results = _muzzle_model(img, conf=MUZZLE_DETECT_CONF, verbose=False)[0]
    except Exception as e:
        log.warning(f"Muzzle detection failed: {e}")
        return None

    if len(results.boxes) == 0:
        return None

    best = max(results.boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = (int(round(float(v))) for v in best.xyxy[0])
    if x2 <= x1 or y2 <= y1:
        return None

    log.info(f"detect_muzzle: conf={float(best.conf[0]):.3f} box=({x1},{y1},{x2},{y2})")
    return x1, y1, x2, y2
