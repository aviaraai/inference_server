"""
pipeline/muzzle_detect.py — Muzzle localization for color sampling.

A dedicated muzzle detector (best_float16.tflite, a YOLOv8n exported to
TFLite) was originally planned here, to localize the muzzle before reading
its color — `crop_cattle()` (pipeline/yolo_crop.py) only finds the WHOLE
ANIMAL, so without localization, muzzle color was sampled from the animal's
neck/chest and reported as coat color instead (e.g. PINK for an obviously
black muzzle on a brown-hided cow).

That model is not deployed and will not be — confirmed with the CTO,
2026-08-17: `appstorage/Models/muzzle_detect/` does not exist on the real
inference host and never will. The fixed center-crop fallback below
(wildlife/color/roi.py's `_get_fixed_muzzle_roi`) is the actual, permanent
muzzle-color path, not a degraded stand-in for a model that's coming later.
This module is kept only so its callers (main.py's startup sequence,
wildlife/color/roi.py) don't need restructuring — every function here is an
intentional no-op, not an attempt-then-fallback.
"""

from typing import Optional

import numpy as np


def load_muzzle_detector(model_path: Optional[str] = None) -> None:
    """No-op — see module docstring. Kept so main.py's startup sequence
    doesn't need editing; does not attempt to load anything."""
    return


def warmup_muzzle_detector() -> None:
    """No-op — see module docstring."""
    return


def detect_muzzle(img: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Always None — see module docstring. Callers (roi.py's
    `_get_localized_muzzle_roi`) already treat a None return as "no
    localization possible, use the fixed center crop," which is exactly
    the intended behavior now."""
    return None
