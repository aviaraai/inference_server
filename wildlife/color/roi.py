"""
roi.py — Handles Region of Interest (ROI) slicing of cattle body and muzzle images.
"""

import numpy as np


def get_body_roi(img: np.ndarray) -> np.ndarray:
    """Extract the central body region from a front wide-angle image.

    Discards:
        Top 20% (often contains background, sky, ears, horns)
        Bottom 20% (often contains grass, legs, shadow)
        Left 15% (often contains environment)
        Right 15% (often contains environment)
    """
    if img is None or img.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)

    h, w = img.shape[:2]
    y1, y2 = int(h * 0.20), int(h * 0.80)
    x1, x2 = int(w * 0.15), int(w * 0.85)

    return img[y1:y2, x1:x2]


def get_muzzle_roi(img: np.ndarray) -> np.ndarray:
    """Extract the core muzzle skin pattern from a cropped muzzle image.

    Discards:
        Top 25% (often contains upper snout hair/nostril boundaries)
        Bottom 15% (often contains lower jaw/lips)
        Left 20% (often contains cheek hair)
        Right 20% (often contains cheek hair)
    """
    if img is None or img.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)

    h, w = img.shape[:2]
    y1, y2 = int(h * 0.25), int(h * 0.85)
    x1, x2 = int(w * 0.20), int(w * 0.80)

    return img[y1:y2, x1:x2]
