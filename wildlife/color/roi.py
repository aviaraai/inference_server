"""
roi.py — Handles Region of Interest (ROI) slicing of cattle body and muzzle images.
"""

from typing import Optional

import cv2
import numpy as np

# Optional: localize the actual animal via the pipeline's YOLO detector before
# sampling body color, instead of trusting a fixed frame-center crop. This
# import reaches outside wildlife/ into pipeline/, which only works when this
# package is running inside the full inference_server repo (not when wildlife/
# is used standalone, e.g. mounted alone in a container or imported from the
# separate Godhaar/Wildlife source repo) — degrade gracefully in that case,
# same pattern pipeline/color.py already uses in the opposite direction for
# optional wildlife availability.
try:
    from pipeline.yolo_crop import detect_primary_animal
except Exception:
    detect_primary_animal = None

try:
    from pipeline.muzzle_detect import detect_muzzle
except Exception:
    detect_muzzle = None

_LOCALIZED_BOX_PAD_PCT = 0.05  # small pad around the detected animal box
_MUZZLE_BOX_INSET_PCT = 0.08   # trim muzzle hair clipped by the detector box edges


def get_body_roi(img: np.ndarray) -> np.ndarray:
    """Extract the cattle body region for color sampling.

    Prefers a YOLO-localized crop of the actual animal, so color statistics
    come from real coat pixels instead of a fixed frame-center crop (which on
    a real photo can be mostly background/shadow — the subject rarely fills a
    fixed 70%x60% center box). Falls back to the fixed-percentage center crop
    when localization is unavailable (pipeline/YOLO not importable, or the
    model isn't loaded) or finds no animal in this specific photo.

    This deliberately does NOT segment foreground from background inside the
    box. An earlier version ran GrabCut here and it looked essential — without
    it, brown animals came back SPOTTED or BLACK. That turned out to be
    misattribution: the real cause was NEUTRAL_THRESHOLD sitting inside the
    brown-coat chroma distribution (see color_constants.py), and GrabCut was
    only nudging cluster centroids across an arbitrary line. Evidence it was
    never doing the job it appeared to: its accuracy was NON-MONOTONIC in
    resolution (6/7 correct at 384px but 5/7 at 512px), i.e. it was landing on
    the right answers by luck. With the threshold calibrated, GrabCut changes
    no label on the real photo set while costing ~50x the runtime
    (~6700ms/image vs ~130ms), so it was removed rather than tuned.
    """
    if img is None or img.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)

    localized = _get_localized_body_roi(img)
    if localized is not None:
        return localized

    return _get_fixed_body_roi(img)


def _get_fixed_body_roi(img: np.ndarray) -> np.ndarray:
    """Fixed-percentage center crop of the raw frame.

    Discards:
        Top 20% (often contains background, sky, ears, horns)
        Bottom 20% (often contains grass, legs, shadow)
        Left 15% (often contains environment)
        Right 15% (often contains environment)
    """
    h, w = img.shape[:2]
    y1, y2 = int(h * 0.20), int(h * 0.80)
    x1, x2 = int(w * 0.15), int(w * 0.85)
    return img[y1:y2, x1:x2]


def _get_localized_body_roi(img: np.ndarray) -> Optional[np.ndarray]:
    """Crop to the YOLO-detected animal box."""
    if detect_primary_animal is None:
        return None

    box = detect_primary_animal(img)
    if box is None:
        return None

    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = int((x2 - x1) * _LOCALIZED_BOX_PAD_PCT)
    pad_y = int((y2 - y1) * _LOCALIZED_BOX_PAD_PCT)
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def get_muzzle_roi(img: np.ndarray) -> np.ndarray:
    """Extract the muzzle skin region for color sampling.

    Prefers a detector-localized crop of the actual muzzle. This matters more
    than it looks: main.py passes `crop_cattle()`'s output here, which is the
    WHOLE-ANIMAL box, so the fixed center crop below was sampling the animal's
    neck/chest and reporting coat color as muzzle color.

    Falls back to the fixed center crop when the detector is unavailable
    (model not deployed, `pipeline/` not importable) or finds no muzzle in
    this particular photo.
    """
    if img is None or img.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)

    localized = _get_localized_muzzle_roi(img)
    if localized is not None:
        return localized

    return _get_fixed_muzzle_roi(img)


def _get_fixed_muzzle_roi(img: np.ndarray) -> np.ndarray:
    """Fixed-percentage center crop.

    Discards:
        Top 25% (often contains upper snout hair/nostril boundaries)
        Bottom 15% (often contains lower jaw/lips)
        Left 20% (often contains cheek hair)
        Right 20% (often contains cheek hair)
    """
    h, w = img.shape[:2]
    y1, y2 = int(h * 0.25), int(h * 0.85)
    x1, x2 = int(w * 0.20), int(w * 0.80)
    return img[y1:y2, x1:x2]


def _get_localized_muzzle_roi(img: np.ndarray) -> Optional[np.ndarray]:
    """Crop to the detected muzzle box, inset slightly to stay on skin.

    The detector's box tracks the nose pad closely, so only a small inset is
    needed — just enough to drop the surrounding muzzle hair that the box
    edges clip. No GrabCut here, unlike the body ROI: a tight muzzle box is
    almost entirely skin already, so there is no background to segment away
    and running it would only risk eating real nostril/lip pixels.
    """
    if detect_muzzle is None:
        return None

    box = detect_muzzle(img)
    if box is None:
        return None

    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    inset_x, inset_y = int(bw * _MUZZLE_BOX_INSET_PCT), int(bh * _MUZZLE_BOX_INSET_PCT)
    crop = img[y1 + inset_y:y2 - inset_y, x1 + inset_x:x2 - inset_x]

    return crop if crop.size > 0 else None
