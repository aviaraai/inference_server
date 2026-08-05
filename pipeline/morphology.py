"""
pipeline/morphology.py — Horn/ear proportion extraction, pluggable interface.

Architecture mirrors pipeline/color.py:
    MorphologyExtractor (ABC)
        └── RuleBasedMorphologyExtractor   ← classical CV heuristic (v1)
            (future: AIMorphologyExtractor, a trained keypoint model)

Returns SCALE-INVARIANT RATIOS, never an absolute length. A phone photo
carries no distance/scale reference (no ruler, no known-distance marker,
no depth camera) — the same horn measures differently depending on how far
the phone was held. Ratios (horn/ear extent ÷ the animal's own crop width)
are comparable animal-to-animal, which is what duplicate-detection/search
actually need — they don't need real centimeters.

⚠️ v1 is an UNVALIDATED HEURISTIC — same caveat CLAUDE.md already documents
for the muzzle-color gate ("near-useless for black cattle... label is not
discriminative"). There is no labeled horn/ear dataset and no trained
keypoint model anywhere in this project (checked both this repo's bundled
wildlife/ and the full Godhaar/Wildlife source — neither has one). This
extractor works off crop_cattle's whole-animal bbox: it isolates the top
band of that crop (same region roi.py already calls out as "often contains
... ears, horns" when EXCLUDING it for body color), finds the largest
silhouette contour there via Canny edges, and measures that contour's
extremities. It has NOT been checked against real photos the way the blur
threshold was (see CLAUDE.md's blur calibration story) — there was no
labeled ground truth available to check it against. Confidence is capped at
0.6 for exactly this reason: never claim more certainty than a heuristic
with zero validation data has earned.

⚠️ A LOW/ZERO READING IS NOT EVIDENCE OF "NO HORNS." A single front photo
cannot tell "this animal has no horns / is polled" (common in Indian
cattle — genuinely hornless or dehorned animals are routine, not rare)
apart from "this animal has horns but they're not visible from this
angle" (backward-curving horns, horns tucked down, occlusion by an ear or
another animal, poor lighting on the crown). Both cases produce the same
thing here: a small `horn_length_ratio` and/or a non-`OK` `status`. This
is a structural limit of single-2D-photo silhouette analysis, not a bug —
telling those two cases apart would need either a horn-specific detector
(none exists — see the docstring above) or a second photo angle (a
capture-flow change, out of scope for this service). Treat any reading
here as "no confirmed horn visible in this photo," never as "confirmed no
horns" — don't use it as negative evidence anywhere.
"""

import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np

log = logging.getLogger("godhaar.morphology")

# Top fraction of the whole-animal crop treated as the head/horn/ear band.
# roi.py excludes the top 20% for body color for the same reason we WANT
# it here; widened slightly to reduce the chance of clipping a tall horn.
HEAD_BAND_FRACTION = 0.35

# Below this, the detected silhouette is more likely background/rope noise
# than the animal — treated as a failed reading, not a low-confidence one.
MIN_CONTOUR_AREA_FRACTION = 0.04

# Heuristic confidence is deliberately capped well below 1.0 — this is an
# unvalidated rule-based estimate, not a calibrated classifier.
MAX_CONFIDENCE = 0.6


def _reading(status: str, reason: str, horn_length_ratio: float = 0.0,
             ear_span_ratio: float = 0.0, confidence: float = 0.0) -> dict:
    """Every code path returns through here so `status`/`reason` are never
    missing — a caller must always be able to tell WHY a reading is what
    it is, not just see a bare number. See module docstring on why a
    silent 0.0 is actively misleading for this feature specifically.
    """
    return {
        "horn_length_ratio": round(float(horn_length_ratio), 4),
        "ear_span_ratio": round(float(ear_span_ratio), 4),
        "confidence": round(float(confidence), 3),
        "status": status,
        "reason": reason,
    }


def average_readings(readings: list[dict]) -> dict:
    """Combine multiple per-image readings (e.g. registration's 2 front
    photos) into one, weighting by each reading's own confidence so a
    failed (confidence=0) reading doesn't silently drag a good one toward
    zero. Returned confidence is the mean of the inputs' — deliberately not
    the max — so "one of two photos failed" still reads as less certain
    than "both succeeded."
    """
    total_conf = sum(r["confidence"] for r in readings)
    ok_count = sum(1 for r in readings if r["status"] == "OK")

    if total_conf <= 0:
        # Nothing usable from either photo — surface the first failure's
        # own status/reason rather than inventing a generic one; if both
        # failed differently, at least one real reason is more useful than
        # a vague combined label.
        first = readings[0]
        return _reading(first["status"], first["reason"])

    horn = sum(r["horn_length_ratio"] * r["confidence"] for r in readings) / total_conf
    ear = sum(r["ear_span_ratio"] * r["confidence"] for r in readings) / total_conf
    avg_conf = total_conf / len(readings)

    if ok_count == len(readings):
        status, reason = "OK", ""
    else:
        status = "PARTIAL"
        reason = f"{len(readings) - ok_count}/{len(readings)} photos produced no reading"

    return _reading(status, reason, horn, ear, avg_conf)


def _largest_contour(gray: np.ndarray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


class MorphologyExtractor(ABC):
    """Abstract interface for extracting horn/ear proportions from a photo.

    Implementations must return scale-invariant ratios, never absolute
    lengths — see module docstring. This allows swapping the rule-based
    heuristic for a trained keypoint model later without touching callers.
    """

    @abstractmethod
    def extract(self, front_img_bgr: np.ndarray) -> dict:
        """Extract horn/ear proportions from a BGR front-facing photo.

        Returns
        -------
        {"horn_length_ratio": float, "ear_span_ratio": float,
         "confidence": float, "status": str, "reason": str}

        `status` is one of "OK", "INVALID_IMAGE", "NO_ANIMAL_DETECTED",
        "NO_CLEAR_SILHOUETTE" (see RuleBasedMorphologyExtractor). Only
        "OK" means the ratios are a real reading — every other status
        means they're the zero default and must not be treated as data.
        """
        ...


class RuleBasedMorphologyExtractor(MorphologyExtractor):
    """Classical-CV heuristic: whole-animal crop → head band → largest
    edge contour → extremity distances, normalized by crop width.

    Gracefully degrades to a zero reading with an explanatory status on any
    detection failure or unexpected image shape — never raises. Same
    fail-open contract as RuleBasedColorExtractor, since this must never be
    able to block a registration or search on its own.
    """

    def extract(self, front_img_bgr: np.ndarray) -> dict:
        # Local import: avoids a hard dependency/circular-import between
        # this module and yolo_crop at import time, matching how color.py
        # keeps its Wildlife import scoped to __init__ rather than top-level.
        from pipeline.yolo_crop import crop_cattle

        if front_img_bgr is None or front_img_bgr.size == 0:
            return _reading("INVALID_IMAGE", "empty or undecodable image")

        try:
            crop, det_status, det_conf = crop_cattle(front_img_bgr)
            if crop is None:
                return _reading(
                    "NO_ANIMAL_DETECTED",
                    f"crop_cattle found no animal (yolo_status={det_status})",
                )

            h, w = crop.shape[:2]
            if h < 20 or w < 20:
                return _reading(
                    "NO_ANIMAL_DETECTED",
                    f"detected crop too small to analyze ({w}x{h}px)",
                )

            band_h = max(1, int(h * HEAD_BAND_FRACTION))
            head_band = crop[:band_h, :]
            gray = cv2.cvtColor(head_band, cv2.COLOR_BGR2GRAY)

            contour = _largest_contour(gray)
            if contour is None:
                return _reading(
                    "NO_CLEAR_SILHOUETTE",
                    "no edge contour found in head band — could be a "
                    "backward/occluded horn, flat lighting, or a "
                    "genuinely hornless animal; not distinguishable here",
                )

            band_area = band_h * w
            contour_area = cv2.contourArea(contour)
            fill_ratio = contour_area / band_area if band_area else 0.0
            if fill_ratio < MIN_CONTOUR_AREA_FRACTION:
                return _reading(
                    "NO_CLEAR_SILHOUETTE",
                    f"head-band contour too small to trust "
                    f"(fill={fill_ratio:.3f}, need >={MIN_CONTOUR_AREA_FRACTION}) — "
                    f"could be a backward/occluded horn, flat lighting, or a "
                    f"genuinely hornless animal; not distinguishable here",
                )

            pts = contour.reshape(-1, 2)
            top_y = int(pts[:, 1].min())
            left_x = int(pts[:, 0].min())
            right_x = int(pts[:, 0].max())

            horn_length_px = band_h - top_y  # extent up from the head-band base
            ear_span_px = right_x - left_x   # left-right extent within the band

            # Capped, never a confident-sounding number — see module docstring.
            confidence = float(np.clip(fill_ratio * 2.0, 0.0, MAX_CONFIDENCE))

            return _reading(
                "OK", "",
                horn_length_ratio=horn_length_px / w,
                ear_span_ratio=ear_span_px / w,
                confidence=confidence,
            )
        except Exception as e:
            log.warning(f"Morphology extraction failed: {e}")
            return _reading("INVALID_IMAGE", f"unexpected error: {e}")
