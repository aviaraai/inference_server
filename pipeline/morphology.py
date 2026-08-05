"""
pipeline/morphology.py — Horn presence/shape extraction, pluggable interface.

Architecture mirrors pipeline/color.py:
    MorphologyExtractor (ABC)
        └── RuleBasedMorphologyExtractor   ← classical CV heuristic (v1)
            (future: AIMorphologyExtractor, a trained keypoint model)

v1 returns CATEGORICAL fields, not a measurement — `has_horns` (bool) and
`horn_shape` (a hardcoded enum, HornShape below), the same pattern
pipeline/color.py already uses for body/muzzle color labels. An earlier
version of this file returned continuous ratios (horn_length_ratio,
ear_span_ratio) instead; that was replaced on the user's explicit
instruction — presence + shape is what's wanted, not a length/width number.

⚠️ v1 is an UNVALIDATED HEURISTIC — same caveat CLAUDE.md already documents
for the muzzle-color gate ("near-useless for black cattle... label is not
discriminative"). There is no labeled horn dataset and no trained
keypoint/shape model anywhere in this project (checked both this repo's
bundled wildlife/ and the full Godhaar/Wildlife source — neither has one).
Shape classification (STRAIGHT vs CURVED) is a straightness measurement on
the detected silhouette's edge contour — a rougher, less validated guess
than presence detection is, because there is no ground truth to calibrate
the straightness cutoff against. Confidence is capped at 0.6 for exactly
this reason: never claim more certainty than a heuristic with zero
validation data has earned.

⚠️ `has_horns=False` IS NOT "CONFIRMED NO HORNS." A single front photo
cannot tell "this animal has no horns / is polled" (common in Indian
cattle — genuinely hornless or dehorned animals are routine, not rare)
apart from "this animal has horns but they're not visible from this
angle" (backward-curving horns, horns tucked down, occlusion by an ear or
another animal, poor lighting on the crown). Both cases look the same
here: no protrusion found above the head silhouette. This is a structural
limit of single-2D-photo silhouette analysis, not a bug — telling those
two cases apart would need either a horn-specific detector (none exists —
see above) or a second photo angle (a capture-flow change, out of scope
for this service). Treat `has_horns=False` as "no horn confirmed visible
in this photo," never as "confirmed hornless" — don't use it as negative
evidence anywhere.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum

import cv2
import numpy as np

log = logging.getLogger("godhaar.morphology")


class HornShape(str, Enum):
    """Only meaningful when has_horns=True. There is deliberately no NONE
    member here — "no horn" is already fully expressed by has_horns=False;
    a separate NONE shape would just be the same fact said twice. When
    there's no horn (or no reading at all), horn_shape is plain `None`,
    not a member of this enum.
    """
    STRAIGHT = "STRAIGHT"  # protrusion found, low curvature
    CURVED = "CURVED"      # protrusion found, notable curvature
    UNKNOWN = "UNKNOWN"    # a horn was found but its shape couldn't be classified


# Top fraction of the whole-animal crop treated as the head/horn/ear band.
# roi.py excludes the top 20% for body color for the same reason we WANT
# it here; widened slightly to reduce the chance of clipping a tall horn.
HEAD_BAND_FRACTION = 0.35

# Below this, the detected silhouette is more likely background/rope noise
# than the animal — treated as a failed reading, not "no horns."
MIN_CONTOUR_AREA_FRACTION = 0.04

# Minimum protrusion above the head-band base (÷ crop width) to call it a
# horn at all, vs. no protrusion (has_horns=False). Picked as a small,
# conservative fraction — not calibrated against real photos, same caveat
# as everything else in this module.
MIN_HORN_PX_FRACTION = 0.03

# Straightness cutoff for STRAIGHT vs CURVED: normalized RMS perpendicular
# distance of the horn-region's contour points from a fitted straight line.
# A rough guess, not a calibrated threshold — see module docstring.
STRAIGHTNESS_THRESHOLD = 0.10

# Heuristic confidence is deliberately capped well below 1.0 — this is an
# unvalidated rule-based estimate, not a calibrated classifier.
MAX_CONFIDENCE = 0.6


def _reading(status: str, reason: str, has_horns=None, horn_shape=None,
             confidence: float = 0.0) -> dict:
    """Every code path returns through here so `status`/`reason` are never
    missing — a caller must always be able to tell WHY a reading is what
    it is, not just see a bare value. See module docstring on why
    has_horns=False is not the same claim as "confirmed no horns."
    """
    return {
        "has_horns": has_horns,
        "horn_shape": horn_shape.value if isinstance(horn_shape, HornShape) else horn_shape,
        "confidence": round(float(confidence), 3),
        "status": status,
        "reason": reason,
    }


def average_readings(readings: list[dict]) -> dict:
    """Combine multiple per-image readings (e.g. registration's 2 front
    photos). Unlike color, this never blocks/raises on disagreement — but
    unlike the old continuous-ratio version, a categorical disagreement
    can't be silently blended either (there's no such thing as "the
    average of STRAIGHT and CURVED"). So:
      - both OK and agree            → that value, confidence = mean
      - both OK, disagree            → status INCONSISTENT, no value
        (returning either single reading would misrepresent the other)
      - only one produced a reading  → that one, status PARTIAL
      - neither produced a reading   → propagate the first failure's
        status/reason
    """
    ok = [r for r in readings if r["status"] == "OK"]

    if not ok:
        first = readings[0]
        return _reading(first["status"], first["reason"])

    if len(ok) < len(readings):
        best = max(ok, key=lambda r: r["confidence"])
        avg_conf = sum(r["confidence"] for r in ok) / len(readings)
        return _reading(
            "PARTIAL",
            f"{len(readings) - len(ok)}/{len(readings)} photos produced no reading",
            has_horns=best["has_horns"],
            horn_shape=best["horn_shape"],
            confidence=avg_conf,
        )

    distinct = {(r["has_horns"], r["horn_shape"]) for r in ok}
    if len(distinct) > 1:
        summary = "; ".join(f"{r['has_horns']}/{r['horn_shape']}" for r in ok)
        return _reading("INCONSISTENT", f"photos disagreed: {summary}")

    avg_conf = sum(r["confidence"] for r in ok) / len(ok)
    return _reading("OK", "", has_horns=ok[0]["has_horns"], horn_shape=ok[0]["horn_shape"], confidence=avg_conf)


def _largest_contour(gray: np.ndarray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _classify_shape(contour: np.ndarray, top_y: int, horn_px: float) -> tuple:
    """Fit a line to the contour points in the horn region (the upper
    portion of the head-band contour) and measure how far they deviate
    from it, normalized by the horn's own extent. Low deviation → a
    roughly straight edge; higher deviation → curved. This is a rough
    proxy, not a validated classifier — see module docstring.

    Returns (HornShape, straightness_value) for logging/reason purposes.
    """
    pts = contour.reshape(-1, 2).astype(np.float32)
    region = pts[pts[:, 1] <= top_y + horn_px * 0.7]
    if len(region) < 5:
        return HornShape.UNKNOWN, None

    vx, vy, x0, y0 = cv2.fitLine(region, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    dists = np.abs((region[:, 0] - x0) * vy - (region[:, 1] - y0) * vx)
    straightness = float(dists.std() / max(horn_px, 1.0))

    shape = HornShape.STRAIGHT if straightness < STRAIGHTNESS_THRESHOLD else HornShape.CURVED
    return shape, straightness


class MorphologyExtractor(ABC):
    """Abstract interface for extracting horn presence/shape from a photo.

    Implementations must return categorical fields (has_horns, horn_shape),
    not a measurement — see module docstring. This allows swapping the
    rule-based heuristic for a trained model later without touching callers.
    """

    @abstractmethod
    def extract(self, front_img_bgr: np.ndarray) -> dict:
        """Extract horn presence/shape from a BGR front-facing photo.

        Returns
        -------
        {"has_horns": bool | None, "horn_shape": str | None,
         "confidence": float, "status": str, "reason": str}

        `status` is one of "OK", "INVALID_IMAGE", "NO_ANIMAL_DETECTED",
        "NO_CLEAR_SILHOUETTE". Only "OK" means has_horns/horn_shape are a
        real reading — every other status means both are None and MUST
        NOT be treated as "no horns."
        """
        ...


class RuleBasedMorphologyExtractor(MorphologyExtractor):
    """Classical-CV heuristic: whole-animal crop → head band → largest
    edge contour → protrusion above the band (has_horns) → straightness
    of that protrusion's edge (horn_shape).

    Gracefully degrades to a null reading with an explanatory status on any
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
            horn_px = band_h - top_y  # extent up from the head-band base

            # Capped, never a confident-sounding number — see module docstring.
            confidence = float(np.clip(fill_ratio * 2.0, 0.0, MAX_CONFIDENCE))

            if horn_px < MIN_HORN_PX_FRACTION * w:
                return _reading(
                    "OK",
                    f"no protrusion above head silhouette (horn_px_frac="
                    f"{horn_px / w:.3f}, need >={MIN_HORN_PX_FRACTION}) — "
                    f"see module docstring: NOT confirmed hornless, just "
                    f"none visible in this photo",
                    has_horns=False,
                    horn_shape=None,
                    confidence=confidence,
                )

            shape, straightness = _classify_shape(contour, top_y, horn_px)
            reason = "" if straightness is None else f"straightness={straightness:.3f}"

            return _reading(
                "OK", reason,
                has_horns=True,
                horn_shape=shape,
                confidence=confidence,
            )
        except Exception as e:
            log.warning(f"Morphology extraction failed: {e}")
            return _reading("INVALID_IMAGE", f"unexpected error: {e}")
