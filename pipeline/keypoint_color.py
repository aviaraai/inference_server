"""
pipeline/keypoint_color.py — keypoint-anchored forehead color, additive only.

wildlife/color/body_color.py's classify_body_color() samples the ENTIRE
YOLO-localized animal bounding box (see roi.py's get_body_roi — it explicitly
does not segment foreground from background inside the box, so legs, tail,
horns and edge background all contribute pixels). This module is a second,
independent color reading anchored to the pose model's eye keypoints instead:
small patches on the forehead, scaled by inter-eye distance so the sampled
region is the same relative anatomical area regardless of camera distance or
crop tightness.

ADDITIVE ONLY. This is computed and returned for a caller to store alongside
body_color/muzzle_color -- it is not consumed by /register's duplicate check,
/search's decision engine, or wired into go-apiserver's decision.go in any
way. See CLAUDE.md before changing that.

Self-contained by design: does its own crop_cattle() + predict_keypoints()
call rather than threading keypoints out of extract_face_geometry (one extra
pose-model inference per front photo). This matches the existing convention
in this codebase -- get_body_roi's detect_primary_animal(), morphology.py's
extract(), and pose.py's extract_face_geometry() each independently re-run
their own detection rather than sharing state across extractors.

Confidence rule (same standard as geometry.py's horn-presence cascade): if
EITHER eye keypoint is missing or below pipeline.pose.KEYPOINT_CONF_THRESHOLD,
the result is UNKNOWN -- this never falls back to bounding-box sampling and
reports a worse number as if it were a good one. A missing/low-confidence
muzzle or crest keypoint only affects which side of the eye-line "forehead"
resolves to, and whether the optional third (crest) patch is available -- it
does not gate the whole result to UNKNOWN.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from typing import Optional

import numpy as np

from pipeline.geometry import CREST, KPT_NAMES, L_EYE, MUZZLE, R_EYE
from pipeline.pose import KEYPOINT_CONF_THRESHOLD, pose_available, predict_keypoints

log = logging.getLogger("godhaar.keypoint_color")

# ── Reach into wildlife/color/ for the LAB classification primitives ────────
# wildlife/color/ is a sibling top-level directory, not a dotted-importable
# package from here (same situation pipeline/color.py's RuleBasedColorExtractor
# already solves) -- reuse its exact sys.path-insertion candidates rather than
# inventing a second list that could disagree with it about where wildlife/
# lives. Done at import time here (not lazily in a constructor, unlike
# RuleBasedColorExtractor) so this module works whether or not a
# RuleBasedColorExtractor has been constructed yet.
_WILDLIFE_COLOR_AVAILABLE = False
try:
    for _search_path in (
        os.path.join(os.path.dirname(__file__), "..", "wildlife"),  # bundled in repo
        "/app/wildlife",  # Docker (COPY . . puts it here)
    ):
        _abs_path = os.path.abspath(_search_path)
        if os.path.isdir(_abs_path) and _abs_path not in sys.path:
            sys.path.insert(0, _abs_path)

    from color import color_constants as C
    from color.body_color import _classify_lab
    from color.quality import check_roi_quality
    from color.utils import (
        aggregate_clusters_by_label,
        bgr_to_lab,
        calculate_median_lab,
        extract_dominant_lab_features,
    )

    _WILDLIFE_COLOR_AVAILABLE = True
except Exception as e:
    log.warning(f"keypoint_color: wildlife/color module not available ({e}); signal disabled.")


class KeypointColorStatus:
    OK = "OK"            # eyes confident, forehead_low sampled plus at least one more patch
    PARTIAL = "PARTIAL"  # eyes confident, only the primary (forehead_low) patch sampled
    UNKNOWN = "UNKNOWN"  # eye keypoints missing/low-confidence, or nothing else usable


# ── Patch geometry, in units of inter-eye distance ("the ruler") ────────────
# Starting guesses, not validated against a labeled dataset -- same caveat as
# CONTEXT_MULTIPLIER in muzzleCrop.ts (Telangana app) or
# SUBJECT_DOMINANCE_RATIO here: reasoned defaults, revisit once real
# leave-one-out comparison data exists (see step 3 of this change).
#
# Offsets are measured from the eye-line midpoint, along the resolved
# forehead direction (see _forehead_direction). 0.35 sits just above the
# brow; 0.85 is high on the forehead, approaching the poll/crest on a
# typical cattle face proportions.
FOREHEAD_LOW_OFFSET_RATIO = 0.35
FOREHEAD_HIGH_OFFSET_RATIO = 0.85

# Full patch side length, as a fraction of inter-eye distance. 0.45 keeps the
# patch comfortably inside the forehead on typical proportions without
# overlapping the eyes or ear-line at either offset above.
PATCH_SIZE_RATIO = 0.45

# Below this side length, a patch is too small to trust a LAB read from (a
# few pixels of JPEG-compressed noise), regardless of what quality gating
# would otherwise say. Comparable in spirit to MIN_MUZZLE_ROI_WIDTH/HEIGHT in
# color_constants.py (a detector-localized ROI is legitimately much smaller
# than a whole-body crop) but smaller still, since a forehead patch is a
# fraction of even that.
MIN_PATCH_SIDE_PX = 12

# check_roi_quality's own defaults (MIN_ROI_WIDTH/HEIGHT = 120) assume a
# whole-body or whole-muzzle ROI; a keypoint patch is legitimately far
# smaller, so it's gated on size separately above and given a much lower
# floor here purely to keep check_roi_quality's own size check from being the
# thing that rejects it.
_PATCH_QUALITY_MIN_SIDE_PX = 6


def _empty_result(reason: str) -> dict:
    return {
        "status": KeypointColorStatus.UNKNOWN,
        "conf_threshold": KEYPOINT_CONF_THRESHOLD,
        "inter_eye_distance_px": None,
        "label": "UNKNOWN",
        "confidence": 0.0,
        "patches": {"forehead_low": None, "forehead_high": None, "crest": None},
        "reasons": {"status": reason},
    }


def _forehead_direction(
    xy: np.ndarray, conf: np.ndarray, eye_mid: np.ndarray, inter_eye_distance_px: float
) -> np.ndarray:
    """Resolve which of the two perpendiculars to the eye-line points toward
    the forehead (as opposed to the muzzle/chin).

    Only affects WHICH side of the eye-line the patches land on -- never
    gates UNKNOWN; that gate is the eye keypoints alone (see the module
    docstring). Preference order, most to least reliable (PCK@0.1 from
    pipeline/pose.py's PROVENANCE numbers):

      1. muzzle (.931): forehead is the perpendicular pointing AWAY from it.
      2. crest (.778): forehead is the perpendicular pointing TOWARD it --
         the crest keypoint sits on the forehead/poll between the horns.
      3. plain image "up" (smaller y): front photos are captured with the
         phone held upright per the capture UI, so this is a reasonable
         last resort when neither of the above cleared confidence -- not a
         confident anatomical read, just the least-bad default.
    """
    eye_axis = (xy[L_EYE] - xy[R_EYE]) / inter_eye_distance_px
    perp_a = np.array([-eye_axis[1], eye_axis[0]])
    perp_b = -perp_a

    if conf[MUZZLE] >= KEYPOINT_CONF_THRESHOLD:
        muzzle_vec = xy[MUZZLE] - eye_mid
        return perp_a if np.dot(perp_a, muzzle_vec) < 0 else perp_b
    if conf[CREST] >= KEYPOINT_CONF_THRESHOLD:
        crest_vec = xy[CREST] - eye_mid
        return perp_a if np.dot(perp_a, crest_vec) > 0 else perp_b
    return perp_a if perp_a[1] < perp_b[1] else perp_b


def _sample_patch(
    crop_bgr: np.ndarray,
    center: np.ndarray,
    half_side: float,
    reason_key: str,
    reasons: dict[str, str],
) -> Optional[dict]:
    """Crop a square patch around `center`, quality-gate it, and classify its
    median LAB the same way classify_body_color does post-ROI (median LAB +
    a 2-cluster label vote, since a small anatomical patch needs far fewer
    clusters than a whole-body ROI to separate lit/shadowed sub-regions of
    the SAME patch of skin/coat).

    Returns None (with `reasons[reason_key]` set) on any failure -- clipped
    out of frame, too small, or failing check_roi_quality. Never returns a
    reading it isn't confident in.
    """
    h, w = crop_bgr.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    x1, y1 = int(round(cx - half_side)), int(round(cy - half_side))
    x2, y2 = int(round(cx + half_side)), int(round(cy + half_side))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    patch = crop_bgr[y1:y2, x1:x2]

    side = min(x2 - x1, y2 - y1)
    if side < MIN_PATCH_SIDE_PX or patch.size == 0:
        reasons[reason_key] = (
            f"patch too small ({side}px after clipping to frame, need >= "
            f"{MIN_PATCH_SIDE_PX}px) -- likely off-frame or ruler too short"
        )
        return None

    is_ok, quality_reason = check_roi_quality(
        patch, min_width=_PATCH_QUALITY_MIN_SIDE_PX, min_height=_PATCH_QUALITY_MIN_SIDE_PX
    )
    if not is_ok:
        reasons[reason_key] = f"LOW_QUALITY: {quality_reason}"
        return None

    img_lab = bgr_to_lab(patch)
    median_lab = calculate_median_lab(img_lab)
    features = extract_dominant_lab_features(img_lab, k=2)
    ranked = aggregate_clusters_by_label(features["clusters"], _classify_lab, centrality_ratio_min=0.0)
    if not ranked:
        reasons[reason_key] = "NO_COLOR_CLUSTERS"
        return None

    primary_label, primary_weight = ranked[0]
    total_weight = sum(weight for _, weight in ranked)
    confidence = primary_weight / total_weight if total_weight > 0 else 0.0
    # Same boundary-uncertainty discount classify_body_color applies: a patch
    # sitting right on the neutral/chromatic edge could go either way on the
    # next photo.
    chroma = math.sqrt(median_lab[1] ** 2 + median_lab[2] ** 2)
    if abs(chroma - C.NEUTRAL_THRESHOLD) < 3.0:
        confidence *= 0.7

    return {
        "label": primary_label,
        "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 4),
        "median_lab": median_lab,
        "center_px": [round(cx, 1), round(cy, 1)],
        "size_px": side,
    }


def extract_keypoint_forehead_color(front_img_bgr: np.ndarray) -> dict:
    """Additive, keypoint-anchored alternative to body_color.py's whole-bbox
    sampling. See the module docstring for the confidence rule and why this
    is self-contained rather than sharing state with extract_face_geometry.

    Returns
    -------
    dict:
        status : "OK" / "PARTIAL" / "UNKNOWN"
        conf_threshold : the keypoint confidence floor used (pipeline.pose's)
        inter_eye_distance_px : the ruler, or None if UNKNOWN
        label, confidence : the PRIMARY (forehead_low) patch's reading --
            mirrors body_color/muzzle_color's flat {label, confidence} shape
            so it's directly comparable at a glance
        patches : {"forehead_low": ..., "forehead_high": ..., "crest": ...},
            each None or {label, confidence, median_lab, center_px, size_px}
            -- the multi-patch structure a future solid-vs-patterned check
            would read
        reasons : UNKNOWN/failure causes, keyed by field name
    """
    if front_img_bgr is None or getattr(front_img_bgr, "size", 0) == 0:
        return _empty_result("empty or undecodable image")

    if not _WILDLIFE_COLOR_AVAILABLE:
        return _empty_result("wildlife/color module not available")

    if not pose_available():
        return _empty_result("pose model not loaded (POSE_MODEL_PATH unset or file missing)")

    try:
        # Local import: same reason color.py / morphology.py / pose.py scope
        # theirs -- avoid an import-time cycle with yolo_crop.
        from pipeline.yolo_crop import crop_cattle

        crop, det_status, _det_conf = crop_cattle(front_img_bgr)
        if crop is None:
            return _empty_result(f"crop_cattle produced no head crop (yolo_status={det_status})")

        kpts = predict_keypoints(crop)
        if kpts is None:
            return _empty_result("pose model detected no face in the head crop")

        xy = kpts[:, :2]
        conf = kpts[:, 2]
        finite = np.isfinite(xy).all(axis=1) & np.isfinite(conf)
        present = (conf >= KEYPOINT_CONF_THRESHOLD) & finite

        if not (present[R_EYE] and present[L_EYE]):
            missing = [KPT_NAMES[i] for i in (R_EYE, L_EYE) if not present[i]]
            return _empty_result(
                f"eye keypoint(s) below {KEYPOINT_CONF_THRESHOLD}: {missing} -- "
                f"no inter-eye ruler, cannot anchor any patch"
            )

        inter_eye_distance_px = float(np.hypot(*(xy[L_EYE] - xy[R_EYE])))
        if inter_eye_distance_px <= 0:
            return _empty_result("both eyes detected but coincide (zero inter-eye distance)")

        eye_mid = (xy[R_EYE] + xy[L_EYE]) / 2.0
        forehead_dir = _forehead_direction(xy, conf, eye_mid, inter_eye_distance_px)
        half_side = max(1.0, PATCH_SIZE_RATIO * inter_eye_distance_px / 2.0)

        reasons: dict[str, str] = {}
        patches = {
            "forehead_low": _sample_patch(
                crop, eye_mid + forehead_dir * (FOREHEAD_LOW_OFFSET_RATIO * inter_eye_distance_px),
                half_side, "forehead_low", reasons,
            ),
            "forehead_high": _sample_patch(
                crop, eye_mid + forehead_dir * (FOREHEAD_HIGH_OFFSET_RATIO * inter_eye_distance_px),
                half_side, "forehead_high", reasons,
            ),
            "crest": None,
        }
        if present[CREST]:
            patches["crest"] = _sample_patch(crop, xy[CREST], half_side, "crest", reasons)
        else:
            reasons["crest"] = (
                f"crest keypoint below {KEYPOINT_CONF_THRESHOLD} -- optional third patch unavailable"
            )

        primary = patches["forehead_low"]
        other_sampled = sum(1 for k, p in patches.items() if k != "forehead_low" and p is not None)

        # NOTE for future consumers: label/confidence below are ONLY
        # forehead_low's reading, flattened for an at-a-glance comparison
        # against body_color. On a genuinely patterned coat the OTHER
        # patches can disagree with it in a way that matters -- confirmed on
        # a real photo (model_training/pose_model's animal_584: a real
        # black-and-white animal with a white forehead blaze) where
        # forehead_low read BLACK(0.65) while forehead_high and crest both
        # read WHITE(~0.85), correctly reflecting the blaze sitting just
        # above where forehead_low sampled. Anything that wants real
        # accuracy on patterned/spotted animals -- not just solid coats --
        # must read `patches`, not this flattened label. See
        # combine_keypoint_color below for the same flattening on the
        # cross-photo combined reading, which is what actually reaches
        # RegisterResponse.
        if primary is None:
            # Eyes were confident but the primary patch itself failed (out of
            # frame at this crop's edges, or bad quality) -- still no usable
            # color reading, but keep the ruler/reasons rather than
            # collapsing to the fully-empty shape.
            status, label, confidence = KeypointColorStatus.UNKNOWN, "UNKNOWN", 0.0
        elif other_sampled >= 1:
            status, label, confidence = KeypointColorStatus.OK, primary["label"], primary["confidence"]
        else:
            status, label, confidence = KeypointColorStatus.PARTIAL, primary["label"], primary["confidence"]

        return {
            "status": status,
            "conf_threshold": KEYPOINT_CONF_THRESHOLD,
            "inter_eye_distance_px": round(inter_eye_distance_px, 2),
            "label": label,
            "confidence": confidence,
            "patches": patches,
            "reasons": reasons,
        }
    except Exception as e:
        log.warning(f"keypoint forehead color extraction failed: {e}")
        return _empty_result(f"unexpected error: {e}")


def combine_keypoint_color(readings: list[dict]) -> dict:
    """Combine the per-front-photo readings (registration sends 2) into one
    stored value. Same spirit as pipeline.pose.combine_geometry: never
    raises, and represents disagreement by preferring the more confident
    reading rather than voting or erroring -- this signal is additive and
    must never influence whether a registration or search succeeds (unlike
    classify_body_color's front1/front2 unanimity requirement, which DOES
    gate registration -- this module intentionally does not mirror that).
    """
    readings = readings or []
    usable = [r for r in readings if r.get("status") in (KeypointColorStatus.OK, KeypointColorStatus.PARTIAL)]

    if not usable:
        reason = (
            readings[0].get("reasons", {}).get("status") if readings else "no front photo supplied"
        )
        return {
            "status": KeypointColorStatus.UNKNOWN,
            "conf_threshold": KEYPOINT_CONF_THRESHOLD,
            "inter_eye_distance_px": None,
            "label": "UNKNOWN",
            "confidence": 0.0,
            "patches": {"forehead_low": None, "forehead_high": None, "crest": None},
            "sources_ok": 0,
            "sources_usable": 0,
            "reason": reason,
        }

    best = max(usable, key=lambda r: r["confidence"])
    ok_count = sum(1 for r in readings if r.get("status") == KeypointColorStatus.OK)
    rulers = [r["inter_eye_distance_px"] for r in usable if r.get("inter_eye_distance_px") is not None]

    # THIS is the reading that actually reaches RegisterResponse
    # .keypoint_forehead_color. Its "label"/"confidence" are STILL only the
    # forehead_low patch's reading (best["label"]/best["confidence"] --
    # carried through from extract_keypoint_forehead_color's own flattening,
    # not re-decided here). On a genuinely patterned/spotted coat the other
    # two patches ("patches"["forehead_high"] / ["crest"]) can disagree with
    # it in a way that matters, not just noise: confirmed on a real photo
    # (model_training/pose_model's animal_584, a real black-and-white animal
    # with a white forehead blaze) where forehead_low read BLACK(0.65) while
    # forehead_high/crest both read WHITE(~0.85), correctly reflecting the
    # blaze. Any future consumer of this field that wants real accuracy on
    # patterned animals -- not just solid coats -- MUST read "patches", not
    # this top-level label.
    return {
        "status": best["status"],
        "conf_threshold": KEYPOINT_CONF_THRESHOLD,
        "inter_eye_distance_px": round(sum(rulers) / len(rulers), 2) if rulers else None,
        "label": best["label"],
        "confidence": best["confidence"],
        "patches": best["patches"],
        "sources_ok": ok_count,
        "sources_usable": len(usable),
        "reason": (
            None if len(usable) == len(readings)
            else f"{len(usable)}/{len(readings)} front photos produced a usable reading"
        ),
    }
