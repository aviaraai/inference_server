"""
pipeline/pose.py — cattle-face 8-keypoint pose model: loader + inference wrapper.

The model (``appstorage/Models/pose_model/best.pt``, YOLO26n-pose fine-tuned on
8 cattle-face keypoints — right/left horn base, crest, right/left ear base,
right/left eye, muzzle) turns a front-face photo into the keypoint array
``pipeline/geometry.py`` consumes, which in turn yields inter-eye-normalised
face proportions (inter-eye distance, horn-base distance ratio, ear-base span
ratio, per-side horn-presence flags).

This is a SUPPLEMENTARY, demote-only signal — it is returned by ``/register``
for the API server to persist and (later) fold into a comparison step, and it
never gates a registration or search. So the model is OPTIONAL at startup, the
same fail-open contract as the YOLO crop model and the LightGlue verifier: if
``POSE_MODEL_PATH`` is unset or the file is missing, every reading is
``NO_FACE`` and ``/register`` returns ``face_geometry`` with nothing in it —
the server runs exactly as before. It is NOT a hard-fail like ``MODEL_PATH``'s
embedding model, which is the primary match signal.

Load path mirrors ``pipeline/yolo_crop.py``: a module-global lazy handle, a
``load_pose_model(path)`` called once from ``main.py``'s lifespan with the
env-var value, and a ``warmup_pose_model()`` in the warmup block. Callers use
the module functions directly (no FastAPI dependency), same as ``crop_cattle``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from pipeline.geometry import (
    GeometryStatus,
    KPT_NAMES,
    N_KPT,
    compute_geometry,
)

log = logging.getLogger("godhaar.pose")

# Lazy-loaded pose model (loaded once on first call / during warmup).
_pose_model = None

# Detection confidence for accepting a face box out of the pose model. Matches
# the value pose_model/pck.py used for its holdout evaluation, so the localizer
# behaves here the way it was measured there. Per-KEYPOINT confidence is a
# separate gate applied inside compute_geometry (its own conf_threshold).
FACE_DETECT_CONF = 0.25

# Per-keypoint confidence floor handed to compute_geometry. geometry.py's own
# default is 0.5; kept explicit here so the extraction gate lives next to the
# tolerance table below.
KEYPOINT_CONF_THRESHOLD = 0.5

# ── Comparison tolerance the FUTURE cross-animal comparison step MUST use ─────
# Not consumed here (this repo computes/stores geometry only — the comparison
# lives in go-apiserver and is explicitly out of scope). Emitted alongside the
# geometry so the consumer doesn't have to re-derive it.
#
# Horn-base keypoint localization is the model's weakest by a wide margin.
# Holdout PCK@0.1 (within 10% of inter-eye distance) from
# appstorage/Models/pose_model/PROVENANCE.txt:
#     right_eye .967   left_eye .967   muzzle .931   crest .778
#     right_ear .778   left_ear .762   left_horn .647   right_horn .500
# i.e. horn bases land inside a ~10% band only 50–65% of the time, vs ~97% for
# eyes. They ARE reliable inside a ~20% band (PCK@0.2 ~0.94), so a comparison
# that treats horn_base_distance_ratio at the same ~0.10 tolerance as
# eye-anchored measurements would reject genuine matches on horn-localization
# noise alone. Widen it to 0.20 for anything horn-derived; keep 0.10 elsewhere.
EYE_ANCHORED_COMPARISON_TOLERANCE = 0.10
HORN_BASE_COMPARISON_TOLERANCE = 0.20

COMPARISON_TOLERANCE: dict[str, float] = {
    "horn_base_distance_ratio": HORN_BASE_COMPARISON_TOLERANCE,
    "ear_base_span_ratio": EYE_ANCHORED_COMPARISON_TOLERANCE,
    "inter_eye_distance_px": EYE_ANCHORED_COMPARISON_TOLERANCE,
}


def load_pose_model(model_path: Optional[str] = None) -> None:
    """Load the pose model into memory. Called once from main.py's lifespan.

    Fail-open: a missing path or file, or an ultralytics import error, logs a
    warning and leaves the model unloaded — it never raises. There is no
    bundled default filename (unlike YOLO_MODEL_NAME), so ``None`` simply
    disables the signal.
    """
    global _pose_model

    if not model_path:
        log.info("POSE_MODEL_PATH not set — face geometry disabled (optional signal).")
        return

    import os

    if not os.path.exists(model_path):
        log.warning(
            f"Pose model not found at '{model_path}' — face geometry disabled "
            f"(optional signal, not fatal)."
        )
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        log.warning("ultralytics not installed — face geometry disabled.")
        return

    try:
        _pose_model = YOLO(model_path, task="pose")
        log.info(f"Pose model loaded from '{model_path}' (task=pose, 8 keypoints).")
    except Exception as e:
        _pose_model = None
        log.warning(f"Pose model failed to load from '{model_path}': {e} — face geometry disabled.")


def warmup_pose_model() -> None:
    """Run one dummy inference so the first real request isn't slow."""
    if _pose_model is None:
        return
    try:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _pose_model.predict(dummy, verbose=False)
        log.info("Pose model warmup complete.")
    except Exception as e:  # warmup must never break startup
        log.warning(f"Pose model warmup failed (non-fatal): {e}")


def pose_available() -> bool:
    """True when a pose model is loaded and ready."""
    return _pose_model is not None


def predict_keypoints(
    face_img_bgr: np.ndarray, *, min_conf: float = FACE_DETECT_CONF
) -> Optional[np.ndarray]:
    """Run the pose model on ONE cropped front-face image.

    Parameters
    ----------
    face_img_bgr
        BGR image — a crop already localized on the animal's head (e.g.
        ``crop_cattle``'s output). The model still runs its own face detection
        inside this, so a slightly wider crop is fine.
    min_conf
        Minimum face-box detection confidence to accept.

    Returns
    -------
    ``np.ndarray`` of shape ``(8, 3)`` — ``[x, y, confidence]`` per keypoint in
    the pixel coordinates of ``face_img_bgr``, ordered exactly as
    ``geometry.KPT_NAMES``. This is the array ``compute_geometry`` expects.

    ``None`` if the model isn't loaded, the image is empty, or no face box
    cleared ``min_conf``. On multiple detections the highest-confidence box
    wins (there is one face on the subject; a background animal nearer the
    camera could otherwise win on size).
    """
    if _pose_model is None:
        return None
    if face_img_bgr is None or getattr(face_img_bgr, "size", 0) == 0:
        return None

    try:
        result = _pose_model.predict(face_img_bgr, conf=min_conf, verbose=False)[0]
    except Exception as e:
        log.warning(f"pose model inference failed: {e}")
        return None

    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return None

    box_conf = result.boxes.conf.cpu().numpy()
    kpts = result.keypoints.data.cpu().numpy()  # (n, 8, 3)
    if kpts.ndim != 3 or kpts.shape[1:] != (N_KPT, 3):
        log.warning(f"pose model returned unexpected keypoint shape {kpts.shape}")
        return None

    best = int(np.argmax(box_conf))
    return kpts[best].astype(float)


def _empty_geometry(status: str, reason: str) -> dict:
    """A ``Geometry.to_dict()``-shaped dict for the paths that never reach
    ``compute_geometry`` (model absent, no face box, bad image) — so a caller
    always gets the same keys back.
    """
    return {
        "status": status,
        "conf_threshold": KEYPOINT_CONF_THRESHOLD,
        "inter_eye_distance_px": None,
        "horn_base_distance_ratio": None,
        "ear_base_span_ratio": None,
        "horn_present_right": None,
        "horn_present_left": None,
        "keypoints_present": {},
        "reasons": {"status": reason},
    }


def extract_face_geometry(front_img_bgr: np.ndarray) -> dict:
    """Full single-photo path: crop the head, run the pose model, compute
    geometry. Return-only and fail-open — never raises, mirrors
    ``RuleBasedMorphologyExtractor.extract`` (which also does its own
    ``crop_cattle`` then analyses the head region).

    Returns a ``Geometry.to_dict()``-shaped dict. ``status`` is one of
    ``OK`` / ``NO_RULER`` / ``NO_FACE`` (from geometry.py). ``NO_FACE`` here
    also covers "pose model not loaded" and "no face box detected" — the
    ``reasons["status"]`` string says which.
    """
    if front_img_bgr is None or getattr(front_img_bgr, "size", 0) == 0:
        return _empty_geometry(GeometryStatus.NO_FACE, "empty or undecodable image")

    if not pose_available():
        return _empty_geometry(GeometryStatus.NO_FACE, "pose model not loaded (POSE_MODEL_PATH unset or file missing)")

    try:
        # Local import: same reason color.py / morphology.py scope theirs —
        # avoid an import-time cycle with yolo_crop.
        from pipeline.yolo_crop import crop_cattle

        crop, det_status, _det_conf = crop_cattle(front_img_bgr)
        if crop is None:
            return _empty_geometry(
                GeometryStatus.NO_FACE,
                f"crop_cattle produced no head crop (yolo_status={det_status})",
            )

        kpts = predict_keypoints(crop)
        if kpts is None:
            return _empty_geometry(
                GeometryStatus.NO_FACE,
                "pose model detected no face in the head crop",
            )

        return compute_geometry(kpts, conf_threshold=KEYPOINT_CONF_THRESHOLD).to_dict()
    except Exception as e:
        log.warning(f"face geometry extraction failed: {e}")
        return _empty_geometry(GeometryStatus.NO_FACE, f"unexpected error: {e}")


def _merge_present(readings: list[dict]) -> dict[str, bool]:
    """A keypoint counts as present in the combined view if any contributing
    reading saw it confidently."""
    out: dict[str, bool] = {}
    for name in KPT_NAMES:
        out[name] = any(bool(r.get("keypoints_present", {}).get(name)) for r in readings)
    return out


def combine_geometry(readings: list[dict]) -> dict:
    """Combine the per-front-photo geometry dicts (registration sends 2) into
    one stored reading. Same spirit as ``morphology.average_readings``:
    numeric ratios average across the photos that produced them; a categorical
    positive (``horn_present_*``) is kept if ANY photo confirmed it (geometry
    never emits ``False``, only ``True``/``None``); never raises.

    Combined ``status`` (a superset of geometry.py's own):
      * ``OK``       — every front photo gave a full reading (ruler + ratios)
      * ``PARTIAL``  — at least one, but not all, gave a full reading
      * ``NO_RULER`` — a face was found but no photo had the inter-eye ruler,
                       so every ratio is null (horn presence may still be set)
      * ``NO_FACE``  — no usable face in any photo (or the model isn't loaded)

    The returned dict is exactly the shape ``schema.FaceGeometry`` expects.
    """
    readings = readings or []
    ok = [r for r in readings if r.get("status") == GeometryStatus.OK]
    with_face = [r for r in readings if r.get("status") in (GeometryStatus.OK, GeometryStatus.NO_RULER)]

    base = {
        "keypoints_present": _merge_present(with_face or readings),
        "sources_ok": len(ok),
        "sources_with_face": len(with_face),
        "comparison_tolerance": dict(COMPARISON_TOLERANCE),
    }

    if not with_face:
        reason = (
            readings[0].get("reasons", {}).get("status", "no face detected")
            if readings else "no front photo supplied"
        )
        return {
            **base,
            "status": GeometryStatus.NO_FACE,
            "inter_eye_distance_px": None,
            "horn_base_distance_ratio": None,
            "ear_base_span_ratio": None,
            "horn_present_right": None,
            "horn_present_left": None,
            "reason": reason,
        }

    horn_right = True if any(r.get("horn_present_right") is True for r in with_face) else None
    horn_left = True if any(r.get("horn_present_left") is True for r in with_face) else None

    if not ok:
        return {
            **base,
            "status": GeometryStatus.NO_RULER,
            "inter_eye_distance_px": None,
            "horn_base_distance_ratio": None,
            "ear_base_span_ratio": None,
            "horn_present_right": horn_right,
            "horn_present_left": horn_left,
            "reason": "a face was detected but no front photo had a confident inter-eye ruler",
        }

    def _avg(fieldname: str) -> Optional[float]:
        vals = [r[fieldname] for r in ok if r.get(fieldname) is not None]
        return float(sum(vals) / len(vals)) if vals else None

    full = len(ok) == len(readings)
    return {
        **base,
        "status": GeometryStatus.OK if full else "PARTIAL",
        "inter_eye_distance_px": _avg("inter_eye_distance_px"),
        "horn_base_distance_ratio": _avg("horn_base_distance_ratio"),
        "ear_base_span_ratio": _avg("ear_base_span_ratio"),
        "horn_present_right": horn_right,
        "horn_present_left": horn_left,
        "reason": None if full else f"{len(ok)}/{len(readings)} front photos produced a full reading",
    }
