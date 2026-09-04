"""Keypoint -> face geometry for the cattle-face pose model.

Takes the model's raw output for one face — 8 keypoints, each ``(x, y,
confidence)`` — and computes the confirmed geometry set:

  * ``inter_eye_distance_px``     — the scale ruler; every ratio divides by it
  * ``horn_base_distance_ratio``  — right/left horn-base distance / ruler
  * ``ear_base_span_ratio``       — right/left ear-base distance / ruler
  * ``horn_present_right/left``   — True if that horn-base keypoint was
                                    confidently detected; UNKNOWN otherwise

Confidence rule (non-negotiable, same standard as ``CONFIRMED_HORNLESS``):
any keypoint below ``conf_threshold`` yields **UNKNOWN** (``None``) for every
value that depends on it — never a silently computed number from a
low-confidence point. In particular:

  * either eye below threshold  -> no ruler -> every ratio cascades to UNKNOWN
  * a horn/ear base below threshold -> that specific ratio is UNKNOWN
  * ``horn_present_*`` is never ``False`` from this layer: a missing keypoint
    means "could not tell", not "confirmed hornless" (a front photo cannot
    tell a polled animal from an occluded horn — that distinction needs a
    separate positive signal).

Everything here is a pure function of the input array — no model, no I/O —
so it is fully unit-testable before a trained model exists
(``tests/test_geometry.py``).

------------------------------------------------------------------------------
VENDORED, VERBATIM, from ``pose_model/geometry.py`` (the model-training tree,
which is not a deployable package — training data/runs/logs live alongside it
and it is not on the inference host). The body below is byte-identical to that
file so its unit tests (``pose_model/tests/test_geometry.py``, 15 cases) still
certify this copy; keep them in sync — if ``pose_model/geometry.py`` changes,
re-copy here rather than editing in place. Consumed by ``pipeline/pose.py``.
------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

# keypoint indices (matches data.yaml)
R_HORN_BASE, L_HORN_BASE = 0, 1
CREST = 2
R_EAR_BASE, L_EAR_BASE = 3, 4
R_EYE, L_EYE = 5, 6
MUZZLE = 7
KPT_NAMES = [
    "right_horn_base", "left_horn_base", "crest", "right_ear_base",
    "left_ear_base", "right_eye", "left_eye", "muzzle",
]
N_KPT = 8

DEFAULT_CONF_THRESHOLD = 0.5
#: below this many confident keypoints, we will not assert a face is present
DEFAULT_MIN_FACE_KEYPOINTS = 2

UNKNOWN = None  # explicit alias for readability


class GeometryStatus:
    OK = "OK"              # face present and inter-eye ruler available
    NO_RULER = "NO_RULER"  # face present but one/both eyes below threshold
    NO_FACE = "NO_FACE"    # too few confident keypoints to assert a face


@dataclass
class Geometry:
    status: str
    conf_threshold: float
    inter_eye_distance_px: float | None
    horn_base_distance_ratio: float | None
    ear_base_span_ratio: float | None
    horn_present_right: bool | None
    horn_present_left: bool | None
    keypoints_present: dict = field(default_factory=dict)
    #: field name -> plain-English reason it is UNKNOWN (only for UNKNOWN fields)
    reasons: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_ok(self) -> bool:
        return self.status == GeometryStatus.OK


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def compute_geometry(
    keypoints,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    min_face_keypoints: int = DEFAULT_MIN_FACE_KEYPOINTS,
) -> Geometry:
    """Compute normalised face geometry from one face's keypoints.

    Parameters
    ----------
    keypoints
        Array-like of shape ``(8, 3)``: ``[x, y, confidence]`` per keypoint,
        ordered as in ``KPT_NAMES``. Pixel coordinates; confidence in ``[0, 1]``.
    conf_threshold
        Keypoints with confidence below this are treated as not detected.
    min_face_keypoints
        Fewer confident keypoints than this -> ``status = NO_FACE``.
    """
    kp = np.asarray(keypoints, dtype=float)
    if kp.shape != (N_KPT, 3):
        raise ValueError(f"keypoints must be shape ({N_KPT}, 3), got {kp.shape}")

    xy = kp[:, :2]
    conf = kp[:, 2]
    finite = np.isfinite(xy).all(axis=1) & np.isfinite(conf)
    present = (conf >= conf_threshold) & finite
    present_by_name = {KPT_NAMES[i]: bool(present[i]) for i in range(N_KPT)}
    reasons: dict[str, str] = {}

    def _missing(*idx) -> list[str]:
        return [KPT_NAMES[i] for i in idx if not present[i]]

    # --- horn presence: depends only on its own keypoint --------------------
    # True when confidently detected; UNKNOWN otherwise. Never False here.
    def _horn_side(idx: int, label: str):
        if present[idx]:
            return True
        reasons[f"horn_present_{label}"] = (
            f"{KPT_NAMES[idx]} confidence {conf[idx]:.3f} < {conf_threshold} "
            f"— cannot confirm presence (not evidence of absence)"
        )
        return UNKNOWN

    horn_present_right = _horn_side(R_HORN_BASE, "right")
    horn_present_left = _horn_side(L_HORN_BASE, "left")

    n_present = int(present.sum())
    face_present = n_present >= min_face_keypoints

    # --- inter-eye ruler ---------------------------------------------------
    ruler: float | None
    if present[R_EYE] and present[L_EYE]:
        ruler = _euclid(xy[R_EYE], xy[L_EYE])
        if ruler <= 0:
            reasons["inter_eye_distance_px"] = "both eyes detected but coincide (zero distance)"
            ruler = UNKNOWN
    else:
        ruler = UNKNOWN
        reasons["inter_eye_distance_px"] = (
            f"eye keypoint(s) below {conf_threshold}: {_missing(R_EYE, L_EYE)} "
            f"— no scale ruler, all normalised ratios are UNKNOWN"
        )

    # --- normalised ratios (cascade UNKNOWN when the ruler is UNKNOWN) -----
    def _ratio(i: int, j: int, field_name: str) -> float | None:
        if ruler is UNKNOWN:
            reasons[field_name] = "no inter-eye ruler"
            return UNKNOWN
        if present[i] and present[j]:
            return _euclid(xy[i], xy[j]) / ruler
        reasons[field_name] = f"keypoint(s) below {conf_threshold}: {_missing(i, j)}"
        return UNKNOWN

    horn_base_distance_ratio = _ratio(R_HORN_BASE, L_HORN_BASE, "horn_base_distance_ratio")
    ear_base_span_ratio = _ratio(R_EAR_BASE, L_EAR_BASE, "ear_base_span_ratio")

    # --- status ----------------------------------------------------------
    if not face_present:
        status = GeometryStatus.NO_FACE
        reasons.setdefault(
            "status",
            f"only {n_present} keypoint(s) >= {conf_threshold} (need {min_face_keypoints}) "
            f"— not asserting a face; all geometry UNKNOWN",
        )
        # nothing is trustworthy without a face
        return Geometry(
            status=status, conf_threshold=conf_threshold,
            inter_eye_distance_px=UNKNOWN,
            horn_base_distance_ratio=UNKNOWN,
            ear_base_span_ratio=UNKNOWN,
            horn_present_right=UNKNOWN, horn_present_left=UNKNOWN,
            keypoints_present=present_by_name, reasons=reasons,
        )

    status = GeometryStatus.OK if ruler is not UNKNOWN else GeometryStatus.NO_RULER

    return Geometry(
        status=status, conf_threshold=conf_threshold,
        inter_eye_distance_px=ruler,
        horn_base_distance_ratio=horn_base_distance_ratio,
        ear_base_span_ratio=ear_base_span_ratio,
        horn_present_right=horn_present_right,
        horn_present_left=horn_present_left,
        keypoints_present=present_by_name, reasons=reasons,
    )
