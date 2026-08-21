"""
cctv/muzzle_crop.py — extract one registration-style muzzle crop per
stable-tracked cow, from CCTV footage.

Layout-independent prerequisite for cross-camera de-duplication (see
CLAUDE.md, "Cross-camera de-duplication"): before two sightings of the same
physical animal on different cameras can ever be compared, each tracked cow
needs at least one crop suitable for that comparison. This module produces
it; it does NOT compare anything — no cross-camera logic lives here yet.

There is no dedicated muzzle-only detector anywhere in this server —
`pipeline/muzzle_detect.py` is a permanent no-op (CTO-confirmed 2026-08-17,
see godhaar/config.py). The only thing that has ever localized a "muzzle
crop" for the embedding path, for /register and /search alike, is
`crop_cattle()` (pipeline/yolo_crop.py) — a whole-animal COCO detector that
happens to yield a muzzle-filling result because the capture UI demands a
20-30cm close-up. This module reuses that exact function, unmodified.

A CCTV frame is not a close-up: the camera sees the whole animal from a
distance, often with several other cattle in frame. Calling crop_cattle() on
the full frame would hand it the same multi-subject ambiguity registration
photos already solved with select_dominant_box() — except here we don't need
that heuristic at all, because BoT-SORT tracking already tells us exactly
which pixels are OUR cow. So this module pre-crops the frame to the tracked
box (generously padded, so crop_cattle's own YOLO pass has context to
re-detect the animal rather than clip it) before handing that sub-image to
crop_cattle() for its normal tight-crop treatment. The result is the same
KIND of crop registration produces — a whole-animal-dominated JPEG, called
"muzzle crop" by this codebase's convention — not a true muzzle-only region,
because that region no longer exists as a concept here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pipeline.yolo_crop import crop_cattle

log = logging.getLogger("cctv.muzzle_crop")

# Fraction of the tracked box's own width/height added as context padding on
# each side before handing the sub-image to crop_cattle() — NOT crop_cattle's
# own CROP_PADDING_PX (that's a final few-pixel pad around ITS detection,
# unrelated). A CCTV tracker box is usually already a tight fit around the
# animal's visible silhouette; crop_cattle's YOLO needs some slack around
# that to redetect the same animal rather than clip a leg or horn tip off
# the edge it's handed. Unvalidated starting guess, same status as the
# Telangana app's CONTEXT_MULTIPLIER=4 for the same class of problem — if
# extraction success rate is low on legs/horns being cut off, raise this
# first; if crop_cattle keeps pulling in a neighbouring animal, lower it.
CONTEXT_PADDING_FRACTION = 0.25
MIN_PADDING_PX = 20


@dataclass
class BestSighting:
    """The highest-tracker-confidence observation of one stable ID so far."""
    stable_id: int
    frame: np.ndarray                              # BGR, owns its own copy
    bbox: tuple[float, float, float, float]         # x1 y1 x2 y2, in `frame`'s coordinates
    confidence: float                               # the TRACKER's detection confidence
    frame_idx: int


@dataclass
class MuzzleCropResult:
    """Outcome of trying to extract a muzzle crop for one stable ID."""
    stable_id: int
    status: str                     # crop_cattle()'s status, or "NO_SIGHTING"
    crop_path: Optional[str]        # set only when status == "OK"
    source_confidence: float        # tracker's confidence on the frame used
    crop_confidence: float          # crop_cattle()'s own confidence, 0.0 if not OK
    source_frame_idx: Optional[int] = None


class BestSightingTracker:
    """Accumulates, per stable ID, the single best frame to extract a crop
    from — the tracker's highest-confidence detection of that ID across the
    whole clip. One high-quality frame is what registration itself works
    from (one accepted photo per muzzle slot), so this mirrors that rather
    than trying to fuse multiple frames.

    Holds one full-resolution frame copy per currently-best stable ID —
    bounded by how many distinct cattle the clip ever tracks, not by frame
    count, and each entry is replaced (not accumulated) as a better
    confidence is seen. Freed by the caller once extract_muzzle_crops() has
    run.
    """

    def __init__(self) -> None:
        self._best: dict[int, BestSighting] = {}

    def observe(
        self,
        frame_idx: int,
        frame: np.ndarray,
        stable_detections: list[tuple[int, tuple[float, float, float, float]]],
        confidences: list[float],
    ) -> None:
        """Record this frame's detections. `frame` is read, never mutated —
        a copy is only made for a stable ID that just improved on its best
        confidence so far, not on every observation."""
        for (sid, bbox), conf in zip(stable_detections, confidences):
            current = self._best.get(sid)
            if current is not None and conf <= current.confidence:
                continue
            self._best[sid] = BestSighting(
                stable_id=sid, frame=frame.copy(), bbox=bbox,
                confidence=conf, frame_idx=frame_idx,
            )

    @property
    def best_sightings(self) -> dict[int, BestSighting]:
        return self._best


def _padded_subimage(
    frame: np.ndarray, bbox: tuple[float, float, float, float]
) -> Optional[np.ndarray]:
    """Crop `frame` to `bbox` plus context padding, clipped to frame bounds.
    Returns None for a degenerate (zero-area) box."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None

    pad_x = max(box_w * CONTEXT_PADDING_FRACTION, MIN_PADDING_PX)
    pad_y = max(box_h * CONTEXT_PADDING_FRACTION, MIN_PADDING_PX)

    px1 = max(0, int(round(x1 - pad_x)))
    py1 = max(0, int(round(y1 - pad_y)))
    px2 = min(w, int(round(x2 + pad_x)))
    py2 = min(h, int(round(y2 + pad_y)))

    if px2 <= px1 or py2 <= py1:
        return None
    return frame[py1:py2, px1:px2]


def extract_muzzle_crops(
    best_sightings: dict[int, BestSighting], out_dir: Path
) -> dict[int, MuzzleCropResult]:
    """Run crop_cattle() on each stable ID's best sighting and save the
    result. Keyed purely by stable_id — deliberately NOT by faiss_id or any
    registered-animal identity, since this has to work for cattle that were
    never registered at all (cross-camera de-dup is "same physical animal
    across two feeds," not "same registered identity" — see CLAUDE.md).

    Never raises: a single cow's crop failing (no detection, multi-cattle in
    the padded sub-image, degenerate box) must not abort extraction for
    every other cow in the clip, and must not fail the pipeline job itself —
    same fail-open convention as the rest of this codebase's optional
    extraction steps (see muzzle_detect.py, muzzle_crop_cache.py).
    """
    crops_dir = out_dir / "muzzle_crops"
    results: dict[int, MuzzleCropResult] = {}

    for sid, sighting in best_sightings.items():
        try:
            sub_img = _padded_subimage(sighting.frame, sighting.bbox)
            if sub_img is None:
                results[sid] = MuzzleCropResult(
                    stable_id=sid, status="DEGENERATE_BOX", crop_path=None,
                    source_confidence=sighting.confidence, crop_confidence=0.0,
                    source_frame_idx=sighting.frame_idx,
                )
                continue

            crop, status, crop_conf = crop_cattle(sub_img)

            if status != "OK" or crop is None:
                log.info(
                    f"muzzle_crop: stable_id={sid} frame={sighting.frame_idx} "
                    f"status={status} — no crop extracted"
                )
                results[sid] = MuzzleCropResult(
                    stable_id=sid, status=status, crop_path=None,
                    source_confidence=sighting.confidence, crop_confidence=0.0,
                    source_frame_idx=sighting.frame_idx,
                )
                continue

            crops_dir.mkdir(parents=True, exist_ok=True)
            crop_path = crops_dir / f"{sid}.jpg"
            ok = cv2.imwrite(str(crop_path), crop)
            if not ok:
                log.warning(f"muzzle_crop: stable_id={sid} — failed to write {crop_path}")
                results[sid] = MuzzleCropResult(
                    stable_id=sid, status="WRITE_FAILED", crop_path=None,
                    source_confidence=sighting.confidence, crop_confidence=crop_conf,
                    source_frame_idx=sighting.frame_idx,
                )
                continue

            results[sid] = MuzzleCropResult(
                stable_id=sid, status="OK", crop_path=str(crop_path),
                source_confidence=sighting.confidence, crop_confidence=crop_conf,
                source_frame_idx=sighting.frame_idx,
            )
        except Exception as e:
            # A crash on one cow's crop is not grounds to lose every other
            # cow's — see module docstring's fail-open note.
            log.warning(f"muzzle_crop: stable_id={sid} — extraction raised: {e}")
            results[sid] = MuzzleCropResult(
                stable_id=sid, status="ERROR", crop_path=None,
                source_confidence=sighting.confidence, crop_confidence=0.0,
                source_frame_idx=sighting.frame_idx,
            )

    return results
