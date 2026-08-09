"""
StableIdMapper — rewrite volatile tracker IDs into monotonic Cow‑ID labels.

BoT‑SORT (and most MOT trackers) can reassign a new raw ID when a cow is
briefly occluded or when tracks fragment.  StableIdMapper catches this:

1.  For every raw‑ID in the current frame, check whether any *recent*
    stable ID (within `memory_frames`) occupied roughly the same region
    (IoU ≥ `iou_thresh`) -- projected forward to now by that ID's last
    known velocity, not compared against where it stood however many
    frames ago it was last seen (see `_project`).
2.  If yes → reuse that stable ID.
3.  If no  → mint a new monotonic ID ("Cow ID N").

Two real bugs were found and fixed here (2026-08-10) after real footage
showed a 15s clip with only 9 cattle ever visible at once minting 34-41
"distinct" stable IDs:

- Two different raw IDs in the SAME frame could be handed the same stable
  ID. The direct raw_id->stable_id cache (step 1 below) was trusted
  unconditionally, even when a sibling detection this same frame had
  already claimed that stable ID -- which happens once BoT-SORT reassigns
  a fresh raw ID mid-track and the old raw ID later reappears. Verified on
  real footage: 37 of 225 frames had a stable ID stamped on two different
  boxes at once. That's not overcounting, it's misidentification -- two
  real animals reporting as the same one. `claimed_this_frame` below
  fixes it: a stable ID already taken this frame is never reused, a new
  one is minted instead. 0 duplicate-ID frames after the fix.
- `memory_frames` was 30 (~1s), shorter than the underlying BoT-SORT
  tracker's own `track_buffer` (90 processed frames, in trackers/*.yaml).
  That's backwards: this layer only has to intervene when BoT-SORT
  ALREADY gave up on a raw ID, which by definition takes longer than
  BoT-SORT's own patience -- so the safety net was expiring before the
  exact case it exists to catch. Now aligned to match. Separately, IoU was
  matched against the box's last known STATIC position with no allowance
  for the animal having walked during the gap -- velocity-projecting the
  remembered box forward before comparing (`_project`) fixes this without
  touching the threshold itself. Real-footage unique-tracked-cattle count:
  41 (after the duplicate-ID fix alone) -> 33 (with both fixes), peak-in-
  frame unchanged at 9.

Neither of these should be treated as fully calibrated -- there is still
no labeled ground-truth clip to validate the *final* cattle count against,
only the internal consistency check available (no stable ID may be handed
to two different boxes in the same frame). Re-run that check and the
ID-lifetime/fragmentation check (both used to diagnose this) against real
footage before trusting any specific count from this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _TrackedBox:
    stable_id: int
    bbox: tuple[float, float, float, float]   # x1 y1 x2 y2
    last_seen_frame: int
    # Center velocity in px/processed-frame, from the last observed step.
    # (0, 0) until there are two observations to diff. Used to project this
    # box forward to the CURRENT frame before IoU-matching against it --
    # comparing a moving cow's new position to where it stood several
    # frames ago is exactly what let real occlusions "expire" into brand
    # new IDs (see module docstring).
    velocity: tuple[float, float] = (0.0, 0.0)


def _iou(a: tuple, b: tuple) -> float:
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _project(bbox: tuple, velocity: tuple[float, float], frames_ahead: int) -> tuple:
    """Slide `bbox` forward by `velocity` (px/frame) for `frames_ahead`
    frames, keeping its width/height fixed. A short gap is not enough time
    for a cow's apparent size to change meaningfully, so only position is
    extrapolated."""
    dx = velocity[0] * frames_ahead
    dy = velocity[1] * frames_ahead
    return (bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy)


class StableIdMapper:
    """
    Maps raw tracker IDs → stable monotonic cow IDs.

    Parameters
    ----------
    iou_thresh : float
        Minimum IoU to consider two boxes the same animal.
    memory_frames : int
        How many frames back to look for a match before dropping
        the track from memory.
    """

    def __init__(self, iou_thresh: float = 0.25, memory_frames: int = 90):
        self.iou_thresh = iou_thresh
        self.memory_frames = memory_frames

        self._next_id: int = 1
        # raw_tracker_id  →  stable_id
        self._raw_to_stable: dict[int, int] = {}
        # stable_id       →  _TrackedBox (last known position)
        self._memory: dict[int, _TrackedBox] = {}

    # ── public API ────────────────────────────────────────────────

    def update(
        self,
        frame_idx: int,
        detections: list[tuple[int, tuple[float, float, float, float]]],
    ) -> list[tuple[int, tuple[float, float, float, float]]]:
        """
        Accept a frame's detections and return them with stable IDs.

        Parameters
        ----------
        frame_idx : int
            Current frame number (used to expire old memory entries).
        detections : list[(raw_id, (x1, y1, x2, y2))]
            Raw tracker output for this frame.

        Returns
        -------
        list[(stable_id, (x1, y1, x2, y2))]
        """
        # prune stale memory entries
        stale = [
            sid
            for sid, tb in self._memory.items()
            if frame_idx - tb.last_seen_frame > self.memory_frames
        ]
        for sid in stale:
            self._memory.pop(sid, None)

        # Two different raw IDs in the SAME frame are, by construction, two
        # different physical boxes -- they must never be handed the same
        # stable ID. `claimed_this_frame` enforces that even when the cache
        # in `_resolve` would otherwise reuse a stable ID a sibling
        # detection already took this frame (see `_resolve`'s docstring).
        results: list[tuple[int, tuple[float, float, float, float]]] = []
        claimed_this_frame: set[int] = set()

        for raw_id, bbox in detections:
            stable_id = self._resolve(raw_id, bbox, frame_idx, claimed_this_frame)
            claimed_this_frame.add(stable_id)

            prev = self._memory.get(stable_id)
            velocity = (0.0, 0.0)
            if prev is not None and frame_idx > prev.last_seen_frame:
                gap = frame_idx - prev.last_seen_frame
                prev_cx = (prev.bbox[0] + prev.bbox[2]) / 2
                prev_cy = (prev.bbox[1] + prev.bbox[3]) / 2
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                velocity = ((cx - prev_cx) / gap, (cy - prev_cy) / gap)

            self._memory[stable_id] = _TrackedBox(
                stable_id=stable_id, bbox=bbox, last_seen_frame=frame_idx,
                velocity=velocity,
            )
            results.append((stable_id, bbox))

        return results

    @property
    def all_stable_ids(self) -> set[int]:
        """Every stable ID minted so far (dead or alive)."""
        return set(self._memory.keys()) | set(self._raw_to_stable.values())

    @property
    def total_minted(self) -> int:
        """Total unique stable IDs minted (= final cattle count)."""
        return self._next_id - 1

    # ── internals ─────────────────────────────────────────────────

    def _resolve(
        self,
        raw_id: int,
        bbox: tuple,
        frame_idx: int,
        claimed_this_frame: set[int],
    ) -> int:
        # 1) direct raw→stable mapping still valid?
        # Only trusted if nothing else in THIS frame already claimed it --
        # a stale cache entry pointing at an ID a sibling detection just
        # took would otherwise stamp one stable ID onto two live boxes at
        # once (this was the actual duplicate-ID bug: a raw ID can end up
        # cached against a stable ID that a *different* raw ID is also
        # validly tracking, if BoT-SORT reassigned IDs mid-track and the
        # old raw ID later reappears).
        if raw_id in self._raw_to_stable:
            sid = self._raw_to_stable[raw_id]
            if sid in self._memory and sid not in claimed_this_frame:
                return sid

        # 2) IoU match against recent memory, excluding anything already
        # claimed by another detection this frame (same reasoning as
        # above). Each candidate's remembered box is projected forward to
        # the CURRENT frame using its last known velocity before comparing
        # -- matching against where a walking cow actually is now, not
        # where it stood however many frames ago it was last seen.
        best_iou = 0.0
        best_sid: int | None = None
        for sid, tb in self._memory.items():
            if sid in claimed_this_frame:
                continue
            frames_ahead = frame_idx - tb.last_seen_frame
            projected = _project(tb.bbox, tb.velocity, frames_ahead)
            score = _iou(bbox, projected)
            if score > best_iou:
                best_iou = score
                best_sid = sid

        if best_sid is not None and best_iou >= self.iou_thresh:
            self._raw_to_stable[raw_id] = best_sid
            return best_sid

        # 3) mint new stable ID
        new_id = self._next_id
        self._next_id += 1
        self._raw_to_stable[raw_id] = new_id
        return new_id
