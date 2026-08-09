"""
StableIdMapper — rewrite volatile tracker IDs into monotonic Cow‑ID labels.

BoT‑SORT (and most MOT trackers) can reassign a new raw ID when a cow is
briefly occluded or when tracks fragment.  StableIdMapper catches this:

1.  For every raw‑ID in the current frame, check whether any *recent*
    stable ID (within `memory_frames`) occupied roughly the same region
    (IoU ≥ `iou_thresh`).
2.  If yes → reuse that stable ID.
3.  If no  → mint a new monotonic ID ("Cow ID N").

This is the same design described in the Cattle AI docs — the IoU‑based
memory window that turns noisy tracker output into clean labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _TrackedBox:
    stable_id: int
    bbox: tuple[float, float, float, float]   # x1 y1 x2 y2
    last_seen_frame: int


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

    def __init__(self, iou_thresh: float = 0.25, memory_frames: int = 30):
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

        results: list[tuple[int, tuple[float, float, float, float]]] = []

        for raw_id, bbox in detections:
            stable_id = self._resolve(raw_id, bbox, frame_idx)
            self._memory[stable_id] = _TrackedBox(
                stable_id=stable_id, bbox=bbox, last_seen_frame=frame_idx,
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

    def _resolve(self, raw_id: int, bbox: tuple, frame_idx: int) -> int:
        # 1) direct raw→stable mapping still valid?
        if raw_id in self._raw_to_stable:
            sid = self._raw_to_stable[raw_id]
            if sid in self._memory:
                return sid

        # 2) IoU match against recent memory
        best_iou = 0.0
        best_sid: int | None = None
        for sid, tb in self._memory.items():
            score = _iou(bbox, tb.bbox)
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
