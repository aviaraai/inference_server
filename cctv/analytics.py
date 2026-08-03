"""
cctv/analytics.py — video analytics engine for the CCTV model.

Takes per-frame tracking data produced by `cctv/pipeline.py` and computes:
  • Movement metrics    — speed, distance, trajectory per cow
  • Density heatmap     — N×N grid showing where cattle cluster
  • Isolation detection — cattle consistently far from the herd
  • Activity breakdown  — stationary / walking / running per cow
  • Cross-video trends  — persisted via cctv/database.py

All analytics are pure NumPy math on existing bounding-box data —
no new models required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from cctv.pipeline import FrameResult


# ── data structures ───────────────────────────────────────────────

@dataclass
class CowMetrics:
    """Per-animal analytics summary."""
    stable_id: int
    frames_visible: int = 0
    total_distance_px: float = 0.0          # sum of inter-frame displacements
    avg_speed_px_per_frame: float = 0.0
    max_speed_px_per_frame: float = 0.0
    avg_bbox_area: float = 0.0
    avg_nearest_neighbour_px: float = 0.0
    is_isolated: bool = False
    activity: str = "unknown"               # stationary / walking / running
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    dwell_zones: list[tuple[float, float, int]] = field(default_factory=list)


@dataclass
class DensityCell:
    """One cell in the density grid."""
    row: int
    col: int
    count: float                            # avg detections per frame in this cell
    normalised: float                       # 0–1 relative to max cell


@dataclass
class AnalyticsResult:
    """Full analytics output for one video."""
    per_cow: list[CowMetrics]
    density_grid: list[list[float]]          # [row][col] avg count
    density_cells: list[DensityCell]
    heatmap_image: Optional[np.ndarray]      # BGR overlay image (or None)
    total_cattle: int
    avg_herd_speed: float
    isolated_cattle: list[int]               # stable IDs flagged
    activity_breakdown: dict[str, int]       # {"stationary": N, "walking": M, …}
    frame_count_series: list[int]            # cattle count per processed frame
    summary_text: str


# ── speed thresholds (pixels/frame) ───────────────────────────────
# These are rough heuristics — tune them for your camera setup / fps.
STATIONARY_THRESH = 3.0       # < 3 px/frame → stationary
RUNNING_THRESH = 25.0         # > 25 px/frame → running
DWELL_RADIUS = 30.0           # px — cluster radius for dwell detection
DWELL_MIN_FRAMES = 10         # minimum frames in cluster → dwell zone


# ── main entry point ──────────────────────────────────────────────

class VideoAnalytics:
    """
    Accumulates per-frame results and computes analytics.

    Usage
    -----
    ```python
    va = VideoAnalytics(frame_w=1920, frame_h=1080, grid_cells=8)
    for frame_result in pipeline_generator:
        va.ingest(frame_result)
    result = va.compute()
    ```
    """

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        grid_cells: int = 8,
        isolation_multiplier: float = 2.0,
        source_fps: float = 30.0,
        vid_stride: int = 1,
    ):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.grid_cells = grid_cells
        self.isolation_mult = isolation_multiplier
        self.source_fps = source_fps
        self.vid_stride = vid_stride

        # per-cow accumulation: stable_id → list of (frame_idx, cx, cy, area)
        self._tracks: dict[int, list[tuple[int, float, float, float]]] = {}
        # per-frame: list of all (cx, cy) for density / NN
        self._frame_positions: list[list[tuple[float, float]]] = []
        # count series
        self._frame_counts: list[int] = []
        self._frame_nn_distances: dict[int, list[float]] = {}

    # ── ingest ────────────────────────────────────────────────────

    def ingest(self, fr: FrameResult) -> None:
        """Feed one frame's detections into the analytics accumulator."""
        positions: list[tuple[float, float]] = []

        for sid, bbox, conf in zip(fr.stable_ids, fr.bboxes, fr.confidences):
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)
            positions.append((cx, cy))

            if sid not in self._tracks:
                self._tracks[sid] = []
            self._tracks[sid].append((fr.frame_idx, cx, cy, area))

        self._frame_positions.append(positions)
        self._frame_counts.append(fr.cattle_in_frame)

        # compute per-cow nearest-neighbour distances this frame
        if len(positions) >= 2:
            for i, (cx_i, cy_i) in enumerate(positions):
                sid = fr.stable_ids[i]
                min_dist = float("inf")
                for j, (cx_j, cy_j) in enumerate(positions):
                    if i == j:
                        continue
                    d = math.hypot(cx_i - cx_j, cy_i - cy_j)
                    if d < min_dist:
                        min_dist = d
                if sid not in self._frame_nn_distances:
                    self._frame_nn_distances[sid] = []
                self._frame_nn_distances[sid].append(min_dist)

    # ── compute ───────────────────────────────────────────────────

    def compute(self) -> AnalyticsResult:
        """Run all analytics and return the result."""
        per_cow = self._compute_per_cow()
        density_grid, density_cells = self._compute_density()
        heatmap = self._render_heatmap(density_grid)
        isolated = [c.stable_id for c in per_cow if c.is_isolated]

        activity_breakdown: dict[str, int] = {}
        for c in per_cow:
            activity_breakdown[c.activity] = activity_breakdown.get(c.activity, 0) + 1

        speeds = [c.avg_speed_px_per_frame for c in per_cow if c.frames_visible > 1]
        avg_herd_speed = sum(speeds) / len(speeds) if speeds else 0.0

        summary_lines = [
            f"Detected {len(per_cow)} unique cattle across {len(self._frame_counts)} processed frames.",
            f"Average herd speed: {avg_herd_speed:.1f} px/frame.",
        ]
        if isolated:
            summary_lines.append(
                f"Isolated cattle (>{self.isolation_mult}× median NN distance): "
                f"Cow IDs {isolated}."
            )
        for act, cnt in sorted(activity_breakdown.items()):
            summary_lines.append(f"  {act}: {cnt} cattle")

        return AnalyticsResult(
            per_cow=per_cow,
            density_grid=density_grid,
            density_cells=density_cells,
            heatmap_image=heatmap,
            total_cattle=len(per_cow),
            avg_herd_speed=round(avg_herd_speed, 2),
            isolated_cattle=isolated,
            activity_breakdown=activity_breakdown,
            frame_count_series=self._frame_counts,
            summary_text="\n".join(summary_lines),
        )

    # ── per-cow metrics ──────────────────────────────────────────

    def _compute_per_cow(self) -> list[CowMetrics]:
        results: list[CowMetrics] = []

        # median NN distance across ALL cattle (for isolation test)
        all_nn: list[float] = []
        for dists in self._frame_nn_distances.values():
            all_nn.extend(dists)
        global_median_nn = float(np.median(all_nn)) if all_nn else 0.0

        for sid, track in self._tracks.items():
            frames_visible = len(track)
            trajectory = [(cx, cy) for _, cx, cy, _ in track]
            areas = [a for _, _, _, a in track]

            # displacement / speed
            speeds: list[float] = []
            total_dist = 0.0
            for i in range(1, len(track)):
                dx = track[i][1] - track[i - 1][1]
                dy = track[i][2] - track[i - 1][2]
                d = math.hypot(dx, dy)
                # normalise by frame gap (in case of non-consecutive detections)
                gap = track[i][0] - track[i - 1][0]
                spd = d / max(gap, 1)
                speeds.append(spd)
                total_dist += d

            avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
            max_speed = max(speeds) if speeds else 0.0

            # activity classification
            if avg_speed < STATIONARY_THRESH:
                activity = "stationary"
            elif avg_speed > RUNNING_THRESH:
                activity = "running"
            else:
                activity = "walking"

            # nearest-neighbour average
            nn_dists = self._frame_nn_distances.get(sid, [])
            avg_nn = sum(nn_dists) / len(nn_dists) if nn_dists else 0.0
            is_isolated = (
                global_median_nn > 0
                and avg_nn > self.isolation_mult * global_median_nn
            )

            # dwell zones — simple clustering of trajectory points
            dwell_zones = self._find_dwell_zones(trajectory)

            results.append(CowMetrics(
                stable_id=sid,
                frames_visible=frames_visible,
                total_distance_px=round(total_dist, 1),
                avg_speed_px_per_frame=round(avg_speed, 2),
                max_speed_px_per_frame=round(max_speed, 2),
                avg_bbox_area=round(sum(areas) / len(areas), 1) if areas else 0,
                avg_nearest_neighbour_px=round(avg_nn, 1),
                is_isolated=is_isolated,
                activity=activity,
                trajectory=trajectory,
                dwell_zones=dwell_zones,
            ))

        return sorted(results, key=lambda c: c.stable_id)

    # ── density heatmap ───────────────────────────────────────────

    def _compute_density(self) -> tuple[list[list[float]], list[DensityCell]]:
        n = self.grid_cells
        grid = [[0.0] * n for _ in range(n)]

        cell_w = self.frame_w / n
        cell_h = self.frame_h / n

        for positions in self._frame_positions:
            for cx, cy in positions:
                col = min(int(cx / cell_w), n - 1)
                row = min(int(cy / cell_h), n - 1)
                grid[row][col] += 1

        # average per frame
        num_frames = max(len(self._frame_positions), 1)
        for r in range(n):
            for c in range(n):
                grid[r][c] /= num_frames

        # normalise + build cell list
        max_val = max(max(row) for row in grid) if grid else 1
        max_val = max(max_val, 1e-9)

        cells: list[DensityCell] = []
        for r in range(n):
            for c in range(n):
                cells.append(DensityCell(
                    row=r, col=c,
                    count=round(grid[r][c], 3),
                    normalised=round(grid[r][c] / max_val, 3),
                ))

        return grid, cells

    def _render_heatmap(self, grid: list[list[float]]) -> np.ndarray:
        """Render a BGR heatmap overlay at the original video resolution."""
        n = len(grid)
        arr = np.array(grid, dtype=np.float32)
        max_val = arr.max() or 1.0
        arr = (arr / max_val * 255).astype(np.uint8)

        # upscale small grid to frame size
        heatmap_small = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
        heatmap = cv2.resize(
            heatmap_small, (self.frame_w, self.frame_h),
            interpolation=cv2.INTER_LINEAR,
        )
        return heatmap

    # ── dwell zones ───────────────────────────────────────────────

    @staticmethod
    def _find_dwell_zones(
        trajectory: list[tuple[float, float]],
    ) -> list[tuple[float, float, int]]:
        """
        Simple greedy clustering: walk the trajectory, start a new cluster
        whenever the cow moves > DWELL_RADIUS from the cluster centre.
        Return clusters with ≥ DWELL_MIN_FRAMES points as dwell zones.
        """
        if not trajectory:
            return []

        zones: list[tuple[float, float, int]] = []
        cluster_pts: list[tuple[float, float]] = [trajectory[0]]
        cx, cy = trajectory[0]

        for px, py in trajectory[1:]:
            if math.hypot(px - cx, py - cy) <= DWELL_RADIUS:
                cluster_pts.append((px, py))
                # update cluster centre as running mean
                cx = sum(p[0] for p in cluster_pts) / len(cluster_pts)
                cy = sum(p[1] for p in cluster_pts) / len(cluster_pts)
            else:
                # emit if large enough
                if len(cluster_pts) >= DWELL_MIN_FRAMES:
                    zones.append((round(cx, 1), round(cy, 1), len(cluster_pts)))
                cluster_pts = [(px, py)]
                cx, cy = px, py

        # final cluster
        if len(cluster_pts) >= DWELL_MIN_FRAMES:
            zones.append((round(cx, 1), round(cy, 1), len(cluster_pts)))

        return zones


# ── trajectory overlay renderer ───────────────────────────────────

def draw_trajectories(
    frame: np.ndarray,
    per_cow: list[CowMetrics],
    tail_length: int = 30,
) -> np.ndarray:
    """Draw coloured trajectory tails on a frame (for the last N points)."""
    overlay = frame.copy()
    palette = [
        (46, 204, 113), (52, 152, 219), (231, 76, 60),
        (241, 196, 15), (155, 89, 182), (26, 188, 156),
        (230, 126, 34), (44, 62, 80),
    ]

    for cow in per_cow:
        colour = palette[cow.stable_id % len(palette)]
        pts = cow.trajectory[-tail_length:]
        for i in range(1, len(pts)):
            alpha = i / len(pts)  # fade in
            thick = max(1, int(alpha * 3))
            p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
            p2 = (int(pts[i][0]), int(pts[i][1]))
            cv2.line(overlay, p1, p2, colour, thick, cv2.LINE_AA)

    return overlay
