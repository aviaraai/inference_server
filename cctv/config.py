"""
cctv/config.py — configuration for the CCTV video-analytics model.

Third model alongside `pipeline/` (detection) and `godhaar/` (identification):
video-based cattle detection, tracking, counting, and movement/density/
isolation analytics. Self-contained — its own presets, paths, and DB, no
shared state with the register/search models.

Every tunable knob lives here. Import `PipelineConfig`/`make_config`;
never scatter magic numbers across modules (same convention as
`godhaar/config.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


# ── paths ──────────────────────────────────────────────────────────
# Scoped under cctv/ so this model's runtime artifacts (uploads, job
# outputs, session DB) never mix with the detection/identification
# models' files at the repo root.
BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
UPLOADS_DIR = BASE_DIR / "uploads"
DB_PATH = BASE_DIR / "cctv.db"

RUNS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)


# ── YOLO cattle class filter ──────────────────────────────────────
# COCO class names that count as "cattle". The pipeline maps these
# to class-IDs at runtime so they survive model swaps.
CATTLE_TERMS: set[str] = {
    "cow", "cattle", "bull", "calf", "buffalo",
    "bovine", "ox", "horse",           # horse kept for COCO compat
}


# ── presets ────────────────────────────────────────────────────────
class Preset(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"
    LIVE_CPU = "live_cpu"
    LIVE_GPU = "live_gpu"


@dataclass
class PipelineConfig:
    """Runtime config for a single video processing job."""

    # model
    model_path: str = "yolo11s.pt"
    img_size: int = 640
    confidence: float = 0.25
    half: bool = False                       # FP16

    # sampling
    vid_stride: int = 2                      # frame-skip

    # tracking
    use_tracking: bool = True
    tracker_yaml: str = str(BASE_DIR / "trackers" / "botsort_cattle_fast.yaml")

    # class filter
    filter_cattle_classes: bool = True

    # stable-ID mapper
    stable_id_iou_thresh: float = 0.25
    stable_id_memory_frames: int = 30

    # analytics
    enable_analytics: bool = True
    density_grid_cells: int = 8              # N×N grid for density map
    isolation_multiplier: float = 2.0        # ×median NN distance → isolated

    # output
    output_dir: Optional[str] = None         # auto-assigned per job


PRESETS: dict[Preset, dict] = {
    Preset.FAST: dict(
        model_path="yolo11s.pt",
        img_size=640,
        vid_stride=2,
        tracker_yaml=str(BASE_DIR / "trackers" / "botsort_cattle_fast.yaml"),
        half=False,
    ),
    Preset.BALANCED: dict(
        model_path="yolo11m.pt",
        img_size=640,
        vid_stride=1,
        tracker_yaml=str(BASE_DIR / "trackers" / "botsort_cattle_fast.yaml"),
        half=False,
    ),
    Preset.ACCURATE: dict(
        model_path="yolo11m.pt",
        img_size=960,
        vid_stride=1,
        tracker_yaml=str(BASE_DIR / "trackers" / "botsort_cattle.yaml"),
        half=False,
    ),
    Preset.LIVE_CPU: dict(
        model_path="yolo11n.pt",
        img_size=416,
        vid_stride=3,
        tracker_yaml=str(BASE_DIR / "trackers" / "botsort_cattle_fast.yaml"),
        half=False,
    ),
    Preset.LIVE_GPU: dict(
        model_path="yolo11m.pt",
        img_size=640,
        vid_stride=1,
        tracker_yaml=str(BASE_DIR / "trackers" / "botsort_cattle.yaml"),
        half=True,
    ),
}


def make_config(preset: Preset = Preset.FAST, **overrides) -> PipelineConfig:
    """Build a PipelineConfig from a preset + any per-job overrides."""
    base = PRESETS.get(preset, {})
    merged = {**base, **overrides}
    return PipelineConfig(**merged)
