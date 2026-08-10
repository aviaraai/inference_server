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
#
# Checked against the actual yolo11s.pt COCO class list: "cattle", "bull",
# "calf", "buffalo", "bovine", "ox" match NOTHING -- COCO (80 classes) has
# no such names, so those six entries were dead vocabulary that never
# matched any real detection. Only "cow" (19) and "horse" (17) ever fired.
#
# That silently undercounts water buffalo -- extremely common in Indian
# goshalas -- since COCO has no buffalo class and a generic detector
# routinely misclassifies one as a visually-similar animal instead. The
# OTHER service in this repo (pipeline/yolo_crop.py) hit and fixed the
# exact same problem already: it matches classes {17,18,19,20,21} =
# horse/sheep/cow/elephant/bear specifically "to catch buffalo
# misclassifications" (see CLAUDE.md). Applying the same, already-
# validated fix here rather than re-deriving it blind.
CATTLE_TERMS: set[str] = {
    "cow", "horse", "sheep", "elephant", "bear",
}


# ── presets ────────────────────────────────────────────────────────
class Preset(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"
    LIVE_CPU = "live_cpu"
    LIVE_GPU = "live_gpu"
    CROWDED = "crowded"                      # dense goshala/shed footage -- see PRESETS comment
    CROWDED_HD = "crowded_hd"                # CROWDED, but inferring at the source's own resolution band


@dataclass
class PipelineConfig:
    """Runtime config for a single video processing job."""

    # model
    model_path: str = "yolo11s.pt"
    img_size: int = 640
    confidence: float = 0.35                 # conf floor -- raised from 0.25 to kill low-confidence duplicate boxes
    nms_iou: float = 0.45                    # NMS IoU threshold -- removes overlapping detections on the same animal
    half: bool = False                       # FP16

    # sampling
    vid_stride: int = 2                      # frame-skip

    # tracking
    use_tracking: bool = True
    tracker_yaml: str = str(BASE_DIR / "trackers" / "botsort_cattle_fast.yaml")

    # class filter
    filter_cattle_classes: bool = True

    # stable-ID mapper
    stable_id_iou_thresh: float = 0.15
    stable_id_memory_frames: int = 90
    min_frames_visible: int = 5              # drop flicker IDs seen in fewer frames than this

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
        # switched from botsort_cattle_fast.yaml (with_reid: false) — appearance
        # ReID re-identifies cattle after occlusion, which is what the flicker/
        # duplicate-ID problem on real footage needed. "Fast" now means only
        # img_size=640/vid_stride=2, not "no ReID" — accept the throughput cost.
        tracker_yaml=str(BASE_DIR / "trackers" / "botsort_cattle.yaml"),
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
    # For dense goshala/shed footage: a long feeding-trough row shot with
    # cattle standing shoulder-to-shoulder, heavily occluded. FAST's
    # confidence=0.35/nms_iou=0.45 (tuned against a SPARSE open-field herd
    # to kill duplicate-box artifacts on a single animal) badly undercounts
    # here — measured on a real 32s dense clip: peak_cattle_in_frame=22
    # against a visual estimate of 35-55+ actually in frame. Root cause,
    # confirmed empirically (scratch_detection_tune.py, scratch_pipeline_tune.py
    # — not committed, see CLAUDE.md): confidence=0.35 was filtering out
    # genuine but lower-confidence detections of small/rear-view/occluded
    # animals further back in the row -- NOT a false-positive problem, so
    # loosening it doesn't reintroduce noise. Lowering to 0.20 (with nms_iou
    # relaxed to 0.55 so adjacent, genuinely-distinct animals standing close
    # together stop getting merged into one box) raised peak count 22->26 on
    # the dense clip with total_detections/avg_confidence scaling
    # proportionately (not exploding, i.e. real detections, not noise), and
    # produced a smaller, same-direction gain (14->16 peak) on a second,
    # less-crowded clip -- no false-positive blow-up there either.
    # conf=0.15 got one step closer (peak=26 too, same ceiling) but tripped
    # ultralytics' "NMS time limit exceeded" warning on the dense clip; 0.20
    # reaches the same peak without that reliability risk. Tried yolo11m
    # (BALANCED/ACCURATE's model) at these same thresholds expecting a
    # further gain -- it did NOT help (peak=23, slightly WORSE than
    # yolo11s's 26, at the same processing cost) — model size isn't the
    # bottleneck here, so this preset stays on yolo11s.
    #
    # Still an HONEST GAP, not a full fix: 26 vs a ~35-55 visual estimate on
    # the same clip means real animals are still being missed even after
    # this tuning -- this looks like it's approaching yolo11s's real
    # detection ceiling on this level of occlusion, which threshold tuning
    # alone can't fully close. A model fine-tuned on genuinely crowded
    # barn footage (this generic COCO-pretrained model was never trained on
    # shoulder-to-shoulder cattle) is the next lever if more accuracy is
    # needed here — out of scope for a config change.
    #
    # Deliberately NOT applied to FAST's defaults: the original 0.35/0.45
    # was tuned against a genuinely different, sparse-herd clip (see
    # CLAUDE.md's CCTV tuning history) that is no longer available to
    # re-verify against, so changing the global default risks silently
    # undoing that fix. This is an opt-in choice instead, same principle as
    # showing peak vs. tracked counts side by side rather than picking one.
    Preset.CROWDED: dict(
        model_path="yolo11s.pt",
        img_size=640,
        vid_stride=2,
        confidence=0.20,
        nms_iou=0.55,
        tracker_yaml=str(BASE_DIR / "trackers" / "botsort_cattle.yaml"),
        half=False,
    ),
    # CROWDED's thresholds, but inferring at 1920 instead of 640. Use this for
    # any source above ~1080p -- i.e. every real government CCTV feed we have.
    #
    # The "honest gap" recorded on Preset.CROWDED above (peak=26 against a
    # ~35-55 visual estimate) was diagnosed there as yolo11s approaching its
    # detection ceiling. That diagnosis was WRONG, and this preset is the
    # correction. The ceiling was never the model -- it was that a 2560x1440
    # frame is scaled to 640 before inference, a 4x reduction, which drops a
    # background animal from ~120px tall to ~30px, under what the detector can
    # resolve. Every knob tuned on CROWDED (confidence, nms_iou, model size)
    # was tuned downstream of that loss, which is why they all plateaued
    # together at the same peak=26.
    #
    # Measured 2026-08-10 on 23 real government goshala clips (2560x1440 @
    # 25fps, CP Plus IP cameras, 4 distinct sheds incl. one 02:00 night scene):
    #   - Raw detections, 60 frames spanning 20 clips of shed 1: +25.4% at
    #     1920 vs 640, EVERY clip positive, extra boxes landing at a median
    #     0.80 confidence -- recovered animals, not scraped-barrel noise.
    #   - Full pipeline peak_cattle_in_frame, the 3 wider-angle sheds:
    #     23->46, 20->43, 21->28. The two wide sheds literally DOUBLE.
    #   - Hand count on one night frame: ~50 animals visible; 640 found 17
    #     (34%), 1920 found 37 (74%).
    #   - average_confidence rises too (0.45->0.56, 0.48->0.53, 0.51->0.64):
    #     more detections AND more certain, which is not what adding false
    #     positives looks like.
    #
    # The gain scales with how WIDE the camera is: a narrow shed where animals
    # already fill the frame gains only 1.3x, a wide shed covering a whole
    # feeding row gains 2.2x. That is the tell that this is a pixels-on-target
    # problem, not a threshold problem.
    #
    # Cost is small because a 4060-class GPU is idling at 640: inference
    # 45ms -> 64ms (1.4x), end-to-end throughput ~12.5 -> ~9.5 fps, still
    # comfortably faster than real time for a 25fps source.
    #
    # Deliberately opt-in rather than a change to CROWDED's img_size, for the
    # same reason CROWDED itself was added opt-in rather than retuning FAST:
    # CROWDED's numbers were tuned against a clip that is no longer available
    # to re-verify against. Callers on genuinely low-resolution feeds should
    # stay on CROWDED -- inferring at 1920 on a 640x480 sub-stream only
    # upscales pixels that were never recorded and wastes the compute.
    Preset.CROWDED_HD: dict(
        model_path="yolo11s.pt",
        img_size=1920,
        vid_stride=2,
        confidence=0.20,
        nms_iou=0.55,
        tracker_yaml=str(BASE_DIR / "trackers" / "botsort_cattle.yaml"),
        half=False,
    ),
}


def make_config(preset: Preset = Preset.FAST, **overrides) -> PipelineConfig:
    """Build a PipelineConfig from a preset + any per-job overrides."""
    base = PRESETS.get(preset, {})
    merged = {**base, **overrides}
    return PipelineConfig(**merged)
