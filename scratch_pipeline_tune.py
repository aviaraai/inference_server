"""
Run the REAL pipeline (detection + tracking + stable-ID flicker filter,
not just raw per-frame YOLO) at different confidence/nms_iou settings, on
both the dense goshala clip and a sparser one, to find a config that
genuinely improves peak-count accuracy without misbehaving on lighter
scenes.
"""
import sys
from dataclasses import replace
from cctv.config import make_config, Preset
from cctv.pipeline import run_pipeline

VIDEOS = {
    "dense_goshala": r"C:\Users\Asus\Downloads\document_6316777104247628263.mp4",
    "moderate": r"C:\Users\Asus\Downloads\WhatsApp Video 2026-08-04 at 5.51.31 PM.mp4",
}

CONFIGS = [
    ("baseline conf=0.35 iou=0.45", dict(confidence=0.35, nms_iou=0.45)),
    ("conf=0.15 iou=0.45", dict(confidence=0.15, nms_iou=0.45)),
    ("conf=0.15 iou=0.6", dict(confidence=0.15, nms_iou=0.6)),
    ("conf=0.20 iou=0.55", dict(confidence=0.20, nms_iou=0.55)),
]

which_video = sys.argv[1] if len(sys.argv) > 1 else "dense_goshala"
video_path = VIDEOS[which_video]
print(f"=== {which_video}: {video_path} ===")

for label, overrides in CONFIGS:
    cfg = make_config(Preset.FAST, **overrides)
    summary = run_pipeline(video_path, cfg, job_id=f"tune_{which_video}_{label[:10].replace(' ','_').replace('=','')}")
    print(f"{label}: peak={summary.max_cattle_in_frame} tracked={summary.unique_tracked_cattle} "
          f"total_detections={summary.total_detections} avg_conf={summary.average_confidence} "
          f"time={summary.processing_seconds}s")
