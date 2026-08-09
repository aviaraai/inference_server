"""
Scratch: sweep stable_id_iou_thresh / stable_id_memory_frames on the two real
tuning clips already used for the confidence/nms_iou tuning, to check whether
the unique_tracked_cattle vs max_cattle_in_frame fragmentation gap responds to
these parameters at all, before touching any defaults in config.py.

Detection settings held at CROWDED's already-tuned values (confidence=0.20,
nms_iou=0.55) so this isolates the stable-ID layer specifically.
"""
from dataclasses import replace
from cctv.config import make_config, Preset
from cctv.pipeline import run_pipeline

VIDEOS = {
    "dense_goshala": r"C:\Users\Asus\Downloads\document_6316777104247628263.mp4",
    "moderate": r"C:\Users\Asus\Downloads\WhatsApp Video 2026-08-04 at 5.51.31 PM.mp4",
}

SWEEP = [
    ("baseline (iou=0.15 mem=90)", dict(stable_id_iou_thresh=0.15, stable_id_memory_frames=90)),
    ("looser iou (iou=0.08 mem=90)", dict(stable_id_iou_thresh=0.08, stable_id_memory_frames=90)),
    ("longer memory (iou=0.15 mem=150)", dict(stable_id_iou_thresh=0.15, stable_id_memory_frames=150)),
    ("both (iou=0.08 mem=150)", dict(stable_id_iou_thresh=0.08, stable_id_memory_frames=150)),
]

for video_name, video_path in VIDEOS.items():
    print(f"\n=== {video_name} ===")
    for label, overrides in SWEEP:
        cfg = make_config(Preset.FAST, confidence=0.20, nms_iou=0.55, **overrides)
        job_id = f"stableidtune_{video_name[:4]}_{label[:6].replace(' ', '').replace('=', '').replace('(', '')}"
        summary = run_pipeline(video_path, cfg, job_id=job_id)
        gap = summary.unique_tracked_cattle - summary.max_cattle_in_frame
        print(
            f"{label}: peak={summary.max_cattle_in_frame} "
            f"tracked={summary.unique_tracked_cattle} gap={gap} "
            f"time={summary.processing_seconds:.1f}s"
        )
