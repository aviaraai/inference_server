from cctv.config import make_config, Preset
from cctv.pipeline import run_pipeline

video_path = r"C:\Users\Asus\Downloads\document_6316777104247628263.mp4"

configs = [
    ("yolo11s conf=0.20 iou=0.55", dict(model_path="yolo11s.pt", confidence=0.20, nms_iou=0.55)),
    ("yolo11m conf=0.20 iou=0.55", dict(model_path="yolo11m.pt", confidence=0.20, nms_iou=0.55)),
]

for label, overrides in configs:
    cfg = make_config(Preset.FAST, **overrides)
    summary = run_pipeline(video_path, cfg, job_id=f"modeltune_{label[:6]}")
    print(f"{label}: peak={summary.max_cattle_in_frame} tracked={summary.unique_tracked_cattle} "
          f"total_detections={summary.total_detections} avg_conf={summary.average_confidence} "
          f"time={summary.processing_seconds}s")
