"""
Experiment: run the CCTV detector directly on the densest ground-truth
frames at different confidence/NMS settings to see how many boxes come
out, compared against a rough visual count, to find out WHY peak count
(22) looks low against the frames.
"""
import sys
from ultralytics import YOLO
from cctv.config import CATTLE_TERMS

model = YOLO("yolo11s.pt")

names = model.names
cattle_ids = {cid for cid, name in names.items() if name.lower() in CATTLE_TERMS}
print("cattle_ids:", cattle_ids, "names matched:", [names[c] for c in cattle_ids])

FRAMES = [
    "cctv/runs/ground_truth_frames/f_020.jpg",
    "cctv/runs/ground_truth_frames/f_030.jpg",
    "cctv/runs/ground_truth_frames/f_040.jpg",
]

configs = [
    ("current (conf=0.35, iou=0.45, imgsz=640)", dict(conf=0.35, iou=0.45, imgsz=640)),
    ("lower conf (conf=0.15, iou=0.45, imgsz=640)", dict(conf=0.15, iou=0.45, imgsz=640)),
    ("lower conf + higher iou tolerance (conf=0.15, iou=0.6, imgsz=640)", dict(conf=0.15, iou=0.6, imgsz=640)),
    ("lower conf + bigger imgsz (conf=0.15, iou=0.6, imgsz=1280)", dict(conf=0.15, iou=0.6, imgsz=1280)),
]

for frame in FRAMES:
    print(f"\n=== {frame} ===")
    for label, kwargs in configs:
        results = model.predict(frame, classes=list(cattle_ids), verbose=False, **kwargs)
        n = len(results[0].boxes) if results[0].boxes is not None else 0
        confs = results[0].boxes.conf.tolist() if n else []
        print(f"  {label}: {n} boxes"
              + (f" (conf range {min(confs):.2f}-{max(confs):.2f})" if confs else ""))
