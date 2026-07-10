"""
debug_yolo.py — Diagnose YOLO detection failures.

Runs YOLO at multiple confidence thresholds and image sizes to understand
why 0 detections are returned for certain images.
"""

import sys
import cv2
import numpy as np
from ultralytics import YOLO

# Load model
model = YOLO("yolov8s.pt")

# COCO class names for reference
COCO_NAMES = model.names  # dict {0: 'person', 1: 'bicycle', ...}

# Target image — pass as CLI arg or use test_buffalo.jpg
img_path = sys.argv[1] if len(sys.argv) > 1 else "test_buffalo.jpg"
img = cv2.imread(img_path)
if img is None:
    print(f"ERROR: Cannot read {img_path}")
    sys.exit(1)

h, w = img.shape[:2]
print(f"\n{'='*72}")
print(f"  Image: {img_path}")
print(f"  Size:  {w}x{h} ({w*h:,} px)")
print(f"{'='*72}")

# ── Test 1: Run with VERY low conf to see ALL detections ──────────────
print(f"\n[Test 1] YOLO conf=0.01 (show everything YOLO sees)")
print(f"-" * 72)
results = model(img, conf=0.01, verbose=False)[0]
print(f"  Total detections: {len(results.boxes)}")
if len(results.boxes) > 0:
    for i, box in enumerate(results.boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [int(round(float(v))) for v in box.xyxy[0]]
        area_pct = max(0, x2-x1) * max(0, y2-y1) / (w * h)
        name = COCO_NAMES.get(cls, f"class_{cls}")
        print(f"  [{i+1}] class={cls} ({name}) conf={conf:.4f} "
              f"bbox=({x1},{y1},{x2},{y2}) area_pct={area_pct:.4f}")
else:
    print("  >>> ZERO detections even at conf=0.01!")

# ── Test 2: Run at your production threshold ───────────────────────────
print(f"\n[Test 2] YOLO conf=0.30 (production threshold)")
print(f"-" * 72)
results2 = model(img, conf=0.30, verbose=False)[0]
print(f"  Total detections: {len(results2.boxes)}")
for i, box in enumerate(results2.boxes):
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    name = COCO_NAMES.get(cls, f"class_{cls}")
    print(f"  [{i+1}] class={cls} ({name}) conf={conf:.4f}")

# ── Test 3: Resize to 640 (YOLO default) and retry ────────────────────
print(f"\n[Test 3] Resized to 640px (YOLO default input size) conf=0.01")
print(f"-" * 72)
scale = 640 / max(h, w)
img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))
rh, rw = img_resized.shape[:2]
print(f"  Resized: {rw}x{rh}")
results3 = model(img_resized, conf=0.01, verbose=False)[0]
print(f"  Total detections: {len(results3.boxes)}")
for i, box in enumerate(results3.boxes):
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    name = COCO_NAMES.get(cls, f"class_{cls}")
    print(f"  [{i+1}] class={cls} ({name}) conf={conf:.4f}")

# ── Test 4: Set explicit imgsz=1280 for high-res images ───────────────
print(f"\n[Test 4] YOLO with imgsz=1280, conf=0.01")
print(f"-" * 72)
results4 = model(img, conf=0.01, imgsz=1280, verbose=False)[0]
print(f"  Total detections: {len(results4.boxes)}")
for i, box in enumerate(results4.boxes):
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    name = COCO_NAMES.get(cls, f"class_{cls}")
    print(f"  [{i+1}] class={cls} ({name}) conf={conf:.4f}")

print(f"\n{'='*72}")
print("  Done.")
print(f"{'='*72}\n")
