"""
debug_yolo_docker.py — Run inside Docker to diagnose YOLO detection failures.

Copy this to the container and run:
  docker exec inference-server.dev python /app/debug_yolo_docker.py

Or save an image into the container first:
  docker cp cow.png inference-server.dev:/tmp/cow.png
  docker exec inference-server.dev python /app/debug_yolo_docker.py /tmp/cow.png
"""

import sys
import os
import cv2
import numpy as np
import torch

print(f"Python: {sys.version}")
print(f"OpenCV: {cv2.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

from ultralytics import YOLO
import ultralytics
print(f"Ultralytics: {ultralytics.__version__}")

# Load YOLO
yolo_path = os.getenv("YOLO_MODEL_PATH", "yolov8s.pt")
print(f"\nLoading YOLO from: {yolo_path}")
model = YOLO(yolo_path)
print(f"Model device: {next(model.model.parameters()).device}")
print(f"Model names (cattle): { {k:v for k,v in model.names.items() if k in [17,18,19,20,21]} }")

# Test with a synthetic image if no real image provided
if len(sys.argv) > 1:
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: Cannot read {img_path}")
        sys.exit(1)
else:
    # Create a test: 3072x4096 black image (to check if YOLO runs at all)
    print("\nNo image provided. Creating synthetic 3072x4096 test image...")
    img = np.zeros((4096, 3072, 3), dtype=np.uint8)

h, w = img.shape[:2]
print(f"\nImage: {w}x{h} ({w*h:,} px)")
print(f"dtype: {img.dtype}, channels: {img.shape[2]}")

# Run YOLO
print(f"\n--- Running YOLO (conf=0.01) ---")
results = model(img, conf=0.01, verbose=True)[0]
print(f"Total detections: {len(results.boxes)}")
for box in results.boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    name = model.names.get(cls, f"class_{cls}")
    x1, y1, x2, y2 = [int(round(float(v))) for v in box.xyxy[0]]
    print(f"  class={cls} ({name}) conf={conf:.4f} bbox=({x1},{y1},{x2},{y2})")

print("\n--- Running YOLO (conf=0.30, production) ---")
results2 = model(img, conf=0.30, verbose=False)[0]
print(f"Total detections: {len(results2.boxes)}")
for box in results2.boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    name = model.names.get(cls, f"class_{cls}")
    print(f"  class={cls} ({name}) conf={conf:.4f}")

print("\nDone.")
