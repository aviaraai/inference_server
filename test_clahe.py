"""
test_clahe.py — Test that CLAHE enhancement helps YOLO detect
dark/low-contrast cattle that the raw image misses.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from pipeline.yolo_crop import _enhance_for_detection

model = YOLO("yolov8s.pt")

# Load the test image
img = cv2.imread(r"C:\Users\Asus\Downloads\cow.png")
h, w = img.shape[:2]
print(f"Original: {w}x{h}")

# Simulate a low-contrast camera: darken + reduce contrast
# This mimics what bad phone cameras produce for dark cattle
dark = (img.astype(np.float32) * 0.35 + 15).clip(0, 255).astype(np.uint8)
cv2.imwrite("test_dark.jpg", dark)

print(f"\n--- Original image ---")
r1 = model(img, conf=0.01, verbose=False)[0]
print(f"  Detections: {len(r1.boxes)}")
for b in r1.boxes:
    cls = int(b.cls[0])
    conf = float(b.conf[0])
    print(f"  class={cls} ({model.names[cls]}) conf={conf:.4f}")

print(f"\n--- Darkened image (simulating bad phone camera) ---")
r2 = model(dark, conf=0.01, verbose=False)[0]
print(f"  Detections: {len(r2.boxes)}")
for b in r2.boxes:
    cls = int(b.cls[0])
    conf = float(b.conf[0])
    print(f"  class={cls} ({model.names[cls]}) conf={conf:.4f}")

print(f"\n--- Darkened + CLAHE enhancement ---")
enhanced = _enhance_for_detection(dark)
cv2.imwrite("test_dark_clahe.jpg", enhanced)
r3 = model(enhanced, conf=0.01, verbose=False)[0]
print(f"  Detections: {len(r3.boxes)}")
for b in r3.boxes:
    cls = int(b.cls[0])
    conf = float(b.conf[0])
    print(f"  class={cls} ({model.names[cls]}) conf={conf:.4f}")

print(f"\n{'='*60}")
if len(r2.boxes) == 0 and len(r3.boxes) > 0:
    print("  ✅ CLAHE RECOVERED detection that raw image missed!")
elif len(r2.boxes) > 0 and len(r3.boxes) > 0:
    dark_conf = max(float(b.conf[0]) for b in r2.boxes if int(b.cls[0]) in {17,18,19,20,21})
    enh_conf = max(float(b.conf[0]) for b in r3.boxes if int(b.cls[0]) in {17,18,19,20,21})
    print(f"  Both detect. Dark conf={dark_conf:.4f} → Enhanced conf={enh_conf:.4f}")
else:
    print("  ⚠️  Neither detected — CLAHE alone may not be enough")
print(f"{'='*60}")
