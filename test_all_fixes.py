"""
test_all_fixes.py — Compare detection rates across all fix strategies
on every test image in the project.
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pipeline.yolo_crop import _enhance_for_detection

model = YOLO("yolov8s.pt")
DIR = r"d:\Group Projects\inference_server"
CATTLE = {17, 18, 19, 20, 21}

images = [f for f in os.listdir(DIR) if f.endswith(".jpeg")]
print(f"Testing {len(images)} images...\n")

header = f"  {'Image':<45} {'Old(0.25)':<12} {'New(0.10)':<12} {'CLAHE+0.10':<12}"
print(header)
print("  " + "-" * 80)

old_fail = new_fail = clahe_fail = 0

for fname in images:
    img = cv2.imread(os.path.join(DIR, fname))
    if img is None:
        continue

    # Old behavior: YOLO default conf=0.25
    r1 = model(img, verbose=False)[0]
    c1 = len([b for b in r1.boxes if int(b.cls[0]) in CATTLE])

    # New fix: conf=0.10
    r2 = model(img, conf=0.10, verbose=False)[0]
    c2 = len([b for b in r2.boxes if int(b.cls[0]) in CATTLE])

    # CLAHE + conf=0.10
    enh = _enhance_for_detection(img)
    r3 = model(enh, conf=0.10, verbose=False)[0]
    c3 = len([b for b in r3.boxes if int(b.cls[0]) in CATTLE])

    flag = " << RECOVERED" if c1 == 0 and (c2 > 0 or c3 > 0) else ""
    if c1 == 0: old_fail += 1
    if c2 == 0: new_fail += 1
    if c3 == 0: clahe_fail += 1

    print(f"  {fname:<45} {c1:<12} {c2:<12} {c3:<12}{flag}")

print(f"\n  Failures:  Old={old_fail}  New={new_fail}  CLAHE={clahe_fail}  (out of {len(images)})")
