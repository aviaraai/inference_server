"""
foo.py — Batch test all WhatsApp buffalo images through crop_cattle().
"""

import os
import cv2
import pipeline.yolo_crop as yc

yc.load_yolo()

IMAGE_DIR = r"d:\Group Projects\inference_server"
WA_IMAGES = [
    "a53ef935-b6e4-45d9-850e-8a40afd03b4b.jpeg",
    "4517254f-904d-4423-97c2-beca494c57f6.jpeg",
    "63ca96ed-15fa-4127-9bc5-7675fc396ec2.jpeg",
    "0094fc1b-8607-4a45-a50a-e0e6462f9b0d.jpeg",
    "ec98d905-bae5-4b01-8b59-29c4a509127a.jpeg",
    "25c187aa-5b0c-46a6-8ef5-57abcdeb993a.jpeg",
    "b0afb529-41a1-4e51-86d5-6fbb2bd10c7b.jpeg",
    "ff701f74-611b-43a7-bb7c-18f6373a6587.jpeg",
    "2fd3b8ca-7775-417c-8582-f2bf97eb7bfc.jpeg",
    "9791a82b-65e7-4195-b87a-76a8c9fddbe6.jpeg",
    "c8fdb10a-c8de-4865-83ff-2b129ea250cd.jpeg",
    "f811e82d-8341-434a-a464-4b879de0a356.jpeg",
]

PASS = FAIL = 0
print("=" * 72)
print(f"  {'#':<3}  {'File':<46}  {'Status':<26}  {'Conf':>5}")
print("=" * 72)

for i, fname in enumerate(WA_IMAGES):
    path = os.path.join(IMAGE_DIR, fname)
    img  = cv2.imread(path)
    if img is None:
        print(f"  {i+1:<3}  {fname:<46}  CANNOT READ")
        continue

    crop, status, conf = yc.crop_cattle(img)
    ok  = status in ("OK", "FULL_IMAGE")
    tag = "PASS" if ok else "FAIL"
    print(f"  {i+1:<3}  {fname:<46}  {tag} [{status:<20}]  {conf:.3f}")

    if ok:
        PASS += 1
        out = os.path.join(IMAGE_DIR, f"crop_{i+1:02d}.jpg")
        cv2.imwrite(out, crop if crop is not None else img)
    else:
        FAIL += 1

print("=" * 72)
print(f"\n  PASSED : {PASS}/{len(WA_IMAGES)}")
print(f"  FAILED : {FAIL}/{len(WA_IMAGES)}")
