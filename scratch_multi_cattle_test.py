"""
Replay the exact 3 muzzle photos that were reported failing with
RECAPTURE_MULTI_CATTLE through the real current pipeline (quality_check ->
crop_cattle -> quality_check_cv2), to check whether the multi-cattle
dominant-subject fix actually resolves them now.
"""
from helpers import _decode_image
from pipeline.quality import quality_check, quality_check_cv2
from pipeline.yolo_crop import crop_cattle, load_yolo

load_yolo()  # must actually load the real model, not fall back to FULL_IMAGE_NO_YOLO

FILES = {
    "muzzle_1": r"C:\Users\Asus\.claude\image-cache\45070f9b-0eff-42ed-a955-a1a985a740d0\4.png",
    "muzzle_2": r"C:\Users\Asus\.claude\image-cache\45070f9b-0eff-42ed-a955-a1a985a740d0\5.png",
    "muzzle_3": r"C:\Users\Asus\.claude\image-cache\45070f9b-0eff-42ed-a955-a1a985a740d0\6.png",
}

errors = []
for label, path in FILES.items():
    with open(path, "rb") as f:
        raw = f.read()

    q_status, q_reason = quality_check(raw)
    if q_status != "GOOD":
        print(f"{label}: quality_check FAILED -> {q_reason}")
        errors.append(f"{label}: {q_reason}")
        continue
    print(f"{label}: quality_check OK")

    img_bgr = _decode_image(raw)
    crop, det_status, det_conf = crop_cattle(img_bgr)
    if crop is None:
        print(f"{label}: crop_cattle FAILED -> {det_status} (conf={det_conf})")
        errors.append(f"{label}: {det_status}")
        continue
    print(f"{label}: crop_cattle OK -> status={det_status} conf={det_conf:.3f} crop_shape={crop.shape}")

    crop_status, crop_reason = quality_check_cv2(crop)
    if crop_status != "GOOD":
        print(f"{label}: quality_check_cv2 on crop FAILED -> {crop_reason}")
        errors.append(f"{label}_crop: {crop_reason}")
        continue
    print(f"{label}: quality_check_cv2 on crop OK")

print()
if errors:
    print("RESULT: would still be REJECTED —", "; ".join(errors))
else:
    print("RESULT: all 3 muzzle photos would now PASS")
