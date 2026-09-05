"""
scratch_keypoint_color_comparison.py — step 3 of the keypoint-anchored
forehead color task: run BOTH the existing whole-bbox body_color method and
the new keypoint-anchored method on the same real images, and report where
they disagree.

NOT genuine field-registration photos. The 218 real Uttarakhand registration
photos this kind of validation should use are not on disk (see CLAUDE.md /
scratch_pose_wire_smoketest.py -- this is now the SECOND time that dataset
has been unavailable for a validation task in this project; flagged
separately, not silently worked around here). Substitutes used instead, same
honesty standard as that smoke test:

  A. model_training/pose_model's own Roboflow valid/images split (81 real
     cattle face photos, IN-domain for the pose model -- the closest
     available thing to a real front registration photo). Caveat: this
     exact checkpoint (appstorage/Models/pose_model/best.pt, per its
     PROVENANCE.txt) used this same 81-image split for early-stopping /
     best-epoch selection -- not literally trained on, but not a fully
     blind holdout either. Fine for THIS purpose (comparing two downstream
     color methods against each other, not measuring pose accuracy), worth
     knowing if reused for anything else.
  B. experiments/cctv_check/raw_frame_*.jpg -- real goshala CCTV frames,
     out-of-domain angles/distances/lighting for the pose model. Only 3
     exist locally.

Deliberately EXCLUDED: cctv/runs/*/muzzle_crops/*.jpg (961 available) --
those are muzzle-only crops per cctv/muzzle_crop.py, with no eyes/forehead
in frame at all. Not a valid input for either method compared here.

Usage: .venv/Scripts/python.exe scratch_keypoint_color_comparison.py
"""

from __future__ import annotations

import glob
import os

os.environ.setdefault("YOLO_MODEL_PATH", "yolov8s.pt")
os.environ.setdefault(
    "POSE_MODEL_PATH", os.path.join("appstorage", "Models", "pose_model", "best.pt")
)

import cv2  # noqa: E402

from pipeline.color import RuleBasedColorExtractor  # noqa: E402
from pipeline.keypoint_color import extract_keypoint_forehead_color  # noqa: E402
from pipeline.pose import load_pose_model, pose_available, warmup_pose_model  # noqa: E402
from pipeline.yolo_crop import load_yolo, warmup_yolo  # noqa: E402

POSE_MODEL_VALID_IMAGES = os.path.join(
    "..", "model_training", "pose_model",
    "Animal Face Keypoint Detection.yolo26 (4)", "valid", "images",
)
CCTV_RAW_FRAMES = os.path.join("experiments", "cctv_check", "raw_frame_*.jpg")


def _gather_images() -> list[tuple[str, str]]:
    """Returns (group_name, path) pairs."""
    out: list[tuple[str, str]] = []
    for path in sorted(glob.glob(os.path.join(POSE_MODEL_VALID_IMAGES, "*.jpg"))):
        out.append(("pose_model_valid", path))
    for path in sorted(glob.glob(CCTV_RAW_FRAMES)):
        out.append(("cctv_raw_frame", path))
    return out


def main() -> None:
    load_yolo(os.environ["YOLO_MODEL_PATH"])
    warmup_yolo()
    load_pose_model(os.environ["POSE_MODEL_PATH"])
    warmup_pose_model()

    print(f"pose model loaded: {pose_available()}")
    if not pose_available():
        print("Pose model failed to load -- every keypoint reading will be UNKNOWN. Aborting.")
        return

    color_extractor = RuleBasedColorExtractor()
    print(f"wildlife/color available: {color_extractor.available}")

    images = _gather_images()
    print(f"\n{len(images)} real substitute images found "
          f"({sum(1 for g, _ in images if g == 'pose_model_valid')} pose_model_valid, "
          f"{sum(1 for g, _ in images if g == 'cctv_raw_frame')} cctv_raw_frame)\n")

    rows = []
    for group, path in images:
        img = cv2.imread(path)
        if img is None:
            print(f"SKIP (unreadable): {path}")
            continue

        old = color_extractor.extract_body(img)
        new = extract_keypoint_forehead_color(img)
        rows.append((group, os.path.basename(path), old, new))

    n = len(rows)
    n_new_ok = sum(1 for *_ , new in rows if new["status"] == "OK")
    n_new_partial = sum(1 for *_ , new in rows if new["status"] == "PARTIAL")
    n_new_unknown = sum(1 for *_ , new in rows if new["status"] == "UNKNOWN")
    n_both_have_label = sum(1 for *_ , old, new in rows if old["label"] != "UNKNOWN" and new["label"] != "UNKNOWN")
    n_agree = sum(
        1 for *_ , old, new in rows
        if old["label"] != "UNKNOWN" and new["label"] != "UNKNOWN" and old["label"] == new["label"]
    )

    print("=" * 88)
    print("COVERAGE (how often the new signal is usable at all on this substitute set)")
    print("=" * 88)
    print(f"  total images:                {n}")
    print(f"  new method OK:               {n_new_ok}  ({n_new_ok/n:.0%})" if n else "")
    print(f"  new method PARTIAL:          {n_new_partial}  ({n_new_partial/n:.0%})" if n else "")
    print(f"  new method UNKNOWN:          {n_new_unknown}  ({n_new_unknown/n:.0%})" if n else "")

    print()
    print("=" * 88)
    print("AGREEMENT (only where BOTH methods produced a non-UNKNOWN label)")
    print("=" * 88)
    print(f"  both usable:                 {n_both_have_label}")
    print(f"  labels agree:                {n_agree}  ({n_agree/n_both_have_label:.0%})" if n_both_have_label else "  (no overlap -- see coverage above)")

    print()
    print("=" * 88)
    print("PER-IMAGE DISAGREEMENTS (both methods gave a label, and they differ)")
    print("=" * 88)
    disagreements = [
        (g, name, old, new) for g, name, old, new in rows
        if old["label"] != "UNKNOWN" and new["label"] != "UNKNOWN" and old["label"] != new["label"]
    ]
    if not disagreements:
        print("  none")
    for g, name, old, new in disagreements:
        print(
            f"  [{g}] {name}: OLD(whole-bbox)={old['label']}({old['confidence']:.2f}) "
            f"vs NEW(keypoint)={new['label']}({new['confidence']:.2f}) "
            f"[new status={new['status']}, ruler={new['inter_eye_distance_px']}]"
        )

    print()
    print("=" * 88)
    print("PER-IMAGE, FULL DUMP")
    print("=" * 88)
    for g, name, old, new in rows:
        flag = "AGREE" if (old["label"] == new["label"] and old["label"] != "UNKNOWN") else (
            "DISAGREE" if (old["label"] != "UNKNOWN" and new["label"] != "UNKNOWN") else "N/A"
        )
        print(
            f"  [{g:16s}] {name:40s} old={old['label']:8s}({old['confidence']:.2f}) "
            f"new={new['label']:8s}({new['confidence']:.2f}) status={new['status']:8s} "
            f"ruler={str(new['inter_eye_distance_px']):>8s}  {flag}"
        )


if __name__ == "__main__":
    main()
