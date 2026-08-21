"""
scratch_cctv_muzzle_crop_test.py — verify cctv/muzzle_crop.py against real
goshala clips (cctv/clips/), one at a time, reporting per-clip and aggregate
extraction success rate.

Throwaway investigation script, same convention as this repo's other
scratch_*.py files — not part of the test suite, not imported by anything.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from cctv.config import RUNS_DIR, Preset, make_config
from cctv.pipeline import run_pipeline
from pipeline.yolo_crop import load_yolo, warmup_yolo

CLIPS_DIR = Path(__file__).resolve().parent / "cctv" / "clips"


def main() -> None:
    clips = sorted(CLIPS_DIR.glob("*.mp4"))
    if not clips:
        print(f"No clips found under {CLIPS_DIR}")
        sys.exit(1)

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(clips)
    clips = clips[:limit]

    # crop_cattle() (pipeline/yolo_crop.py) needs its own YOLO loaded — this
    # is normally done once at server startup (main.py's lifespan); this
    # script drives the pipeline directly, outside the FastAPI app, so it
    # has to do that itself.
    print("Loading crop_cattle's YOLO model...")
    load_yolo()
    warmup_yolo()

    total_qualifying = 0
    total_ok = 0
    status_counts: dict[str, int] = {}
    per_clip_rows: list[tuple[str, int, int, float]] = []

    for clip in clips:
        job_id = f"muzzlecroptest_{clip.stem}"
        out_dir = RUNS_DIR / job_id
        cfg = make_config(Preset.FAST, output_dir=str(out_dir), enable_analytics=False)

        t0 = time.perf_counter()
        try:
            summary = run_pipeline(clip, cfg, job_id=job_id)
        except Exception as e:
            print(f"{clip.name}: PIPELINE FAILED — {e}")
            continue
        elapsed = time.perf_counter() - t0

        crops = summary.muzzle_crops
        n_qualifying = len(crops)
        n_ok = sum(1 for r in crops.values() if r.status == "OK")
        total_qualifying += n_qualifying
        total_ok += n_ok

        for r in crops.values():
            status_counts[r.status] = status_counts.get(r.status, 0) + 1

        rate = (n_ok / n_qualifying * 100) if n_qualifying else 0.0
        per_clip_rows.append((clip.name, n_ok, n_qualifying, rate))
        print(
            f"{clip.name}: {n_ok}/{n_qualifying} crops extracted "
            f"({rate:.0f}%), {elapsed:.1f}s, "
            f"unique_tracked_cattle={summary.unique_tracked_cattle}"
        )
        for sid, r in sorted(crops.items()):
            if r.status != "OK":
                print(
                    f"    stable_id={sid}: {r.status} "
                    f"(tracker_conf={r.source_confidence:.2f}, "
                    f"frame={r.source_frame_idx})"
                )

    print("\n" + "=" * 60)
    print("AGGREGATE")
    print("=" * 60)
    for clip_name, n_ok, n_qual, rate in per_clip_rows:
        print(f"  {clip_name:30s} {n_ok:3d}/{n_qual:<3d} ({rate:5.1f}%)")

    overall_rate = (total_ok / total_qualifying * 100) if total_qualifying else 0.0
    print(f"\nTotal qualifying tracked cattle: {total_qualifying}")
    print(f"Total crops successfully extracted: {total_ok}")
    print(f"Overall success rate: {overall_rate:.1f}%")
    print(f"\nStatus breakdown: {status_counts}")


if __name__ == "__main__":
    main()
