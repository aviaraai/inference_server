"""
scripts/backfill_muzzle_cache.py — backfill the LightGlue muzzle crop/feature
cache (pipeline/muzzle_crop_cache.py) for animals registered before that
cache existed (commit 5c21646, 2026-08-14).

Confirmed live on the deployed host (2026-09-05): every animal registered
AFTER the cache shipped has a complete crop+features pair; nothing is
silently failing there. Animals registered BEFORE it shipped have nothing
cached at all -- not a bug, just missing coverage -- and get zero LightGlue
protection on every future search until backfilled. This script closes that
gap without needing to re-run /register.

Why this can't be a pure inference_server script: this service has no DB
connection and no object-storage credentials (see muzzle_crop_cache.py's own
module docstring) -- it has no way to know which faiss_ids exist or where
their source images live. go-apiserver has both, so it exposes them read-only
via GET /api/web/v1/debug/muzzle-embeddings (developer-role JWT required,
same auth the analytics dashboard uses) -- see
internal/database/debug/repository.go's MuzzleEmbeddings /
internal/server/web/handlers/debug/routes.go's listMuzzleEmbeddings on the
go-apiserver side. This script is the other half: it has the ML models
(crop_cattle, DISK/LightGlue) go-apiserver doesn't.

Designed to run ON THE DEPLOYED inference_server HOST (needs YOLO_MODEL_PATH
+ POSE_MODEL_PATH... actually just YOLO_MODEL_PATH for crop_cattle, and the
LightGlue verifier for feature extraction -- same env this service's own
/register handler runs under). Uses only the Python standard library for the
network calls (urllib, not requests/httpx) specifically so it has no
dependency beyond what this service's own venv already guarantees --
"standalone" means it must not need anything installed that isn't already
part of running inference_server itself.

SAFE TO RE-RUN. Idempotent: before doing anything, it lists what's already
in MUZZLE_CROP_CACHE_DIR and skips any faiss_id that already has BOTH its
crop and its features cached. A partial prior run (e.g. crashed halfway)
just picks up where it left off.

DRY-RUN BY DEFAULT. Reports exactly what it would do -- counts, and which
faiss_ids -- and does not download or write anything until run with
--execute. Read the report before adding that flag.

Usage:
    export GODHAAR_API_URL="https://<go-apiserver host>/api/web/v1"
    export GODHAAR_API_TOKEN="<developer-role JWT>"
    export YOLO_MODEL_PATH="/appstorage/Models/detection_crop/yolov8s.pt"

    .venv/Scripts/python.exe scripts/backfill_muzzle_cache.py          # report only
    .venv/Scripts/python.exe scripts/backfill_muzzle_cache.py --execute # actually write

GODHAAR_API_TOKEN: the same bearer token the analytics dashboard stores at
localStorage['godhaar.admin.auth'].access_token for a developer-role account
-- log into the dashboard once and copy it out, or mint one however the team
already does for scripted access. This script never sees or needs a password.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from godhaar.config import MUZZLE_CROP_CACHE_DIR  # noqa: E402
from pipeline.lightglue_verify import (  # noqa: E402
    LIGHTGLUE_TIEBREAKER_ENABLED,
    available as lightglue_available,
    extract_features_np,
    load_lightglue,
)
from pipeline.muzzle_crop_cache import (  # noqa: E402
    _features_path_for,
    _path_for,
    save_crop,
    save_features,
)
from pipeline.yolo_crop import crop_cattle, load_yolo  # noqa: E402


def _api_get(url: str, token: str) -> list[dict]:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_image(url: str) -> np.ndarray | None:
    """GET raw bytes from a presigned storage URL and decode as BGR. No auth
    header -- presigned URLs carry their own signed credential in the query
    string, same as every other place this codebase fetches one."""
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"    download failed: {e}")
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img  # None on decode failure, same contract as muzzle_crop_cache.load_crop


def _already_cached(faiss_id: int) -> tuple[bool, bool]:
    """(has_crop, has_features) — read directly off disk, not through
    load_crop/load_features, so a corrupt-but-present file still counts as
    "cached" here (matches what /search's tiebreaker would actually see:
    load_crop/load_features return None on a decode failure too, but that's
    a different problem from "never written" and backfilling over a corrupt
    file that happens to exist is out of scope for this pass)."""
    return _path_for(faiss_id).is_file(), _features_path_for(faiss_id).is_file()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="Actually download+write. Default is report-only.")
    parser.add_argument("--api-url", default=os.environ.get("GODHAAR_API_URL"), help="Base URL, e.g. https://host/api/web/v1")
    parser.add_argument("--token", default=os.environ.get("GODHAAR_API_TOKEN"), help="Developer-role bearer JWT")
    args = parser.parse_args()

    if not args.api_url or not args.token:
        parser.error("--api-url/--token (or GODHAAR_API_URL/GODHAAR_API_TOKEN) are required")

    yolo_path = os.environ.get("YOLO_MODEL_PATH")
    if not yolo_path:
        parser.error("YOLO_MODEL_PATH must be set (same value /register runs under) -- crop_cattle needs it")
    load_yolo(yolo_path)

    if not LIGHTGLUE_TIEBREAKER_ENABLED:
        print("WARNING: LIGHTGLUE_TIEBREAKER_ENABLED=false in this environment -- "
              "crops will still be backfilled, but features will not be (there would be "
              "nothing to verify them against anyway if the tiebreaker itself is off).")
    else:
        load_lightglue()
        if not lightglue_available():
            print("WARNING: LightGlue failed to load -- crops will be backfilled, features will not be.")

    print(f"Fetching muzzle embeddings from {args.api_url}/debug/muzzle-embeddings ...")
    rows = _api_get(f"{args.api_url}/debug/muzzle-embeddings", args.token)
    print(f"{len(rows)} registered muzzle embeddings on record.\n")

    needs_crop: list[dict] = []
    needs_features: list[dict] = []
    fully_cached = 0

    for row in rows:
        has_crop, has_features = _already_cached(row["faiss_id"])
        if has_crop and (has_features or not lightglue_available()):
            fully_cached += 1
            continue
        if not has_crop:
            needs_crop.append(row)
        if not has_features and lightglue_available():
            needs_features.append(row)

    print("=" * 78)
    print("REPORT")
    print("=" * 78)
    print(f"  already fully cached:        {fully_cached}")
    print(f"  missing crop:                {len(needs_crop)}")
    print(f"  missing features:            {len(needs_features)}")
    to_process = {r["faiss_id"]: r for r in (needs_crop + needs_features)}
    print(f"  distinct faiss_ids to touch: {len(to_process)}")
    if to_process:
        print("\n  faiss_ids: " + ", ".join(str(fid) for fid in sorted(to_process)))

    if not args.execute:
        print("\nDry run only -- nothing downloaded or written. Re-run with --execute to apply.")
        return

    if not to_process:
        print("\nNothing to do.")
        return

    print(f"\nExecuting: backfilling {len(to_process)} faiss_ids ...")
    crops_ok = crops_fail = features_ok = features_fail = 0

    for fid, row in sorted(to_process.items()):
        print(f"  faiss_id={fid} (godhaar_id={row['godhaar_id']}, sequence={row['sequence']})")
        img = _download_image(row["image_url"])
        if img is None:
            print(f"    SKIP: could not download/decode image")
            crops_fail += 1
            features_fail += 1
            continue

        crop, det_status, _det_conf = crop_cattle(img)
        if crop is None:
            print(f"    SKIP: crop_cattle found no animal (status={det_status})")
            crops_fail += 1
            features_fail += 1
            continue

        if fid in {r["faiss_id"] for r in needs_crop}:
            if save_crop(fid, crop):
                crops_ok += 1
            else:
                crops_fail += 1
                print(f"    crop write failed (see muzzle_crop_cache warning above)")

        if fid in {r["faiss_id"] for r in needs_features}:
            try:
                if save_features(fid, **extract_features_np(crop)):
                    features_ok += 1
                else:
                    features_fail += 1
                    print(f"    feature write failed (see muzzle_crop_cache warning above)")
            except Exception as e:
                features_fail += 1
                print(f"    feature extraction failed: {e}")

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    print(f"  crops:    {crops_ok} written, {crops_fail} failed/skipped")
    print(f"  features: {features_ok} written, {features_fail} failed/skipped")
    print("\nRe-run this script (still safe) to retry anything that failed --")
    print("already-written faiss_ids are skipped automatically.")


if __name__ == "__main__":
    main()
