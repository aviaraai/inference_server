"""
pipeline/muzzle_crop_cache.py — local on-disk cache of registered muzzle crops
AND their pre-extracted DISK features.

Exists for main.py's /register handler (writes, right after FAISS assigns a
faiss_id) and main.py's /search handler (reads, by the top-1 candidate's
faiss_id) for the LightGlue fusion tiebreaker. See MUZZLE_CROP_CACHE_DIR in
godhaar/config.py for why this is a local file cache and not a GCS/DB lookup:
inference_server has no image storage credentials and no DB connection today,
and this keeps it that way.

Two things are cached per faiss_id, not one:
  - the crop itself (save_crop/load_crop) — the fallback, always sufficient
    on its own to run the tiebreaker (just slower: needs live DISK extraction
    on the candidate side too).
  - its DISK features (save_features/load_features) — keypoints/descriptors/
    image_size, extracted once at registration time via
    pipeline.lightglue_verify.extract_features_np. When present, /search's
    tiebreaker skips DISK extraction for the candidate entirely and only
    extracts the query side, which is most of the added latency the resize
    cap + this cache together are meant to fix (see CLAUDE.md, "latency
    optimization, round 2"). Feature caching is a pure speed optimization —
    it must produce the same num_matches a live extraction would, since it's
    the same extract_features() call, just run earlier and persisted.

Fail-open in both directions, for both cached artifacts. A write failure must
never fail a registration the FAISS index already accepted; a read miss
(not yet cached, or a save that failed) must never crash a search -- callers
fall back to the next-best thing (features -> crop -> tiebreaker skipped
entirely), same as for any animal registered before either cache existed.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from godhaar.config import MUZZLE_CROP_CACHE_DIR

log = logging.getLogger("godhaar.muzzle_crop_cache")


def _path_for(faiss_id: int) -> Path:
    return Path(MUZZLE_CROP_CACHE_DIR) / f"{faiss_id}.jpg"


def _features_path_for(faiss_id: int) -> Path:
    return Path(MUZZLE_CROP_CACHE_DIR) / f"{faiss_id}.features.npz"


def save_crop(faiss_id: int, crop_bgr: np.ndarray) -> bool:
    """Persist one registered muzzle crop, keyed by its faiss_id.

    Called once per accepted muzzle image at registration time. Never raises:
    a cache-write failure (disk full, permissions, missing mount) must not
    fail the registration itself, since FAISS has already committed the
    embedding by the time this runs.

    Returns True on a successful write so the caller can log a summary count —
    without that, a write that never happened (which silently disables the
    /search tiebreaker for this animal forever) is indistinguishable from one
    that succeeded.
    """
    try:
        os.makedirs(MUZZLE_CROP_CACHE_DIR, exist_ok=True)
        path = _path_for(faiss_id)
        ok, buf = cv2.imencode(".jpg", crop_bgr)
        if not ok:
            log.warning(f"muzzle_crop_cache: failed to encode crop for faiss_id={faiss_id}")
            return False
        path.write_bytes(buf.tobytes())
        log.info(f"muzzle_crop_cache: saved crop for faiss_id={faiss_id} ({len(buf)} bytes)")
        return True
    except Exception as e:
        log.warning(f"muzzle_crop_cache: failed to save faiss_id={faiss_id}: {e}")
        return False


def load_crop(faiss_id: int) -> Optional[np.ndarray]:
    """Load a previously cached muzzle crop, or None if never cached.

    A None return is the expected, common case for any animal registered
    before this cache existed, or if a write above ever failed -- callers
    must treat it as "tiebreaker not available for this candidate", not as
    an error.
    """
    path = _path_for(faiss_id)
    try:
        if not path.is_file():
            return None
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return img  # cv2.imread already returns None on decode failure
    except Exception as e:
        log.warning(f"muzzle_crop_cache: failed to load faiss_id={faiss_id}: {e}")
        return None


def save_features(faiss_id: int, keypoints: np.ndarray, descriptors: np.ndarray, image_size: np.ndarray) -> bool:
    """Persist one registered muzzle's DISK features, keyed by faiss_id.

    Called once per accepted muzzle image at registration time, right after
    save_crop for the same faiss_id — see main.py. `keypoints`/`descriptors`/
    `image_size` are exactly pipeline.lightglue_verify.extract_features_np()'s
    output; this function doesn't know or care what DISK's output shape is
    beyond "three numpy arrays to persist together." Never raises, same
    fail-open contract as save_crop. Returns True on a successful write.
    """
    try:
        os.makedirs(MUZZLE_CROP_CACHE_DIR, exist_ok=True)
        path = _features_path_for(faiss_id)
        np.savez(path, keypoints=keypoints, descriptors=descriptors, image_size=image_size)
        log.info(f"muzzle_crop_cache: saved features for faiss_id={faiss_id}")
        return True
    except Exception as e:
        log.warning(f"muzzle_crop_cache: failed to save features for faiss_id={faiss_id}: {e}")
        return False


def load_features(faiss_id: int) -> Optional[dict]:
    """Load previously cached DISK features, or None if never cached (or a
    save above failed, or this animal predates feature caching).

    Returns {"keypoints": np.ndarray, "descriptors": np.ndarray,
    "image_size": np.ndarray} — ready for
    pipeline.lightglue_verify.verify_with_cached_candidate(). A None return
    means "no fast path for this candidate" — callers must fall back to
    load_crop() + live extraction, not treat this as an error.
    """
    path = _features_path_for(faiss_id)
    try:
        if not path.is_file():
            return None
        data = np.load(path)
        return {
            "keypoints": data["keypoints"],
            "descriptors": data["descriptors"],
            "image_size": data["image_size"],
        }
    except Exception as e:
        log.warning(f"muzzle_crop_cache: failed to load features for faiss_id={faiss_id}: {e}")
        return None
