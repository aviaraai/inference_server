"""
pipeline/lightglue_verify.py — DISK + LightGlue keypoint-matching tiebreaker
for /search, used only when the DINOv2/FAISS score is ambiguous.

Reuses exactly the approach validated in experiments/lightglue_poc/
match_muzzles.py against 13 real Dehradun search false-positive incidents
(see CLAUDE.md) — same DISK feature extractor, same LightGlue matcher. That
script stays where it is as the standalone CLI/experiment entry point; this
module is the same logic made importable and loadable-once-at-startup so
/search can actually call it, the same way YOLO and the muzzle detector are
loaded once in main.py's lifespan rather than per request.

This is a SIGNAL, not a decision-maker. It never determines MATCH/REVIEW/
UNKNOWN -- that stays go-apiserver's job. See main.py's /search handler for
where its output is attached to the response as inert, additive fields.

Latency history (see CLAUDE.md, "/search fusion tiebreaker" for the full
story): the first cut (single fixed-size extraction per call, no cache)
measured 3-5s on real large crops -- well over budget. Round 2 added the
resize cap (LIGHTGLUE_MAX_DIM) and the candidate-side feature cache below,
re-validated against the same real 40-image dataset: worst case dropped to
172ms, and zone separation on the clean population went from an exact tie
(96 == 96) to a clean gap (139 vs 150, zero overlap) -- both gates passed,
so LIGHTGLUE_TIEBREAKER_ENABLED defaults on. The pre-resize thresholds/flag
default are kept in git history, not restated here.
"""

import logging
import os

import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch

log = logging.getLogger("godhaar.lightglue_verify")

_disk = None
_lightglue = None
_device: torch.device = torch.device("cpu")

NUM_FEATURES = 1024

# Longest-side cap applied before DISK feature extraction (both query and
# candidate crops go through this). DISK's cost scales with pixel count, and
# crop_cattle crops range from tight close-ups (~130K px) to near-full-frame
# (~1.7M px) — the latter measured at 3-5s, well over budget, with no resize.
# 512 was the starting hypothesis (matches the offline verifier draft's own
# choice) and held up: re-validated against the real 40-image dataset (see
# CLAUDE.md) with worst-case latency at 172ms and zone separation actually
# improving over the pre-resize baseline. Not re-tuned beyond that first
# guess since it already cleared both bars — no reason to search further.
LIGHTGLUE_MAX_DIM = int(os.getenv("LIGHTGLUE_MAX_DIM", "512"))

# Top-K re-ranking (promotes LightGlue from a demote-only veto on the single
# top-1 candidate to a signal across all embedding candidates — see
# pipeline/rerank.py and scratch_rerank_eval.py's Stage 1 gate for the
# measured numbers this shipped on). Bounds how many of /search's top_matches
# get a LightGlue comparison per request — cost scales linearly with this,
# so it is not simply "as many as possible".
#
# scratch_rerank_eval.py, 619 leave-one-out queries / 216 Uttarakhand
# animals, baseline top-1 (fused+whitened embedding alone) = 72.2%:
#
#   K    re-ranked top-1   oracle (top-K recall)   gain     median latency
#   3    73.8%             86.1%                   +1.6pp   167ms
#   5    75.0%             88.7%                    +2.7pp   276ms
#   10   76.9%             93.5%                   +4.7pp   552ms
#   20   78.7%             95.5%                   +6.5pp  1099ms
#
# Real, positive, monotonic gain at every K -- this is not a wash. But the
# RRF combination (pipeline/rerank.py) captures only a fraction of the
# available headroom: at K=10 it recovers 4.7 of a possible 21.3 points
# (72.2% -> 93.5%). Picked K=10: ~17% relative top-1 error reduction
# (27.8%->23.1%) for a bounded, real latency cost; K=20 nearly doubles that
# cost (1.1s median) for only +1.8pp more -- clearly diminishing returns
# past 10. Closing more of the oracle gap needs a smarter combiner than
# plain RRF (e.g. weighting LightGlue's contribution by its own within-K
# confidence spread) -- flagged as a follow-up, not attempted here.
LIGHTGLUE_RERANK_TOP_K = int(os.getenv("LIGHTGLUE_RERANK_TOP_K", "10"))

# Master switch — defaults ON. Both re-validation gates passed (see CLAUDE.md,
# "/search fusion tiebreaker, round 2"): worst-case latency 172ms (budget was
# ~1s) and zone separation on the clean population went from an exact tie
# (96 == 96, pre-resize) to a clean gap (max FP 139, min TP 150, zero
# overlap). Kept as an env-var kill switch, not deleted outright, in case a
# future regression needs a fast way to disable this without a redeploy.
LIGHTGLUE_TIEBREAKER_ENABLED = os.getenv("LIGHTGLUE_TIEBREAKER_ENABLED", "true").lower() == "true"

# ── Zone thresholds ───────────────────────────────────────────────────────────
# Round 2: re-derived from lightglue_fp_results_v2_resize_cache.csv (same
# 40-image dataset as round 1, same 57 pairs, this time run through the
# ACTUAL production code path — pipeline/lightglue_verify.py +
# pipeline/muzzle_crop_cache.py, resize cap AND candidate feature cache both
# active, cache hits confirmed for every pair, not benchmarking a fallback
# path). Restricted to the 51 pairs whose both images cleared production's
# own quality/detection gates — same population-selection rule as round 1,
# for a fair before/after comparison (see CLAUDE.md).
#
#   28 clean false-positive pairs: num_matches 28..139 (max 139, 2nd-highest 119)
#   23 clean true-positive pairs:  num_matches 150..833 (min 150, 2nd-lowest 249)
#
# Zero overlap this time — a real gap, not a tie: every clean FP scored below
# every clean TP. (Round 1, pre-resize: max FP == min TP == 96, an exact tie.
# Resizing to 512px did not trade accuracy for speed; if anything it separated
# the two populations more cleanly, plausibly because the coarser resolution
# discards some of the high-frequency texture noise that was producing
# spurious keypoint matches on the false-positive pairs.) Thresholds keep a
# margin either side of the empirical 139/150 boundary rather than cutting
# exactly at it — this is 51 pairs from 9-10 real animals, not a
# large-sample calibration, and a razor's-edge cutoff would overfit to it:
#
#   < 140        likely_different   (28/28 clean FPs)
#   140 .. 150   ambiguous          (nothing empirically here — a deliberate
#                                     safety margin around the boundary)
#   > 150        likely_same        (23/23 clean TPs; the boundary value
#                                     itself, 150, reads as ambiguous rather
#                                     than a confident claim)
LIKELY_DIFFERENT_MAX = 140
LIKELY_SAME_MIN = 150


def load_lightglue() -> None:
    """Load DISK + LightGlue once. Called during startup, alongside YOLO and
    the muzzle detector.

    Fail-open: both models are downloaded from the kornia/torch hub cache on
    first use (see CLAUDE.md for the exact cached filenames and the
    deployment note about pre-seeding them). A missing cache with no network
    reachable at container start must not crash the server -- the /search
    tiebreaker just never runs, exactly like a missing muzzle detector today.
    """
    global _disk, _lightglue, _device
    try:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _disk = KF.DISK.from_pretrained("depth").to(_device).eval()
        _lightglue = KF.LightGlue("disk").to(_device).eval()
        log.info(f"LightGlue verifier loaded on {_device}")
    except Exception as e:
        log.warning(f"LightGlue verifier failed to load ({e}). /search tiebreaker disabled.")
        _disk = None
        _lightglue = None


def warmup_lightglue() -> None:
    """Run a dummy pair through the pipeline so the first real ambiguous
    search isn't the one paying for lazy CUDA/kernel init."""
    if _disk is None or _lightglue is None:
        return
    try:
        # Random noise, NOT zeros/a flat color — DISK finds zero keypoints on
        # a textureless image, which then throws inside LightGlue's internal
        # top-k reduction (`kthvalue(): ... non-zero size`). Noise guarantees
        # local gradients everywhere, same as any real photo would have.
        rng = np.random.default_rng(0)
        dummy = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
        verify(dummy, dummy)
        log.info("LightGlue verifier warmup complete.")
    except Exception as e:
        log.warning(f"LightGlue verifier warmup failed: {e}")


def available() -> bool:
    """True if the verifier is loaded and usable."""
    return _disk is not None and _lightglue is not None


def _resize_for_lightglue(img_bgr: np.ndarray, max_dim: int = None) -> np.ndarray:
    """Downscale so the longest side is at most max_dim, preserving aspect
    ratio. Never upscales — a crop already smaller than the cap is left
    alone, since upscaling adds no real detail and small crops are already
    fast. See LIGHTGLUE_MAX_DIM above for why this exists."""
    if max_dim is None:
        max_dim = LIGHTGLUE_MAX_DIM
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img_bgr
    scale = max_dim / longest
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = K.image.image_to_tensor(img_rgb, False).float() / 255.0
    return t.to(_device)


def extract_features(img_bgr: np.ndarray, num_features: int = NUM_FEATURES) -> dict:
    """Resize (see _resize_for_lightglue) + run DISK once. Returns tensors
    ready to feed LightGlue directly, on this module's device.

    This is the ONE place resizing and DISK extraction happen — used both for
    a fresh (query-side) extraction and, at registration time, for what gets
    cached. Keeping it single-sourced is what guarantees a cached candidate's
    features are extracted exactly the same way a live one would be.
    """
    if _disk is None:
        raise RuntimeError("LightGlue verifier not loaded — check available() first")

    resized = _resize_for_lightglue(img_bgr)
    t = _to_tensor(resized)
    with torch.inference_mode():
        feats = _disk(t, n=num_features, pad_if_not_divisible=True)[0]
    image_size = torch.tensor(t.shape[-2:][::-1], device=_device)  # (2,) — (W, H)
    return {
        "keypoints": feats.keypoints,
        "descriptors": feats.descriptors,
        "image_size": image_size,
    }


def features_to_numpy(feats: dict) -> dict:
    """extract_features()'s device tensors -> plain numpy, for caching
    (pipeline/muzzle_crop_cache.py's save_features)."""
    return {
        "keypoints": feats["keypoints"].detach().cpu().numpy(),
        "descriptors": feats["descriptors"].detach().cpu().numpy(),
        "image_size": feats["image_size"].detach().cpu().numpy(),
    }


def _features_from_numpy(feats_np: dict) -> dict:
    """The inverse of features_to_numpy — cached numpy arrays back onto this
    module's device, ready for LightGlue. Input is exactly what
    muzzle_crop_cache.load_features() returns."""
    return {
        "keypoints": torch.from_numpy(feats_np["keypoints"]).to(_device),
        "descriptors": torch.from_numpy(feats_np["descriptors"]).to(_device),
        "image_size": torch.from_numpy(feats_np["image_size"]).to(_device),
    }


def _match_feature_sets(feats_a: dict, feats_b: dict) -> dict:
    """LightGlue over two already-extracted feature sets (each either fresh
    from extract_features() or reloaded via _features_from_numpy())."""
    if _lightglue is None:
        raise RuntimeError("LightGlue verifier not loaded — check available() first")

    lg_input = {
        "image0": {
            "keypoints": feats_a["keypoints"][None],
            "descriptors": feats_a["descriptors"][None],
            "image_size": feats_a["image_size"][None],
        },
        "image1": {
            "keypoints": feats_b["keypoints"][None],
            "descriptors": feats_b["descriptors"][None],
            "image_size": feats_b["image_size"][None],
        },
    }
    with torch.inference_mode():
        out = _lightglue(lg_input)

    matches = out["matches"][0]
    scores = out["scores"][0]
    n_kp_a = feats_a["keypoints"].shape[0]
    n_kp_b = feats_b["keypoints"].shape[0]
    n_matches = int(matches.shape[0])
    mean_score = float(scores.mean().item()) if n_matches > 0 else 0.0

    return {
        "num_matches": n_matches,
        "match_ratio": n_matches / max(min(n_kp_a, n_kp_b), 1),
        "mean_match_score": round(mean_score, 4),
    }


def verify(img_a: np.ndarray, img_b: np.ndarray) -> dict:
    """Extract both sides fresh, then match. For callers with no cached
    features on either side — the offline experiment script, warmup, and the
    fallback path when a candidate's features were never cached."""
    return _match_feature_sets(extract_features(img_a), extract_features(img_b))


def match_features(query_feats: dict, candidate_feats: dict) -> dict:
    """Public wrapper over _match_feature_sets, for callers that already
    hold on-device feature dicts from extract_features() on BOTH sides
    (e.g. the top-K reranker matching one query extraction against a
    freshly-extracted, uncached candidate crop) and don't need this
    module's other cache-handling conveniences.
    """
    return _match_feature_sets(query_feats, candidate_feats)


def match_precomputed(query_feats: dict, candidate_features_np: dict) -> dict:
    """Match an already-extracted (on-device) query feature set against a
    CACHED (numpy, off-device) candidate feature set. Used by the top-K
    reranker (main.py) to extract the query ONCE per search and reuse it
    across every candidate, instead of verify_with_cached_candidate's
    per-call re-extraction — the same query image never changes across a
    single search's K candidates.
    """
    candidate_feats = _features_from_numpy(candidate_features_np)
    return _match_feature_sets(query_feats, candidate_feats)


def verify_with_cached_candidate(query_img_bgr: np.ndarray, candidate_features_np: dict) -> dict:
    """Query extracted fresh (always, since it's per-request); candidate
    features reused from the cache (pipeline/muzzle_crop_cache.load_features)
    instead of re-extracted. This is the fast path — no DISK forward pass on
    the candidate side at all."""
    query_feats = extract_features(query_img_bgr)
    candidate_feats = _features_from_numpy(candidate_features_np)
    return _match_feature_sets(query_feats, candidate_feats)


def extract_features_np(img_bgr: np.ndarray, num_features: int = NUM_FEATURES) -> dict:
    """Convenience: extract + convert to numpy in one call, for main.py's
    /register handler, which only needs the cacheable form."""
    return features_to_numpy(extract_features(img_bgr, num_features=num_features))


def classify_zone(num_matches: int) -> str:
    """Map a raw match count onto the three calibrated zones (round 2, post
    resize+cache). See the "Zone thresholds" block above for the derivation."""
    if num_matches < LIKELY_DIFFERENT_MAX:
        return "likely_different"
    if num_matches > LIKELY_SAME_MIN:
        return "likely_same"
    return "ambiguous"
