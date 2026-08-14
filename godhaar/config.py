"""
godhaar/config.py — ML pipeline constants for the inference server.

This file contains ONLY model/pipeline constants.
Business logic thresholds (MATCH_THRESHOLD, REVIEW_THRESHOLD, GPS bonuses,
COLOR_MISMATCH_PENALTY, etc.) belong in the API server, NOT here.
"""

import os

# ── Model & Preprocessing ────────────────────────────────────────────────────
IMG_SIZE = 518                                  # DINOv2 ViT-B/14 native resolution
IMG_MEAN = (0.485, 0.456, 0.406)                # ImageNet mean
IMG_STD  = (0.229, 0.224, 0.225)                # ImageNet std
EMB_DIM  = 256                                  # GodhaarModel output contract

# ── YOLO Crop ─────────────────────────────────────────────────────────────────
YOLO_MODEL_NAME    = "yolov8s.pt"
YOLO_COW_CLASS_ID  = 19                         # COCO "cow" (primary)
# Extended set: close-up buffalo shots are often misclassified as bear/sheep/
# horse by YOLOv8 because the body shape context is missing. Accept any of
# these large-animal classes so the crop pipeline still fires.
YOLO_CATTLE_CLASS_IDS = {17, 18, 19, 20, 21}   # horse, sheep, cow, elephant, bear
YOLO_CONF          = 0.30                       # pipeline filter: accept detections ≥ this
YOLO_INTERNAL_CONF = 0.10                       # passed to YOLO inference — must be BELOW
                                                # YOLO_CONF so we see (and log) all candidates
                                                # before filtering. YOLO default (0.25) silently
                                                # drops dark-cattle detections we need to see.
MIN_BBOX_AREA_PCT  = 0.05                       # bbox must be ≥5% of image area
# ── Muzzle detector (color sampling only, NOT the embedding path) ────────────
# Single-class YOLOv8n (`cattle_muzzle_3`) exported to TFLite — the same
# weights the Telangana Android app ships in its assets. Used to localize the
# muzzle before reading its color; without it, muzzle color is sampled from
# the center of the whole-animal crop, i.e. the coat. See
# pipeline/muzzle_detect.py. `appstorage/` is gitignored, so this file must be
# placed on the deployment volume alongside the other models.
MUZZLE_MODEL_PATH  = os.getenv(
    "MUZZLE_MODEL_PATH", "appstorage/Models/muzzle_detect/best_float16.tflite"
)
MUZZLE_DETECT_CONF = 0.30                       # measured 0.82–0.91 on real field photos
MAX_CATTLE_PER_IMAGE = 1                        # detections above this must resolve to a
                                                # single dominant subject (see below) or the
                                                # image is rejected as multi-cattle
CROP_PADDING_PX    = 10                         # pixels to pad around detection box
# ── Dominant-subject selection (goshala multi-cattle photos) ─────────────────
# In a goshala the cattle stand shoulder to shoulder, so a correctly-aimed
# photo of ONE animal routinely has neighbours in frame. Rejecting every such
# photo as RECAPTURE_MULTI_CATTLE is unusable there — instead, when several
# boxes survive, the SUBJECT is the one the photographer clearly aimed at:
# large in frame AND near its center. Each box scores
#     area_fraction * (1 - center_distance) ** SUBJECT_CENTER_WEIGHT_POWER
# (center_distance normalized 0=frame center, 1=corner) and the top box wins
# only if it beats the runner-up by SUBJECT_DOMINANCE_RATIO. Below that the two
# animals are comparably prominent, nobody can say which one was meant, and the
# image is still rejected — which is the whole point of the single-animal gate:
# an embedding must not be ambiguous about WHICH animal it represents.
#
# Both values are calibrated against the reported goshala photos (5 real
# images, 2 cattle detected in 4 of them): measured winner:runner-up score
# ratios were 5.9, 7.2 and 42 — pure AREA ratios for the same photos were only
# 1.62 and 2.28, which is why area alone (the old unused DOMINANT_AREA_RATIO,
# 3.0) could not do this job. 2.0 sits well below the observed population with
# margin, and squaring the center term is what opens the gap: the neighbour
# animal sits far off-center in every one of these photos.
SUBJECT_DOMINANCE_RATIO   = 2.0                 # top score must be >= 2x the runner-up
SUBJECT_CENTER_WEIGHT_POWER = 2.0               # how sharply off-center boxes are penalized

# ── Quality Gate ──────────────────────────────────────────────────────────────
MIN_SHORT_SIDE     = 96                         # minimum short edge in pixels
BLUR_THRESHOLD     = 20.0                       # Laplacian variance minimum
MIN_EXPOSURE       = 30.0                       # mean pixel intensity floor
MAX_EXPOSURE       = 225.0                      # mean pixel intensity ceiling
# Dark-coated cattle/buffalo (common in Indian breeds) push whole-frame mean
# intensity below MIN_EXPOSURE even in a well-lit, detailed photo — the mean
# alone can't tell a naturally dark subject from a genuinely underexposed one.
# A real underexposed shot is dark AND flat (crushed toward black, low local
# contrast); a well-lit dark animal is dark but still has real contrast. Only
# reject on low exposure when contrast is ALSO below this floor.
MIN_EXPOSURE_STD   = 15.0                       # pixel-intensity std-dev floor,
                                                 # paired with MIN_EXPOSURE

# ── Body-color agreement (register's 2 front photos) ─────────────────────────
# When the two front photos read DIFFERENT body colors, /register accepts the
# reading that claims at least this share of the sampled coat, instead of
# rejecting outright (see main.py::_resolve_disagreeing_body_colors).
#
# This is NOT a tuned number: body_color.py defines confidence as the winning
# label's share of the sampled coat, so 0.50 is exactly "this color covers
# most of the animal" — the same majority idea muzzle color already uses
# across its 3 crops. It only ever applies to photos that previously produced
# a hard 422, so it cannot change the outcome for any animal whose two front
# photos agree.
BODY_COLOR_MAJORITY_CONFIDENCE = 0.50

# ── Duplicate Detection ──────────────────────────────────────────────────────
DUPLICATE_THRESHOLD = 0.80                      # cosine similarity above which
                                                # embeddings are considered the
                                                # same muzzle (duplicate).
                                                # Lowered from 0.95 → 0.80 to catch
                                                # re-registrations under different
                                                # lighting/angle while still giving
                                                # enough margin to avoid false positives
                                                # on genuinely different cattle.

# Above this, embedding similarity alone is treated as sufficient proof of a
# duplicate -- color agreement is no longer required (see main.py's duplicate
# check). Reported live: a real animal was double-registered because its
# second registration's front photo was shot in a crowded goshala stall,
# detect_primary_animal() (pipeline/yolo_crop.py) picked a neighbouring
# animal's coat instead of the subject's (it has no quality/multi-cattle
# gate, unlike the muzzle path), so body_color mismatched and the AND-gated
# duplicate check silently let a real duplicate through. Not independently
# calibrated -- chosen as a conservative midpoint between DUPLICATE_THRESHOLD
# (0.80) and the one measured genuine same-animal score on record (0.9712,
# see CLAUDE.md's crop-bug investigation). Revisit if real registrations
# show this is too tight/loose once more same-animal score data exists.
DUPLICATE_HIGH_CONFIDENCE_THRESHOLD = 0.90

# ── Muzzle crop cache (search fusion tiebreaker) ─────────────────────────────
# Local, on-disk cache of each registered muzzle's accepted crop, written at
# /register time and read at /search time by the LightGlue tiebreaker (see
# pipeline/lightglue_verify.py and main.py's /search handler). Keyed by
# faiss_id, which is 1:1 with a specific muzzle crop -- exactly the crop that
# produced whichever embedding scored highest, so no "which of the 3 photos"
# ambiguity. Lives under the same host-mounted volume as FAISS_INDEX_PATH
# (docker-compose.yaml's /appstorage mount), so it survives restarts/redeploys
# the same way the index does, with no new infrastructure.
#
# Only covers animals registered AFTER this shipped -- there is no backfill
# for the existing FAISS index. A search whose top-1 candidate has no cached
# crop just skips the tiebreaker (lightglue_checked=False); this is expected
# and not an error. See CLAUDE.md.
MUZZLE_CROP_CACHE_DIR = os.getenv("MUZZLE_CROP_CACHE_DIR", "/appstorage/muzzle_crops")

IMAGE_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".webp"}

# ── Versioning ────────────────────────────────────────────────────────────────
MODEL_VERSION      = "dinov2_arcface_v1"
