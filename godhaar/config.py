"""
godhaar/config.py — ML pipeline constants for the inference server.

This file contains ONLY model/pipeline constants.
Business logic thresholds (MATCH_THRESHOLD, REVIEW_THRESHOLD, GPS bonuses,
COLOR_MISMATCH_PENALTY, etc.) belong in the API server, NOT here.
"""

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
MAX_CATTLE_PER_IMAGE = 1                        # reject multi-cattle images
CROP_PADDING_PX    = 10                         # pixels to pad around detection box
CLOSE_UP_AREA_PCT  = 0.55                       # if best box fills >55% of frame
                                                # → treat as close-up, use full img
DOMINANT_AREA_RATIO = 3.0                       # if top box is >=3x larger than next,
                                                # drop smaller boxes (background blur)

# ── Quality Gate ──────────────────────────────────────────────────────────────
MIN_SHORT_SIDE     = 96                         # minimum short edge in pixels
BLUR_THRESHOLD     = 20.0                       # Laplacian variance minimum
MIN_EXPOSURE       = 30.0                       # mean pixel intensity floor
MAX_EXPOSURE       = 225.0                      # mean pixel intensity ceiling

# ── Duplicate Detection ──────────────────────────────────────────────────────
DUPLICATE_THRESHOLD = 0.95                      # cosine similarity above which
                                                # embeddings are considered the
                                                # same muzzle (duplicate)

IMAGE_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".webp"}

# ── Versioning ────────────────────────────────────────────────────────────────
MODEL_VERSION      = "dinov2_arcface_v1"
