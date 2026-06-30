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
YOLO_COW_CLASS_ID  = 19                         # COCO "cow"
YOLO_CONF          = 0.30                       # minimum detection confidence
MIN_BBOX_AREA_PCT  = 0.05                       # bbox must be ≥5% of image area
MAX_CATTLE_PER_IMAGE = 1                        # reject multi-cattle images
CROP_PADDING_PX    = 10                         # pixels to pad around detection box

# ── Quality Gate ──────────────────────────────────────────────────────────────
MIN_SHORT_SIDE     = 96                         # minimum short edge in pixels
BLUR_THRESHOLD     = 20.0                       # Laplacian variance minimum
MIN_EXPOSURE       = 30.0                       # mean pixel intensity floor
MAX_EXPOSURE       = 225.0                      # mean pixel intensity ceiling

IMAGE_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".webp"}

# ── Versioning ────────────────────────────────────────────────────────────────
MODEL_VERSION      = "dinov2_arcface_v1"
