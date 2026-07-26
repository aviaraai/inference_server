"""
color_constants.py — Configurable color spaces, enums, and thresholds.
"""

# ---------------------------------------------------------------------------
# Color Labels
# ---------------------------------------------------------------------------
LABEL_BLACK = "BLACK"
LABEL_WHITE = "WHITE"
LABEL_BROWN = "BROWN"
LABEL_GREY  = "GREY"
LABEL_SPOTTED = "SPOTTED"
LABEL_MIXED = "MIXED"
LABEL_PINK  = "PINK"
LABEL_UNKNOWN = "UNKNOWN"

BODY_COLORS = [
    LABEL_BLACK,
    LABEL_WHITE,
    LABEL_BROWN,
    LABEL_GREY,
    LABEL_SPOTTED,
    LABEL_UNKNOWN,
]

MUZZLE_COLORS = [
    LABEL_BLACK,
    LABEL_PINK,
    LABEL_MIXED,
    LABEL_UNKNOWN,
]

# ---------------------------------------------------------------------------
# LAB Color Space Thresholds (Initial defaults, TODO: calibrate using calibrate.py)
# ---------------------------------------------------------------------------
NEUTRAL_THRESHOLD = 15.0  # TODO: calibrate (chroma sqrt(a^2 + b^2) below this is neutral)
L_BLACK_MAX       = 40.0  # TODO: calibrate (max L* value for Black)
L_WHITE_MIN       = 75.0  # TODO: calibrate (min L* value for White)

# Chromatic bounds in the a*-b* plane
BROWN_ANGLE_MIN   = 10.0  # TODO: calibrate (degrees in a*-b* plane)
BROWN_ANGLE_MAX   = 80.0  # TODO: calibrate

# Spotted ratio (body) / Mixed ratio (muzzle)
# If secondary dominant color represents more than this ratio of the main color pixels,
# classify as Spotted/Mixed.
SPOTTED_RATIO_MIN = 0.25  # TODO: calibrate
MIXED_RATIO_MIN   = 0.25  # TODO: calibrate

# ---------------------------------------------------------------------------
# Quality Gate Boundaries
# ---------------------------------------------------------------------------
MIN_ROI_WIDTH      = 120   # minimum width of cropped region
MIN_ROI_HEIGHT     = 120   # minimum height of cropped region
BRIGHTNESS_MIN     = 30.0  # minimum average pixel intensity
BRIGHTNESS_MAX     = 225.0 # maximum average pixel intensity
MIN_CONTRAST       = 10.0  # minimum standard deviation of pixel values
BLUR_THRESHOLD     = 15.0  # minimum Laplacian variance
