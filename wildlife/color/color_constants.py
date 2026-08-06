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
# Chroma sqrt(a*^2 + b*^2) below this is treated as neutral (black/grey/white).
#
# CALIBRATED against real cattle photos (was 15.0, an uncalibrated guess).
# Measured cluster chroma on real field images:
#   genuinely achromatic (white/grey Brahman calf): 0.05, 2.1, 3.2, 6.4, 7.1
#   brown Gir/Sahiwal coat:                         12.4, 13.5, 13.8, 14.4, 16.7
# At 15.0 the threshold sat INSIDE the brown distribution, so lit portions of a
# brown coat fell to GREY and shadowed portions to BLACK — this, not the
# segmentation, was why brown animals kept coming back BLACK/GREY. It also made
# BROWN nearly unreachable in practice. 9.0 sits in the empty gap between the
# two populations. Verified: 7/7 correct on the real photo set, and both front
# photos of every animal agree (so /register's consistency check passes).
NEUTRAL_THRESHOLD = 9.0
L_BLACK_MAX       = 40.0  # TODO: calibrate (max L* value for Black)
L_WHITE_MIN       = 75.0  # TODO: calibrate (min L* value for White)

# Chromatic bounds in the a*-b* plane
BROWN_ANGLE_MIN   = 10.0  # TODO: calibrate (degrees in a*-b* plane)
BROWN_ANGLE_MAX   = 80.0  # TODO: calibrate

# Spotted ratio (body) / Mixed ratio (muzzle)
# Ratio of the RUNNER-UP COLOR's total weight to the WINNING COLOR's total
# weight (weights summed per label across clusters — see
# utils.aggregate_clusters_by_label), above which the coat/muzzle is called
# two-tone rather than solid.
#
# Measured on real Gir front photos (brown coat, white face blaze): the same
# animal gave white/brown ratios of 0.51 and 0.22 on its two register photos,
# purely because the head fills more of one frame than the other. Anything
# between those two values would label one photo SPOTTED and the other BROWN
# and trip /register's "front images disagree" 422 on a perfectly good pair.
# 0.60 sits above both, so that animal reads BROWN consistently, while a
# genuinely two-tone coat (roughly balanced patches, ratio near 1.0) still
# reads SPOTTED.
SPOTTED_RATIO_MIN = 0.60  # TODO: recalibrate against a labeled SPOTTED set
MIXED_RATIO_MIN   = 0.60  # TODO: recalibrate against a labeled PINK/BLACK muzzle set

# A secondary color cluster only counts as a spot/patch if it sits on the
# subject rather than in the surrounding scene. Clusters carry a "centrality"
# score (see utils.extract_dominant_lab_features); a secondary cluster must be
# at least this fraction as central as the dominant one to count. Without this,
# an un-localized crop of a plain white cow on dark ground reads as SPOTTED,
# because the ground is a large, genuinely different-colored cluster.
# Irrelevant when a foreground mask is present — segmentation already removed
# the background, and every surviving cluster scores centrality 1.0.
SPOTTED_CENTRALITY_RATIO_MIN = 0.5  # TODO: calibrate

# ---------------------------------------------------------------------------
# Quality Gate Boundaries
# ---------------------------------------------------------------------------
MIN_ROI_WIDTH      = 120   # minimum width of cropped region
MIN_ROI_HEIGHT     = 120   # minimum height of cropped region

# The muzzle ROI is detector-localized (pipeline/muzzle_detect.py), so it is a
# tight box around the nose pad rather than a fixed slice of the whole frame —
# legitimately smaller. Real field photos produced 117x113 muzzle crops, which
# the 120px gate above rejected as "too small", discarding the best pixels
# available in favour of nothing. ~64x64 is still ~4k pixels, ample for color
# clustering, and the blur/exposure/contrast gates below still apply.
MIN_MUZZLE_ROI_WIDTH  = 64
MIN_MUZZLE_ROI_HEIGHT = 64
BRIGHTNESS_MIN     = 30.0  # minimum average pixel intensity
BRIGHTNESS_MAX     = 225.0 # maximum average pixel intensity
MIN_CONTRAST       = 10.0  # minimum standard deviation of pixel values
BLUR_THRESHOLD     = 15.0  # minimum Laplacian variance
