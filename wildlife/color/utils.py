"""
utils.py — Helper functions for color space conversion, statistics, and histogram peak detection.
"""

import cv2
import numpy as np


def bgr_to_lab(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to CIE L*a*b*."""
    if img_bgr is None or img_bgr.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)


def parse_raw_lab(pixel_lab: np.ndarray | list | tuple) -> list[float]:
    """Convert OpenCV representation of LAB [0-255] back to standard CIE LAB scale:

        L* in [0, 100]
        a* in [-128, 127]
        b* in [-128, 127]
    """
    l_raw, a_raw, b_raw = pixel_lab[0], pixel_lab[1], pixel_lab[2]
    l_std = float(l_raw) * 100.0 / 255.0
    a_std = float(a_raw) - 128.0
    b_std = float(b_raw) - 128.0
    return [round(l_std, 2), round(a_std, 2), round(b_std, 2)]


def calculate_median_lab(img_lab: np.ndarray) -> list[float]:
    """Calculate the median L*, a*, b* values of the image on standard CIE scale."""
    if img_lab is None or img_lab.size == 0:
        return [0.0, 0.0, 0.0]

    # Compute median along spatial dimensions
    median_raw = np.median(img_lab.reshape(-1, 3), axis=0)
    return parse_raw_lab(median_raw)


def extract_dominant_lab_features(img_lab: np.ndarray, bin_size: int = 16) -> dict:
    """Build a 2D histogram on the a* and b* chromatic channels.

    Finds the primary and secondary peak regions, calculates their ratios
    for spotted classification, and extracts dominant coordinates.

    Returns
    -------
    dict containing:
        dominant_lab : list[float]  # [L*, a*, b*] of the dominant peak
        confidence : float         # area percentage of the primary peak
        peak_ratio : float         # area of secondary peak / primary peak
        is_multimodal : bool       # True if there's a strong secondary peak
    """
    if img_lab is None or img_lab.size == 0:
        return {
            "dominant_lab": [0.0, 0.0, 0.0],
            "confidence": 0.0,
            "peak_ratio": 0.0,
            "is_multimodal": False,
        }

    # Flatten spatial layout
    pixels = img_lab.reshape(-1, 3)

    # 1. Compute 2D Histogram on chromaticity channels (a* and b* are index 1 & 2)
    # Range is 0 to 256 for OpenCV uint8 channel
    hist, xedges, yedges = np.histogram2d(
        pixels[:, 1], pixels[:, 2],
        bins=256 // bin_size,
        range=[[0, 256], [0, 256]]
    )

    # Total pixel count
    total_pixels = len(pixels)

    # Find highest peaks
    flat_indices = np.argsort(hist.flatten())[::-1]

    # Primary Peak
    primary_idx = flat_indices[0]
    p_x, p_y = np.unravel_index(primary_idx, hist.shape)
    primary_count = hist[p_x, p_y]

    # Find median L* value for pixels falling in the primary chromaticity bin
    x_min, x_max = xedges[p_x], xedges[p_x + 1]
    y_min, y_max = yedges[p_y], yedges[p_y + 1]

    bin_mask = (
        (pixels[:, 1] >= x_min) & (pixels[:, 1] < x_max) &
        (pixels[:, 2] >= y_min) & (pixels[:, 2] < y_max)
    )
    bin_pixels = pixels[bin_mask]

    if len(bin_pixels) > 0:
        dominant_l_raw = np.median(bin_pixels[:, 0])
        # Bin center for a* and b*
        dominant_a_raw = (x_min + x_max) / 2.0
        dominant_b_raw = (y_min + y_max) / 2.0
        dominant_lab = parse_raw_lab([dominant_l_raw, dominant_a_raw, dominant_b_raw])
    else:
        # Fallback to general median
        dominant_lab = calculate_median_lab(img_lab)

    # Secondary Peak (must be physically separated from primary bin in grid coordinates)
    secondary_count = 0.0
    for idx in flat_indices[1:]:
        s_x, s_y = np.unravel_index(idx, hist.shape)
        # Check if the bin is non-adjacent (Manhattan distance > 1) to be a distinct color peak
        if abs(s_x - p_x) > 1 or abs(s_y - p_y) > 1:
            secondary_count = hist[s_x, s_y]
            break

    peak_ratio = float(secondary_count / primary_count) if primary_count > 0 else 0.0
    confidence = float(primary_count / total_pixels) if total_pixels > 0 else 0.0

    return {
        "dominant_lab": dominant_lab,
        "confidence": round(confidence, 4),
        "peak_ratio": round(peak_ratio, 4),
        "is_multimodal": peak_ratio >= 0.25,
    }
