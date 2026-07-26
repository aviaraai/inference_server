"""
Wildlife/color — Cattle color extraction and classification module.
"""

from .body_color import classify_body_color
from .muzzle_color import classify_muzzle_color

__all__ = [
    "classify_body_color",
    "classify_muzzle_color",
]
