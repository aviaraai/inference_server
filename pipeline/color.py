"""
pipeline/color.py — Color extraction with pluggable interface.

Architecture:
    ColorExtractor (ABC)
        ├── RuleBasedColorExtractor   ← uses Wildlife/color/ LAB logic (current)
        └── AIColorExtractor          ← future: trained classifier

Enums match exactly what the current Wildlife classifiers output.
"""

import logging
import os
import sys
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

log = logging.getLogger("godhaar.color")


# ── Color Enums ───────────────────────────────────────────────────────────────
# These match the labels in Wildlife/color/color_constants.py.


class BodyColor(str, Enum):
    UNKNOWN = "UNKNOWN"
    BLACK = "BLACK"
    WHITE = "WHITE"
    BROWN = "BROWN"
    GREY = "GREY"
    SPOTTED = "SPOTTED"
    MIXED = "MIXED"


class MuzzleColor(str, Enum):
    UNKNOWN = "UNKNOWN"
    BLACK = "BLACK"
    PINK = "PINK"
    MIXED = "MIXED"


# ── Interface ─────────────────────────────────────────────────────────────────


class ColorExtractor(ABC):
    """Abstract interface for extracting cattle colors from images.

    Implementations must return one of the Enum values above.
    This allows swapping the rule-based system with an AI classifier
    without touching any other code.
    """

    @abstractmethod
    def extract_body(self, img_bgr: np.ndarray) -> dict:
        """Extract body coat color from a BGR image.

        Returns
        -------
        {"label": str, "confidence": float}
        """
        ...

    @abstractmethod
    def extract_muzzle(self, img_bgr: np.ndarray) -> dict:
        """Extract muzzle skin color from a BGR image.

        Returns
        -------
        {"label": str, "confidence": float}
        """
        ...


# Rule-Based Implementation
class RuleBasedColorExtractor(ColorExtractor):
    """Uses the existing Wildlife/color/ LAB-histogram classifiers.

    Gracefully degrades to UNKNOWN if the Wildlife module is not available
    (e.g. not volume-mounted in Docker).
    """

    def __init__(self) -> None:
        self._body_fn = None
        self._muzzle_fn = None
        self._available = False

        try:
            # Try importing from the Wildlife color package.
            # In Docker, this is volume-mounted to /wildlife/color.

            # Check common mount locations
            for search_path in [
                "/wildlife",  # Docker mount
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "Wildlife"
                ),  # Local dev
            ]:
                abs_path = os.path.abspath(search_path)
                if os.path.isdir(abs_path) and abs_path not in sys.path:
                    sys.path.insert(0, abs_path)

            from color.body_color import classify_body_color
            from color.muzzle_color import classify_muzzle_color

            self._body_fn = classify_body_color
            self._muzzle_fn = classify_muzzle_color
            self._available = True
            log.info("RuleBasedColorExtractor: Wildlife color module loaded.")
        except Exception as e:
            log.warning(
                f"RuleBasedColorExtractor: Wildlife color module not available ({e}). "
                "All colors will be UNKNOWN."
            )

    @property
    def available(self) -> bool:
        return self._available

    def extract_body(self, img_bgr: np.ndarray) -> dict:
        if not self._available or img_bgr is None:
            return {
                "label": BodyColor.UNKNOWN.value,
                "confidence": 0.0,
            }

        try:
            result = self._body_fn(img_bgr)
            # Validate the label against our enum
            label = result.get("label", "UNKNOWN")
            try:
                label = BodyColor(label).value
            except ValueError:
                label = BodyColor.UNKNOWN.value
            return {
                "label": label,
                "confidence": float(result.get("confidence", 0.0)),
            }
        except Exception as e:
            log.warning(f"Body color extraction failed: {e}")
            return {
                "label": BodyColor.UNKNOWN.value,
                "confidence": 0.0,
            }

    def extract_muzzle(self, img_bgr: np.ndarray) -> dict:
        if not self._available or img_bgr is None:
            return {
                "label": MuzzleColor.UNKNOWN.value,
                "confidence": 0.0,
            }

        try:
            result = self._muzzle_fn(img_bgr)
            label = result.get("label", "UNKNOWN")
            try:
                label = MuzzleColor(label).value
            except ValueError:
                label = MuzzleColor.UNKNOWN.value
            return {
                "label": label,
                "confidence": float(result.get("confidence", 0.0)),
            }
        except Exception as e:
            log.warning(f"Muzzle color extraction failed: {e}")
            return {
                "label": MuzzleColor.UNKNOWN.value,
                "confidence": 0.0,
            }
