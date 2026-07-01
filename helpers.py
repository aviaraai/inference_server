import time

import cv2
import numpy as np
from fastapi import HTTPException


def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes to a BGR numpy array."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="cannot_decode_image")
    return img


def _ms_since(start: float) -> int:
    """Milliseconds elapsed since `start` (from time.monotonic())."""
    return int((time.monotonic() - start) * 1000)
