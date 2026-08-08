import time

import cv2
import numpy as np

from errors import unreadable_image_error


def _decode_image(image_bytes: bytes, slot: str = "image") -> np.ndarray:
    """Decode raw bytes to a BGR numpy array.

    Undecodable bytes are a verdict about that upload, not a protocol fault, so
    this raises the IMAGE_UNREADABLE envelope rather than a bare 422 — which
    means callers should pass the `slot` they are decoding ("muzzle_2",
    "front_1"). The default exists only so the ad-hoc scripts that import this
    keep working; every call inside the request path names its slot.

    Note this fires even while BYPASS_QUALITY_GATES is on. That flag downgrades
    quality *judgements*; bytes that produce no pixels cannot be downgraded
    into an embedding.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise unreadable_image_error(slot)
    return img


def _ms_since(start: float) -> int:
    """Milliseconds elapsed since `start` (from time.monotonic())."""
    return int((time.monotonic() - start) * 1000)
