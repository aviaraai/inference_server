"""
pipeline/preprocess.py — Image preprocessing for GodhaarModel inference.

Matches the exact transform used in src/embed.py and src/identify.py:
  518×518 resize + ImageNet mean/std normalisation.
"""

import io
from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from godhaar.config import IMG_SIZE, IMG_MEAN, IMG_STD


# Build the transform once at module level for reuse.
_val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
])


def preprocess(image_input: Union[bytes, np.ndarray, Image.Image]) -> torch.Tensor:
    """Convert a single image to a model-ready (1, 3, 518, 518) tensor.

    Parameters
    ----------
    image_input : bytes | np.ndarray (BGR) | PIL.Image
        A single image in any common format.

    Returns
    -------
    torch.Tensor of shape (1, 3, 518, 518), float32.
    """
    if isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        # Assume BGR (OpenCV convention) → RGB
        import cv2
        rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    tensor = _val_transform(pil_img)
    return tensor.unsqueeze(0)  # (1, 3, 518, 518)


def preprocess_batch(images: list[Union[bytes, np.ndarray, Image.Image]]) -> torch.Tensor:
    """Preprocess a list of images into a single batched tensor.

    Parameters
    ----------
    images : list of image inputs (bytes, ndarray, or PIL.Image)

    Returns
    -------
    torch.Tensor of shape (B, 3, 518, 518), float32.
    """
    tensors = []
    for img in images:
        t = preprocess(img)         # (1, 3, 518, 518)
        tensors.append(t.squeeze(0))  # (3, 518, 518)
    return torch.stack(tensors)  # (B, 3, 518, 518)
