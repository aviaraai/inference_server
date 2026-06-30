"""
pipeline/muzzle.py — Muzzle embedding pipeline.

Handles single-image and batch inference through GodhaarModel.
"""

from typing import Any, Union

import torch
import numpy as np
from PIL import Image

from pipeline.preprocess import preprocess, preprocess_batch


def embed_single(image_bytes: bytes, model: Any, device: torch.device) -> torch.Tensor:
    """Embed a single muzzle image.

    Parameters
    ----------
    image_bytes : bytes
        Raw image file content.
    model : GodhaarModel
        Loaded model in eval mode.
    device : torch.device

    Returns
    -------
    torch.Tensor of shape (1, 256), unit-norm, float32.
    """
    tensor = preprocess(image_bytes).to(device)  # (1, 3, 518, 518)

    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            embedding = model(tensor)  # (1, 256)

    return embedding.float().cpu()


def embed_batch(
    images: list[bytes],
    model: Any,
    device: torch.device,
) -> torch.Tensor:
    """Embed a batch of muzzle images in a single forward pass.

    Parameters
    ----------
    images : list[bytes]
        List of raw image file contents.
    model : GodhaarModel
    device : torch.device

    Returns
    -------
    torch.Tensor of shape (B, 256), unit-norm, float32.
    """
    if len(images) == 0:
        return torch.empty(0, 256)

    batch_tensor = preprocess_batch(images).to(device)  # (B, 3, 518, 518)

    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            embeddings = model(batch_tensor)  # (B, 256)

    return embeddings.float().cpu()
