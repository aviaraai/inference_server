"""
pipeline/muzzle.py — Muzzle embedding pipeline.

Handles single-image and batch inference through GodhaarModel.
"""

from typing import Any

import torch

from pipeline.preprocess import preprocess_batch


def embed_batch(
    images: list[bytes],
    model: Any,
    device: torch.device,
) -> torch.Tensor:
    """Embed a batch of muzzle images in a single forward pass.

    Signature and output contract are frozen: (B, 256), unit-norm, float32
    -- faiss_index.py and everything downstream must not notice which
    encoder actually ran. Dispatches to one of two encoders:

    - ``model`` is a ``pipeline.fusion_encoder.FusionEncoder`` (production,
      post fusion+whitening upgrade): calls its ``embed_images(images)``,
      which does its own per-sub-model resizing directly from the raw
      image bytes (it does NOT go through ``preprocess_batch``'s single
      518x518 transform below -- DINOv2/ResNet50@384/ResNet50@448 each need
      a different resize).
    - ``model`` is a plain ``GodhaarModel`` (legacy / scratch-script /
      unit-test usage): unchanged original path, 518x518 resize + forward.

    Parameters
    ----------
    images : list[bytes]
        List of raw image file contents.
    model : GodhaarModel | pipeline.fusion_encoder.FusionEncoder
    device : torch.device

    Returns
    -------
    torch.Tensor of shape (B, 256), unit-norm, float32.
    """
    if len(images) == 0:
        return torch.empty(0, 256)

    if hasattr(model, "embed_images"):
        return model.embed_images(images).float().cpu()

    batch_tensor = preprocess_batch(images).to(device)  # (B, 3, 518, 518)

    with torch.inference_mode():
        with torch.amp.autocast(
            device_type=device.type, enabled=(device.type == "cuda")
        ):
            embeddings = model(batch_tensor)  # (B, 256)

    return embeddings.float().cpu()
