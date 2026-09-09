"""
pipeline/fusion_encoder.py — DINOv2 + ResNet50(x2) fusion encoder with frozen
PCA whitening.

Replaces the plain DINOv2-only encoder as the thing `pipeline/muzzle.py`'s
`embed_batch()` calls. See CLAUDE.md / the fusion-upgrade investigation for
why: on a 222-animal disjoint-identity leave-one-out test, this pipeline
measured top-1 58.2% -> ~81-83%, and — the real point — flipped the sign of
genuine-minus-impostor separation from negative to positive (production's
matchThreshold could do no useful work at all before this).

Pipeline (whole photo, NO YOLO crop)
-------------------------------------
    photo bytes
      -> DINOv2 @518 (GodhaarModel, existing production encoder) -> 256-d, L2
      -> ImageNet ResNet50 (IMAGENET1K_V2, fc=Identity) @384       -> 2048-d, L2
      -> same ResNet50 @448                                        -> 2048-d, L2
      -> concat                                                    -> 4352-d
      -> subtract whitening mean, project (frozen PCA), / sqrt(eigval)
      -> L2-normalize                                              -> 256-d

Every one of the three encoders and the whitening apply reproduces, in the
real code path, what scratch_ablation_fusion.py / scratch_ablation_pca.py
measured inline — see scratch_fusion_validation.py, which is the Stage 1
proof gate. Do not change the resize sizes (384/448) or the encoder set
without re-running that gate: this module's whole purpose is to be a
byte-for-byte match of what was measured.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.models import resnet50

from pipeline.whitening import WhiteningTransform

log = logging.getLogger("godhaar.fusion_encoder")

# Resize sizes fixed by the measured ablation — see scratch_ablation_fusion.py.
_DINO_SIZE = 518
_RESNET_SIZES = (384, 448)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(size: int) -> T.Compose:
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def _decode_rgb(image_bytes: bytes) -> Image.Image:
    import io
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def load_resnet50(weights_path: str, device: torch.device) -> nn.Module:
    """Load ImageNet ResNet50 (IMAGENET1K_V2 weights) from a LOCAL file.

    Deliberately does NOT use `ResNet50_Weights.IMAGENET1K_V2` directly —
    that triggers a network download on first use, and the deploy
    container has no reliable egress (same reasoning as every other model
    file in this repo living under /appstorage). `weights_path` must point
    at the raw state_dict `.pth` (the same file torchvision's own weights
    enum would have downloaded to its hub cache) staged onto the deploy
    volume ahead of time.
    """
    model = resnet50(weights=None)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    return model


class FusionEncoder(nn.Module):
    """DINOv2 + ResNet50@384 + ResNet50@448 -> concat -> PCA-whiten -> L2.

    `embed_images(images: list[bytes]) -> torch.Tensor (B, 256)` is the
    contract `pipeline/muzzle.py::embed_batch` calls. Unlike the plain
    DINOv2 path, this encoder does its OWN resizing per sub-model directly
    from the original photo bytes (three different resize targets), so it
    does not go through `pipeline/preprocess.py`'s single 518x518 transform.
    """

    def __init__(
        self,
        dino_model: Any,
        resnet_model: nn.Module,
        whitening: WhiteningTransform | None,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.dino = dino_model
        self.resnet = resnet_model
        self.whitening = whitening
        self.device = device
        self._tf_dino = _build_transform(_DINO_SIZE)
        self._tf_resnet = {s: _build_transform(s) for s in _RESNET_SIZES}

        self.dino.eval()
        self.resnet.eval()

    def eval(self) -> "FusionEncoder":  # noqa: D102 — keep nn.Module contract
        super().eval()
        self.dino.eval()
        self.resnet.eval()
        return self

    @staticmethod
    def _l2norm(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n = np.clip(n, 1e-12, None)
        return x / n

    @torch.inference_mode()
    def embed_fused_raw(self, images: list[bytes]) -> np.ndarray:
        """DINOv2 + ResNet50@384 + ResNet50@448, concatenated and L2-normed,
        BEFORE whitening. (B, 4352). This is the offline PCA-fitting input
        (scripts/fit_whitening.py) as well as the first half of the served
        `embed_images` path below -- kept as one method so both call sites
        run the identical concat/normalize logic, never two copies.
        """
        if len(images) == 0:
            return np.empty((0, 4352), dtype=np.float32)

        pil_images = [_decode_rgb(b) for b in images]

        dino_batch = torch.stack([self._tf_dino(im) for im in pil_images]).to(self.device)
        dino_emb = self.dino(dino_batch).float().cpu().numpy()  # (B, 256)
        dino_emb = self._l2norm(dino_emb)

        resnet_embs = []
        for size in _RESNET_SIZES:
            tf = self._tf_resnet[size]
            batch = torch.stack([tf(im) for im in pil_images]).to(self.device)
            feat = self.resnet(batch).float().cpu().numpy()  # (B, 2048)
            resnet_embs.append(self._l2norm(feat))

        fused = np.hstack([dino_emb, *resnet_embs])  # (B, 4352)
        return self._l2norm(fused)

    @torch.inference_mode()
    def embed_images(self, images: list[bytes]) -> torch.Tensor:
        """Embed a batch of RAW PHOTO bytes (no YOLO crop) into (B, 256)
        unit-norm PCA-whitened fusion embeddings.
        """
        if self.whitening is None:
            raise RuntimeError(
                "FusionEncoder.embed_images called with no whitening artifact "
                "loaded -- construct with a real WhiteningTransform, or use "
                "embed_fused_raw() directly if you're fitting one."
            )
        if len(images) == 0:
            return torch.empty(0, 256)
        fused = self.embed_fused_raw(images)
        whitened = self.whitening.apply(fused)  # (B, 256), L2-normed inside
        return torch.from_numpy(whitened.astype(np.float32))
