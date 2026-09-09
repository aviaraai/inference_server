"""pipeline/tests/test_fusion_encoder_determinism.py

Stage 2 hard requirement: embedding the same image twice through the real
FusionEncoder must be bit-identical. Both sub-encoders are held in eval()
mode (no dropout/batchnorm-update randomness) and the whitening apply is
pure linear algebra, so nondeterminism here would mean a real bug (e.g. a
stray .train() call, or cuDNN nondeterministic kernels).

Requires real model files -- skipped (not failed) if they aren't present,
consistent with this repo's other model-dependent tests.
"""
import io
import os

import numpy as np
import pytest
import torch
from PIL import Image

MODEL_PATH = os.getenv(
    "TEST_DINO_MODEL_PATH", r"D:\Group Projects\inference_server\appstorage\Models\model.pt"
)
RESNET_PATH = os.getenv(
    "TEST_RESNET_MODEL_PATH",
    r"D:\Group Projects\inference_server\appstorage\Models\resnet50\resnet50_imagenet1k_v2.pth",
)
WHITENING_PATH = os.getenv(
    "TEST_WHITENING_MODEL_PATH",
    r"D:\Group Projects\inference_server\appstorage\Models\whitening\whitening_v1.npz",
)

_missing = [p for p in (MODEL_PATH, RESNET_PATH, WHITENING_PATH) if not os.path.exists(p)]


@pytest.mark.skipif(bool(_missing), reason=f"model files not present: {_missing}")
def test_embed_images_is_deterministic():
    from godhaar.model import GodhaarModel
    from pipeline.fusion_encoder import FusionEncoder, load_resnet50
    from pipeline.whitening import WhiteningTransform

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino, _ = GodhaarModel.load_checkpoint(MODEL_PATH, device=device)
    dino.eval()
    resnet = load_resnet50(RESNET_PATH, device)
    whitening = WhiteningTransform.load(WHITENING_PATH)
    encoder = FusionEncoder(dino, resnet, whitening, device)
    encoder.eval()

    img = Image.new("RGB", (640, 480), color=(60, 90, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    jpg_bytes = buf.getvalue()

    e1 = encoder.embed_images([jpg_bytes]).numpy()
    e2 = encoder.embed_images([jpg_bytes]).numpy()

    assert np.array_equal(e1, e2), "FusionEncoder.embed_images is not bit-identical across repeated calls"
    assert e1.shape == (1, 256)
    assert np.isclose(np.linalg.norm(e1[0]), 1.0, atol=1e-4)
