import io

import numpy as np
import torch
from PIL import Image


def preprocess(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize, normalize, etc.
    image = image.resize((224, 224))

    image = np.asarray(image, dtype=np.float32) / 255.0

    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW

    tensor = torch.from_numpy(image).unsqueeze(0)

    return tensor
