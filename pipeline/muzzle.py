from typing import Any

import torch
from preprocess import preprocess


def pipeline(image, model):
    image_bytes = image.file.read()

    tensor = preprocess(image_bytes)

    embedding = run_inference(tensor, model)

    return embedding


def run_inference(input_tensor: torch.Tensor, model: Any):
    with torch.inference_mode():
        output_tensor = model(input_tensor)

    return output_tensor
