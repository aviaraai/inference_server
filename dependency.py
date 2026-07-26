"""
dependency.py — FastAPI dependency injection for shared resources.
"""

from typing import Any

from fastapi import HTTPException, Request

from faiss_index import FaissIndex
from pipeline.color import ColorExtractor


def get_model(request: Request) -> Any:
    """Retrieve the loaded GodhaarModel from application state."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="model_not_loaded",
        )
    return model


def get_device(request: Request):
    """Retrieve the torch device from application state."""
    device = getattr(request.app.state, "device", None)
    if device is None:
        raise HTTPException(
            status_code=500,
            detail="device_not_configured",
        )
    return device


def get_faiss_index(request: Request) -> FaissIndex:
    """Retrieve the FAISS index from application state."""
    faiss_index = getattr(request.app.state, "faiss_index", None)
    if faiss_index is None:
        raise HTTPException(
            status_code=500,
            detail="faiss_index_not_loaded",
        )
    return faiss_index


def get_color_extractor(request: Request) -> ColorExtractor:
    """Retrieve the color extractor from application state."""
    extractor = getattr(request.app.state, "color_extractor", None)
    if extractor is None:
        raise HTTPException(
            status_code=500,
            detail="color_extractor_not_loaded",
        )
    return extractor
