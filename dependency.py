"""
dependency.py — FastAPI dependency injection for shared resources.
"""

from typing import Any

from fastapi import HTTPException, Request

from faiss_index import FaissIndex
from id_store import IDStore
from pipeline.color import ColorExtractor


# ── Model ─────────────────────────────────────────────────────────────────────

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


# ── FAISS ─────────────────────────────────────────────────────────────────────

def get_faiss_index(request: Request) -> FaissIndex:
    """Retrieve the FAISS index from application state."""
    faiss_index = getattr(request.app.state, "faiss_index", None)
    if faiss_index is None:
        raise HTTPException(
            status_code=500,
            detail="faiss_index_not_loaded",
        )
    return faiss_index


# ── ID Store ──────────────────────────────────────────────────────────────────

def get_id_store(request: Request) -> IDStore:
    """Retrieve the SQLite ID store from application state."""
    id_store = getattr(request.app.state, "id_store", None)
    if id_store is None:
        raise HTTPException(
            status_code=500,
            detail="id_store_not_loaded",
        )
    return id_store


# ── Color Extractor ───────────────────────────────────────────────────────────

def get_color_extractor(request: Request) -> ColorExtractor:
    """Retrieve the color extractor from application state."""
    extractor = getattr(request.app.state, "color_extractor", None)
    if extractor is None:
        raise HTTPException(
            status_code=500,
            detail="color_extractor_not_loaded",
        )
    return extractor
