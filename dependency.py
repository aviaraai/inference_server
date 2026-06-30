from typing import Any

from fastapi import HTTPException, Request


# Dependency function to retrieve the loaded model
def get_model(request: Request) -> Any:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Machine learning model is not loaded or initialized.",
        )
    return model


# Dependency function to retrieve the faiss index
def get_faiss_index(request: Request) -> Any:
    faiss_index = getattr(request.app.state, "faiss_index", None)
    if faiss_index is None:
        raise HTTPException(
            status_code=500,
            detail="Machine learning model is not loaded or initialized.",
        )
    return faiss_index
