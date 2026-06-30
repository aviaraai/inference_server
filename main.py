import os
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import Depends, FastAPI, HTTPException

from dependency import get_faiss_index, get_model
from faiss_index import FaissIndex
from pipeline.front import pipeline as front_pipeline
from pipeline.muzzle import pipeline as muzzle_pipeline
from schema import Register, Search


# Modern FastAPI lifespan manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Retrieve model path from environment variable or use a default
    model_path = os.getenv("MODEL_PATH", "model.pt")

    # For seamless local development and testing, create a dummy model if it doesn't exist
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Generating a dummy model...")
        dummy_model = torch.nn.Linear(10, 2)
        torch.save(dummy_model, model_path)
        print(f"Dummy model successfully saved to '{model_path}'")

    try:
        # Load the PyTorch (.pt) model into memory
        # 'map_location=torch.device("cpu")' ensures it loads fine on any system without GPU
        # 'weights_only=False' is set to allow loading full modules safely from a trusted source
        model = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=False
        )

        # Set to evaluation mode if it's a torch.nn.Module
        if isinstance(model, torch.nn.Module):
            model.eval()

        app.state.model = model
        # Embedding dimension set to 256 as per your latest message.
        # TODO: Modify if required
        app.state.faiss_index = FaissIndex(embedding_dim=256)
        print(
            f"Successfully loaded PyTorch model from '{model_path}' and stored it in application state."
        )
    except Exception as e:
        app.state.model = None
        print(f"Error loading PyTorch model from '{model_path}': {e}")
        raise RuntimeError(f"Could not load ML model: {e}")

    yield

    # Clean up resources on shutdown
    app.state.model = None
    print("Application shutdown: Model cleared from memory.")


# Initialize FastAPI application with the lifespan context manager
app = FastAPI(lifespan=lifespan)


@app.post("register")
def register(
    register: Register,
    faiss_index: FaissIndex = Depends(get_faiss_index),
    model: Any = Depends(get_model),
):
    try:
        # Begins inference on all muzzle images serially
        # These tasks are done serially, not parallely to not increase load on the server
        embeddings = [
            muzzle_pipeline(register.muzzle_1, model),
            muzzle_pipeline(register.muzzle_2, model),
            muzzle_pipeline(register.muzzle_3, model),
        ]

        rules = [front_pipeline(register.front_1), front_pipeline(register.front_2)]

        ids = []

        for embedding in embeddings:
            embedding = embedding.squeeze(0).cpu().numpy()

            ids.append(faiss_index.add(embedding))

        return {"status": "success", "embedding_ids": ids, "rules": rules}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error occurred: {e}",
        )


@app.post("/search")
def predict(
    search: Search,
    faiss_index: FaissIndex = Depends(get_faiss_index),
    model: Any = Depends(get_model),
):
    try:
        embedding = muzzle_pipeline(search.muzzle, model).squeeze(0).cpu().numpy()
        id = faiss_index.add(embedding)
        rule = front_pipeline(search.front)
        return {"status": "success", "id": id, "rule": rule}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error occurred: {e}",
        )
