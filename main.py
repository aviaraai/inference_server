"""
main.py — Godhaar Inference Server.

A pure ML microservice exposing:
  POST /register  — embed + store cattle in FAISS
  POST /search    — embed + search FAISS + return ML scores
  GET  /health    — liveness check

Business logic (MATCH/REVIEW/NOT_REGISTERED, GPS, policy rules) lives
in the API server, NOT here. This server only returns raw ML scores,
color classifications, and gap calculations.
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

import cv2
import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from dependency import (
    get_color_extractor,
    get_device,
    get_faiss_index,
    get_model,
)
from faiss_index import FaissIndex
from godhaar.config import EMB_DIM, MODEL_VERSION
from godhaar.model import GodhaarModel
from helpers import _decode_image, _ms_since
from pipeline.color import RuleBasedColorExtractor
from pipeline.muzzle import embed_batch
from pipeline.quality import quality_check, quality_check_cv2
from pipeline.yolo_crop import crop_cattle, load_yolo, warmup_yolo
from schema import (
    ColorResult,
    ExtractedColors,
    HealthResponse,
    MatchCandidate,
    RegisterRequest,
    RegisterResponse,
    SearchResponse,
    VersionInfo,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("godhaar.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML resources at startup, clean up on shutdown."""

    start = time.monotonic()

    # 1. Resolve device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.device = device
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. Load GodhaarModel
    model_path = os.getenv("MODEL_PATH")
    if not model_path or not os.path.exists(model_path):
        log.error(f"Model checkpoint not found: {model_path}")
        raise RuntimeError(f"Model checkpoint not found: {model_path}")

    log.info(f"Loading GodhaarModel from {model_path}...")
    model, ckpt = GodhaarModel.load_checkpoint(model_path, device=device)
    model.eval()
    app.state.model = model
    log.info(f"GodhaarModel loaded (epoch={ckpt.get('epoch', '?')})")

    # 3. Load FAISS index (hybrid: load existing, allow online additions)
    faiss_index_path = os.getenv("FAISS_INDEX_PATH")
    if not faiss_index_path:
        log.error(f"FAISS index not found: {faiss_index_path}")
        raise RuntimeError(f"FAISS index not found: {faiss_index_path}")
    faiss_index = FaissIndex(embedding_dim=EMB_DIM)
    faiss_index.load(faiss_index_path)
    app.state.faiss_index = faiss_index
    log.info(f"FAISS index: {len(faiss_index)} vectors")

    # 4. Load Color Extractor
    color_extractor = RuleBasedColorExtractor()
    app.state.color_extractor = color_extractor

    # 5. Load YOLO
    yolo_path = os.getenv("YOLO_MODEL_PATH", None)
    load_yolo(yolo_path)

    # 6. Warmup — run dummy inference through both models
    log.info("Running warmup inference...")
    dummy = torch.randn(1, 3, 518, 518, device=device)
    with torch.inference_mode():
        model(dummy)
    warmup_yolo()
    log.info("Warmup complete.")

    elapsed = time.monotonic() - start
    log.info(
        f"Server ready in {elapsed:.1f}s — FAISS={len(faiss_index)} vectors, model={MODEL_VERSION}"
    )

    yield

    # Shutdown
    log.info("Shutting down — saving FAISS index...")
    await faiss_index.save(faiss_index_path)
    app.state.model = None
    log.info("Server shutdown complete.")


app = FastAPI(
    title="Godhaar Inference Server",
    version="1.0.0",
    description="Pure ML microservice for cattle muzzle re-identification.",
    lifespan=lifespan,
)


@app.post("/register", response_model=RegisterResponse)
async def register(
    register: RegisterRequest,
    model: Any = Depends(get_model),
    device: Any = Depends(get_device),
    faiss_index: FaissIndex = Depends(get_faiss_index),
    color_extractor: Any = Depends(get_color_extractor),
):
    """
    Register endpoint accepts image inputs and returns the faiss indices used to store the embeddings
    """

    t_start = time.monotonic()

    # Read all image files with no blocking
    muzzle_bytes, front_bytes = await asyncio.gather(
        asyncio.gather(
            register.muzzle_1.read(),
            register.muzzle_2.read(),
            register.muzzle_3.read(),
        ),
        asyncio.gather(
            register.front_1.read(),
            register.front_2.read(),
        ),
    )

    # ── Everything CPU/GPU-bound runs in a single thread-offloaded call ───
    # One seam, not five — nothing inside _run_registration_pipeline needs
    # its own asyncio.to_thread wrapper.
    embeddings_np, body_color, muzzle_color = await asyncio.to_thread(
        _run_registration_pipeline,
        muzzle_bytes,
        front_bytes,
        model,
        device,
        color_extractor,
    )

    # ── FAISS write stays awaited on its own: it has real async locking ───
    faiss_ids = await faiss_index.add_batch(embeddings_np)

    t_total = _ms_since(t_start)
    log.info(f"/register done | faiss_ids={faiss_ids} | {t_total}ms")

    return RegisterResponse(
        status="success",
        embedding_ids=faiss_ids,
        extracted_colors=ExtractedColors(
            body=ColorResult(**body_color),
            muzzle=ColorResult(**muzzle_color),
        ),
        versions=VersionInfo(
            model=MODEL_VERSION,
            faiss=faiss_index.faiss_version_label,
            embedding="v1",
        ),
        registered_at=datetime.now(timezone.utc).isoformat(),
    )


def _run_registration_pipeline(
    muzzle_bytes: tuple[bytes, ...],
    front_bytes: tuple[bytes, ...],
    model: Any,
    device: torch.device,
    color_extractor: Any,
) -> tuple[np.ndarray, dict, dict]:
    """
    Runs the full synchronous CPU/GPU pipeline: quality gates, YOLO crop,
    embedding, and color extraction. Executed entirely inside a single
    worker thread via asyncio.to_thread — nothing in here should ever
    need to be async itself.

    Raises HTTPException on any validation/quality failure; it's safe to
    raise HTTPException from inside a thread because asyncio.to_thread
    re-raises it on the awaiting coroutine, where FastAPI's normal
    exception handling picks it up.
    """
    # ── Quality gate on raw muzzle images ──────────────────────────────
    for i, mb in enumerate(muzzle_bytes, 1):
        status, reason = quality_check(mb)
        if status != "GOOD":
            raise HTTPException(status_code=422, detail=f"muzzle_{i}: {reason}")

    # ── YOLO crop + quality check on crops ──────────────────────────────
    cropped_images = []
    for i, mb in enumerate(muzzle_bytes, 1):
        img_bgr = _decode_image(mb)
        crop, det_status, _det_conf = crop_cattle(img_bgr)
        if crop is None:
            raise HTTPException(status_code=422, detail=f"muzzle_{i}: {det_status}")

        crop_status, crop_reason = quality_check_cv2(crop)
        if crop_status != "GOOD":
            raise HTTPException(
                status_code=422, detail=f"muzzle_{i}_crop: {crop_reason}"
            )

        cropped_images.append(crop)

    # ── Embed (batched forward pass) ─────────────────────────────────────
    jpg_bytes = [cv2.imencode(".jpg", img)[1].tobytes() for img in cropped_images]
    embeddings = embed_batch(jpg_bytes, model, device)

    # ── Color extraction from front images ───────────────────────────────
    front_img_1 = _decode_image(front_bytes[0])
    body_color = color_extractor.extract_body(front_img_1)
    muzzle_color = color_extractor.extract_muzzle(cropped_images[0])

    return embeddings.numpy(), body_color, muzzle_color


@app.post("/search", response_model=SearchResponse)
async def search(
    muzzle: UploadFile = File(...),
    front: UploadFile = File(...),
    top_k: int = Form(10),
    model: Any = Depends(get_model),
    device: Any = Depends(get_device),
    faiss_index: FaissIndex = Depends(get_faiss_index),
    color_extractor: Any = Depends(get_color_extractor),
):
    request_id = uuid.uuid4().hex
    t_start = time.monotonic()
    log.info(f"[{request_id}] /search top_k={top_k}")

    # ── Real async I/O: read uploads concurrently ──────────────────────────
    muzzle_bytes, front_bytes = await asyncio.gather(
        muzzle.read(),
        front.read(),
    )

    # ── CPU/GPU pipeline: one thread-offload seam ───────────────────────────
    t_embed_start = time.monotonic()
    emb_np, muzzle_color, body_color = await asyncio.to_thread(
        _run_search_pipeline,
        muzzle_bytes,
        front_bytes,
        model,
        device,
        color_extractor,
    )
    t_embed = _ms_since(t_embed_start)

    # ── FAISS search ─────────────────────────────────────────────────────
    t_faiss_start = time.monotonic()
    matches = await faiss_index.search(emb_np, top_k=top_k)
    t_faiss = _ms_since(t_faiss_start)

    t_total = _ms_since(t_start)
    log.info(
        f"[{request_id}] /search done | matches={len(matches)} | "
        f"{t_total}ms (embed={t_embed} faiss={t_faiss}) | "
        f"top1_faiss_id={matches[0]['faiss_id'] if matches else 'none'} "
        f"score={matches[0]['score'] if matches else 0}"
    )

    return SearchResponse(
        request_id=request_id,
        query_colors=ExtractedColors(
            body=ColorResult(**body_color),
            muzzle=ColorResult(**muzzle_color),
        ),
        top_matches=[
            MatchCandidate(**m) for m in matches
        ],  # {"faiss_id": int, "score": float}
        versions=VersionInfo(
            model=MODEL_VERSION,
            faiss=faiss_index.faiss_version_label,
            embedding=None,
        ),
    )


def _run_search_pipeline(
    muzzle_bytes: bytes,
    front_bytes: Optional[bytes],
    model: Any,
    device: torch.device,
    color_extractor: Any,
) -> tuple[np.ndarray, dict, dict]:
    """
    Synchronous CPU/GPU pipeline for /search: quality gate, crop, embed,
    color extraction. Runs entirely inside asyncio.to_thread.
    """
    q_status, q_reason = quality_check(muzzle_bytes)
    if q_status != "GOOD":
        raise HTTPException(status_code=422, detail=q_reason)

    img_bgr = _decode_image(muzzle_bytes)
    crop, det_status, _det_conf = crop_cattle(img_bgr)
    if crop is None:
        raise HTTPException(status_code=422, detail=det_status)

    crop_status, crop_reason = quality_check_cv2(crop)
    if crop_status != "GOOD":
        raise HTTPException(status_code=422, detail=f"muzzle_crop: {crop_reason}")

    crop_bytes = cv2.imencode(".jpg", crop)[1].tobytes()
    embedding = embed_batch([crop_bytes], model, device)  # (1, 256)
    emb_np = embedding.squeeze(0).numpy()  # (256,)

    muzzle_color = color_extractor.extract_muzzle(crop)
    if front_bytes:
        front_img = _decode_image(front_bytes)
        body_color = color_extractor.extract_body(front_img)
    else:
        body_color = {"label": "UNKNOWN", "confidence": 0.0, "method": "NO_FRONT_IMAGE"}

    return emb_np, muzzle_color, body_color


# ── GET /health ───────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health(
    model: Any = Depends(get_model),
    faiss_index: FaissIndex = Depends(get_faiss_index),
    color_extractor: Any = Depends(get_color_extractor),
):
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        faiss_size=len(faiss_index),
        gpu_available=torch.cuda.is_available(),
        model_version=MODEL_VERSION,
        color_extractor_available=color_extractor.available,
    )
