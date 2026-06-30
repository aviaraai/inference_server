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
    get_id_store,
    get_model,
)
from faiss_index import FaissIndex
from godhaar.config import EMB_DIM, MODEL_VERSION
from godhaar.model import GodhaarModel
from id_store import IDStore
from pipeline.color import RuleBasedColorExtractor
from pipeline.muzzle import embed_batch, embed_single
from pipeline.quality import quality_check, quality_check_cv2
from pipeline.yolo_crop import crop_cattle, load_yolo, warmup_yolo
from schema import (
    ColorResult,
    ExtractedColors,
    HealthResponse,
    LatencyMs,
    MatchCandidate,
    RegisterResponse,
    SearchResponse,
    VersionInfo,
)


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("godhaar.server")


# ── Lifespan (Startup / Shutdown) ────────────────────────────────────────────

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
    model_path = os.getenv("MODEL_PATH", "model.pt")
    if not os.path.exists(model_path):
        log.error(f"Model checkpoint not found: {model_path}")
        raise RuntimeError(f"Model checkpoint not found: {model_path}")

    log.info(f"Loading GodhaarModel from {model_path}...")
    model, ckpt = GodhaarModel.load_checkpoint(model_path, device=device)
    model.eval()
    app.state.model = model
    log.info(f"GodhaarModel loaded (epoch={ckpt.get('epoch', '?')})")

    # 3. Load FAISS index (hybrid: load existing, allow online additions)
    faiss_index_path = os.getenv("FAISS_INDEX_PATH", "/appstorage/indexes/faiss.index")
    faiss_index = FaissIndex(embedding_dim=EMB_DIM)
    faiss_index.load(faiss_index_path)
    app.state.faiss_index = faiss_index
    log.info(f"FAISS index: {len(faiss_index)} vectors")

    # 4. Load SQLite ID store
    id_store_path = os.getenv("ID_STORE_PATH", "/appstorage/indexes/id_store.db")
    app.state.id_store = IDStore(id_store_path)

    # 5. Load Color Extractor
    color_extractor = RuleBasedColorExtractor()
    app.state.color_extractor = color_extractor

    # 6. Load YOLO
    yolo_path = os.getenv("YOLO_MODEL_PATH", None)
    load_yolo(yolo_path)

    # 7. Warmup — run dummy inference through both models
    log.info("Running warmup inference...")
    dummy = torch.randn(1, 3, 518, 518, device=device)
    with torch.inference_mode():
        _ = model(dummy)
    warmup_yolo()
    log.info(f"Warmup complete.")

    elapsed = time.monotonic() - start
    log.info(f"Server ready in {elapsed:.1f}s — FAISS={len(faiss_index)} vectors, model={MODEL_VERSION}")

    yield

    # Shutdown
    log.info("Shutting down — saving FAISS index...")
    await faiss_index.save(faiss_index_path)
    app.state.model = None
    log.info("Server shutdown complete.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Godhaar Inference Server",
    version="1.0.0",
    description="Pure ML microservice for cattle muzzle re-identification.",
    lifespan=lifespan,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes to a BGR numpy array."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="cannot_decode_image")
    return img


def _ms_since(start: float) -> int:
    """Milliseconds elapsed since `start` (from time.monotonic())."""
    return int((time.monotonic() - start) * 1000)


# ── POST /register ───────────────────────────────────────────────────────────

@app.post("/register", response_model=RegisterResponse)
async def register(
    cattle_id: str = Form(...),
    muzzle_1: UploadFile = File(...),
    muzzle_2: UploadFile = File(...),
    muzzle_3: UploadFile = File(...),
    front_1: UploadFile = File(...),
    front_2: UploadFile = File(...),
    model: Any = Depends(get_model),
    device: Any = Depends(get_device),
    faiss_index: FaissIndex = Depends(get_faiss_index),
    id_store: IDStore = Depends(get_id_store),
    color_extractor: Any = Depends(get_color_extractor),
):
    request_id = uuid.uuid4().hex
    t_start = time.monotonic()

    log.info(f"[{request_id}] /register cattle_id={cattle_id}")

    # Read all files
    muzzle_bytes = [
        await muzzle_1.read(),
        await muzzle_2.read(),
        await muzzle_3.read(),
    ]
    front_bytes = [
        await front_1.read(),
        await front_2.read(),
    ]

    # ── Quality gate on all muzzle images ────────────────────────────────
    for i, mb in enumerate(muzzle_bytes, 1):
        status, reason = quality_check(mb)
        if status != "GOOD":
            raise HTTPException(
                status_code=422,
                detail=f"muzzle_{i}: {reason}",
            )

    # ── YOLO Crop + Quality on cropped images ────────────────────────────
    t_crop_start = time.monotonic()
    cropped_images = []
    for i, mb in enumerate(muzzle_bytes, 1):
        img_bgr = _decode_image(mb)
        crop, det_status, det_conf = crop_cattle(img_bgr)
        if crop is None:
            raise HTTPException(
                status_code=422,
                detail=f"muzzle_{i}: {det_status}",
            )
        # Quality check on the crop itself
        crop_status, crop_reason = quality_check_cv2(crop)
        if crop_status != "GOOD":
            raise HTTPException(
                status_code=422,
                detail=f"muzzle_{i}_crop: {crop_reason}",
            )
        cropped_images.append(crop)
    t_crop = _ms_since(t_crop_start)

    # ── Embed (batched) ──────────────────────────────────────────────────
    t_embed_start = time.monotonic()
    # Convert cropped BGR images to bytes for the preprocessor
    embeddings = embed_batch(
        [cv2.imencode(".jpg", img)[1].tobytes() for img in cropped_images],
        model,
        device,
    )
    t_embed = _ms_since(t_embed_start)

    # ── Color extraction from front images ───────────────────────────────
    t_color_start = time.monotonic()
    front_img_1 = _decode_image(front_bytes[0])
    front_img_2 = _decode_image(front_bytes[1])

    body_color = color_extractor.extract_body(front_img_1)
    muzzle_color = color_extractor.extract_muzzle(cropped_images[0])
    t_color = _ms_since(t_color_start)

    # ── Add to FAISS + IDStore (locked) ──────────────────────────────────
    t_faiss_start = time.monotonic()
    emb_np = embeddings.numpy()  # (3, 256)
    faiss_ids = await faiss_index.add_batch(emb_np)
    id_store.add_batch([(fid, cattle_id) for fid in faiss_ids])
    t_faiss = _ms_since(t_faiss_start)

    t_total = _ms_since(t_start)

    log.info(
        f"[{request_id}] /register done | cattle_id={cattle_id} | "
        f"faiss_ids={faiss_ids} | {t_total}ms (crop={t_crop} embed={t_embed} "
        f"color={t_color} faiss={t_faiss})"
    )

    return RegisterResponse(
        status="success",
        cattle_id=cattle_id,
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
        latency_ms=LatencyMs(
            total=t_total, crop=t_crop, embed=t_embed, color=t_color, faiss=t_faiss,
        ),
    )


# ── POST /search ─────────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse)
async def search(
    muzzle: UploadFile = File(...),
    front: Optional[UploadFile] = File(None),
    top_k: int = Form(10),
    model: Any = Depends(get_model),
    device: Any = Depends(get_device),
    faiss_index: FaissIndex = Depends(get_faiss_index),
    id_store: IDStore = Depends(get_id_store),
    color_extractor: Any = Depends(get_color_extractor),
):
    request_id = uuid.uuid4().hex
    t_start = time.monotonic()

    log.info(f"[{request_id}] /search top_k={top_k}")

    muzzle_bytes = await muzzle.read()
    front_bytes = await front.read() if front else None

    # ── Quality gate ─────────────────────────────────────────────────────
    q_status, q_reason = quality_check(muzzle_bytes)
    if q_status != "GOOD":
        raise HTTPException(status_code=422, detail=q_reason)

    # ── YOLO Crop ────────────────────────────────────────────────────────
    t_crop_start = time.monotonic()
    img_bgr = _decode_image(muzzle_bytes)
    crop, det_status, det_conf = crop_cattle(img_bgr)
    if crop is None:
        raise HTTPException(status_code=422, detail=det_status)
    t_crop = _ms_since(t_crop_start)

    # ── Embed ────────────────────────────────────────────────────────────
    t_embed_start = time.monotonic()
    crop_bytes = cv2.imencode(".jpg", crop)[1].tobytes()
    embedding = embed_single(crop_bytes, model, device)
    emb_np = embedding.squeeze(0).numpy()  # (256,)
    t_embed = _ms_since(t_embed_start)

    # ── Color extraction ─────────────────────────────────────────────────
    t_color_start = time.monotonic()
    muzzle_color = color_extractor.extract_muzzle(crop)

    if front_bytes:
        front_img = _decode_image(front_bytes)
        body_color = color_extractor.extract_body(front_img)
    else:
        body_color = {"label": "UNKNOWN", "confidence": 0.0, "method": "NO_FRONT_IMAGE"}
    t_color = _ms_since(t_color_start)

    # ── FAISS search ─────────────────────────────────────────────────────
    t_faiss_start = time.monotonic()
    matches = faiss_index.cattle_search(emb_np, id_store, top_k=top_k)
    t_faiss = _ms_since(t_faiss_start)

    t_total = _ms_since(t_start)

    log.info(
        f"[{request_id}] /search done | matches={len(matches)} | "
        f"{t_total}ms (crop={t_crop} embed={t_embed} color={t_color} faiss={t_faiss}) | "
        f"top1={matches[0]['cattle_id'] if matches else 'none'} "
        f"score={matches[0]['score'] if matches else 0}"
    )

    return SearchResponse(
        request_id=request_id,
        query_colors=ExtractedColors(
            body=ColorResult(**body_color),
            muzzle=ColorResult(**muzzle_color),
        ),
        top_matches=[MatchCandidate(**m) for m in matches],
        versions=VersionInfo(
            model=MODEL_VERSION,
            faiss=faiss_index.faiss_version_label,
        ),
        latency_ms=LatencyMs(
            total=t_total, crop=t_crop, embed=t_embed, color=t_color, faiss=t_faiss,
        ),
    )


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health(
    model: Any = Depends(get_model),
    faiss_index: FaissIndex = Depends(get_faiss_index),
    id_store: IDStore = Depends(get_id_store),
    color_extractor: Any = Depends(get_color_extractor),
):
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        faiss_size=len(faiss_index),
        id_store_size=id_store.count(),
        gpu_available=torch.cuda.is_available(),
        model_version=MODEL_VERSION,
        color_extractor_available=color_extractor.available,
    )
