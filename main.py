"""
main.py — Godhaar Inference Server.

A pure ML microservice exposing:
  POST /register  — embed + store cattle in FAISS
  POST /search    — embed + rank provided candidates + return ML scores
  GET  /health    — liveness check

Business logic (MATCH/REVIEW/NOT_REGISTERED, GPS, policy rules) lives
in the API server, NOT here. This server only returns raw ML scores,
color classifications, and gap calculations.
"""

import asyncio
import json
from collections import Counter
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from dependency import (
    get_color_extractor,
    get_device,
    get_faiss_index,
    get_model,
    get_morphology_extractor,
)
from faiss_index import FaissIndex
from godhaar.config import DUPLICATE_THRESHOLD, EMB_DIM, MODEL_VERSION
from godhaar.model import GodhaarModel
from helpers import _decode_image, _ms_since
from pipeline.color import RuleBasedColorExtractor
from pipeline.morphology import RuleBasedMorphologyExtractor, average_readings
from pipeline.muzzle import embed_batch
from pipeline.quality import quality_check, quality_check_cv2
from pipeline.yolo_crop import crop_cattle, load_yolo, warmup_yolo
from schema import (
    CandidateInfo,
    ColorResult,
    ExtractedColors,
    HealthResponse,
    MatchCandidate,
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

    # 4b. Load Morphology Extractor (horn/ear proportions — unvalidated v1
    #     heuristic, see pipeline/morphology.py)
    app.state.morphology_extractor = RuleBasedMorphologyExtractor()

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


@app.post("/register", response_model=RegisterResponse, status_code=201)
async def register(
    muzzle_images: list[UploadFile] = File(...),
    front_images: list[UploadFile] = File(...),
    candidate_json: str = Form(..., alias="candidates"),
    model: Any = Depends(get_model),
    device: Any = Depends(get_device),
    faiss_index: FaissIndex = Depends(get_faiss_index),
    color_extractor: Any = Depends(get_color_extractor),
    morphology_extractor: Any = Depends(get_morphology_extractor),
):
    """
    Register a cattle animal.

    Expects exactly 3 muzzle images and 2 front images.
    Requires ``candidates`` — a JSON string of nearby cattle
    (pre-filtered by GPS) with their stored colors:
        [{"faiss_id": 123, "body_color": "BLACK", "muzzle_color": "PINK"}, ...]
    Send "[]" if no nearby cattle exist (first registration in the area).

    Duplicate detection
    -------------------
    A candidate is a duplicate if BOTH conditions are true:
      1. Embedding cosine similarity ≥ DUPLICATE_THRESHOLD (0.95)
      2. Body color AND muzzle color match the newly extracted colors

    If duplicate → HTTP 409, nothing stored in FAISS.

    HTTP status codes
    -----------------
    201 — registered successfully
    409 — duplicate muzzle detected, not stored
    422 — bad input (wrong image count, quality failure, no detection)
    500 — FAISS or internal system failure
    """
    if len(muzzle_images) != 3:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 3 muzzle images, got {len(muzzle_images)}",
        )
    if len(front_images) != 2:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 2 front images, got {len(front_images)}",
        )

    # Parse candidates JSON — always required, send "[]" if no nearby cattle
    try:
        candidate_list: list[CandidateInfo] = [CandidateInfo(**c) for c in json.loads(candidate_json)]
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid candidates format: {e}",
        )

    t_start = time.monotonic()

    # Read all image files concurrently — pure async I/O, no thread needed.
    muzzle_bytes, front_bytes = await asyncio.gather(
        asyncio.gather(*[img.read() for img in muzzle_images]),
        asyncio.gather(*[img.read() for img in front_images]),
    )

    # ── Everything CPU/GPU-bound runs in a single thread-offloaded call ───
    embeddings_np, body_color, muzzle_color, morphology = await asyncio.to_thread(
        _run_registration_pipeline,
        muzzle_bytes,
        front_bytes,
        model,
        device,
        color_extractor,
        morphology_extractor,
    )

    new_body  = body_color["label"]    # e.g. "BLACK"
    new_muzzle = muzzle_color["label"] # e.g. "PINK"

    # ── Duplicate check: embedding similarity + color match ───────────────
    potential_matches: list[MatchCandidate] = []

    if candidate_list:
        candidate_ids = [c.faiss_id for c in candidate_list]
        candidate_colors = {c.faiss_id: c for c in candidate_list}

        try:
            avg_embedding = embeddings_np.mean(axis=0)  # (256,)
            matches = await faiss_index.restricted_search(
                avg_embedding, candidate_ids=candidate_ids, top_k=5
            )
            potential_matches = [MatchCandidate(**m) for m in matches]
        except Exception as e:
            log.error(f"FAISS duplicate check failed: {e}")
            raise HTTPException(status_code=500, detail=f"faiss_error: {e}")

        for match in potential_matches:
            if match.score < DUPLICATE_THRESHOLD:
                break  # sorted descending — no point checking further

            stored = candidate_colors[match.faiss_id]
            color_match = (
                stored.body_color == new_body
                and stored.muzzle_color == new_muzzle
            )

            if color_match:
                log.info(
                    f"/register 409 duplicate | "
                    f"score={match.score:.4f} | "
                    f"body={new_body} muzzle={new_muzzle} | "
                    f"matched_faiss_id={match.faiss_id}"
                )
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "duplicate_muzzle",
                        "top_score": match.score,
                        "matched_faiss_id": match.faiss_id,
                    },
                )

    # ── Store in FAISS ────────────────────────────────────────────────────
    try:
        faiss_ids = await faiss_index.add_batch(embeddings_np)
    except Exception as e:
        log.error(f"FAISS write failed: {e}")
        raise HTTPException(status_code=500, detail=f"faiss_error: {e}")

    t_total = _ms_since(t_start)
    log.info(f"/register 201 | faiss_ids={faiss_ids} | {t_total}ms")

    return RegisterResponse(
        status="success",
        embedding_ids=faiss_ids,
        extracted_colors=ExtractedColors(
            body=ColorResult(**body_color),
            muzzle=ColorResult(**muzzle_color),
        ),
        horn_shape=morphology["horn_shape"],
        potential_matches=potential_matches,
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
    morphology_extractor: Any,
) -> tuple[np.ndarray, dict, dict, dict]:
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
    # ── Quality gate + YOLO crop + crop-quality, evaluated for ALL 3 images ──
    # Every muzzle image is checked before any failure is raised, so a single
    # 422 names every bad slot at once instead of the caller discovering them
    # one retake at a time (each retake is a full network round trip). Each
    # image still stops at its OWN first failure (quality, then detection,
    # then crop-quality) — only the across-images fail-fast was removed.
    errors: list[str] = []
    cropped_images: list[np.ndarray | None] = [None] * len(muzzle_bytes)
    for i, mb in enumerate(muzzle_bytes, 1):
        status, reason = quality_check(mb)
        if status != "GOOD":
            errors.append(f"muzzle_{i}: {reason}")
            continue

        img_bgr = _decode_image(mb)
        crop, det_status, _det_conf = crop_cattle(img_bgr)
        if crop is None:
            errors.append(f"muzzle_{i}: {det_status}")
            continue

        crop_status, crop_reason = quality_check_cv2(crop)
        if crop_status != "GOOD":
            errors.append(f"muzzle_{i}_crop: {crop_reason}")
            continue

        cropped_images[i - 1] = crop

    if errors:
        raise HTTPException(status_code=422, detail="; ".join(errors))

    # ── Embed (batched forward pass) ─────────────────────────────────────
    jpg_bytes = [cv2.imencode(".jpg", img)[1].tobytes() for img in cropped_images]
    embeddings = embed_batch(jpg_bytes, model, device)

    # ── Color extraction with consistency check ──────────────────────────
    #
    # Body (2 front images): BOTH must agree on the same label.
    #   Disagree → 422, ask user to retake.
    #
    # Muzzle (3 crops): majority (≥2/3) must agree on the same label.
    #   Majority found → accept majority label (avg confidence of agreeing images).
    #   All 3 different → 422, ask user to retake.

    body_colors = [color_extractor.extract_body(_decode_image(fb)) for fb in front_bytes]
    body_labels = [c["label"] for c in body_colors]

    if body_labels[0] != body_labels[1]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"body_color_inconsistent: front images disagree "
                f"({body_labels[0]} vs {body_labels[1]}), retake photos"
            ),
        )
    # Both agree — pick highest confidence reading
    body_color = max(body_colors, key=lambda c: c["confidence"])

    muzzle_colors = [color_extractor.extract_muzzle(crop) for crop in cropped_images]
    muzzle_labels = [c["label"] for c in muzzle_colors]

    # Count votes per label
    vote_counts = Counter(muzzle_labels)
    majority_label, majority_count = vote_counts.most_common(1)[0]

    if majority_count < 2:
        # All 3 different — no majority
        raise HTTPException(
            status_code=422,
            detail=(
                f"muzzle_color_inconsistent: no majority among crops "
                f"({', '.join(muzzle_labels)}), retake photos"
            ),
        )
    # Majority found — use avg confidence of agreeing images
    agreeing = [c for c in muzzle_colors if c["label"] == majority_label]
    avg_conf  = sum(c["confidence"] for c in agreeing) / len(agreeing)
    muzzle_color = {"label": majority_label, "confidence": avg_conf}

    # ── Morphology (horn/ear proportions) — return-only, not a gate ───────
    # Unlike color, there's no majority/consistency check here: this is a
    # continuous, unvalidated heuristic (see pipeline/morphology.py), not a
    # categorical label, so "the 2 photos disagree" isn't a retake-worthy
    # error the way a color mismatch is. Both readings are combined via a
    # confidence-weighted average instead.
    morphology_readings = [
        morphology_extractor.extract(_decode_image(fb)) for fb in front_bytes
    ]
    morphology = average_readings(morphology_readings)

    return embeddings.numpy(), body_color, muzzle_color, morphology


@app.post("/search", response_model=SearchResponse)
async def search(
    muzzle: UploadFile = File(...),
    front: UploadFile = File(...),
    top_k: int = Form(5),
    candidate_json: str = Form(..., alias="candidates"),
    model: Any = Depends(get_model),
    device: Any = Depends(get_device),
    faiss_index: FaissIndex = Depends(get_faiss_index),
    color_extractor: Any = Depends(get_color_extractor),
    morphology_extractor: Any = Depends(get_morphology_extractor),
):
    """
    Search endpoint: receives candidates from the API server (same shape as
    /register's `candidates`, faiss_id + whatever stored color/morphology
    the caller has), embeds the query muzzle, then ranks only those
    candidates via restricted_search (reconstruct → dot product → sort).

    Each stored candidate's color/horn_shape is echoed back on its
    corresponding entry in `top_matches`, alongside the query's own
    freshly-extracted `horn_shape`/`query_colors` at the top level — so
    both sides are available to compare without a second lookup. No
    comparison/similarity math is computed here; this endpoint returns
    data only, same as /register does for horn_shape (see CLAUDE.md).

    No index-wide FAISS search is performed. The API server decides
    which candidates to send based on GPS / Supabase filtering.

    ⚠️ Contract change: this used to accept a bare repeated `candidate_ids`
    form field. It now expects a `candidates` field carrying the same JSON
    shape /register already uses (list of CandidateInfo). Callers built
    against the old bare-ID contract will get a 422 until updated.
    """
    request_id = uuid.uuid4().hex
    t_start = time.monotonic()

    try:
        candidate_list: list[CandidateInfo] = [CandidateInfo(**c) for c in json.loads(candidate_json)]
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid candidates format: {e}")

    if not candidate_list:
        raise HTTPException(status_code=422, detail="candidates must contain at least one entry")

    candidate_lookup = {c.faiss_id: c for c in candidate_list}
    candidate_ids = [c.faiss_id for c in candidate_list]

    log.info(f"[{request_id}] /search top_k={top_k} candidates={len(candidate_ids)}")

    # ── Real async I/O: read uploads concurrently ──────────────────────────
    muzzle_bytes, front_bytes = await asyncio.gather(
        muzzle.read(),
        front.read()
    )

    # ── CPU/GPU pipeline: one thread-offload seam ───────────────────────────
    t_embed_start = time.monotonic()
    emb_np, muzzle_color, body_color, morphology = await asyncio.to_thread(
        _run_search_pipeline,
        muzzle_bytes,
        front_bytes,
        model,
        device,
        color_extractor,
        morphology_extractor,
    )
    t_embed = _ms_since(t_embed_start)

    # ── Restricted search: rank only the provided candidates ─────────────
    t_faiss_start = time.monotonic()
    matches = await faiss_index.restricted_search(emb_np, candidate_ids=candidate_ids, top_k=top_k)
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
        horn_shape=morphology["horn_shape"],
        top_matches=[
            MatchCandidate(
                **m,
                body_color=candidate_lookup[m["faiss_id"]].body_color,
                muzzle_color=candidate_lookup[m["faiss_id"]].muzzle_color,
                horn_shape=candidate_lookup[m["faiss_id"]].horn_shape,
            )
            for m in matches
        ],
        versions=VersionInfo(
            model=MODEL_VERSION,
            faiss=faiss_index.faiss_version_label,
            embedding=None,
        ),
    )


def _run_search_pipeline(
    muzzle_bytes: bytes,
    front_bytes: bytes,
    model: Any,
    device: torch.device,
    color_extractor: Any,
    morphology_extractor: Any,
) -> tuple[np.ndarray, dict, dict, dict]:
    """
    Synchronous CPU/GPU pipeline for /search: quality gate, crop, embed,
    color extraction. Runs entirely inside asyncio.to_thread.
    """
    q_status, q_reason = quality_check(muzzle_bytes)
    if q_status != "GOOD":
        log.error(f"Quality status: {q_status}")
        log.error(f"Quality reason: {q_reason}")
        raise HTTPException(status_code=422, detail=q_reason)

    img_bgr = _decode_image(muzzle_bytes)
    crop, det_status, _det_conf = crop_cattle(img_bgr)
    if crop is None:
        log.error(f"Crop: {crop} | Det Status: {det_status} | Det conf: {_det_conf}")
        raise HTTPException(status_code=422, detail=det_status)

    crop_status, crop_reason = quality_check_cv2(crop)
    if crop_status != "GOOD":
        log.error(f"Crop Status: {crop_status} | Crop reason: {crop_reason}")
        raise HTTPException(status_code=422, detail=f"muzzle_crop: {crop_reason}")

    crop_bytes = cv2.imencode(".jpg", crop)[1].tobytes()
    embedding = embed_batch([crop_bytes], model, device)  # (1, 256)
    emb_np = embedding.squeeze(0).numpy()  # (256,)

    muzzle_color = color_extractor.extract_muzzle(crop)
    front_img = _decode_image(front_bytes)
    body_color = color_extractor.extract_body(front_img)
    morphology = morphology_extractor.extract(front_img)

    return emb_np, muzzle_color, body_color, morphology


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
