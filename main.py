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
from typing import Any, Optional

import cv2
import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from cctv.database import init_db as init_cctv_db
from cctv.routes import router as cctv_router
from dependency import (
    get_color_extractor,
    get_device,
    get_faiss_index,
    get_model,
    get_morphology_extractor,
)
from errors import (
    DomainError,
    classify_detection_status,
    classify_quality_reason,
    color_inconsistency_error,
    color_readings,
    domain_error_handler,
    duplicate_animal_error,
    image_quality_error,
)
from faiss_index import FaissIndex
from godhaar.config import (
    BODY_COLOR_MAJORITY_CONFIDENCE,
    DUPLICATE_HIGH_CONFIDENCE_THRESHOLD,
    DUPLICATE_THRESHOLD,
    EMB_DIM,
    MODEL_VERSION,
)
from godhaar.model import GodhaarModel
from helpers import _decode_image, _ms_since
from pipeline.color import RuleBasedColorExtractor
from pipeline.morphology import RuleBasedMorphologyExtractor, average_readings
from pipeline.muzzle import embed_batch
from pipeline.pose import (
    combine_geometry,
    extract_face_geometry,
    load_pose_model,
    pose_available,
    warmup_pose_model,
)
from pipeline.quality import quality_check, quality_check_cv2
from pipeline.muzzle_detect import load_muzzle_detector, warmup_muzzle_detector
from pipeline.muzzle_crop_cache import (
    load_crop as load_cached_muzzle_crop,
    load_features as load_cached_muzzle_features,
    save_crop as save_muzzle_crop,
    save_features as save_muzzle_features,
)
from pipeline.lightglue_verify import (
    LIGHTGLUE_TIEBREAKER_ENABLED,
    available as lightglue_available,
    classify_zone as lightglue_classify_zone,
    extract_features_np as lightglue_extract_features_np,
    load_lightglue,
    verify as lightglue_verify,
    verify_with_cached_candidate as lightglue_verify_with_cached_candidate,
    warmup_lightglue,
)
from pipeline.yolo_crop import crop_cattle, load_yolo, warmup_yolo
from schema import (
    CandidateInfo,
    ColorResult,
    ErrorCode,
    ExtractedColors,
    FaceGeometry,
    HealthResponse,
    ImageFailure,
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

# ── TEMPORARY: accept every offline registration regardless of quality ────────
# Set True on the user's explicit instruction, 2026-08-08: officers are
# registering cattle offline right now and every 422 (blur, exposure, no
# detection, body/muzzle color inconsistency) is blocking real field data from
# ever syncing. While this is True, every quality/consistency gate below still
# RUNS and still LOGS what it would have rejected, but falls back to the best
# available reading instead of raising -- nothing is silently skipped, it is
# visibly downgraded to "accepted anyway".
#
# Deliberately NOT covering the 409 duplicate-animal check -- that stays
# exactly as built (embedding + color, with the tag_no veto). Confirmed
# explicitly: this flag is about image/consistency quality only.
#
# REVERTED 2026-08-09: field sync is done, quality gates matter again.
BYPASS_QUALITY_GATES = False

# ── TEMPORARY: accept every registration regardless of duplicate match ────────
# Set True on the user's explicit instruction, 2026-08-09: real field
# registrations are still hitting 409 DUPLICATE_ANIMAL because go-apiserver
# does not send tag_no on candidates yet (the veto above has nothing to
# compare against, so it never fires and every color+embedding match still
# rejects). Until that's wired up, disable the duplicate check outright
# rather than block real data collection on a check that cannot currently
# use its own escape hatch.
#
# A SEPARATE flag from BYPASS_QUALITY_GATES on purpose: once syncing is done
# these need to be re-enabled independently, not as a pair.
#
# While True, the check still RUNS and LOGS what it would have rejected
# (including the tag_no veto's own reasoning, when it has data to reason
# with) -- it just never raises.
#
# REVERTED 2026-08-09: field sync is done, duplicate check runs for real
# again (now against the tiered verdict above, not the old flat AND-gate).
BYPASS_DUPLICATE_CHECK = False

# ── /search fusion tiebreaker: LightGlue as an additive, inert signal ────────
# These mirror go-apiserver's decision.go thresholds (matchThreshold=0.82,
# reviewThreshold=0.72, gapThreshold=0.08) — NOT a copy of business logic
# living here (godhaar/config.py's docstring rule against that still holds).
# inference_server never decides MATCH/REVIEW/UNKNOWN; this is used only to
# decide whether it's worth spending the extra GPU time to compute an optional
# signal that go-apiserver may or may not ever read. Applied to
# inference_server's OWN top_matches ranking — raw per-embedding FAISS scores,
# before go-apiserver aggregates multiple embeddings per animal and applies
# its attribute-agreement adjustment — so this is an approximation of what
# go-apiserver will ultimately decide, not identical to it. There is no
# shared source between the two repos: if decision.go's thresholds change,
# these need updating by hand. See CLAUDE.md.
_SEARCH_MATCH_THRESHOLD_MIRROR = 0.82
_SEARCH_REVIEW_THRESHOLD_MIRROR = 0.72
_SEARCH_GAP_THRESHOLD_MIRROR = 0.08


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

    # 5b. No-op — the dedicated muzzle detector is permanently absent by
    #     design (CTO-confirmed, 2026-08-17), not a missing deployment step.
    #     Muzzle color is sampled from a fixed center crop of the
    #     whole-animal box; see pipeline/muzzle_detect.py.
    load_muzzle_detector()

    # 5c. Load the LightGlue /search fusion tiebreaker. Optional, same
    #     fail-open contract as the muzzle detector: if absent (no network to
    #     fetch cached weights on first boot, see CLAUDE.md), /search just
    #     never runs the tiebreaker — lightglue_checked stays False.
    load_lightglue()

    # 5d. Load the cattle-face pose model (8 keypoints → face geometry, see
    #     pipeline/pose.py). OPTIONAL, env-var-driven exactly like
    #     YOLO_MODEL_PATH above — no bundled default. This is a supplementary
    #     demote-only signal, NOT the primary match, so a missing/unloadable
    #     model is a warning, never a startup failure (unlike MODEL_PATH):
    #     /register then just returns face_geometry with status=NO_FACE.
    load_pose_model(os.getenv("POSE_MODEL_PATH", None))

    # 6. Warmup — run dummy inference through both models
    log.info("Running warmup inference...")
    dummy = torch.randn(1, 3, 518, 518, device=device)
    with torch.inference_mode():
        model(dummy)
    warmup_yolo()
    warmup_muzzle_detector()
    warmup_lightglue()
    warmup_pose_model()
    log.info("Warmup complete.")

    # 7. Init CCTV model's session DB (third model — video analytics)
    init_cctv_db()
    log.info("CCTV model ready — session DB initialised.")

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

# Third model, alongside detection (pipeline/) and identification (godhaar/):
# video-based cattle counting/tracking/analytics. Self-contained under /cctv.
app.include_router(cctv_router)

# Renders DomainError as {"error_code": ..., "detail": {...}}. Plain
# HTTPExceptions are untouched and keep answering with FastAPI's bare
# {"detail": ...} — see the module docstring in errors.py for why that split
# matters.
#
# This supersedes the inline InferenceError handler the cctv branch carried:
# both put the same envelope on the wire, and errors.py is the factored version
# the /register and /search paths on this branch already raise through.
app.add_exception_handler(DomainError, domain_error_handler)


@app.post("/register", response_model=RegisterResponse, status_code=201)
async def register(
    muzzle_images: list[UploadFile] = File(...),
    front_images: list[UploadFile] = File(...),
    candidate_json: str = Form(..., alias="candidates"),
    # The animal's physical ear-tag number (not a purchase price). go-apiserver
    # sends it as its own `tag_no` form field. NOT persisted here — go-apiserver
    # owns storing the real tag_no — but IS used below as a veto signal against
    # the embedding+color duplicate verdict.
    tag_no: Optional[str] = Form(None),
    model: Any = Depends(get_model),
    device: Any = Depends(get_device),
    faiss_index: FaissIndex = Depends(get_faiss_index),
    color_extractor: Any = Depends(get_color_extractor),
    morphology_extractor: Any = Depends(get_morphology_extractor),
):
    """
    Register a cattle animal.

    Expects exactly 3 muzzle images and 2 front images.

    ``tag_no`` is the animal's physical ear-tag number (not a price) — see
    the parameter comment above.
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

    Error bodies
    ------------
    Rejections this server decides on (409, and 422s about the photos) carry
    {"error_code": ..., "detail": {...}} — see schema.ErrorCode. A 409 always
    includes detail.matched_faiss_id so the caller can name the animal being
    duplicated.

    Failures caused by the caller sending the wrong thing — image counts,
    malformed candidates — keep FastAPI's plain {"detail": ...} with no
    error_code, which is how the API server tells the two cases apart.
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
    embeddings_np, body_color, muzzle_color, morphology, face_geometry, cropped_images = await asyncio.to_thread(
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
            body_match = stored.body_color == new_body
            muzzle_match = stored.muzzle_color == new_muzzle

            # Tiered, not a flat AND-gate. body_color comes from
            # detect_primary_animal() (pipeline/yolo_crop.py), which has NO
            # quality/multi-cattle gate — unlike muzzle_color, which only
            # ever comes from photos that already passed one. Requiring both
            # to agree let one bad body_color read (a crowded goshala front
            # photo picking a neighbour's coat) silently defeat a genuinely
            # correct muzzle-embedding match — reported live as a real
            # double-registration. See DUPLICATE_HIGH_CONFIDENCE_THRESHOLD's
            # comment in godhaar/config.py for the full incident.
            if match.score >= DUPLICATE_HIGH_CONFIDENCE_THRESHOLD:
                # Embedding alone is decisive at this similarity — color
                # cannot veto it, only corroborate/contradict for the log.
                is_duplicate = True
                if not (body_match and muzzle_match):
                    log.warning(
                        f"/register duplicate at score={match.score:.4f} "
                        f"(>= {DUPLICATE_HIGH_CONFIDENCE_THRESHOLD}) despite "
                        f"color mismatch — body_match={body_match} "
                        f"muzzle_match={muzzle_match}; rejecting anyway"
                    )
            else:
                # Ambiguous band: require the more trustworthy signal
                # (muzzle_color) to corroborate. body_color is deliberately
                # NOT required here — see comment above.
                is_duplicate = muzzle_match

            if is_duplicate:
                # tag_no veto: the signals above say "same animal", but if the
                # officer entered a tag that differs from this candidate's
                # stored tag, that is a human-verified signal they are
                # DIFFERENT physical animals (this is precisely the
                # same-breed/same-color goshala case tag_no exists to fix).
                tags_differ = tag_no and stored.tag_no and tag_no != stored.tag_no

                # FOR NOW (explicit instruction, 2026-08-08): only ONE side
                # having a tag isn't enough to PROVE they're the same animal
                # either, and not every animal is guaranteed to be tagged yet
                # during this rollout -- so don't let that partial comparison
                # block a registration. Revisit once every animal reliably
                # carries a tag; today this just means "can't confirm from
                # tags alone" gets the benefit of the doubt.
                one_sided = bool(tag_no) != bool(stored.tag_no)

                if tags_differ or one_sided:
                    log.info(
                        f"/register duplicate candidate cleared by tag_no "
                        f"({'mismatch' if tags_differ else 'one-sided, FOR NOW'}) | "
                        f"score={match.score:.4f} | new_tag={tag_no!r} | "
                        f"stored_tag={stored.tag_no!r} | matched_faiss_id={match.faiss_id}"
                    )
                    continue

                if BYPASS_DUPLICATE_CHECK:
                    log.warning(
                        f"/register bypassing duplicate (BYPASS_DUPLICATE_CHECK) | "
                        f"score={match.score:.4f} | "
                        f"body={new_body} muzzle={new_muzzle} | "
                        f"matched_faiss_id={match.faiss_id}"
                    )
                    continue

                log.info(
                    f"/register 409 duplicate | "
                    f"score={match.score:.4f} | "
                    f"body={new_body} muzzle={new_muzzle} | "
                    f"matched_faiss_id={match.faiss_id}"
                )
                # body/muzzle colour ride along because they are part of WHY
                # this is a duplicate, and the API server records the whole
                # detail against the failure row.
                raise duplicate_animal_error(
                    matched_faiss_id=match.faiss_id,
                    top_score=match.score,
                    body_color=new_body,
                    muzzle_color=new_muzzle,
                )

    # ── Store in FAISS ────────────────────────────────────────────────────
    try:
        faiss_ids = await faiss_index.add_batch(embeddings_np)
    except Exception as e:
        log.error(f"FAISS write failed: {e}")
        raise HTTPException(status_code=500, detail=f"faiss_error: {e}")

    # ── Cache each crop locally, keyed by its faiss_id, for /search's
    #    LightGlue tiebreaker (see pipeline/muzzle_crop_cache.py). Runs AFTER
    #    the FAISS write succeeds, and a cache-write failure is fail-open
    #    (logged, not raised) — registration must not fail over an optional
    #    signal for a feature that isn't the embedding index itself.
    try:
        await asyncio.to_thread(
            lambda: [save_muzzle_crop(fid, crop) for fid, crop in zip(faiss_ids, cropped_images)]
        )
    except Exception as e:
        log.warning(f"muzzle crop cache write failed (non-fatal): {e}")

    # ── Also pre-extract and cache DISK features for the same crops, so
    #    /search's tiebreaker can skip live extraction on the candidate side
    #    entirely (see pipeline/lightglue_verify.py's "latency optimization,
    #    round 2" note and CLAUDE.md). Same fail-open contract; skipped
    #    outright if the verifier never loaded.
    if lightglue_available():
        try:
            await asyncio.to_thread(
                lambda: [
                    save_muzzle_features(fid, **lightglue_extract_features_np(crop))
                    for fid, crop in zip(faiss_ids, cropped_images)
                ]
            )
        except Exception as e:
            log.warning(f"muzzle feature cache write failed (non-fatal): {e}")

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
        face_geometry=FaceGeometry(**face_geometry),
        potential_matches=potential_matches,
        versions=VersionInfo(
            model=MODEL_VERSION,
            faiss=faiss_index.faiss_version_label,
            embedding="v1",
        ),
        registered_at=datetime.now(timezone.utc).isoformat(),
    )


def _resolve_disagreeing_body_colors(body_colors: list[dict]) -> dict:
    """Resolve two front photos that read different body colors.

    This used to be an unconditional 422 ("retake photos"). That was the wrong
    response to the most common cause of disagreement, which is not a bad
    photo at all: the two front shots frame the animal differently, so the
    coat's share of the sampled pixels shifts and a coat sitting near a
    classifier boundary lands on either side of it. The officer cannot fix
    that by retaking — the same two angles produce the same split — so the
    endpoint rejected registrations that no retake would ever repair.

    It is also disproportionate to what the label is worth downstream.
    go-apiserver deliberately does NOT hard-filter search on these labels
    ("classifier confidence is unreliable"), and CLAUDE.md records the body
    thresholds as explicitly uncalibrated. A signal too weak to filter a
    search result should not be strong enough to block a registration.

    The asymmetry it leaves behind is the giveaway: MUZZLE color takes 3
    samples and accepts a MAJORITY, while body color took 2 and demanded
    UNANIMITY. Body was not stricter because it is more reliable — it is
    stricter only because you cannot have a majority out of 2. This restores
    the muzzle rule's spirit for a 2-sample vote.

    The tie-break uses confidence as what body_color.py actually defines it to
    be: the winning label's SHARE of the sampled coat. So a reading at or
    above BODY_COLOR_MAJORITY_CONFIDENCE means "this color covers most of the
    animal" — a substantive claim, not a tuned number.

      - Exactly one reading claims a majority of the coat → take it. The other
        photo saw no color clearly enough to outvote it.
      - Both claim a majority, and disagree → still 422. Two confident,
        contradictory readings is the case a retake genuinely serves (e.g.
        the two front photos are of different animals).
      - Neither claims a majority → still 422. Nothing here is trustworthy
        enough to store.

    Confidence of an accepted reading is HALVED, following the same
    convention average_readings() uses for morphology's PARTIAL status: one of
    two photos failed to support this label, so the result must read as less
    certain than two agreeing photos would.
    """
    confident = [c for c in body_colors if c["confidence"] >= BODY_COLOR_MAJORITY_CONFIDENCE]

    if len(confident) != 1:
        labels = " vs ".join(
            f"{c['label']}({c['confidence']:.2f})" for c in body_colors
        )
        if BYPASS_QUALITY_GATES:
            best = max(body_colors, key=lambda c: c["confidence"])
            log.warning(
                f"body_color: bypassing inconsistency ({labels}), "
                f"accepting best-guess reading {best['label']}({best['confidence']:.2f})"
            )
            result = dict(best)
            result["confidence"] = round(result["confidence"] / 2.0, 4)
            return result
        detail = (
            "body_color_inconsistent: front images disagree and neither is "
            "decisive"
            if not confident
            else "body_color_inconsistent: front images give conflicting "
                 "confident readings"
        )
        raise color_inconsistency_error(
            ErrorCode.BODY_COLOR_INCONSISTENT,
            readings=color_readings("front", body_colors),
            summary=f"{detail} ({labels})",
        )

    winner = dict(confident[0])
    loser = next(c for c in body_colors if c is not confident[0])
    log.warning(
        f"body_color: front photos disagree ({winner['label']} "
        f"{winner['confidence']:.2f} vs {loser['label']} "
        f"{loser['confidence']:.2f}) — accepting the decisive reading"
    )
    winner["confidence"] = round(winner["confidence"] / 2.0, 4)
    winner["reason"] = (
        f"RESOLVED_DISAGREEMENT: other front photo read "
        f"{loser['label']} at {loser['confidence']:.2f}"
    )
    return winner


def _run_registration_pipeline(
    muzzle_bytes: tuple[bytes, ...],
    front_bytes: tuple[bytes, ...],
    model: Any,
    device: torch.device,
    color_extractor: Any,
    morphology_extractor: Any,
) -> tuple[np.ndarray, dict, dict, dict, dict, list[np.ndarray]]:
    """
    Runs the full synchronous CPU/GPU pipeline: quality gates, YOLO crop,
    embedding, color extraction, morphology, and pose-model face geometry.
    Executed entirely inside a single worker thread via asyncio.to_thread —
    nothing in here should ever need to be async itself.

    Returns (embeddings, body_color, muzzle_color, morphology, face_geometry,
    cropped_images).

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
    failures: list[ImageFailure] = []
    cropped_images: list[np.ndarray | None] = [None] * len(muzzle_bytes)
    t_muzzle_gate = time.monotonic()
    for i, mb in enumerate(muzzle_bytes, 1):
        slot = f"muzzle_{i}"
        t_slot = time.monotonic()

        # Decoded unconditionally (not just after quality_check passes) so a
        # BYPASS_QUALITY_GATES fallback always has raw pixels to fall back to,
        # regardless of which check below is the one that would have failed.
        img_bgr = _decode_image(mb, slot)

        status, reason = quality_check(mb)
        if status != "GOOD":
            if BYPASS_QUALITY_GATES:
                log.warning(f"{slot}: bypassing quality failure ({reason}), using raw frame")
                cropped_images[i - 1] = img_bgr
                continue
            failures.append(ImageFailure(
                slot=slot,
                stage="quality",
                error_code=classify_quality_reason(reason),
                reason=reason,
            ))
            continue

        crop, det_status, _det_conf = crop_cattle(img_bgr)
        log.info(f"{slot}: crop_cattle took {_ms_since(t_slot)}ms (status={det_status})")
        if crop is None:
            if BYPASS_QUALITY_GATES:
                log.warning(f"{slot}: bypassing detection failure ({det_status}), using raw frame")
                cropped_images[i - 1] = img_bgr
                continue
            failures.append(ImageFailure(
                slot=slot,
                stage="detection",
                error_code=classify_detection_status(det_status),
                reason=det_status,
            ))
            continue

        crop_status, crop_reason = quality_check_cv2(crop)
        if crop_status != "GOOD":
            if BYPASS_QUALITY_GATES:
                log.warning(f"{slot}_crop: bypassing crop-quality failure ({crop_reason})")
                cropped_images[i - 1] = crop
                continue
            failures.append(ImageFailure(
                slot=slot,
                stage="crop_quality",
                error_code=classify_quality_reason(crop_reason),
                reason=crop_reason,
            ))
            continue

        cropped_images[i - 1] = crop

    log.info(f"muzzle quality/crop gate (3 images): {_ms_since(t_muzzle_gate)}ms")

    if failures:
        raise image_quality_error(failures)

    # ── Embed (batched forward pass) ─────────────────────────────────────
    t_embed = time.monotonic()
    jpg_bytes = [cv2.imencode(".jpg", img)[1].tobytes() for img in cropped_images]
    embeddings = embed_batch(jpg_bytes, model, device)
    log.info(f"embed_batch (3 images): {_ms_since(t_embed)}ms")

    # ── Color extraction with consistency check ──────────────────────────
    #
    # Body (2 front images): BOTH must agree on the same label.
    #   Disagree → 422, ask user to retake.
    #
    # Muzzle (3 crops): majority (≥2/3) must agree on the same label.
    #   Majority found → accept majority label (avg confidence of agreeing images).
    #   All 3 different → 422, ask user to retake.

    t_body_color = time.monotonic()
    body_colors = [
        color_extractor.extract_body(_decode_image(fb, f"front_{i}"))
        for i, fb in enumerate(front_bytes, 1)
    ]
    log.info(f"body_color extraction (2 images): {_ms_since(t_body_color)}ms")
    body_labels = [c["label"] for c in body_colors]

    if body_labels[0] == body_labels[1]:
        # Both agree — pick highest confidence reading
        body_color = max(body_colors, key=lambda c: c["confidence"])
    else:
        body_color = _resolve_disagreeing_body_colors(body_colors)

    t_muzzle_color = time.monotonic()
    muzzle_colors = [color_extractor.extract_muzzle(crop) for crop in cropped_images]
    log.info(f"muzzle_color extraction (3 images): {_ms_since(t_muzzle_color)}ms")
    muzzle_labels = [c["label"] for c in muzzle_colors]

    # Count votes per label
    vote_counts = Counter(muzzle_labels)
    majority_label, majority_count = vote_counts.most_common(1)[0]

    if majority_count < 2:
        # All 3 different — no majority
        if BYPASS_QUALITY_GATES:
            best = max(muzzle_colors, key=lambda c: c["confidence"])
            log.warning(
                f"muzzle_color: bypassing no-majority ({', '.join(muzzle_labels)}), "
                f"accepting best-guess reading {best['label']}({best['confidence']:.2f})"
            )
            muzzle_color = {"label": best["label"], "confidence": round(best["confidence"] / 2.0, 4)}
        else:
            raise color_inconsistency_error(
                ErrorCode.MUZZLE_COLOR_INCONSISTENT,
                readings=color_readings("muzzle", muzzle_colors),
                summary=(
                    f"muzzle_color_inconsistent: no majority among crops "
                    f"({', '.join(muzzle_labels)})"
                ),
            )
    else:
        # Majority found — use avg confidence of agreeing images
        agreeing = [c for c in muzzle_colors if c["label"] == majority_label]
        avg_conf  = sum(c["confidence"] for c in agreeing) / len(agreeing)
        muzzle_color = {"label": majority_label, "confidence": avg_conf}

    # ── Morphology (horn/e ar proportions) — return-only, not a gate ───────
    # Unlike color, there's no majority/consistency check here: this is a
    # continuous, unvalidated heuristic (see pipeline/morphology.py), not a
    # categorical label, so "the 2 photos disagree" isn't a retake-worthy
    # error the way a color mismatch is. Both readings are combined via a
    # confidence-weighted average instead.
    t_morphology = time.monotonic()
    morphology_readings = [
        morphology_extractor.extract(_decode_image(fb, f"front_{i}"))
        for i, fb in enumerate(front_bytes, 1)
    ]
    log.info(f"morphology extraction (2 images): {_ms_since(t_morphology)}ms")
    morphology = average_readings(morphology_readings)

    # ── Face geometry (pose keypoints → inter-eye-normalised proportions) ──
    # Same return-only contract as morphology above: never a gate, never
    # raises. For each front photo: crop_cattle → 8-keypoint pose model →
    # geometry.py, then combine the 2 readings. If POSE_MODEL_PATH wasn't set
    # / didn't load, every reading is NO_FACE and the combined struct just
    # says so — nothing downstream changes. extract_face_geometry does its
    # own crop_cattle internally (like RuleBasedMorphologyExtractor.extract).
    t_pose = time.monotonic()
    geometry_readings = [
        extract_face_geometry(_decode_image(fb, f"front_{i}"))
        for i, fb in enumerate(front_bytes, 1)
    ]
    log.info(f"face geometry extraction (2 images): {_ms_since(t_pose)}ms")
    face_geometry = combine_geometry(geometry_readings)
    # Coverage line, ships WITH the feature rather than as a follow-up: the
    # only way to know how often this signal is even usable in the field is
    # to watch it from the first production registration onward — there is
    # nowhere else this is recorded (not persisted downstream as of this
    # commit; see the go-apiserver /register response). grep on
    # "face_geometry status=" for the OK/PARTIAL/NO_RULER/NO_FACE split, and
    # on "horn_base_ratio=" for how often that specific field is usable.
    log.info(
        f"face_geometry status={face_geometry['status']} "
        f"reason={face_geometry.get('reason')} "
        f"sources_ok={face_geometry['sources_ok']}/{face_geometry['sources_with_face']} "
        f"horn_base_ratio={'present' if face_geometry.get('horn_base_distance_ratio') is not None else 'UNKNOWN'} "
        f"ear_base_ratio={'present' if face_geometry.get('ear_base_span_ratio') is not None else 'UNKNOWN'}"
    )

    log.info(f"_run_registration_pipeline total: {_ms_since(t_muzzle_gate)}ms")
    return embeddings.numpy(), body_color, muzzle_color, morphology, face_geometry, cropped_images


async def _run_lightglue_tiebreaker(
    matches: list[dict],
    query_crop: np.ndarray,
    request_id: str,
    candidate_tags: Optional[dict[int, Optional[str]]] = None,
) -> tuple[bool, Optional[int], Optional[str]]:
    """The /search fusion tiebreaker: additive only, computed only when the
    top-1 embedding score is ambiguous. Never touches `matches` itself or any
    existing SearchResponse field — see CLAUDE.md.

    Pulled out of the /search handler as its own function specifically so it
    can be exercised directly in tests without booting the full FastAPI app
    (which needs a real GodhaarModel checkpoint this function has nothing to
    do with).

    Returns (lightglue_checked, lightglue_num_matches, lightglue_zone) —
    (False, None, None) whenever the tiebreaker didn't run, for any reason
    (disabled, not ambiguous, verifier unavailable, no cached crop, or any
    internal failure — this is a fail-open signal, never allowed to raise).

    Gated on LIGHTGLUE_TIEBREAKER_ENABLED (pipeline/lightglue_verify.py,
    defaults on — the resize cap + feature cache were re-validated against
    real data before this shipped, see CLAUDE.md, "/search fusion
    tiebreaker, round 2"). Kept as an env-var kill switch for a fast disable
    without a redeploy, not as a "not ready yet" gate.

    TEMPORARY (2026-09-05): `candidate_tags` maps faiss_id -> tag_no, passed
    in so this function can log which registered animal each of the raw
    top-2 embedding matches actually belongs to at the moment ambiguity is
    decided. This is the go/no-go check for a suspected root cause: the
    ambiguity gate below compares the top-2 EMBEDDINGS, which can be two
    muzzle photos of the SAME correctly-matched animal (each animal has up
    to 3 registered embeddings) rather than two different animals — in which
    case running LightGlue at all is answering a question nobody asked, and
    a low keypoint count against a mediocre cached crop can demote a verdict
    that was never actually ambiguous at the animal level. go-apiserver's own
    `gap` in decision.go is measured post-aggregation (one score per animal)
    and would NOT show this; this raw-level log is the only way to see it.
    No godhaar_id is available here — inference_server is never sent one
    (see client.go's Candidate struct) — tag_no is the closest identifier
    that's actually on the wire, and is unique per animal (`animals.tag_id`
    is a UNIQUE column), so it's sufficient to tell "same animal" apart from
    "different animal". Remove this parameter and the log line below once
    the hypothesis is confirmed or refuted and the real fix (or a decision
    not to change anything) lands.
    """
    if not LIGHTGLUE_TIEBREAKER_ENABLED:
        return False, None, None

    if not matches:
        return False, None, None

    top1 = matches[0]
    ambiguous = (
        _SEARCH_REVIEW_THRESHOLD_MIRROR <= top1["score"] <= _SEARCH_MATCH_THRESHOLD_MIRROR
        or top1["gap"] < _SEARCH_GAP_THRESHOLD_MIRROR
    )

    # TEMPORARY (2026-09-05, remove with the rest of the candidate_tags
    # plumbing above): log what the top-2 raw embeddings actually are the
    # moment ambiguity is decided, regardless of whether LightGlue itself
    # goes on to run. tag2/faiss_id2 are None when there's only one candidate
    # at all (nothing to be ambiguous against).
    if ambiguous:
        tags = candidate_tags or {}
        top2 = matches[1] if len(matches) > 1 else None
        tag1 = tags.get(top1["faiss_id"])
        tag2 = tags.get(top2["faiss_id"]) if top2 else None
        log.info(
            f"[{request_id}] lightglue ambiguity check | "
            f"top1 faiss_id={top1['faiss_id']} tag_no={tag1!r} score={top1['score']:.4f} | "
            f"top2 faiss_id={top2['faiss_id'] if top2 else None} tag_no={tag2!r} "
            f"score={top2['score'] if top2 else None} | "
            f"gap={top1['gap']:.4f} | "
            f"SAME_ANIMAL={bool(tag1 and top2 and tag1 == tag2)}"
        )
    if not (ambiguous and lightglue_available()):
        return False, None, None

    try:
        # Fast path: candidate's DISK features were pre-extracted at
        # registration time — skip live extraction on that side entirely.
        candidate_features = await asyncio.to_thread(load_cached_muzzle_features, top1["faiss_id"])
        if candidate_features is not None:
            lg_result = await asyncio.to_thread(
                lightglue_verify_with_cached_candidate, query_crop, candidate_features
            )
            num_matches = lg_result["num_matches"]
            return True, num_matches, lightglue_classify_zone(num_matches)

        # Fallback: no cached features (pre-existing registration, or a past
        # write failure) — fall back to the cached crop and extract live on
        # both sides, same as before feature caching existed.
        candidate_crop = await asyncio.to_thread(load_cached_muzzle_crop, top1["faiss_id"])
        if candidate_crop is None:
            log.info(
                f"[{request_id}] lightglue tiebreaker skipped: no cached crop "
                f"or features for top1 faiss_id={top1['faiss_id']} (registered "
                f"before either cache existed, or its registration-time write failed)"
            )
            return False, None, None

        log.info(
            f"[{request_id}] lightglue tiebreaker: feature cache miss for "
            f"faiss_id={top1['faiss_id']}, falling back to live extraction"
        )
        lg_result = await asyncio.to_thread(lightglue_verify, query_crop, candidate_crop)
        num_matches = lg_result["num_matches"]
        return True, num_matches, lightglue_classify_zone(num_matches)
    except Exception as e:
        # Fail-open: an inert additive signal must never break a search the
        # embedding pipeline already answered.
        log.warning(f"[{request_id}] lightglue tiebreaker failed (non-fatal): {e}")
        return False, None, None


@app.post("/search", response_model=SearchResponse)
async def search(
    muzzle: UploadFile = File(...),
    front: UploadFile = File(...),
    top_k: int = Form(5),
    candidate_json: str = Form(..., alias="candidates"),
    tag_no: str | None = Form(None),
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

    log.info(f"[{request_id}] /search top_k={top_k} candidates={len(candidate_ids)} tag_no={tag_no!r}")

    # ── Real async I/O: read uploads concurrently ──────────────────────────
    muzzle_bytes, front_bytes = await asyncio.gather(
        muzzle.read(),
        front.read()
    )

    # ── CPU/GPU pipeline: one thread-offload seam ───────────────────────────
    t_embed_start = time.monotonic()
    emb_np, muzzle_color, body_color, morphology, query_crop = await asyncio.to_thread(
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
    faiss_top_k = len(candidate_ids) if tag_no else top_k
    matches = await faiss_index.restricted_search(emb_np, candidate_ids=candidate_ids, top_k=faiss_top_k)
    t_faiss = _ms_since(t_faiss_start)

    if tag_no:
        tag_match = None
        remaining = []
        for m in matches:
            stored_tag = candidate_lookup[m["faiss_id"]].tag_no
            if stored_tag and stored_tag == tag_no:
                tag_match = m
            elif stored_tag and stored_tag != tag_no:
                log.info(
                    f"[{request_id}] /search dropping candidate proven different by "
                    f"tag_no | faiss_id={m['faiss_id']} score={m['score']:.4f} "
                    f"query_tag={tag_no!r} stored_tag={stored_tag!r}"
                )
                continue
            else:
                remaining.append(m)
        ordered = ([tag_match] if tag_match else []) + remaining
        if tag_match:
            log.info(
                f"[{request_id}] /search promoted tag_no match to rank 1 | "
                f"faiss_id={tag_match['faiss_id']} score={tag_match['score']:.4f}"
            )
        # Ranks/gaps were computed against the original FAISS order; both are
        # meaningless after reordering/dropping entries, so recompute them
        # the same way faiss_index._restricted_search_sync does.
        matches = ordered[:top_k]
        for idx, m in enumerate(matches):
            m["rank"] = idx + 1
            m["gap"] = (
                m["score"] - matches[idx + 1]["score"]
                if idx + 1 < len(matches)
                else 0.0
            )

    # ── LightGlue fusion tiebreaker — additive only; see _run_lightglue_tiebreaker.
    t_lightglue_start = time.monotonic()
    lightglue_checked, lightglue_num_matches, lightglue_zone = await _run_lightglue_tiebreaker(
        matches, query_crop, request_id,
        candidate_tags={fid: c.tag_no for fid, c in candidate_lookup.items()},
    )
    t_lightglue = _ms_since(t_lightglue_start)

    t_total = _ms_since(t_start)
    log.info(
        f"[{request_id}] /search done | matches={len(matches)} | "
        f"{t_total}ms (embed={t_embed} faiss={t_faiss} lightglue={t_lightglue}) | "
        f"top1_faiss_id={matches[0]['faiss_id'] if matches else 'none'} "
        f"score={matches[0]['score'] if matches else 0} | "
        f"lightglue_checked={lightglue_checked} lightglue_zone={lightglue_zone}"
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
        lightglue_checked=lightglue_checked,
        lightglue_num_matches=lightglue_num_matches,
        lightglue_zone=lightglue_zone,
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
) -> tuple[np.ndarray, dict, dict, dict, np.ndarray]:
    """
    Synchronous CPU/GPU pipeline for /search: quality gate, crop, embed,
    color extraction. Runs entirely inside asyncio.to_thread.

    Also returns the query's own muzzle `crop` (not just its embedding) —
    needed by the /search route handler for the LightGlue fusion tiebreaker,
    which compares raw pixels, not embeddings.
    """
    q_status, q_reason = quality_check(muzzle_bytes)
    if q_status != "GOOD":
        log.error(f"Quality status: {q_status}")
        log.error(f"Quality reason: {q_reason}")
        raise image_quality_error([ImageFailure(
            slot="muzzle",
            stage="quality",
            error_code=classify_quality_reason(q_reason),
            reason=q_reason,
        )])

    img_bgr = _decode_image(muzzle_bytes, "muzzle")
    crop, det_status, _det_conf = crop_cattle(img_bgr)
    if crop is None:
        log.error(f"Crop: {crop} | Det Status: {det_status} | Det conf: {_det_conf}")
        raise image_quality_error([ImageFailure(
            slot="muzzle",
            stage="detection",
            error_code=classify_detection_status(det_status),
            reason=det_status,
        )])

    crop_status, crop_reason = quality_check_cv2(crop)
    if crop_status != "GOOD":
        log.error(f"Crop Status: {crop_status} | Crop reason: {crop_reason}")
        raise image_quality_error([ImageFailure(
            slot="muzzle",
            stage="crop_quality",
            error_code=classify_quality_reason(crop_reason),
            reason=crop_reason,
        )])

    crop_bytes = cv2.imencode(".jpg", crop)[1].tobytes()
    embedding = embed_batch([crop_bytes], model, device)  # (1, 256)
    emb_np = embedding.squeeze(0).numpy()  # (256,)

    muzzle_color = color_extractor.extract_muzzle(crop)
    front_img = _decode_image(front_bytes, "front")
    body_color = color_extractor.extract_body(front_img)
    morphology = morphology_extractor.extract(front_img)

    return emb_np, muzzle_color, body_color, morphology, crop


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
        pose_model_loaded=pose_available(),
    )
