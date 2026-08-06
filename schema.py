"""
schema.py — Pydantic response models for the inference server API.

Request parameters use Form() + File() directly in route signatures.
These models define the response shape only.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# ── Errors ────────────────────────────────────────────────────────────────────
#
# Every deliberate rejection this server makes is reported as
#
#     {"error_code": "<ErrorCode>", "detail": {...}}
#
# The API server switches on error_code alone and never parses the prose, so a
# message can be reworded here without breaking a caller. An error_code it does
# not recognise is treated as a version mismatch and reported to the user as a
# service fault, never guessed at — so codes can be added on either side in
# either order, but an existing code must not change meaning.
#
# Failures that are the API server's own fault — wrong image count, malformed
# candidates JSON — deliberately keep FastAPI's plain {"detail": ...} shape with
# no error_code. That is what lets the caller tell "the photos are unusable"
# apart from "these two services disagree about the contract", which are the
# same 422 otherwise.


class ErrorCode(str, Enum):
    """Machine-readable rejection reasons. Values are the wire contract."""

    # /register only — the muzzle matches an animal already in the index.
    DUPLICATE_ANIMAL = "DUPLICATE_ANIMAL"

    # No cattle found in the frame by the detector.
    NO_ANIMAL_DETECTED = "NO_ANIMAL_DETECTED"

    # Single-image quality gate failures, split by cause because each one
    # implies different advice to whoever is holding the phone.
    IMAGE_TOO_BLURRY = "IMAGE_TOO_BLURRY"
    IMAGE_BAD_EXPOSURE = "IMAGE_BAD_EXPOSURE"
    IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL"
    IMAGE_UNREADABLE = "IMAGE_UNREADABLE"

    # Cross-image disagreement: the photos are individually fine but cannot
    # describe one animal.
    BODY_COLOR_INCONSISTENT = "BODY_COLOR_INCONSISTENT"
    MUZZLE_COLOR_INCONSISTENT = "MUZZLE_COLOR_INCONSISTENT"

    # Umbrella for a set of image failures with more than one distinct cause.
    POOR_IMAGE_QUALITY = "POOR_IMAGE_QUALITY"


class ImageFailure(BaseModel):
    """One image's rejection, so the app can point at the photo to retake
    rather than making the user guess which of the five was bad."""

    slot: str = Field(..., description="Which upload failed, e.g. 'muzzle_1', 'front_2'")
    stage: str = Field(..., description="quality | detection | crop_quality")
    error_code: ErrorCode = Field(..., description="This image's specific failure")
    reason: str = Field(..., description="Raw pipeline reason, for developers — not for display")


class ImageQualityDetail(BaseModel):
    """detail for any image-quality rejection."""

    message: str = Field(..., description="Human-readable summary, for logs — not for display")
    failures: list[ImageFailure] = Field(default_factory=list)


class ColorReading(BaseModel):
    slot: str
    label: str
    confidence: float


class ColorInconsistencyDetail(BaseModel):
    """detail for BODY_COLOR_INCONSISTENT / MUZZLE_COLOR_INCONSISTENT."""

    message: str
    readings: list[ColorReading] = Field(default_factory=list)


class DuplicateDetail(BaseModel):
    """detail for DUPLICATE_ANIMAL.

    matched_faiss_id is the whole point of this payload: the API server is the
    only side that knows which animal a faiss_id belongs to, so it maps this
    back to a godhaar_id and tells the user which registration this duplicates.
    """

    matched_faiss_id: int = Field(..., description="FAISS id of the animal this duplicates")
    top_score: float = Field(..., description="Cosine similarity to that animal")
    body_color: Optional[str] = Field(None, description="Body colour extracted from the new photos")
    muzzle_color: Optional[str] = Field(None, description="Muzzle colour extracted from the new photos")


class ErrorResponse(BaseModel):
    """The body of every deliberate rejection. Documented so it shows up in
    the OpenAPI schema alongside the success responses."""

    error_code: ErrorCode
    detail: dict = Field(default_factory=dict)


# ── Shared ────────────────────────────────────────────────────────────────────

class VersionInfo(BaseModel):
    model: str = Field(..., description="GodhaarModel version identifier")
    faiss: Optional[str] = Field(None, description="FAISS index date label")
    embedding: Optional[str] = Field(None, description="Embedding contract version")


class ColorResult(BaseModel):
    label: str = Field(..., description="Color enum label (e.g. BLACK, PINK, UNKNOWN)")
    confidence: float = Field(..., description="Classifier confidence 0.0–1.0")


class ExtractedColors(BaseModel):
    body: ColorResult
    muzzle: ColorResult


HORN_SHAPE_DESCRIPTION = (
    "One of pipeline.morphology.HornShape (STRAIGHT, CURVED, UNKNOWN), or "
    "None if no horn is confirmed visible in the photo(s). None does NOT "
    "mean 'confirmed hornless' — a genuinely polled animal and a "
    "backward/occluded horn look identical to this heuristic, and any "
    "non-reading status (bad image, no animal detected, no clear "
    "silhouette, or — register only — the 2 front photos disagreeing) "
    "also collapses to None here. See pipeline/morphology.py."
)


# ── Candidate input (register duplicate-check AND search) ─────────────────────

class CandidateInfo(BaseModel):
    """A nearby cattle candidate sent by the API server, for duplicate
    checking (/register) or for echoing stored data back on each match
    (/search). `horn_shape` is optional so a caller that doesn't have it
    yet can omit it — it defaults to None, not a fabricated value.
    """
    faiss_id: int = Field(..., description="FAISS integer ID of the stored embedding")
    body_color: str = Field(..., description="Stored body color label (e.g. BLACK)")
    muzzle_color: str = Field(..., description="Stored muzzle color label (e.g. PINK)")
    horn_shape: Optional[str] = Field(None, description=HORN_SHAPE_DESCRIPTION)


# ── Match ─────────────────────────────────────────────────────────────────────

class MatchCandidate(BaseModel):
    faiss_id: int = Field(..., description="FAISS integer ID for this embedding")
    score: float = Field(..., description="Cosine similarity score")
    rank: int = Field(..., description="1-indexed rank among returned candidates")
    gap: float = Field(..., description="Score delta to the next candidate (0.0 for last)")
    body_color: Optional[str] = Field(
        None, description="This candidate's stored body color, echoed back from the request's candidates list, if provided."
    )
    muzzle_color: Optional[str] = Field(
        None, description="This candidate's stored muzzle color, echoed back from the request's candidates list, if provided."
    )
    horn_shape: Optional[str] = Field(
        None, description="This candidate's stored horn shape, echoed back so it can be compared against the query's own `horn_shape` field. " + HORN_SHAPE_DESCRIPTION
    )


# ── Register ──────────────────────────────────────────────────────────────────
class RegisterResponse(BaseModel):
    status: str = Field("success", description="Registration status")
    embedding_ids: list[int] = Field(
        ..., description="FAISS integer IDs for the stored embeddings"
    )
    extracted_colors: ExtractedColors
    horn_shape: Optional[str] = Field(None, description=HORN_SHAPE_DESCRIPTION)
    potential_matches: list[MatchCandidate] = Field(
        default_factory=list,
        description="Top matches against candidate_ids (empty if no candidates provided)",
    )
    versions: VersionInfo
    registered_at: str = Field(..., description="ISO 8601 UTC timestamp")


# ── Search ────────────────────────────────────────────────────────────────────

class SearchResponse(BaseModel):
    request_id: str = Field(..., description="UUID for request tracing")
    query_colors: ExtractedColors
    horn_shape: Optional[str] = Field(None, description=HORN_SHAPE_DESCRIPTION)
    top_matches: list[MatchCandidate]
    versions: VersionInfo


# ── Health ────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = Field("ok")
    model_loaded: bool
    faiss_size: int = Field(..., description="Total vectors in FAISS index")
    gpu_available: bool
    model_version: str
    color_extractor_available: bool
    morphology_extractor_available: bool = Field(
        True, description="Always true — the rule-based morphology extractor has no external dependency to fail."
    )
