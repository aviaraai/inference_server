"""
schema.py — Pydantic response models for the inference server API.

Request parameters use Form() + File() directly in route signatures.
These models define the response shape only.
"""

from typing import Optional

from pydantic import BaseModel, Field

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
    # NOT a price. This animal's tag_no, piggybacked on the `cost` field name
    # end to end (app -> go-apiserver -> here) -- see the `cost` parameter on
    # /register in main.py. Optional so a candidate registered before tag_no
    # existed can still be sent without one; the duplicate-check veto below
    # only fires when BOTH sides have a tag to compare.
    cost: Optional[str] = Field(None, description="Stored tag_no, not a price")


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
