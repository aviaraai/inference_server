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


class MorphologyResult(BaseModel):
    has_horns: Optional[bool] = Field(
        None,
        description=(
            "True/False if a reading was produced, else None. False means "
            "'no horn confirmed visible in this photo' — NOT 'confirmed "
            "hornless' (a genuinely polled animal and a backward/occluded "
            "horn look identical to this heuristic). See pipeline/morphology.py."
        ),
    )
    horn_shape: Optional[str] = Field(
        None,
        description=(
            "One of pipeline.morphology.HornShape (STRAIGHT, CURVED, UNKNOWN) "
            "when has_horns=True. Null whenever has_horns is False or None — "
            "there is no separate 'NONE' shape value, since has_horns=False "
            "already says there's no horn; repeating that as a shape would "
            "just be the same fact twice."
        ),
    )
    confidence: float = Field(
        ...,
        description=(
            "Heuristic confidence 0.0-0.6 (capped). v1 rule-based estimate, not "
            "validated against labeled data — see pipeline/morphology.py."
        ),
    )
    status: str = Field(
        ...,
        description=(
            "OK, INVALID_IMAGE, NO_ANIMAL_DETECTED, NO_CLEAR_SILHOUETTE, "
            "INCONSISTENT (register only — the 2 front photos disagreed), or "
            "PARTIAL (register only — some but not all front photos produced "
            "a reading). Only OK/PARTIAL carry a real reading; anything else "
            "means has_horns/horn_shape are null and MUST NOT be read as "
            "data. See pipeline/morphology.py."
        ),
    )
    reason: str = Field(
        "",
        description="Human-readable detail for `status` when it isn't OK (empty string otherwise).",
    )


# ── Candidate input (register duplicate-check AND search) ─────────────────────

class CandidateInfo(BaseModel):
    """A nearby cattle candidate sent by the API server, for duplicate
    checking (/register) or for echoing stored data back on each match
    (/search). Morphology fields are optional so a caller that doesn't
    have them yet (nothing persists horn/ear data today) can omit them —
    they default to "unknown," not a fabricated 0.
    """
    faiss_id: int = Field(..., description="FAISS integer ID of the stored embedding")
    body_color: str = Field(..., description="Stored body color label (e.g. BLACK)")
    muzzle_color: str = Field(..., description="Stored muzzle color label (e.g. PINK)")
    has_horns: Optional[bool] = Field(
        None, description="This candidate's stored horn presence, if known — see pipeline/morphology.py."
    )
    horn_shape: Optional[str] = Field(
        None, description="This candidate's stored horn shape (pipeline.morphology.HornShape), if known."
    )
    morphology_confidence: Optional[float] = Field(
        None, description="Confidence of this candidate's stored morphology reading, if known."
    )


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
    has_horns: Optional[bool] = Field(
        None, description="This candidate's stored horn presence, echoed back so it can be compared against the query's own `morphology` field."
    )
    horn_shape: Optional[str] = Field(
        None, description="This candidate's stored horn shape, echoed back so it can be compared against the query's own `morphology` field."
    )


# ── Register ──────────────────────────────────────────────────────────────────
class RegisterResponse(BaseModel):
    status: str = Field("success", description="Registration status")
    embedding_ids: list[int] = Field(
        ..., description="FAISS integer IDs for the stored embeddings"
    )
    extracted_colors: ExtractedColors
    morphology: MorphologyResult
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
    morphology: MorphologyResult
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
