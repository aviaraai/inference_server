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
    horn_length_ratio: float = Field(
        ...,
        description=(
            "Horn/head extent above the head band, ÷ crop width. Scale-invariant "
            "proxy, NOT a real-world measurement — see pipeline/morphology.py."
        ),
    )
    ear_span_ratio: float = Field(
        ...,
        description=(
            "Left-right silhouette extent within the head band, ÷ crop width. "
            "Scale-invariant proxy, NOT a real-world measurement — see pipeline/morphology.py."
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
            "OK, INVALID_IMAGE, NO_ANIMAL_DETECTED, NO_CLEAR_SILHOUETTE, or "
            "PARTIAL (register only — some but not all front photos produced "
            "a reading). Only OK/PARTIAL carry a real (possibly blended) "
            "reading; anything else means the ratios are the zero default "
            "and MUST NOT be read as data. A non-OK status does NOT mean "
            "'no horns' — it means 'no horn confirmed visible in this "
            "photo,' which is also what a genuinely hornless/polled animal "
            "or a backward-facing horn look like. See pipeline/morphology.py."
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
    horn_length_ratio: Optional[float] = Field(
        None, description="This candidate's stored horn/head ratio, if known — see pipeline/morphology.py."
    )
    ear_span_ratio: Optional[float] = Field(
        None, description="This candidate's stored ear-span ratio, if known — see pipeline/morphology.py."
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
    horn_length_ratio: Optional[float] = Field(
        None, description="This candidate's stored horn/head ratio, echoed back so it can be compared against the query's own `morphology` field."
    )
    ear_span_ratio: Optional[float] = Field(
        None, description="This candidate's stored ear-span ratio, echoed back so it can be compared against the query's own `morphology` field."
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
