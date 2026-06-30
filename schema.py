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
    method: Optional[str] = Field(None, description="Classification method used")


class ExtractedColors(BaseModel):
    body: ColorResult
    muzzle: ColorResult


class LatencyMs(BaseModel):
    total: int = Field(..., description="Total request latency in ms")
    crop: Optional[int] = Field(None, description="YOLO crop time in ms")
    embed: Optional[int] = Field(None, description="GodhaarModel inference time in ms")
    color: Optional[int] = Field(None, description="Color extraction time in ms")
    faiss: Optional[int] = Field(None, description="FAISS search time in ms")


# ── Register ──────────────────────────────────────────────────────────────────

class RegisterResponse(BaseModel):
    status: str = Field("success", description="Registration status")
    cattle_id: str
    embedding_ids: list[int] = Field(..., description="FAISS integer IDs for the stored embeddings")
    extracted_colors: ExtractedColors
    versions: VersionInfo
    registered_at: str = Field(..., description="ISO 8601 UTC timestamp")
    latency_ms: LatencyMs


# ── Search ────────────────────────────────────────────────────────────────────

class MatchCandidate(BaseModel):
    rank: int
    cattle_id: str
    score: float = Field(..., description="Cosine similarity score")
    gap: Optional[float] = Field(None, description="Score gap to next candidate (ML metric)")


class SearchResponse(BaseModel):
    request_id: str = Field(..., description="UUID for request tracing")
    query_colors: ExtractedColors
    top_matches: list[MatchCandidate]
    versions: VersionInfo
    latency_ms: LatencyMs


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field("ok")
    model_loaded: bool
    faiss_size: int = Field(..., description="Total vectors in FAISS index")
    id_store_size: int = Field(..., description="Total entries in ID store")
    gpu_available: bool
    model_version: str
    color_extractor_available: bool
