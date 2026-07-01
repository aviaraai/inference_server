"""
schema.py — Pydantic response models for the inference server API.

Request parameters use Form() + File() directly in route signatures.
These models define the response shape only.
"""

from typing import Optional

from fastapi import UploadFile
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


# ── Register ──────────────────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    muzzle_1: UploadFile
    muzzle_2: UploadFile
    muzzle_3: UploadFile
    front_1: UploadFile
    front_2: UploadFile


class RegisterResponse(BaseModel):
    status: str = Field("success", description="Registration status")
    embedding_ids: list[int] = Field(
        ..., description="FAISS integer IDs for the stored embeddings"
    )
    extracted_colors: ExtractedColors
    versions: VersionInfo
    registered_at: str = Field(..., description="ISO 8601 UTC timestamp")


# ── Search ────────────────────────────────────────────────────────────────────


class MatchCandidate(BaseModel):
    rank: int
    cattle_id: str
    score: float = Field(..., description="Cosine similarity score")
    gap: Optional[float] = Field(
        None, description="Score gap to next candidate (ML metric)"
    )


class SearchResponse(BaseModel):
    request_id: str = Field(..., description="UUID for request tracing")
    query_colors: ExtractedColors
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
