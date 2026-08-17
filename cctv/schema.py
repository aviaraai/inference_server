"""
cctv/schema.py — Pydantic response models for the CCTV video-analytics API.

Mirrors the convention of the root `schema.py` (register/search models):
response shapes only, request params use Form()/File() directly in route
signatures (see `cctv/routes.py`).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class JobCreate(BaseModel):
    preset: str = "fast"
    location_tag: Optional[str] = None
    model_path: Optional[str] = None
    img_size: Optional[int] = None
    confidence: Optional[float] = None
    vid_stride: Optional[int] = None
    enable_analytics: bool = True


class JobStatus(BaseModel):
    job_id: str
    status: str                # queued | processing | done | failed
    progress: float = 0.0      # 0-1
    frames_processed: int = 0
    total_frames: int = 0
    cattle_so_far: int = 0
    error: Optional[str] = None


class JobResult(BaseModel):
    job_id: str
    final_cattle_count: int = Field(
        ..., description="The single best count for this clip: max_cattle_in_frame for a static/fixed camera, unique_tracked_cattle for a clip classified as panning (see count_method_used). Prefer this field when only one number can be shown."
    )
    count_method: str
    count_method_used: str = Field(
        "peak_in_frame",
        description="Which figure final_cattle_count above actually is: \"peak_in_frame\" (max_cattle_in_frame) or \"tracking_estimate\" (unique_tracked_cattle), decided by automatic panning detection (cctv/panning.py). Additive/diagnostic field — not surfaced in the dashboard UI today.",
    )
    max_cattle_in_frame: int = Field(
        ..., description="\"Cattle in view (peak)\" — highest count visible in any single frame. Accurate for a mostly-static camera; undercounts a camera panning across a large herd."
    )
    unique_tracked_cattle: int = Field(
        ..., description="\"Cattle observed (tracking)\" — distinct tracked IDs seen for at least min_frames_visible frames. Accurate for a panning shot; can overcount a static herd via tracker ID churn."
    )
    average_confidence: float
    total_detections: int
    throughput_fps: float
    processing_seconds: float
    classify_seconds: float = Field(
        0.0, description="How long the separate FAST-preset classification pass took (see count_method_used). 0.0 means this job predates the decoupled classify/count pipeline and used a single pass."
    )
    frames_processed: int
    frames_with_cattle: int
    video_url: str = Field(
        ..., description="Path (relative to this server) to GET for the annotated video — bounding boxes, stable Cow IDs, and confidence drawn on every frame. H.264, browser-playable, range-request seekable."
    )
    output_report: str
    output_csv: str


class AnalyticsSummary(BaseModel):
    job_id: str
    total_cattle: int = Field(
        ..., description="\"Cattle in view (peak)\" — highest count visible in any single frame."
    )
    unique_tracked_cattle: int = Field(
        ..., description="\"Cattle observed (tracking)\" — distinct tracked IDs seen for at least min_frames_visible frames."
    )
    avg_herd_speed: float
    isolated_cattle: list[int]
    activity_breakdown: dict[str, int]
    density_grid: list[list[float]]
    per_cow: list[dict]
    summary_text: str


class TrendPoint(BaseModel):
    date: str
    count: int = Field(..., description="\"Cattle in view (peak)\" for this session.")
    unique_tracked_cattle: Optional[int] = Field(
        None, description="\"Cattle observed (tracking)\" for this session. None for sessions recorded before this field existed."
    )
    avg_speed: float
    isolated_count: int
    location: Optional[str]
    job_id: str


class SessionInfo(BaseModel):
    job_id: str
    created_at: str
    location_tag: Optional[str]
    video_filename: Optional[str]
    final_cattle_count: int = Field(
        ..., description="The single best count for this session — peak-in-frame for a static camera, tracked-ID count for a panning one. See count_method_used."
    )
    unique_tracked_cattle: Optional[int] = Field(
        None, description="\"Cattle observed (tracking)\" for this session. None for sessions recorded before this field existed."
    )
    count_method_used: Optional[str] = Field(
        None, description="Which figure final_cattle_count is: \"peak_in_frame\" or \"tracking_estimate\". None for sessions recorded before automatic panning detection existed."
    )
    avg_herd_speed: Optional[float]
    processing_sec: Optional[float]
    video_url: str = Field(
        ..., description="Path (relative to this server) to GET this session's annotated video — bounding boxes and all, same as JobResult.video_url."
    )


class CctvHealthResponse(BaseModel):
    status: str = "ok"
    service: str = "cctv"
    active_jobs: int = 0
    total_sessions: int = 0
