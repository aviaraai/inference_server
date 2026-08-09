"""
cctv/routes.py — REST endpoints for the CCTV video-analytics model.

Third model in this server, alongside `pipeline/` (detection) and
`godhaar/` (identification). Wired into `main.py` via
`app.include_router(cctv_router)`.

Endpoints
---------
GET    /cctv/health           Liveness check for this model
POST   /cctv/analyze          Upload video → start background job
GET    /cctv/jobs/{id}/status  Poll progress
GET    /cctv/jobs/{id}/result  Final pipeline result
GET    /cctv/jobs/{id}/analytics  Analytics (movement, density, isolation)
GET    /cctv/jobs/{id}/video   Download annotated video
GET    /cctv/history           List past sessions
GET    /cctv/trends            Cross-video trend data
DELETE /cctv/jobs/{id}         Remove a job's outputs
"""

from __future__ import annotations

import shutil
import threading
import traceback
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from cctv import database as db
from cctv.analytics import VideoAnalytics
from cctv.config import RUNS_DIR, UPLOADS_DIR, Preset, make_config
from cctv.pipeline import run_pipeline
from cctv.schema import (
    AnalyticsSummary,
    CctvHealthResponse,
    JobResult,
    JobStatus,
    SessionInfo,
    TrendPoint,
)

router = APIRouter(prefix="/cctv", tags=["cctv"])

# ── in-memory job tracker ─────────────────────────────────────────
# Same caveat as the original prototype: production traffic should move
# this to Redis / Celery. For a single-server deployment this is fine.

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _set_job(job_id: str, **fields):
    with _lock:
        if job_id not in _jobs:
            _jobs[job_id] = {"status": "queued", "progress": 0}
        _jobs[job_id].update(fields)


def _get_job(job_id: str) -> dict | None:
    with _lock:
        return _jobs.get(job_id, {}).copy() if job_id in _jobs else None


# ── background worker ─────────────────────────────────────────────

def _run_job(job_id: str, video_path: Path, cfg, location_tag: str | None, filename: str | None):
    try:
        _set_job(job_id, status="processing")

        # analytics accumulator
        va = VideoAnalytics(
            frame_w=0, frame_h=0,  # set after first frame
            grid_cells=cfg.density_grid_cells,
            isolation_multiplier=cfg.isolation_multiplier,
            min_frames_visible=cfg.min_frames_visible,
        )
        first_frame_seen = False

        def on_frame(fr):
            nonlocal first_frame_seen
            if not first_frame_seen:
                va.frame_w = fr.annotated_frame.shape[1]
                va.frame_h = fr.annotated_frame.shape[0]
                first_frame_seen = True

            if cfg.enable_analytics:
                va.ingest(fr)

            _set_job(
                job_id,
                frames_processed=fr.processed_frame_idx,
                cattle_so_far=len(set(fr.stable_ids)),
            )

        summary = run_pipeline(video_path, cfg, job_id=job_id, on_frame=on_frame)

        # compute analytics
        analytics_result = va.compute() if cfg.enable_analytics else None

        # persist to database
        if analytics_result:
            db.save_session(summary, analytics_result, location_tag, filename)

        _set_job(
            job_id,
            status="done",
            progress=1.0,
            summary=summary,
            analytics=analytics_result,
            frames_processed=summary.frames_processed,
            total_frames=summary.total_frames,
            cattle_so_far=summary.final_cattle_count,
        )

    except Exception as exc:
        _set_job(job_id, status="failed", error=str(exc))
        traceback.print_exc()
    finally:
        # cleanup the raw upload — only the annotated output under RUNS_DIR
        # needs to stick around for later retrieval.
        try:
            video_path.unlink(missing_ok=True)
        except Exception:
            pass


# ── endpoints ─────────────────────────────────────────────────────

@router.get("/health", response_model=CctvHealthResponse)
async def cctv_health():
    with _lock:
        active = sum(1 for j in _jobs.values() if j.get("status") in ("queued", "processing"))
    try:
        total_sessions = len(db.list_sessions(limit=10_000))
    except Exception:
        total_sessions = 0
    return CctvHealthResponse(active_jobs=active, total_sessions=total_sessions)


@router.post("/analyze", response_model=JobStatus)
async def analyze_video(
    video: UploadFile = File(...),
    preset: str = Form("fast"),
    location_tag: Optional[str] = Form(None),
    enable_analytics: bool = Form(True),
    img_size: Optional[int] = Form(None),
    confidence: Optional[float] = Form(None),
    nms_iou: Optional[float] = Form(None),
    vid_stride: Optional[int] = Form(None),
):
    """Upload a video and start processing in the background."""
    job_id = uuid.uuid4().hex[:12]

    # save upload
    upload_path = UPLOADS_DIR / f"{job_id}_{video.filename}"
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # build config
    overrides: dict = {"enable_analytics": enable_analytics}
    if img_size:
        overrides["img_size"] = img_size
    if confidence:
        overrides["confidence"] = confidence
    if nms_iou:
        overrides["nms_iou"] = nms_iou
    if vid_stride:
        overrides["vid_stride"] = vid_stride

    try:
        preset_enum = Preset(preset)
    except ValueError:
        preset_enum = Preset.FAST

    cfg = make_config(preset_enum, **overrides)

    _set_job(job_id, status="queued", progress=0, total_frames=0)

    # launch in background thread
    t = threading.Thread(
        target=_run_job,
        args=(job_id, upload_path, cfg, location_tag, video.filename),
        daemon=True,
    )
    t.start()

    return JobStatus(job_id=job_id, status="queued")


@router.get("/jobs/{job_id}/status", response_model=JobStatus)
async def get_job_status(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JobStatus(
        job_id=job_id,
        status=job.get("status", "unknown"),
        progress=job.get("progress", 0),
        frames_processed=job.get("frames_processed", 0),
        total_frames=job.get("total_frames", 0),
        cattle_so_far=job.get("cattle_so_far", 0),
        error=job.get("error"),
    )


@router.get("/jobs/{job_id}/result", response_model=JobResult)
async def get_job_result(job_id: str):
    job = _get_job(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "Job not done or not found")

    s = job["summary"]
    return JobResult(
        job_id=s.job_id,
        # Primary displayed count is the peak simultaneous-in-frame count,
        # not the unique-tracked-ID count — visually verifiable against the
        # video, unlike unique_tracked_cattle which is sensitive to tracker
        # ID churn. See unique_tracked_cattle below for the tracking-based
        # figure.
        final_cattle_count=s.max_cattle_in_frame,
        count_method=s.count_method,
        unique_tracked_cattle=s.unique_tracked_cattle,
        max_cattle_in_frame=s.max_cattle_in_frame,
        average_confidence=s.average_confidence,
        total_detections=s.total_detections,
        throughput_fps=s.throughput_fps,
        processing_seconds=s.processing_seconds,
        frames_processed=s.frames_processed,
        frames_with_cattle=s.frames_with_cattle,
        video_url=f"/cctv/jobs/{job_id}/video",
        output_report=s.output_report,
        output_csv=s.output_csv,
    )


@router.get("/jobs/{job_id}/analytics", response_model=AnalyticsSummary)
async def get_job_analytics(job_id: str):
    job = _get_job(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "Job not done or not found")

    a = job.get("analytics")
    if not a:
        raise HTTPException(404, "Analytics not available for this job")

    return AnalyticsSummary(
        job_id=job_id,
        total_cattle=a.total_cattle,
        unique_tracked_cattle=a.unique_tracked_cattle,
        avg_herd_speed=a.avg_herd_speed,
        isolated_cattle=a.isolated_cattle,
        activity_breakdown=a.activity_breakdown,
        density_grid=a.density_grid,
        per_cow=[
            {
                "stable_id": c.stable_id,
                "frames_visible": c.frames_visible,
                "total_distance_px": c.total_distance_px,
                "avg_speed": c.avg_speed_px_per_frame,
                "max_speed": c.max_speed_px_per_frame,
                "avg_bbox_area": c.avg_bbox_area,
                "is_isolated": c.is_isolated,
                "activity": c.activity,
                "dwell_zones": c.dwell_zones,
            }
            for c in a.per_cow
        ],
        summary_text=a.summary_text,
    )


@router.get("/jobs/{job_id}/video")
async def download_video(job_id: str):
    job = _get_job(job_id)
    if job and job.get("status") == "done":
        video_path = job["summary"].output_video
    else:
        # Not in the in-memory job tracker — either the server restarted
        # since this job ran, or it's an older session. _jobs is never
        # persisted, but the sessions table and the on-disk file both
        # survive a restart, so fall back to those rather than 404ing on
        # every session that isn't the most recent server lifetime's.
        row = db.get_session(job_id)
        if not row:
            raise HTTPException(404, "Job not done or not found")
        video_path = row["output_video"]

    path = Path(video_path)
    if not path.exists():
        raise HTTPException(404, "Video file missing")
    return FileResponse(path, media_type="video/mp4", filename=f"cctv_{job_id}.mp4")


@router.get("/history", response_model=list[SessionInfo])
async def list_history(
    location: Optional[str] = None,
    limit: int = 50,
):
    rows = db.list_sessions(location_tag=location, limit=limit)
    return [
        SessionInfo(
            job_id=r["job_id"],
            created_at=r["created_at"],
            location_tag=r.get("location_tag"),
            video_filename=r.get("video_filename"),
            final_cattle_count=r["final_cattle_count"],
            unique_tracked_cattle=r.get("unique_tracked_cattle"),
            avg_herd_speed=r.get("avg_herd_speed"),
            processing_sec=r.get("processing_sec"),
            video_url=f"/cctv/jobs/{r['job_id']}/video",
        )
        for r in rows
    ]


@router.get("/trends", response_model=list[TrendPoint])
async def get_trends(
    location: Optional[str] = None,
    limit: int = 100,
):
    return db.get_trend_data(location_tag=location, limit=limit)


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    out_dir = RUNS_DIR / job_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    with _lock:
        _jobs.pop(job_id, None)
    return {"deleted": job_id}
