"""
cctv/pipeline.py — core detection & tracking pipeline for the CCTV model.

Owns the full lifecycle of a single video job:
  upload → read frames → YOLO detect → BoT-SORT track →
  StableIdMapper → annotate → write outputs → collect metrics.

The pipeline is a generator — `process_video()` yields per-frame
results so the caller (a background job in `cctv/routes.py`) decides
what to do with them.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np

from cctv.config import CATTLE_TERMS, RUNS_DIR, PipelineConfig
from cctv.stable_id import StableIdMapper

# try importing ultralytics — hard-fail if missing
try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "ultralytics is required. Install with:  pip install ultralytics"
    ) from exc


# ── data structures ───────────────────────────────────────────────

@dataclass
class FrameResult:
    """Per-frame output yielded by the pipeline generator."""
    frame_idx: int
    processed_frame_idx: int          # sequential index of frames we actually ran
    annotated_frame: np.ndarray       # BGR frame with boxes drawn
    stable_ids: list[int]
    bboxes: list[tuple[float, float, float, float]]   # x1 y1 x2 y2
    confidences: list[float]
    raw_tracker_ids: list[int]
    cattle_in_frame: int
    elapsed_sec: float


@dataclass
class VideoSummary:
    """Final summary after the pipeline finishes."""
    job_id: str
    final_cattle_count: int
    count_method: str                 # "tracking" or "max_in_frame"
    unique_tracked_cattle: int
    unique_track_ids: list[int]
    raw_tracker_ids: list[int]
    max_cattle_in_frame: int
    average_confidence: float
    total_detections: int
    throughput_fps: float
    processing_seconds: float
    source_fps: float
    source_width: int
    source_height: int
    total_frames: int
    frames_processed: int
    frames_with_cattle: int
    output_video: str
    output_report: str
    output_csv: str


# ── helpers ───────────────────────────────────────────────────────

def _resolve_cattle_class_ids(model) -> set[int]:
    """Map CATTLE_TERMS to the model's numeric class IDs."""
    names: dict[int, str] = model.names  # {0: "person", 19: "cow", …}
    ids = set()
    for cid, name in names.items():
        if name.lower() in CATTLE_TERMS:
            ids.add(cid)
    return ids


def _draw_stable_tracks(
    frame: np.ndarray,
    stable_detections: list[tuple[int, tuple[float, float, float, float]]],
    confidences: list[float],
) -> np.ndarray:
    """Draw bounding boxes and 'Cow ID N' labels on the frame."""
    annotated = frame.copy()
    for (sid, (x1, y1, x2, y2)), conf in zip(stable_detections, confidences):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        colour = _id_colour(sid)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
        label = f"Cow ID {sid}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(
            annotated, label, (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return annotated


def _id_colour(sid: int) -> tuple[int, int, int]:
    """Deterministic colour per stable ID (BGR)."""
    palette = [
        (46, 204, 113), (52, 152, 219), (231, 76, 60),
        (241, 196, 15), (155, 89, 182), (26, 188, 156),
        (230, 126, 34), (44, 62, 80),   (192, 57, 43),
        (22, 160, 133), (142, 68, 173), (39, 174, 96),
    ]
    return palette[sid % len(palette)]


# ── pipeline generator ────────────────────────────────────────────

def process_video(
    video_path: str | Path,
    cfg: PipelineConfig,
    job_id: Optional[str] = None,
) -> Generator[FrameResult, None, VideoSummary]:
    """
    Process a video through the full pipeline, yielding per-frame results.

    Usage
    -----
    ```python
    gen = process_video("clip.mp4", cfg)
    for frame_result in gen:
        # render frame_result.annotated_frame, update progress, etc.
        pass
    summary = gen.value   # available after StopIteration
    ```
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    job_id = job_id or uuid.uuid4().hex[:12]
    out_dir = Path(cfg.output_dir) if cfg.output_dir else RUNS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load model ────────────────────────────────────────────────
    model = YOLO(cfg.model_path)
    cattle_ids = _resolve_cattle_class_ids(model) if cfg.filter_cattle_classes else None

    # ── open video ────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # cv2.isOpened() can return True for a file it cannot actually decode
    # (e.g. a non-video file with a .mp4 extension) — CAP_PROP_FRAME_COUNT
    # then comes back as -1 (or an out-of-range float that overflows on
    # int() cast), which used to sail through as a "successful" job with
    # garbage stats instead of a clear rejection.
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(
            f"Unusable video file — cannot decode frames: {video_path}"
        )

    # output video writer
    out_video_path = out_dir / "annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = src_fps / cfg.vid_stride
    writer = cv2.VideoWriter(str(out_video_path), fourcc, out_fps, (src_w, src_h))

    # ── stable-ID mapper & accumulators ───────────────────────────
    mapper = StableIdMapper(
        iou_thresh=cfg.stable_id_iou_thresh,
        memory_frames=cfg.stable_id_memory_frames,
    )

    csv_rows: list[dict] = []
    all_confidences: list[float] = []
    total_detections = 0
    max_in_frame = 0
    frames_with_cattle = 0
    processed_count = 0
    all_raw_ids: set[int] = set()
    frames_visible: dict[int, int] = {}   # stable_id -> count of frames it appeared in

    t_start = time.perf_counter()
    frame_idx = -1

    # ── frame loop ────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # stride — skip frames
        if frame_idx % cfg.vid_stride != 0:
            continue

        processed_count += 1
        elapsed = time.perf_counter() - t_start

        # ── detect / track ────────────────────────────────────────
        if cfg.use_tracking:
            results = model.track(
                frame,
                imgsz=cfg.img_size,
                conf=cfg.confidence,
                half=cfg.half,
                stream=True,
                tracker=cfg.tracker_yaml,
                persist=True,
                verbose=False,
                classes=list(cattle_ids) if cattle_ids else None,
            )
        else:
            results = model.predict(
                frame,
                imgsz=cfg.img_size,
                conf=cfg.confidence,
                half=cfg.half,
                stream=True,
                verbose=False,
                classes=list(cattle_ids) if cattle_ids else None,
            )

        # collect detections from results
        raw_detections: list[tuple[int, tuple[float, float, float, float]]] = []
        frame_confidences: list[float] = []

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf_val = float(boxes.conf[i].cpu())
                raw_id = int(boxes.id[i].cpu()) if boxes.id is not None else i
                raw_detections.append((raw_id, tuple(xyxy)))
                frame_confidences.append(conf_val)
                all_raw_ids.add(raw_id)

        # ── stable-ID remap ──────────────────────────────────────
        stable_detections = mapper.update(frame_idx, raw_detections)

        stable_ids = [sid for sid, _ in stable_detections]
        bboxes = [bb for _, bb in stable_detections]
        raw_ids_frame = [rid for rid, _ in raw_detections]

        for sid in stable_ids:
            frames_visible[sid] = frames_visible.get(sid, 0) + 1

        cattle_count = len(stable_detections)
        total_detections += cattle_count
        if cattle_count > max_in_frame:
            max_in_frame = cattle_count
        if cattle_count > 0:
            frames_with_cattle += 1
        all_confidences.extend(frame_confidences)

        # ── annotate frame ────────────────────────────────────────
        annotated = _draw_stable_tracks(frame, stable_detections, frame_confidences)
        writer.write(annotated)

        # ── CSV row ───────────────────────────────────────────────
        csv_rows.append({
            "frame": frame_idx,
            "cattle_in_frame": cattle_count,
            "stable_ids": stable_ids,
            "raw_tracker_ids": raw_ids_frame,
            "unique_ids_so_far": mapper.total_minted,
            "avg_confidence": (
                round(sum(frame_confidences) / len(frame_confidences), 4)
                if frame_confidences else 0
            ),
            "elapsed_sec": round(elapsed, 3),
        })

        # ── yield to caller ───────────────────────────────────────
        yield FrameResult(
            frame_idx=frame_idx,
            processed_frame_idx=processed_count,
            annotated_frame=annotated,
            stable_ids=stable_ids,
            bboxes=bboxes,
            confidences=frame_confidences,
            raw_tracker_ids=raw_ids_frame,
            cattle_in_frame=cattle_count,
            elapsed_sec=elapsed,
        )

    # ── cleanup ───────────────────────────────────────────────────
    cap.release()
    writer.release()

    # A positive CAP_PROP_FRAME_COUNT doesn't guarantee any frame actually
    # decoded successfully (some corrupt/truncated files misreport a frame
    # count but fail every cap.read()) — belt-and-suspenders on top of the
    # total_frames check above.
    if processed_count == 0:
        raise RuntimeError(
            f"Unusable video file — no frames could be read: {video_path}"
        )

    total_time = time.perf_counter() - t_start

    # ── write CSV ─────────────────────────────────────────────────
    csv_path = out_dir / "metrics.csv"
    import csv as csv_mod

    with open(csv_path, "w", newline="") as f:
        fieldnames = list(csv_rows[0].keys()) if csv_rows else []
        w = csv_mod.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in csv_rows:
            # serialise lists for CSV
            row_copy = dict(row)
            row_copy["stable_ids"] = str(row_copy["stable_ids"])
            row_copy["raw_tracker_ids"] = str(row_copy["raw_tracker_ids"])
            w.writerow(row_copy)

    # ── build summary ─────────────────────────────────────────────
    # Drop flicker IDs: a track only counts if it was actually visible for
    # at least `min_frames_visible` frames. `mapper.total_minted`/
    # `all_stable_ids` still reflect every ID ever minted (kept for
    # debugging via raw_track_ids-style inspection) — the reported count
    # uses the filtered set.
    qualifying_ids = {
        sid for sid, count in frames_visible.items()
        if count >= cfg.min_frames_visible
    }
    unique_stable = len(qualifying_ids)
    count_method = "tracking" if cfg.use_tracking and unique_stable > 0 else "max_in_frame"
    final_count = unique_stable if count_method == "tracking" else max_in_frame

    summary = VideoSummary(
        job_id=job_id,
        final_cattle_count=final_count,
        count_method=count_method,
        unique_tracked_cattle=unique_stable,
        unique_track_ids=sorted(qualifying_ids),
        raw_tracker_ids=sorted(all_raw_ids),
        max_cattle_in_frame=max_in_frame,
        average_confidence=(
            round(sum(all_confidences) / len(all_confidences), 4)
            if all_confidences else 0
        ),
        total_detections=total_detections,
        throughput_fps=round(processed_count / total_time, 2) if total_time > 0 else 0,
        processing_seconds=round(total_time, 2),
        source_fps=round(src_fps, 2),
        source_width=src_w,
        source_height=src_h,
        total_frames=total_frames,
        frames_processed=processed_count,
        frames_with_cattle=frames_with_cattle,
        output_video=str(out_video_path),
        output_report=str(out_dir / "report.json"),
        output_csv=str(csv_path),
    )

    # write JSON report
    with open(summary.output_report, "w") as f:
        json.dump(summary.__dict__, f, indent=2, default=str)

    return summary


# ── convenience wrapper ───────────────────────────────────────────

def run_pipeline(
    video_path: str | Path,
    cfg: PipelineConfig,
    job_id: Optional[str] = None,
    on_frame: Optional[callable] = None,
) -> VideoSummary:
    """
    Run the full pipeline, optionally calling `on_frame(FrameResult)`
    for each processed frame. Returns the final VideoSummary.
    """
    gen = process_video(video_path, cfg, job_id)
    summary = None
    try:
        while True:
            fr = next(gen)
            if on_frame:
                on_frame(fr)
    except StopIteration as e:
        summary = e.value
    return summary
