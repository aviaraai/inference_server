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
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

import cv2
import imageio_ffmpeg
import numpy as np

from cctv.config import CATTLE_TERMS, RUNS_DIR, PipelineConfig, Preset, make_config
from cctv.muzzle_crop import BestSightingTracker, MuzzleCropResult, extract_muzzle_crops
from cctv.panning import detect_panning
from cctv.profiling import StageProfiler
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
    # Automatic peak-vs-tracking metric selection (see cctv/panning.py).
    # Additive fields, computed but not yet consumed by the OLD
    # final_cattle_count/count_method above (those keep their existing
    # meaning -- see the comments at their computation site below). The
    # smart selection between max_cattle_in_frame and unique_tracked_cattle
    # happens where the response is actually built (cctv/routes.py's
    # /result handler), using these two fields.
    is_panning: bool = False
    panning_ratio: float = 0.0
    count_method_used: str = "peak_in_frame"  # "peak_in_frame" | "tracking_estimate"
    # Set only by run_classify_and_count() below — how long the FAST-preset
    # classification pass took, separate from processing_seconds above
    # (which is this summary's own pass, i.e. the count pass). 0.0 for a
    # summary produced by a plain run_pipeline() call.
    classify_seconds: float = 0.0
    # One extraction attempt per stable ID that ever qualified (see
    # cctv/muzzle_crop.py) — infrastructure for cross-camera de-duplication,
    # not yet consumed by anything (see CLAUDE.md, "Cross-camera
    # de-duplication"). Keyed by stable_id, not by any registered-animal
    # identity. Empty for the classify-only pass in run_classify_and_count()
    # (extraction only runs on the real count pass — see there).
    muzzle_crops: dict[int, MuzzleCropResult] = field(default_factory=dict)


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
    profile: Optional[bool] = None,
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

    `profile`: measurement-only stage timing (cctv/profiling.py), off by
    default. `None` (the default) reads the `CCTV_PROFILE` env var, so
    production is zero-cost without anyone having to remember to pass
    `profile=False`; pass `True`/`False` explicitly to override the env var
    (e.g. from a one-off script). When enabled, prints a per-stage summary
    table to stdout after the run — does not change any detection/tracking
    logic or written output (annotated.mp4/metrics.csv/report.json/
    muzzle_crops/*.jpg are byte-identical to a run with profiling off).
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

    # ── output video: one persistent ffmpeg process, not cv2.VideoWriter ──
    # Previously: cv2.VideoWriter wrote mp4v/FMP4 during the loop (this
    # machine's OpenCV/FFmpeg build can't encode H.264 directly — bundled
    # libopenh264 fails to load), then a SEPARATE subprocess.run() batch pass
    # re-decoded and re-encoded the ENTIRE finished file to H.264 after the
    # loop — every frame paid for encoding twice, and the second pass
    # couldn't start until the first was 100% done and the file closed
    # (measured at CROWDED_HD on clip_01.mp4: encode_frame 27.2s +
    # encode_final 24.3s = 51.5s of a 185.7s total). Fixed by opening ffmpeg
    # itself as a persistent subprocess before the loop and piping each
    # annotated frame's raw BGR bytes to its stdin as it's produced — ffmpeg
    # encodes H.264 directly, once, overlapped with later frames' detection/
    # tracking, with no intermediate mp4v file at all.
    out_video_path = out_dir / "annotated.mp4"
    out_fps = src_fps / cfg.vid_stride
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    # ffmpeg writes its own diagnostic/progress text to stderr continuously
    # while running. PIPE-ing that without draining it is a classic
    # subprocess deadlock: the OS pipe buffer fills, ffmpeg blocks trying to
    # write to it, we're blocked writing frames to stdin, neither side ever
    # unblocks. Routed to a real file instead — no fixed buffer to fill —
    # and only read back if ffmpeg actually fails, for a useful error message.
    ffmpeg_stderr_path = out_dir / "ffmpeg_stderr.log"
    ffmpeg_stderr_file = open(ffmpeg_stderr_path, "wb")
    ffmpeg_proc = subprocess.Popen(
        [
            ffmpeg_path, "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{src_w}x{src_h}",
            "-r", f"{out_fps:.6f}",
            "-i", "pipe:0",
            "-vcodec", "libx264",
            "-preset", "fast",
            "-crf", "23",
            # Explicit, not left to ffmpeg's default: raw BGR input carries no
            # chroma-subsampling hint, and without this libx264 can pick a
            # pixel format (e.g. yuv444p) that isn't universally
            # browser-decodable. yuv420p is what the old mp4v-source path
            # produced and is what this codebase's earlier browser-
            # playability fix (see CLAUDE.md) was built around — this keeps
            # that same compatibility guarantee under the new encode path.
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_video_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=ffmpeg_stderr_file,
    )

    # ── stable-ID mapper & accumulators ───────────────────────────
    mapper = StableIdMapper(
        iou_thresh=cfg.stable_id_iou_thresh,
        memory_frames=cfg.stable_id_memory_frames,
    )
    # Only feed real crop candidates: only=True on the count pass avoids
    # copying frames for the throwaway classify pass (run_classify_and_count
    # below), which never reads muzzle_crops.
    sighting_tracker = BestSightingTracker() if cfg.extract_muzzle_crops else None

    csv_rows: list[dict] = []
    all_confidences: list[float] = []
    total_detections = 0
    max_in_frame = 0
    frames_with_cattle = 0
    processed_count = 0
    all_raw_ids: set[int] = set()
    frames_visible: dict[int, int] = {}   # stable_id -> count of frames it appeared in

    # Measurement-only (cctv/profiling.py): off unless CCTV_PROFILE=1 or
    # `profile=True` is passed explicitly. A disabled profiler's stage()
    # calls are true no-ops, so this instruments the pipeline unconditionally
    # without changing behaviour when profiling is off.
    #
    # There is no separate "preprocessing" stage below — checked, there
    # isn't one in this file to time. `model.track()`/`model.predict()`
    # with `stream=True` returns a generator; the actual resize/normalize/
    # to-tensor preprocessing, the forward pass, AND (when a tracker is
    # attached) BoT-SORT's own update all run lazily, fused together,
    # inside the `for r in results:` loop below, not at the call site — so
    # "yolo_track" below is preprocess+inference+BoT-SORT as one
    # inseparable unit, not "inference" alone. Splitting them would mean
    # patching ultralytics' internal Predictor, which is exactly the kind
    # of pipeline-logic change this instrumentation is not supposed to
    # make. Only that stage touches CUDA (confirmed by reading
    # cctv/stable_id.py, this file's cv2 calls, and cv2.VideoWriter — none
    # import torch) — cctv/profiling.py's `gpu=True` synchronize() calls
    # are scoped to it alone.
    profiler = StageProfiler(enabled=profile)

    t_start = time.perf_counter()
    frame_idx = -1

    # ── frame loop ────────────────────────────────────────────────
    while True:
        with profiler.stage("decode"):
            ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # stride — skip frames
        if frame_idx % cfg.vid_stride != 0:
            continue

        processed_count += 1
        elapsed = time.perf_counter() - t_start

        # Wraps everything from here to the yield — used only to compute
        # the "other" row in the profile report (CSV-row bookkeeping,
        # muzzle-crop sighting tracking, FrameResult construction): whatever
        # this stage's total doesn't attribute to yolo_track/stable_id/
        # encode. Deliberately does NOT wrap the yield itself (the
        # `with` closes before it) -- the caller's own time between pulling
        # frames from this generator (e.g. routes.py's on_frame analytics
        # callback) is the caller's cost, not this pipeline's.
        with profiler.stage("frame_total_processed"):
            # ── detect / track ────────────────────────────────────────
            with profiler.stage("yolo_track", gpu=True):
                if cfg.use_tracking:
                    results = model.track(
                        frame,
                        imgsz=cfg.img_size,
                        conf=cfg.confidence,
                        iou=cfg.nms_iou,
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
                        iou=cfg.nms_iou,
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
            with profiler.stage("stable_id"):
                stable_detections = mapper.update(frame_idx, raw_detections)

                stable_ids = [sid for sid, _ in stable_detections]
                bboxes = [bb for _, bb in stable_detections]
                raw_ids_frame = [rid for rid, _ in raw_detections]

                for sid in stable_ids:
                    frames_visible[sid] = frames_visible.get(sid, 0) + 1

            # Muzzle-crop bookkeeping (cctv/muzzle_crop.py) is a separate
            # feature from BoT-SORT/StableIdMapper tracking, so it's left
            # out of the "stable_id" stage above — it lands in "other"
            # rather than inflating the tracking number with an unrelated
            # cost.
            if sighting_tracker is not None:
                sighting_tracker.observe(frame_idx, frame, stable_detections, frame_confidences)

            cattle_count = len(stable_detections)
            total_detections += cattle_count
            if cattle_count > max_in_frame:
                max_in_frame = cattle_count
            if cattle_count > 0:
                frames_with_cattle += 1
            all_confidences.extend(frame_confidences)

            # ── annotate + stream to ffmpeg ────────────────────────────
            with profiler.stage("encode"):
                annotated = _draw_stable_tracks(frame, stable_detections, frame_confidences)
                try:
                    ffmpeg_proc.stdin.write(annotated.tobytes())
                except BrokenPipeError:
                    # ffmpeg exited early (bad input, codec error, etc.) —
                    # surface ITS reason rather than a bare pipe error.
                    ffmpeg_proc.wait()
                    ffmpeg_stderr_file.close()
                    stderr_text = ffmpeg_stderr_path.read_text(errors="replace")
                    raise RuntimeError(
                        f"ffmpeg exited early (code {ffmpeg_proc.returncode}) "
                        f"while encoding frame {frame_idx}:\n{stderr_text}"
                    )

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

            frame_result = FrameResult(
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

        # ── yield to caller ───────────────────────────────────────
        yield frame_result

    # ── cleanup ───────────────────────────────────────────────────
    cap.release()

    # A positive CAP_PROP_FRAME_COUNT doesn't guarantee any frame actually
    # decoded successfully (some corrupt/truncated files misreport a frame
    # count but fail every cap.read()) — belt-and-suspenders on top of the
    # total_frames check above.
    if processed_count == 0:
        # No frames ever reached ffmpeg's stdin — stop it rather than leak
        # a subprocess still waiting on input that will never arrive.
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.kill()
        ffmpeg_proc.wait()
        ffmpeg_stderr_file.close()
        raise RuntimeError(
            f"Unusable video file — no frames could be read: {video_path}"
        )

    # ── finish encoding ───────────────────────────────────────────────
    # Closing stdin tells ffmpeg no more frames are coming; it then flushes
    # whatever it still has buffered (encoder lookahead, muxer finalization
    # — the moov atom/faststart index) and exits on its own. This replaces
    # the old separate batch re-encode pass entirely: there is no second
    # file, no second full-video decode — the frames piped in during the
    # loop above ARE the H.264 encode, this is just its tail.
    with profiler.stage("encode_finalize"):
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
    ffmpeg_stderr_file.close()

    if ffmpeg_proc.returncode != 0:
        stderr_text = ffmpeg_stderr_path.read_text(errors="replace")
        raise RuntimeError(
            f"ffmpeg exited with code {ffmpeg_proc.returncode}:\n{stderr_text}"
        )

    total_time = time.perf_counter() - t_start
    profiler.report(total_time)

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

    # One muzzle-crop extraction attempt per QUALIFYING stable ID — a
    # flicker ID that never met min_frames_visible isn't a real tracked cow
    # either, same filter the final count already applies. See
    # cctv/muzzle_crop.py; this is infrastructure only, nothing reads
    # muzzle_crops yet.
    muzzle_crop_results: dict[int, MuzzleCropResult] = {}
    if sighting_tracker is not None:
        qualifying_sightings = {
            sid: s for sid, s in sighting_tracker.best_sightings.items()
            if sid in qualifying_ids
        }
        muzzle_crop_results = extract_muzzle_crops(qualifying_sightings, out_dir)

    # Automatic peak-vs-tracking metric selection (cctv/panning.py). Uses
    # the same unique_ids_so_far series already written to metrics.csv
    # above -- one point per processed frame, in order.
    is_panning, panning_ratio = detect_panning(
        [row["unique_ids_so_far"] for row in csv_rows]
    )
    count_method_used = "tracking_estimate" if is_panning else "peak_in_frame"

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
        is_panning=is_panning,
        panning_ratio=round(panning_ratio, 4),
        count_method_used=count_method_used,
        muzzle_crops=muzzle_crop_results,
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
    profile: Optional[bool] = None,
) -> VideoSummary:
    """
    Run the full pipeline, optionally calling `on_frame(FrameResult)`
    for each processed frame. Returns the final VideoSummary.

    `profile`: forwarded to process_video() — see its docstring.
    """
    gen = process_video(video_path, cfg, job_id, profile=profile)
    summary = None
    try:
        while True:
            fr = next(gen)
            if on_frame:
                on_frame(fr)
    except StopIteration as e:
        summary = e.value
    return summary


# ── decoupled classification + counting ────────────────────────────

def run_classify_and_count(
    video_path: str | Path,
    count_cfg: PipelineConfig,
    job_id: Optional[str] = None,
    on_frame: Optional[callable] = None,
) -> VideoSummary:
    """
    Runs panning classification and cattle counting as two independent
    pipeline passes and combines them into one VideoSummary. Classification
    ALWAYS runs at Preset.FAST, regardless of what `count_cfg` is — counting
    runs at whatever `count_cfg` says (normally Preset.CROWDED_HD).

    Why these can't share one pass: confirmed on a real 23-clip validation
    (see CLAUDE.md, "Classification and counting run at different
    resolutions, on purpose") that CROWDED_HD's confidence (0.20 vs FAST's
    0.35) and NMS (0.55 vs 0.45) settings measurably corrupt the panning
    signal — 2 of 5 known false positives stayed wrongly classified, 2 NEW
    false positives appeared, and the clearest real-panning reference clip
    flipped to "static". FAST's ratios matched the original 3-clip
    calibration; CROWDED_HD's did not. Counting, separately, is genuinely
    better at CROWDED_HD (+41% to +207% peak-count recall measured earlier)
    — the fix is to stop making one preset do both jobs, not to prefer one
    preset over the other.

    The classify pass's own detection/count numbers are discarded — only
    `is_panning`/`panning_ratio` are kept — and its output directory is
    deleted immediately after, so a real job does not silently double its
    on-disk footprint forever. `on_frame` (progress callback) is wired only
    to the count pass, so a job's progress reporting stays silent during
    the classification pass — a known, accepted UX gap, not a bug: see
    CLAUDE.md.
    """
    job_id = job_id or uuid.uuid4().hex[:12]

    classify_out_dir = RUNS_DIR / f"_classify_{job_id}"
    classify_cfg = make_config(
        Preset.FAST,
        output_dir=str(classify_out_dir),
        enable_analytics=False,
        extract_muzzle_crops=False,
    )
    t0 = time.perf_counter()
    classify_summary = run_pipeline(video_path, classify_cfg, job_id=f"{job_id}_classify")
    classify_seconds = round(time.perf_counter() - t0, 2)
    shutil.rmtree(classify_out_dir, ignore_errors=True)

    count_summary = run_pipeline(video_path, count_cfg, job_id=job_id, on_frame=on_frame)

    count_summary.is_panning = classify_summary.is_panning
    count_summary.panning_ratio = classify_summary.panning_ratio
    count_summary.count_method_used = (
        "tracking_estimate" if classify_summary.is_panning else "peak_in_frame"
    )
    count_summary.classify_seconds = classify_seconds

    return count_summary
    return summary
