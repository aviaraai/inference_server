"""
cctv/profiling.py — non-invasive, off-by-default stage timing for
cctv/pipeline.py's process_video().

Purpose: find out which of decode / detect+track / StableIdMapper / encode
(streamed to ffmpeg per-frame, plus its post-loop finalize/drain) is
actually slow, before optimizing any of them. Measurement only — importing
or using this module must never change what process_video() computes or
writes to disk.

Zero-cost when disabled: StageProfiler.stage() is a context manager that,
when `enabled=False`, does nothing but `yield` — no timer call, no dict
write, no torch call. The pipeline code calls `with profiler.stage(...):`
unconditionally either way, so process_video() never has to branch on
whether profiling is on; the no-op path IS the off switch.

Toggle: env var CCTV_PROFILE=1 (checked once, at StageProfiler
construction), or pass `profile=True/False` explicitly to
process_video()/run_pipeline() to override the env var programmatically
(e.g. from a test script).
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

try:
    import torch
    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:  # pragma: no cover — torch is a hard dependency elsewhere
    torch = None  # type: ignore[assignment]
    _CUDA_AVAILABLE = False


def profiling_enabled_from_env() -> bool:
    return os.getenv("CCTV_PROFILE", "").strip().lower() in ("1", "true", "yes")


@dataclass
class _StageStats:
    total_ms: float = 0.0
    count: int = 0


class StageProfiler:
    """Accumulates wall-clock time per named stage across a whole video run.

    GPU-touching stages must be opened with `gpu=True`: CUDA dispatches
    asynchronously, so without an explicit `torch.cuda.synchronize()`
    immediately before starting and immediately after stopping the timer,
    queued GPU work from a previous stage (or work this stage queued but
    never itself waited on) can silently "leak" its cost into whichever
    stage happens to call `.cpu()` or synchronize next — attributing one
    stage's real cost to a different one. Every CPU-only stage
    (StableIdMapper, cv2 drawing/encoding, the video decode itself — none
    of which touch torch/CUDA, confirmed by reading each module before
    instrumenting it) is left `gpu=False` on purpose: an unnecessary
    synchronize() would itself add a real stall to a stage that never
    needed one, which would corrupt the exact comparison this exists to
    make.
    """

    def __init__(self, enabled: Optional[bool] = None):
        self.enabled = profiling_enabled_from_env() if enabled is None else enabled
        self._stats: dict[str, _StageStats] = defaultdict(_StageStats)
        self._order: list[str] = []  # first-seen order, for a stable report

    @contextmanager
    def stage(self, name: str, gpu: bool = False):
        if not self.enabled:
            yield
            return

        if gpu and _CUDA_AVAILABLE:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if gpu and _CUDA_AVAILABLE:
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if name not in self._stats:
                self._order.append(name)
            s = self._stats[name]
            s.total_ms += elapsed_ms
            s.count += 1

    def report(self, total_seconds: float, file=None) -> None:
        """Print the summary table. No-op if profiling was never enabled —
        there is nothing accumulated to show, and printing an empty table
        would look like a bug rather than "profiling was off"."""
        if not self.enabled:
            return

        file = file or sys.stdout
        total_ms = total_seconds * 1000.0

        # "other_per_frame" isn't timed directly — it's whatever the
        # frame_total wrapper measured that the three named per-frame
        # stages didn't account for (CSV-row bookkeeping, muzzle-crop
        # sighting tracking, FrameResult construction, the yield itself).
        # Computed here, not stored as its own stage() call, so it can
        # never itself be double-counted inside frame_total.
        #
        # "encode" here is only its PER-FRAME portion (the stdin.write()
        # call inside the loop) — deliberately excludes "encode_finalize"
        # (the post-loop stdin-close+wait drain), which happens outside
        # frame_total's scope entirely. Including it here would net the
        # drain against "other" and silently under-report bookkeeping time
        # instead of showing the drain as its own line, same as the old
        # one-shot "encode_final" row did.
        frame_total = self._stats.get("frame_total_processed")
        named_per_frame = ("yolo_track", "stable_id", "encode")
        other_ms = 0.0
        other_count = 0
        if frame_total is not None:
            accounted = sum(self._stats[n].total_ms for n in named_per_frame if n in self._stats)
            other_ms = max(0.0, frame_total.total_ms - accounted)
            other_count = frame_total.count

        rows: list[tuple[str, float, Optional[float], int]] = []
        for name in self._order:
            if name == "frame_total_processed":
                continue  # bookkeeping wrapper only, not a real pipeline stage
            s = self._stats[name]
            avg = s.total_ms / s.count if s.count else 0.0
            rows.append((name, s.total_ms, avg, s.count))
        if other_count:
            rows.append(("other (bookkeeping/yield)", other_ms, other_ms / other_count, other_count))

        header = f"{'Stage':<28} {'Total ms':>12} {'Avg ms/frame':>14} {'% of total':>11}"
        sep = "-" * len(header)
        print("\nCCTV pipeline profile", file=file)
        print(sep, file=file)
        print(header, file=file)
        print(sep, file=file)
        for name, stage_total_ms, avg_ms, count in rows:
            pct = (stage_total_ms / total_ms * 100.0) if total_ms > 0 else 0.0
            # A stage that only ever ran once in this whole run (the
            # ffmpeg re-encode, always) isn't a per-frame cost — dividing
            # by 1 would print a real number that LOOKS like a per-frame
            # average but means something completely different.
            avg_str = "(one-shot)" if count <= 1 else f"{avg_ms:.2f}"
            print(
                f"{name:<28} {stage_total_ms:>12.1f} {avg_str:>14} {pct:>10.1f}%",
                file=file,
            )
        print(sep, file=file)
        print(f"{'TOTAL (wall clock)':<28} {total_ms:>12.1f} {'':>14} {'100.0':>10}%", file=file)
        print(sep, file=file)
