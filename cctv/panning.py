"""
cctv/panning.py — detects whether a clip's camera was panning/moving across
a large area, using a signal the pipeline already computes: the running
count of distinct stable track IDs minted so far, per processed frame
(the `unique_ids_so_far` column already written to metrics.csv).

Why this signal: a fixed/static camera's rate of *new* individual discovery
drops toward zero once the visible herd has actually been seen -- there's
nothing left to discover. A panning camera keeps revealing new parts of the
shed/field throughout the clip, so it keeps minting new IDs at an
undiminished rate right up to the end. The shape of that curve, not its
absolute height, is what tells the two apart -- which is why this works
even though `unique_ids_so_far` is the *unfiltered* running count (includes
transient/flicker IDs the min_frames_visible filter would later drop): the
noise affects both segments compared below roughly equally, but a genuine
plateau vs. a genuine climb is a much bigger effect than that noise.

Validated against the real labelled data available (see CLAUDE.md,
"Automatic peak-vs-tracking metric selection"):
  - the known panning clip (max_cattle_in_frame=22 vs
    unique_tracked_cattle=62, camera motion confirmed by inspecting frames
    from 4 different points in the clip -- each shows a visibly different
    part of the shed): ratio 2.0-2.2, consistent across every re-run/config
    variant of that same source clip.
  - the one visually-confirmed STATIC clip on file (a fixed close-up on a
    feeding trough -- same background railing/trough structure start to
    finish across the whole clip): ratio 0.506.
PANNING_RATIO_THRESHOLD=1.0 sits with wide margin on both sides of those two
confirmed real examples -- not a number tuned to split a narrow gap.
"""

from __future__ import annotations

# Minimum number of processed frames needed for a first-half vs
# final-quarter comparison to mean anything. Below this, default to "not
# panning" -- i.e. keep today's already-trusted peak-in-frame behaviour
# rather than guess from too little data.
MIN_FRAMES_FOR_DETECTION = 20

# Below this many total distinct IDs ever minted, there's no meaningful
# "is the herd still being discovered" question to ask -- a handful of
# animals is well within what peak-in-frame already counts correctly, and
# a ratio computed on single-digit counts is statistically unstable.
MIN_UNIQUE_IDS_FOR_DETECTION = 4

PANNING_RATIO_THRESHOLD = 1.0


def detect_panning(unique_ids_so_far: list[int]) -> tuple[bool, float]:
    """
    `unique_ids_so_far`: the per-processed-frame running count of distinct
    stable IDs minted so far, in frame order (non-decreasing) -- exactly
    the column already written to metrics.csv.

    Returns (is_panning, ratio). `ratio` is returned even when the guard
    clauses below short-circuit to `False`, so a borderline/insufficient-
    data case is visible in logs/reports rather than collapsing silently
    into a bare boolean.
    """
    n = len(unique_ids_so_far)
    if n < MIN_FRAMES_FOR_DETECTION or unique_ids_so_far[-1] < MIN_UNIQUE_IDS_FOR_DETECTION:
        return False, 0.0

    half = n // 2
    q3 = int(n * 0.75)

    first_half_rate = (unique_ids_so_far[half] - unique_ids_so_far[0]) / half if half > 0 else 0.0
    tail = n - 1 - q3
    final_quarter_rate = (
        (unique_ids_so_far[-1] - unique_ids_so_far[q3]) / tail if tail > 0 else 0.0
    )

    if first_half_rate <= 0:
        # Nothing discovered in the first half at all. If the second half
        # genuinely picked up from a standing start, that's panning-like on
        # its own terms; if it also stayed flat, this just isn't a clip
        # with much happening in it -- not panning.
        is_panning = final_quarter_rate > 0
        return is_panning, (float("inf") if is_panning else 0.0)

    ratio = final_quarter_rate / first_half_rate
    return ratio >= PANNING_RATIO_THRESHOLD, ratio
